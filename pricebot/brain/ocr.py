"""Reading photos and scans without Gemini.

* page_words(): Tesseract OCR of the whole page (table lines removed first,
  image scaled so the text is a comfortable size) -> words with boxes.
* LocalReader: reads each price several independent ways and only accepts a
  number when the reads agree:
    1. the page OCR,
    2. Tesseract on the cut-out number (two different renderings),
    3. a glyph memory built from this very page: every digit shape is
       compared with the same digits elsewhere in the list (same font), so a
       mistake Tesseract makes on one price is out-voted by the others.
"""
from __future__ import annotations

import logging
import os
import shutil
import statistics
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .. import config, raster
from ..numfmt import ARABIC, LATIN, PERSIAN, digit_script, parse_number, to_latin_digits
from .layout import Box, digits_of

log = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional until installed
    pytesseract = None

_LANGS: set[str] | None = None
# Each Tesseract run otherwise starts one thread per core; several runs at once then
# fight over the CPU and a page that takes 1 s can take minutes.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
# Tesseract is memory hungry on big pages: limit how many run at once.
_SLOTS = threading.BoundedSemaphore(max(1, config.OCR_PARALLEL))
TIMEOUT = 180


def available() -> bool:
    global _LANGS
    if pytesseract is None or not shutil.which("tesseract"):
        return False
    if _LANGS is None:
        try:
            _LANGS = set(pytesseract.get_languages(config=""))
        except Exception:  # noqa: BLE001
            _LANGS = set()
    return "eng" in _LANGS


def _lang(script: str) -> str:
    if script in ("persian", "arabic") and _LANGS and "fas" in _LANGS:
        return "fas"
    return "eng"


def _page_lang() -> str:
    return "fas+eng" if _LANGS and "fas" in _LANGS else "eng"


# ============================================================== page OCR ==

def text_height(gray: np.ndarray) -> float:
    """Typical height of text characters (median of glyph-sized ink blobs)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [int(stats[k, 3]) for k in range(1, n)
          if 4 <= stats[k, 3] <= 200 and stats[k, 2] <= 3 * stats[k, 3] and stats[k, 4] >= 6]
    return float(statistics.median(hs)) if len(hs) >= 5 else 20.0


def _remove_lines(gray: np.ndarray, th: float) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(25, int(th * 3))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    lines = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[lines > 0] = 255
    return out


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # light text on dark cells: Tesseract wants dark on light
    if float(np.median(gray)) < 110:
        gray = 255 - gray
    return gray


def page_words(rgb: np.ndarray, psm: int = 11) -> list[tuple[str, Box, float]]:
    """(text, box in rgb pixels, confidence 0-100) of every word on the page."""
    gray = _to_gray(rgb)
    th = text_height(gray)
    s = min(4.0, max(0.6, 30.0 / th))
    if max(gray.shape) * s > 4200:          # bigger only costs time (a cover photo can take minutes)
        s = 4200 / max(gray.shape)
    big = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    clean = _remove_lines(big, th * s)
    with _SLOTS:
        data = pytesseract.image_to_data(clean, lang=_page_lang(), config=f"--psm {psm}",
                                         output_type=pytesseract.Output.DICT, timeout=TIMEOUT)
    out = []
    for k, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        x, y, w, h = data["left"][k], data["top"][k], data["width"][k], data["height"][k]
        conf = float(data["conf"][k]) if str(data["conf"][k]) not in ("", "-1") else 0.0
        out.append((text, (x / s, y / s, (x + w) / s, (y + h) / s), conf))
    return out


# ========================================================= local reading ==

_SEPS = ",./٬،٫"


def _whitelist(script: str) -> str:
    digits = PERSIAN if script == "persian" else ARABIC if script == "arabic" else LATIN
    return digits + _SEPS


def _crop(ink: raster.InkMap, t: raster.Target, height: int, pad: float = 0.35) -> np.ndarray:
    x0, y0, x1, y1 = t.box
    H, W = ink.gray.shape
    m = max(2, int(pad * (y1 - y0)))
    g = ink.gray[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    if t.polarity == "light":
        g = 255 - g
    s = height / max(1, g.shape[0])
    return cv2.resize(g, (max(1, round(g.shape[1] * s)), height),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def _sheet_reads(ink: raster.InkMap, targets: list[raster.Target], height: int, binarize: bool,
                 script: str) -> list[str]:
    """OCR many number crops in one Tesseract call (stacked, one per line)."""
    if not targets:
        return []
    crops = []
    for t in targets:
        c = _crop(ink, t, height)
        if binarize:
            _, c = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        crops.append(c)
    gap = height
    width = max(c.shape[1] for c in crops) + 2 * gap
    sheet = np.full((gap + len(crops) * (height + gap), width), 255, np.uint8)
    for k, c in enumerate(crops):
        y = gap + k * (height + gap)
        sheet[y:y + height, gap:gap + c.shape[1]] = c
    cfg = f"--psm 6 -c tessedit_char_whitelist={_whitelist(script)}"
    try:
        with _SLOTS:
            data = pytesseract.image_to_data(sheet, lang=_lang(script), config=cfg,
                                             output_type=pytesseract.Output.DICT, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        log.warning("tesseract failed: %s", exc)
        return [""] * len(targets)
    parts: list[list[tuple[int, str]]] = [[] for _ in targets]
    for k, text in enumerate(data["text"]):
        text = (text or "").strip()
        if not text:
            continue
        yc = data["top"][k] + data["height"][k] / 2
        row = int((yc - gap) // (height + gap))
        inside = (yc - gap) - row * (height + gap)
        if 0 <= row < len(targets) and -0.2 * height <= inside <= 1.2 * height:
            parts[row].append((data["left"][k], text))
    return ["".join(t for _, t in sorted(p)) for p in parts]


def _clean(text: str) -> str:
    text = text.strip().strip(_SEPS)
    return text if parse_number(text) else ""


# ----------------------------------------------------------- glyph bank ---

@dataclass
class Glyph:
    vec: np.ndarray
    geo: np.ndarray


def _glyphs(ink: raster.InkMap, t: raster.Target) -> list[Glyph] | None:
    mask = raster.ink_mask(ink, t, relative=True).astype(np.uint8)
    if mask.sum() < 5:
        return None
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [tuple(int(v) for v in st[k]) for k in range(1, n) if st[k, 4] >= 2]
    if not comps:
        return None
    comps.sort(key=lambda c: c[0])
    merged: list[list[int]] = []
    for x, y, w, h, a in comps:
        if merged:
            px, py, pw, ph, pa = merged[-1]
            ov = min(px + pw, x + w) - max(px, x)
            if ov >= 0.6 * min(pw, w):
                nx0, ny0 = min(px, x), min(py, y)
                nx1, ny1 = max(px + pw, x + w), max(py + ph, y + h)
                merged[-1] = [nx0, ny0, nx1 - nx0, ny1 - ny0, pa + a]
                continue
        merged.append([x, y, w, h, a])
    heights = [m[3] for m in merged]
    H = float(max(heights))
    big = [m for m in merged if m[3] >= 0.55 * H]
    base = float(statistics.median([m[1] + m[3] for m in big])) if big else H
    out = []
    for x, y, w, h, _ in merged:
        piece = mask[y:y + h, x:x + w].astype(np.float32)
        vec = cv2.resize(piece, (10, 14), interpolation=cv2.INTER_AREA).flatten()
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm else vec
        geo = np.array([h / H, w / H, (y + h / 2 - base) / H], np.float32)
        out.append(Glyph(vec, geo))
    return out


class GlyphBank:
    def __init__(self) -> None:
        self.vecs: list[np.ndarray] = []
        self.geos: list[np.ndarray] = []
        self.labels: list[str] = []

    def add(self, glyphs: list[Glyph], text: str) -> bool:
        chars = [c for c in text if not c.isspace()]
        if len(chars) != len(glyphs):
            return False
        for g, ch in zip(glyphs, chars):
            self.vecs.append(g.vec)
            self.geos.append(g.geo)
            self.labels.append(ch)
        return True

    def ready(self) -> bool:
        return len(self.labels) >= 20

    def read(self, glyphs: list[Glyph]) -> str | None:
        if not self.ready():
            return None
        V = np.stack(self.vecs)
        G = np.stack(self.geos)
        out = []
        for g in glyphs:
            d = np.linalg.norm(V - g.vec, axis=1) + 2.0 * np.abs(G - g.geo).sum(axis=1)
            order = np.argsort(d)[:5]
            if d[order[0]] > 0.6:
                return None
            votes: dict[str, float] = {}
            for k in order:
                votes[self.labels[k]] = votes.get(self.labels[k], 0.0) + 1.0 / (0.05 + d[k])
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            if len(ranked) > 1 and ranked[1][1] > 0.6 * ranked[0][1]:
                return None
            out.append(ranked[0][0])
        return "".join(out)


# ---------------------------------------------------------------- reader ---

@dataclass
class Read:
    text: str | None          # accepted read, or None when the reads disagree
    votes: list[str]
    how: str = ""


def read_targets(ink: raster.InkMap, targets: list[raster.Target], first_reads: list[str]) -> list[Read]:
    """Independent reads of every target; accepted only where they agree."""
    if not targets:
        return []
    scripts = [digit_script(r) if r else "latin" for r in first_reads]
    script = max(set(scripts), key=scripts.count)
    a = _sheet_reads(ink, targets, 44, False, script)
    b = _sheet_reads(ink, targets, 64, True, script)
    first = [_clean(r) for r in first_reads]
    a = [_clean(r) for r in a]
    b = [_clean(r) for r in b]

    glyphs = [_glyphs(ink, t) for t in targets]
    bank = GlyphBank()
    for k in range(len(targets)):
        tess = [r for r in (first[k], a[k], b[k]) if r]
        if len(tess) >= 2 and len(set(tess)) == 1 and glyphs[k]:
            bank.add(glyphs[k], tess[0])

    out = []
    for k in range(len(targets)):
        g = bank.read(glyphs[k]) if glyphs[k] else None
        g = _clean(g) if g else ""
        votes = [first[k], a[k], b[k], g]
        tess = [r for r in (first[k], a[k], b[k]) if r]
        text, how = None, ""
        if g:
            if sum(1 for r in tess if r == g) >= 1:
                text, how = g, "glyph+ocr"
        elif len(tess) == 3 and len(set(tess)) == 1:
            text, how = tess[0], "ocr×3"
        out.append(Read(text, votes, how))

    # a column's prices share one number style
    seps = [parse_number(r.text).fmt.group_sep for r in out if r.text]
    if seps:
        common = max(set(seps), key=seps.count)
        if seps.count(common) >= 0.8 * len(seps):
            for r in out:
                if r.text and parse_number(r.text).fmt.group_sep != common:
                    r.text, r.how = None, "style"
    return out


def same_number(a: str, b: str) -> bool:
    return bool(a) and bool(b) and digits_of(a) == digits_of(b) and \
        to_latin_digits(a).strip() == to_latin_digits(b).strip()
