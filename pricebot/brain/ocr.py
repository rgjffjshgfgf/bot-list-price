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
from ..numfmt import ARABIC, LATIN, PERSIAN, digit_script, parse_number, to_latin_digits, to_script
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


def _latin(text: str) -> str:
    return to_latin_digits(text or "").strip()


# ------------------------------------------------------ digit script ---

_REFS: dict[str, list[np.ndarray]] | None = None


def _reference_digits() -> dict[str, list[np.ndarray]]:
    """Shape vectors of 0-9 drawn as Latin and as Persian digits (to tell which
    kind a list uses when its PDF stores Latin codes behind Persian-looking glyphs)."""
    global _REFS
    if _REFS is not None:
        return _REFS
    from PIL import Image, ImageDraw, ImageFont

    from .. import fonts
    refs: dict[str, list[np.ndarray]] = {}
    for script, digits in (("latin", LATIN), ("persian", PERSIAN)):
        path = next((f for f in fonts.candidate_fonts() if fonts.supports(f, digits)), None)
        if path is None:
            continue
        font = ImageFont.truetype(path, 64)
        vecs = []
        for ch in digits:
            im = Image.new("L", (120, 120), 0)
            ImageDraw.Draw(im).text((20, 10), ch, fill=255, font=font)
            a = (np.asarray(im) > 127).astype(np.float32)
            ys, xs = np.nonzero(a)
            piece = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else a
            v = cv2.resize(piece, (10, 14), interpolation=cv2.INTER_AREA).flatten()
            vecs.append(v / (np.linalg.norm(v) or 1))
        refs[script] = vecs
    _REFS = refs
    return refs


def visual_script(glyphs: list["Glyph"], text: str) -> str | None:
    """Which digits the picture really shows (latin / persian), or None if unsure."""
    refs = _reference_digits()
    if "latin" not in refs or "persian" not in refs:
        return None
    score = {"latin": 0.0, "persian": 0.0}
    chars = [c for c in to_latin_digits(text) if not c.isspace()]
    for g, ch in zip(glyphs, chars):
        if ch.isdigit() and ch not in "01":          # 0/1 look alike in both scripts' neighbours
            for sc in score:
                score[sc] += float(np.linalg.norm(refs[sc][int(ch)] - g.vec))
    if score["latin"] == score["persian"]:
        return None
    best, other = sorted(score, key=score.get)
    return best if score[other] - score[best] > 0.15 else None


# ---------------------------------------------------------- glyph bank ---

class GlyphBank:
    """Digit shapes with their labels. One bank per page (from reads that agree),
    and a lasting library the bot builds from checked lists."""
    PER_LABEL = 400

    def __init__(self) -> None:
        self.vecs: list[np.ndarray] = []
        self.geos: list[np.ndarray] = []
        self.labels: list[str] = []        # Latin digits / separators
        self.scripts: list[str] = []       # what the digit looked like: latin / persian / arabic / ""
        self._cache = None

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, glyphs: list[Glyph], text: str, script: str = "") -> bool:
        chars = [c for c in to_latin_digits(text) if not c.isspace()]
        if len(chars) != len(glyphs):
            return False
        for g, ch in zip(glyphs, chars):
            self.vecs.append(g.vec.astype(np.float32))
            self.geos.append(g.geo.astype(np.float32))
            self.labels.append(ch)
            self.scripts.append(script if ch.isdigit() else "")
        self._cache = None
        return True

    def trim(self) -> None:
        """Keep at most PER_LABEL shapes of each character (the newest)."""
        keep, count = [], {}
        for k in range(len(self.labels) - 1, -1, -1):
            c = self.labels[k]
            if count.get(c, 0) < self.PER_LABEL:
                count[c] = count.get(c, 0) + 1
                keep.append(k)
        keep.reverse()
        self.vecs = [self.vecs[k] for k in keep]
        self.geos = [self.geos[k] for k in keep]
        self.labels = [self.labels[k] for k in keep]
        self.scripts = [self.scripts[k] for k in keep]
        self._cache = None

    def extend(self, other: "GlyphBank") -> None:
        self.vecs += other.vecs
        self.geos += other.geos
        self.labels += other.labels
        self.scripts += other.scripts
        self._cache = None

    def ready(self) -> bool:
        return len(self.labels) >= 20

    def _arrays(self):
        if self._cache is None:
            G = np.stack(self.geos)
            L = np.array(self.labels)
            widths = {c: float(np.median(G[L == c, 1])) for c in set(self.labels)}
            self._cache = (np.stack(self.vecs), G, L, np.array(self.scripts), widths)
        return self._cache

    def read(self, glyphs: list[Glyph]) -> tuple[str | None, bool]:
        """(text, sure). `sure`: every character's 5 nearest shapes agree and are close."""
        if not self.ready():
            return None, False
        V, G, L, S, widths = self._arrays()
        out, sure = [], True
        looks: dict[str, int] = {}
        for g in glyphs:
            d = np.linalg.norm(V - g.vec, axis=1) + 2.0 * np.abs(G - g.geo).sum(axis=1)
            order = np.argsort(d)[:5]
            if d[order[0]] > 0.6:
                return None, False
            votes: dict[str, float] = {}
            for k in order:
                votes[L[k]] = votes.get(L[k], 0.0) + 1.0 / (0.05 + d[k])
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            if len(ranked) > 1 and ranked[1][1] > 0.6 * ranked[0][1]:
                return None, False
            if len(ranked) > 1 or d[order[0]] > 0.35 or len(order) < 5:
                sure = False
            label = ranked[0][0]
            if g.geo[1] > 1.6 * widths.get(label, 9.0) + 0.1:
                return None, False          # a blob far wider than this character: several glued together
            for k in order:
                if S[k]:
                    looks[S[k]] = looks.get(S[k], 0) + 1
            out.append(label)
        text = "".join(out)
        # zeros read as separators ("۳۴,۷۵۴,,,,") must not be tidied into a shorter number
        if not text or text[0] in _SEPS or text[-1] in _SEPS or any(
                a in _SEPS and b in _SEPS for a, b in zip(text, text[1:])):
            return None, False
        script = max(looks, key=looks.get) if looks else "latin"
        return (to_script(text, script) if script != "latin" else text), sure

    # ---- persistence (lasting library) ----
    def save(self, path) -> None:
        if not self.labels:
            return
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, vecs=(np.stack(self.vecs) * 255).astype(np.uint8),
                            geos=np.stack(self.geos).astype(np.float16), labels=np.array(self.labels),
                            scripts=np.array(self.scripts))
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "GlyphBank":
        bank = cls()
        try:
            with np.load(path, allow_pickle=False) as z:
                bank.vecs = list(z["vecs"].astype(np.float32) / 255)
                bank.geos = list(z["geos"].astype(np.float32))
                bank.labels = [to_latin_digits(str(x)) for x in z["labels"]]
                bank.scripts = [str(x) for x in z["scripts"]] if "scripts" in z else [""] * len(bank.labels)
        except (OSError, ValueError, KeyError):
            pass
        return bank


def harvest(ink: raster.InkMap, targets: list[raster.Target], texts: list[str], library: GlyphBank) -> int:
    """Add the digit shapes of checked numbers to the library, labelled with the
    digits the picture really shows. Returns how many numbers were added."""
    n = 0
    for t, text in zip(targets, texts):
        if t is None or not text:
            continue
        g = _glyphs(ink, t)
        if not g:
            continue
        script = visual_script(g, text) or ""
        if library.add(g, text, script):
            n += 1
    if n:
        library.trim()
    return n


# ---------------------------------------------------------------- reader ---

@dataclass
class Read:
    text: str | None          # accepted read, or None when the reads disagree
    votes: list[str]
    how: str = ""
    glyph: str = ""           # what the shape library alone read (for measuring it)
    glyph_sure: bool = False


# Confidence given to numbers the shape library found on a page: marks them as
# NOT an independent read (the library cannot confirm itself).
GLYPH_CONF = -1.0


def complete(ink: raster.InkMap, t: raster.Target) -> bool:
    """False when more ink sits glued to the number on its line (then only a piece
    of a bigger number was found, and rewriting it would corrupt the price)."""
    x0, y0, x1, y1 = t.box
    h = max(1, y1 - y0)
    for w in ink.words:
        if w.polarity != t.polarity or min(w.y1, y1) - max(w.y0, y0) < 0.5 * min(w.h, h):
            continue
        if w.x0 >= x0 and w.x1 <= x1:
            continue                                   # part of the number itself
        if w.h < 0.3 * h:
            continue                                   # dust
        gap = max(w.x0 - x1, x0 - w.x1)
        if raster._vline_between(ink, min(x1, w.x1), max(x0, w.x0), y0, y1):
            continue                                   # another cell
        if gap < 0.45 * h:
            return False
        # a wider gap still continues the number when the neighbour is digit-sized ink
        # (fonts with wide thousands separators: "۹,۷۶۴  ,۰۰۰")
        if gap < 1.1 * h and 0.3 * h <= w.h <= 1.3 * h and w.w <= 4 * h:
            return False
    return True


def glyph_words(rgb: np.ndarray, words: list[tuple[str, Box, float]],
                library: GlyphBank) -> list[tuple[str, Box, float]]:
    """Add the numbers the shape library recognises on the page to the OCR words
    (fonts Tesseract cannot read, e.g. bold Persian digits). An OCR word lying on
    such a number is replaced by it."""
    if len(library) < 200:
        return words
    gray = _to_gray(rgb)
    th = text_height(gray)
    ink = raster.InkMap(rgb, th)
    found: list[tuple[str, Box, float]] = []
    cand = sorted((w for w in ink.words if 0.5 * th <= w.h <= 3 * th and w.w <= 25 * th),
                  key=lambda w: (w.polarity, round(w.yc / th), w.x0))
    # rows of words; a number may be split into several ink groups ("۳/۶۹۶" + "/۰۰۰")
    rows: list[list] = []
    for w in cand:
        if rows and rows[-1][-1].polarity == w.polarity and \
                min(rows[-1][-1].y1, w.y1) - max(rows[-1][-1].y0, w.y0) >= 0.5 * min(rows[-1][-1].h, w.h):
            rows[-1].append(w)
        else:
            rows.append([w])

    def read_span(ws) -> tuple[str, bool, tuple, list] | None:
        x0, y0 = min(w.x0 for w in ws), min(w.y0 for w in ws)
        x1, y1 = max(w.x1 for w in ws), max(w.y1 for w in ws)
        t = raster.Target((x0, y0, x1, y1), ws[0].polarity, "", (x0, x1))
        g = _glyphs(ink, t)
        if not g or not 2 <= len(g) <= 20:
            return None
        # finding numbers may be a little generous: every number is read again, strictly, before it changes
        text, sure = library.read(g)
        if text and _plausible(_clean(text), g):
            return _clean(text), sure, (float(x0), float(y0), float(x1), float(y1)), g
        return None

    for row in rows:
        row.sort(key=lambda w: w.x0)
        i = 0
        while i < len(row):
            j = i
            while j + 1 < len(row) and row[j + 1].x0 - row[j].x1 <= 0.8 * max(row[j].h, row[j + 1].h) \
                    and j - i < 4:
                j += 1
            hit, used = None, i
            for k in range(j, i - 1, -1):             # the longest run that reads as one number
                hit = read_span(row[i:k + 1])
                if hit:
                    used = k
                    break
            if hit:
                found.append((hit[0], hit[2], GLYPH_CONF))
            i = used + 1
    if not found:
        return words
    from .layout import overlap_share
    kept = [x for x in words if not any(overlap_share(x[1], f[1]) > 0.3 for f in found)]
    return kept + found


def _plausible(text: str, glyphs: list[Glyph] | None) -> bool:
    """A read must fit the ink: not far fewer characters than glyphs seen (a whole
    price read as "0" happens with unknown fonts), and be a real amount."""
    parsed = parse_number(text)
    if parsed is None or parsed.value <= 0:
        return False
    digits = [c for c in to_latin_digits(text) if c.isdigit()]
    if len(digits) < 2:
        return False
    chars = [c for c in text if not c.isspace()]
    return not glyphs or len(chars) >= len(glyphs) - 1


def read_targets(ink: raster.InkMap, targets: list[raster.Target], first_reads: list[str],
                 library: GlyphBank | None = None, trust_glyphs: bool = False) -> list[Read]:
    """Independent reads of every target; accepted only where they agree.

    library: lasting digit shapes learned from checked lists. trust_glyphs: its
    reads have proven themselves, so a sure shape read alone is accepted (for
    fonts Tesseract cannot read)."""
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
    if library is not None and len(library):
        bank.extend(library)
    for k in range(len(targets)):
        tess = [r for r in (first[k], a[k], b[k]) if r]
        if len(tess) >= 2 and len({_latin(r) for r in tess}) == 1 and glyphs[k]:
            bank.add(glyphs[k], tess[0], digit_script(tess[0]))

    out = []
    for k in range(len(targets)):
        g, sure = bank.read(glyphs[k]) if glyphs[k] else (None, False)
        g = _clean(g) if g else ""
        votes = [first[k], a[k], b[k], g]
        tess = [r for r in (first[k], a[k], b[k]) if r]
        text, how = None, ""
        agree = sum(1 for r in tess if _latin(r) == _latin(g)) if g else 0
        conflict = len(tess) - agree if g else 0
        if g and agree >= 1 and conflict == 0:          # any disagreeing read vetoes the number
            text, how = g, "glyph+ocr"
        elif g and sure and trust_glyphs and conflict == 0:
            text, how = g, "glyph"
        elif not g and len(tess) == 3 and len({_latin(r) for r in tess}) == 1:
            text, how = tess[0], "ocr×3"
        if text and not _plausible(text, glyphs[k]):
            text, how = None, "shape"
        if text and not complete(ink, targets[k]):
            text, how = None, "piece"
        out.append(Read(text, votes, how, g, sure))

    # a price far shorter than the others (a piece: "۹,۷۶۴" of 9,764,000) is not trusted
    lens = sorted(len([c for c in to_latin_digits(r.text) if c.isdigit()]) for r in out if r.text)
    if len(lens) >= 3:
        typical = lens[len(lens) // 2]
        for r in out:
            if r.text and len([c for c in to_latin_digits(r.text) if c.isdigit()]) <= typical - 3:
                r.text, r.how = None, "short"

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
