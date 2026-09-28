"""Font discovery, rendering and "which font does this number look like" matching.

Used whenever a price has to be re-drawn from pixels (photos, scanned PDFs) or
when a PDF's own embedded font lacks a needed digit.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from . import config
from .numfmt import to_latin_digits, to_script

log = logging.getLogger(__name__)

# Families that look like what price lists are usually typed in.
_KNOWN = {
    # Latin digits
    "arial.ttf", "arialbd.ttf", "arialn.ttf", "arialnb.ttf",
    "liberationsans-regular.ttf", "liberationsans-bold.ttf",
    "liberationsansnarrow-regular.ttf", "liberationsansnarrow-bold.ttf",
    "calibri.ttf", "calibrib.ttf", "carlito-regular.ttf", "carlito-bold.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "dejavusans.ttf", "dejavusans-bold.ttf", "dejavusanscondensed.ttf", "dejavusanscondensed-bold.ttf",
    "times.ttf", "timesbd.ttf", "liberationserif-regular.ttf", "liberationserif-bold.ttf",
    "segoeui.ttf", "segoeuib.ttf", "verdana.ttf", "verdanab.ttf",
    "notosans-regular.ttf", "notosans-bold.ttf", "roboto-regular.ttf", "roboto-bold.ttf",
    "cambria.ttc", "cambriab.ttf", "consola.ttf", "consolab.ttf", "cour.ttf", "courbd.ttf",
    # Persian / Arabic digits
    "vazirmatn-regular.ttf", "vazirmatn-medium.ttf", "vazirmatn-bold.ttf", "vazirmatn-black.ttf",
    "vazirmatn-light.ttf", "vazirmatn-semibold.ttf", "vazirmatn-extrabold.ttf",
    "notosansarabic-regular.ttf", "notosansarabic-bold.ttf",
    "notonaskharabic-regular.ttf", "notonaskharabic-bold.ttf",
    "notokufiarabic-regular.ttf", "notokufiarabic-bold.ttf",
}


def _font_dirs() -> list[Path]:
    dirs = [config.FONTS_DIR]
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
                 Path.home() / ".local/share/fonts"]
    return [d for d in dirs if d.exists()]


@functools.lru_cache(maxsize=1)
def candidate_fonts() -> tuple[str, ...]:
    """All font files worth trying, user-supplied fonts first."""
    found: dict[str, str] = {}
    user_dir = config.FONTS_DIR.resolve() if config.FONTS_DIR.exists() else None
    for d in _font_dirs():
        for root, _, files in os.walk(d):
            for name in files:
                low = name.lower()
                if not low.endswith((".ttf", ".otf")):
                    continue
                in_user_dir = user_dir is not None and Path(root).resolve().is_relative_to(user_dir)
                if in_user_dir or low in _KNOWN:
                    found.setdefault(low, str(Path(root) / name))
    paths = sorted(found.values(), key=lambda p: (not _is_user_font(p), p.lower()))
    log.info("font candidates: %d", len(paths))
    return tuple(paths)


def _is_user_font(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(config.FONTS_DIR.resolve())
    except (OSError, ValueError):
        return False


@functools.lru_cache(maxsize=512)
def _coverage(path: str) -> frozenset[str]:
    try:
        font = pymupdf.Font(fontfile=path)
    except Exception:  # noqa: BLE001 - unreadable font files are just skipped
        return frozenset()
    probe = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩,./٬،'’٫ "
    return frozenset(ch for ch in probe if font.has_glyph(ord(ch)))


def supports(path: str, text: str) -> bool:
    cov = _coverage(path)
    return bool(cov) and all(ch in cov for ch in text)


@functools.lru_cache(maxsize=256)
def _pil_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)


@dataclass
class Rendered:
    alpha: np.ndarray      # float32 0..1, final (downsampled) coverage
    baseline: float        # y of the baseline inside `alpha`
    ink: tuple[int, int, int, int]  # x0, y0, x1, y1 of ink inside `alpha` (x1/y1 exclusive)


def render(path: str, text: str, size_px: float, hscale: float = 1.0,
           frac: tuple[float, float] = (0.0, 0.0), ss: int = 4) -> Rendered | None:
    """Render `text` as an anti-aliased coverage mask.

    `frac` is a sub-pixel offset applied before downsampling so the final
    placement can be accurate to a fraction of a pixel.
    """
    size = max(4, int(round(size_px * ss)))
    font = _pil_font(path, size)
    l, t, r, b = font.getbbox(text, anchor="ls")
    pad = 3 * ss
    w = int(r - l) + 2 * pad
    h = int(b - t) + 2 * pad
    if w <= 0 or h <= 0:
        return None
    ox = -l + pad + frac[0] * ss / max(hscale, 1e-6)
    oy = -t + pad + frac[1] * ss
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((ox, oy), text, font=font, fill=255, anchor="ls")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    out_w = max(1, int(round(w * hscale / ss)))
    out_h = max(1, int(round(h / ss)))
    small = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small > 0.25)
    if len(xs) == 0:
        return None
    ink = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Rendered(small, oy / ss * (out_h * ss / h), ink)


@dataclass
class FontStyle:
    path: str
    size_px: float
    hscale: float
    script: str            # digit script the new text must be written in
    score: float

    @property
    def name(self) -> str:
        return Path(self.path).stem


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    total = float(a.sum() + b.sum())
    return 2 * inter / total if total else 0.0


def _script_variants(text: str) -> list[tuple[str, str]]:
    lat = to_latin_digits(text)
    variants = [("latin", lat), ("persian", to_script(lat, "persian"))]
    if any("٠" <= ch <= "٩" for ch in text):
        variants.append(("arabic", to_script(lat, "arabic")))
    return variants


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / den if den else 0.0


def match_font(samples: list[tuple[np.ndarray, str]], prefer_script: str | None = None) -> FontStyle | None:
    """Pick the font (and digit script) that best reproduces the sample crops.

    `samples` are (ink coverage 0..1 or a binary mask, displayed text). Grey
    levels are compared too: at 10-15 px text sizes the anti-aliasing carries
    most of the shape information.
    """
    fonts = candidate_fonts()
    tight = []
    for cov, text in samples:
        cov = cov.astype(np.float32)
        ys, xs = np.nonzero(cov > 0.5)
        if len(xs):
            tight.append((cov[ys.min():ys.max() + 1, xs.min():xs.max() + 1], text))
    samples = tight
    if not fonts or not samples:
        return None
    best: FontStyle | None = None
    for path in fonts:
        for script, _ in _script_variants(samples[0][1]):
            scores, sizes, ratios = [], [], []
            ok = True
            for cov, text in samples:
                variant = dict(_script_variants(text)).get(script)
                if variant is None or not supports(path, variant):
                    ok = False
                    break
                h_o, w_o = cov.shape
                probe = render(path, variant, 100, ss=1)
                if probe is None:
                    ok = False
                    break
                x0, y0, x1, y1 = probe.ink
                scale = h_o / max(1, (y1 - y0))
                ratio = w_o / max(1.0, (x1 - x0) * scale)
                hs = min(max(ratio, 0.8), 1.25)
                rr = render(path, variant, 100 * scale, hscale=hs, ss=4)
                if rr is None:
                    ok = False
                    break
                rx0, ry0, rx1, ry1 = rr.ink
                crop = cv2.resize(rr.alpha[ry0:ry1, rx0:rx1], (w_o, h_o), interpolation=cv2.INTER_AREA)
                shape = 0.5 * _corr(crop, cov) + 0.5 * _dice(crop > 0.5, cov > 0.5)
                scores.append(shape - 0.35 * abs(math.log(max(ratio, 1e-3))))
                sizes.append(100 * scale)
                ratios.append(ratio)
            if not ok or not scores:
                continue
            score = float(np.mean(scores))
            if prefer_script and script == prefer_script:
                # the vision model saw the digits; trust its script unless the pixels strongly disagree
                score += 0.15
            if best is None or score > best.score:
                ratio = float(np.median(ratios))
                hscale = min(max(ratio, 0.85), 1.18) if abs(ratio - 1) > 0.02 else 1.0
                best = FontStyle(path, float(np.median(sizes)), hscale, script, score)
    if best:
        log.info("font match: %s script=%s size=%.1f hscale=%.3f score=%.3f",
                 best.name, best.script, best.size_px, best.hscale, best.score)
    return best
