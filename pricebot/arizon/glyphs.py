"""Naming glyphs a PDF leaves unnamed.

Some PDFs (old Word files with B Nazanin & co., some Calibri exports) draw
text with glyphs that have no character attached: the text layer reads
"�". The glyph's outline, however, is in the embedded font. The same
outline in another PDF that did name it tells which letter it is, so a memory
of outline -> character (assets/glyphs.json, built from real price lists by
tools/learn_glyphs.py) names it back. Exact outlines only: a glyph that is
not in the memory stays unnamed (and its cell is shown as a picture or read by
Gemini).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

MEMORY = Path(__file__).resolve().parent / "assets" / "glyphs.json"
# one letter written with different code points by different programs counts as one
_SAME = str.maketrans({**{a: b for a, b in zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")},
                       "ي": "ی", "ى": "ی", "ك": "ک"})


def _canon(ch: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", ch).translate(_SAME)


@lru_cache(maxsize=1)
def memory() -> dict[str, str]:
    try:
        return json.loads(MEMORY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def outline_key(font, gid: int) -> str | None:
    """A short fingerprint of a glyph's outline (None for an empty glyph)."""
    from fontTools.pens.recordingPen import DecomposingRecordingPen

    try:
        name = font.getGlyphOrder()[gid]
        glyph_set = font.getGlyphSet()
        pen = DecomposingRecordingPen(glyph_set)    # composite glyphs drawn out in full
        glyph_set[name].draw(pen)
    except Exception:  # noqa: BLE001 - odd or missing glyph
        return None
    if not pen.value:
        return None
    data = repr([(op, [tuple(round(v) for v in p) for p in pts]) for op, pts in pen.value])
    return hashlib.md5(data.encode()).hexdigest()[:16]


def load_font(doc: pymupdf.Document, xref: int):
    from fontTools.ttLib import TTFont

    try:
        buf = doc.extract_font(xref)[3]
        return TTFont(io.BytesIO(buf), lazy=True) if buf else None
    except Exception:  # noqa: BLE001 - Type3 / broken fonts
        return None


def _font_name(name: str) -> str:
    return name.split("+", 1)[-1]


def page_fonts(page: pymupdf.Page) -> dict[str, int]:
    """Font name as the text trace reports it -> xref of the embedded font."""
    out = {}
    for f in page.get_fonts(full=True):
        xref, base = f[0], f[3]
        out.setdefault(_font_name(base), xref)
        out.setdefault(base, xref)
    return out


def unnamed_glyphs(page: pymupdf.Page) -> dict[tuple[float, float], str]:
    """Origin (x, y) of every glyph the page does not name, mapped to the
    character the memory knows for its outline."""
    mem = memory()
    if not mem:
        return {}
    try:
        trace = page.get_texttrace()
    except Exception:  # noqa: BLE001
        return {}
    wanted = [(sp, c) for sp in trace for c in sp["chars"] if c[0] == 0xFFFD or 0xE000 <= c[0] <= 0xF8FF]
    if not wanted:
        return {}
    xrefs = page_fonts(page)
    fonts: dict[str, object] = {}
    keys: dict[tuple[str, int], str | None] = {}
    out: dict[tuple[float, float], str] = {}
    for sp, c in wanted:
        name = sp["font"]
        if name not in fonts:
            xref = xrefs.get(name) or xrefs.get(_font_name(name))
            fonts[name] = load_font(page.parent, xref) if xref else None
        font = fonts[name]
        if font is None:
            continue
        k = (name, c[1])
        if k not in keys:
            keys[k] = outline_key(font, c[1])
        ch = mem.get(keys[k] or "")
        if ch:
            out[(round(c[2][0], 1), round(c[2][1], 1))] = ch
    if out:
        log.debug("glyph memory named %d of %d glyphs on page %d", len(out), len(wanted), page.number + 1)
    return out


def learn(docs: list[pymupdf.Document]) -> dict[str, str]:
    """Outline -> character from PDFs that name their glyphs, keeping only
    outlines always named the same way (digits in any script count as one)."""
    from collections import Counter, defaultdict  # noqa: F401

    seen: dict[str, Counter] = defaultdict(Counter)
    for doc in docs:
        cache: dict[int, object] = {}
        for page in doc:
            xrefs = page_fonts(page)
            for sp in page.get_texttrace():
                xref = xrefs.get(sp["font"]) or xrefs.get(_font_name(sp["font"]))
                if not xref:
                    continue
                if xref not in cache:
                    cache[xref] = load_font(doc, xref)
                font = cache[xref]
                if font is None:
                    continue
                # one glyph can stand for several letters (the «لا» ligature): the PDF then
                # lists the letters one after the other at the same place with the same glyph
                runs: list[list] = []
                for c in sp["chars"]:
                    if runs and (runs[-1][0][1] == c[1] or c[1] < 0) and abs(runs[-1][0][2][0] - c[2][0]) < 0.05:
                        runs[-1].append(c)
                    else:
                        runs.append([c])
                for run in runs:
                    c = run[0]
                    text = "".join(chr(x[0]) for x in run)
                    if any(x[0] == 0xFFFD or 0xE000 <= x[0] <= 0xF8FF for x in run) or text.isspace():
                        continue
                    if c[3][2] - c[3][0] < 0.05 * sp["size"]:
                        continue        # zero-width parts (dots, marks) are not letters
                    key = outline_key(font, c[1])
                    if key:
                        seen[key]["".join(_canon(x) for x in text)] += 1
    out = {}
    for key, counts in seen.items():
        ch, n = counts.most_common(1)[0]
        if n >= 2 and n >= 0.97 * sum(counts.values()) and len(ch) <= 3 and not ch.isspace():
            out[key] = ch
    return out
