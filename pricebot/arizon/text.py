"""The text a person sees in any part of a PDF page, in reading order.

Persian PDFs are full of traps for text extraction: the parts of a ligature
come out in the wrong order ("کاال" for «کالا»), zero-width glyph parts sit on
a raised baseline, MuPDF adds fake spaces around them, fake-bold text is drawn
twice, and legacy fonts (B Nazanin & co. from old converters) put gibberish in
the text layer. This module rebuilds clean, logical text from the glyphs.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pymupdf

from ..pdftext import page_chars

_ARABIC_LETTER = re.compile("[ؠ-يٮ-ۓەۺ-ۿﭐ-﷿ﹰ-ﻼ]")
_LTR = re.compile(r"[A-Za-z0-9٠-٩۰-۹À-ɏ]")
_ALEFS = "اآأإ"
_LAMS = "ل"
# Characters an old non-Unicode Persian font turns into (the text layer then reads like "Z^fYÁ").
_LEGACY = re.compile(r"[\u0080-¿À-ÿƒˆ˜‘-„†-•…‰‹›€™]")
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")
_SEPS = set(",./:-_+×*%٫٬'#")          # signs that stick to a Latin word or number
_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
                          "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
                          "‎": "", "‏": "", "‪": "", "‫": "", "‬": "",
                          "‭": "", "‮": "", "﻿": "", "­": "", "ـ": ""})


@dataclass
class Glyph:
    c: str
    x0: float
    y0: float
    x1: float
    y1: float
    base: float          # baseline y
    size: float
    bold: bool
    color: int
    seq: int             # position in the content stream

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0


@dataclass
class Unit:
    """One visible glyph plus the zero-width ligature parts that belong to it."""
    text: str
    glyphs: list[Glyph] = field(default_factory=list)
    space: bool = False

    @property
    def g(self) -> Glyph:
        return self.glyphs[0]

    @property
    def x0(self) -> float:
        return min(g.x0 for g in self.glyphs)

    @property
    def x1(self) -> float:
        return max(g.x1 for g in self.glyphs)

    @property
    def cls(self) -> str:
        """R (Persian/Arabic letter), L (Latin letter or digit), N (neutral)."""
        if self.space:
            return "N"
        if _ARABIC_LETTER.search(self.text):
            return "R"
        if _LTR.search(self.text):
            return "L"
        return "N"


def page_glyphs(page: pymupdf.Page) -> list[Glyph]:
    """Every visible, horizontal glyph of the page in content-stream order,
    without the duplicates drawn for fake bold."""
    out: list[Glyph] = []
    for k, ch in enumerate(page_chars(page)):
        b = ch.bbox
        bold = bool(re.search(r"bold|black|heavy|semibold|demi", ch.font, re.I))
        g = Glyph(ch.c, b.x0, b.y0, b.x1, b.y1, ch.origin[1], ch.size, bold, ch.color, k)
        dup = False
        for o in out[-4:]:
            if o.c == g.c and not g.c.isspace() and abs(o.x0 - g.x0) < 0.25 * g.size \
                    and abs(o.base - g.base) < 0.25 * g.size and min(o.w, g.w) > 0.08 * g.size:
                o.bold = True    # drawn twice = fake bold
                dup = True
                break
        if not dup:
            out.append(g)
    return out


def is_legacy(text: str) -> bool:
    """True when a text layer is gibberish from an old non-Unicode Persian font."""
    letters = [c for c in text if c.isalpha() or _LEGACY.match(c)]
    if len(letters) < 4:
        return False
    legacy = sum(1 for c in letters if _LEGACY.match(c))
    return legacy >= 0.25 * len(letters)


# ================================================================ units ==

def _units(glyphs: list[Glyph]) -> list[Unit]:
    """Glyphs (stream order) -> units, zero-width parts attached to their glyph."""
    units: list[Unit] = []
    pending_alef: Glyph | None = None
    for i, g in enumerate(glyphs):
        if g.c.isspace():
            if g.w > 0.1 * g.size:      # zero-width "spaces" are artefacts of ligatures
                units.append(Unit(" ", [g], space=True))
            continue
        zero = g.w < 0.08 * g.size
        if zero and g.c in _ALEFS:
            nxt = glyphs[i + 1] if i + 1 < len(glyphs) else None
            if nxt is not None and nxt.c in _LAMS and abs(nxt.x1 - g.x0) < 0.35 * g.size:
                pending_alef = g            # «لا» written as alef + lam: put it back in order
                continue
        if pending_alef is not None:
            if g.c in _LAMS:
                units.append(Unit(g.c + pending_alef.c, [g, pending_alef]))
                pending_alef = None
                continue
            units.append(Unit(pending_alef.c, [pending_alef]))
            pending_alef = None
        host = next((u for u in reversed(units[-3:]) if not u.space), None)
        if zero and host is not None and abs(host.g.base - g.base) < 1.2 * g.size:
            host.text += g.c
            host.glyphs.append(g)
            continue
        units.append(Unit(g.c, [g]))
    if pending_alef is not None:
        units.append(Unit(pending_alef.c, [pending_alef]))
    # spaces MuPDF invents over a glyph (around zero-width parts) are not real
    solid = [u for u in units if not u.space]
    kept = []
    for u in units:
        if u.space:
            s = u.g
            cover = sum(max(0.0, min(s.x1, v.x1) - max(s.x0, v.x0)) for v in solid
                        if abs(v.g.base - s.base) < 0.6 * s.size)
            if s.w > 0 and cover > 0.75 * s.w:
                continue
        kept.append(u)
    return kept


def _lines(units: list[Unit]) -> list[list[Unit]]:
    """Units grouped into text lines (top to bottom), stream order kept inside."""
    lines: list[list[Unit]] = []
    for u in units:
        base, size = u.g.base, u.g.size
        for ln in lines:
            ref = next((v for v in ln if not v.space), ln[0])
            if abs(ref.g.base - base) <= 0.45 * max(size, ref.g.size):
                ln.append(u)
                break
        else:
            lines.append([u])
    lines.sort(key=lambda ln: min(v.g.base for v in ln))
    return lines


def _is_latin(u: Unit) -> bool:
    return any(ch.isalpha() and not _ARABIC_LETTER.match(ch) for ch in u.text)


def _line_text(line: list[Unit]) -> str:
    """The text of one line. Some PDFs name a bracket by the shape drawn (so a
    bracket in right-to-left text reads mirrored), others by its meaning: both
    readings are made and the one whose brackets pair up is kept."""
    plain = _line_text_as(line, False)
    if not any(c in plain for c in "()[]{}«»"):
        return plain
    mirrored = _line_text_as(line, True)
    best = mirrored if _bracket_errors(mirrored) < _bracket_errors(plain) else plain
    return _pair_brackets(best) if _bracket_errors(best) else best


def _pair_brackets(text: str) -> str:
    """Brackets that still do not pair up: turn the fewest of them around."""
    at = [k for k, c in enumerate(text) if c in "()"]
    if not at or len(at) > 8:
        return text
    best, best_key = text, (_bracket_errors(text), 0)
    for mask in range(1, 1 << len(at)):
        chars = list(text)
        for b, k in enumerate(at):
            if mask >> b & 1:
                chars[k] = chars[k].translate(_MIRROR)
        cand = "".join(chars)
        key = (_bracket_errors(cand), bin(mask).count("1"))
        if key < best_key:
            best, best_key = cand, key
    return best


def _bracket_errors(text: str) -> int:
    errors = 0
    depth = {"(": 0, "[": 0, "{": 0, "«": 0}
    close = {")": "(", "]": "[", "}": "{", "»": "«"}
    for c in text:
        if c in depth:
            depth[c] += 1
        elif c in close:
            if depth[close[c]]:
                depth[close[c]] -= 1
            else:
                errors += 1
    errors += sum(depth.values())
    # «()» and «( word» at the end of a line are signs of a wrong reading too
    return errors + text.count("()") + text.count("[]")


def _line_text_as(line: list[Unit], mirror: bool) -> str:
    """Units of one text line -> reading order, from their positions alone.

    The content stream order cannot be trusted (Word writes «1/754/500» as
    500, 754, 1 and puts a closing bracket before the words), but the page
    positions can: sort by x to get the visual order, then undo the bidi
    reordering for right-to-left lines."""
    solid = [u for u in line if not u.space]
    if not solid:
        return ""
    size = max(u.g.size for u in solid)
    # a visible gap is a word break too (titles often have no space glyph)
    # (with real space glyphs on the line only a clearly wider gap counts: justified text
    # stretches the gaps inside words too)
    gap_min = (0.45 if any(u.space for u in line) else 0.22) * size
    vis = sorted(line, key=lambda u: (u.g.xc, u.g.seq))
    # visual string: glyph units plus the spaces seen on the page
    seq: list[Unit | None] = []          # None = a space
    for k, u in enumerate(vis):
        if u.space:
            if seq and seq[-1] is not None:
                seq.append(None)
            continue
        if seq and seq[-1] is not None:
            p = seq[-1]
            if u.x0 - p.x1 > gap_min:
                seq.append(None)
        seq.append(u)
    while seq and seq[-1] is None:
        seq.pop()
    if not any(u is not None and u.cls == "R" for u in seq):
        return "".join(" " if u is None else u.text for u in seq)
    # right-to-left line: reverse, then turn left-to-right runs back. A run
    # keeps exactly the order seen on the page, so the template shows the same
    # thing (e.g. «DM_ 4 PK 855 _Tu5/R2» written by Excel inside Persian text).
    rev = seq[::-1]
    out: list[str] = []
    i = 0
    while i < len(rev):
        u = rev[i]
        if u is None or u.cls != "L":
            # a bracket outside a left-to-right run is drawn mirrored in right-to-left text
            out.append(" " if u is None else u.text.translate(_MIRROR) if mirror else u.text)
            i += 1
            continue
        j = i + 1
        while j < len(rev):
            v = rev[j]
            if v is not None and v.cls == "L":
                j += 1
                continue
            nxt = rev[j + 1] if j + 1 < len(rev) else None
            glued = v is not None and v.text in _SEPS
            if nxt is not None and nxt.cls == "L" and (v is None or glued):
                j += 2
                continue
            if glued and (nxt is None or nxt.cls != "R"):
                j += 1          # «DM_», «(+)»: a sign stuck to a Latin word
                continue
            break
        run = rev[i:j]
        if any(x is not None and _is_latin(x) for x in run):
            out.extend(" " if x is None else x.text for x in run[::-1])
        else:
            # only numbers: each space-separated number reads left to right,
            # the numbers themselves follow the right-to-left line
            groups: list[list[Unit]] = [[]]
            for x in run:
                if x is None:
                    groups.append([])
                else:
                    groups[-1].append(x)
            out.append(" ".join("".join(x.text for x in grp[::-1]) for grp in groups if grp))
        i = j
    return "".join(out)


# ============================================================ public API ==

def clean(text: str) -> str:
    """Normal Persian letters and digits, tidy spaces and brackets."""
    t = unicodedata.normalize("NFKC", text or "").translate(_PERSIAN)
    t = re.sub(r"[ \t  -​]+", " ", t).strip()
    t = re.sub(r"([ؠ-ي٠-ۿ])\(", r"\1 (", t)       # Persian word(…) -> word (…)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\s+([،,:؛.])(\s|$)", r"\1\2", t)
    return t


def text_of(glyphs: list[Glyph]) -> str:
    """Glyphs of one region (e.g. a table cell) -> its text, lines joined."""
    if not glyphs:
        return ""
    units = _units(sorted(glyphs, key=lambda g: g.seq))
    lines = [_line_text(ln) for ln in _lines(units)]
    return clean(" ".join(x for x in lines if x))


def lines_of(glyphs: list[Glyph]) -> list[tuple[str, float, float, bool]]:
    """Glyphs -> text lines as (text, top y, font size, bold), top to bottom."""
    units = _units(sorted(glyphs, key=lambda g: g.seq))
    out = []
    for ln in _lines(units):
        t = clean(_line_text(ln))
        if t:
            solid = [u for u in ln if not u.space] or ln
            size = max(u.g.size for u in solid)
            bold = sum(u.g.bold for u in solid) >= 0.5 * len(solid)
            out.append((t, min(u.g.y0 for u in solid), size, bold))
    return out


def rotated_lines(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Vertical text (e.g. a group name written sideways in a merged cell)."""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            if abs(line["dir"][1]) < 0.5:
                continue
            text = clean("".join(s["text"] for s in line["spans"]))
            if text and not is_legacy(text):
                out.append((pymupdf.Rect(line["bbox"]), text))
    return out
