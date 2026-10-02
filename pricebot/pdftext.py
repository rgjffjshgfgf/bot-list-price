"""Numbers in a PDF's text layer: extraction with exact geometry, and in-place
replacement that keeps the original font, size, colour, position and the
cell background untouched.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

from . import fonts
from .numfmt import DIGIT_CHARS, SEP_CHARS, SPACE_CHARS, ParsedNumber, parse_number, to_latin_digits, to_script

log = logging.getLogger(__name__)

CLIPPED = 64
FILLED = 16
STROKED = 32
BOLD = 16              # in a span's "flags" (not its "char_flags")


@dataclass
class Char:
    c: str
    bbox: pymupdf.Rect
    origin: tuple[float, float]
    size: float
    font: str
    color: int
    char_flags: int
    alpha: int
    bold: bool = False


@dataclass
class TextToken:
    page: int
    chars: list[Char]
    shadows: list[Char]           # duplicate glyphs drawn on top (fake bold etc.)
    parsed: ParsedNumber
    attached: bool = False        # glued to letters, e.g. "L90", "پژو405"
    align: str = "center"
    bounds: tuple[float, float] = (0.0, 0.0)

    @property
    def text(self) -> str:
        return "".join(ch.c for ch in self.chars)

    @property
    def bbox(self) -> pymupdf.Rect:
        r = pymupdf.Rect(self.chars[0].bbox)
        for ch in self.chars[1:]:
            r |= ch.bbox
        return r

    @property
    def size(self) -> float:
        return self.chars[0].size

    @property
    def font(self) -> str:
        return self.chars[0].font

    @property
    def bold(self) -> bool:
        return self.chars[0].bold

    @property
    def baseline(self) -> float:
        return self.chars[0].origin[1]


def _visible(span: dict) -> bool:
    flags = span.get("char_flags")
    if flags is not None:
        if flags & CLIPPED:
            return False
        if not flags & (FILLED | STROKED):   # render mode 3: invisible OCR layer
            return False
    if span.get("alpha", 255) == 0:
        return False
    return True


def page_chars(page: pymupdf.Page) -> list[Char]:
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_MEDIABOX_CLIP
    raw = page.get_text("rawdict", flags=flags)
    out: list[Char] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            dx, dy = line["dir"]
            if abs(dy) > 0.01 or dx < 0:
                continue   # rotated text is left alone
            for span in line["spans"]:
                if not _visible(span) or span["size"] < 2:
                    continue
                for ch in span["chars"]:
                    out.append(Char(ch["c"], pymupdf.Rect(ch["bbox"]), tuple(ch["origin"]), span["size"],
                                    span["font"], span["color"], span.get("char_flags", FILLED),
                                    span.get("alpha", 255), bool(span.get("flags", 0) & BOLD)))
    return out


def _lines(chars: list[Char]) -> list[list[Char]]:
    chars = sorted(chars, key=lambda c: (c.origin[1], c.bbox.x0))
    lines: list[list[Char]] = []
    for ch in chars:
        placed = False
        for ln in reversed(lines[-6:]):
            ref = ln[0]
            if abs(ch.origin[1] - ref.origin[1]) <= 0.3 * ref.size and 0.75 <= ch.size / ref.size <= 1.33:
                ln.append(ch)
                placed = True
                break
        if not placed:
            lines.append([ch])
    for ln in lines:
        ln.sort(key=lambda c: c.bbox.x0)
    return lines


def _is_num(c: str) -> bool:
    return c in DIGIT_CHARS or c in SEP_CHARS


def _is_letter(c: str) -> bool:
    return unicodedata.category(c).startswith("L")


def extract_tokens(page: pymupdf.Page, page_index: int) -> list[TextToken]:
    tokens: list[TextToken] = []
    for line in _lines(page_chars(page)):
        # Whitespace glyphs are unreliable (Excel even places them inside numbers);
        # word breaks are judged from the visible gaps instead.
        line = [c for c in line if c.c.strip()]
        # drop duplicated glyphs (same char drawn twice at almost the same spot)
        kept: list[Char] = []
        shadows: dict[int, list[Char]] = {}
        for ch in line:
            dup = None
            for k in range(len(kept) - 1, max(-1, len(kept) - 4), -1):
                o = kept[k]
                if o.c == ch.c and abs(o.bbox.x0 - ch.bbox.x0) < 0.25 * o.size \
                        and abs(o.origin[1] - ch.origin[1]) < 0.25 * o.size and ch.c.strip():
                    dup = k
                    break
            if dup is None:
                kept.append(ch)
            else:
                shadows.setdefault(id(kept[dup]), []).append(ch)

        i = 0
        while i < len(kept):
            if kept[i].c not in DIGIT_CHARS:
                i += 1
                continue
            # extend left over separators? numbers start with a digit, so no.
            run = [kept[i]]
            j = i + 1
            while j < len(kept) and _is_num(kept[j].c):
                prev = run[-1]
                gap = kept[j].bbox.x0 - prev.bbox.x1
                if gap > 0.6 * prev.size:
                    break
                run.append(kept[j])
                j += 1
            # trim trailing separators
            while run and run[-1].c not in DIGIT_CHARS:
                run.pop()
            before = kept[i - 1] if i > 0 else None
            after = kept[i + len(run)] if i + len(run) < len(kept) else None
            for sub in _split_run(run):
                tok = _make_token(page_index, sub, shadows)
                if tok is None:
                    continue
                first, last = sub[0], sub[-1]
                b = before if sub[0] is run[0] else None
                a = after if sub[-1] is run[-1] else None
                if b is not None and _is_letter(b.c) and first.bbox.x0 - b.bbox.x1 < 0.12 * first.size:
                    tok.attached = True
                if a is not None and _is_letter(a.c) and a.bbox.x0 - last.bbox.x1 < 0.12 * last.size:
                    tok.attached = True
                tokens.append(tok)
            i = j
    return tokens


def _run_text(run: list[Char]) -> str:
    """Text of a run, with a virtual space where glyphs are visibly apart."""
    out = []
    for k, ch in enumerate(run):
        if k and ch.c not in SEP_CHARS and run[k - 1].c not in SEP_CHARS:
            if ch.bbox.x0 - run[k - 1].bbox.x1 > 0.18 * ch.size:
                out.append(" ")
        out.append(ch.c)
    return "".join(out)


def _split_run(run: list[Char]) -> list[list[Char]]:
    if not run:
        return []
    if parse_number(_run_text(run)) is not None:
        return [run]
    # "100 3,696,000" style: split at spaces / visible gaps and try each part
    parts: list[list[Char]] = [[]]
    for k, ch in enumerate(run):
        gap = ch.bbox.x0 - run[k - 1].bbox.x1 if k else 0
        if ch.c in SPACE_CHARS or (k and gap > 0.18 * ch.size and ch.c not in SEP_CHARS
                                    and run[k - 1].c not in SEP_CHARS):
            if parts[-1]:
                parts.append([])
            if ch.c in SPACE_CHARS:
                continue
        parts[-1].append(ch)
    out = []
    for p in parts:
        while p and p[-1].c not in DIGIT_CHARS:
            p.pop()
        while p and p[0].c not in DIGIT_CHARS:
            p.pop(0)
        if p and parse_number(_run_text(p)) is not None:
            out.append(p)
    return out


def _make_token(page_index: int, run: list[Char], shadows: dict[int, list[Char]]) -> TextToken | None:
    run = [ch for ch in run if ch.c not in SPACE_CHARS]
    if not run:
        return None
    parsed = parse_number(_run_text(run))
    if parsed is None:
        return None
    shadow = [s for ch in run for s in shadows.get(id(ch), [])]
    return TextToken(page_index, run, shadow, parsed)


# ------------------------------------------------------------ geometry ----

def vertical_edges(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """(x, y0, y1) of vertical rules and cell edges drawn on the page."""
    edges = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - broken vector content should not stop us
        return edges
    for d in drawings:
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) < 0.8 and abs(p1.y - p2.y) > 2:
                    edges.append((p1.x, min(p1.y, p2.y), max(p1.y, p2.y)))
            elif kind == "re":
                r = item[1]
                if r.height < 2:
                    continue
                if r.width < 1.6:
                    edges.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
                else:
                    edges.append((r.x0, r.y0, r.y1))
                    edges.append((r.x1, r.y0, r.y1))
            elif kind == "qu":
                q = item[1]
                r = q.rect
                if r.height >= 2:
                    edges.append((r.x0, r.y0, r.y1))
                    edges.append((r.x1, r.y0, r.y1))
    return edges


def compute_layout(page: pymupdf.Page, tokens: list[TextToken], all_chars: list[Char] | None = None) -> None:
    """Fill `bounds` (free horizontal space) and `align` for each token."""
    if not tokens:
        return
    edges = vertical_edges(page)
    chars = all_chars if all_chars is not None else page_chars(page)
    width = page.rect.width
    for t in tokens:
        bb = t.bbox
        yc = (bb.y0 + bb.y1) / 2
        left, right = 0.0, width
        for x, y0, y1 in edges:
            if y0 - 0.5 <= yc <= y1 + 0.5:
                if x <= bb.x0 + 0.3 and x > left:
                    left = x
                elif x >= bb.x1 - 0.3 and x < right:
                    right = x
        own = {id(c) for c in t.chars} | {id(c) for c in t.shadows}
        for ch in chars:
            if id(ch) in own or not ch.c.strip():
                continue
            if abs(ch.origin[1] - t.baseline) > 0.5 * t.size and not (ch.bbox.y0 < yc < ch.bbox.y1):
                continue
            if ch.bbox.x1 <= bb.x0 + 0.1 and ch.bbox.x1 > left:
                left = ch.bbox.x1
            elif ch.bbox.x0 >= bb.x1 - 0.1 and ch.bbox.x0 < right:
                right = ch.bbox.x0
        t.bounds = (left, right)

    # alignment: prefer what the whole column does, else the cell gaps
    cols = _columns(tokens)
    for col in cols:
        align = None
        ws = [t.bbox.width for t in col]
        if len(col) >= 2 and max(ws) - min(ws) > 0.3 * col[0].size:
            s0 = np.std([t.bbox.x0 for t in col])
            s1 = np.std([t.bbox.x1 for t in col])
            sc = np.std([(t.bbox.x0 + t.bbox.x1) / 2 for t in col])
            best = min((s1, "right"), (sc, "center"), (s0, "left"))
            if best[0] < 0.2 * col[0].size:
                align = best[1]
        for t in col:
            if align:
                t.align = align
                continue
            bb = t.bbox
            gl, gr = bb.x0 - t.bounds[0], t.bounds[1] - bb.x1
            span = t.bounds[1] - t.bounds[0]
            if abs(gl - gr) <= max(1.0, 0.12 * span):
                t.align = "center"
            else:
                t.align = "right" if gr < gl else "left"


def _columns(tokens: list[TextToken]) -> list[list[TextToken]]:
    cols: list[list[TextToken]] = []
    for t in sorted(tokens, key=lambda t: t.bbox.x0):
        bb = t.bbox
        for col in cols:
            ref = col[-1].bbox
            ov = min(bb.x1, ref.x1) - max(bb.x0, ref.x0)
            if ov >= 0.3 * min(bb.width, ref.width) and t.page == col[-1].page:
                col.append(t)
                break
        else:
            cols.append([t])
    return cols


# ------------------------------------------------------------ rewriting ---

_SUBSET = re.compile(r"^[A-Z]{6}\+")
_BASE14 = [("courier", "cour", "cobo"), ("times", "tiro", "tibo"),
           ("arial", "helv", "hebo"), ("helvetica", "helv", "hebo")]


def _norm_font(name: str) -> str:
    return re.sub(r"[\s,_-]", "", _SUBSET.sub("", name)).lower()


class FontBank:
    """Finds a font that can draw a replacement number in a given span's style."""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._embedded: dict[str, list[tuple[int, bytes, pymupdf.Font]]] = {}
        self._loaded = False
        self._registered: dict[tuple[int, str], str] = {}
        self._counter = 0
        self._files: dict[str, pymupdf.Font] = {}
        self._styles: dict[tuple[str, float], fonts.FontStyle | None] = {}
        self.borrowed: set[str] = set()   # fonts whose own copy is not used for any number

    def plan(self, reps: list["Replacement"]) -> None:
        """Before anything is written: a font whose embedded copy cannot draw every
        new number of the document draws none of them - all its numbers take the
        same substitute, so a price column never mixes two typefaces."""
        for rep in reps:
            font = rep.token.font
            if font not in self.borrowed and self.embedded_for(font, rep.new_text) is None:
                self.borrowed.add(font)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        seen = set()
        for page in self.doc:
            for f in page.get_fonts(full=True):
                xref, basefont = f[0], f[3]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    _, _, _, buf = self.doc.extract_font(xref)
                    if not buf:
                        continue
                    font = pymupdf.Font(fontbuffer=buf)
                except Exception:  # noqa: BLE001 - unusable embedded font
                    continue
                self._embedded.setdefault(_norm_font(basefont), []).append((xref, buf, font))

    def embedded_for(self, span_font: str, text: str) -> tuple[int, bytes, pymupdf.Font] | None:
        self._load()
        key = _norm_font(span_font)
        for cand_key, fonts_ in self._embedded.items():
            if cand_key == key or cand_key.endswith(key) or key.endswith(cand_key):
                for entry in fonts_:
                    if all(entry[2].has_glyph(ord(ch)) for ch in text):
                        return entry
        return None

    def font_file(self, path: str) -> pymupdf.Font:
        if path not in self._files:
            self._files[path] = pymupdf.Font(fontfile=path)
        return self._files[path]

    def look_alike(self, page: pymupdf.Page, tok: TextToken) -> fonts.FontStyle | None:
        """The installed font that looks most like the number as the page shows it
        (asked before the number is erased; once per font and size of a document)."""
        key = (tok.font, round(tok.size, 1))
        if key not in self._styles:
            self._styles[key] = _fallback_style(page, tok)
        return self._styles[key]

    def register(self, page: pymupdf.Page, key: str, **font_src) -> str:
        reg = (page.number, key)
        if reg not in self._registered:
            self._counter += 1
            alias = f"PBF{self._counter}"
            page.insert_font(fontname=alias, **font_src)
            self._registered[reg] = alias
        return self._registered[reg]


@dataclass
class Replacement:
    token: TextToken
    new_text: str                 # already formatted like the original
    result: dict = field(default_factory=dict)


def _base14_for(font_name: str) -> str | None:
    low = font_name.lower()
    bold = "bold" in low or "bd" in low
    for key, regular, boldname in _BASE14:
        if key in low:
            return boldname if bold else regular
    return None


def _fallback_style(page: pymupdf.Page, tok: TextToken) -> fonts.FontStyle | None:
    """Match a system font against how the number actually looks on the page."""
    bb = tok.bbox
    zoom = max(2.0, 48.0 / max(1.0, bb.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=bb, alpha=False, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    bg = float(np.median(arr))
    cov = np.clip((bg - arr.astype(np.float32)) / max(1.0, bg - float(arr.min())), 0, 1)
    if not (cov > 0.5).any():
        return None
    style = fonts.match_font([(cov, tok.text)])
    if style:
        style.size_px /= zoom
    return style


@dataclass
class _Pen:
    """How one replacement number is drawn: the font, and the text in its digits."""
    key: str                          # registration key of the font on the page
    src: dict                         # pymupdf insert_font arguments; empty for a built-in font
    base14: str | None                # a built-in PDF font, used by name
    measure: pymupdf.Font
    same: pymupdf.Font | None         # the original font (or its twin): its widths give the spacing
    text: str
    size: float
    hscale: float = 1.0
    how: str = ""


_ASSETS = Path(__file__).resolve().parent / "arizon" / "assets"
_LATIN_ONLY = re.compile(r"^[0-9,./ \-]*$")


def _last_resort(tok: TextToken, new_text: str) -> _Pen:
    """A font that can always draw the number: Helvetica for Latin digits, the
    bundled Vazirmatn for Persian/Arabic ones."""
    if _LATIN_ONLY.match(new_text):
        name = "hebo" if tok.bold else "helv"
        f = pymupdf.Font(name)
        return _Pen(name, {}, name, f, None, new_text, tok.size, how="last-resort")
    path = str(_ASSETS / ("Vazirmatn-Bold.ttf" if tok.bold else "Vazirmatn-Regular.ttf"))
    f = pymupdf.Font(fontfile=path)
    text = new_text if all(f.has_glyph(ord(ch)) or ch.isspace() for ch in new_text) else to_latin_digits(new_text)
    return _Pen(path, {"fontfile": path}, None, f, None, text, tok.size, how="last-resort")


def _choose(page: pymupdf.Page, tok: TextToken, new_text: str, bank: FontBank) -> _Pen:
    """The font a number is redrawn in, chosen while the page still shows the old number:
    1. the PDF's own font, when its embedded copy has every glyph the document's new
       numbers need (see FontBank.plan);
    2. the built-in twin of Arial/Times/Courier;
    3. an installed font of the same family and widths (Excel's Calibri -> Carlito...);
    4. the installed font that looks most like the number as drawn on the page;
    5. a font that can always draw it."""
    entry = None if tok.font in bank.borrowed else bank.embedded_for(tok.font, new_text)
    if entry is not None:
        xref, buf, font = entry
        return _Pen(f"x{xref}", {"fontbuffer": buf}, None, font, font, new_text, tok.size, how="own")
    base14 = _base14_for(tok.font)
    if base14 and _LATIN_ONLY.match(new_text):
        f = pymupdf.Font(base14)
        return _Pen(base14, {}, base14, f, f, new_text, tok.size, how="base14")
    twin = fonts.twin_font(tok.font, tok.bold, new_text)
    if twin is not None:
        f = bank.font_file(twin)
        return _Pen(twin, {"fontfile": twin}, None, f, f, new_text, tok.size, how="twin")
    style = bank.look_alike(page, tok)
    if style is not None:
        text = to_script(to_latin_digits(new_text), style.script)
        if fonts.supports(style.path, text):
            return _Pen(style.path, {"fontfile": style.path}, None, bank.font_file(style.path), None, text,
                        style.size_px, style.hscale, how="look-alike")
    return _last_resort(tok, new_text)


def apply_page(page: pymupdf.Page, reps: list[Replacement], bank: FontBank) -> list[str]:
    """Replace numbers on one page. Returns warnings (in Persian, for the user).
    Every number's font is chosen before anything is erased, and a number that
    cannot be redrawn keeps its old value on the page - a price is never lost."""
    warnings: list[str] = []
    if not reps:
        return warnings
    pens: list[_Pen | None] = []
    for rep in reps:
        try:
            pens.append(_choose(page, rep.token, rep.new_text, bank))
        except Exception:  # noqa: BLE001 - this number stays as it is
            log.exception("no font for %s", rep.token.text)
            pens.append(None)
    for rep, pen in zip(reps, pens):
        if pen is None:
            warnings.append(f"صفحه {page.number + 1}: قیمت {rep.token.text} عوض نشد (فونت مناسب پیدا نشد) "
                            "و همان قیمت قبلی ماند.")
            continue
        for ch in rep.token.chars + rep.token.shadows:
            r = ch.bbox
            cx, w, h = (r.x0 + r.x1) / 2, r.width, r.height
            page.add_redact_annot(pymupdf.Rect(cx - 0.3 * w, r.y0 + 0.3 * h, cx + 0.3 * w, r.y1 - 0.3 * h),
                                  fill=False, cross_out=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)

    for rep, pen in zip(reps, pens):
        if pen is None:
            continue
        rep.result["font"] = pen.how
        try:
            _insert(page, rep.token, pen, bank)
        except Exception:  # noqa: BLE001 - the old number is gone: draw it with the font that always works
            log.exception("drawing %s with %s failed", rep.new_text, pen.key)
            try:
                _insert(page, rep.token, _last_resort(rep.token, rep.new_text), bank)
                rep.result["font"] = "last-resort"
            except Exception:  # noqa: BLE001
                log.exception("drawing %s failed", rep.new_text)
                warnings.append(f"صفحه {page.number + 1}: قیمت جدید {rep.new_text} نوشته نشد.")
    return warnings


def _insert(page: pymupdf.Page, tok: TextToken, pen: _Pen, bank: FontBank) -> None:
    size = pen.size
    ch0 = tok.chars[0]
    color = pymupdf.sRGB_to_pdf(ch0.color)
    flags = ch0.char_flags
    render_mode = 0
    if flags & STROKED and flags & FILLED:
        render_mode = 2
    elif flags & STROKED:
        render_mode = 1

    text = pen.text
    hscale = pen.hscale
    alias = pen.base14 or bank.register(page, pen.key, **pen.src)
    measure = pen.measure

    # Reproduce any character spacing / horizontal scaling of the original.
    if pen.same is not None:
        natural_old = pen.same.text_length(tok.text, size)
        old_w = tok.chars[-1].bbox.x1 - tok.chars[0].bbox.x0
        ratio = old_w / natural_old if natural_old else 1.0
        if 0.7 < ratio < 1.3 and abs(ratio - 1) > 0.02:
            hscale = ratio
    new_w = measure.text_length(text, size) * hscale

    left_bound, right_bound = tok.bounds
    avail = right_bound - left_bound - 0.3 * size
    if avail > 0 and new_w > avail:
        shrink = avail / new_w
        new_hs = max(hscale * shrink, 0.85 * hscale)
        rest = shrink * hscale / new_hs
        hscale = new_hs
        if rest < 1:
            size *= rest
        new_w = measure.text_length(text, size) * hscale

    bb = tok.bbox
    if tok.align == "right":
        x = bb.x1 - new_w
    elif tok.align == "left":
        x = bb.x0
    else:
        x = (bb.x0 + bb.x1) / 2 - new_w / 2
    if right_bound > left_bound:
        x = min(max(x, left_bound + 0.1 * size), right_bound - 0.1 * size - new_w)
    y = tok.baseline
    shadows = sorted({(round(s.origin[0] - c.origin[0], 3), round(s.origin[1] - c.origin[1], 3))
                      for c in tok.chars for s in tok.shadows if s.c == c.c
                      and abs(s.bbox.x0 - c.bbox.x0) < 0.25 * c.size})
    for dx, dy in [(0.0, 0.0)] + [o for o in shadows if o != (0.0, 0.0)]:
        pt = pymupdf.Point(x + dx, y + dy)
        morph = (pt, pymupdf.Matrix(hscale, 0, 0, 1, 0, 0)) if hscale != 1.0 else None
        page.insert_text(pt, text, fontsize=size, fontname=alias, color=color,
                         render_mode=render_mode, border_width=0.03, morph=morph)
