"""Reading a price list into clean tables for the Arizon template.

Text PDFs are read from their own text layer (exact words, no AI, a page in
milliseconds). The table grid comes from the ruling lines, or - when a list
has no vertical lines - from the columns its words line up in. Merged
"group" columns, group-title rows, header rows repeated on every page and
lists printed as two or three tables side by side all become one clean
table. Photos, scans and PDFs whose text layer is gibberish are transcribed
by Gemini (or by the OCR of the bot's own AI); a page that still cannot be
rebuilt is kept as a picture, so nothing is ever lost.
"""
from __future__ import annotations

import bisect
import io
import logging
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import pymupdf

from ..models import Analysis
from .model import Cell, Content, Frame, Row, Table
from .text import Glyph, clean, is_legacy, page_glyphs, rotated_lines, text_of

log = logging.getLogger(__name__)
if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()

FrameSource = Callable[[int], Frame | None]
CROP_DPI = 170          # product photos and text pictures cut out of a page (render.SNAP_DPI matches)

_ARABIC = re.compile("[ؠ-يٮ-ۓۺ-ۿ]")
_LATIN = re.compile("[A-Za-z]")
_CANON = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "‌": ""})

# header words, matched inside a normalised header text
_ROLE_WORDS = {
    "row": ["ردیف", "ردبف", "row", "#"],
    "code": ["کد", "شماره", "فنی", "بارکد", "پارت", "code", "part", "sku", "ref", "barcode"],
    "price": ["قیمت", "مبلغ", "فی", "بها", "price", "ریال", "تومان", "جدید", "لیست", "فروش", "همکار", "مصرف",
              "نماینده", "عمده", "amount", "cost"],
    "qty": ["تعداد", "کارتن", "بسته", "qty", "quantity", "موجودی", "بندی", "تیراژ"],
    "unit": ["واحد", "unit", "سنجش"],
    "image": ["عکس", "تصویر", "image", "photo", "picture"],
    "name": ["شرح", "نام", "کالا", "محصول", "عنوان", "توضیح", "description", "name", "item", "product", "مدل"],
}
_UNITS = {"عدد", "دست", "ست", "جفت", "بسته", "کارتن", "متر", "لیتر", "کیلو", "گالن", "حلقه", "رول", "شاخه", "pcs", "set"}
_JUNK_LINE = re.compile(r"^(page\s*\d+\s*(of\s*\d+)?|صفحه\s*[\d۰-۹]+(\s*از\s*[\d۰-۹]+)?|[\d۰-۹\s/\-.]+)$", re.I)


def canon(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "").translate(_CANON).lower()
    return re.sub(r"[\s\W_]+", "", t)


_WHOLE = {"فی", "#", "بها", "row", "part", "ref"}     # short words that only count on their own
_PREFIX = {"کد"}                                    # ... or at the start of a word («کدکالا»)


def _has_word(text: str, words: list[str]) -> bool:
    c = canon(text)
    if not c and "#" not in text:
        return False
    tokens = [canon(t) for t in re.split(r"[\s()\[\]/\-_:.,،]+", text) if t.strip()]
    tokens += ["#"] if text.strip() == "#" else []
    for w in words:
        k = canon(w) or w
        if w in _WHOLE:
            hit = k in tokens
        elif w in _PREFIX:
            hit = any(t.startswith(k) for t in tokens)
        else:
            hit = k in c
        if hit:
            return True
    return False


# ============================================================ page pieces ==

@dataclass
class Phrase:
    glyphs: list[Glyph]

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(min(g.x0 for g in self.glyphs), min(g.y0 for g in self.glyphs),
                            max(g.x1 for g in self.glyphs), max(g.y1 for g in self.glyphs))


def _text_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """Glyphs grouped by baseline, each line sorted left to right. Zero-width
    ligature parts (often on a raised baseline) stay with their glyph."""
    lines: list[list[Glyph]] = []
    line_of: dict[int, list[Glyph]] = {}
    zero = [g for g in glyphs if g.w < 0.08 * g.size and not g.c.isspace()]
    zero_ids = {id(g) for g in zero}
    for g in sorted((g for g in glyphs if id(g) not in zero_ids), key=lambda g: (g.base, g.x0)):
        for ln in reversed(lines[-8:]):
            ref = ln[0]
            if abs(ref.base - g.base) <= 0.45 * max(ref.size, g.size) and 0.6 <= g.size / ref.size <= 1.7:
                ln.append(g)
                break
        else:
            lines.append([g])
        line_of[g.seq] = lines[-1] if lines[-1][-1] is g else next(ln for ln in lines if ln[-1] is g)
    for z in zero:
        host = line_of.get(z.seq - 1) or line_of.get(z.seq + 1)
        if host is None:
            host = min(lines, key=lambda ln: abs(ln[0].base - z.base), default=None)
        if host is None:
            lines.append([z])
        else:
            host.append(z)
    for ln in lines:
        ln.sort(key=lambda g: (g.x0, g.seq))
    lines.sort(key=lambda ln: min(g.base for g in ln))
    return lines


def _phrases(glyphs: list[Glyph], edges: Callable[[float], list[float]]) -> list[Phrase]:
    """Runs of text that belong together: split at wide gaps and at column
    edges that fall in a visible gap (never inside a word)."""
    out: list[Phrase] = []
    for ln in _text_lines(glyphs):
        solid = [g for g in ln if not g.c.isspace()]
        if not solid:
            continue
        xs = edges((solid[0].y0 + solid[0].y1) / 2)
        cur: list[Glyph] = []
        right = -1e9
        last: Glyph | None = None
        for g in ln:
            if g.c.isspace():
                if cur:
                    cur.append(g)
                continue
            if cur:
                gap = g.x0 - right
                size = g.size
                cut = gap > 0.9 * size or (gap > 0.3 * size and any(right - 0.5 <= x <= g.x0 + 0.5 for x in xs))
                # text running over a column line into the next cell's number («…مشکی15»)
                if not cut and last is not None and g.c.isdigit() != last.c.isdigit():
                    cut = any(last.xc <= x <= g.xc for x in xs)
                if cut:
                    out.append(Phrase(_strip(cur)))
                    cur = []
                    right = -1e9
            cur.append(g)
            right = max(right, g.x1)
            last = g
        if cur and any(not c.c.isspace() for c in cur):
            out.append(Phrase(_strip(cur)))
    return out


def _strip(glyphs: list[Glyph]) -> list[Glyph]:
    """A phrase keeps the spaces between its words, not around it."""
    solid = [k for k, g in enumerate(glyphs) if not g.c.isspace()]
    return glyphs[solid[0]:solid[-1] + 1]


def _prices_on_page(analysis: Analysis, page: int) -> list[tuple[str, pymupdf.Rect]]:
    """(item id, box in PDF points) of every price found on a page."""
    out = []
    info = analysis.pages[page] if page < len(analysis.pages) else None
    for it in analysis.items:
        if it.page != page:
            continue
        r = pymupdf.Rect(it.bbox)
        if it.kind == "raster" and info is not None and info.zoom:
            r = r / info.zoom
        out.append((it.id, r))
    return out


# ================================================================== grid ==

@dataclass
class GCell:
    rect: pymupdf.Rect
    r0: int
    r1: int
    c0: int
    c1: int
    glyphs: list[Glyph] = field(default_factory=list)
    prices: list[str] = field(default_factory=list)
    image: pymupdf.Rect | None = None
    text: str = ""
    bold: bool = False
    suspect: bool = False     # the text layer of this cell cannot be trusted

    def has(self) -> bool:
        return bool(self.text or self.prices or self.image)


@dataclass
class Grid:
    bbox: pymupdf.Rect
    xs: list[float]
    ys: list[float]
    cells: list[GCell]
    hlines: list[tuple[float, float, float]] = field(default_factory=list)   # (y, x0, x1) drawn on the page
    _at: dict[tuple[int, int], GCell] | None = None

    def cell_at(self, x: float, y: float) -> GCell | None:
        """The cell containing a point (grid index lookup, not a scan)."""
        if not (self.xs[0] - 0.5 <= x <= self.xs[-1] + 0.5 and self.ys[0] - 0.5 <= y <= self.ys[-1] + 0.5):
            return None
        if self._at is None:
            self._at = {}
            for c in self.cells:
                for r in range(c.r0, c.r1):
                    for k in range(c.c0, c.c1):
                        self._at.setdefault((r, k), c)
        k = max(0, min(self.ncols - 1, bisect.bisect_right(self.xs, x) - 1))
        r = max(0, min(self.nrows - 1, bisect.bisect_right(self.ys, y) - 1))
        cell = self._at.get((r, k))
        if cell is not None and cell.rect.x0 - 0.5 <= x <= cell.rect.x1 + 0.5 \
                and cell.rect.y0 - 0.5 <= y <= cell.rect.y1 + 0.5:
            return cell
        return next((c for c in self.cells if c.rect.x0 <= x <= c.rect.x1 and c.rect.y0 <= y <= c.rect.y1), None)

    @property
    def ncols(self) -> int:
        return len(self.xs) - 1

    @property
    def nrows(self) -> int:
        return len(self.ys) - 1


def _cluster(values: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def _index(edges: list[float], v: float) -> int:
    return min(range(len(edges)), key=lambda k: abs(edges[k] - v))


def _grid(rects: list[pymupdf.Rect]) -> Grid | None:
    uniq: dict[tuple, pymupdf.Rect] = {}
    for r in rects:
        if r.width > 1 and r.height > 1:
            uniq.setdefault(tuple(round(v, 1) for v in r), r)
    rects = list(uniq.values())
    if len(rects) < 2:
        return None
    xs = _cluster([r.x0 for r in rects] + [r.x1 for r in rects], 1.5)
    ys = _cluster([r.y0 for r in rects] + [r.y1 for r in rects], 1.5)
    cells = []
    for r in rects:
        c0, c1, r0, r1 = _index(xs, r.x0), _index(xs, r.x1), _index(ys, r.y0), _index(ys, r.y1)
        if c1 > c0 and r1 > r0:
            cells.append(GCell(r, r0, r1, c0, c1))
    bbox = pymupdf.Rect(xs[0], ys[0], xs[-1], ys[-1])
    return Grid(bbox, xs, ys, cells)


def _find_grids(page: pymupdf.Page, **kw) -> list[Grid]:
    try:
        tabs = page.find_tables(**kw).tables
    except Exception as exc:  # noqa: BLE001 - table finder is best effort
        log.debug("find_tables failed: %s", exc)
        return []
    out = []
    for t in tabs:
        g = _grid([pymupdf.Rect(c) for row in t.rows for c in row.cells if c])
        if g is not None:
            out.append(g)
    return out


def _fill(grid: Grid, phrases: list[Phrase], prices: list[tuple[str, pymupdf.Rect]],
          images: list[pymupdf.Rect]) -> list[Phrase]:
    """Put phrases, prices and pictures into the cells; returns the phrases
    that are outside the grid."""
    outside = []
    for ph in phrases:
        r = ph.rect
        # a phrase belongs to the cell where it starts: text too long for its cell runs
        # over into the next one (Excel does that) but is still that cell's text
        size = ph.glyphs[0].size
        yc = (r.y0 + r.y1) / 2
        rtl = any(_ARABIC.match(g.c) for g in ph.glyphs)
        cell = grid.cell_at(r.x1 - 0.3 * size if rtl else r.x0 + 0.3 * size, yc)
        if cell is None:
            cell = grid.cell_at((r.x0 + r.x1) / 2, yc)
        if cell is None and grid.bbox.intersects(r):
            best = max(grid.cells, key=lambda c: (c.rect & r).get_area(), default=None)
            if best is not None and (best.rect & r).get_area() >= 0.5 * max(1e-6, r.get_area()):
                cell = best
        if cell is None:
            outside.append(ph)
        else:
            cell.glyphs.extend(ph.glyphs)
    for pid, box in prices:
        cell = grid.cell_at((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        if cell is not None:
            cell.prices.append(pid)
    for im in images:
        cell = grid.cell_at((im.x0 + im.x1) / 2, (im.y0 + im.y1) / 2)
        # a picture in a cell without text is a product photo; behind text it is shading
        if cell is not None and not cell.glyphs and im.get_area() <= 1.3 * cell.rect.get_area() \
                and cell.image is None:
            cell.image = (im & cell.rect) + (1.5, 1.5, -1.5, -1.5)    # not the cell's border lines
    for c in grid.cells:
        c.text = text_of(c.glyphs)
        solid = [g for g in c.glyphs if not g.c.isspace()]
        c.bold = bool(solid) and sum(g.bold for g in solid) >= 0.6 * len(solid)
        c.suspect = any(unreadable(g) for g in c.glyphs)
    return outside


def unreadable(g: Glyph) -> bool:
    """A glyph the PDF does not name correctly: no character at all, or a
    zero-width dot mark it calls a space (Word + Calibri write «سبز» as «ستز»)."""
    return g.c == "\ufffd" or "\ue000" <= g.c <= "\uf8ff" or (g.c == " " and g.w < 0.05 * g.size)


# ============================================= columns of a list without lines ==

def _word_boxes(line: list[Glyph]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for g in line:
        if g.c.isspace():
            continue
        if out and g.x0 - out[-1][1] <= 0.22 * g.size:
            out[-1][1] = max(out[-1][1], g.x1)
        else:
            out.append([g.x0, g.x1])
    return [(a, b) for a, b in out]


def _text_columns(glyphs: list[Glyph], prices: list[tuple[str, pymupdf.Rect]],
                  region: pymupdf.Rect) -> list[float]:
    """x positions separating the columns of a list, from where its words line up."""
    lines = [ln for ln in _text_lines([g for g in glyphs if region.contains(pymupdf.Point(g.xc, g.yc))])]
    rows = []
    for ln in lines:
        y0, y1 = min(g.y0 for g in ln), max(g.y1 for g in ln)
        if any(b.y0 < y1 and y0 < b.y1 for _, b in prices):
            rows.append(ln)
    if len(rows) < 3:
        return []
    x0 = min(g.x0 for ln in rows for g in ln)
    x1 = max(g.x1 for ln in rows for g in ln)
    width = int(x1 - x0) + 2
    cover = [0] * width
    for ln in rows:
        for a, b in _word_boxes(ln):
            for x in range(max(0, int(a - x0)), min(width, int(b - x0) + 1)):
                cover[x] += 1
    size = statistics.median(g.size for ln in rows for g in ln)
    seps: list[float] = []
    run = 0
    for x in range(width):
        if cover[x] == 0:
            run += 1
        else:
            if run >= max(2.0, 0.3 * size) and x - run > 0:
                seps.append(x0 + x - run / 2)
            run = 0
    # outer edges just outside the text: lines at the page edge are ignored by the table finder
    lo = min(g.x0 for ln in lines for g in ln)
    hi = max(g.x1 for ln in lines for g in ln)
    return [max(region.x0, lo - 4)] + seps + [min(region.x1, hi + 4)]


def _text_grid(page: pymupdf.Page, glyphs: list[Glyph], prices: list[tuple[str, pymupdf.Rect]],
               near: pymupdf.Rect | None) -> Grid | None:
    if len(prices) < 3:
        return None
    pr = pymupdf.Rect(prices[0][1])
    for _, b in prices[1:]:
        pr |= b
    region = pymupdf.Rect(near) if near is not None else pymupdf.Rect(0, pr.y0, page.rect.width, pr.y1)
    region.x0, region.x1 = 0, page.rect.width
    region.y0 = max(0.0, region.y0 - 110)     # room for a tall header
    region.y1 = min(page.rect.height, max(region.y1, pr.y1) + 20)
    xs = _text_columns(glyphs, prices, region)
    if len(xs) < 4:
        return None
    # rows from the ruling lines if there are any, else from the text lines
    for strategy in ("lines", "text"):
        grids = _find_grids(page, vertical_strategy="explicit", vertical_lines=xs,
                            horizontal_strategy=strategy, clip=region)
        grids = [g for g in grids if g.nrows >= 3]
        if grids:
            return max(grids, key=lambda g: g.bbox.get_area())
    return None


# =========================================================== logical table ==

@dataclass
class RawTable:
    """A table as found on one page, before the pages are joined."""
    headers: list[str]
    rows: list[Row]
    rtl: bool
    title_rows: list[str] = field(default_factory=list)
    note_rows: list[str] = field(default_factory=list)
    top: float = 0.0
    bottom: float = 0.0
    x0: float = 0.0
    late_headers: list[str] = field(default_factory=list)   # a header met only further down
    cells: int = 0              # text cells of the items
    suspect: int = 0            # ... of them with an unreadable text layer (headers count too)
    tail: list[str] = field(default_factory=list)   # group bars under the last row (belong to what follows)
    priceless: bool = False     # no price in it: only kept as the continuation of a table


def _looks_header(cells: list[GCell]) -> bool:
    texts = [c.text for c in cells if c.text]
    if len(texts) < 2 or any(c.prices for c in cells):
        return False
    # a header names columns: no product codes, no error values from a spreadsheet
    if any(re.search(r"[\d۰-۹٠-٩]{4,}|#N/A|#REF|#VALUE", t) for t in texts):
        return False
    hits = sum(1 for t in texts if any(_has_word(t, w) for w in _ROLE_WORDS.values()))
    return hits >= max(1, len(texts) // 3)


def _sparse_group_columns(grid: Grid, item_rows: set[int], known: set[int]) -> dict[int, dict[int, str]]:
    """Group names in a column the table finder cut into one cell per row: the
    name sits in one cell of the group, and only the lines drawn across the
    column show where each group starts. Returns column -> {first row: name}."""
    out: dict[int, dict[int, str]] = {}
    if not grid.hlines or not item_rows:
        return out
    lo, hi = min(item_rows), max(item_rows)
    for c in range(grid.ncols):
        if c in known:
            continue
        cells = [x for x in grid.cells if x.c0 == c and x.c1 == c + 1 and lo <= x.r0 <= hi]
        named = [x for x in cells if x.text]
        if not named or any(x.prices for x in cells) or len(named) > 0.3 * len(item_rows) \
                or any(len(x.text) > 30 for x in named):
            continue
        x0, x1 = grid.xs[c], grid.xs[c + 1]
        width = x1 - x0

        def ruled(y: float) -> bool:
            return any(abs(ly - y) <= 1.5 and min(lx1, x1) - max(lx0, x0) >= 0.8 * width
                       for ly, lx0, lx1 in grid.hlines)
        # group boundaries: grid rows whose top line crosses this column
        bounds = [r for r in range(lo, hi + 1) if ruled(grid.ys[r])]
        if not bounds or bounds[0] != lo:
            bounds = [lo] + bounds
        if len(bounds) >= 0.8 * (hi - lo + 1):
            continue          # every row ruled: an ordinary column that is mostly empty
        spans = list(zip(bounds, bounds[1:] + [hi + 1]))
        starts: dict[int, str] = {}
        for a, b in spans:
            names = [x.text for x in sorted(named, key=lambda x: x.r0) if a <= x.r0 < b]
            if names:
                starts[a] = " ".join(names)
        if starts:
            out[c] = starts
    return out


def _hlines(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """Horizontal lines drawn on the page (strokes, hairline rectangles, rectangle edges)."""
    out = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return out
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.y - b.y) < 1:
                    out.append(((a.y + b.y) / 2, min(a.x, b.x), max(a.x, b.x)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 2.5:
                    out.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
                elif d.get("color") is not None:
                    out += [(r.y0, r.x0, r.x1), (r.y1, r.x0, r.x1)]
    return out


def _group_columns(grid: Grid, item_rows: set[int]) -> set[int]:
    """Columns whose cells are merged down over many rows and hold a group name
    (e.g. «پیکان» written once beside all Peykan rows)."""
    out = set()
    for c in range(grid.ncols):
        cells = [x for x in grid.cells if x.c0 == c and x.c1 == c + 1]
        tall = [x for x in cells if x.r1 - x.r0 >= 2 and x.text and not x.prices]
        covered = sum(1 for r in item_rows if any(x.r0 <= r < x.r1 for x in tall))
        # a group cell stands beside several items; a cell merely drawn over two grid
        # rows (lines of other columns not aligned) stands beside one
        per_cell = covered / max(1, len(tall))
        if tall and item_rows and covered >= 0.5 * len(item_rows) and per_cell >= 1.8 \
                and not any(x.prices for x in cells):
            out.add(c)
    return out


_TITLE_WORDS = re.compile(r"لیست|فهرست|قیمت|تاریخ|شرکت|بازرگانی|گروه صنعتی|فروشگاه|نمایندگی|price|list|"
                          r"(?<![ؠ-ي])(فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|[اآ]بان|[اآ]ذر|دی|بهمن|اسفند)"
                          r"(\s*ماه)?(?![ؠ-ي])|"
                          r"[\d۰-۹]{2,4}\s*[/\-.]\s*[\d۰-۹]{1,2}\s*[/\-.]\s*[\d۰-۹]{1,4}", re.I)


_NOTE_WORDS = re.compile(r"می\s*باشد|میباشد|\bاست\b|هستند|باشد|نمایید|فرمایید|گردد|می\s*شود|میشود|"
                         r"شده|توجه|نکته|لطفا|\*|مالیات|ارزش افزوده|گارانتی|ضمانت|اعتبار|معتبر|موقت|نقدی|چکی", re.I)


def is_note_line(text: str) -> bool:
    return bool(_NOTE_WORDS.search(unicodedata.normalize("NFKC", text).translate(_CANON).replace("\u200c", "")))


# notes that belong to the supplier (who, where, which edition): not repeated in the Arizon list
_SUPPLIER_NOTE = re.compile(
    r"تاریخ|شرکت|بازرگانی|دفتر|ادرس|آدرس|تلفن|تماس|فکس|همراه|واتس|تلگرام|اینستا|ایتا|روبیکا|"
    r"whatsapp|telegram|instagram|www|http|@|\.com|\.ir|شماره حساب|شماره کارت|کارت|شبا|حساب|"
    r"فروشگاه|نمایندگی|منتشر|لیست قبلی|فاقد اعتبار|نام مشتری|امضا|مهر و|"
    r"[\d۰-۹]{2,4}\s*[/\-.]\s*[\d۰-۹]{1,2}\s*[/\-.]\s*[\d۰-۹]{1,4}|[\d۰-۹]{7,}|"
    r"(?<![ؠ-ي])(فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|[اآ]بان|[اآ]ذر|دی|بهمن|اسفند)(?![ؠ-ي])", re.I)


_PRICE_NOTE = re.compile(r"مالیات|ارزش افزوده|احتساب|vat", re.I)


def is_price_note(text: str) -> bool:
    return bool(_PRICE_NOTE.search(text))


def is_supplier_note(text: str) -> bool:
    return bool(_SUPPLIER_NOTE.search(unicodedata.normalize("NFKC", text).translate(_CANON)))


def is_title_line(text: str) -> bool:
    t = unicodedata.normalize("NFKC", text).translate(_CANON)
    return bool(_TITLE_WORDS.search(t)) and not is_note_line(text)


def logical_table(grid: Grid, rtl: bool, crop: Callable[[pymupdf.Rect], bytes | None]) -> RawTable | None:
    """A detected grid -> header, group rows and item rows in reading order."""
    starts: dict[int, list[GCell]] = {}
    for c in grid.cells:
        starts.setdefault(c.r0, []).append(c)
    price_rows = {r for r, cs in starts.items() if any(c.prices for c in cs)}
    priceless = not price_rows
    if priceless:
        # rows without any price (e.g. the last rows of a table, alone at the top of the
        # next page): kept only if they turn out to continue a table (see assemble)
        price_rows = {r for r, cs in starts.items() if sum(1 for c in cs if c.text) >= 3
                      and not _looks_header([c for c in cs if c.text])}
        if not price_rows:
            return None
    group_cols = _group_columns(grid, price_rows)
    sparse = _sparse_group_columns(grid, price_rows, group_cols)
    group_cols |= set(sparse)

    def in_group_col(c: GCell) -> bool:
        return c.c0 in group_cols and c.c1 == c.c0 + 1

    group_text: dict[int, str] = {c.r0: c.text for c in grid.cells if in_group_col(c) and c.text
                                  and c.c0 not in sparse}
    for starts_of in sparse.values():
        group_text.update(starts_of)
    kinds: dict[int, str] = {}
    for r in sorted(starts):
        cells = [c for c in starts[r] if c.has() and not in_group_col(c)]
        if not cells:
            continue
        if any(c.prices for c in cells) or (priceless and r in price_rows):
            kinds[r] = "item"
        elif len(cells) == 1 and not cells[0].image:
            kinds[r] = "banner"
        elif _looks_header(cells):
            kinds[r] = "header"
        else:
            kinds[r] = "item"
    first_item = min(price_rows)
    # the table's header: the header rows right above the first item
    header_rows: list[int] = []
    for r in sorted(k for k in kinds if k < first_item):
        if kinds[r] == "header":
            if header_rows and any(kinds.get(k) in ("item", "banner") for k in range(header_rows[-1] + 1, r)):
                header_rows = []
            header_rows.append(r)
    late_rows = [] if header_rows else [r for r in sorted(kinds) if kinds[r] == "header"][:1]
    # above the header there are no items: logo, company name, date… (read as titles)
    above = {k for k in kinds if header_rows and k < header_rows[0] and kinds[k] == "item"}
    for r in above:
        kinds[r] = "banner"

    item_cells = [c for c in grid.cells if kinds.get(c.r0) == "item" and c.has() and not in_group_col(c)]
    cols = sorted({c.c0 for c in item_cells})
    if priceless:     # its empty price column must stay, to line up with the table it continues
        cols = [c for c in range(grid.ncols) if c not in group_cols]
    if len(cols) < 2:
        return None
    order = cols[::-1] if rtl else cols
    pos = {c: k for k, c in enumerate(order)}

    def place(cell: GCell) -> tuple[int, int] | None:
        inside = [c for c in cols if cell.c0 <= c < cell.c1]
        if not inside:
            return None
        ks = [pos[c] for c in inside]
        return min(ks), max(ks) - min(ks) + 1

    def header_of(hrows: list[int]) -> list[str]:
        out = [""] * len(cols)
        for r in hrows:
            for cell in starts[r]:
                got = place(cell) if cell.text else None
                if got is None:
                    continue
                k0, span = got
                for k in range(k0, k0 + span):
                    if cell.text not in out[k]:
                        out[k] = (out[k] + " " + cell.text).strip()
        return out

    headers = header_of(header_rows)
    late = header_of(late_rows) if late_rows else []

    rows: list[Row] = []
    titles: list[str] = []
    notes: list[str] = []
    for r in range(grid.nrows):
        if r in group_text and r >= first_item - 1 and kinds.get(r) != "header":
            rows.append(Row([Cell(group_text[r])], "group"))
        kind = kinds.get(r)
        if kind is None or kind == "header":
            continue
        cells = [c for c in starts[r] if c.has() and not in_group_col(c)]
        if kind == "banner":
            text = " ".join(c.text for c in cells if c.text)
            if not text:
                continue
            if len(text) > 70 or is_note_line(text):
                notes.append(text)
            elif r < first_item and (is_title_line(text) or r in above):
                titles.append(text)
            elif any(c.suspect for c in cells):
                bad = [c for c in cells if c.suspect][0]
                rows.append(Row([Cell(text, snapshot=crop(_ink_box(bad), True),
                                      snap_size=statistics.median(g.size for g in bad.glyphs))], "group"))
            else:
                rows.append(Row([Cell(text)], "group"))
            continue
        line = [Cell() for _ in cols]
        for cell in cells:
            got = place(cell)
            if got is None:
                continue
            k0 = got[0]
            img = crop(cell.image) if cell.image is not None else None
            snap, size = None, 0.0
            if cell.suspect and not cell.prices:
                snap = crop(_ink_box(cell), True)
                size = statistics.median(g.size for g in cell.glyphs)
            prev = line[k0]
            line[k0] = Cell((prev.text + " " + cell.text).strip(), prev.price_ids + cell.prices, prev.image or img,
                            prev.snapshot or snap, prev.snap_size or size)
        if not all(c.empty() for c in line):
            rows.append(Row(line, "item"))
    # group rows with nothing after them head the next table (or were notes)
    tail: list[str] = []
    while rows and rows[-1].kind == "group":
        tail.insert(0, rows.pop().cells[0].text)
    headers, rows, late = _merge_split_columns(headers, rows, late)
    headers, rows, late = _collapse_spanned(headers, rows, late)
    if not priceless:
        headers, rows, late = _drop_unused_columns(headers, rows, late)
    used = [c for c in grid.cells if c.text and not c.prices and (kinds.get(c.r0) in ("item", "banner")
                                                                  or c.r0 in header_rows)]
    bad = sum(1 for c in used if c.suspect) + sum(3 for r in header_rows for c in starts[r] if c.suspect)
    return RawTable(headers, rows, rtl, titles, notes, grid.bbox.y0, grid.bbox.y1, grid.bbox.x0, late,
                    len(used), bad, tail, priceless)


def _ink_box(cell: GCell) -> pymupdf.Rect:
    solid = [g for g in cell.glyphs if not g.c.isspace()] or cell.glyphs
    r = pymupdf.Rect(min(g.x0 for g in solid), min(g.y0 for g in solid),
                     max(g.x1 for g in solid), max(g.y1 for g in solid))
    return (r + (-1.5, -1, 1.5, 1)) & cell.rect


def _collapse_spanned(headers: list[str], rows: list[Row],
                      late: list[str]) -> tuple[list[str], list[Row], list[str]]:
    """Columns under one header cell (e.g. «کد مشترک» over a ★ column and a code
    column) are one column: their texts are joined. Two price columns are
    never joined."""
    items = [r for r in rows if r.kind == "item"]
    k = 0
    while k + 1 < len(headers):
        h = canon(headers[k])
        if not h or canon(headers[k + 1]) != h or any(
                k + 1 < len(r.cells) and r.cells[k].price_ids and r.cells[k + 1].price_ids for r in items):
            k += 1
            continue
        for r in items:
            if k + 1 >= len(r.cells):
                continue
            a, b = r.cells[k], r.cells[k + 1]
            r.cells[k] = Cell(" ".join(x for x in (a.text, b.text) if x), a.price_ids + b.price_ids,
                              a.image or b.image, a.snapshot or b.snapshot, a.snap_size or b.snap_size)
            del r.cells[k + 1]
        del headers[k + 1]
        if len(late) > k + 1:
            del late[k + 1]
    return headers, rows, late


def _drop_unused_columns(headers: list[str], rows: list[Row],
                         late: list[str]) -> tuple[list[str], list[Row], list[str]]:
    """A column without header that no item uses (a stray line split a cell in two).
    A named column may just be empty on this page: the whole table decides later."""
    items = [r for r in rows if r.kind == "item"]
    keep = [k for k in range(len(headers))
            if canon(headers[k]) or any(k < len(r.cells) and not r.cells[k].empty() for r in items)]
    if len(keep) == len(headers) or not keep:
        return headers, rows, late
    for r in items:
        r.cells = [r.cells[k] for k in keep if k < len(r.cells)]
    late = [late[k] for k in keep if k < len(late)] if late else late
    return [headers[k] for k in keep], rows, late


def _merge_split_columns(headers: list[str], rows: list[Row],
                         late: list[str]) -> tuple[list[str], list[Row], list[str]]:
    """Neighbouring columns that are never filled in the same row are one
    column whose cell edges moved (rows drawn with different grids)."""
    items = [r for r in rows if r.kind == "item"]
    k = 0
    while k + 1 < len(headers):
        both = any(not r.cells[k].empty() and not r.cells[k + 1].empty() for r in items)
        used = [any(not r.cells[j].empty() for r in items) for j in (k, k + 1)]
        ha, hb = canon(headers[k]), canon(headers[k + 1])
        if not both and all(used) and (not ha or not hb or ha == hb):
            headers[k] = headers[k] or headers[k + 1]
            del headers[k + 1]
            if len(late) > k + 1:
                late[k] = late[k] or late[k + 1]
                del late[k + 1]
            for r in items:
                a, b = r.cells[k], r.cells[k + 1]
                r.cells[k] = a if not a.empty() else b
                del r.cells[k + 1]
            continue
        k += 1
    return headers, rows, late


def split_blocks(t: RawTable) -> RawTable:
    """«ردیف | نام | قیمت | ردیف | نام | قیمت | …»: a list printed as several
    tables side by side becomes one long table (right block first)."""
    n = len(t.headers)
    keys = [canon(h) for h in (t.headers if any(t.headers) else t.late_headers or t.headers)]
    for p in range(2, n // 2 + 1):
        if n % p or not any(keys[:p]):
            continue
        if all(keys[k] == keys[k % p] for k in range(n)):
            break
    else:
        return t
    return _split(t, p)


def _split(t: RawTable, p: int) -> RawTable:
    blocks = len(t.headers) // p
    rows: list[Row] = []
    segment: list[Row] = []

    def flush() -> None:
        for b in range(blocks):
            for row in segment:
                part = row.cells[b * p:(b + 1) * p]
                if not all(c.empty() for c in part):
                    rows.append(Row(part, "item"))
        segment.clear()

    for row in t.rows:
        if row.kind == "group":
            flush()
            rows.append(row)
        else:
            segment.append(row)
    flush()
    return RawTable(t.headers[:p], rows, t.rtl, t.title_rows, t.note_rows, t.top, t.bottom, t.x0,
                    t.late_headers[:p], t.cells, t.suspect, t.tail, t.priceless)


# ================================================================ roles ==

def column_roles(t: Table) -> list[str]:
    n = t.ncols
    roles = ["text"] * n
    items = [r for r in t.rows if r.kind == "item"]
    stats = []
    for k in range(n):
        cells = [r.cells[k] for r in items if k < len(r.cells)]
        full = [c for c in cells if not c.empty()]
        texts = [c.text for c in full if c.text]
        stats.append({
            "n": len(full),
            "price": sum(1 for c in full if c.price_ids),
            "image": sum(1 for c in full if c.image is not None and not c.text),
            "len": statistics.mean([len(x) for x in texts]) if texts else 0.0,
            "int": sum(1 for x in texts if re.fullmatch(r"[\d۰-۹٠-٩]{1,4}", x.strip())),
            "code": sum(1 for x in texts if re.fullmatch(r"[A-Za-z0-9۰-۹٠-٩\-_/.]{4,}", x.strip().replace(" ", ""))
                        and len(x.split()) <= 3),
            "unit": sum(1 for x in texts if x.strip() in _UNITS),
            "fa": sum(1 for x in texts if _ARABIC.search(x)),
        })
    head = t.headers + [""] * (n - len(t.headers))
    for k, s in enumerate(stats):
        h = head[k]
        if s["n"] and (s["price"] >= 0.4 * s["n"] or (s["price"] and _has_word(h, _ROLE_WORDS["price"]))):
            roles[k] = "price"
        elif s["n"] and s["image"] >= 0.5 * s["n"]:
            roles[k] = "image"
        elif _has_word(h, _ROLE_WORDS["image"]) and s["len"] == 0:
            roles[k] = "image"
        elif _has_word(h, _ROLE_WORDS["row"]) or (s["n"] and s["int"] >= 0.8 * s["n"] and _increasing(t, k)):
            roles[k] = "row"
        elif _has_word(h, _ROLE_WORDS["code"]) and s["fa"] <= 0.5 * max(1, s["n"]):
            roles[k] = "code"
        elif _has_word(h, _ROLE_WORDS["unit"]) or (s["n"] and s["unit"] >= 0.7 * s["n"]):
            roles[k] = "unit"
        elif _has_word(h, _ROLE_WORDS["qty"]) or (s["n"] and s["int"] >= 0.8 * s["n"]):
            roles[k] = "qty"
        elif s["n"] and s["code"] >= 0.7 * s["n"] and s["fa"] <= 0.2 * s["n"]:
            roles[k] = "code"
    free = [k for k in range(n) if roles[k] == "text"]
    if free:
        name = max(free, key=lambda k: (stats[k]["len"] * (1.5 if _has_word(head[k], _ROLE_WORDS["name"]) else 1)))
        if stats[name]["len"] >= 3:
            roles[name] = "name"
    return roles


def _increasing(t: Table, k: int) -> bool:
    vals = []
    for r in t.rows:
        if r.kind == "item" and k < len(r.cells):
            d = re.sub(r"\D", "", unicodedata.normalize("NFKC", r.cells[k].text))
            d = d.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
            if d:
                vals.append(int(d))
    if len(vals) < 3:
        return False
    ups = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
    return ups >= 0.7 * (len(vals) - 1)


# ============================================================ PDF pages ==

@dataclass
class PageRead:
    tables: list[RawTable]
    above: list[str]          # text lines above the first table (titles)
    below: list[str]          # text lines under the tables (notes)
    placed: int               # prices that ended up in a table cell
    total: int
    in_title: int = 0         # "prices" in the title lines above the tables (not shown)

    @property
    def unreadable(self) -> float:
        """Share of the text cells whose text layer is gibberish."""
        cells = sum(t.cells for t in self.tables)
        return sum(t.suspect for t in self.tables) / max(1, cells)


def _page_rtl(glyphs: list[Glyph]) -> bool:
    text = "".join(g.c for g in glyphs)
    return len(_ARABIC.findall(text)) >= len(_LATIN.findall(text))


def _outside_lines(phrases: list[Phrase]) -> list[tuple[str, float]]:
    """Phrases outside every table -> text lines (top to bottom) with their y."""
    glyphs = [g for ph in phrases for g in ph.glyphs]
    out = []
    for ln in _text_lines(glyphs):
        # a line can hold several separate phrases (e.g. title left, date right)
        chunks: list[list[Glyph]] = [[]]
        for g in ln:
            if chunks[-1] and g.x0 - max(x.x1 for x in chunks[-1]) > 3 * g.size:
                chunks.append([])
            chunks[-1].append(g)
        for ch in chunks:
            t = text_of(ch)
            if t and not _JUNK_LINE.match(t):
                out.append((t, min(g.y0 for g in ch)))
    return out


def read_pdf_page(analysis: Analysis, page_no: int) -> PageRead | None:
    """The tables of one text-layer PDF page, or None when its text cannot be
    trusted (gibberish font) or no table holding its prices is found."""
    doc = pymupdf.open(analysis.source)
    try:
        page = doc[page_no]
        glyphs = page_glyphs(page)
        if not glyphs or is_legacy("".join(g.c for g in glyphs)):
            return None
        prices = _prices_on_page(analysis, page_no)
        if not prices:
            return None
        rtl = _page_rtl(glyphs)
        images = [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
        images = [r for r in images if not r.is_empty and r.get_area() < 0.2 * page.rect.get_area()
                  and r.width > 8 and r.height > 8 and not _flat(page, r)]

        whole: list = []      # the page drawn once, cut up for every picture

        def crop(r: pymupdf.Rect, text: bool = False) -> bytes | None:
            """A picture of part of the page (text: the ink alone, see _ink_only)."""
            import numpy as np
            from PIL import Image

            try:
                if not whole:
                    pix = page.get_pixmap(dpi=CROP_DPI, alpha=False)
                    whole.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n))
                img = whole[0]
                z = CROP_DPI / 72
                box = pymupdf.Rect(r) * page.rotation_matrix * pymupdf.Matrix(z, z)
                x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
                x1, y1 = min(img.shape[1], int(box.x1 + 1)), min(img.shape[0], int(box.y1 + 1))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    return None
                part = np.ascontiguousarray(img[y0:y1, x0:x1, :3])
                if text:
                    return _ink_only(part)
                buf = io.BytesIO()
                Image.fromarray(part).save(buf, "JPEG", quality=88)
                return buf.getvalue()
            except Exception:  # noqa: BLE001
                return None

        best: PageRead | None = None
        best_grids: list[Grid] = []
        # a list drawn without any ruling lines on its last pages has none on this one either
        no_lines = analysis.cache.get("arizon_no_lines", 0)
        for attempt in ("lines", "text"):
            if attempt == "lines":
                if no_lines >= 2:
                    continue
                grids = _find_grids(page)
                analysis.cache["arizon_no_lines"] = 0 if grids else no_lines + 1
            else:
                near = max(best_grids, key=lambda g: g.bbox.get_area()).bbox if best_grids else None
                g = _text_grid(page, glyphs, prices, near)
                grids = [g] if g is not None else []
            if attempt == "lines":
                best_grids = grids
            read = _read_grids(page, glyphs, grids, prices, images, rtl, crop)
            if read is not None and (best is None or _score(read) > _score(best)):
                best = read
            if best is not None and _good(best):
                break
        return best
    finally:
        doc.close()


def _ink_only(rgb) -> bytes:
    """Text on a shaded cell -> the text alone, on a transparent background."""
    import numpy as np
    from PIL import Image

    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bg = np.percentile(lum, 70)                     # the cell's own background
    k = np.clip((bg - lum) / max(1.0, bg - 40.0), 0.0, 1.0) ** 0.6     # full strength for the strokes
    ink = np.minimum(rgb, 40)                                          # the same near-black as the rest
    out = np.dstack([np.where(lum[..., None] < bg - 60, ink, rgb), (255 * k).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def _flat(page: pymupdf.Page, r: pymupdf.Rect) -> bool:
    """A single-colour picture (cell shading, a coloured bar) is not a photo."""
    try:
        pix = page.get_pixmap(clip=r, dpi=24, alpha=False, colorspace=pymupdf.csGRAY)
    except Exception:  # noqa: BLE001
        return True
    data = pix.samples
    if not data:
        return True
    mean = sum(data) / len(data)
    var = sum((v - mean) ** 2 for v in data) / len(data)
    return var < 60


def _score(r: PageRead) -> float:
    cols = [len(t.headers) for t in r.tables] or [0]
    return r.placed / max(1, r.total) + 0.01 * min(max(cols), 6)


def _good(r: PageRead) -> bool:
    if r.placed < 0.9 * r.total or not r.tables:
        return False
    for t in r.tables:
        if len(t.headers) < 2 and not t.priceless:
            return False
        # a price squeezed into the same cell as a long description = columns not found
        for row in t.rows:
            for c in row.cells:
                if c.price_ids and len(_ARABIC.findall(c.text)) > 6:
                    return False
    return True


def _read_grids(page: pymupdf.Page, glyphs: list[Glyph], grids: list[Grid], prices, images, rtl: bool,
                crop) -> PageRead | None:
    if not grids:
        return None

    def edges(y: float) -> list[float]:
        return [x for g in grids if g.bbox.y0 - 2 <= y <= g.bbox.y1 + 2 for x in g.xs]

    phrases = _phrases(glyphs, edges)
    outside = phrases
    rotated = rotated_lines(page)
    hlines = _hlines(page)
    for g in grids:
        g.hlines = hlines
        outside = _fill(g, outside, prices, images)
        for box, text in rotated:
            centre = pymupdf.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            cell = next((c for c in g.cells if c.rect.contains(centre) and not c.text and not c.prices), None)
            if cell is not None:
                cell.text = text
    tables = []
    for g in grids:
        t = logical_table(g, rtl, crop)
        if t is not None:
            tables.append(split_blocks(t))
    priced = [t for t in tables if not t.priceless]
    if not priced:
        return None
    # a table without prices counts only above the first priced one (a table's last
    # rows carried over to the top of this page)
    first = min(t.top for t in priced)
    tables = [t for t in tables if not t.priceless or t.bottom <= first + 2]
    placed_ids = {pid for t in tables for r in t.rows for c in r.cells for pid in c.price_ids}
    placed = len(placed_ids)
    top = min(t.top for t in priced)
    # a number in the list's own title (e.g. the year) that was taken for a price: the
    # template does not show that title, so nothing wrong can appear
    in_title = sum(1 for pid, box in prices if pid not in placed_ids and box.y1 <= top + 1)
    bottom = max(t.bottom for t in priced)
    lines = _outside_lines(outside)
    above = [t for t, y in lines if y < top]
    below = [t for t, y in lines if y >= bottom - 2]
    tables.sort(key=lambda t: (round(t.top / 20), -t.x0 if rtl else t.x0))
    return PageRead(tables, above, below, placed, len(prices), in_title)


# ============================================================ whole list ==

def _to_table(raw: RawTable) -> Table:
    return Table(list(raw.headers), list(raw.rows), "", raw.rtl)


def _continues(a: Table, b: Table) -> bool:
    """b goes on where a stopped: the same header, or no header and as many columns."""
    if len(a.headers) != len(b.headers):
        return False
    # (a header cell the page lost, or one cell misread, does not make a new table)
    same = sum(1 for x, y in zip((canon(h) for h in a.headers), (canon(h) for h in b.headers))
               if x == y or not x or not y)
    return same == len(a.headers) or (len(a.headers) >= 4 and same >= len(a.headers) - 1)


def assemble(pages: list[tuple[int, PageRead | Table | Frame | list]]) -> Content:
    """Join what was read page by page into one document."""
    content = Content()
    seen_titles: set[str] = set()
    seen_notes: set[str] = set()

    def note(text: str) -> None:
        # only notes about how the prices are meant (tax included...) are kept; the
        # supplier's own terms, dates and contact details are not Arizon's
        k = canon(text)
        if not is_price_note(text) or is_supplier_note(text) or "\ufffd" in text or "لیست قیمت" in text:
            return
        if k and k not in seen_notes and len(k) > 3:
            seen_notes.add(k)
            content.notes.append(text)

    def title(text: str) -> None:
        k = canon(text)
        if k and k not in seen_titles:
            seen_titles.add(k)
            content.titles.append(text)

    carry: list[str] = []
    for page, got in pages:
        items: list = got if isinstance(got, list) else [got]
        for obj in items:
            if isinstance(obj, Frame):
                content.blocks.append(obj)
                continue
            if isinstance(obj, PageRead):
                for t in obj.above:
                    if is_note_line(t) or len(t) > 70:
                        note(t)
                    else:
                        title(t)
                raws = obj.tables
                for t in obj.below:
                    note(t)
            elif isinstance(obj, RawTable):
                raws = [obj]
            else:
                raws = []
            for raw in raws:
                for t in raw.title_rows:
                    if len(t) > 60:
                        note(t)
                    else:
                        title(t)
                for t in raw.note_rows:
                    note(t)
                last = next((b for b in reversed(content.blocks) if isinstance(b, Table)), None)
                # a page without header, as wide as two or three tables of the previous page:
                # the same list printed side by side
                n = len(raw.headers)
                if last is not None and not any(canon(h) for h in raw.headers + raw.late_headers) \
                        and 2 <= len(last.headers) < n and n % len(last.headers) == 0:
                    raw = _split(raw, len(last.headers))
                table = _to_table(raw)
                table.rows[:0] = [Row([Cell(g)], "group") for g in carry]
                carry = list(raw.tail)
                if last is not None and content.blocks[-1] is last and _continues(last, table):
                    last.rows.extend(table.rows)
                    last.headers = [x or y for x, y in zip(last.headers, table.headers)]
                    continue
                if raw.priceless:
                    continue
                if not any(table.headers) and raw.late_headers:
                    table.headers = list(raw.late_headers)
                # the list's own title lines name the supplier: the template has its own title
                content.blocks.append(table)
    for b in content.blocks:
        if isinstance(b, Table):
            b.headers, b.rows, _ = _collapse_spanned(b.headers, b.rows, [])
            b.roles = column_roles(b)
            _drop_empty_columns(b)
            _order_by_row_number(b)
    content.currency = currency_of(content)
    return content


_CURRENCIES = [("ریال", r"ریال|رىال|\bریا\b|rial"), ("تومان", r"تومان|toman"), ("یورو", r"یورو|euro|€"),
               ("دلار", r"دلار|dollar|usd|\$"), ("درهم", r"درهم|aed|dirham")]


def currency_of(content: Content) -> str:
    """The currency named in the price headers (else the titles and notes)."""
    def find(texts: list[str]) -> str:
        joined = " ".join(texts).lower()
        hits = [(name, len(re.findall(rx, joined))) for name, rx in _CURRENCIES]
        name, n = max(hits, key=lambda h: h[1])
        return name if n else ""
    heads = [h for b in content.blocks if isinstance(b, Table)
             for h, r in zip(b.headers, b.roles or [""] * len(b.headers)) if r == "price"]
    return find(heads) or find(content.titles + content.notes)


def _order_by_row_number(t: Table) -> None:
    """Rows numbered 1, 2, 3… but read out of order (blocks side by side, pages
    split in columns) are put back in the order of their numbers. Group bars
    move with the row that follows them."""
    if "row" not in t.roles:
        return
    k = t.roles.index("row")
    units: list[list[Row]] = []
    pending: list[Row] = []
    nums = []
    for r in t.rows:
        if r.kind != "item":
            pending.append(r)
            continue
        d = re.sub(r"\D", "", unicodedata.normalize("NFKC", r.cells[k].text if k < len(r.cells) else ""))
        d = d.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
        if not d or len(d) > 5:
            return
        nums.append(int(d))
        units.append(pending + [r])
        pending = []
    # (a few numbers used twice by mistake do not matter: the sort keeps their order)
    if len(nums) < 3 or len(set(nums)) < 0.95 * len(nums) or nums == sorted(nums):
        return
    order = sorted(range(len(units)), key=lambda i: nums[i])
    t.rows = [r for i in order for r in units[i]] + pending


def _drop_empty_columns(t: Table) -> None:
    n = t.ncols
    keep = []
    for k in range(n):
        if any(k < len(r.cells) and not r.cells[k].empty() for r in t.rows if r.kind == "item"):
            keep.append(k)
    if len(keep) == n:
        return
    t.headers = [t.headers[k] if k < len(t.headers) else "" for k in keep]
    t.roles = [t.roles[k] for k in keep] if t.roles else []
    for r in t.rows:
        if r.kind == "item":
            r.cells = [r.cells[k] if k < len(r.cells) else Cell() for k in keep]
