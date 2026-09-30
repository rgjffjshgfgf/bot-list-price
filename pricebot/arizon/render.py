"""Drawing the Arizon template: the price list rebuilt as clean, branded
A4 pages (black header band with the vector logo, yellow/black table header,
zebra rows, highlighted new prices, group bars, page numbers).

Text is laid out by MuPDF's HTML engine (Story), which shapes Persian and
mixes right-to-left and left-to-right text correctly; the table is split into
pages here, so every page repeats the header and no row is ever cut.
"""
from __future__ import annotations

import html
import logging
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pymupdf

from ..models import PriceItem
from .model import Cell, Content, Frame, Row, Table

log = logging.getLogger(__name__)
ASSETS = Path(__file__).resolve().parent / "assets"
TEHRAN = ZoneInfo("Asia/Tehran")

YELLOW = "#FFF112"
RED = "#EC3237"
BLACK = "#0B0B0B"
INK = "#1A1A1A"
GREY = "#6B6B6B"
LINE = "#D5D5D5"
ZEBRA = "#F5F5F2"
PRICE_BG = "#FFFCE0"
PRICE_BG_Z = "#FFF8C4"

A4 = pymupdf.paper_rect("a4")
MARGIN = 26.0
BAND_H = 74.0
FOOT_H = 30.0
SNAP_DPI = 170     # resolution of the crops made by extract.py
PAD = 7.5          # horizontal padding + border of a table cell
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

_CSS = """
@font-face {font-family: vz; src: url(Vazirmatn-Regular.ttf);}
@font-face {font-family: vz; font-weight: bold; src: url(Vazirmatn-Bold.ttf);}
@font-face {font-family: vzm; src: url(Vazirmatn-Medium.ttf);}
@font-face {font-family: vzb; src: url(Vazirmatn-Black.ttf);}
p, div, table, body {margin: 0; padding: 0;}
body {font-family: vz; font-size: %(fs)spt; color: #1A1A1A;}
table {border-collapse: collapse; border: 0.8pt solid #C8C8C8;}
td {border: 0.5pt solid #D5D5D5; padding: %(pv)spt 3.5pt; text-align: center; vertical-align: middle;}
td.s {text-align: left;}
tr.h td {background-color: #0B0B0B; color: #FFF112; font-weight: bold; border: 0.5pt solid #3A3A3A;
         padding: %(ph)spt 3pt; font-size: %(hfs)spt;}
tr.z td {background-color: #F5F5F2;}
td.p {font-family: vzm; font-weight: bold; background-color: #FFFCE0; color: #0B0B0B; white-space: nowrap;}
tr.z td.p {background-color: #FFF8C4;}
td.n {color: #6B6B6B; font-size: %(sfs)spt;}
td.c {font-size: %(sfs)spt; color: #333333;}
tr.g td {background-color: #FFF112; color: #0B0B0B; font-weight: bold; font-size: %(gfs)spt;
         border: 0.5pt solid #E0D400; padding: %(ph)spt 8pt;}
p.t {font-family: vzb; font-size: 12pt; color: #0B0B0B;}
"""


def fa(n) -> str:
    return str(n).translate(FA_DIGITS)


# ================================================================== dates ==

def jalali(d: datetime) -> tuple[int, int, int]:
    """Gregorian -> Jalali (Solar Hijri) date."""
    gy, gm, gd = d.year, d.month, d.day
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + gd + g_d_m[gm - 1]
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm, jd = 1 + days // 31, 1 + days % 31
    else:
        jm, jd = 7 + (days - 186) // 30, 1 + (days - 186) % 30
    return jy, jm, jd


MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]


def today_fa(now: datetime | None = None) -> tuple[str, str]:
    """('۱۴۰۵/۰۷/۰۷', '۷ مهر ۱۴۰۵') for today in Tehran."""
    y, m, d = jalali(now or datetime.now(TEHRAN))
    return fa(f"{y}/{m:02d}/{d:02d}"), f"{fa(d)} {MONTHS[m - 1]} {fa(y)}"


# =================================================================== logo ==

@lru_cache(maxsize=1)
def _logo() -> dict:
    return json.loads((ASSETS / "logo.json").read_text(encoding="utf-8"))


def _hex(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def draw_logo(page: pymupdf.Page, box: pymupdf.Rect, parts=("emblem", "diamond", "word")) -> pymupdf.Rect:
    """The Arizon logo as vector shapes, fitted (centred) into box. Returns the
    rectangle actually used."""
    data = _logo()
    bb = None
    for p in parts:
        r = pymupdf.Rect(data["boxes"][p])
        bb = r if bb is None else bb | r
    s = min(box.width / bb.width, box.height / bb.height)
    ox = box.x0 + (box.width - bb.width * s) / 2 - bb.x0 * s
    oy = box.y0 + (box.height - bb.height * s) / 2 - bb.y0 * s
    colors = {"emblem": YELLOW, "word": YELLOW, "diamond": RED}
    for p in parts:
        shape = page.new_shape()
        for poly in data["parts"][p]:
            pts = [pymupdf.Point(ox + x * s, oy + y * s) for x, y in poly]
            shape.draw_polyline(pts + [pts[0]])
        shape.finish(fill=_hex(colors[p]), color=None, even_odd=True, closePath=True)
        shape.commit()
    return pymupdf.Rect(ox + bb.x0 * s, oy + bb.y0 * s, ox + bb.x1 * s, oy + bb.y1 * s)


# ============================================================ cell values ==
# One consistent look: Persian digits in Persian text and prices, Latin digits
# in codes and next to Latin words (e.g. «HNBR-2000», «EF7»).

_NUM = re.compile(r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩,٬،/.'’٫  ]*[0-9۰-۹٠-٩]|[0-9۰-۹٠-٩]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


_DIGIT_RUN = re.compile(r"[0-9۰-۹٠-٩]+")
_LETTER = re.compile(r"[A-Za-zؠ-يٮ-ۓۺ-ۿ]")


def digits(text: str, role: str, rtl: bool) -> str:
    if role == "code" or not rtl:
        return text.translate(TO_LATIN)

    def one(m: re.Match) -> str:
        a, b = m.start(), m.end()
        near = [text[a - 1] if a else "", text[b] if b < len(text) else ""]
        letters = [c for c in near if c and _LETTER.match(c)]
        if not letters:
            # no letter right beside it: the closest letters of the same word decide
            left = re.search(r"(\S*?)$", text[:a]).group(1)
            right = re.match(r"\S*", text[b:]).group(0)
            letters = [c for c in left if _LETTER.match(c)][-1:] + [c for c in right if _LETTER.match(c)][:1]
        use_latin = bool(letters) and all(_LATIN_LETTER.match(c) for c in letters)
        return m.group(0).translate(TO_LATIN if use_latin else FA_DIGITS)
    return _DIGIT_RUN.sub(one, text)


_SHEET_ERROR = re.compile(r"#(N/A|REF!?|VALUE!?|DIV/0!?|NAME\??|NULL!?|NUM!?)", re.I)


def _no_errors(text: str) -> str:
    """Spreadsheet error values (#N/A…) left in the list are shown as a dash."""
    return _SHEET_ERROR.sub("—", text)


def money(value: Decimal, decimals: int, rtl: bool) -> str:
    q = value.quantize(Decimal(1).scaleb(-decimals))
    out = f"{q:,.{decimals}f}"
    return out.replace(".", "٫").translate(FA_DIGITS) if rtl else out


def price_text(cell: Cell, items: dict[str, PriceItem], values: dict[str, Decimal], rtl: bool) -> str:
    """The new price(s) of a cell, plus any words the cell had (e.g. «تومان»)."""
    news = []
    for pid in cell.price_ids:
        it = items.get(pid)
        if it is None:
            continue
        news.append(money(values.get(pid, it.value), it.fmt.decimals, rtl))
    words = _NUM.sub("", cell.text).strip(" -/،,")
    text = " / ".join(news)
    if words:
        text = f"{text} {digits(words, '', rtl)}"
    return text


# ================================================================ layout ==

@dataclass
class Style:
    font_size: float = 8.6
    pad_v: float = 2.6

    def css(self) -> str:
        fs = self.font_size
        return _CSS % {"fs": fs, "pv": self.pad_v, "ph": self.pad_v + 1.4, "hfs": fs + 0.2, "sfs": fs - 0.6,
                       "gfs": fs + 0.8}


@lru_cache(maxsize=4)
def _font(name: str) -> pymupdf.Font:
    return pymupdf.Font(fontfile=str(ASSETS / name))


def _text_w(text: str, size: float, bold: bool = False) -> float:
    f = _font("Vazirmatn-Bold.ttf" if bold else "Vazirmatn-Regular.ttf")
    return f.text_length(text, fontsize=size)


@dataclass
class Col:
    role: str
    header: str
    width: float = 0.0
    align: str = ""        # "", "r", "l"
    klass: str = ""        # "p" price, "n" row number, "c" code


@dataclass
class Prepared:
    """A table ready to draw: columns in visual order (left to right) and the
    rows as HTML."""
    table: Table
    cols: list[Col]
    head_html: str
    rows_html: list[str]
    heights: list[float] = field(default_factory=list)
    head_h: float = 0.0
    title_h: float = 0.0
    width: float = 0.0


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _col_widths(t: Table, texts: list[list[str]], images: list[bool], avail: float, fs: float) -> list[float]:
    n = t.ncols
    need = []
    for k in range(n):
        role = t.roles[k] if k < len(t.roles) else "text"
        head = t.headers[k] if k < len(t.headers) else ""
        vals = sorted((_text_w(x, fs, role == "price") for x in texts[k] if x), reverse=True)
        # the widest few cells decide, but one freak cell must not blow the column up
        body = vals[min(len(vals) - 1, max(0, len(vals) // 50))] if vals else 0.0
        words = head.split()
        head_w = max([_text_w(w, fs + 0.2, True) for w in words] or [0.0])
        head_full = _text_w(head, fs + 0.2, True)
        w = max(body, min(head_full, max(head_w, body) * 1.0), head_w) + 9
        if images[k]:
            w = max(w, 58)
        if role == "row":
            w = max(w, 24)
        need.append(w)
    names = [k for k in range(n) if (t.roles[k] if k < len(t.roles) else "") == "name"]
    total = sum(need)
    if total <= avail:
        extra = avail - total
        grow = names or [max(range(n), key=lambda k: need[k])]
        for k in grow:
            need[k] += extra / len(grow)
        return need
    # too wide: squeeze the long text columns (they wrap), keep numbers intact
    fixed = [k for k in range(n) if (t.roles[k] if k < len(t.roles) else "") in ("row", "price", "qty", "unit")]
    fixed_w = sum(need[k] for k in fixed)
    flex = [k for k in range(n) if k not in fixed]
    room = max(avail - fixed_w, 40.0 * len(flex))
    flex_w = sum(need[k] for k in flex) or 1.0
    for k in flex:
        need[k] = max(36.0, need[k] * room / flex_w)
    s = avail / sum(need)
    return [w * s for w in need] if s < 1 else need


# a column whose header the list does not show (or shows unreadably)
DEFAULT_HEADERS = {"row": "ردیف", "code": "کد کالا", "name": "شرح کالا", "price": "قیمت", "qty": "تعداد",
                   "unit": "واحد", "image": "تصویر"}


def prepare(t: Table, items: dict[str, PriceItem], values: dict[str, Decimal], avail: float,
            style: Style, images: dict[str, bytes]) -> Prepared:
    n = t.ncols
    roles = (t.roles + ["text"] * n)[:n]
    # a header the list does not show, or shows with unreadable letters, gets a plain name
    headers = [digits(h, "", t.rtl) if h and "\ufffd" not in h else (DEFAULT_HEADERS.get(r, "") if t.rtl else "")
               for h, r in zip((t.headers + [""] * n)[:n], roles)]
    texts: list[list[str]] = [[] for _ in range(n)]
    has_img = [False] * n
    shown: list[list[str] | None] = []
    for row in t.rows:
        if row.kind != "item":
            shown.append(None)
            continue
        vals = []
        for k in range(n):
            c = row.cells[k] if k < len(row.cells) else Cell()
            v = price_text(c, items, values, t.rtl) if c.price_ids else digits(_no_errors(c.text), roles[k], t.rtl)
            vals.append(v)
            texts[k].append(v)
            has_img[k] |= c.image is not None
        shown.append(vals)
    widths = _col_widths(t, texts, has_img, avail, style.font_size)
    cols = []
    for k in range(n):
        role = roles[k]
        klass = {"price": "p", "row": "n", "code": "c"}.get(role, "")
        align = ""
        if role in ("name", "text") and texts[k] and sum(len(x) for x in texts[k]) / max(1, len(texts[k])) > 18:
            align = "s"     # long descriptions start at the reading edge
        cols.append(Col(role, headers[k], widths[k], align, klass))
    order = list(range(n))[::-1] if t.rtl else list(range(n))

    # header: neighbouring columns under one spanning header cell are shown merged
    head = []
    k = 0
    vis = [(i, cols[i]) for i in order]
    while k < len(vis):
        i, c = vis[k]
        span = 1
        while k + span < len(vis) and vis[k + span][1].header and vis[k + span][1].header == c.header:
            span += 1
        w = sum(v[1].width for v in vis[k:k + span])
        attrs = f' colspan="{span}"' if span > 1 else ""
        dh = ' dir="rtl"' if t.rtl else ""
        head.append(f'<td{attrs}{dh} style="width:{w - PAD:.1f}pt">{_esc(c.header)}</td>')
        k += span
    head_html = f'<tr class="h" id="hd">{"".join(head)}</tr>'

    rows_html = []
    zebra = 0
    # under dir="rtl" MuPDF lays text out right to left and "left" means the start edge
    d = ' dir="rtl"' if t.rtl else ""
    for r, (row, vals) in enumerate(zip(t.rows, shown)):
        if vals is None:
            c = row.cells[0] if row.cells else Cell()
            inner = _esc(digits(c.text, "", t.rtl))
            if c.snapshot is not None:
                name = f"im{len(images)}.png"
                images[name] = c.snapshot
                h = _snap_h(c.snapshot) * (style.font_size + 0.8) / (c.snap_size or style.font_size)
                inner = f'<img src="{name}" style="height:{min(h, 30.0):.1f}pt"/>'
            rows_html.append(f'<tr class="g" id="r{r}"><td colspan="{n}" class="s"{d}>{inner}</td></tr>')
            zebra = 0
            continue
        tds = []
        for i in order:
            c = row.cells[i] if i < len(row.cells) else Cell()
            col = cols[i]
            klass = " ".join(x for x in (col.klass, col.align) if x)
            inner = _esc(vals[i])
            if c.image is not None:
                name = f"im{len(images)}.jpg"
                images[name] = c.image
                inner = f'<img src="{name}" style="height:34pt"/>' + (f"<br/>{inner}" if inner else "")
            elif c.snapshot is not None and not c.price_ids:
                # text the PDF names wrongly: shown exactly as printed
                name = f"im{len(images)}.png"
                images[name] = c.snapshot
                h = _snap_h(c.snapshot) * style.font_size / (c.snap_size or style.font_size)
                inner = f'<img src="{name}" style="height:{min(h, 40.0):.1f}pt"/>'
            kl = f' class="{klass}"' if klass else ""
            tds.append(f'<td{kl}{d} style="width:{col.width - PAD:.1f}pt">{inner}</td>')
        zc = ' class="z"' if zebra % 2 else ""
        rows_html.append(f'<tr{zc} id="r{r}">{"".join(tds)}</tr>')
        zebra += 1
    return Prepared(t, cols, head_html, rows_html, width=sum(widths))


def _snap_h(data: bytes) -> float:
    """Printed height (pt) of a text snapshot."""
    try:
        pix = pymupdf.Pixmap(data)
        return pix.height * 72 / SNAP_DPI
    except Exception:  # noqa: BLE001
        return 12.0


def _archive(images: dict[str, bytes]) -> pymupdf.Archive:
    arch = pymupdf.Archive(str(ASSETS))
    for name, data in images.items():
        arch.add((data, name))
    return arch


def _table_html(p: Prepared, rows: list[int], title: str = "") -> str:
    out = []
    if title:
        d = ' dir="rtl"' if p.table.rtl else ""
        out.append(f'<p class="t"{d} style="text-align:left; margin-bottom:4pt">{_esc(digits(title, "", p.table.rtl))}</p>')
    body = "".join(p.rows_html[r] for r in rows)
    out.append(f'<table style="width:{p.width:.1f}pt">{p.head_html}{body}</table>')
    return "".join(out)


def measure(p: Prepared, css: str, arch: pymupdf.Archive) -> None:
    """Height of the header and of every row at the final column widths."""
    got: dict[str, list[float]] = {}

    def cb(pos) -> None:
        if pos.id:
            r = pos.rect
            v = got.setdefault(pos.id, [r[1], r[3]])
            v[0], v[1] = min(v[0], r[1]), max(v[1], r[3])

    story = pymupdf.Story(html=_table_html(p, list(range(len(p.rows_html)))), user_css=css, archive=arch)
    story.place(pymupdf.Rect(0, 0, p.width + 40, 1e6))
    story.element_positions(cb)
    p.head_h = (got["hd"][1] - got["hd"][0]) if "hd" in got else 20.0
    heights = []
    for r in range(len(p.rows_html)):
        v = got.get(f"r{r}")
        heights.append(v[1] - v[0] if v else 16.0)
    p.heights = heights
    if p.table.title:
        tst = pymupdf.Story(html=_table_html(p, [], p.table.title), user_css=css, archive=arch)
        tst.place(pymupdf.Rect(0, 0, p.width + 40, 1e6))
        pos: dict[str, float] = {}
        tst.element_positions(lambda e: pos.__setitem__("y", max(pos.get("y", 0), e.rect[3])))
        p.title_h = max(0.0, pos.get("y", 0) - p.head_h) + 2


# ============================================================ pagination ==

@dataclass
class Slice:
    prep: Prepared
    rows: list[int]
    title: str = ""
    height: float = 0.0


@dataclass
class PageSpec:
    slices: list[Slice] = field(default_factory=list)
    frame: Frame | None = None
    notes: list[str] = field(default_factory=list)
    notes_h: float = 0.0
    notes_box: pymupdf.Rect | None = None


def paginate(preps: list[Prepared | Frame], first_top: float, top: float, bottom: float,
             gap: float = 12.0) -> list[PageSpec]:
    pages: list[PageSpec] = [PageSpec()]
    y = first_top

    def new_page() -> None:
        nonlocal y
        pages.append(PageSpec())
        y = top

    for p in preps:
        if isinstance(p, Frame):
            if pages[-1].slices or pages[-1].frame:
                new_page()
            pages[-1].frame = p
            new_page()
            continue
        rows = list(range(len(p.rows_html)))
        title = p.table.title
        k = 0
        while k < len(rows):
            need_head = (p.title_h if title else 0) + p.head_h
            # at least the header and two rows (or a group bar and its first row) together
            first = p.heights[k] + (p.heights[k + 1] if k + 1 < len(rows) else 0)
            if y + need_head + first > bottom and (pages[-1].slices or pages[-1].frame):
                new_page()
            h = need_head
            chunk = []
            while k < len(rows) and y + h + p.heights[k] <= bottom:
                chunk.append(rows[k])
                h += p.heights[k]
                k += 1
            if not chunk:            # a single row taller than a page: place it anyway
                chunk.append(rows[k])
                h += p.heights[k]
                k += 1
            # never end a page on a group bar
            while len(chunk) > 1 and p.rows_html[chunk[-1]].startswith('<tr class="g"') and k < len(rows):
                k -= 1
                h -= p.heights[chunk.pop()]
            pages[-1].slices.append(Slice(p, chunk, title, h))
            y += h + gap
            title = ""
            if k < len(rows):
                new_page()
    if not pages[-1].slices and not pages[-1].frame and len(pages) > 1:
        pages.pop()
    return pages


# ================================================================ drawing ==
# Under dir="rtl" MuPDF mirrors text-align: "left" is the start (right) edge.

@dataclass
class Meta:
    title: str = "لیست قیمت محصولات"
    subtitle: str = ""            # the user's line under the title («باباپارت»), empty: none
    currency: str = ""
    item_count: int = 0
    date: datetime | None = None


TITLE_TOP = 17.0                  # where the title starts in the band when it stands alone
SUBTITLE_SIZES = (13.0, 12.0, 11.0, 10.0, 9.0)
SUBTITLE_MIN = 8.5                # below this a long line wraps onto a second line instead
SUBTITLE_MAX = 70                 # characters: two lines of the band at the smallest size
_PERSIAN = re.compile("[\u0600-\u06ff]")
_INVISIBLE = re.compile("[\u200b\u200d-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def clean_subtitle(text: str) -> str:
    """The user's line under the title as it can be drawn: one line, Persian
    letters (ي/ك typed on an Arabic keyboard become ی/ک), no direction marks, no
    characters the template's font does not have (emoji), digits in the script
    of the words beside them."""
    t = unicodedata.normalize("NFKC", text or "").replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    t = _INVISIBLE.sub("", t)
    font = _font("Vazirmatn-Bold.ttf")
    t = "".join(c if c.isspace() or c == "\u200c" or font.has_glyph(ord(c)) else " " for c in t)
    t = re.sub(r"\s+", " ", t).strip(" \u200c")
    if _PERSIAN.search(t):
        t = digits(t, "", True)
    return t


def subtitle_size(text: str, width: float) -> float:
    """The largest size at which the line under the title fits the band on one line."""
    for size in SUBTITLE_SIZES:
        if _text_w(text, size, bold=True) * 1.04 <= width:
            return size
    return SUBTITLE_MIN


def _band_html(meta: Meta, sub_size: float = SUBTITLE_SIZES[0]) -> str:
    sub = (f'<p dir="rtl" style="text-align:left; font-family:vz; font-weight:bold; font-size:{sub_size}pt; '
           f'color:{YELLOW}; margin-top:1.5pt">{_esc(meta.subtitle)}</p>') if meta.subtitle else ""
    return (f'<p dir="rtl" style="text-align:left; font-family:vzb; font-size:19pt; color:#FFFFFF">'
            f'{_esc(meta.title)}</p>{sub}')


def _band(meta: Meta, css: str, arch, width: float) -> tuple[str, float]:
    """The title (and the line under it) and where it starts: alone, the title stands
    where it always has; with a line under it, the two are centred in the band."""
    if not meta.subtitle:
        return _band_html(meta), TITLE_TOP
    size = subtitle_size(meta.subtitle, width)
    html_text = _band_html(meta, size)
    h = _story_h(html_text, css, arch, width)
    return html_text, max(5.0, (BAND_H - h) / 2 + 1.0)


def _date_html(meta: Meta) -> str:
    num, words = today_fa(meta.date)
    return (f'<p dir="rtl" style="text-align:right; font-family:vzm; font-size:7.5pt; color:#BDBDBD">'
            f'تاریخ به‌روزرسانی</p>'
            f'<p dir="rtl" style="text-align:right; font-family:vzb; font-size:14pt; color:{YELLOW}">{num}</p>'
            f'<p dir="rtl" style="text-align:right; font-family:vz; font-size:8pt; color:#FFFFFF">{words}</p>')


@dataclass
class Pill:
    label: str
    value: str
    lab: pymupdf.Rect
    val: pymupdf.Rect


def _pills(meta: Meta, W: float, y: float) -> list[Pill]:
    """Small label/value chips under the band, laid out from the right."""
    data = []
    if meta.item_count:
        data.append(("تعداد اقلام", fa(meta.item_count)))
    if meta.currency:
        data.append(("واحد قیمت‌ها", meta.currency))
    x = W - MARGIN
    out = []
    for label, value in data:
        lw = _text_w(label, 8.0) * 1.08 + 16
        vw = _text_w(value, 9.0, True) * 1.08 + 18
        lab = pymupdf.Rect(x - lw, y, x, y + 18)
        val = pymupdf.Rect(x - lw - vw, y, x - lw, y + 18)
        out.append(Pill(label, value, lab, val))
        x -= lw + vw + 10
    return out


def _chip_html(text: str, color: str, size: float, font: str) -> str:
    return (f'<p dir="rtl" style="text-align:center; font-family:{font}; font-size:{size}pt; color:{color}">'
            f'{_esc(text)}</p>')


def _footer_html(page: int, pages: int) -> str:
    return (f'<p dir="rtl" style="text-align:right; font-family:vzm; font-size:8pt; color:#FFFFFF">'
            f'صفحه {fa(page)} از {fa(pages)}</p>')


def _notes_html(notes: list[str]) -> str:
    items = "".join(f'<p dir="rtl" style="text-align:left; margin-top:2.5pt">■ {_esc(n)}</p>' for n in notes)
    return (f'<div style="font-size:8.3pt; color:#333333">'
            f'<p dir="rtl" style="text-align:left; font-family:vzb; font-size:9.5pt; color:{BLACK}; '
            f'margin-bottom:2pt">توضیحات</p>{items}</div>')


def _story_h(html_text: str, css: str, arch, width: float) -> float:
    st = pymupdf.Story(html=html_text, user_css=css, archive=arch)
    _, filled = st.place(pymupdf.Rect(0, 0, width, 1e5))
    return filled[3]


def render(content: Content, items: dict[str, PriceItem], values: dict[str, Decimal], out_pdf: Path,
           meta: Meta) -> int:
    """Writes the Arizon PDF. Returns the number of pages."""
    style = Style()
    css = style.css()
    images: dict[str, bytes] = {}
    tables = [b for b in content.blocks if isinstance(b, Table)]
    widest = max((t.ncols for t in tables), default=0)
    landscape = widest >= 8
    page_rect = pymupdf.Rect(0, 0, A4.height, A4.width) if landscape else pymupdf.Rect(A4)
    W, H = page_rect.width, page_rect.height
    avail = W - 2 * MARGIN
    preps: list[Prepared | Frame] = []
    for b in content.blocks:
        preps.append(b if isinstance(b, Frame) else prepare(b, items, values, avail, style, images))
    arch = _archive(images)
    for p in preps:
        if isinstance(p, Prepared):
            measure(p, css, arch)

    pills = _pills(meta, W, BAND_H + 13)
    top = BAND_H + 18
    first_top = top + (26 if pills else 0)
    bottom = H - FOOT_H - 12
    pages = paginate(preps, first_top, top, bottom)

    notes_html = _notes_html(content.notes) if content.notes else ""
    notes_h = _story_h(notes_html, css, arch, avail - 24) + 18 if notes_html else 0.0
    # the notes go under the last table, or on a page of their own
    if notes_html:
        last = pages[-1]
        used = (first_top if len(pages) == 1 else top) + sum(s.height + 12 for s in last.slices)
        if last.frame is None and used + notes_h <= bottom:
            last.notes, last.notes_h = content.notes, used
        else:
            pages.append(PageSpec(notes=content.notes, notes_h=top))
    for spec in pages:
        if spec.notes:
            spec.notes_box = pymupdf.Rect(MARGIN, spec.notes_h, W - MARGIN, spec.notes_h + notes_h - 6)

    band_x0, band_x1 = MARGIN + 190, W - MARGIN - LOGO_W - 16
    band_html, band_top = _band(meta, css, arch, band_x1 - band_x0)

    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    total = len(pages)
    for k, spec in enumerate(pages):
        dev = writer.begin_page(page_rect)

        def draw(html_text: str, rect: pymupdf.Rect) -> None:
            st = pymupdf.Story(html=html_text, user_css=css, archive=arch)
            st.place(rect)
            st.draw(dev)

        draw(band_html, pymupdf.Rect(band_x0, band_top, band_x1, BAND_H - 2))
        draw(_date_html(meta), pymupdf.Rect(MARGIN, 11, MARGIN + 150, BAND_H - 2))
        if k == 0:
            for pl in pills:
                for text, r, color, size, font in ((pl.label, pl.lab, YELLOW, 8.0, "vzm"),
                                                   (pl.value, pl.val, BLACK, 9.0, "vzb")):
                    dy = (r.height - 1.2 * size) / 2
                    draw(_chip_html(text, color, size, font), pymupdf.Rect(r.x0, r.y0 + dy, r.x1, r.y1 + 6))
        y = first_top if k == 0 else top
        for s in spec.slices:
            p = s.prep
            x0 = MARGIN + (avail - p.width) / 2
            draw(_table_html(p, s.rows, s.title), pymupdf.Rect(x0, y, x0 + p.width + 2, y + s.height + 40))
            y += s.height + 12
        if spec.notes:
            b = spec.notes_box
            draw(notes_html, pymupdf.Rect(b.x0 + 12, b.y0 + 8, b.x1 - 12, bottom + 4))
        draw(_footer_html(k + 1, total), pymupdf.Rect(MARGIN, H - FOOT_H + 9.5, W / 2, H))
        writer.end_page()
    writer.close()

    doc = pymupdf.open("pdf", buf.getvalue())
    for k, (page, spec) in enumerate(zip(doc, pages)):
        _decorate(page, spec, W, H, pills if k == 0 else [])
    title = f"{meta.title} — {meta.subtitle}" if meta.subtitle else meta.title
    doc.set_metadata({"title": title, "creator": "Arizon", "producer": "Arizon"})
    try:
        _fix_text_layer(doc)
    except Exception:  # noqa: BLE001 - fontTools missing: the pages look the same either way
        log.warning("arizon: text layer not fixed", exc_info=True)
    doc.save(out_pdf, garbage=3, deflate=True)
    n = doc.page_count
    doc.close()
    return n


LOGO_W = 112.0


def _decorate(page: pymupdf.Page, spec: PageSpec, W: float, H: float, pills: list[Pill]) -> None:
    """Band, stripes, logo, chips, footer bar and note box: vector shapes under the text."""
    black, yellow, red = _hex(BLACK), _hex(YELLOW), _hex(RED)
    ops: list = []    # drawn under the text; overlay=False prepends, so they are applied in reverse

    def rect(r: pymupdf.Rect, fill, radius=None) -> None:
        ops.append(lambda: page.draw_rect(r, color=None, fill=fill, overlay=False, radius=radius))

    def poly(pts: list[pymupdf.Point], fill) -> None:
        ops.append(lambda: page.draw_polyline(pts, color=None, fill=fill, closePath=True, overlay=False))

    rect(pymupdf.Rect(0, 0, W, BAND_H), black)
    rect(pymupdf.Rect(0, BAND_H, W, BAND_H + 2.4), red)
    rect(pymupdf.Rect(0, BAND_H + 2.4, W, BAND_H + 4.4), yellow)
    # the date sits in its own darker panel, cut on a slant like the emblem
    x = MARGIN + 150
    poly([pymupdf.Point(0, 0), pymupdf.Point(x + 14, 0), pymupdf.Point(x - 8, BAND_H), pymupdf.Point(0, BAND_H)],
         _hex("#1E1E1E"))
    poly([pymupdf.Point(x + 14, 0), pymupdf.Point(x + 20, 0), pymupdf.Point(x - 2, BAND_H),
          pymupdf.Point(x - 8, BAND_H)], red)
    draw_logo(page, pymupdf.Rect(W - MARGIN - LOGO_W, 9, W - MARGIN, BAND_H - 9))
    for pl in pills:
        rect(pl.lab, black, 0.25)
        rect(pl.val, _hex("#F2F2EE"), 0.25)
        rect(pymupdf.Rect(pl.lab.x1 - 3, pl.lab.y0, pl.lab.x1, pl.lab.y1), yellow)
    # footer
    rect(pymupdf.Rect(0, H - FOOT_H + 4, W, H), black)
    rect(pymupdf.Rect(0, H - FOOT_H + 2, W, H - FOOT_H + 4), yellow)
    draw_logo(page, pymupdf.Rect(W - MARGIN - 64, H - FOOT_H + 10, W - MARGIN, H - 7), parts=("word",))
    if spec.frame is not None:
        f = spec.frame
        area = pymupdf.Rect(MARGIN, BAND_H + 18 + (26 if pills else 0), W - MARGIN, H - FOOT_H - 12)
        s = min(area.width / f.width, area.height / f.height)
        w, h = f.width * s, f.height * s
        r = pymupdf.Rect(area.x0 + (area.width - w) / 2, area.y0, area.x0 + (area.width + w) / 2, area.y0 + h)
        page.insert_image(r, stream=f.png)
        page.draw_rect(r, color=_hex(LINE), width=0.6)
    if spec.notes:
        b = spec.notes_box
        rect(b, _hex("#F7F7F3"))
        rect(pymupdf.Rect(b.x1 - 3, b.y0, b.x1, b.y1), red)
    for op in reversed(ops):
        op()


# ============================================================= text layer ==
# MuPDF names the contextual forms of Persian letters (reached through the
# font's substitution tables, not its character map) with made-up characters,
# so the PDF looked right but copied / searched as «Уژو» for «پژو». Each
# embedded Vazirmatn gets a character map rebuilt from the font itself.

_FONT_FILES = {"Vazirmatn Regular": "Vazirmatn-Regular.ttf", "Vazirmatn Medium": "Vazirmatn-Medium.ttf",
               "Vazirmatn Bold": "Vazirmatn-Bold.ttf", "Vazirmatn Black": "Vazirmatn-Black.ttf"}


def _is_presentation(ch: str) -> bool:
    return "\ufb50" <= ch <= "\ufdff" or "\ufe70" <= ch <= "\ufeff"


@lru_cache(maxsize=4)
def _glyph_text(file: str) -> dict[int, str]:
    """Glyph id -> the text it stands for, from the font's cmap and GSUB."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(ASSETS / file))
    order = font.getGlyphOrder()
    gid = {name: k for k, name in enumerate(order)}
    text: dict[str, str] = {}
    for code, name in sorted(font.getBestCmap().items()):
        ch = chr(code)
        old = text.get(name)
        # a glyph reached from a letter and from its presentation form stands for the letter
        if old is None or (_is_presentation(old) and not _is_presentation(ch)):
            text[name] = ch
    lookups = font["GSUB"].table.LookupList.Lookup if "GSUB" in font else []
    for _ in range(4):                   # forms of forms (init -> calt alternate...)
        before = len(text)
        for lookup in lookups:
            for sub in lookup.SubTable:
                kind = lookup.LookupType
                if kind == 7:
                    kind, sub = sub.ExtensionLookupType, sub.ExtSubTable
                if kind == 1:
                    for a, b in sub.mapping.items():
                        if a in text and b not in text:
                            text[b] = text[a]
                elif kind == 3:
                    for a, alts in sub.alternates.items():
                        for b in alts:
                            if a in text and b not in text:
                                text[b] = text[a]
                elif kind == 4:
                    for first, ligs in sub.ligatures.items():
                        for lig in ligs:
                            parts = [first] + list(lig.Component)
                            if lig.LigGlyph not in text and all(x in text for x in parts):
                                text[lig.LigGlyph] = "".join(text[x] for x in parts)
        if len(text) == before:
            break
    out = {}
    for name, t in text.items():
        if name in gid:
            out[gid[name]] = unicodedata.normalize("NFKC", t) if any(_is_presentation(c) for c in t) else t
    return out


def _cmap_stream(mapping: dict[int, str]) -> bytes:
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def",
             "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def",
             "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange"]
    items = sorted(mapping.items())
    for k in range(0, len(items), 100):
        chunk = items[k:k + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        for g, t in chunk:
            hexes = "".join(f"{b:02X}" for b in t.encode("utf-16-be"))
            lines.append(f"<{g:04X}> <{hexes}>")
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return "\n".join(lines).encode("ascii")


def _fix_text_layer(doc: pymupdf.Document) -> None:
    done: set[int] = set()
    for page in doc:
        for xref, _ext, kind, base, *_ in page.get_fonts(full=True):
            file = _FONT_FILES.get(base.split("+")[-1])
            if xref in done or kind != "Type0" or file is None:
                continue
            done.add(xref)
            new = doc.get_new_xref()
            doc.update_object(new, "<<>>")
            doc.update_stream(new, _cmap_stream(_glyph_text(file)))
            doc.xref_set_key(xref, "ToUnicode", f"{new} 0 R")
