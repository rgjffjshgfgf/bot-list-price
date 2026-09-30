"""Groups of a price list: the section titles that split it («گروه پژو 405»,
«ترموستات ها», «ایران خودرو»...) and which prices fall under each.

The bot's own AI sees groups the way it sees prices: on the words of a page
(PDF text layer or OCR of a photo), so what it learns on one kind of file
helps the other. Lists show groups in two ways:

* a title bar: a row of its own across the table, often shaded, bold or
  centred, usually followed by row numbers that start again at 1;
* a group column: the name written once beside all rows of the group (a
  merged cell), sometimes sideways.

Title bars are recognised by a learned model (features of each line that holds
no price: where it is, how it looks, what it says, what follows it), group
columns by their shape. A group goes on over the page break until the next
title; a page without any title continues the group of the page before.
"""
from __future__ import annotations

import math
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from ..numfmt import to_latin_digits
from .layout import Box, PageDoc, Tok, _cluster, canon, digits_of, keyword_groups

# ============================================================== prior model ==

PRIOR: dict[str, float] = {
    "bias": -3.2,
    "pos:inside": 1.6, "pos:above": 0.6, "pos:top": -2.2, "pos:below": -0.4, "pos:foot": -3.5,
    "hdr_next": 1.3, "hdr_prev": -0.8, "restart": 2.2, "rowno": -4.5, "near_price": -2.0,
    "ntok:1": 0.5, "ntok:2-3": 0.8, "ntok:4-6": 0.2, "ntok:7+": -1.2,
    "chunks:1": 1.6, "chunks:2": -0.6, "chunks:3+": -3.2,
    "align:center": 0.9, "align:start": 0.3, "align:end": -0.3, "wide": 0.2,
    "noletters": -3.5, "kw:group": 2.6, "kw:title": -2.2, "kw:note": -3.2, "kw:hdr": -1.6, "kw:hdr2": -3.8,
    "kw:date": -2.6, "kw:phone": -3.2, "kw:money": -1.2, "kw:unit": -0.8,
    "len:long": -1.8, "len:short": 0.3, "big": 0.8, "small": -0.6,
    "fill": 1.4, "fill:own": 1.0, "repeat": -2.6, "code": -1.8, "outside": -2.0,
    "in_col:text": -0.6, "in_col:num": -2.4, "span_cols": 0.6, "kw:company": -3.0, "num_cell": -3.0,
    "junk": -9.0, "src:ocr": -0.5, "ocr:long": -1.5, "ocr:bar": 1.0,
}

TITLE_WORDS = re.compile(r"لیست|فهرست|قیمت|تاریخ|price|list", re.I)
COMPANY_WORDS = re.compile(r"شرکت|بازرگانی|صنع[تن]ی|صنایع|تولیدی|تولیدکننده|فروشگاه|نمایندگی|www|http|@|\.com|\.ir", re.I)
NOTE_WORDS = re.compile(r"می\s*باشد|میباشد|(?<![\u0600-\u06ff])است(?![\u0600-\u06ff])|هستند|باشد|نمایید|فرمایید|گردد|می\s*شود|"
                        r"میشود|شده|توجه|نکته|لطفا|مالیات|ارزش افزوده|گارانتی|ضمانت|اعتبار|معتبر|موقت|نقدی|چکی")
_MONTH = r"(?<![\u0600-\u06ff])(فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|[اآ]\s?بان|[اآ]\s?ذر|دی|بهمن|اسفند)(?![\u0600-\u06ff])"
DATE = re.compile(r"[\d۰-۹]{2,4}\s*[/\-.]\s*[\d۰-۹]{1,2}\s*[/\-.]\s*[\d۰-۹]{1,4}|"
                  r"[\d۰-۹]{1,2}\s*" + _MONTH + "|" + _MONTH + r"\s*(ماه)?\s*[\d۰-۹]{2,4}|" + _MONTH + r"\s*ماه")
HEADER_WORDS = {canon(w) for w in ["ردیف", "ردبف", "کد", "کالا", "شرح", "نام", "قیمت", "تعداد", "واحد", "عکس",
                                    "تصویر", "مارک", "برند", "کشور", "سازنده", "توضیحات", "مبلغ", "فی", "کارتن",
                                    "بسته", "شماره", "فنی", "محصول", "row", "code", "price", "qty", "description",
                                    "unit", "brand", "image", "photo", "item", "no"]}
_ARABIC = re.compile("[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
_LETTER = re.compile(r"[^\W\d_]")
_LTR = re.compile(r"^[A-Za-z0-9۰-۹٠-٩.,/:%+\-_()#]+$")
_UNREADABLE = re.compile("[\ufffd\ue000-\uf8ff]")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


class GroupModel:
    """Logistic model over line features, on top of the prior weights."""

    def __init__(self, weights: dict | None = None, g2: dict | None = None):
        self.w: dict[str, float] = dict(weights or {})
        self.g2: dict[str, float] = dict(g2 or {})

    def logit(self, feats: list[str]) -> float:
        return sum(PRIOR.get(k, 0.0) + self.w.get(k, 0.0) for k in feats)

    def prob(self, feats: list[str]) -> float:
        return _sigmoid(self.logit(feats))

    def train(self, samples: list[tuple[list[str], int]], epochs: int = 4, lr: float = 0.3,
              l2: float = 1e-3) -> None:
        data = list(samples)
        for _ in range(epochs):
            random.shuffle(data)
            for feats, y in data:
                g = _sigmoid(self.logit(feats)) - y
                for k in feats:
                    wk = self.w.get(k, 0.0)
                    grad = g + l2 * wk
                    acc = self.g2.get(k, 0.0) + grad * grad
                    self.g2[k] = acc
                    wk -= lr * grad / (math.sqrt(acc) + 1e-8)
                    if abs(wk) < 1e-6:
                        self.w.pop(k, None)
                    else:
                        self.w[k] = max(-10.0, min(10.0, wk))


# ================================================================ graphics ==

@dataclass
class Graphics:
    """What the page draws besides text: shaded areas, ruling lines, sideways text."""
    fills: list[tuple[Box, int]] = field(default_factory=list)          # (box, colour 0xRRGGBB)
    hlines: list[tuple[float, float, float]] = field(default_factory=list)   # (y, x0, x1)
    rotated: list[tuple[Box, str]] = field(default_factory=list)
    bg: Callable[[Box], int | None] | None = None      # photos: background colour behind a box
    bars: list[tuple[Box, int]] = field(default_factory=list)   # photos: shaded bands across the page


def _rgb_int(c) -> int | None:
    if c is None:
        return None
    try:
        r, g, b = (max(0, min(255, int(round(v * 255)))) for v in list(c)[:3])
    except (TypeError, ValueError):
        return None
    return (r << 16) | (g << 8) | b


def _whiteish(c: int | None) -> bool:
    if c is None:
        return True
    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
    return min(r, g, b) >= 238


def _similar(a: int | None, b: int | None, tol: int = 24) -> bool:
    if a is None or b is None:
        return a is b
    return all(abs(((a >> s) & 255) - ((b >> s) & 255)) <= tol for s in (16, 8, 0))


def pdf_graphics(page) -> Graphics:
    """Fills, horizontal lines and sideways text of a PDF page (page coordinates
    as the PageDoc uses them: unrotated)."""
    import pymupdf

    g = Graphics()
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - graphics are extra evidence only
        drawings = []
    for d in drawings:
        fill = _rgb_int(d.get("fill"))
        for item in d.get("items", []):
            if item[0] == "re":
                r = item[1]
                if fill is not None and not _whiteish(fill) and r.width > 4 and r.height > 3:
                    g.fills.append(((r.x0, r.y0, r.x1, r.y1), fill))
                if r.height < 2.5:
                    g.hlines.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
                elif d.get("color") is not None:
                    g.hlines += [(r.y0, r.x0, r.x1), (r.y1, r.x0, r.x1)]
            elif item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.y - b.y) < 1:
                    g.hlines.append(((a.y + b.y) / 2, min(a.x, b.x), max(a.x, b.x)))
            elif item[0] == "qu" and fill is not None and not _whiteish(fill):
                r = item[1].rect
                if r.width > 4 and r.height > 3:
                    g.fills.append(((r.x0, r.y0, r.x1, r.y1), fill))
    # sideways text (a group name written vertically in a merged cell)
    try:
        flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES
        for block in page.get_text("dict", flags=flags).get("blocks", []):
            for line in block.get("lines", []):
                if abs(line["dir"][1]) < 0.5:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text and not _UNREADABLE.search(text):
                    b = line["bbox"]
                    g.rotated.append(((b[0], b[1], b[2], b[3]), unicodedata.normalize("NFKC", text)))
    except Exception:  # noqa: BLE001
        pass
    return g


def image_graphics(rgb) -> Graphics:
    """Photos: the background colour behind any box is read from the pixels."""
    import numpy as np

    h, w = rgb.shape[:2]

    def bg(box: Box) -> int | None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        pad = max(2, (y1 - y0) // 3)
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 < 3 or y1 - y0 < 2:
            return None
        part = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
        lum = part @ np.array([299, 587, 114]) // 1000
        light = part[lum >= np.percentile(lum, 55)]
        if not len(light):
            return None
        r, g, b = (int(v) for v in np.median(light, axis=0))
        return (r << 16) | (g << 8) | b
    return Graphics(bg=bg, bars=_shaded_bands(rgb))


def _shaded_bands(rgb) -> list[tuple[Box, int]]:
    """Horizontal bands of one flat colour (not white) across much of a photo:
    title bars, header rows, shaded rows. The text on them is the minority."""
    import numpy as np

    h, w = rgb.shape[:2]
    s = min(1.0, 700 / max(1, w))
    step = max(1, int(round(1 / s)))
    small = rgb[::step, ::step].astype(np.int32)
    sh, sw = small.shape[:2]
    med = np.median(small[:, int(0.1 * sw):int(0.9 * sw)], axis=1)                 # per row
    close = (np.abs(small - med[:, None, :]).max(axis=2) <= 28).mean(axis=1)
    shaded = (med.min(axis=1) < 232) & (close >= 0.55)
    out: list[tuple[Box, int]] = []
    y = 0
    while y < sh:
        if not shaded[y]:
            y += 1
            continue
        y0 = y
        while y + 1 < sh and shaded[y + 1] and np.abs(med[y + 1] - med[y0]).max() <= 30:
            y += 1
        y1 = y + 1
        colour = med[y0:y1].mean(axis=0)
        cols = (np.abs(small[y0:y1] - colour[None, None, :]).max(axis=2) <= 30).mean(axis=0) >= 0.5
        xs = np.nonzero(cols)[0]
        if y1 - y0 >= 3 and len(xs) >= 0.3 * sw:
            r, g, b = (int(v) for v in colour)
            out.append(((float(xs[0] * step), float(y0 * step), float((xs[-1] + 1) * step), float(y1 * step)),
                        (r << 16) | (g << 8) | b))
        y += 1
    return out


# =================================================================== lines ==

@dataclass
class Line:
    toks: list[Tok]
    prices: set[int]                  # indices into doc.nums of the prices on this line
    y0: float
    y1: float
    x0: float
    x1: float
    text: str = ""                    # what the line says, in reading order (lines without a price)
    bar: bool = False                 # photos: a shaded bar read on its own

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x1, self.y1)


def reading_text(toks: list[Tok], rtl: bool) -> str:
    """Words of one line in reading order; Latin/digit runs inside Persian text
    keep their own left-to-right order («گروه MVM X33»)."""
    order = sorted(toks, key=lambda t: -t.xc if rtl else t.xc)
    if rtl:
        out: list[Tok] = []
        run: list[Tok] = []
        for t in order:
            if _LTR.match(t.text) and not _ARABIC.search(t.text):
                run.append(t)
                continue
            out += run[::-1]
            run = []
            out.append(t)
        out += run[::-1]
        order = out
    return clean_name(" ".join(t.text for t in order))


_FA_LETTER = "[\u0621-\u064a\u066e-\u06d3]"      # Persian/Arabic letters (not its digits)


def clean_name(text: str) -> str:
    """A group name as shown: normal Persian letters, no unreadable glyphs, tidy spaces
    (a Persian word and a number apart: «پژو405» -> «پژو 405», as the page shows it)."""
    t = unicodedata.normalize("NFKC", text or "").replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    t = _UNREADABLE.sub(" ", t)
    t = re.sub(f"(?<={_FA_LETTER})(?=\\d)|(?<=\\d)(?={_FA_LETTER})", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" :-–—|،,.*_")
    return t


_KEY = str.maketrans({"ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "آ": "ا",
                      "ؤ": "و", "\u200c": "", "ـ": ""})


def name_key(text: str) -> str:
    """Comparable form of a group name (spacing, letter forms, digit scripts)."""
    t = to_latin_digits(clean_name(text)).translate(_KEY).lower()
    return " ".join(w for w in re.split(r"[^\w]+", t) if w)


def same_group(a: str, b: str) -> bool:
    """Exactly the same group name (spacing, letter forms and digit scripts aside):
    «ایکس 33» and «ایکس 33 کراس» are two groups."""
    ka, kb = name_key(a).replace(" ", ""), name_key(b).replace(" ", "")
    return bool(ka) and ka == kb


def likeness(a: str, b: str) -> float:
    """How alike two names read (0-1, letter pairs in common): an OCR reading of a
    title against its real name («تروه ربو» ~ «گروه ریو»)."""
    def pairs(t: str) -> list[str]:
        k = name_key(t).replace(" ", "")
        return [k[i:i + 2] for i in range(len(k) - 1)] or ([k] if k else [])
    pa, pb = pairs(a), pairs(b)
    if not pa or not pb:
        return 0.0
    rest = list(pb)
    common = 0
    for x in pa:
        if x in rest:
            rest.remove(x)
            common += 1
    return 2 * common / (len(pa) + len(pb))


def same_name(a: str, b: str) -> bool:
    """The same group read by two different readers: equal, or one a large part
    of the other (a name cut by the page break, a reader that missed a word)."""
    ka, kb = name_key(a).replace(" ", ""), name_key(b).replace(" ", "")
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long_ = sorted((ka, kb), key=len)
    return len(short) >= 3 and short in long_ and len(short) >= 0.6 * len(long_)


def _lines(doc: PageDoc, selected: set[int]) -> list[Line]:
    extra = list(getattr(doc, "small", [])) + list(getattr(doc, "hidden", []))
    toks = [(t, i) for i, t in enumerate(doc.nums)] + [(t, -1) for t in doc.words + extra]
    toks.sort(key=lambda p: p[0].yc)
    lh = doc.line_h
    lines: list[list[tuple[Tok, int]]] = []
    for t, i in toks:
        for ln in reversed(lines[-4:]):
            yc = statistics.mean(x.yc for x, _ in ln)
            if abs(t.yc - yc) <= 0.45 * max(lh, min(t.h, max(x.h for x, _ in ln))):
                ln.append((t, i))
                break
        else:
            lines.append([(t, i)])
    out = []
    for ln in lines:
        ts = [t for t, _ in ln]
        out.append(Line(ts, {i for _, i in ln if i >= 0 and i in selected},
                        min(t.box[1] for t in ts), max(t.box[3] for t in ts),
                        min(t.box[0] for t in ts), max(t.box[2] for t in ts)))
    out.sort(key=lambda ln: ln.yc)
    return out


def _text(ctx: "PageContext", toks: list[Tok], box: Box, side: float = 0.3) -> str:
    """What a group of tokens says: the PDF's own text of that spot when there is
    one (its glyphs named even where the file does not), else the words.
    side: room around the box to read in (0: the box is a whole bar)."""
    if ctx.namer is not None:
        try:
            exact = clean_name(ctx.namer(box, side))
        except Exception:  # noqa: BLE001 - the words of the page are still there
            exact = ""
        if exact and _LETTER.search(exact):
            return exact
    return reading_text(toks, ctx.rtl)


def _chunks(toks: list[Tok], gap: float) -> int:
    xs = sorted((t.box[0], t.box[2]) for t in toks)
    n, right = 0, None
    for a, b in xs:
        if right is None or a - right > gap:
            n += 1
        right = b if right is None else max(right, b)
    return n


def _small_int(t: Tok) -> int | None:
    n = t.num
    if n is None or n.fmt.group_sep or n.fmt.decimals or n.value != int(n.value):
        return None
    d = digits_of(t.text)
    return int(d) if d and len(d) <= 4 else None


def _header_hits(text: str) -> int:
    hits = 0
    for w in text.split():
        c = canon(w)
        if not c:
            continue
        if c in HEADER_WORDS or keyword_groups(w) & {"price", "code", "row", "qty"}:
            hits += 1
    return hits


# ============================================================== one page ==

@dataclass
class Cand:
    """A line with no price that may be a group title."""
    line: Line
    name: str
    feats: list[str]
    prob: float = 0.0

    @property
    def box(self) -> Box:
        return self.line.box


@dataclass
class Title:
    """A group title found on a page: prices below `y` (and above the next title)
    belong to it; a column group ends at `end`."""
    name: str
    y: float
    box: Box
    kind: str                         # "bar" | "column"
    end: float | None = None
    score: float = 1.0
    x0: float = -1e9                  # the part of the page it heads (side-by-side tables)
    x1: float = 1e9


@dataclass
class PageContext:
    doc: PageDoc
    selected: set[int]
    graphics: Graphics
    lines: list[Line] = field(default_factory=list)
    price_lines: list[Line] = field(default_factory=list)
    header_lines: list[Line] = field(default_factory=list)
    tx0: float = 0.0
    tx1: float = 0.0
    rows: dict[int, int] = field(default_factory=dict)       # id(price line) -> row number
    row_x: tuple[float, float] | None = None
    price_h: float = 10.0
    rtl: bool = True
    price_fills: list[int | None] = field(default_factory=list)
    bands: list[tuple] = field(default_factory=list)   # ordinary columns: (x0, x1, "text"|"num", usual digits)
    prev_row: int | None = None          # last row number of the page before
    y0: float = 0.0                      # the table rows: from the first to the last row
    y1: float = 0.0
    namer: Callable[[Box], str] | None = None

    def last_row(self) -> int | None:
        nums = [self.rows[id(ln)] for ln in self.price_lines if id(ln) in self.rows]
        return nums[-1] if nums else None


def _bands(ctx: "PageContext") -> list[tuple]:
    """The ordinary columns of the table: filled on most rows with prices."""
    toks = [t for ln in ctx.price_lines for t in ln.toks]
    if not toks:
        return []
    labels = _cluster([t.box for t in toks])
    by: dict[int, list[Tok]] = {}
    for t, lab in zip(toks, labels):
        by.setdefault(lab, []).append(t)
    lh = ctx.doc.line_h
    out = []
    for ts in by.values():
        hit = sum(1 for ln in ctx.price_lines if any(ln.y0 - 0.2 * lh <= t.yc <= ln.y1 + 0.2 * lh for t in ts))
        if hit >= 0.5 * len(ctx.price_lines):
            words = sum(1 for t in ts if t.num is None and _LETTER.search(t.text))
            digits = [len(digits_of(t.text)) for t in ts if digits_of(t.text)]
            out.append((min(t.box[0] for t in ts), max(t.box[2] for t in ts), "text" if words >= 0.5 * len(ts) else "num",
                        statistics.median(digits) if digits else 0))
    return sorted(out)


def _fill_at(g: Graphics, box: Box, min_w: float) -> int | None:
    """Colour of a shaded area under the box (wide enough to be a bar or a row)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if g.bg is not None:
        c = g.bg(box)
        return None if _whiteish(c) else c
    best = None
    for (x0, y0, x1, y1), col in g.fills:
        if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1 and x1 - x0 >= min_w and y1 - y0 <= 6 * max(1.0, box[3] - box[1]):
            if best is None or (x1 - x0) * (y1 - y0) < best[0]:
                best = ((x1 - x0) * (y1 - y0), col)
    return best[1] if best else None


def context(doc: PageDoc, selected: set[int], graphics: Graphics | None = None,
            namer: Callable[[Box], str] | None = None) -> PageContext:
    ctx = PageContext(doc, set(selected), graphics or Graphics(), namer=namer)
    ctx.lines = _lines(doc, ctx.selected)
    ctx.rtl = doc.rtl()
    ctx.price_lines = [ln for ln in ctx.lines if ln.prices]
    if not ctx.price_lines:
        return ctx
    ctx.y0, ctx.y1 = ctx.price_lines[0].y0, ctx.price_lines[-1].y1
    _band_lines(ctx)
    for ln in ctx.lines:
        if not ln.prices and not ln.text:
            ln.text = _text(ctx, ln.toks, ln.box)
    ctx.tx0 = min(ln.x0 for ln in ctx.price_lines)
    ctx.tx1 = max(ln.x1 for ln in ctx.price_lines)
    ctx.price_h = statistics.median(t.h for ln in ctx.price_lines for t in ln.toks if t.h > 0) or doc.line_h
    # the header: column names spread over the columns (a title naming the price is one piece)
    # (an item row whose price could not be read is no header: it has a code or long numbers)
    ctx.header_lines = [ln for ln in ctx.lines if not ln.prices and _header_hits(ln.text) >= 2
                        and (_chunks(ln.toks, 1.2 * doc.line_h) >= 2 or _header_hits(ln.text) >= 3)
                        and ln.y1 >= ctx.price_lines[0].y0 - 8 * doc.line_h
                        and not any(len(digits_of(t.text)) >= 5 for t in ln.toks)]
    # a header word wrapped onto a line of its own («تعداد در» / «کارتن») is header too
    grown = True
    while grown:
        grown = False
        for ln in ctx.lines:
            if ln.prices or ln in ctx.header_lines or not _header_hits(ln.text) or len(ln.toks) > 3:
                continue
            if any(abs(ln.yc - h.yc) <= 1.7 * doc.line_h for h in ctx.header_lines):
                ctx.header_lines.append(ln)
                grown = True
    ctx.bands = _bands(ctx)
    # the row-number column: small whole numbers lined up, counting up
    small = [(t, ln) for ln in ctx.price_lines for t in ln.toks if t.num is not None and _small_int(t) is not None
             and not any(t is ctx.doc.nums[i] for i in ln.prices)]
    if small:
        labels = _cluster([t.box for t, _ in small])
        best = None
        for lab in set(labels):
            members = [small[k] for k in range(len(small)) if labels[k] == lab]
            if len(members) < max(3, 0.5 * len(ctx.price_lines)):
                continue
            members.sort(key=lambda m: m[1].yc)
            vals = [_small_int(t) for t, _ in members]
            ups = sum(1 for a, b in zip(vals, vals[1:]) if b == a + 1 or b == 1 or (b < a and b <= 3))
            if ups >= 0.7 * (len(vals) - 1) and (best is None or len(members) > len(best)):
                best = members
        if best:
            ctx.rows = {id(ln): _small_int(t) for t, ln in best}
            ctx.row_x = (min(t.box[0] for t, _ in best), max(t.box[2] for t, _ in best))
    ctx.price_fills = [_fill_at(ctx.graphics, ln.box, 0.3 * (ctx.tx1 - ctx.tx0)) for ln in ctx.price_lines]
    ctx.y0, ctx.y1 = ctx.price_lines[0].y0, ctx.price_lines[-1].y1
    if ctx.row_x is not None:
        # rows without a price (not available, «0») still belong to the table
        numbered = [ln for ln in ctx.lines if any(t.num is not None and _small_int(t) is not None
                                                 and t.box[2] >= ctx.row_x[0] - 2 and t.box[0] <= ctx.row_x[1] + 2
                                                 for t in ln.toks)]
        if numbered:
            ctx.y0 = min(ctx.y0, min(ln.y0 for ln in numbered))
            ctx.y1 = max(ctx.y1, max(ln.y1 for ln in numbered))
    return ctx


_GROUP_WORD = re.compile("گروه|هورگ")
_PERSIAN_WORD = re.compile("^[\u0600-\u06ff\u200c]{2,}$")
_MODEL = re.compile(r"^[A-Z0-9][A-Z0-9\-/.]*[A-Z0-9]$|^[A-Z]{2,}[a-z]*$")
_JUNK_CHAR = re.compile(r"[“”‘’=_—~|\\\[\]{}<>@©®«»•^*%$#]")


def ocr_junk(text: str) -> bool:
    """OCR that read no real words (a title on a dark bar read as «ee Oy —»): no
    group name may come from it."""
    good = bad = 0
    for w in unicodedata.normalize("NFKC", text).split():
        w = w.strip(".,،:;()-")
        if not w:
            continue
        if _JUNK_CHAR.search(w):
            bad += 1
        elif _PERSIAN_WORD.match(w):
            if len(w.replace("\u200c", "")) >= 3:
                good += 1                 # (short ones - «و», «تا», or OCR crumbs «سس» - say nothing)
        elif w.isdigit() or re.fullmatch(r"[\d۰-۹]+", w):
            continue                      # numbers say nothing either way
        elif _MODEL.match(w):
            good += 1
        else:
            bad += 1
    return good == 0 or bad > good


def _band_lines(ctx: PageContext) -> None:
    """Photos: a shaded bar is read as one line on its own. The page OCR often
    finds nothing or only scraps on light letters over a dark colour; the lines
    it did find inside the bar give way to the bar's own reading."""
    if not ctx.graphics.bars or ctx.namer is None:
        return
    lh = ctx.doc.line_h
    changed = False
    for box, _ in ctx.graphics.bars:
        x0, y0, x1, y1 = box
        if not (0.7 * lh <= y1 - y0 <= 8 * lh) or y1 < ctx.y0 - 12 * lh or y0 > ctx.y1 + 4 * lh:
            continue
        inside = [ln for ln in ctx.lines if y0 - 0.2 * lh <= ln.yc <= y1 + 0.2 * lh]
        if any(ln.prices for ln in inside):
            continue                     # a shaded row of the table
        text = _text(ctx, [], box, 0.0)
        if not text or not _LETTER.search(text):
            continue
        ctx.lines = [ln for ln in ctx.lines if ln not in inside]
        ctx.lines.append(Line([Tok(text, box)], set(), y0, y1, x0, x1, text, bar=True))
        changed = True
    if changed:
        ctx.lines.sort(key=lambda ln: ln.yc)


def candidates(ctx: PageContext) -> list[Cand]:
    """Lines without a price that could be group titles, with their features."""
    if not ctx.price_lines:
        return []
    lh = ctx.doc.line_h
    first, last = ctx.price_lines[0], ctx.price_lines[-1]
    width = max(1.0, ctx.tx1 - ctx.tx0)
    center = (ctx.tx0 + ctx.tx1) / 2
    out = []
    headers = {id(h) for h in ctx.header_lines}
    for k, ln in enumerate(ctx.lines):
        if ln.prices or id(ln) in headers:
            continue
        text = ln.text
        if not text or len(text) < 2:
            continue
        if ln.x1 < ctx.tx0 - 0.5 * width or ln.x0 > ctx.tx1 + 0.5 * width:
            continue
        f = ["bias"]
        # where it is
        if first.y0 - 0.2 * lh <= ln.yc <= last.y1 + 0.2 * lh:
            f.append("pos:inside")
        elif ln.yc < first.y0:
            f.append("pos:above" if first.y0 - ln.y1 <= 7 * lh else "pos:top")
        else:
            f.append("pos:below" if ln.y0 - last.y1 <= 3 * lh else "pos:foot")
        if ln.x1 < ctx.tx0 - 2 or ln.x0 > ctx.tx1 + 2:
            f.append("outside")
        # what follows / precedes it
        below = [x for x in ctx.lines[k + 1:k + 3] if x.y0 >= ln.y1 - 0.3 * lh]
        above = [x for x in ctx.lines[max(0, k - 2):k] if x.y1 <= ln.y0 + 0.3 * lh]
        if below and any(b in ctx.header_lines for b in below[:1]):
            f.append("hdr_next")
        if above and above[-1] in ctx.header_lines:
            f.append("hdr_prev")
        nxt = next((p for p in ctx.price_lines if p.yc > ln.yc), None)
        prv = next((p for p in reversed(ctx.price_lines) if p.yc < ln.yc), None)
        if nxt is not None and id(nxt) in ctx.rows:
            # numbering starts again under it (compared with the rows above, or the page before)
            r = ctx.rows[id(nxt)]
            before = ctx.rows.get(id(prv)) if prv is not None else ctx.prev_row
            if before is not None and r < before and (r <= 1 or prv is not None):
                f.append("restart")
        near = min((abs(p.yc - ln.yc) for p in ctx.price_lines), default=99 * lh)
        if near < 0.8 * lh:
            f.append("near_price")
        if ctx.row_x is not None and any(t.num is not None and _small_int(t) is not None
                                         and t.box[2] >= ctx.row_x[0] - 2 and t.box[0] <= ctx.row_x[1] + 2
                                         for t in ln.toks) and not ln.prices:
            f.append("rowno")
        # how it looks
        n = len(ln.toks)
        f.append("ntok:" + ("1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+"))
        ch = _chunks(ln.toks, 1.6 * lh)
        f.append("chunks:" + ("1" if ch == 1 else "2" if ch == 2 else "3+"))
        mid = (ln.x0 + ln.x1) / 2
        if abs(mid - center) <= 0.12 * width:
            f.append("align:center")
        elif (ctx.rtl and mid > center) or (not ctx.rtl and mid < center):
            f.append("align:start")
        else:
            f.append("align:end")
        if ln.x1 - ln.x0 >= 0.5 * width:
            f.append("wide")
        inside = [b for b in ctx.bands if min(ln.x1, b[1]) - max(ln.x0, b[0]) >= 0.85 * max(1.0, ln.x1 - ln.x0)]
        crossed = [b for b in ctx.bands if min(ln.x1, b[1]) - max(ln.x0, b[0]) > 0.2 * (b[1] - b[0])]
        if inside and len(crossed) <= 1:
            f.append("in_col:" + inside[0][2])
        elif len(crossed) >= 2:
            f.append("span_cols")
        if not _LETTER.search(text):
            f.append("noletters")
        norm = unicodedata.normalize("NFKC", text).replace("ي", "ی").replace("ك", "ک")
        if ctx.doc.source == "ocr":
            f.append("src:ocr")
            if ocr_junk(text):
                f.append("junk")
            if len(norm) > 32:
                f.append("ocr:long")
            if ln.bar:
                f.append("ocr:bar")
        hs = statistics.median(t.h for t in ln.toks if t.h > 0) if any(t.h > 0 for t in ln.toks) else ctx.price_h
        if hs >= 1.2 * ctx.price_h:
            f.append("big")
        elif hs <= 0.75 * ctx.price_h:
            f.append("small")
        # what it says
        if _GROUP_WORD.search(norm):
            f.append("kw:group")
        if TITLE_WORDS.search(norm):
            f.append("kw:title")
        if COMPANY_WORDS.search(norm):
            f.append("kw:company")
        if NOTE_WORDS.search(norm.replace("\u200c", "")):
            f.append("kw:note")
        hh = _header_hits(text)
        if hh >= 2:
            f.append("kw:hdr2")
        elif hh == 1:
            f.append("kw:hdr")
        if DATE.search(norm) or re.search(r"(?<![\d۰-۹])(13|14|۱۳|۱۴)[\d۰-۹]{2}(?![\d۰-۹])", norm):
            f.append("kw:date")
        kws = set()
        for t in ln.toks:
            kws |= keyword_groups(t.text)
        if "phone" in kws:
            f.append("kw:phone")
        if any(canon(t.text) in {canon("ریال"), canon("تومان"), canon("تومن")} for t in ln.toks):
            f.append("kw:money")
        if re.search(r"(^|\s)(عدد|بسته|کارتن|جفت|ست|دست)(\s|$)", norm):
            f.append("kw:unit")
        if len(norm) > 45:
            f.append("len:long")
        elif len(norm) <= 25:
            f.append("len:short")
        if any(len(digits_of(t.text)) >= 5 or re.fullmatch(r"[A-Za-z]{1,4}[\-_]?\d{2,}", t.text) for t in ln.toks):
            f.append("code")
        # a number standing in an ordinary column of numbers like the others there
        # (a code or a quantity of a row whose price is missing)
        for t in ln.toks:
            d = len(digits_of(t.text))
            if d < 4 or t.num is None and _LETTER.search(t.text):
                continue
            if any(b[2] == "num" and b[0] - 2 <= t.box[0] and t.box[2] <= b[1] + 2 and abs(d - b[3]) <= 1
                   for b in ctx.bands if len(b) > 3):
                f.append("num_cell")
                break
        fill = _fill_at(ctx.graphics, ln.box, 0.4 * width)
        if fill is not None:
            f.append("fill")
            same_rows = sum(1 for pf in ctx.price_fills if _similar(pf, fill))
            if same_rows <= 0.2 * max(1, len(ctx.price_fills)):
                f.append("fill:own")
        out.append(Cand(ln, text, f))
    return out


def column_titles(ctx: PageContext) -> list[Title]:
    """Group names written once beside the rows of each group (merged cells),
    upright or sideways."""
    if len(ctx.price_lines) < 3:
        return []
    lh = ctx.doc.line_h
    y_top, y_bot = ctx.y0 - 0.6 * lh, ctx.y1 + 0.6 * lh
    # every token beside the price rows, by column
    toks = [t for ln in ctx.lines if y_top <= ln.yc <= y_bot for t in ln.toks]
    rot = [Tok(text, box) for box, text in ctx.graphics.rotated
           if y_top <= (box[1] + box[3]) / 2 <= y_bot
           and ctx.tx0 - 2 * lh <= (box[0] + box[2]) / 2 <= ctx.tx1 + 2 * lh and _LETTER.search(text)
           and not _header_hits(text)]
    if not toks:
        return []
    labels = _cluster([t.box for t in toks])
    bands: dict[int, list[Tok]] = {}
    for t, lab in zip(toks, labels):
        bands.setdefault(lab, []).append(t)
    n_rows = len(ctx.price_lines)
    full = []            # x ranges of the ordinary columns (filled on most rows)
    for lab, ts in bands.items():
        rows_hit = sum(1 for ln in ctx.price_lines if any(ln.y0 - 0.2 * lh <= t.yc <= ln.y1 + 0.2 * lh for t in ts))
        if rows_hit >= 0.5 * n_rows:
            full.append((min(t.box[0] for t in ts), max(t.box[2] for t in ts)))
    if not full:
        return []
    fx0, fx1 = min(a for a, _ in full), max(b for _, b in full)
    found: list[tuple[list[Tok], str]] = []
    for lab, ts in bands.items():
        x0, x1 = min(t.box[0] for t in ts), max(t.box[2] for t in ts)
        if any(min(x1, b) - max(x0, a) > 0.3 * min(x1 - x0, b - a) for a, b in full):
            continue                   # shares its place with an ordinary column
        named = [t for t in ts if _LETTER.search(t.text) or _UNREADABLE.search(t.text)
                 or (len(digits_of(t.text)) <= 4 and t.num is not None)]
        if not named or len(named) > 0.35 * n_rows:
            continue
        # the header above this column calls it a group column («گروه کالایی»)
        says_group = False
        for hl in ctx.header_lines:
            if hl.yc > min(t.yc for t in ts):
                continue
            hdr = [w for w in hl.toks if w.box[2] >= x0 - 0.5 * lh and w.box[0] <= x1 + 0.5 * lh]
            box = (x0 - 0.5 * lh, hl.y0, x1 + 0.5 * lh, hl.y1)
            if hdr and _GROUP_WORD.search(_text(ctx, hdr, box)):
                says_group = True
        outer = x0 >= fx1 - 0.5 * lh or x1 <= fx0 + 0.5 * lh
        if says_group or outer:
            found.append((named, "header" if says_group else "outer"))
    if rot:
        # a sideways name written over two lines: two narrow boxes side by side
        rot.sort(key=lambda t: t.box[0])
        merged: list[Tok] = []
        for t in rot:
            m = next((x for x in merged if abs(x.box[2] - t.box[0]) <= 0.8 * lh
                      and min(x.box[3], t.box[3]) - max(x.box[1], t.box[1]) > 0.3 * min(x.h, t.h)), None)
            if m is None:
                merged.append(Tok(t.text, t.box))
            else:
                # the line further right is read first (sideways Persian text reads top-down)
                first, second = (m, t) if m.box[0] > t.box[0] else (t, m)
                m.text = first.text + " " + second.text
                m.box = (min(m.box[0], t.box[0]), min(m.box[1], t.box[1]), max(m.box[2], t.box[2]), max(m.box[3], t.box[3]))
        found.append((merged, "rotated"))
    titles: list[Title] = []
    for ts, why in found:
        # a name written over two lines is one name
        ts = sorted(ts, key=lambda t: t.yc)
        groups: list[list[Tok]] = []
        for t in ts:
            if groups and t.box[1] - max(x.box[3] for x in groups[-1]) <= 0.6 * lh and why != "rotated":
                groups[-1].append(t)
            else:
                groups.append([t])
        if why == "outer" and not _restarts_back(ctx, groups):
            continue
        for g in groups:
            box = (min(t.box[0] for t in g), min(t.box[1] for t in g), max(t.box[2] for t in g), max(t.box[3] for t in g))
            if why == "rotated":
                name = clean_name(" ".join(t.text for t in g))
            else:
                lines: dict[int, list[Tok]] = {}
                for t in g:
                    lines.setdefault(round(t.yc / lh), []).append(t)
                name = " ".join(_text(ctx, v, (min(t.box[0] for t in v), min(t.box[1] for t in v),
                                                max(t.box[2] for t in v), max(t.box[3] for t in v)))
                                for _, v in sorted(lines.items()))
            if not name:
                continue
            titles.append(Title(name, (box[1] + box[3]) / 2, box, "column"))
    if not titles:
        return []
    titles.sort(key=lambda t: t.y)
    _column_bounds(ctx, titles)
    return titles


def _restarts_back(ctx: PageContext, groups: list[list[Tok]]) -> bool:
    """An outer column of names is a group column only if the rows start again
    (row numbers back to 1, or the header repeated) where the names change."""
    if len(groups) < 2:
        return False
    starts = [ln.yc for ln in ctx.price_lines if ctx.rows.get(id(ln)) == 1]
    starts += [h.yc for h in ctx.header_lines]
    if not starts:
        return False
    ys = sorted((min(t.yc for t in g) + max(t.yc for t in g)) / 2 for g in groups)
    between = sum(1 for a, b in zip(ys, ys[1:]) if any(a < s < b for s in starts))
    return between >= 0.7 * (len(ys) - 1)


def _column_bounds(ctx: PageContext, titles: list[Title]) -> None:
    """Where each column group starts and ends: at a row where numbering starts
    again, a repeated header or a ruling line across the name's column - the
    one that leaves the name centred in its merged cell."""
    lh = ctx.doc.line_h
    first_y = ctx.y0 - 0.5 * lh
    last_y = ctx.y1 + 0.5 * lh
    base = {round(ln.y0 - 0.3 * lh, 1) for ln in ctx.price_lines if ctx.rows.get(id(ln)) == 1}
    base |= {round(h.y0 - 0.3 * lh, 1) for h in ctx.header_lines if first_y < h.yc < last_y}

    def marks(t: Title) -> list[float]:
        rules = {round(y, 1) for y, a, b in ctx.graphics.hlines
                 if a <= t.box[0] + 1 and b >= t.box[2] - 1 and first_y - lh < y < last_y + lh}
        return sorted(base | rules)

    def centre(t: Title) -> float:
        return (t.box[1] + t.box[3]) / 2

    def end_of(k: int, start: float) -> float:
        t = titles[k]
        c = centre(t)
        guess = 2 * c - start
        if k + 1 == len(titles):
            # the last group ends at a line across its column below the name, or with the rows
            below = [m for m in marks(t) if t.box[3] - 0.3 * lh < m < last_y - 0.5 * lh
                     and any(m < ln.yc < last_y for ln in ctx.price_lines)]
            if below and guess < last_y - 0.5 * lh:
                return min(below, key=lambda m: abs(m - guess))
            return last_y
        c_next = centre(titles[k + 1])
        between = [m for m in marks(t) if c < m < c_next]
        if between:
            return min(between, key=lambda m: abs(m - guess))
        return guess if t.box[3] < guess < titles[k + 1].box[1] else (t.box[3] + titles[k + 1].box[1]) / 2

    # the first group starts where the rows start, or lower at a mark when the rows
    # above it still belong to the group of the page before
    t0 = titles[0]
    starts = [first_y] + [m for m in marks(t0) if first_y + 0.5 * lh < m < t0.box[1]
                          and any(first_y < ln.yc < m for ln in ctx.price_lines)]
    start = min(starts, key=lambda s: (abs(centre(t0) - (s + end_of(0, s)) / 2) - (0.5 * lh if s == first_y else 0)))
    for k, t in enumerate(titles):
        t.y = start
        t.end = end_of(k, start)
        start = t.end


# ============================================================= whole list ==

@dataclass
class PageIn:
    """One page as the group finder gets it."""
    doc: PageDoc | None
    selected: set[int]
    graphics: Graphics | None = None
    namer: Callable[[Box], str] | None = None    # exact text inside a box (PDF text layer)


@dataclass
class PageOut:
    titles: list[Title] = field(default_factory=list)
    cands: list[Cand] = field(default_factory=list)
    carry_in: str = ""                       # the group going on from the page before
    ctx: PageContext | None = None
    tail_y: float | None = None              # rows below this belong to a group named on the next page
    tail: str = ""

    def group_of(self, box: Box) -> str:
        """The group of a price at `box` on this page."""
        yc, xc = (box[1] + box[3]) / 2, (box[0] + box[2]) / 2
        best = self.carry_in
        for t in self.titles:
            if t.kind == "column":
                if t.y - 1 <= yc <= (t.end if t.end is not None else 1e12) + 1:
                    return t.name
                continue
            if t.y <= yc and t.x0 <= xc <= t.x1:
                best = t.name
        if self.tail_y is not None and yc > self.tail_y:
            return self.tail
        if any(t.kind == "column" and t.end is not None and yc > t.end for t in self.titles) \
                and not any(t.kind == "bar" and t.y <= yc for t in self.titles):
            return self.tail or best
        return best

    def last(self) -> str:
        if self.tail_y is not None and self.tail:
            return self.tail
        if not self.titles:
            return self.carry_in
        t = max(self.titles, key=lambda t: (t.end if t.kind == "column" and t.end is not None else t.y))
        return t.name


def find(pages: list[PageIn], model: GroupModel, threshold: float = 0.5) -> list[PageOut]:
    """Group titles of every page of one list, and the group each page starts in."""
    outs: list[PageOut] = []
    ctxs: list[PageContext | None] = []
    prev_row = None
    for p in pages:
        if p.doc is None:
            outs.append(PageOut())
            ctxs.append(None)
            continue
        ctx = context(p.doc, p.selected, p.graphics, p.namer)
        ctx.prev_row = prev_row
        prev_row = ctx.last_row() or prev_row
        ctxs.append(ctx)
        outs.append(PageOut(cands=candidates(ctx), ctx=ctx))
    # text written the same on most pages (a banner, the company name) is no group
    seen: dict[str, set[int]] = {}
    for k, o in enumerate(outs):
        for c in o.cands:
            seen.setdefault(name_key(c.name), set()).add(k)
    n_pages = sum(1 for o in outs if o.ctx is not None and o.ctx.price_lines)
    for o in outs:
        for c in o.cands:
            if n_pages >= 3 and len(seen.get(name_key(c.name), ())) >= max(3, 0.6 * n_pages):
                c.feats.append("repeat")
            c.prob = model.prob(c.feats)
    for k, (o, p) in enumerate(zip(outs, pages)):
        ctx = ctxs[k]
        if ctx is None or not ctx.price_lines:
            continue
        cols = column_titles(ctx)
        bars = [Title(c.name, c.line.y1 - 0.1 * ctx.doc.line_h, c.box, "bar", score=c.prob)
                for c in o.cands if c.prob >= threshold]
        if cols:
            # names in a group column: bars inside that column's rows are sub-lines of it
            bars = [b for b in bars if not any(t.y <= b.box[1] <= (t.end or 1e12) and same_name(t.name, b.name)
                                              for t in cols)]
        _split_side_by_side(bars, ctx)
        o.titles = sorted(bars + cols, key=lambda t: t.y)
    _drop_empty(outs)
    # a name cut by the page break: the end of it at the foot of one page, the whole
    # (or the rest) at the top of the next - one group, with the fuller name
    live = [o for o in outs if o.ctx is not None and o.ctx.price_lines]
    for o, q in zip(live, live[1:]):
        a_ = max((t for t in o.titles if t.kind == "column" and t.end is not None), key=lambda t: t.y, default=None)
        b_ = min((t for t in q.titles if t.kind == "column"), key=lambda t: t.y, default=None)
        if a_ is None or b_ is None or a_.end < o.ctx.y1 or b_.y > q.ctx.y0:
            continue
        ka, kb = name_key(a_.name).replace(" ", ""), name_key(b_.name).replace(" ", "")
        if ka and kb and (ka in kb or kb in ka):
            a_.name = b_.name = max(a_.name, b_.name, key=len)
    # rows under the last column group of a page start a new merged cell (a line
    # across the column ends the group above them); its name is written further
    # on, on the page where most of that cell is - pages in between have no name
    pending: list[PageOut] = []
    for o in outs:
        ctx = o.ctx
        if ctx is None or not ctx.price_lines:
            continue
        cols = sorted((t for t in o.titles if t.kind == "column" and t.end is not None), key=lambda t: t.y)
        if not cols:
            if pending and not o.titles:
                o.tail_y = -1e12            # the whole page is inside the unnamed cell
                pending.append(o)
            else:
                pending = []
            continue
        if pending and cols[0].y <= ctx.y0:
            for q in pending:
                q.tail = cols[0].name
        pending = []
        end = cols[-1].end
        if any(ln.yc > end + 1 for ln in ctx.price_lines):
            o.tail_y = end
            pending = [o]
    carry = ""
    for o in outs:
        o.carry_in = carry
        carry = o.last() if (o.ctx is not None and o.ctx.price_lines) or o.titles else carry
    return outs


def _split_side_by_side(bars: list[Title], ctx: PageContext) -> None:
    """Two titles on the same height head two tables printed side by side."""
    lh = ctx.doc.line_h
    for b in bars:
        same = [o for o in bars if abs(o.y - b.y) <= 1.5 * lh]
        if len(same) < 2:
            continue
        same.sort(key=lambda t: (t.box[0] + t.box[2]) / 2)
        for k, t in enumerate(same):
            left = (same[k - 1].box[2] + t.box[0]) / 2 if k else -1e9
            right = (t.box[2] + same[k + 1].box[0]) / 2 if k + 1 < len(same) else 1e9
            t.x0, t.x1 = left, right


def _drop_empty(outs: list[PageOut]) -> None:
    """A title with no price under it before the next title (a banner, the upper
    line of a two-line title) does not start a group."""
    flat = [(k, t) for k, o in enumerate(outs) for t in sorted(o.titles, key=lambda t: t.y)]
    keep: set[int] = set()
    for n, (k, t) in enumerate(flat):
        # the next title further down (a title beside this one heads another table)
        nxt = next(((j, u) for j, u in flat[n + 1:]
                    if j > k or u.y > t.y + 1.5 * (outs[k].ctx.doc.line_h if outs[k].ctx else 10)), None)
        has = False
        for j in range(k, len(outs)):
            ctx = outs[j].ctx
            if nxt is not None and j > nxt[0]:
                break
            for ln in (ctx.price_lines if ctx is not None else []):
                y = ln.yc
                if j == k and y <= t.y:
                    continue
                if j == k and t.kind == "column" and t.end is not None and y > t.end:
                    break
                if nxt is not None and j == nxt[0] and y > nxt[1].y:
                    break
                has = True
                break
            if has:
                break
        if has:
            keep.add(id(t))
    for o in outs:
        o.titles = [t for t in o.titles if id(t) in keep]


# ================================================================ teaching ==

def labels(out: PageOut, names_below: Callable[[Line], str | None]) -> list[tuple[list[str], int]]:
    """Training samples of one page: `names_below(line)` gives the checked group of
    the first price under a line (None = unknown). A line is a group title when
    its text names that group (on a photo: reads much like it)."""
    ocr = out.ctx is not None and out.ctx.doc.source == "ocr"
    samples = []
    for c in out.cands:
        truth = names_below(c.line)
        if truth is None:
            continue
        if ocr:
            sim = likeness(c.name, truth) if truth else 0.0
            if 0.3 <= sim < 0.55:
                continue                   # too garbled to say either way
            samples.append((list(c.feats), 1 if sim >= 0.55 else 0))
        else:
            samples.append((list(c.feats), 1 if truth and same_name(c.name, truth) else 0))
    return samples
