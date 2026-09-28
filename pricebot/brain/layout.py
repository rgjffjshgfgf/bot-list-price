"""One page as the bot's own AI sees it: every word and every number with its
box, grouped into columns, with column headers, row labels and the currency.

The same structure is built from a PDF's text layer (exact) or from OCR of a
photo/scan, so everything learned on one kind of file helps the other.
"""
from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

from ..numfmt import ParsedNumber, parse_number, to_latin_digits

Box = tuple[float, float, float, float]

_CANON = str.maketrans({"ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا",
                        "آ": "ا", "ؤ": "و", "‌": "", "‍": "", "‎": "", "‏": "",
                        "ـ": "", "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "",
                        "ِ": "", "ّ": "", "ْ": ""})
_ARABIC = re.compile("[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")


def canon(word: str) -> str:
    """Comparable form of a word. Persian words are made order-agnostic because
    some PDFs store them in visual (reversed) order."""
    w = to_latin_digits(unicodedata.normalize("NFKC", word)).translate(_CANON).lower()
    w = re.sub(r"[^\w%$#]", "", w)
    if _ARABIC.search(w):
        w = min(w, w[::-1])
    return w


def _kw(words: list[str]) -> list[str]:
    return [canon(w) for w in words]


# Keyword groups (matched on canonical words; short ones must match exactly).
KEYWORDS: dict[str, list[str]] = {
    "price": _kw(["قیمت", "فی", "مبلغ", "بها", "ریال", "تومان", "تومن", "همکار", "مصرف", "فروش", "عمده",
                  "نماینده", "خرید", "نقد", "نقدی", "چک", "اعتباری", "price", "amount", "cost",
                  "rial", "rials", "toman", "usd", "irr", "دلار", "یورو", "درهم", "total", "جمع", "fee"]),
    "code": _kw(["کد", "شماره", "فنی", "بارکد", "سریال", "پارت", "code", "part", "barcode", "sku", "ref",
                 "serial", "مدل", "model", "id"]),
    "row": _kw(["ردیف", "ردبف", "row", "no", "#", "رديف", "ش"]),
    "qty": _kw(["تعداد", "کارتن", "عدد", "qty", "quantity", "موجودی", "بسته", "وزن", "گرم", "کیلو", "سایز",
                "ابعاد", "size", "weight", "درصد", "تخفیف", "percent", "%", "حجم", "ظرفیت", "واحد", "pcs",
                "min", "حداقل", "سفارش"]),
    "date": _kw(["تاریخ", "date", "سال", "year", "بروزرسانی", "به‌روزرسانی"]),
    "phone": _kw(["تلفن", "تماس", "موبایل", "همراه", "tel", "phone", "فکس", "fax", "واتساپ", "whatsapp"]),
}
_SHORT = 3

CURRENCIES = {canon(k): v for k, v in {
    "ریال": "ریال", "rial": "ریال", "rials": "ریال", "irr": "ریال",
    "تومان": "تومان", "تومن": "تومان", "toman": "تومان",
    "دلار": "دلار", "usd": "دلار", "$": "دلار", "یورو": "یورو", "eur": "یورو", "درهم": "درهم", "aed": "درهم",
}.items()}


def keyword_groups(word: str) -> set[str]:
    c = canon(word)
    if not c:
        return set()
    out = set()
    for group, kws in KEYWORDS.items():
        for k in kws:
            if (len(k) <= _SHORT and c == k) or (len(k) > _SHORT and (k in c or k[::-1] in c)):
                out.add(group)
                break
    return out


@dataclass
class Tok:
    text: str
    box: Box
    num: ParsedNumber | None = None
    attached: bool = False
    conf: float = 100.0

    @property
    def xc(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def yc(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def digits_of(text: str) -> str:
    return "".join(ch for ch in to_latin_digits(text) if ch.isdigit())


@dataclass
class PageDoc:
    width: float
    height: float
    words: list[Tok]              # text (non-number) words
    nums: list[Tok]               # numbers: the things that may be prices
    source: str                   # "pdf" | "ocr"
    line_h: float = 10.0
    col_of: list[int] = field(default_factory=list)
    columns: list[list[int]] = field(default_factory=list)
    _headers: dict[int, list[Tok]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hs = [t.h for t in self.words + self.nums if t.h > 0]
        self.line_h = float(statistics.median(hs)) if hs else 10.0
        self.col_of = _cluster([t.box for t in self.nums])
        cols: dict[int, list[int]] = {}
        for i, c in enumerate(self.col_of):
            cols.setdefault(c, []).append(i)
        self.columns = [sorted(v, key=lambda i: self.nums[i].yc) for _, v in sorted(cols.items())]
        remap = {old: new for new, (old, _) in enumerate(sorted(cols.items()))}
        self.col_of = [remap[c] for c in self.col_of]

    # ---- geometry ---------------------------------------------------------
    def col_span(self, c: int) -> tuple[float, float]:
        idx = self.columns[c]
        return min(self.nums[i].box[0] for i in idx), max(self.nums[i].box[2] for i in idx)

    def norm_span(self, c: int) -> tuple[float, float]:
        x0, x1 = self.col_span(c)
        return x0 / self.width, x1 / self.width

    def col_rank_right(self, c: int) -> int:
        """0 = rightmost column holding at least two numbers."""
        big = [k for k in range(len(self.columns)) if len(self.columns[k]) >= 2]
        if c not in big:
            return 9
        order = sorted(big, key=lambda k: -self.col_span(k)[1])
        return order.index(c)

    def col_rank_left(self, c: int) -> int:
        big = [k for k in range(len(self.columns)) if len(self.columns[k]) >= 2]
        if c not in big:
            return 9
        order = sorted(big, key=lambda k: self.col_span(k)[0])
        return order.index(c)

    # ---- context ----------------------------------------------------------
    def header(self, c: int) -> list[Tok]:
        """Words written above a column (its header), nearest line(s) first."""
        if c in self._headers:
            return self._headers[c]
        x0, x1 = self.col_span(c)
        pad = 0.4 * self.line_h
        top = min(self.nums[i].box[1] for i in self.columns[c])
        above = [w for w in self.words
                 if _overlap(w.box[0], w.box[2], x0 - pad, x1 + pad) >= 0.3 * max(1.0, min(w.box[2] - w.box[0], x1 - x0))
                 and w.box[3] <= top + 0.3 * self.line_h and top - w.box[1] <= 9 * self.line_h]
        above.sort(key=lambda w: -w.box[3])
        out: list[Tok] = []
        if above:
            first = above[0].box[3]
            out = [w for w in above if first - w.box[3] <= 2.6 * self.line_h][:8]
        self._headers[c] = out
        if not out:
            # e.g. a row-number column whose first numbers are single digits: its header
            # sits on the same line as the other columns' headers, far above
            bands = [(min(w.box[1] for w in h), max(w.box[3] for w in h))
                     for k, h in self._headers.items() if k != c and h] or \
                    [(min(w.box[1] for w in h), max(w.box[3] for w in h))
                     for h in (self._near_header(k) for k in range(len(self.columns)) if k != c) if h]
            for y0, y1 in bands:
                if y1 > top:
                    continue
                hit = [w for w in self.words if w.box[1] >= y0 - 0.3 * self.line_h and w.box[3] <= y1 + 0.3 * self.line_h
                       and _overlap(w.box[0], w.box[2], x0 - pad, x1 + pad) > 0]
                if hit:
                    out = hit[:8]
                    break
            self._headers[c] = out
        return out

    def _near_header(self, c: int) -> list[Tok]:
        """Header words close above column c (no fallback, no caching)."""
        x0, x1 = self.col_span(c)
        top = min(self.nums[i].box[1] for i in self.columns[c])
        return [w for w in self.words
                if _overlap(w.box[0], w.box[2], x0, x1) > 0 and w.box[3] <= top + 0.3 * self.line_h
                and top - w.box[1] <= 3 * self.line_h]

    def above(self, i: int) -> list[Tok]:
        """The nearest words straight above one number (a local sub-header)."""
        t = self.nums[i]
        x0, x1 = self.col_span(self.col_of[i])
        cand = [w for w in self.words if w.box[3] <= t.box[1] + 0.2 * self.line_h
                and _overlap(w.box[0], w.box[2], x0, x1) > 0]
        if not cand:
            return []
        y = max(w.box[3] for w in cand)
        # stop at a number sitting between the word and this token
        between = [n for k, n in enumerate(self.nums) if k != i and self.col_of[k] == self.col_of[i]
                   and y <= n.yc <= t.box[1]]
        if between:
            return []
        return [w for w in cand if y - w.box[3] <= 0.6 * self.line_h][:4]

    def row(self, i: int) -> tuple[list[Tok], list[int]]:
        """Words and other numbers on the same row as number i."""
        t = self.nums[i]
        h = max(t.h, 0.6 * self.line_h)

        def same(o: Tok) -> bool:
            return _overlap(o.box[1], o.box[3], t.box[1], t.box[3]) >= 0.45 * min(max(o.h, 1.0), h)
        words = [w for w in self.words if same(w)]
        nums = [k for k, n in enumerate(self.nums) if k != i and same(n)]
        return words, nums

    def rtl(self) -> bool:
        persian = sum(1 for w in self.words if _ARABIC.search(w.text))
        return persian >= 0.3 * max(1, len(self.words))

    def label(self, i: int) -> str:
        """Row number (if any) + the product words on the row, reading order."""
        words, nums = self.row(i)
        rtl = self.rtl()
        words.sort(key=lambda w: -w.xc if rtl else w.xc)
        row_no = ""
        for k in nums:
            n = self.nums[k]
            if n.num and not n.num.fmt.group_sep and n.num.value < 10000 and len(digits_of(n.text)) <= 4:
                c = self.col_of[k]
                if _is_sequence(self, c):
                    row_no = digits_of(n.text)
                    break
        text = " ".join(w.text for w in words[:10])
        return f"{row_no} - {text}" if row_no and text else (text or row_no)

    def currency(self) -> str:
        counts: dict[str, int] = {}
        for w in self.words:
            cur = CURRENCIES.get(canon(w.text))
            if cur:
                counts[cur] = counts.get(cur, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    def signature(self) -> set[str]:
        return {c for c in (canon(w.text) for w in self.words) if len(c) >= 2 and not any(ch.isdigit() for ch in c)}

    def title(self) -> str:
        top = sorted(self.words, key=lambda w: (w.box[1], -w.xc))
        if not top:
            return ""
        y = top[0].box[1]
        line = [w for w in top if w.box[1] - y <= 0.8 * self.line_h]
        line.sort(key=lambda w: -w.xc if self.rtl() else w.xc)
        return " ".join(w.text for w in line)[:60]


def _cluster(boxes: list[Box]) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(order)):
        i = order[a]
        bi = boxes[i]
        for b in range(a + 1, len(order)):
            j = order[b]
            bj = boxes[j]
            if bj[0] > bi[2]:
                break
            if _overlap(bi[0], bi[2], bj[0], bj[2]) >= 0.3 * min(bi[2] - bi[0], bj[2] - bj[0]):
                parent[find(i)] = find(j)
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(boxes))]


def _is_sequence(doc: PageDoc, c: int) -> bool:
    vals = []
    for i in doc.columns[c]:
        n = doc.nums[i].num
        if n is None or n.fmt.group_sep or n.fmt.decimals or n.value != int(n.value):
            return False
        vals.append(int(n.value))
    if len(vals) < 3:
        return False
    steps = [b - a for a, b in zip(vals, vals[1:])]
    return sum(1 for s in steps if s == 1) >= 0.7 * len(steps)


# ================================================================ builders ==

_NUM_EDGE = re.compile(r"^[^\d۰-۹٠-٩]+|[^\d۰-۹٠-٩]+$")
_LETTER = re.compile(r"[^\W\d_]")


def ocr_number(text: str) -> ParsedNumber | None:
    """A word from OCR that is one clean number (stray bars/dots at the edges removed)."""
    core = _NUM_EDGE.sub("", text.strip())
    if not core or _LETTER.search(to_latin_digits(core)):
        return None
    return parse_number(core)


def from_ocr(words: list[tuple[str, Box, float]], width: float, height: float) -> PageDoc:
    text_words, nums = [], []
    for text, box, conf in words:
        num = ocr_number(text)
        if num is not None and len(digits_of(text)) >= 1:
            core = _NUM_EDGE.sub("", text.strip())
            nums.append(Tok(core, box, num, False, conf))
        elif text.strip():
            attached = bool(_LETTER.search(text)) and bool(re.search(r"[\d۰-۹٠-٩]", text))
            text_words.append(Tok(text, box, None, attached, conf))
    return PageDoc(width, height, text_words, nums, "ocr")


def _char_kind(c: str) -> str:
    c = to_latin_digits(c)
    if c.isdigit():
        return "d"
    return "l" if unicodedata.category(c).startswith("L") else "o"


def pdf_words(page) -> list[Tok]:
    """Words from the PDF's characters: split at visible gaps (cell borders often
    have no space character) and where digits meet letters ("10رینگ")."""
    from ..pdftext import _lines, page_chars
    out: list[Tok] = []
    for line in _lines(page_chars(page)):
        run: list = []

        def flush() -> None:
            if run and any(_char_kind(ch.c) == "l" for ch in run):
                text = "".join(ch.c for ch in run)
                if _ARABIC.search(text):
                    text = text[::-1]          # characters are in visual order
                text = unicodedata.normalize("NFKC", text)
                x0 = min(ch.bbox.x0 for ch in run)
                out.append(Tok(text, (x0, min(ch.bbox.y0 for ch in run), max(ch.bbox.x1 for ch in run),
                                      max(ch.bbox.y1 for ch in run))))
            run.clear()

        for ch in line:
            if not ch.c.strip():
                flush()
                continue
            if run:
                prev = run[-1]
                gap = ch.bbox.x0 - prev.bbox.x1
                glued = ({_char_kind(prev.c), _char_kind(ch.c)} == {"d", "l"}
                         and bool(_ARABIC.search(prev.c + ch.c)))      # "10رینگ", not "EF7"
                if gap > 0.25 * ch.size or glued:
                    flush()
            run.append(ch)
        flush()
    return out


def from_pdf(page, cands) -> PageDoc:
    """cands: the pipeline's number tokens (pdftext.TextToken) of this page, in order."""
    nums = [Tok(t.text, tuple(t.bbox), t.parsed, t.attached) for t in cands]
    rect = page.rect * page.derotation_matrix
    return PageDoc(abs(rect.width) or 1.0, abs(rect.height) or 1.0, pdf_words(page), nums, "pdf")


def box_iou(a: Box, b: Box) -> float:
    inter = _overlap(a[0], a[2], b[0], b[2]) * _overlap(a[1], a[3], b[1], b[3])
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_in(a: Box, b: Box, pad: float = 0.0) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad


def magnitude(value) -> int:
    v = abs(float(value))
    return int(math.log10(v)) if v >= 1 else 0
