"""What the bot process needs from /shot without loading the image code: the numbers people type or put in
captions, turned into the text Telegram would show, and the error the user sees."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal


class ShotError(Exception):
    """A problem worth telling the user about (the message is meant for them)."""


@dataclass
class ShotOut:
    """One screenshot read in the engine process, as the bot process gets it back."""
    current: list[str]                        # each view count top to bottom as read ("" = unreadable)
    warnings: list[str] = field(default_factory=list)
    photo: bytes | None = None                # the finished screenshot (the caption had its numbers)
    notes: list[str] = field(default_factory=list)
    question: str | None = None               # the caption's numbers do not fit this screenshot
    preview: bytes | None = None              # the screenshot with the view counts boxed (no numbers yet)
    blob: bytes = b""                         # what the engine needs to build it later (opaque here)

    @property
    def count(self) -> int:
        return len(self.current)


def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


_FA = {**{ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}, **{ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")},
       ord("٫"): ".", ord("٬"): "", ord(","): "", ord("،"): ""}


def format_views(n: int) -> str:
    """The way Telegram prints a view count: 999, 1K, 1.2K, 27K, 24.6K (one decimal, cut not rounded)."""
    if n < 0:
        raise ShotError("عدد بازدید نمی‌تواند منفی باشد.")
    if n < 1000:
        return str(n)
    for div, suffix in ((1_000_000, "M"), (1000, "K")):
        if n >= div:
            tenths = n * 10 // div
            if tenths >= 10000:      # 1000K -> 1M
                continue
            whole, frac = divmod(tenths, 10)
            return f"{whole}{suffix}" if frac == 0 else f"{whole}.{frac}{suffix}"
    return str(n)


def parse_value(text: str, plain_thousands: bool = True) -> str:
    """The text to draw for what the user typed: '36.2' / '36.2k' / '۳۶٫۲ هزار' -> '36.2K', '2500' -> '2.5K',
    '0.5' -> '500'. A plain number under 1000 counts in thousands (views are almost always in K) unless
    `plain_thousands` is off; Telegram's own rounding (one decimal, cut) is applied."""
    t = (text or "").strip().translate(_FA).replace(" ", "").upper()
    t = t.replace("هزار", "K").replace("میلیون", "M")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KM]?)", t)
    if not m:
        raise ShotError("عدد را درست نفهمیدم. مثلاً بنویس: 36.2 (یعنی 36.2K) یا 5 یا 2500")
    num, suffix = Decimal(m.group(1)), m.group(2)
    if not suffix and plain_thousands and num < 1000:
        suffix = "K"
    return format_views(int(num * {"": 1, "K": 1000, "M": 1_000_000}[suffix]))



# ---------------------------------------------------------- what was typed --

_SHOT_DIGITS = str.maketrans({**{c: str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")},
                               **{c: str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}, "٫": "."})
_SHOT_TOKEN = re.compile(
    r"(?P<row>(?:ردیف|شماره|مورد|پیام|row)\s*(?P<ri>\d+))"
    r"|(?P<word>بالایی|بالا|اولی|پایینی|پایین|آخری|وسطی|وسط|دومی|سومی|همه|top|bottom|middle|all)"
    r"|(?P<num>\d+(?:\.\d+)?)\s*(?P<suf>k|هزار|m|میلیون)?")
_SHOT_WORD_ROW = {"بالایی": 0, "بالا": 0, "اولی": 0, "top": 0, "دومی": 1, "سومی": 2,
                  "پایینی": -1, "پایین": -1, "آخری": -1, "bottom": -1, "وسطی": -2, "وسط": -2, "middle": -2,
                  "همه": -3, "all": -3}
# what may stand around the numbers in a caption that is only numbers ("بازدید ۳۴.۸"، "۶.۲ و ۵۳.۷")
_SHOT_FILLER = re.compile(r"\b(?:بازدید|بازدیدها|ویو|views?|بشود|بشوند|بشه|بشن|شود|کن|بکن|و|and)\b|[\s,،;:|/\\\-–+=.()*×]+")
_ALL = -3


def read_values(text: str, n: int, strict: bool = False) -> tuple[dict[int, str] | None, str | None]:
    """The new number of each view count (0-based, top to bottom) from a message or a caption.

    «۳۴.۸» -> bottom 34.8K; «۶.۲ ۵۳.۷» -> top 6.2K, bottom 53.7K (fewer numbers than view counts fill from the
    bottom); «بالا ۱۰ پایین ۵»; «ردیف ۲ بشود ۱۰»; «همه ۵». A plain number is in thousands (see shot.parse_value).
    `strict` (captions): anything besides numbers and these words means the caption is not an instruction.
    Returns (values, None), (None, a question for the user) or (None, None) when there are no numbers."""
    t = (text or "").translate(_SHOT_DIGITS).lower()
    t = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", t)          # 2,500 -> 2500
    named: dict[int, str] = {}
    loose: list[str] = []
    slot: int | None = None
    problem: str | None = None
    rest_text = []
    last = 0
    for m in _SHOT_TOKEN.finditer(t):
        rest_text.append(t[last:m.start()])
        last = m.end()
        if m.group("row"):
            slot = int(m.group("ri")) - 1
            if not 0 <= slot < n:
                problem = problem or f"ردیف {_fa(slot + 1)} وجود ندارد؛ شماره‌ها از ۱ تا {_fa(n)} است."
        elif m.group("word"):
            slot = _SHOT_WORD_ROW[m.group("word")]
            slot = n - 1 if slot == -1 else (1 if n == 3 else None) if slot == -2 else slot
            if slot is None or slot >= n:
                problem = problem or f"«{m.group('word')}» را نفهمیدم؛ در این عکس {_fa(n)} بازدید هست."
        else:
            suf = m.group("suf") or ""
            value = m.group("num") + ("K" if suf in ("k", "هزار") else "M" if suf in ("m", "میلیون") else "")
            if slot is None:
                loose.append(value)
            else:
                named[slot] = value
                slot = None
    rest_text.append(t[last:])
    if strict and _SHOT_FILLER.sub("", "".join(rest_text)):
        return None, None
    if not named and not loose:
        return None, None
    if problem:
        return None, problem
    values = {i: named[_ALL] for i in range(n)} if _ALL in named else {}
    named.pop(_ALL, None)
    rest = [i for i in range(n) if i not in named]
    if len(loose) > len(rest):
        return None, (f"در این عکس {_fa(n)} بازدید هست ولی {_fa(len(loose) + len(named))} عدد نوشتی؛ "
                      "عددها را از بالا به پایین بنویس (یک عدد فقط پایینی را عوض می‌کند).")
    values.update(named)
    values.update(zip(rest[len(rest) - len(loose):], loose))
    return values, None
