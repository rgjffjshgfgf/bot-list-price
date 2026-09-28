"""Parsing and formatting of numbers exactly the way they appear in a list.

A price such as ``۳/۶۹۶/۰۰۰`` or ``58,000,000`` is parsed into its value and a
``NumberFormat`` (digit script, grouping separator, decimals) so the new price
can be written back in precisely the same style.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

LATIN = "0123456789"
PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
ARABIC = "٠١٢٣٤٥٦٧٨٩"
DIGIT_CHARS = frozenset(LATIN + PERSIAN + ARABIC)

# Characters that may appear between the digit groups of a number.
SPACE_CHARS = frozenset("     ")
SEP_CHARS = frozenset(",٬،/.'’٫") | SPACE_CHARS

_TO_LATIN = {ord(c): str(i) for i, c in enumerate(PERSIAN)}
_TO_LATIN.update({ord(c): str(i) for i, c in enumerate(ARABIC)})
_FROM_LATIN = {
    "latin": None,
    "persian": {ord(str(i)): c for i, c in enumerate(PERSIAN)},
    "arabic": {ord(str(i)): c for i, c in enumerate(ARABIC)},
}


def to_latin_digits(text: str) -> str:
    return text.translate(_TO_LATIN)


def to_script(text: str, script: str) -> str:
    table = _FROM_LATIN.get(script)
    return text.translate(table) if table else text


def digit_script(text: str) -> str:
    counts = {"latin": 0, "persian": 0, "arabic": 0}
    for ch in text:
        if ch in LATIN:
            counts["latin"] += 1
        elif ch in PERSIAN:
            counts["persian"] += 1
        elif ch in ARABIC:
            counts["arabic"] += 1
    return max(counts, key=counts.get)


@dataclass(frozen=True)
class NumberFormat:
    script: str = "latin"      # latin | persian | arabic
    group_sep: str = ""        # "" means the number is written without grouping
    decimal_sep: str = ""
    decimals: int = 0

    def format(self, value: Decimal) -> str:
        quantum = Decimal(1).scaleb(-self.decimals)
        v = value.quantize(quantum, rounding=ROUND_HALF_UP)
        sign = "-" if v < 0 else ""
        int_part, _, frac = f"{abs(v):f}".partition(".")
        if self.group_sep:
            int_part = _group(int_part, self.group_sep)
        out = int_part
        if self.decimals:
            out += (self.decimal_sep or ".") + frac.ljust(self.decimals, "0")[: self.decimals]
        return sign + to_script(out, self.script)


def _group(digits: str, sep: str) -> str:
    head = len(digits) % 3 or 3
    parts = [digits[:head]] + [digits[i:i + 3] for i in range(head, len(digits), 3)]
    return sep.join(parts)


@dataclass(frozen=True)
class ParsedNumber:
    value: Decimal
    fmt: NumberFormat
    text: str          # the original text, trimmed


def parse_number(text: str) -> ParsedNumber | None:
    """Parse a number written with any digit script and grouping style.

    Returns None for things that are not a single clean number, e.g. dates
    like ``1405/06/23`` or ``05.06.21``.
    """
    s = text.strip()
    s = "".join(ch for ch in s if ch in DIGIT_CHARS or ch in SEP_CHARS)
    s = s.strip("".join(SEP_CHARS))
    if not s or not any(ch in DIGIT_CHARS for ch in s):
        return None
    script = digit_script(s)
    lat = to_latin_digits(s)
    # A space glued to a real separator ("3 /432/000") is layout noise.
    lat = re.sub(r"\s*([,٬،/.'’٫])\s*", r"\1", lat)
    lat = re.sub(r"[    ]", " ", lat)
    lat = re.sub(r" +", " ", lat)
    groups = re.findall(r"\d+", lat)
    seps = re.findall(r"[^\d]+", lat)
    if len(seps) != len(groups) - 1 or any(len(x) != 1 for x in seps):
        return None

    if not seps:
        return ParsedNumber(Decimal(groups[0]), NumberFormat(script=script), text.strip())

    distinct = set(seps)
    first_ok = 1 <= len(groups[0]) <= 3
    if len(distinct) == 1:
        sep = seps[0]
        if first_ok and all(len(g) == 3 for g in groups[1:]):
            value = Decimal("".join(groups))
            return ParsedNumber(value, NumberFormat(script=script, group_sep=sep), text.strip())
        if len(seps) == 1 and sep != " " and 1 <= len(groups[1]) <= 4:
            value = Decimal(f"{groups[0]}.{groups[1]}")
            fmt = NumberFormat(script=script, decimal_sep=sep, decimals=len(groups[1]))
            return ParsedNumber(value, fmt, text.strip())
        return None

    if len(distinct) == 2 and seps[-1] not in seps[:-1] and len(set(seps[:-1])) == 1:
        group_sep, dec_sep = seps[0], seps[-1]
        if dec_sep == " ":
            return None
        if first_ok and all(len(g) == 3 for g in groups[1:-1]) and 1 <= len(groups[-1]) <= 4:
            value = Decimal("".join(groups[:-1]) + "." + groups[-1])
            fmt = NumberFormat(script=script, group_sep=group_sep, decimal_sep=dec_sep,
                               decimals=len(groups[-1]))
            return ParsedNumber(value, fmt, text.strip())
    return None


def looks_like_price(p: ParsedNumber) -> bool:
    """Heuristic used when no AI is available: grouped numbers >= 1000."""
    return bool(p.fmt.group_sep) and p.value >= 1000


def fmt_plain(value: Decimal) -> str:
    """Human readable Latin formatting for chat messages."""
    if value == value.to_integral_value():
        return f"{int(value):,}"
    return f"{value:,f}".rstrip("0").rstrip(".")
