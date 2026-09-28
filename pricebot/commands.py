"""Understanding "10 درصد افزایش" style instructions and computing new prices."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

from .models import PriceItem
from .numfmt import fmt_plain, to_latin_digits


@dataclass
class Rule:
    op: str          # percent | add | set | multiply
    value: Decimal

    def apply(self, x: Decimal) -> Decimal:
        if self.op == "percent":
            return x * (1 + self.value / 100)
        if self.op == "add":
            return x + self.value
        if self.op == "set":
            return self.value
        if self.op == "multiply":
            return x * self.value
        raise ValueError(self.op)

    def describe(self) -> str:
        v = self.value
        if self.op == "percent":
            return f"{fmt_plain(abs(v))}٪ {'افزایش' if v >= 0 else 'کاهش'}"
        if self.op == "add":
            return f"{'افزودن' if v >= 0 else 'کم کردن'} {fmt_plain(abs(v))}"
        if self.op == "set":
            return f"تنظیم روی {fmt_plain(v)}"
        return f"ضرب در {fmt_plain(v)}"


@dataclass
class Plan:
    rules: list[tuple[set[str] | None, Rule]] = field(default_factory=list)  # None scope = all
    round_step: Decimal = Decimal(0)     # 0 = none, -1 = auto (keep the original precision)
    round_mode: str = "nearest"
    summary: str = ""
    command: str = ""

    def rule_for(self, item_id: str) -> Rule | None:
        found = None
        for scope, rule in self.rules:
            if scope is None or item_id in scope:
                found = rule
        return found

    def with_rounding(self, step: Decimal, mode: str = "nearest") -> "Plan":
        return Plan(list(self.rules), step, mode, self.summary, self.command)


def _round(value: Decimal, step: Decimal, mode: str) -> Decimal:
    if step <= 0:
        return value
    rounding = {"nearest": ROUND_HALF_UP, "up": ROUND_CEILING, "down": ROUND_FLOOR}[mode]
    return (value / step).quantize(Decimal(1), rounding=rounding) * step


def _auto_step(original: Decimal, decimals: int = 0) -> Decimal:
    """Round to the precision the original price was written with (max 1000)."""
    if decimals or original != original.to_integral_value():
        return Decimal(1).scaleb(-decimals)
    n = int(original)
    zeros = 0
    while n and n % 10 == 0 and zeros < 3:
        n //= 10
        zeros += 1
    return Decimal(10) ** zeros if zeros else Decimal(1)


def compute(plan: Plan, items: list[PriceItem], prefix: str = "") -> dict[str, Decimal]:
    """New value per item id (only items whose price actually changes).
    `prefix` namespaces ids when several files share one plan."""
    out: dict[str, Decimal] = {}
    for it in items:
        rule = plan.rule_for(prefix + it.id)
        if rule is None:
            continue
        new = rule.apply(it.value)
        step = _auto_step(it.value, it.fmt.decimals) if plan.round_step < 0 else plan.round_step
        new = _round(new, step, plan.round_mode)
        quantum = Decimal(1).scaleb(-it.fmt.decimals)
        new = new.quantize(quantum, rounding=ROUND_HALF_UP)
        if new < 0:
            new = Decimal(0)
        if new != it.value:
            out[it.id] = new
    return out


# ------------------------------------------------------------ local parser --

_NORMALIZE = str.maketrans({"ي": "ی", "ك": "ک", "٪": "%", "ة": "ه", "‌": " ", "،": ",", "٫": "."})
_UP = re.compile(r"افزایش|اضافه|زیاد|بالا|بیشتر|گرون|گران|بیفزا|افزودن|increase|raise|\bup\b|\+")
_DOWN = re.compile(r"کاهش|کم\s*کن|کم\b|کمتر|پایین|تخفیف|ارزون|ارزان|کسر|decrease|discount|\bdown\b|reduce|(?<![\d.])-")
_PERCENT = re.compile(r"([+-]?\s*\d+(?:[.,/]\d+)?)\s*(?:%|درصد|percent)|(?:%|درصد)\s*([+-]?\d+(?:[.,/]\d+)?)")
_ROUND = re.compile(r"رند|گرد|round")
_COMPLEX = re.compile(
    r"ردیف|فقط|به\s*جز|بجز|غیر\s*از|بقیه|باقی|گروه|صفحه|ستون|اگر|بیشتر\s*از|کمتر\s*از|بالای|زیر|"
    r"تومان|تومن|ریال|هزار(?!\s*تایی)|میلیون|ملیون|تا\s*\d|\d\s*تا\s*\d|کالا|قطعه|محصول|برای|روی\s+\S+ها|"
    r"only|except|row|page")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", to_latin_digits(text).translate(_NORMALIZE)).strip().lower()


@dataclass
class LocalParse:
    plan: Plan | None = None
    ask_direction: Decimal | None = None   # percent found, direction unknown
    needs_ai: bool = False


def _parse_rounding(text: str) -> tuple[Decimal, str, str]:
    """Returns (step, mode, text-without-rounding-clause)."""
    m = _ROUND.search(text)
    if not m:
        return Decimal(0), "nearest", text
    clause = text[m.start():]
    rest = text[:m.start()]
    mode = "nearest"
    if re.search(r"بالا|up", clause):
        mode = "up"
    elif re.search(r"پایین|down", clause):
        mode = "down"
    num = re.search(r"(\d+(?:\.\d+)?)", clause)
    mult = Decimal(1)
    if "میلیون" in clause or "ملیون" in clause:
        mult = Decimal(1_000_000)
    elif "هزار" in clause:
        mult = Decimal(1000)
    elif "صد" in clause and not num:
        mult = Decimal(100)
    if num:
        step = Decimal(num.group(1)) * mult
    elif mult > 1:
        step = mult
    else:
        step = Decimal(-1)   # "رند کن" without a unit: keep the original precision
    return step, mode, rest


def parse_local(command: str) -> LocalParse:
    text = normalize(command)
    step, mode, core = _parse_rounding(text)
    percents = [m.group(1) or m.group(2) for m in _PERCENT.finditer(core)]
    if len(percents) != 1:
        return LocalParse(needs_ai=True)
    generic = re.sub(r"(برای|روی)\s+(همه|کل|تمام)\S*(\s+(قیمت|کالا|محصول|قطعه)\S*)?(\s+ها)?", " ", core)
    generic = re.sub(r"(همه|کل|تمام)\s*(ی)?\s+(قیمت|کالا|محصول|قطعه)\S*(\s+ها)?", " ", generic)
    if _COMPLEX.search(generic):
        return LocalParse(needs_ai=True)
    raw = percents[0].replace(" ", "").replace(",", ".").replace("/", ".")
    value = Decimal(raw)
    rest = _PERCENT.sub(" ", core)
    up = bool(_UP.search(rest)) or raw.startswith("+")
    down = bool(_DOWN.search(rest)) or raw.startswith("-")
    value = abs(value)
    if up and down:
        return LocalParse(needs_ai=True)
    if not up and not down:
        return LocalParse(ask_direction=value)
    signed = value if up else -value
    rule = Rule("percent", signed)
    plan = Plan([(None, rule)], step, mode, rule.describe(), command)
    return LocalParse(plan=plan)


def plan_from_ai(data: dict, command: str, valid_ids: set[str]) -> Plan:
    rules: list[tuple[set[str] | None, Rule]] = []
    for r in data.get("rules", []):
        op = r.get("op")
        if op not in {"percent", "add", "set", "multiply"}:
            continue
        value = Decimal(str(r.get("value", 0)))
        if r.get("scope") == "all":
            rules.append((None, Rule(op, value)))
        else:
            ids = {i for i in r.get("ids", []) if i in valid_ids}
            if ids:
                rules.append((ids, Rule(op, value)))
    rounding = data.get("rounding") or {}
    step = Decimal(str(rounding.get("step") or 0))
    mode = rounding.get("mode") or "nearest"
    return Plan(rules, step, mode if mode in {"nearest", "up", "down"} else "nearest",
                data.get("summary", ""), command)
