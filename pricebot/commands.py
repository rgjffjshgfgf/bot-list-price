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


# ------------------------------------------------- commands by list group --

_REST = re.compile(r"(?<!\S)(بقیه|باقی|سایر|مابقی|الباقی|دیگر)\S*")
_GROUP_STOP = {"گروه", "خودرو", "محصولات", "قطعات", "لوازم", "و", "یا", "ها", "های", "سری", "مدل", "انواع", "کلیه"}
_OTHER = re.compile(r"ردیف|صفحه|ستون|تومان|تومن|ریال|هزار(?!\s*تایی)|میلیون|ملیون|اگر|بیشتر\s*از|کمتر\s*از|به\s*جز|بجز|"
                    r"غیر\s*از|بالای|زیر|only|except|row|page")


def _words(text: str) -> list[tuple[str, int]]:
    """(word, position) with plural endings dropped («پرایدها» -> «پراید»)."""
    out = []
    for m in re.finditer(r"[^\s،,؛;:.!?()«»\-/]+", text):
        w = re.sub(r"(های|ها|هایی)$", "", m.group(0)) if len(m.group(0)) > 4 else m.group(0)
        out.append((w, m.start()))
    return out


_FA_NUM = re.compile(r"(?<=[\u0621-\u064a\u066e-\u06d3])(?=\d)|(?<=\d)(?=[\u0621-\u064a\u066e-\u06d3])")


def parse_groups(command: str, groups: list[str]) -> list[tuple[list[str] | None, Rule]] | None:
    """A price change by the list's own groups, understood without Gemini:
    «گروه پژو ۱۰ درصد افزایش، پراید ۵ درصد، بقیه ۳ درصد». Returns rules as
    (group names, rule) - None names = every other price - or None when the
    command is not (only) that."""
    if not groups:
        return None
    text = normalize(command)
    _, _, core = _parse_rounding(text)
    if _OTHER.search(core):
        return None
    # «پژو405» is the group «پژو 405»: a Persian word and a number apart, as in the names
    core = _FA_NUM.sub(" ", core)
    keys = {g: [w for w in _FA_NUM.sub(" ", normalize(g)).split() if w not in _GROUP_STOP] for g in groups}
    events: list[tuple[int, int, str, object]] = []          # (start, end, kind, value)
    for m in _PERCENT.finditer(core):
        events.append((m.start(), m.end(), "pct", (m.group(1) or m.group(2)).replace(" ", "")))
    for m in _REST.finditer(core):
        events.append((m.start(), m.end(), "rest", None))
    # a group by its whole name («پژو 405»), else by a word of it («پژو» = every Peugeot group)
    taken: set[int] = set()
    for g, ks in sorted(keys.items(), key=lambda kv: -len(" ".join(kv[1]))):
        full = " ".join(ks)
        if len(full) < 2:
            continue
        for m in re.finditer(r"(?<!\S)" + re.escape(full) + r"(?:ها|های)?(?!\S)", core):
            if not taken & set(range(m.start(), m.end())):
                events.append((m.start(), m.end(), "group", {g}))
                taken.update(range(m.start(), m.end()))
    for w, pos in _words(core):
        if pos in taken or w in _GROUP_STOP or len(w) < 2 or w.isdigit():
            continue
        hit = {g for g, ks in keys.items() if w in ks}
        if hit:
            events.append((pos, pos + len(w), "group", hit))
    events.sort()
    if not any(e[2] == "pct" for e in events) or not any(e[2] == "group" for e in events):
        return None

    parts: list[tuple[set[str] | None, str, Rule | None]] = []
    names: set[str] = set()
    rest = False
    used: set[int] = set()
    clause_start = 0
    for k, (start, stop, kind, val) in enumerate(events):
        if k in used:
            continue
        if kind == "group":
            names |= val
            continue
        if kind == "rest":
            rest = True
            continue
        if not names and not rest:
            # «۱۰ درصد افزایش برای پژو»: the groups named right after it
            j = k + 1
            while j < len(events) and events[j][2] == "group":
                names |= events[j][3]
                used.add(j)
                j += 1
            if not names:
                return None
        nxt = next((e[0] for j, e in enumerate(events) if j > k and j not in used), len(core))
        parts.append((set(names) if names else None, val, _direction(core[clause_start:nxt], val)))
        clause_start = nxt
        names, rest = set(), False
    if names:                          # groups named with no percentage of their own
        return None
    if any(r is None for _, _, r in parts):
        # a direction said once for all («پژو ۱۰ درصد و پراید ۵ درصد افزایش»)
        up = _sign(core)
        if up is None:
            return None
        parts = [(g, v, r or Rule("percent", abs(Decimal(v.replace(",", ".").replace("/", "."))) * (1 if up else -1)))
                 for g, v, r in parts]
    if all(g is None for g, _, _ in parts):
        return None
    rules = [(sorted(g) if g is not None else None, r) for g, _, r in parts]
    # the general rule first: a group's own rule wins over «بقیه»
    rules.sort(key=lambda gr: gr[0] is not None)
    return rules


def _sign(text: str) -> bool | None:
    up, down = bool(_UP.search(text)), bool(_DOWN.search(text))
    return None if up == down else up


def _direction(clause: str, raw: str) -> Rule | None:
    value = Decimal(raw.replace(",", ".").replace("/", "."))
    if raw.startswith("-") or raw.startswith("+"):
        return Rule("percent", value)
    s = _sign(_PERCENT.sub(" ", clause))
    if s is None:
        return None
    return Rule("percent", abs(value) if s else -abs(value))


def plan_for_groups(command: str, rules: list[tuple[list[str] | None, Rule]],
                    ids_by_group: dict[str, set[str]]) -> Plan:
    """The plan of parse_groups() with the price ids of each group."""
    step, mode, _ = _parse_rounding(normalize(command))
    out: list[tuple[set[str] | None, Rule]] = []
    parts = []
    for groups, rule in rules:
        if groups is None:
            out.append((None, rule))
            parts.append(f"بقیه {rule.describe()}")
        else:
            out.append(({i for g in groups for i in ids_by_group.get(g, set())}, rule))
            parts.append(f"{'، '.join(groups)}: {rule.describe()}")
    return Plan(out, step, mode, " — ".join(parts), command)


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
