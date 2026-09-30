"""Google Gemini calls on the free tier.

Two models work together:
  * Gemini Flash      – smarter, small free quota: photos/scans and free-form commands.
  * Gemini Flash-Lite – large free quota: text-layer PDFs, Excel transcription, re-reads.
Each request is counted per model and day (Google resets free quotas at midnight
Pacific time), requests per minute are paced client-side, and when a model's
daily quota is used up the other one takes over automatically.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, TypeVar
from zoneinfo import ZoneInfo

from google import genai
from google.genai import errors, types
from PIL import Image

from . import config

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 2576
PACIFIC = ZoneInfo("America/Los_Angeles")
TEHRAN = ZoneInfo("Asia/Tehran")
T = TypeVar("T")


class AIError(RuntimeError):
    pass


class QuotaExceeded(AIError):
    """Every model's free daily quota is used up."""

    def __init__(self, message: str, reset_at: datetime):
        super().__init__(message)
        self.reset_at = reset_at


def fit_for_ai(img: Image.Image) -> tuple[Image.Image, float]:
    """Downscale so the long edge stays reasonable. Returns (image, scale)."""
    long_edge = max(img.size)
    if long_edge <= MAX_IMAGE_EDGE:
        return img, 1.0
    s = MAX_IMAGE_EDGE / long_edge
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS), s


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92, subsampling=0)
    return buf.getvalue()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# =================================================================== usage ==

def next_reset() -> datetime:
    now = datetime.now(PACIFIC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _today() -> str:
    return datetime.now(PACIFIC).date().isoformat()


class UsageBook:
    """Requests/tokens per model for the current quota day, kept on disk so a
    restart does not forget what was already spent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = config.USAGE_FILE
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"day": _today(), "models": {}, "limits": {}}

    def _roll(self) -> None:
        if self._data.get("day") != _today():
            self._data = {"day": _today(), "models": {}, "limits": self._data.get("limits", {})}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            log.debug("could not save usage: %s", exc)

    def _entry(self, model: str) -> dict:
        return self._data["models"].setdefault(model, {"requests": 0, "input": 0, "output": 0, "exhausted": False})

    def record(self, model: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self._roll()
            e = self._entry(model)
            e["requests"] += 1
            e["input"] += tokens_in
            e["output"] += tokens_out
            self._save()

    def mark_exhausted(self, model: str, limit: int | None) -> None:
        with self._lock:
            self._roll()
            e = self._entry(model)
            e["exhausted"] = True
            if limit:
                self._data.setdefault("limits", {})[model] = limit   # learned from Google's own error
                e["requests"] = max(e["requests"], limit)
            self._save()

    def get(self, model: str) -> dict:
        with self._lock:
            self._roll()
            return dict(self._entry(model))

    def learned_limit(self, model: str) -> int | None:
        with self._lock:
            return self._data.get("limits", {}).get(model)


USAGE = UsageBook()


class Pacer:
    """Client-side requests-per-minute limit, so bursts never hit Google's 429."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                delay = 60 - (now - self._times[0]) + 0.2
            time.sleep(min(delay, 5))


# ================================================================== models ==

@dataclass
class QuotaInfo:
    daily: bool
    limit: int | None
    retry_after: float | None


def _quota_info(exc: errors.APIError) -> QuotaInfo:
    text = json.dumps(exc.details, ensure_ascii=False) if exc.details is not None else str(exc)
    daily = bool(re.search(r"PerDay|per day|daily", text, re.I))
    limit = None
    m = re.search(r'"quotaValue"\s*:\s*"?(\d+)', text)
    if m:
        limit = int(m.group(1))
    retry = None
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text) or re.search(r"retry in ([\d.]+)s", text, re.I)
    if m:
        retry = float(m.group(1))
    return QuotaInfo(daily, limit, retry)


_THINKING = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}


def _plain_schema(schema):
    """Gemini's JSON-schema mode does not need (and may reject) additionalProperties."""
    if isinstance(schema, dict):
        return {k: _plain_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_plain_schema(v) for v in schema]
    return schema


_SKIP_MODELS = ("image", "tts", "audio", "live", "embedding", "robotics", "computer", "exp", "native")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def discover_model(lite: bool, exclude: set[str]) -> str | None:
    """Pick a working Flash / Flash-Lite model from the key's own model list
    (used when the configured name no longer exists)."""
    names = []
    for m in _client().models.list(config={"page_size": 200}):
        if "generateContent" not in (m.supported_actions or []):
            continue
        n = (m.name or "").removeprefix("models/")
        if ("flash" in n and ("lite" in n) == lite and n not in exclude
                and not any(s in n for s in _SKIP_MODELS)):
            names.append(n)
    if not names:
        return None
    # "-latest" alias first, then the newest stable version, previews last.
    names.sort(key=lambda n: (n.endswith("-latest"), "preview" not in n, _version(n)), reverse=True)
    return names[0]


def short_error(exc: BaseException | str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    return text if len(text) <= limit else text[:limit] + "…"


class GeminiModel:
    def __init__(self, name: str, model: str, label: str, rpm: int, rpd: int):
        self.name = name
        self.model = model
        self.label = label
        self.default_rpd = rpd
        self.pacer = Pacer(rpm)
        self._thinking_ok = True
        self._media_ok = True
        self._schema_ok = True
        self._tried: set[str] = {model}
        self.last_error = ""
        self.last_ok: datetime | None = None

    def _degrade(self, has_media: bool) -> str | None:
        """Drop the next optional request feature after an unexplained 400."""
        if self._thinking_ok:
            self._thinking_ok = False
            return "thinking_config"
        if has_media and self._media_ok:
            self._media_ok = False
            return "media_resolution"
        if self._schema_ok:
            self._schema_ok = False
            return "response_json_schema"
        return None

    def _switch_model(self) -> bool:
        try:
            new = discover_model(self.name == "lite", self._tried)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not list gemini models: %s", exc)
            return False
        if not new:
            return False
        log.warning("%s not found, switching to %s", self.model, new)
        self._tried.add(new)
        self.model = new
        return True

    # ---- quota ------------------------------------------------------------
    @property
    def limit(self) -> int:
        return USAGE.learned_limit(self.model) or self.default_rpd

    def used(self) -> int:
        return USAGE.get(self.model)["requests"]

    def remaining(self) -> int:
        e = USAGE.get(self.model)
        return 0 if e["exhausted"] else max(0, self.limit - e["requests"])

    def available(self) -> bool:
        return bool(config.GEMINI_API_KEY) and self.remaining() > 0

    # ---- call -------------------------------------------------------------
    def call(self, system: str, text: str, images: list[tuple[Image.Image, str]], schema: dict,
             schema_name: str, effort: str, max_tokens: int) -> dict:
        try:
            data = self._call(system, text, images, schema, schema_name, effort, max_tokens)
        except Exception as exc:
            self.last_error = f"{_fa(datetime.now(TEHRAN).strftime('%H:%M'))} — {short_error(exc, 400)}"
            raise
        self.last_error = ""
        self.last_ok = datetime.now(TEHRAN)
        return data

    def _call(self, system: str, text: str, images: list[tuple[Image.Image, str]], schema: dict,
              schema_name: str, effort: str, max_tokens: int) -> dict:
        if not self.available():
            raise QuotaExceeded(f"{self.label}: daily quota used up", next_reset())
        media = [types.Part.from_bytes(data=_jpeg(img) if fmt == "JPEG" else _png(img),
                                       mime_type="image/jpeg" if fmt == "JPEG" else "image/png")
                 for img, fmt in images]
        attempts = 0
        empty_answers = 0
        # a slow or broken Gemini must never keep the user waiting for minutes
        deadline = time.monotonic() + config.AI_CALL_BUDGET
        while True:
            attempts += 1
            if attempts > 8 or time.monotonic() > deadline:
                raise AIError(f"gemini kept failing or too slow on {self.model}")
            prompt = text
            cfg = {
                "system_instruction": system,
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens,
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if self._schema_ok:
                cfg["response_json_schema"] = _plain_schema(schema)
            else:
                prompt = (f"{text}\n\nAnswer with one JSON object that follows this JSON schema exactly:\n"
                          f"{json.dumps(_plain_schema(schema), ensure_ascii=False)}")
            if self._thinking_ok:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level=_THINKING.get(effort, "MEDIUM"))
            if media and self._media_ok:
                cfg["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
            parts = [*media, types.Part.from_text(text=prompt)]
            self.pacer.wait()
            try:
                resp = _client().models.generate_content(
                    model=self.model, contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(**cfg))
            except errors.APIError as exc:
                msg = f"{exc.message or ''} {exc.status or ''}".lower()
                log.warning("%s error %s: %s", self.model, exc.code, short_error(exc.message or exc))
                if exc.code == 429:
                    info = _quota_info(exc)
                    if info.daily:
                        USAGE.mark_exhausted(self.model, info.limit)
                        log.warning("%s daily quota exhausted (limit %s)", self.model, info.limit)
                        raise QuotaExceeded(f"{self.label}: daily quota used up", next_reset()) from exc
                    wait = min(info.retry_after or 10.0, 20.0)
                    if attempts > 2 or time.monotonic() + wait > deadline:
                        raise AIError(f"gemini busy (per-minute limit): {exc.message}") from exc
                    log.info("%s per-minute limit hit, waiting %.0fs", self.model, wait)
                    time.sleep(wait)
                    continue
                if "api key" in msg or (exc.code in (401, 403) and "permission" in msg):
                    raise AIError(f"gemini authentication failed: {exc.message}") from exc
                if "location is not supported" in msg:
                    raise AIError("gemini is not available in this server's region") from exc
                if exc.code == 400 and self._thinking_ok and "thinking" in msg:
                    self._thinking_ok = False
                    continue
                if exc.code == 400 and self._media_ok and "media" in msg:
                    self._media_ok = False
                    continue
                if exc.code == 400:
                    dropped = self._degrade(bool(media))     # some models reject optional features
                    if dropped:
                        log.warning("%s rejected the request, retrying without %s", self.model, dropped)
                        continue
                if exc.code == 404 and self._switch_model():
                    continue
                if exc.code in (500, 502, 503, 504) and attempts <= 2 and time.monotonic() + 3 < deadline:
                    time.sleep(3)                 # "model overloaded" is common on the free tier
                    continue
                if exc.code == 404:
                    raise AIError(f"gemini model not found: {self.model}") from exc
                raise AIError(f"gemini error {exc.code}: {exc.message}") from exc
            except Exception as exc:  # noqa: BLE001 - network hiccups
                slow = "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
                if not slow and attempts <= 1 and time.monotonic() + 2 < deadline:
                    time.sleep(2)
                    continue
                raise AIError(f"gemini {'too slow' if slow else 'unreachable'}: {exc}") from exc
            um = resp.usage_metadata
            tin = (um.prompt_token_count or 0) if um else 0
            tout = ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0
            USAGE.record(self.model, tin, tout)
            log.info("%s %s: in=%s out=%s", self.model, schema_name, tin, tout)
            cand = resp.candidates[0] if resp.candidates else None
            finish = str(getattr(cand, "finish_reason", "") or "")
            if cand is None:
                raise AIError(f"gemini returned no answer ({getattr(resp, 'prompt_feedback', '')})")
            if "MAX_TOKENS" in finish:
                raise AIError("gemini answer was cut off (max tokens)")
            try:
                return _parse_json(resp.text or "")
            except AIError:
                empty_answers += 1
                if empty_answers > 1:
                    raise
                log.warning("%s gave an unusable answer (finish %s), asking again", self.model, finish)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    if not text.startswith("{") and "{" in text and "}" in text:   # prose around the JSON (no-schema mode)
        text = text[text.index("{"): text.rindex("}") + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIError(f"gemini returned invalid JSON: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise AIError("gemini returned a non-object JSON")
    return data


_client_lock = threading.Lock()
_client_obj: genai.Client | None = None


def _client() -> genai.Client:
    global _client_obj
    with _client_lock:
        if _client_obj is None:
            _client_obj = genai.Client(api_key=config.GEMINI_API_KEY,
                                       http_options=types.HttpOptions(timeout=config.AI_TIMEOUT * 1000))
        return _client_obj


MODELS: dict[str, GeminiModel] = {
    "flash": GeminiModel("flash", config.GEMINI_MODEL, "Gemini Flash", config.GEMINI_RPM, config.GEMINI_RPD),
    "lite": GeminiModel("lite", config.GEMINI_LITE_MODEL, "Gemini Flash-Lite", config.GEMINI_LITE_RPM,
                        config.GEMINI_LITE_RPD),
}
Provider = GeminiModel

# Which model leads for each kind of work (the other one takes over when needed).
ROUTES = {
    "image": ("flash", "lite"),
    "command": ("flash", "lite"),
    "text": ("lite", "flash"),
    "table": ("lite", "flash"),
    "verify": ("lite", "flash"),
}


def providers(task: str = "verify") -> list[GeminiModel]:
    """Models with quota left, in the preferred order for `task`."""
    return [MODELS[n] for n in ROUTES.get(task, ("lite", "flash")) if MODELS[n].available()]


def enabled() -> bool:
    return bool(config.GEMINI_API_KEY)


def with_failover(fn: Callable[[GeminiModel], T], task: str) -> T:
    order = providers(task)
    if not order:
        if not enabled():
            raise AIError("GEMINI_API_KEY is not set")
        raise QuotaExceeded("free daily quota used up", next_reset())
    errors_seen = []
    for p in order:
        try:
            return fn(p)
        except QuotaExceeded as exc:
            errors_seen.append(f"{p.name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - hand over to the other model
            log.warning("%s failed: %s", p.model, exc)
            errors_seen.append(f"{p.name}: {exc}")
    if not providers(task):
        raise QuotaExceeded("; ".join(errors_seen), next_reset())
    raise AIError("; ".join(errors_seen))


def describe() -> str:
    if not enabled():
        return "غیرفعال (GEMINI_API_KEY تنظیم نشده)"
    return " + ".join(f"{m.label} [{m.model}]" for m in MODELS.values())


# =============================================================== reporting ==

def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _bar(used: int, limit: int, width: int = 10) -> str:
    filled = min(width, round(width * used / limit)) if limit else width
    return "▓" * filled + "░" * (width - filled)


def _until(reset: datetime) -> str:
    delta = reset - datetime.now(PACIFIC)
    h, rem = divmod(max(0, int(delta.total_seconds())), 3600)
    return f"{_fa(h)} ساعت و {_fa(rem // 60)} دقیقه دیگر"


def reset_text() -> str:
    reset = next_reset()
    return f"ساعت {_fa(reset.astimezone(TEHRAN).strftime('%H:%M'))} به وقت تهران ({_until(reset)})"


def usage_report() -> str:
    if not enabled():
        return "📊 هوش مصنوعی فعال نیست (GEMINI_API_KEY تنظیم نشده)."
    lines = ["📊 سهمیه رایگان Gemini — امروز"]
    total_tokens = 0
    for m in MODELS.values():
        e = USAGE.get(m.model)
        used, limit = min(e["requests"], m.limit), m.limit
        left = m.remaining()
        state = "✅" if left > limit * 0.2 else ("⚠️" if left > 0 else "⛔️")
        lines.append(f"\n{state} {m.label}  ({m.model})")
        lines.append(f"{_bar(used, limit)}  {_fa(used)} از {_fa(limit)} درخواست — {_fa(left)} باقی‌مانده")
        if m.last_error:
            lines.append(f"❗️ آخرین خطا: {m.last_error}")
        total_tokens += e["input"] + e["output"]
    lines.append(f"\n🔤 توکن مصرف‌شده امروز: {_fa(f'{total_tokens:,}')}")
    lines.append(f"⏰ تمدید سهمیه: {reset_text()}")
    lines.append(f"📄 ظرفیت تقریبی باقی‌مانده: {capacity_text()}")
    lines.append("\nℹ️ این آمار را خود ربات می‌شمارد؛ آمار دقیق پروژه در aistudio.google.com است.")
    return "\n".join(lines)


def self_test() -> str:
    """Send one tiny request to each model and report what Google answers."""
    if not enabled():
        return "❌ GEMINI_API_KEY تنظیم نشده."
    lines = ["🩺 آزمایش اتصال به Gemini"]
    for m in MODELS.values():
        started = time.monotonic()
        try:
            data = m.call("Reply in JSON.", "Say OK.", [], _obj({"answer": _STR}), "self_test", "low", 2048)
        except QuotaExceeded:
            lines.append(f"\n⛔️ {m.label} ({m.model})\nسهمیه امروز تمام شده — تمدید: {reset_text()}")
            continue
        except Exception as exc:  # noqa: BLE001
            lines.append(f"\n❌ {m.label} ({m.model})\n{short_error(exc, 500)}")
            continue
        took = time.monotonic() - started
        notes = [n for n, ok in (("بدون thinking", m._thinking_ok), ("بدون media_resolution", m._media_ok),
                                 ("بدون schema", m._schema_ok)) if not ok]
        extra = f" ({'، '.join(notes)})" if notes else ""
        lines.append(f"\n✅ {m.label} ({m.model})\nپاسخ داد در {_fa(f'{took:.1f}')} ثانیه{extra}: "
                     f"{short_error(data.get('answer', ''), 40)}")
    lines.append("\n(این آزمایش از هر مدل ۱ درخواست سهمیه مصرف می‌کند.)")
    return "\n".join(lines)


def capacity_text() -> str:
    flash, lite = MODELS["flash"].remaining(), MODELS["lite"].remaining()
    pdf_pages = lite + flash                 # one request per text-PDF page
    photos = (lite + flash) // 2             # a photo: one read + one independent re-read
    return f"حدود {_fa(pdf_pages)} صفحه PDF یا {_fa(photos)} عکس"


def usage_line() -> str:
    if not enabled():
        return ""
    parts = [f"{'Flash' if m.name == 'flash' else 'Lite'}: {_fa(m.remaining())}/{_fa(m.limit)}" for m in MODELS.values()]
    return "📊 سهمیه باقی‌مانده امروز — " + " • ".join(parts)


def estimate_requests(kind: str, pages: int) -> int:
    """Rough number of requests a file will need (analysis + verification)."""
    return pages * 2 if kind == "image" else pages


def remaining_total() -> int:
    return sum(m.remaining() for m in MODELS.values())


# ================================================================= schemas ==

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOX = {"type": "array", "items": _INT,
        "description": "[ymin, xmin, ymax, xmax] normalised to 0-1000 (0,0 = top-left of the image)"}

PAGE_SCHEMA = _obj({
    "currency": {"type": "string", "description": "ریال / تومان / دلار ... as stated on the page, or empty"},
    "columns": {"type": "array", "items": _obj({"column_id": _INT, "header": _STR})},
    "text_prices": {"type": "array", "items": _obj({"id": _INT, "column_id": _INT, "label": _STR, "group": _STR})},
    "image_prices": {"type": "array", "items": _obj({"text": _STR, "box_2d": _BOX, "column_id": _INT,
                                                     "label": _STR, "group": _STR})},
})

PAGE_SYSTEM = """You read price lists (mostly Persian, sometimes English) for a bot that rewrites the prices in place, keeping the document otherwise identical. You get an image of one page or photo and, for PDFs, the number tokens found in the file's text layer.

Decide exactly which numbers are PRICES: monetary amounts of items that must change when the seller raises or lowers prices.

Prices: values in columns such as قیمت، فی، مبلغ، قیمت فروش، قیمت مصرف کننده، قیمت همکار، قیمت عمده، قیمت نماینده، price, and money amounts written next to items (e.g. «۲۵۰,۰۰۰ تومان»). When there are several price columns (e.g. wholesale and retail), all of them are prices. Money totals are prices too.

Never prices: row numbers (ردیف), product/part/technical codes (کد کالا، شماره فنی، کد)، barcodes, quantities and pack sizes (تعداد، تعداد در کارتن، عدد)، dates and years (1405/06/23، 1405)، phone numbers, page numbers, percentages and discount rates, model numbers or specs inside product names (405، 206، L90، EF7، ۷۵ درجه، 76/5)، weights, dimensions.

Groups: many lists are split into groups (sections) - a title row or bar across the table («گروه پژو 405»، «ترموستات ها»، «ایران خودرو») or a group name written once beside its rows (a merged cell, sometimes sideways). For every price give "group": that title exactly as written on this page, the one the row falls under. Leave it empty when the page shows no such title above the row (a list without groups, or a group that began on an earlier page). Never use the list's own title, the company name, a column header or a product name as a group.

Boxes use box_2d = [ymin, xmin, ymax, xmax] normalised to 0-1000 over the whole image.

Work carefully: read the column headers first, then go row by row. Missing one price or marking a code as a price both ruin the output."""

VERIFY_SCHEMA = _obj({"reads": {"type": "array", "items": _obj({"index": _INT, "text": _STR})}})

VERIFY_SYSTEM = """You transcribe numbers cut out of a price list. Each strip in the image has a red index tag on its left and one number on its right. Copy every number exactly: all digits in order (keep Persian ۰-۹ as Persian, Latin 0-9 as Latin) and the separators as shown. If a strip is unreadable or does not show exactly one number, return an empty text for it. Do not guess."""

COMMAND_SCHEMA = _obj({
    "status": {"type": "string", "enum": ["ok", "clarify", "not_a_price_command"]},
    "question": _STR,
    "summary": _STR,
    "rules": {"type": "array", "items": _obj({
        "scope": {"type": "string", "enum": ["all", "ids"]},
        "ids": {"type": "array", "items": _STR},
        "op": {"type": "string", "enum": ["percent", "add", "set", "multiply"]},
        "value": {"type": "number"},
    })},
    "rounding": _obj({"step": {"type": "number"}, "mode": {"type": "string", "enum": ["nearest", "up", "down"]}}),
})

COMMAND_SYSTEM = """You turn a seller's instruction (usually Persian, informal) about changing the prices of a price list into an exact plan. You see every price with an id, page, column, row label and current value.

Rules are applied in order; each price takes the LAST rule whose scope contains it; prices matched by no rule stay unchanged.
- op "percent": value is the signed percentage (+10 = ten percent increase, -5 = five percent decrease).
- op "add": value is a signed amount in the SAME unit as the listed prices. 1 تومان = 10 ریال; هزار = 1,000; میلیون = 1,000,000. If the list is in ریال and the user speaks in تومان, convert.
- op "set": value is the new absolute price (same unit as the list).
- op "multiply": value is the factor.
rounding.step: 0 = no rounding; otherwise round every NEW price to a multiple of step ("رند به هزار" = 1000). mode nearest/up/down.

Selecting items: match row labels semantically (e.g. "پرایدها" = rows whose label mentions پراید; "ردیف ۱ تا ۱۰" = rows 1..10 by their row number in the label, or by order if there is none; "صفحه ۲" = page 2). The list's groups (sections) are given per price ("-" = none): "گروه پژو" / "پژوها" when the list has such a group means every price of that group (all groups whose name matches, e.g. پژو 405 and پژو 206), whatever the row labels say; "بقیه" = every price not already chosen. Row numbers often start again at 1 in every group: "ردیف ۱ تا ۱۰" inside a named group means those rows of that group; without a group named, and with rows numbered per group, ask which group (status clarify). Use the ids exactly as given.

status "clarify" only when the instruction is genuinely ambiguous (e.g. the direction of the change is unclear); then ask a short question in Persian in "question". status "not_a_price_command" if the text is not about changing prices. "summary" is a short Persian description of what will be done."""

TABLE_SCHEMA = _obj({
    "tables": {"type": "array", "items": _obj({
        "title": _STR,
        "direction": {"type": "string", "enum": ["rtl", "ltr"]},
        "headers": {"type": "array", "items": _STR},
        "rows": {"type": "array", "items": _obj({
            "cells": {"type": "array", "items": _obj({"text": _STR, "price_id": _STR})},
        })},
    })},
})

TABLE_SYSTEM = """You transcribe price-list tables from an image into a spreadsheet, cell by cell, exactly as written (Persian or English). You also get the ids of the price cells, which must be referenced instead of copied.

- One entry per table on the page, top to bottom. title: the heading written above the table (or the page title), else empty.
- direction "rtl" for Persian tables: the FIRST cell of every row is the RIGHTMOST column. "ltr" for left-to-right tables.
- headers: the header row (join a multi-line header with a space). Every row must have exactly as many cells as there are headers; use empty text for empty cells and for cells that only hold a picture.
- A cell that shows one of the listed prices: price_id = its id and text = the price as shown. Every other cell: price_id = "".
- A row that only names a group of the list (a title bar across the table such as «گروه پژو 405»): one row whose first cell holds that title and all other cells empty, where it stands between the rows.
- Copy product names, codes and numbers exactly; do not translate or reorder words."""


# ============================================================ coordinates ==

def _to_norm(box: tuple[int, int, int, int], w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [round(y0 * 1000 / h), round(x0 * 1000 / w), round(y1 * 1000 / h), round(x1 * 1000 / w)]


def _from_norm(b: list, w: int, h: int) -> list[int] | None:
    if not isinstance(b, list) or len(b) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    box = [round(xmin * w / 1000), round(ymin * h / 1000), round(xmax * w / 1000), round(ymax * h / 1000)]
    return box if box[2] > box[0] and box[3] > box[1] else None


# ============================================================ public calls ==

def analyze_page_all(img: Image.Image, candidates: list[tuple[int, str, tuple[int, int, int, int]]],
                     raster_only: bool) -> list[tuple[str, dict]]:
    """Analyse one page. Returns [(model name, answer)], answer boxes converted to
    pixels [x0, y0, x1, y1] of `img` under the key "bbox"."""
    W, H = img.size
    if raster_only:
        text = (
            "There is no usable text layer, so read the prices from the image.\n"
            "Put every price in \"image_prices\":\n"
            "- text: exactly as displayed — the same digits (Persian ۰-۹, Arabic ٠-٩ or Latin 0-9) and the same "
            "separators (, / . ٬ or space).\n"
            "- box_2d: tight around the digits of that one number only (not the whole cell, no currency word).\n"
            "- label: the row number if the table has one, then the product name as written, e.g. "
            "«12 - ترموستات پراید» (max ~10 words). column_id: 1 = rightmost price column on this page, 2 = the "
            "next one to the left, and so on.\n"
            "Skip empty cells and words like «به زودی», «تماس بگیرید», «ناموجود», «COMING SOON».\n"
            "Leave \"text_prices\" empty."
        )
    else:
        lines = "\n".join(f"{cid} | {t} | {_to_norm(b, W, H)}" for cid, t, b in candidates)
        text = (
            "Number tokens from the PDF text layer (id | text | box_2d):\n"
            f"{lines}\n\n"
            "Put the id of every candidate that is a price in \"text_prices\", with a label (the row number if the "
            "table has one, then the product name as written, e.g. «12 - ترموستات پراید», max ~10 words) and "
            "column_id (1 = rightmost price column on this page, 2 = next to the left...).\n"
            "If the image shows prices that are NOT in the candidate list (for example inside a picture), add "
            "them to \"image_prices\" with a tight box_2d and the text exactly as displayed. Otherwise leave "
            "\"image_prices\" empty."
        )
    effort = config.AI_EFFORT_IMAGE if raster_only else config.AI_EFFORT_TEXT
    task = "image" if raster_only else "text"

    def run(p: GeminiModel) -> tuple[str, dict]:
        data = p.call(PAGE_SYSTEM, text, [(img, "JPEG")], PAGE_SCHEMA, "price_page", effort, 32768)
        prices = []
        for item in data.get("image_prices", []) or []:
            box = _from_norm(item.get("box_2d"), W, H)
            if box:
                prices.append({**item, "bbox": box})
        return p.name, {**data, "image_prices": prices}

    order = providers(task)
    if config.AI_CONSENSUS and len(order) >= 2:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(run, p) for p in order[:2]]
        results, errs = [], []
        for fut in futs:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - the other model's answer is still usable
                errs.append(str(exc))
        if results:
            return results
        raise AIError("; ".join(errs))
    return [with_failover(run, task)]


def verify_reads(sheet: Image.Image, indexes: list[int], provider: GeminiModel) -> dict[int, str]:
    text = (f"There are {len(indexes)} strips, tagged {indexes[0]} to {indexes[-1]}. "
            "Return one read per tag, using the tag number as index.")
    data = provider.call(VERIFY_SYSTEM, text, [(sheet, "PNG")], VERIFY_SCHEMA, "number_reads",
                         config.AI_EFFORT_IMAGE, 16384)
    out = {}
    for r in data.get("reads", []) or []:
        try:
            out[int(r["index"])] = str(r.get("text", ""))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def interpret_command(command: str, items: list[dict], currency: str) -> dict:
    lines = "\n".join(f"{it['id']} | p{it['page']} | c{it['column']} | g{it.get('group', '-')} | {it['label']} | "
                      f"{it['value']}" for it in items)
    text = (f"Currency of the list: {currency or 'unknown'}\n"
            f"Prices (id | page | column | group | row label | current value):\n{lines}\n\n"
            f"Instruction: «{command}»")
    return with_failover(lambda p: p.call(COMMAND_SYSTEM, text, [], COMMAND_SCHEMA, "price_plan",
                                          config.AI_EFFORT_COMMAND, 32768), "command")


def transcribe_tables(img: Image.Image, prices: list[tuple[str, str, tuple[int, int, int, int]]]) -> dict:
    W, H = img.size
    lines = "\n".join(f"{pid} | {t} | {_to_norm(b, W, H)}" for pid, t, b in prices)
    text = f"Price cells on this page (id | price as shown | box_2d):\n{lines or '(none)'}\n\nTranscribe the table(s)."
    return with_failover(lambda p: p.call(TABLE_SYSTEM, text, [(img, "JPEG")], TABLE_SCHEMA, "price_tables",
                                          config.AI_EFFORT_TABLE, 65536), "table")
