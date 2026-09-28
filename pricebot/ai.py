"""Vision/text model calls through two independent providers: Claude
(Anthropic SDK) and OpenAI (Responses API).

Pages are analysed by both providers when both keys are configured and the
answers are merged (see pipeline._merge_page); every other call uses the
preferred provider and fails over to the other one.
"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import logging
import threading
from typing import Callable, TypeVar

import anthropic
import openai
from PIL import Image

from . import config

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 2576
T = TypeVar("T")


class AIError(RuntimeError):
    pass


def _encode(img: Image.Image, fmt: str) -> tuple[str, str]:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=92, subsampling=0)
        media = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        media = "image/png"
    return media, base64.standard_b64encode(buf.getvalue()).decode("ascii")


def fit_for_ai(img: Image.Image) -> tuple[Image.Image, float]:
    """Downscale so the long edge fits the models' native resolution. Returns (image, scale)."""
    long_edge = max(img.size)
    if long_edge <= MAX_IMAGE_EDGE:
        return img, 1.0
    s = MAX_IMAGE_EDGE / long_edge
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS), s


# ================================================================ providers ==

class Provider:
    name = ""
    label = ""

    @property
    def model(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError

    def call(self, system: str, text: str, images: list[tuple[Image.Image, str]], schema: dict,
             schema_name: str, effort: str, max_tokens: int) -> dict:
        """One request whose answer must be a JSON object matching `schema`.
        `images` are (image, "JPEG"|"PNG") and are placed before the text."""
        raise NotImplementedError


class ClaudeProvider(Provider):
    name = "claude"
    label = "Claude"

    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None
        self._lock = threading.Lock()
        self._fallbacks_supported = True

    @property
    def model(self) -> str:
        return config.CLAUDE_MODEL

    def available(self) -> bool:
        return bool(config.ANTHROPIC_API_KEY)

    def _get(self) -> anthropic.Anthropic:
        with self._lock:
            if self._client is None:
                self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=300.0, max_retries=3)
            return self._client

    def call(self, system, text, images, schema, schema_name, effort, max_tokens):
        content = []
        for img, fmt in images:
            media, data = _encode(img, fmt)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": text})
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        client = self._get()
        msg = None
        try:
            if self._fallbacks_supported:
                # server-side fallback: a declined request is retried on another Claude model
                try:
                    with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                                     fallbacks="default", **kwargs) as stream:
                        msg = stream.get_final_message()
                except anthropic.BadRequestError as exc:
                    if "fallback" not in str(exc).lower():
                        raise
                    log.warning("server-side fallbacks not available for %s; continuing without", self.model)
                    self._fallbacks_supported = False
            if msg is None:
                with client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise AIError(f"claude authentication failed: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise AIError(f"claude bad request: {exc}") from exc
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise AIError(f"claude unavailable: {exc}") from exc
        if msg.stop_reason == "refusal":
            raise AIError("claude declined the request")
        if msg.stop_reason == "max_tokens":
            raise AIError("claude answer was cut off (max_tokens)")
        log.info("claude %s: in=%s out=%s", schema_name, msg.usage.input_tokens, msg.usage.output_tokens)
        out = next((b.text for b in msg.content if b.type == "text"), "")
        return _parse_json(out, "claude")


class OpenAIProvider(Provider):
    name = "openai"
    label = "OpenAI"

    def __init__(self) -> None:
        self._client: openai.OpenAI | None = None
        self._lock = threading.Lock()
        self._responses_supported = True

    @property
    def model(self) -> str:
        return config.OPENAI_MODEL

    def available(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def _get(self) -> openai.OpenAI:
        with self._lock:
            if self._client is None:
                kwargs = {"api_key": config.OPENAI_API_KEY, "timeout": 300.0, "max_retries": 3}
                if config.OPENAI_BASE_URL:
                    kwargs["base_url"] = config.OPENAI_BASE_URL
                self._client = openai.OpenAI(**kwargs)
            return self._client

    def call(self, system, text, images, schema, schema_name, effort, max_tokens):
        client = self._get()
        encoded = [(_encode(img, fmt)) for img, fmt in images]
        try:
            if self._responses_supported:
                try:
                    return self._responses(client, system, text, encoded, schema, schema_name, effort, max_tokens)
                except openai.NotFoundError:
                    # Some OpenAI-compatible gateways only implement Chat Completions.
                    log.warning("Responses API not available at this endpoint; using Chat Completions")
                    self._responses_supported = False
            return self._chat(client, system, text, encoded, schema, schema_name, effort, max_tokens)
        except openai.AuthenticationError as exc:
            raise AIError(f"openai authentication failed: {exc}") from exc
        except openai.BadRequestError as exc:
            raise AIError(f"openai bad request: {exc}") from exc
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            raise AIError(f"openai unavailable: {exc}") from exc

    def _responses(self, client, system, text, encoded, schema, schema_name, effort, max_tokens):
        content = [{"type": "input_image", "image_url": f"data:{media};base64,{data}",
                    "detail": config.OPENAI_IMAGE_DETAIL} for media, data in encoded]
        content.append({"type": "input_text", "text": text})
        resp = client.responses.create(
            model=self.model,
            instructions=system,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            reasoning={"effort": effort},
            max_output_tokens=max_tokens,
        )
        if getattr(resp, "status", "completed") == "incomplete":
            raise AIError(f"openai answer incomplete: {getattr(resp, 'incomplete_details', '')}")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            log.info("openai %s: in=%s out=%s", schema_name, usage.input_tokens, usage.output_tokens)
        return _parse_json(resp.output_text or "", "openai")

    def _chat(self, client, system, text, encoded, schema, schema_name, effort, max_tokens):
        content = [{"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}", "detail": "high"}}
                   for media, data in encoded]
        content.append({"type": "text", "text": text})
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
            reasoning_effort=effort,
            max_completion_tokens=max_tokens,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise AIError("openai answer was cut off (length)")
        if getattr(choice.message, "refusal", None):
            raise AIError("openai declined the request")
        return _parse_json(choice.message.content or "", "openai")


def _parse_json(text: str, who: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIError(f"{who} returned invalid JSON: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise AIError(f"{who} returned {type(data).__name__}, expected an object")
    return data


_REGISTRY: dict[str, Provider] = {"claude": ClaudeProvider(), "openai": OpenAIProvider()}


def providers() -> list[Provider]:
    """Configured providers that have a key, in preference order."""
    out = []
    for name in config.AI_PROVIDERS:
        p = _REGISTRY.get(name)
        if p is not None and p.available() and p not in out:
            out.append(p)
    return out


def enabled() -> bool:
    return bool(providers())


def with_failover(fn: Callable[[Provider], T], order: list[Provider] | None = None) -> T:
    """Run `fn` on the preferred provider; on failure try the next one."""
    order = order or providers()
    if not order:
        raise AIError("no AI provider configured (set ANTHROPIC_API_KEY and/or OPENAI_API_KEY)")
    errors = []
    for p in order:
        try:
            return fn(p)
        except Exception as exc:  # noqa: BLE001 - any failure of one provider hands over to the next
            log.warning("%s failed: %s", p.name, exc)
            errors.append(f"{p.name}: {exc}")
    raise AIError("; ".join(errors))


def describe() -> str:
    ps = providers()
    if not ps:
        return "هیچ"
    mode = " (هر دو، با رأی‌گیری)" if config.AI_CONSENSUS and len(ps) > 1 else ""
    return " + ".join(f"{p.label} [{p.model}]" for p in ps) + mode


# ================================================================== schemas ==

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
_INT = {"type": "integer"}

PAGE_SCHEMA = _obj({
    "currency": {"type": "string", "description": "ریال / تومان / دلار ... as stated on the page, or empty"},
    "columns": {"type": "array", "items": _obj({"column_id": _INT, "header": _STR})},
    "text_prices": {"type": "array", "items": _obj({"id": _INT, "column_id": _INT, "label": _STR})},
    "image_prices": {"type": "array", "items": _obj({
        "text": _STR,
        "bbox": {"type": "array", "items": _INT},
        "column_id": _INT,
        "label": _STR,
    })},
    "notes": _STR,
})

PAGE_SYSTEM = """You read price lists (mostly Persian, sometimes English) for a bot that rewrites the prices in place, keeping the document otherwise identical. You get an image of one page or photo and, for PDFs, the number tokens found in the file's text layer.

Decide exactly which numbers are PRICES: monetary amounts of items that must change when the seller raises or lowers prices.

Prices: values in columns such as قیمت، فی، مبلغ، قیمت فروش، قیمت مصرف کننده، قیمت همکار، قیمت عمده، قیمت نماینده، price, and money amounts written next to items (e.g. «۲۵۰,۰۰۰ تومان»). When there are several price columns (e.g. wholesale and retail), all of them are prices. Money totals are prices too.

Never prices: row numbers (ردیف), product/part/technical codes (کد کالا، شماره فنی، کد)، barcodes, quantities and pack sizes (تعداد، تعداد در کارتن، عدد)، dates and years (1405/06/23، 1405)، phone numbers, page numbers, percentages and discount rates, model numbers or specs inside product names (405، 206، L90، EF7، ۷۵ درجه، 76/5)، weights, dimensions.

Coordinates are pixels of the image you are given: x to the right, y down, origin at the top-left corner.

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
    "rounding": _obj({
        "step": {"type": "number"},
        "mode": {"type": "string", "enum": ["nearest", "up", "down"]},
    }),
})

COMMAND_SYSTEM = """You turn a seller's instruction (usually Persian, informal) about changing the prices of a price list into an exact plan. You see every price with an id, page, column, row label and current value.

Rules are applied in order; each price takes the LAST rule whose scope contains it; prices matched by no rule stay unchanged.
- op "percent": value is the signed percentage (+10 = ten percent increase, -5 = five percent decrease).
- op "add": value is a signed amount in the SAME unit as the listed prices. 1 تومان = 10 ریال; هزار = 1,000; میلیون = 1,000,000. If the list is in ریال and the user speaks in تومان, convert.
- op "set": value is the new absolute price (same unit as the list).
- op "multiply": value is the factor.
rounding.step: 0 = no rounding; otherwise round every NEW price to a multiple of step ("رند به هزار" = 1000). mode nearest/up/down.

Selecting items: match row labels semantically (e.g. "پرایدها" = rows whose label mentions پراید; "ردیف ۱ تا ۱۰" = rows 1..10 by their row number in the label, or by order if there is none; "صفحه ۲" = page 2). Use the ids exactly as given.

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
- Copy product names, codes and numbers exactly; do not translate or reorder words."""


# ============================================================ public calls ==

def analyze_page_all(img: Image.Image, candidates: list[tuple[int, str, tuple[int, int, int, int]]],
                     raster_only: bool) -> list[tuple[str, dict]]:
    """Analyse one page. With consensus on and two providers configured, both
    run in parallel and both answers are returned (preferred first)."""
    W, H = img.size
    if raster_only:
        text = (
            f"Image size: {W}x{H} pixels. There is no usable text layer, so read the prices from the image.\n"
            "Put every price in \"image_prices\":\n"
            "- text: exactly as displayed — the same digits (Persian ۰-۹, Arabic ٠-٩ or Latin 0-9) and the same "
            "separators (, / . ٬ or space).\n"
            "- bbox: [x0, y0, x1, y1] in pixels of this image, tight around the digits of that one number only "
            "(not the whole cell, no currency word).\n"
            "- label: the row number if the table has one, then the product name as written, e.g. "
            "«12 - ترموستات پراید» (max ~10 words). column_id: 1 = rightmost price column on this page, 2 = the "
            "next one to the left, and so on.\n"
            "Skip empty cells and words like «به زودی», «تماس بگیرید», «ناموجود», «COMING SOON».\n"
            "Leave \"text_prices\" empty."
        )
    else:
        lines = "\n".join(f"{cid} | {t} | {b[0]},{b[1]},{b[2]},{b[3]}" for cid, t, b in candidates)
        text = (
            f"Image size: {W}x{H} pixels.\n"
            "Number tokens from the PDF text layer (id | text | x0,y0,x1,y1 in pixels of this image):\n"
            f"{lines}\n\n"
            "Put the id of every candidate that is a price in \"text_prices\", with a label (the row number if the "
            "table has one, then the product name as written, e.g. «12 - ترموستات پراید», max ~10 words) and "
            "column_id (1 = rightmost price column on this page, 2 = next to the left...).\n"
            "If the image shows prices that are NOT in the candidate list (for example inside a picture), add "
            "them to \"image_prices\" with a tight pixel bbox and the text exactly as displayed. Otherwise leave "
            "\"image_prices\" empty."
        )
    effort = config.AI_EFFORT_IMAGE if raster_only else config.AI_EFFORT_TEXT

    def run(p: Provider) -> dict:
        return p.call(PAGE_SYSTEM, text, [(img, "JPEG")], PAGE_SCHEMA, "price_page", effort, 32000)

    provs = providers()
    if not provs:
        raise AIError("no AI provider configured")
    if config.AI_CONSENSUS and len(provs) >= 2:
        pair = provs[:2]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(run, p) for p in pair]
        results, errors = [], []
        for p, fut in zip(pair, futs):
            try:
                results.append((p.name, fut.result()))
            except Exception as exc:  # noqa: BLE001 - the other provider's answer is still usable
                log.warning("%s failed on page: %s", p.name, exc)
                errors.append(f"{p.name}: {exc}")
        if not results:
            raise AIError("; ".join(errors))
        return results
    return [with_failover(lambda p: (p.name, run(p)))]


def verify_reads(sheet: Image.Image, indexes: list[int], provider: Provider) -> dict[int, str]:
    text = (f"There are {len(indexes)} strips, tagged {indexes[0]} to {indexes[-1]}. "
            "Return one read per tag, using the tag number as index.")
    data = provider.call(VERIFY_SYSTEM, text, [(sheet, "PNG")], VERIFY_SCHEMA, "number_reads",
                         config.AI_EFFORT_IMAGE, 16000)
    out = {}
    for r in data.get("reads", []):
        try:
            out[int(r["index"])] = str(r.get("text", ""))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def interpret_command(command: str, items: list[dict], currency: str) -> dict:
    lines = "\n".join(
        f"{it['id']} | p{it['page']} | c{it['column']} | {it['label']} | {it['value']}" for it in items)
    text = (f"Currency of the list: {currency or 'unknown'}\n"
            f"Prices (id | page | column | row label | current value):\n{lines}\n\n"
            f"Instruction: «{command}»")
    return with_failover(lambda p: p.call(COMMAND_SYSTEM, text, [], COMMAND_SCHEMA, "price_plan",
                                          config.AI_EFFORT_COMMAND, 32000))


def transcribe_tables(img: Image.Image, prices: list[tuple[str, str, tuple[int, int, int, int]]]) -> dict:
    W, H = img.size
    lines = "\n".join(f"{pid} | {t} | {b[0]},{b[1]},{b[2]},{b[3]}" for pid, t, b in prices)
    text = (f"Image size: {W}x{H} pixels.\nPrice cells on this page (id | price as shown | x0,y0,x1,y1):\n"
            f"{lines or '(none)'}\n\nTranscribe the table(s).")
    return with_failover(lambda p: p.call(TABLE_SYSTEM, text, [(img, "JPEG")], TABLE_SCHEMA, "price_tables",
                                          config.AI_EFFORT_TABLE, 64000))
