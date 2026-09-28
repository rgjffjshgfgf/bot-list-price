"""Telegram bot: send a price list (PDF or photo), say how prices should change,
then pick the output format (PDF / Excel / image)."""
from __future__ import annotations

import asyncio
import io
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaDocument, Message, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler,
                          filters)

import pymupdf

from pricebot import ai, brain, commands, config, pipeline
from pricebot.commands import Plan, Rule
from pricebot.models import Analysis
from pricebot.numfmt import fmt_plain
from pricebot.outputs import Exporter, original_format

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "telegram.ext"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("bot")

WORK_ROOT = config.WORK_DIR
MAX_DOWNLOAD = 20 * 1024 * 1024   # Telegram bot API download limit
FORMAT_NAMES = {"pdf": "PDF", "xlsx": "Excel", "image": "عکس"}
FORMAT_ICONS = {"pdf": "📄", "xlsx": "📊", "image": "🖼"}

HELP = (
    "سلام! 👋 من قیمت‌های لیست شما را تغییر می‌دهم و همان لیست را با همان ظاهر، فقط با قیمت‌های جدید تحویل می‌دهم.\n\n"
    "۱) فایل لیست قیمت را بفرست: PDF یا عکس (عکس را ترجیحاً به صورت «فایل» بفرست تا کیفیتش کم نشود).\n"
    "۲) بگو چه تغییری بدهم، مثلاً:\n"
    "• ۱۰ درصد افزایش\n"
    "• ۵ درصد کاهش\n"
    "• ۱۵ درصد افزایش و رند به هزار\n"
    "• فقط پرایدها ۲۰ درصد، بقیه ۱۰ درصد\n"
    "• ردیف ۱ تا ۱۰ رو ۵۰۰ هزار تومن اضافه کن\n"
    "۳) فرمت خروجی را انتخاب کن: PDF، Excel یا عکس.\n\n"
    "دستور را می‌توانی در کپشن فایل هم بنویسی. چند فایل پشت سر هم هم قبول است؛ دستور روی همه اعمال می‌شود.\n"
    "هر دستور روی فایل اصلی اعمال می‌شود (نه روی خروجی قبلی).\n\n"
    "🧠 ربات هوش مصنوعی خودش را دارد: قالب‌های جدید را با کمک Gemini (طرح رایگان) یاد می‌گیرد و بعد از "
    "چند بار تأیید، همان قالب را بدون Gemini و بدون مصرف سهمیه پردازش می‌کند. زیر عکس پیش‌نمایش با 👍 / 👎 "
    "بگو تشخیص درست بود یا نه تا سریع‌تر یاد بگیرد. وضعیت یادگیری: /brain\n\n"
    "/help راهنما   /brain هوش ربات   /usage سهمیه Gemini   /test آزمایش اتصال Gemini   /status وضعیت   "
    "/reset شروع دوباره   /id شناسه شما"
)


@dataclass
class Pending:
    """A computed price change waiting for (or already given) an output format."""
    key: str
    plan: Plan
    values: list[dict[str, Decimal]]
    exporter: Exporter
    sent: set[str] = field(default_factory=set)


@dataclass
class Session:
    workdir: Path
    analyses: list[Analysis] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    pending: Pending | None = None

    def expired(self) -> bool:
        return time.time() - self.created > config.SESSION_TTL_MINUTES * 60

    @property
    def items(self):
        return [it for a in self.analyses for it in a.items]


SESSIONS: dict[int, Session] = {}
CHAT_LOCKS: dict[int, asyncio.Lock] = {}


def _chat_lock(chat_id: int) -> asyncio.Lock:
    return CHAT_LOCKS.setdefault(chat_id, asyncio.Lock())


def _drop_session(chat_id: int) -> None:
    s = SESSIONS.pop(chat_id, None)
    if s:
        shutil.rmtree(s.workdir, ignore_errors=True)


def _purge_expired() -> None:
    for chat_id, s in list(SESSIONS.items()):
        if s.expired():
            _drop_session(chat_id)


def _allowed(update: Update) -> bool:
    user = update.effective_user
    return not config.ALLOWED_USER_IDS or (user is not None and user.id in config.ALLOWED_USER_IDS)


async def _deny(update: Update) -> None:
    uid = update.effective_user.id if update.effective_user else "?"
    await update.effective_message.reply_text(f"⛔️ شما اجازه استفاده از این ربات را ندارید.\nشناسه شما: {uid}")


def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _ai_error_text(exc: Exception) -> str:
    if isinstance(exc, ai.QuotaExceeded):
        return ("⛔️ سهمیه رایگان امروز Gemini تمام شد.\n"
                f"⏰ تمدید: {ai.reset_text()}\n"
                "تا آن موقع PDFهای متنی با تشخیص ساده کار می‌کنند؛ عکس‌ها باید تا تمدید صبر کنند.")
    low = str(exc).lower()
    if "authentication" in low or "api key" in low:
        return "کلید Gemini نامعتبر است (GEMINI_API_KEY را بررسی کن)."
    if "region" in low:
        return "Gemini در منطقه سرور در دسترس نیست."
    return ("سرویس هوش مصنوعی پاسخ نداد. چند لحظه بعد دوباره امتحان کن.\n"
            f"جزئیات: {ai.short_error(exc)}\n"
            "برای آزمایش اتصال: /test")


async def _safe_edit(message: Message, text: str, **kwargs) -> None:
    try:
        await message.edit_text(text, **kwargs)
    except BadRequest as exc:
        if "not modified" not in str(exc).lower():
            log.debug("edit failed: %s", exc)


# ---------------------------------------------------------------- keyboards --

def _quick_keyboard() -> InlineKeyboardMarkup:
    row1 = [InlineKeyboardButton(f"+{_fa(p)}٪", callback_data=f"pct:{p}") for p in (5, 10, 15, 20)]
    row2 = [InlineKeyboardButton(f"−{_fa(p)}٪", callback_data=f"pct:-{p}") for p in (5, 10)]
    return InlineKeyboardMarkup([row1, row2])


def _format_row(session: Session, key: str, exclude: set[str] = frozenset()) -> list[InlineKeyboardButton]:
    originals = {original_format(a) for a in session.analyses}
    row = []
    for fmt in ("pdf", "xlsx", "image"):
        if fmt in exclude:
            continue
        mark = " (اصلی)" if fmt in originals and len(originals) == 1 else ""
        row.append(InlineKeyboardButton(f"{FORMAT_ICONS[fmt]} {FORMAT_NAMES[fmt]}{mark}", callback_data=f"out:{fmt}:{key}"))
    return row


def _output_keyboard(session: Session, pending: Pending) -> InlineKeyboardMarkup:
    k = pending.key
    return InlineKeyboardMarkup([
        _format_row(session, k),
        [InlineKeyboardButton("📦 همه فرمت‌ها", callback_data=f"out:all:{k}")],
        [InlineKeyboardButton("رند ۱٬۰۰۰", callback_data=f"round:1000:{k}"),
         InlineKeyboardButton("رند ۱۰٬۰۰۰", callback_data=f"round:10000:{k}"),
         InlineKeyboardButton("رند ۱۰۰٬۰۰۰", callback_data=f"round:100000:{k}")],
        [InlineKeyboardButton("بدون رند", callback_data=f"round:0:{k}")],
    ])


def _more_keyboard(session: Session, pending: Pending) -> InlineKeyboardMarkup | None:
    row = _format_row(session, pending.key, exclude=pending.sent)
    return InlineKeyboardMarkup([row]) if row else None


# ----------------------------------------------------------------- commands --

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    extra = "" if ai.enabled() else (
        "\n\n⚠️ کلید Gemini تنظیم نشده؛ فقط PDFهای متنی با تشخیص ساده پشتیبانی می‌شوند.")
    await update.effective_message.reply_text(HELP + extra)


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(f"شناسه عددی شما: {update.effective_user.id}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    sess = SESSIONS.get(update.effective_chat.id)
    files = (f"{_fa(len(sess.analyses))} فایل، {_fa(len(sess.items))} قیمت"
             if sess and not sess.expired() else "فایلی باز نیست")
    await update.effective_message.reply_text(
        f"🤖 Gemini: {ai.describe()}\n🧠 {_brain_short()}\n📂 جلسه فعلی: {files}\n\n{ai.usage_report()}")


async def cmd_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    await update.effective_message.reply_text(ai.usage_report())


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    status = await update.effective_message.reply_text("🩺 در حال آزمایش اتصال به Gemini…")
    report = await asyncio.to_thread(ai.self_test)
    await _safe_edit(status, report)


def _brain_short() -> str:
    if not brain.enabled():
        return "هوش ربات: خاموش"
    with brain.BRAIN.lock:
        tpls = list(brain.BRAIN.templates)
    ready = sum(1 for t in tpls if t.get("streak", 0) >= brain.BRAIN.trust_after)
    return f"هوش ربات: {_fa(len(tpls))} قالب یادگرفته، {_fa(ready)} مستقل — جزئیات: /brain"


async def cmd_brain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗑 پاک کردن همه یادگرفته‌ها", callback_data="brainreset:ask")]])
    await update.effective_message.reply_text(brain.report(), reply_markup=kb if brain.enabled() else None)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    async with _chat_lock(update.effective_chat.id):
        _drop_session(update.effective_chat.id)
    await update.effective_message.reply_text("🗑 پاک شد. فایل جدید را بفرست.")


# -------------------------------------------------------------------- files --

async def on_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    msg = update.effective_message
    chat_id = update.effective_chat.id
    if msg.photo:
        tg_file = msg.photo[-1]
        filename = f"photo_{msg.message_id}.jpg"
        size = tg_file.file_size or 0
    else:
        doc = msg.document
        filename = doc.file_name or f"file_{msg.message_id}"
        mime = (doc.mime_type or "").lower()
        if not (mime == "application/pdf" or mime.startswith("image/")
                or filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))):
            await msg.reply_text("این نوع فایل پشتیبانی نمی‌شود. لطفاً PDF یا عکس بفرست.")
            return
        tg_file = doc
        size = doc.file_size or 0
    if size > MAX_DOWNLOAD:
        await msg.reply_text("حجم فایل بیشتر از ۲۰ مگابایت است (محدودیت تلگرام برای ربات‌ها). لطفاً فایل کوچک‌تری بفرست.")
        return

    async with _chat_lock(chat_id):
        _purge_expired()
        session = SESSIONS.get(chat_id)
        if session is None or session.pending is not None or session.expired():
            _drop_session(chat_id)
            workdir = WORK_ROOT / f"{chat_id}_{uuid.uuid4().hex[:8]}"
            workdir.mkdir(parents=True, exist_ok=True)
            session = SESSIONS[chat_id] = Session(workdir)

        status = await msg.reply_text("📥 فایل دریافت شد، در حال بررسی…")
        fdir = session.workdir / f"f{len(session.analyses) + 1}"
        fdir.mkdir(parents=True, exist_ok=True)
        src = fdir / ("source" + (Path(filename).suffix or ""))
        try:
            f = await context.bot.get_file(tg_file.file_id)
            await f.download_to_drive(src)
        except Exception:  # noqa: BLE001
            log.exception("download failed")
            await _safe_edit(status, "❌ دانلود فایل از تلگرام انجام نشد. دوباره بفرست.")
            return

        loop = asyncio.get_running_loop()
        last_edit = [0.0]

        def progress(done: int, total: int) -> None:
            if total <= 1 or time.time() - last_edit[0] < 3:
                return
            last_edit[0] = time.time()
            text = f"🔎 در حال بررسی… صفحه {_fa(done)} از {_fa(total)}"
            asyncio.run_coroutine_threadsafe(_safe_edit(status, text), loop)

        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        await _quota_warning(msg, src)
        try:
            analysis = await asyncio.to_thread(pipeline.analyze, src, filename, fdir / "work", progress)
        except pipeline.UserError as exc:
            await _safe_edit(status, f"❌ {exc}")
            return
        except ai.AIError as exc:
            log.warning("AI error: %s", exc)
            await _safe_edit(status, "❌ " + _ai_error_text(exc))
            return
        except Exception:  # noqa: BLE001
            log.exception("analysis failed")
            await _safe_edit(status, "❌ در بررسی فایل خطایی رخ داد.")
            return
        session.analyses.append(analysis)
        await _report_analysis(status, analysis, session)
        await _send_preview(context, chat_id, analysis)

        caption = (msg.caption or "").strip()
        if caption and analysis.items:
            await _handle_command_text(update, context, session, caption)


async def _quota_warning(msg: Message, src: Path) -> None:
    """Tell the user up front when today's free quota will not cover this file."""
    if not ai.enabled():
        return
    kind = pipeline.detect_kind(src)
    pages = 1
    if kind == "pdf":
        try:
            with pymupdf.open(src) as doc:
                pages = doc.page_count
        except Exception:  # noqa: BLE001 - the analysis step reports broken files
            return
    need = ai.estimate_requests(kind or "pdf", pages)
    left = ai.remaining_total()
    if left >= need:
        return
    if brain.enabled():
        if left == 0:
            await msg.reply_text("⚠️ سهمیه رایگان امروز Gemini تمام شده؛ این فایل را هوش خود ربات بررسی می‌کند "
                                 f"(تمدید سهمیه: {ai.reset_text()}).")
        return
    if left == 0:
        text = ("⚠️ سهمیه رایگان امروز Gemini تمام شده.\n"
                + ("این PDF با تشخیص ساده (بدون هوش مصنوعی) بررسی می‌شود." if kind == "pdf"
                   else f"عکس‌ها بعد از تمدید سهمیه قابل بررسی‌اند: {ai.reset_text()}"))
    else:
        text = (f"⚠️ این فایل حدود {_fa(need)} درخواست لازم دارد ولی از سهمیه رایگان امروز فقط "
                f"{_fa(left)} درخواست مانده؛ صفحه‌هایی که به سهمیه نرسند با تشخیص ساده بررسی می‌شوند.")
    await msg.reply_text(text)


def _brain_notes(analysis: Analysis) -> list[str]:
    notes: list[str] = []
    for page in sorted(analysis.brain):
        note = brain.page_note(analysis.brain[page])
        if note and note not in notes:
            notes.append(note)
    local = sum(1 for i in analysis.brain.values() if i.get("how") == "local")
    if local and analysis.page_count > 1:
        notes.append(f"🧠 {_fa(local)} از {_fa(analysis.page_count)} صفحه بدون Gemini پردازش شد.")
    return notes[:3]


async def _report_analysis(status: Message, analysis: Analysis, session: Session) -> None:
    n = len(analysis.items)
    if n == 0:
        text = "⚠️ هیچ قیمتی در این فایل پیدا نشد."
        if analysis.warnings:
            text += "\n" + "\n".join(f"• {w}" for w in analysis.warnings[:5])
        await _safe_edit(status, text)
        return
    pages = len({it.page for it in analysis.items})
    lines = [f"✅ {_fa(n)} قیمت" + (f" در {_fa(pages)} صفحه" if analysis.kind == "pdf" else "") + " پیدا شد."]
    if analysis.currency:
        lines.append(f"واحد: {analysis.currency}")
    for it in analysis.items[:4]:
        label = f"{it.label}: " if it.label else ""
        lines.append(f"• {label}{it.text}")
    if n > 4:
        lines.append("• …")
    notes = _brain_notes(analysis)
    if notes:
        lines.append("")
        lines += notes
    if analysis.warnings:
        lines.append("")
        lines += [f"⚠️ {w}" for w in analysis.warnings[:4]]
    if len(session.analyses) > 1:
        lines.append(f"\n📚 {_fa(len(session.analyses))} فایل آماده ({_fa(len(session.items))} قیمت). "
                     "دستور روی همه اعمال می‌شود.")
    if ai.enabled():
        lines.append("\n" + ai.usage_line())
    lines.append("\nحالا بگو چه تغییری بدم (مثلاً «۱۰ درصد افزایش») یا یکی از دکمه‌ها را بزن:")
    await _safe_edit(status, "\n".join(lines), reply_markup=_quick_keyboard())


def _feedback_keyboard(analysis: Analysis) -> InlineKeyboardMarkup | None:
    if not brain.enabled() or not analysis.brain:
        return None
    aid = analysis.cache.setdefault("aid", uuid.uuid4().hex[:8])
    return InlineKeyboardMarkup([[InlineKeyboardButton("👍 تشخیص درست است", callback_data=f"fb:ok:{aid}"),
                                  InlineKeyboardButton("👎 اشتباه دارد", callback_data=f"fb:bad:{aid}")]])


async def _send_preview(context: ContextTypes.DEFAULT_TYPE, chat_id: int, analysis: Analysis) -> None:
    if not analysis.items:
        return
    page = analysis.items[0].page
    try:
        im = await asyncio.to_thread(pipeline.preview, analysis, page)
    except Exception:  # noqa: BLE001
        log.exception("preview failed")
        return
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    buf.seek(0)
    cap = "قیمت‌های پیدا شده با کادر سبز مشخص شده‌اند" + (f" (صفحه {_fa(page + 1)})" if analysis.kind == "pdf" else "")
    await context.bot.send_photo(chat_id, buf, caption=cap, reply_markup=_feedback_keyboard(analysis))


# --------------------------------------------------------- price commands ---

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    chat_id = update.effective_chat.id
    async with _chat_lock(chat_id):   # waits for a file that is still being analysed
        session = SESSIONS.get(chat_id)
        if session is None or session.expired() or not session.items:
            await update.effective_message.reply_text(
                "اول فایل لیست قیمت (PDF یا عکس) را بفرست، بعد بگو چه تغییری بدهم.\n/help")
            return
        await _handle_command_text(update, context, session, update.effective_message.text or "")


async def _handle_command_text(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session,
                               text: str) -> None:
    msg = update.effective_message
    local = commands.parse_local(text)
    if local.plan is not None:
        return await _prepare(update, context, session, local.plan)
    if local.ask_direction is not None:
        v = fmt_plain(local.ask_direction)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"افزایش {_fa(v)}٪", callback_data=f"pct:{v}"),
            InlineKeyboardButton(f"کاهش {_fa(v)}٪", callback_data=f"pct:-{v}"),
        ]])
        await msg.reply_text("افزایش یا کاهش؟", reply_markup=kb)
        return
    if not ai.enabled():
        await msg.reply_text("متوجه نشدم 🤔 مثلاً بنویس: «۱۰ درصد افزایش» یا «۵ درصد کاهش و رند به هزار».")
        return
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    thinking = await msg.reply_text("🤔 در حال فهمیدن دستور…")
    listing = []
    for fi, a in enumerate(session.analyses):
        for it in a.items:
            listing.append({"id": f"f{fi + 1}-{it.id}", "page": it.page + 1, "column": it.column or "-",
                            "label": it.label or "-", "value": fmt_plain(it.value)})
    currency = next((a.currency for a in session.analyses if a.currency), "")
    try:
        data = await asyncio.to_thread(ai.interpret_command, text, listing, currency)
    except Exception as exc:  # noqa: BLE001
        log.warning("command AI failed: %s", exc)
        detail = _ai_error_text(exc) if isinstance(exc, ai.AIError) else ""
        await _safe_edit(thinking, "❌ نتوانستم دستور را بفهمم. ساده‌تر بنویس، مثلاً «۱۰ درصد افزایش»."
                         + (f"\n\n{detail}" if detail else ""))
        return
    status = data.get("status")
    if status == "clarify":
        await _safe_edit(thinking, "❓ " + (data.get("question") or "لطفاً دقیق‌تر بگو."))
        return
    if status != "ok":
        await _safe_edit(thinking, "این پیام دستور تغییر قیمت نبود. مثلاً بنویس: «۱۰ درصد افزایش».")
        return
    plan = commands.plan_from_ai(data, text, {x["id"] for x in listing})
    if not plan.rules:
        await _safe_edit(thinking, "❓ هیچ قیمتی با این دستور انتخاب نشد. دقیق‌تر بگو.")
        return
    await _safe_edit(thinking, f"🧠 {plan.summary}" if plan.summary else "🧠 فهمیدم.")
    await _prepare(update, context, session, plan)


def _label(plan: Plan) -> str:
    if len(plan.rules) == 1 and plan.rules[0][0] is None and plan.rules[0][1].op == "percent":
        v = plan.rules[0][1].value
        label = f"{'+' if v >= 0 else '-'}{fmt_plain(abs(v))}%"
    else:
        label = "updated"
    if plan.round_step > 0:
        label += f" r{fmt_plain(plan.round_step).replace(',', '')}"
    return label


def _compute(session: Session, plan: Plan, key: str) -> Pending:
    namespaced = any(scope is not None for scope, _ in plan.rules)
    values = [commands.compute(plan, a.items, f"f{fi + 1}-" if namespaced else "")
              for fi, a in enumerate(session.analyses)]
    out_dir = session.workdir / f"out_{key}"
    exporter = Exporter([(a, v) for a, v in zip(session.analyses, values) if v], out_dir, _label(plan),
                        plan.summary or _label(plan))
    return Pending(key, plan, values, exporter)


def _summary_text(session: Session, pending: Pending) -> str:
    plan = pending.plan
    changed = sum(len(v) for v in pending.values)
    lines = [f"✅ {_fa(changed)} قیمت محاسبه شد — {plan.summary or _label(plan)}"]
    if plan.round_step:
        lines.append("رند: " + ("هم‌دقت قیمت اصلی" if plan.round_step < 0 else _fa(fmt_plain(plan.round_step))))
    shown = 0
    for a, values in zip(session.analyses, pending.values):
        for it in a.items:
            if it.id in values and shown < 5:
                label = f"{it.label}: " if it.label else ""
                lines.append(f"• {label}{it.fmt.format(values[it.id])}  (قبلی: {it.text})")
                shown += 1
    lines.append("\n📦 خروجی رو با چه فرمتی می‌خوای؟")
    return "\n".join(lines)


async def _prepare(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session, plan: Plan) -> None:
    """Compute the new prices, show a summary and ask for the output format."""
    msg = update.effective_message
    pending = _compute(session, plan, uuid.uuid4().hex[:8])
    if not any(pending.values):
        await msg.reply_text("هیچ قیمتی تغییر نکرد (شاید دستور روی هیچ ردیفی صدق نکرد).")
        return
    _replace_pending(session, pending)
    await msg.reply_text(_summary_text(session, pending), reply_markup=_output_keyboard(session, pending))


def _replace_pending(session: Session, pending: Pending) -> None:
    old = session.pending
    if old is not None and old.exporter.out_dir != pending.exporter.out_dir:
        shutil.rmtree(old.exporter.out_dir, ignore_errors=True)
    session.pending = pending


async def _deliver(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session, pending: Pending,
                   formats: list[str]) -> None:
    chat_id = update.effective_chat.id
    msg = update.effective_message
    for fmt in formats:
        status = await msg.reply_text(f"⏳ در حال ساخت خروجی {FORMAT_NAMES[fmt]}…")
        await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
        try:
            paths = await asyncio.to_thread(pending.exporter.build, fmt)
        except ai.AIError as exc:
            log.warning("export %s failed: %s", fmt, exc)
            await _safe_edit(status, "❌ " + _ai_error_text(exc))
            continue
        except Exception:  # noqa: BLE001
            log.exception("export %s failed", fmt)
            await _safe_edit(status, f"❌ ساخت خروجی {FORMAT_NAMES[fmt]} با خطا روبه‌رو شد.")
            continue
        await _send_files(context, chat_id, paths)
        pending.sent.add(fmt)
        try:
            await status.delete()
        except Exception:  # noqa: BLE001
            pass
    warnings = list(dict.fromkeys(pending.exporter.warnings))
    text = "✅ آماده شد." + (f"\n{ai.usage_line()}" if ai.enabled() else "")
    if warnings:
        text += "\n" + "\n".join(f"⚠️ {w}" for w in warnings[:4])
    more = _more_keyboard(session, pending)
    if more:
        text += "\n\nفرمت دیگری هم می‌خوای؟ یا دستور جدید بنویس."
    await msg.reply_text(text, reply_markup=more)


async def _send_files(context: ContextTypes.DEFAULT_TYPE, chat_id: int, paths: list[Path]) -> None:
    if len(paths) == 1:
        with paths[0].open("rb") as fh:
            await context.bot.send_document(chat_id, fh, filename=paths[0].name)
        return
    for start in range(0, len(paths), 10):   # Telegram albums hold up to 10 files
        chunk = paths[start:start + 10]
        handles = [p.open("rb") for p in chunk]
        try:
            media = [InputMediaDocument(h, filename=p.name) for h, p in zip(handles, chunk)]
            await context.bot.send_media_group(chat_id, media)
        finally:
            for h in handles:
                h.close()


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not _allowed(update):
        await q.answer("دسترسی ندارید", show_alert=True)
        return
    await q.answer()
    if (q.data or "").startswith("brainreset:"):
        return await _on_brain_reset(q)
    async with _chat_lock(update.effective_chat.id):
        session = SESSIONS.get(update.effective_chat.id)
        if (q.data or "").startswith("fb:"):
            return await _on_feedback(update, context, session)
        if session is None or session.expired() or not session.items:
            await q.message.reply_text("جلسه قبلی تمام شده. فایل را دوباره بفرست.")
            return
        parts = (q.data or "").split(":")
        kind = parts[0]
        if kind == "pct":
            rule = Rule("percent", Decimal(parts[1]))
            await _prepare(update, context, session, Plan([(None, rule)], summary=rule.describe(), command=q.data))
            return
        pending = session.pending
        if pending is None or len(parts) < 3 or parts[2] != pending.key:
            await q.message.reply_text("این دکمه مربوط به یک دستور قدیمی است؛ دستور را دوباره بفرست.")
            return
        if kind == "round":
            new = _compute(session, pending.plan.with_rounding(Decimal(parts[1])), uuid.uuid4().hex[:8])
            _replace_pending(session, new)
            await _safe_edit(q.message, _summary_text(session, new), reply_markup=_output_keyboard(session, new))
        elif kind == "out":
            formats = ["pdf", "xlsx", "image"] if parts[1] == "all" else [parts[1]]
            await _deliver(update, context, session, pending, formats)


async def _on_brain_reset(q) -> None:
    if q.data == "brainreset:ask":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("بله، همه را پاک کن", callback_data="brainreset:yes"),
                                    InlineKeyboardButton("نه", callback_data="brainreset:no")]])
        await q.message.reply_text("مطمئنی؟ هرچه ربات تا الان یاد گرفته پاک می‌شود و از صفر شروع می‌کند.",
                                   reply_markup=kb)
    elif q.data == "brainreset:yes":
        await asyncio.to_thread(brain.BRAIN.reset)
        await _safe_edit(q.message, "🗑 حافظه هوش ربات پاک شد.")
    else:
        await _safe_edit(q.message, "لغو شد.")


def _learn_from_user(analysis: Analysis, good: bool) -> None:
    """👍: every page becomes a checked example. 👎: formats lose their trust."""
    ids: set[str] = set()
    wrong_models: list[str] = []
    for info in analysis.brain.values():
        tpl = info.get("format") or {}
        if good and not info.get("taught") and info.get("doc") is not None:
            brain.BRAIN.learn(info["doc"], set(info.get("selected", set())), info.get("decision"), "user")
            info["taught"] = True
            continue
        if tpl.get("id"):
            ids.add(tpl["id"])
        if not good and info.get("how") in ("local", "fallback") and info.get("by") == "model":
            wrong_models.append(info["doc"].source)
    brain.BRAIN.feedback(ids, good, wrong_models)


async def _on_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session | None) -> None:
    q = update.callback_query
    _, verdict, aid = (q.data.split(":") + ["", ""])[:3]
    analysis = next((a for a in session.analyses if a.cache.get("aid") == aid), None) if session else None
    if analysis is None:
        await q.message.reply_text("این پیش‌نمایش مربوط به فایل قدیمی است.")
        return
    if analysis.cache.get("feedback"):
        return
    analysis.cache["feedback"] = verdict
    good = verdict == "ok"
    await asyncio.to_thread(_learn_from_user, analysis, good)
    try:
        await q.message.edit_reply_markup(None)
    except BadRequest:
        pass
    if good:
        await q.message.reply_text("👍 ممنون، یاد گرفتم. دفعه بعد این قالب را بهتر می‌شناسم.")
        return
    was_local = any(i.get("how") in ("local", "fallback") for i in analysis.brain.values())
    if not (was_local and ai.enabled()):
        await q.message.reply_text("👎 ثبت شد؛ این قالب دوباره با احتیاط بررسی می‌شود تا درست یاد بگیرد."
                                   + ("" if ai.enabled() else "\nبرای بررسی دوباره، کلید Gemini لازم است."))
        return
    # the brain got it wrong: ask the teacher now, and learn from its answer
    status = await q.message.reply_text("🔁 با Gemini دوباره بررسی می‌کنم و ربات از جوابش یاد می‌گیرد…")
    idx = session.analyses.index(analysis)
    try:
        fresh = await asyncio.to_thread(pipeline.analyze, analysis.source, analysis.filename,
                                        analysis.workdir.parent / f"work_{uuid.uuid4().hex[:6]}", None, True)
    except Exception as exc:  # noqa: BLE001
        log.warning("re-check failed: %s", exc)
        await _safe_edit(status, "❌ " + (_ai_error_text(exc) if isinstance(exc, ai.AIError) else "بررسی دوباره انجام نشد."))
        return
    session.analyses[idx] = fresh
    session.pending = None
    await _report_analysis(status, fresh, session)
    await _send_preview(context, update.effective_chat.id, fresh)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("unhandled error", exc_info=context.error)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    if not ai.enabled():
        log.warning("no AI key set: images/scans will not work, PDFs use simple detection")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    app = (Application.builder().token(config.TELEGRAM_BOT_TOKEN).concurrent_updates(True)
           .read_timeout(60).write_timeout(180).connect_timeout(30).media_write_timeout(180).build())
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler(["usage", "quota"], cmd_usage))
    app.add_handler(CommandHandler(["test", "diag"], cmd_test))
    app.add_handler(CommandHandler(["brain", "learn", "ai"], cmd_brain))
    app.add_handler(CommandHandler(["reset", "cancel"], cmd_reset))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, on_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_error_handler(on_error)
    if brain.enabled() and not brain.ocr.available():
        log.warning("Tesseract OCR not found: the brain can learn PDFs only, photos still need Gemini")
    log.info("bot started — AI: %s — brain: %s (%s)", ai.describe(), brain.mode(), config.BRAIN_DIR)
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
