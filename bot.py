"""Telegram bot: send a price list (PDF or photo), say how prices should change,
then pick the output format (PDF / Excel / image / the Arizon template)."""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaDocument, Message, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler,
                          filters)

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps

from pricebot import ai, brain, commands, config, pipeline, shot
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
FORMAT_NAMES = {"pdf": "PDF", "xlsx": "Excel", "image": "عکس", "arizon": "قالب آریزون (PDF)",
                "arizon_image": "قالب آریزون (عکس)"}
FORMAT_ICONS = {"pdf": "📄", "xlsx": "📊", "image": "🖼", "arizon": "✨", "arizon_image": "✨"}
ARIZON_FORMATS = ("arizon", "arizon_image")
SUBTITLES_FILE = config.DATA_DIR / "arizon_subtitles.json"   # each chat's last lines under the Arizon title
RECENT_SUBTITLES = 3
# a typed line that may be a price command rather than a line for the Arizon header
_COMMANDISH = re.compile(r"افزایش|کاهش|درصد|٪|%|(?<!\S)رند(?!\S)|اضافه\s*کن|کم\s*کن|بکن|ردیف\s*[\d۰-۹]")

HELP = (
    "سلام! 👋 من قیمت‌های لیست شما را تغییر می‌دهم و همان لیست را با همان ظاهر، فقط با قیمت‌های جدید تحویل می‌دهم.\n\n"
    "۱) فایل لیست قیمت را بفرست: PDF یا عکس (عکس را ترجیحاً به صورت «فایل» بفرست تا کیفیتش کم نشود).\n"
    "۲) بگو چه تغییری بدهم، مثلاً:\n"
    "• ۱۰ درصد افزایش\n"
    "• ۵ درصد کاهش\n"
    "• ۱۵ درصد افزایش و رند به هزار\n"
    "• فقط پرایدها ۲۰ درصد، بقیه ۱۰ درصد\n"
    "• گروه پژو ۱۵ درصد افزایش، بقیه ۵ درصد (اگر لیست گروه‌بندی دارد)\n"
    "• ردیف ۱ تا ۱۰ رو ۵۰۰ هزار تومن اضافه کن\n"
    "۳) فرمت خروجی را انتخاب کن: PDF، Excel یا عکس، یا «✨ قالب آریزون»: لیست با قیمت‌های جدید "
    "در طراحی اختصاصی آریزون. برای قالب آریزون می‌پرسم زیر «لیست قیمت محصولات» چه نوشته شود "
    "(مثلاً اسم فروشگاه)، یا بدون متن اضافه.\n\n"
    "دستور را می‌توانی در کپشن فایل هم بنویسی. چند فایل پشت سر هم هم قبول است؛ دستور روی همه اعمال می‌شود.\n"
    "هر دستور روی فایل اصلی اعمال می‌شود (نه روی خروجی قبلی).\n\n"
    "🧠 ربات هوش مصنوعی خودش را دارد: قالب‌های جدید را با کمک Gemini (طرح رایگان) یاد می‌گیرد و بعد از "
    "چند بار تأیید، همان قالب را بدون Gemini و بدون مصرف سهمیه پردازش می‌کند. زیر عکس پیش‌نمایش با 👍 / 👎 "
    "بگو تشخیص درست بود یا نه تا سریع‌تر یاد بگیرد. وضعیت یادگیری: /brain\n\n"
    "🖼 تغییر عدد بازدید در اسکرین‌شات تلگرام: /shot\n\n"
    "/help راهنما   /shot تغییر بازدید اسکرین‌شات   /brain هوش ربات   /backup بکاپ   /restore بازیابی   /usage سهمیه Gemini   "
    "/test آزمایش اتصال Gemini   /status وضعیت   "
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
class SubtitleWait:
    """Arizon formats waiting for the line under the template's title."""
    key: str                          # the price change (Pending) they belong to
    formats: list[str]
    offered: list[str]                # the chat's recent lines, shown as buttons
    question: Message | None = None
    typed: str = ""                   # a typed line that may also be a price command: which one?
    raw: str = ""


@dataclass
class Session:
    workdir: Path
    analyses: list[Analysis] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    pending: Pending | None = None
    sub_wait: SubtitleWait | None = None

    def expired(self) -> bool:
        return time.time() - self.created > config.SESSION_TTL_MINUTES * 60

    @property
    def items(self):
        return [it for a in self.analyses for it in a.items]


SESSIONS: dict[int, Session] = {}
CHAT_LOCKS: dict[int, asyncio.Lock] = {}


@dataclass
class ShotSession:
    """A screenshot that came without numbers (see /shot); its question message carries the buttons."""
    analysis: shot.Analysis
    source: Message                                          # the user's screenshot; the result answers it
    created: float = field(default_factory=time.time)
    picks: dict[int, str] = field(default_factory=dict)     # numbers chosen with the buttons, per view count

    def expired(self) -> bool:
        return time.time() - self.created > config.SESSION_TTL_MINUTES * 60


SHOTS: dict[int, dict[int, ShotSession]] = {}   # chat -> question message id -> screenshot
SHOT_LAST: dict[int, int] = {}                  # chat -> the newest question (typed numbers go there)
SHOT_MODE_FILE = config.DATA_DIR / "shot_mode.json"
SHOT_WORDS = ("بازدید", "ویو", "view", "اسکرین")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


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
    for chat in SHOTS.values():
        for key, t in list(chat.items()):
            if t.expired():
                chat.pop(key, None)


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


def _arizon_row(key: str, exclude: set[str] = frozenset()) -> list[InlineKeyboardButton]:
    """«receive it as the Arizon template»: the list rebuilt in Arizon's own design."""
    labels = {"arizon": "✨ قالب آریزون (PDF)", "arizon_image": "✨ قالب آریزون (عکس)"}
    return [InlineKeyboardButton(labels[f], callback_data=f"out:{f}:{key}") for f in ARIZON_FORMATS if f not in exclude]


def _output_keyboard(session: Session, pending: Pending) -> InlineKeyboardMarkup:
    k = pending.key
    return InlineKeyboardMarkup([
        _format_row(session, k),
        _arizon_row(k),
        [InlineKeyboardButton("📦 همه فرمت‌ها", callback_data=f"out:all:{k}")],
        [InlineKeyboardButton("رند ۱٬۰۰۰", callback_data=f"round:1000:{k}"),
         InlineKeyboardButton("رند ۱۰٬۰۰۰", callback_data=f"round:10000:{k}"),
         InlineKeyboardButton("رند ۱۰۰٬۰۰۰", callback_data=f"round:100000:{k}")],
        [InlineKeyboardButton("بدون رند", callback_data=f"round:0:{k}")],
    ])


def _more_keyboard(session: Session, pending: Pending) -> InlineKeyboardMarkup | None:
    rows = [r for r in (_format_row(session, pending.key, exclude=pending.sent),
                        _arizon_row(pending.key, exclude=pending.sent)) if r]
    if any(f in pending.sent for f in ARIZON_FORMATS):
        rows.append([InlineKeyboardButton("✏️ قالب آریزون با متن دیگری زیر عنوان",
                                          callback_data=f"sub:ask:{pending.key}")])
    return InlineKeyboardMarkup(rows) if rows else None


# ----------------------------------------------------------------- commands --

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    extra = "" if ai.enabled() else (
        "\n\n⚠️ کلید Gemini تنظیم نشده؛ فقط PDFهای متنی با تشخیص ساده پشتیبانی می‌شوند.")
    chat_id = update.effective_chat.id
    command = ((update.effective_message.text or "").split() or ["/start"])[0].split("@")[0].lower()
    if command == "/help" and chat_id in SHOT_MODE:
        await update.effective_message.reply_text(SHOT_HELP + "\n\nبرای برگشتن به لیست قیمت: /start")
        return
    head = ""
    if command == "/start" and chat_id in SHOT_MODE:
        _leave_shot(chat_id)
        head = "🖼 حالت تغییر بازدید بسته شد.\n\n"
    await update.effective_message.reply_text(head + HELP + extra)


def _leave_shot(chat_id: int) -> None:
    _set_shot_mode(chat_id, False)
    SHOTS.pop(chat_id, None)
    SHOT_LAST.pop(chat_id, None)


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(f"شناسه عددی شما: {update.effective_user.id}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    sess = SESSIONS.get(update.effective_chat.id)
    files = (f"{_fa(len(sess.analyses))} فایل، {_fa(len(sess.items))} قیمت"
             if sess and not sess.expired() else "فایلی باز نیست")
    await update.effective_message.reply_text(
        f"🤖 Gemini: {ai.describe()}\n🧠 {_brain_short()}\n👁 {_ocr_status()}\n📂 جلسه فعلی: {files}\n\n"
        f"{ai.usage_report()}")


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


def _ocr_status() -> str:
    if not brain.ocr.available():
        return "OCR (Tesseract): نصب نیست — عکس‌ها فقط با Gemini"
    langs = sorted(brain.ocr._LANGS or [])
    fa = "fas" in langs
    return f"OCR (Tesseract): فعال — زبان‌ها: {', '.join(langs)}" + ("" if fa else " ⚠️ فارسی نصب نیست")


def _brain_short() -> str:
    if not brain.enabled():
        return "هوش ربات: خاموش"
    with brain.BRAIN.lock:
        tpls = list(brain.BRAIN.templates)
    ready = sum(1 for t in tpls if t.get("streak", 0) >= brain.BRAIN.trust_after)
    return f"هوش ربات: {_fa(len(tpls))} قالب یادگرفته، {_fa(ready)} مستقل — جزئیات: /brain"


def _brain_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💾 گرفتن بکاپ", callback_data="mem:backup"),
         InlineKeyboardButton("♻️ بازیابی بکاپ", callback_data="mem:restore")],
        [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data="mem:refresh"),
         InlineKeyboardButton("🗑 پاک کردن حافظه", callback_data="mem:reset")],
    ])


async def cmd_brain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    await update.effective_message.reply_text(brain.report(), reply_markup=_brain_keyboard() if brain.enabled() else None)


async def cmd_backup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    await _send_backup(context, update.effective_chat.id)


async def cmd_restore(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    await _ask_for_backup(update.effective_message, update.effective_chat.id)


# ------------------------------------------------------- memory backup ----

RESTORE_WAIT: dict[int, float] = {}        # chat -> time the bot started waiting for a backup file
RESTORE_FILES: dict[int, Path] = {}        # chat -> received backup waiting for "merge / replace"
BACKUP_DIR = config.DATA_DIR / "brain_backups"


def _fa_date(ts: float) -> str:
    return _fa(time.strftime("%Y/%m/%d %H:%M", time.localtime(ts)))


def _auto_backup(reason: str) -> Path:
    """A safety copy kept on the server before the memory is replaced or wiped (last 5 kept)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"auto-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.zip"
    brain.BRAIN.export_zip(path)
    for old in sorted(BACKUP_DIR.glob("auto-*.zip"))[:-5]:
        old.unlink(missing_ok=True)
    return path


async def _send_backup(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    if not brain.enabled():
        await context.bot.send_message(chat_id, "هوش ربات خاموش است (BRAIN_MODE=off).")
        return
    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
    path = WORK_ROOT / f"pricebot-brain-{time.strftime('%Y-%m-%d_%H-%M')}.zip"
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    info = await asyncio.to_thread(brain.BRAIN.export_zip, path)
    caption = (f"💾 بکاپ حافظه هوش ربات\n"
               f"📚 {_fa(info['formats'])} قالب ({_fa(info['trusted'])} مستقل) — {_fa(info['samples'])} نمونه آموزشی\n"
               f"🕒 {_fa_date(info['created'])}\n\n"
               "این فایل را نگه دار. برای برگرداندن: /restore یا /brain ← ♻️ بازیابی بکاپ")
    try:
        with path.open("rb") as fh:
            await context.bot.send_document(chat_id, fh, filename=path.name, caption=caption)
    finally:
        path.unlink(missing_ok=True)


async def _ask_for_backup(message: Message, chat_id: int) -> None:
    RESTORE_WAIT[chat_id] = time.time()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✖️ لغو", callback_data="mem:cancel")]])
    await message.reply_text("♻️ فایل بکاپ (zip) را همین‌جا بفرست.\n"
                             "همان فایلی که با «💾 گرفتن بکاپ» گرفته‌ای.", reply_markup=kb)


def _is_backup_upload(chat_id: int, filename: str) -> bool:
    name = filename.lower()
    waiting = time.time() - RESTORE_WAIT.get(chat_id, 0) < 600
    return name.endswith(".zip") and (waiting or name.startswith("pricebot-brain"))


async def _on_backup_file(update: Update, context: ContextTypes.DEFAULT_TYPE, doc) -> None:
    chat_id = update.effective_chat.id
    msg = update.effective_message
    RESTORE_WAIT.pop(chat_id, None)
    path = WORK_ROOT / f"restore_{chat_id}_{uuid.uuid4().hex[:6]}.zip"
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        f = await context.bot.get_file(doc.file_id)
        await f.download_to_drive(path)
        info, _, _ = await asyncio.to_thread(brain.BRAIN.read_backup, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("bad backup file: %s", exc)
        path.unlink(missing_ok=True)
        await msg.reply_text("❌ این فایل بکاپ حافظه ربات نیست یا خراب است.\n"
                             "فقط فایلی که خود ربات با «💾 گرفتن بکاپ» داده قابل بازیابی است.")
        return
    old = RESTORE_FILES.pop(chat_id, None)
    if old:
        old.unlink(missing_ok=True)
    RESTORE_FILES[chat_id] = path
    now = brain.BRAIN.summary()
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ ادغام با حافظه فعلی (پیشنهادی)", callback_data="mem:merge")],
        [InlineKeyboardButton("🔁 جایگزینی کامل", callback_data="mem:replace"),
         InlineKeyboardButton("✖️ لغو", callback_data="mem:cancel")],
    ])
    await msg.reply_text(
        "📦 فایل بکاپ سالم است:\n"
        f"• تاریخ بکاپ: {_fa_date(info.get('created', 0))}\n"
        f"• داخل بکاپ: {_fa(info.get('formats', 0))} قالب، {_fa(info.get('samples', 0))} نمونه\n"
        f"• حافظه فعلی ربات: {_fa(now['formats'])} قالب، {_fa(now['samples'])} نمونه\n\n"
        "➕ ادغام: هرچه در بکاپ هست به حافظه فعلی اضافه می‌شود و چیزی پاک نمی‌شود.\n"
        "🔁 جایگزینی: حافظه فعلی کنار می‌رود و دقیقاً همان بکاپ برمی‌گردد.\n"
        "(در هر دو حالت یک نسخه از حافظه فعلی خودکار نگه داشته می‌شود و قابل برگشت است.)",
        reply_markup=kb)


def _undo_keyboard(path: Path) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("↩️ برگرداندن به قبل", callback_data=f"mem:undo:{path.name}")]])


async def _on_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    chat_id = update.effective_chat.id
    parts = (q.data or "").split(":", 2)
    action = parts[1] if len(parts) > 1 else ""
    if action == "backup":
        await _send_backup(context, chat_id)
    elif action == "restore":
        await _ask_for_backup(q.message, chat_id)
    elif action == "refresh":
        await _safe_edit(q.message, brain.report(), reply_markup=_brain_keyboard())
    elif action == "cancel":
        RESTORE_WAIT.pop(chat_id, None)
        path = RESTORE_FILES.pop(chat_id, None)
        if path:
            path.unlink(missing_ok=True)
        await _safe_edit(q.message, "لغو شد.")
    elif action in ("merge", "replace"):
        path = RESTORE_FILES.pop(chat_id, None)
        if path is None or not path.exists():
            await _safe_edit(q.message, "فایل بکاپ دیگر در دسترس نیست؛ دوباره بفرست. /restore")
            return
        await _safe_edit(q.message, "⏳ در حال بازیابی…")
        try:
            safety = await asyncio.to_thread(_auto_backup, "before-restore")
            _, data, samples = await asyncio.to_thread(brain.BRAIN.read_backup, path)
            glyphs = await asyncio.to_thread(brain.BRAIN.backup_glyphs, path)
            n = await asyncio.to_thread(brain.BRAIN.restore, data, samples, action == "merge", glyphs)
        except Exception:  # noqa: BLE001
            log.exception("restore failed")
            await _safe_edit(q.message, "❌ بازیابی انجام نشد؛ حافظه فعلی دست نخورد.")
            return
        finally:
            path.unlink(missing_ok=True)
        now = brain.BRAIN.summary()
        head = (f"✅ ادغام شد: {_fa(n)} قالب تازه اضافه شد." if action == "merge"
                else f"✅ بازیابی کامل شد: {_fa(n)} قالب برگشت.")
        await _safe_edit(q.message, f"{head}\n🧠 حافظه الان: {_fa(now['formats'])} قالب "
                                    f"({_fa(now['trusted'])} مستقل)، {_fa(now['samples'])} نمونه.",
                         reply_markup=_undo_keyboard(safety))
    elif action == "undo":
        path = BACKUP_DIR / parts[2] if len(parts) > 2 else None
        if path is None or not path.exists() or path.parent != BACKUP_DIR:
            await q.message.reply_text("نسخه قبلی دیگر موجود نیست.")
            return
        _, data, samples = await asyncio.to_thread(brain.BRAIN.read_backup, path)
        glyphs = await asyncio.to_thread(brain.BRAIN.backup_glyphs, path)
        await asyncio.to_thread(brain.BRAIN.restore, data, samples, False, glyphs)
        now = brain.BRAIN.summary()
        await _safe_edit(q.message, f"↩️ حافظه به حالت قبل برگشت: {_fa(now['formats'])} قالب، "
                                    f"{_fa(now['samples'])} نمونه.")
    elif action == "reset":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🗑 بله، پاک کن", callback_data="mem:reset_yes"),
                                    InlineKeyboardButton("✖️ نه", callback_data="mem:cancel")]])
        await q.message.reply_text("⚠️ مطمئنی؟ هرچه ربات یاد گرفته پاک می‌شود و فقط درس‌های پایه می‌ماند.\n"
                                   "(قبل از پاک کردن یک نسخه خودکار نگه داشته می‌شود.)", reply_markup=kb)
    elif action == "reset_yes":
        safety = await asyncio.to_thread(_auto_backup, "before-reset")
        await asyncio.to_thread(brain.BRAIN.reset)
        await asyncio.to_thread(brain.BRAIN.merge_seed, brain.SEED_DIR)
        await _safe_edit(q.message, "🗑 حافظه پاک شد و ربات به درس‌های پایه برگشت.",
                         reply_markup=_undo_keyboard(safety))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    async with _chat_lock(update.effective_chat.id):
        _drop_session(update.effective_chat.id)
        _leave_shot(update.effective_chat.id)
    await update.effective_message.reply_text("🗑 پاک شد. فایل جدید را بفرست.")


# ------------------------------------------------------ screenshot numbers --

SHOT_HELP = (
    "🖼 حالت تغییر بازدید روشن شد.\n"
    "تا وقتی /start نزنی همین‌جا می‌مانی و هر عکسی بفرستی اسکرین‌شات حساب می‌شود.\n\n"
    "⚡️ سریع‌ترین راه: عددها را در کپشن هر عکس بنویس و هر چند تا عکس می‌خواهی یک‌جا بفرست یا فوروارد کن؛ "
    "جواب‌ها به همان ترتیب، به صورت عکس برمی‌گردند.\n"
    "• «۳۴.۸» ← فقط پایینی می‌شود ۳۴.۸K\n"
    "• «۶.۲ ۵۳.۷» ← بالایی ۶.۲K و پایینی ۵۳.۷K\n"
    "• عدد بدون K یعنی هزار: «۵» یعنی ۵K؛ «۲۵۰۰» یعنی ۲.۵K و «۰.۵» یعنی ۵۰۰.\n\n"
    "عکس بدون کپشن هم بفرستی، بازدیدهایش را نشان می‌دهم و عدد را می‌پرسم.\n"
    "عدد با همان فونت، رنگ و فاصله‌ی خود تلگرام دوباره کشیده می‌شود و بقیه‌ی عکس دست نمی‌خورد."
)
SHOT_VALUES = ["1K", "2K", "5K", "10K", "20K", "50K"]
SHOT_KB_ROWS = 4                             # view counts that get their own row of buttons; the rest are typed


def _load_shot_mode() -> set[int]:
    try:
        return {int(x) for x in json.loads(SHOT_MODE_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError, TypeError):
        return set()


def _set_shot_mode(chat_id: int, on: bool) -> None:
    """/shot mode is kept on disk so a redeploy does not drop the user back into the price-list flow."""
    (SHOT_MODE.add if on else SHOT_MODE.discard)(chat_id)
    try:
        SHOT_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SHOT_MODE_FILE.write_text(json.dumps(sorted(SHOT_MODE)), encoding="utf-8")
    except OSError as exc:
        log.warning("could not save shot mode: %s", exc)


SHOT_MODE: set[int] = _load_shot_mode()


def _shot_names(n: int) -> list[str]:
    """What the view counts are called, top to bottom."""
    if n == 1:
        return ["بازدید"]
    if n == 2:
        return ["بالایی", "پایینی"]
    if n == 3:
        return ["بالایی", "وسطی", "پایینی"]
    return [f"ردیف {_fa(i + 1)}" for i in range(n)]


def _shot_keyboard(sess: ShotSession) -> InlineKeyboardMarkup:
    """One row of numbers per view count (a single row when there is only one); a ✔ marks what is picked."""
    n = len(sess.analysis.footers)
    if n == 1:
        return InlineKeyboardMarkup([[InlineKeyboardButton(_fa(v), callback_data=f"shot:val:{v}")
                                      for v in SHOT_VALUES]])
    names = _shot_names(n)
    rows = []
    for i in range(min(n, SHOT_KB_ROWS)):
        pick = sess.picks.get(i)
        rows.append([InlineKeyboardButton(f"{names[i]}: {_fa(pick) if pick else '—'}", callback_data="shot:nop")])
        rows.append([InlineKeyboardButton(("✔" if v == pick else "") + _fa(v), callback_data=f"shot:set:{i}:{v}")
                     for v in SHOT_VALUES])
    rows.append([InlineKeyboardButton("✅ ساختن", callback_data="shot:go")])
    return InlineKeyboardMarkup(rows)


def _shot_ask(n: int) -> str:
    tip = "\n💡 دفعه‌ی بعد عدد را در کپشن عکس بنویس تا بدون سؤال ساخته شود."
    if n == 1:
        return "چه عددی بشود؟ بنویس یا دکمه بزن. عدد بدون K یعنی هزار: «۳۴.۸» یعنی ۳۴.۸K." + tip
    if n == 2:
        return ("عدد بالایی و پایینی را با هم بنویس، اول بالایی: «۶.۲ ۵۳.۷». یک عدد تنها فقط پایینی را عوض "
                "می‌کند. عدد بدون K یعنی هزار. یا برای هر کدام دکمه بزن." + tip)
    return ("عددها را از بالا به پایین با هم بنویس؛ اگر کمتر بنویسی از پایین پر می‌شوند (یک عدد فقط پایینی). "
            "«ردیف ۲ بشود ۱۰» هم می‌شود. عدد بدون K یعنی هزار." + tip)


def _is_image_message(msg: Message) -> bool:
    if msg.photo:
        return True
    doc = msg.document
    if doc is None:
        return False
    name = (doc.file_name or "").lower()
    return (doc.mime_type or "").lower().startswith("image/") or name.endswith(IMAGE_EXT)


def _wants_shot(chat_id: int, caption: str) -> bool:
    if chat_id in SHOT_MODE:
        return True
    low = caption.lower()
    return caption.startswith("/shot") or any(w in low for w in SHOT_WORDS)


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


def _shot_values(text: str, n: int, strict: bool = False) -> tuple[dict[int, str] | None, str | None]:
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


def _read_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _photo_bytes(img: np.ndarray) -> bytes:
    """The result as a normal Telegram photo (Telegram re-encodes photos anyway; this keeps the upload small)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


async def _tg(call, *args, **kwargs):
    """A Telegram call that waits out flood limits (answering a batch of 30 screenshots can hit them)."""
    for _ in range(4):
        try:
            return await call(*args, **kwargs)
        except RetryAfter as exc:
            wait = exc.retry_after
            await asyncio.sleep((wait.total_seconds() if hasattr(wait, "total_seconds") else float(wait)) + 0.5)
    return await call(*args, **kwargs)


async def cmd_shot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        return await _deny(update)
    _set_shot_mode(update.effective_chat.id, True)
    await update.effective_message.reply_text(SHOT_HELP + "\n\nحالا اسکرین‌شات‌ها را بفرست 👇")


def _keep_shot(chat_id: int, question: Message, sess: ShotSession) -> None:
    chat = SHOTS.setdefault(chat_id, {})
    chat[question.message_id] = sess
    SHOT_LAST[chat_id] = question.message_id
    for old in sorted(chat)[:-30]:                 # keep memory bounded: the newest 30 questions stay usable
        chat.pop(old, None)


def _find_shot(chat_id: int, msg: Message | None = None) -> ShotSession | None:
    """The screenshot a typed answer is for: the question it replies to, else the newest one."""
    chat = SHOTS.get(chat_id) or {}
    reply = msg.reply_to_message if msg is not None else None
    key = reply.message_id if reply is not None and reply.message_id in chat else SHOT_LAST.get(chat_id)
    sess = chat.get(key) if key is not None else None
    return None if sess is None or sess.expired() else sess


# --- one chat's screenshots: worked on side by side, answered in the order they were sent

@dataclass
class ShotOutcome:
    analysis: shot.Analysis | None = None
    photo: bytes | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    question: str | None = None                  # the caption had numbers that do not fit this screenshot


class ShotLine:
    def __init__(self) -> None:
        self.tail: asyncio.Task | None = None
        self.made = self.asked = self.failed = 0


SHOT_LINES: dict[int, ShotLine] = {}
SHOT_WORKERS = asyncio.Semaphore(3)              # screenshots downloaded and redrawn at the same time
SHOT_ALBUMS: dict[str, str] = {}                 # media group -> its caption (Telegram puts it on one photo)


def _build(analysis: shot.Analysis, values: dict[int, str]) -> tuple[bytes, shot.Result]:
    result = shot.apply_views(analysis, values)
    return _photo_bytes(result.img), result


async def _shot_work(context: ContextTypes.DEFAULT_TYPE, file_id: str, caption: str) -> ShotOutcome:
    out = ShotOutcome()
    async with SHOT_WORKERS:
        try:
            f = await _tg(context.bot.get_file, file_id)
            img = await asyncio.to_thread(_read_image, bytes(await f.download_as_bytearray()))
        except Exception:  # noqa: BLE001
            log.exception("shot download failed")
            out.error = "دریافت یا خواندن عکس انجام نشد. دوباره بفرست."
            return out
        try:
            out.analysis = await asyncio.to_thread(shot.analyze, img)
            raw, out.question = _shot_values(caption, len(out.analysis.footers), strict=True)
            if raw is None:
                return out
            values = {i: shot.parse_value(v) for i, v in sorted(raw.items())}
            out.photo, result = await asyncio.to_thread(_build, out.analysis, values)
            out.notes = result.notes
        except shot.ShotError as exc:
            out.error = str(exc)
        except Exception:  # noqa: BLE001
            log.exception("shot work failed")
            out.error = "در ساخت این عکس خطایی رخ داد."
    return out


def _queue_shot(update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file, caption: str) -> None:
    """Start on a screenshot right away and line its answer up behind the ones sent before it.

    Nothing here waits, so screenshots join the line in the order Telegram delivered them."""
    msg = update.effective_message
    chat_id = update.effective_chat.id
    _purge_expired()
    if msg.media_group_id:
        if caption:
            SHOT_ALBUMS[msg.media_group_id] = caption
            while len(SHOT_ALBUMS) > 200:
                SHOT_ALBUMS.pop(next(iter(SHOT_ALBUMS)))
        else:
            caption = SHOT_ALBUMS.get(msg.media_group_id, "")
    line = SHOT_LINES.setdefault(chat_id, ShotLine())
    work = asyncio.ensure_future(_shot_work(context, tg_file.file_id, caption))
    line.tail = context.application.create_task(_shot_answer(update, context, line, line.tail, work), update=update)


async def _shot_answer(update: Update, context: ContextTypes.DEFAULT_TYPE, line: ShotLine,
                       before: asyncio.Task | None, work: asyncio.Future) -> None:
    msg = update.effective_message
    chat_id = update.effective_chat.id
    if before is not None:
        with contextlib.suppress(Exception):
            await before
    out = await work
    try:
        if out.photo is not None:
            await _tg(context.bot.send_chat_action, chat_id, ChatAction.UPLOAD_PHOTO)
            await _tg(msg.reply_photo, out.photo, do_quote=True)
            if out.notes:
                await _tg(msg.reply_text, "\n".join(f"⚠️ {n}" for n in out.notes[:3]), do_quote=True)
            line.made += 1
        elif out.error:
            await _tg(msg.reply_text, f"❌ {out.error}", do_quote=True)
            line.failed += 1
        else:
            await _ask_shot(context, msg, out.analysis, out.question)
            line.asked += 1
    except Exception:  # noqa: BLE001
        log.exception("shot answer failed")
        line.failed += 1
    if line.tail is asyncio.current_task():          # the last one in line: report the batch, start a new one
        made, asked, failed = line.made, line.asked, line.failed
        line.made = line.asked = line.failed = 0
        if made + asked + failed > 1:
            parts = [f"✅ {_fa(made)} عکس آماده شد"]
            if asked:
                parts.append(f"{_fa(asked)} عکس کپشن عددی نداشت و عددش را پرسیدم")
            if failed:
                parts.append(f"❌ {_fa(failed)} عکس ساخته نشد (جواب همان پیام‌ها داده شد)")
            with contextlib.suppress(Exception):
                await _tg(context.bot.send_message, chat_id, "؛ ".join(parts) + ".")


async def _ask_shot(context: ContextTypes.DEFAULT_TYPE, source: Message, analysis: shot.Analysis,
                    question: str | None) -> None:
    """A screenshot without numbers: one photo (found view counts boxed) with what they are now and the buttons."""
    sess = ShotSession(analysis, source)
    n = len(analysis.footers)
    names = _shot_names(n)
    now = [_fa(f.views_text) if f.readable else "؟" for f in analysis.footers]
    if n == 1:
        lines = [f"✅ بازدید فعلی: {now[0]}"]
    else:
        lines = [f"✅ {_fa(n)} بازدید پیدا شد:"]
        lines += [f"{'⬆️' if i == 0 else '⬇️' if i == n - 1 else '▫️'} {names[i]}: {v}" for i, v in enumerate(now)]
    lines += [f"⚠️ {w}" for w in analysis.warnings[:2]]
    if question:
        lines.append(f"❓ {question}")
    lines.append(_shot_ask(n))
    prev = await asyncio.to_thread(shot.preview, analysis)
    ok, buf = cv2.imencode(".jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 85])
    question_msg = await _tg(source.reply_photo, buf.tobytes(), caption="\n".join(lines)[:1024],
                             reply_markup=_shot_keyboard(sess), do_quote=True)
    _keep_shot(source.chat_id, question_msg, sess)


async def _apply_shot(context: ContextTypes.DEFAULT_TYPE, sess: ShotSession, raw: dict[int, str],
                      reply_to: Message) -> None:
    """Build a screenshot whose numbers came by text or buttons; the photo answers the screenshot itself."""
    try:
        values = {i: shot.parse_value(v) for i, v in sorted(raw.items())}
    except shot.ShotError as exc:
        await reply_to.reply_text(f"❓ {exc}")
        return
    await _tg(context.bot.send_chat_action, sess.source.chat_id, ChatAction.UPLOAD_PHOTO)
    try:
        photo, result = await asyncio.to_thread(_build, sess.analysis, values)
    except shot.ShotError as exc:
        await reply_to.reply_text(f"❌ {exc}")
        return
    except Exception:  # noqa: BLE001
        log.exception("shot edit failed")
        await reply_to.reply_text("❌ ساخت عکس با خطا روبه‌رو شد.")
        return
    try:
        await _tg(sess.source.reply_photo, photo, do_quote=True)
    except BadRequest:                               # the screenshot was deleted meanwhile
        await _tg(context.bot.send_photo, sess.source.chat_id, photo)
    if result.notes:
        await reply_to.reply_text("\n".join(f"⚠️ {n}" for n in result.notes[:3]))


async def _on_shot_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """shot:val:<v> sets the only view count; shot:set:<i>:<v> picks one row (built once every row has a number);
    shot:go builds with what is picked so far. The buttons stay for another try."""
    q = update.callback_query
    _, action, *args = q.data.split(":")
    if action == "nop":
        return
    chat_id = update.effective_chat.id
    sess = (SHOTS.get(chat_id) or {}).get(q.message.message_id)
    if sess is None or sess.expired():
        await q.message.reply_text("این اسکرین‌شات منقضی شده؛ دوباره بفرستش.")
        return
    n = len(sess.analysis.footers)
    if action == "val":
        await _apply_shot(context, sess, {n - 1: args[0]}, q.message)
        return
    if action == "set":
        i, value = int(args[0]), args[1]
        if not 0 <= i < n:
            return
        sess.picks[i] = value
        if len(sess.picks) < min(n, SHOT_KB_ROWS):
            with contextlib.suppress(BadRequest):
                await q.message.edit_reply_markup(reply_markup=_shot_keyboard(sess))
            return
    elif not sess.picks:
        await q.message.reply_text("اول برای بازدیدها عدد انتخاب کن (یا عددها را بنویس).")
        return
    picks = dict(sess.picks)
    sess.picks.clear()
    with contextlib.suppress(BadRequest):
        await q.message.edit_reply_markup(reply_markup=_shot_keyboard(sess))
    await _apply_shot(context, sess, picks, q.message)


async def _on_shot_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Typed numbers for a screenshot. Returns False when the text is not for /shot."""
    msg = update.effective_message
    chat_id = update.effective_chat.id
    sess = _find_shot(chat_id, msg)
    if chat_id not in SHOT_MODE:
        price = SESSIONS.get(chat_id)
        if sess is None or (price is not None and not price.expired() and price.items
                            and price.created > sess.created):
            return False
    if sess is None:
        await msg.reply_text("اسکرین‌شات‌ها را بفرست؛ عدد هر کدام را در کپشنش بنویس (مثلاً «۳۴.۸»).\n"
                             "برای برگشتن به لیست قیمت: /start")
        return True
    n = len(sess.analysis.footers)
    values, question = _shot_values(msg.text or "", n)
    if values is None:
        await msg.reply_text(f"❓ {question or _shot_ask(n)}")
        return True
    await _apply_shot(context, sess, values, msg)
    return True


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
        if _is_backup_upload(chat_id, filename):
            return await _on_backup_file(update, context, doc)
        if not (mime == "application/pdf" or mime.startswith("image/")
                or filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))):
            await msg.reply_text("این نوع فایل پشتیبانی نمی‌شود. لطفاً PDF یا عکس بفرست.")
            return
        if chat_id in SHOT_MODE and not _is_image_message(msg):
            await msg.reply_text("الان در حالت تغییر بازدید هستی و فقط اسکرین‌شات می‌گیرم.\n"
                                 "برای لیست قیمت اول /start را بزن.")
            return
        tg_file = doc
        size = doc.file_size or 0
    if size > MAX_DOWNLOAD:
        await msg.reply_text("حجم فایل بیشتر از ۲۰ مگابایت است (محدودیت تلگرام برای ربات‌ها). لطفاً فایل کوچک‌تری بفرست.")
        return

    if _is_image_message(msg) and _wants_shot(chat_id, (msg.caption or "").strip()):
        return _queue_shot(update, context, tg_file, (msg.caption or "").strip())

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

        live = {"stage": "🔎 در حال بررسی", "pages": "", "start": time.monotonic()}

        def progress(done: int, total: int) -> None:
            if total > 1:
                live["pages"] = f" — صفحه {_fa(done)} از {_fa(total)}"

        def stage(text: str) -> None:
            live["stage"] = text

        async def ticker() -> None:
            """Keep the status message alive: what is happening and for how long."""
            shown = ""
            while True:
                await asyncio.sleep(3)
                secs = int(time.monotonic() - live["start"])
                text = f"{live['stage']}…{live['pages']}\n⏱ {_fa(secs)} ثانیه"
                if text != shown:
                    shown = text
                    await _safe_edit(status, text)
                with contextlib.suppress(Exception):
                    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        await _quota_warning(msg, src)
        tick = asyncio.create_task(ticker())
        try:
            analysis = await asyncio.to_thread(pipeline.analyze, src, filename, fdir / "work", progress, False, stage)
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
        finally:
            tick.cancel()
        analysis.cache["seconds"] = time.monotonic() - live["start"]
        log.info("%s analysed in %.1fs", filename, analysis.cache["seconds"])
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
    groups = _groups_note(analysis)
    if groups:
        lines.append("")
        lines += groups
    notes = _brain_notes(analysis)
    if analysis.cache.get("seconds"):
        notes.append(f"⏱ زمان بررسی: {_fa(round(analysis.cache['seconds']))} ثانیه")
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
    if await _on_shot_text(update, context):
        return
    async with _chat_lock(chat_id):   # waits for a file that is still being analysed
        session = SESSIONS.get(chat_id)
        if session is None or session.expired() or not session.items:
            await update.effective_message.reply_text(
                "اول فایل لیست قیمت (PDF یا عکس) را بفرست، بعد بگو چه تغییری بدهم.\n/help")
            return
        wait = session.sub_wait
        if wait is not None and session.pending is not None and wait.key == session.pending.key:
            return await _on_subtitle_text(update, context, session, update.effective_message.text or "")
        await _handle_command_text(update, context, session, update.effective_message.text or "")


async def _handle_command_text(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session,
                               text: str) -> None:
    msg = update.effective_message
    local = commands.parse_local(text)
    if local.plan is not None:
        return await _prepare(update, context, session, local.plan)
    by_group = _group_plan(session, text)
    if by_group is not None:
        # «گروه پژو ۱۰ درصد، بقیه ۵ درصد»: understood from the list's own groups, no Gemini needed
        await msg.reply_text(f"🗂 {by_group.summary}")
        return await _prepare(update, context, session, by_group)
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
                            "group": it.group or "-", "label": it.label or "-", "value": fmt_plain(it.value)})
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


def _session_groups(session: Session) -> list[str]:
    names: list[str] = []
    for a in session.analyses:
        for g, _ in a.groups():
            if g not in names:
                names.append(g)
    return names


def _group_plan(session: Session, text: str) -> Plan | None:
    rules = commands.parse_groups(text, _session_groups(session))
    if rules is None:
        return None
    ids: dict[str, set[str]] = {}
    for fi, a in enumerate(session.analyses):
        for it in a.items:
            if it.group:
                ids.setdefault(it.group, set()).add(f"f{fi + 1}-{it.id}")
    return commands.plan_for_groups(text, rules, ids)


def _groups_note(analysis: Analysis) -> list[str]:
    """The list's groups, for the analysis message."""
    groups = analysis.groups()
    if not groups:
        return []
    shown = "، ".join(f"{g} ({_fa(n)})" for g, n in groups[:8])
    more = f" و {_fa(len(groups) - 8)} گروه دیگر" if len(groups) > 8 else ""
    return [f"🗂 {_fa(len(groups))} گروه: {shown}{more}",
            "برای تغییر یک گروه بنویس مثلاً: «گروه " + _short_group(groups[0][0]) + " ۱۰ درصد افزایش، بقیه ۵ درصد»"]


def _short_group(name: str) -> str:
    return name.replace("گروه", "", 1).strip() if name.startswith("گروه") else name


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
    session.sub_wait = None           # a question about the old change's Arizon header is over


async def _deliver(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session, pending: Pending,
                   formats: list[str], subtitle: str = "") -> None:
    """subtitle: the line under the Arizon template's title (other formats ignore it)."""
    chat_id = update.effective_chat.id
    msg = update.effective_message
    for fmt in formats:
        status = await msg.reply_text(f"⏳ در حال ساخت خروجی {FORMAT_NAMES[fmt]}…")
        await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
        try:
            paths = await asyncio.to_thread(pending.exporter.build, fmt, subtitle)
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
    if (q.data or "").startswith("mem:"):
        return await _on_memory(update, context)
    if (q.data or "").startswith("shot:"):
        return await _on_shot_button(update, context)
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
            formats = ["pdf", "xlsx", "image", "arizon"] if parts[1] == "all" else [parts[1]]
            if any(f not in FORMAT_NAMES for f in formats):
                return
            if any(f in ARIZON_FORMATS for f in formats):
                # the Arizon template first asks what goes under its title
                return await _ask_subtitle(update, session, pending, formats)
            await _deliver(update, context, session, pending, formats)
        elif kind == "sub":
            await _on_subtitle_button(update, context, session, pending, parts[1])


# ----------------------------------------------------- Arizon header line ---

def _recent_subtitles(chat_id: int) -> list[str]:
    try:
        data = json.loads(SUBTITLES_FILE.read_text(encoding="utf-8"))
        recent = data.get(str(chat_id), []) if isinstance(data, dict) else []
        return [t for t in recent if isinstance(t, str) and t][:RECENT_SUBTITLES]
    except (OSError, ValueError):
        return []


def _remember_subtitle(chat_id: int, text: str) -> None:
    """The chat's recent lines, newest first (kept on the data volume, offered as buttons next time)."""
    try:
        data = json.loads(SUBTITLES_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    old = data.get(str(chat_id), [])
    data[str(chat_id)] = ([text] + [t for t in (old if isinstance(old, list) else []) if t != text])[:RECENT_SUBTITLES]
    try:
        SUBTITLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SUBTITLES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(SUBTITLES_FILE)
    except OSError as exc:
        log.warning("could not save the Arizon header line: %s", exc)


def _clip(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


async def _ask_subtitle(update: Update, session: Session, pending: Pending, formats: list[str]) -> None:
    """Ask what goes under «لیست قیمت محصولات» in the Arizon template."""
    wait = SubtitleWait(pending.key, formats, _recent_subtitles(update.effective_chat.id))
    session.sub_wait = wait
    lines = ["✏️ زیر عنوان «لیست قیمت محصولات» در قالب آریزون چه نوشته شود؟", "",
             "متن را همین‌جا بنویس و بفرست؛ مثلاً: باباپارت"]
    if wait.offered:
        lines.append("یا یکی از متن‌های قبلی را بزن.")
    lines.append("اگر متنی لازم نیست، «بدون متن» را بزن.")
    if any(f not in ARIZON_FORMATS for f in formats):
        lines.append("\n(بعد از جواب، همه فرمت‌ها با هم ساخته می‌شوند.)")
    rows = [[InlineKeyboardButton(f"✅ {_clip(t)}", callback_data=f"sub:r{j}:{wait.key}")]
            for j, t in enumerate(wait.offered)]
    rows.append([InlineKeyboardButton("➖ بدون متن (فقط «لیست قیمت محصولات»)", callback_data=f"sub:none:{wait.key}")])
    wait.question = await update.effective_message.reply_text("\n".join(lines),
                                                              reply_markup=InlineKeyboardMarkup(rows))


def _looks_like_command(session: Session, text: str) -> bool:
    local = commands.parse_local(text)
    return (local.plan is not None or local.ask_direction is not None or _COMMANDISH.search(text) is not None
            or _group_plan(session, text) is not None)


async def _on_subtitle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session, raw: str) -> None:
    """A typed line for the Arizon header: checked, then the waiting formats are made with it."""
    from pricebot.arizon.render import SUBTITLE_MAX, clean_subtitle

    msg = update.effective_message
    wait = session.sub_wait
    text = clean_subtitle(raw)
    if not text:
        await msg.reply_text("این متن را نمی‌شود در سربرگ نوشت (فقط شکلک یا علامت بود). متن را با حروف بنویس، "
                             "یا «بدون متن» را بزن.")
        return
    if len(text) > SUBTITLE_MAX:
        await msg.reply_text(f"متن طولانی است ({_fa(len(text))} حرف). حداکثر {_fa(SUBTITLE_MAX)} حرف بنویس "
                             "تا در سربرگ جا شود.")
        return
    if _looks_like_command(session, raw):
        wait.typed, wait.raw = text, raw
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✏️ زیر عنوان بنویس", callback_data=f"sub:use:{wait.key}"),
            InlineKeyboardButton("🔁 دستور جدید قیمت است", callback_data=f"sub:cmd:{wait.key}"),
        ]])
        await msg.reply_text(f"«{text}» را زیر عنوان قالب آریزون بنویسم، یا دستور جدید تغییر قیمت است؟",
                             reply_markup=kb)
        return
    await _use_subtitle(update, context, session, wait, text)


async def _on_subtitle_button(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session,
                              pending: Pending, choice: str) -> None:
    q = update.callback_query
    if choice == "ask":
        # the Arizon output again, with another line under the title
        formats = [f for f in ARIZON_FORMATS if f in pending.sent] or ["arizon"]
        return await _ask_subtitle(update, session, pending, formats)
    wait = session.sub_wait
    if wait is None or wait.key != pending.key:
        await q.message.reply_text("این دکمه مربوط به سؤال قبلی است؛ دوباره «✨ قالب آریزون» را انتخاب کن.")
        return
    with contextlib.suppress(BadRequest):
        await q.message.edit_reply_markup(None)       # each question is answered once
    if choice == "cmd":
        session.sub_wait = None
        return await _handle_command_text(update, context, session, wait.raw)
    if choice == "none":
        text = ""
    elif choice == "use" and wait.typed:
        text = wait.typed
    elif choice.startswith("r") and choice[1:].isdigit() and int(choice[1:]) < len(wait.offered):
        text = wait.offered[int(choice[1:])]
    else:
        return
    await _use_subtitle(update, context, session, wait, text)


async def _use_subtitle(update: Update, context: ContextTypes.DEFAULT_TYPE, session: Session, wait: SubtitleWait,
                        text: str) -> None:
    session.sub_wait = None
    if wait.question is not None:
        chosen = f"زیر عنوان: «{text}»" if text else "بدون متن زیر عنوان"
        with contextlib.suppress(BadRequest):
            await wait.question.edit_text(f"✏️ قالب آریزون — {chosen}")
    if text:
        _remember_subtitle(update.effective_chat.id, text)
    await _deliver(update, context, session, session.pending, wait.formats, subtitle=text)


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
        if (info.get("layout") or {}).get("id"):
            ids.add(info["layout"]["id"])
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


async def _set_menu(app: Application) -> None:
    """The command list shown by Telegram's menu button."""
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "راهنمای ربات"),
            BotCommand("shot", "تغییر عدد بازدید در اسکرین‌شات تلگرام"),
            BotCommand("brain", "هوش ربات: وضعیت یادگیری، بکاپ و بازیابی"),
            BotCommand("backup", "گرفتن بکاپ از حافظه ربات"),
            BotCommand("restore", "بازیابی بکاپ حافظه ربات"),
            BotCommand("usage", "سهمیه امروز Gemini"),
            BotCommand("test", "آزمایش اتصال به Gemini"),
            BotCommand("status", "وضعیت ربات و فایل‌های باز"),
            BotCommand("reset", "شروع دوباره (بستن فایل‌های باز)"),
            BotCommand("id", "شناسه عددی شما"),
        ])
    except Exception as exc:  # noqa: BLE001 - the menu is a nicety
        log.warning("could not set the command menu: %s", exc)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    if not ai.enabled():
        log.warning("no AI key set: images/scans will not work, PDFs use simple detection")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    app = (Application.builder().token(config.TELEGRAM_BOT_TOKEN).concurrent_updates(True).post_init(_set_menu)
           .read_timeout(60).write_timeout(180).connect_timeout(30).media_write_timeout(180).build())
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler(["shot", "views"], cmd_shot))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler(["usage", "quota"], cmd_usage))
    app.add_handler(CommandHandler(["test", "diag"], cmd_test))
    app.add_handler(CommandHandler(["brain", "learn", "ai"], cmd_brain))
    app.add_handler(CommandHandler("backup", cmd_backup))
    app.add_handler(CommandHandler("restore", cmd_restore))
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
