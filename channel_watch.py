"""
خوراکِ پستِ کانال از سمتِ *ربات* - نه از سمتِ اکانت‌ها.

چرا این ماژول ساخته شد
----------------------
حلقه‌ی اسکنِ بازدیدِ خودکار هر ۴۵ ثانیه یه اکانتِ واقعی رو مجبور می‌کرد
`messages.getHistory` بزنه تا ببینه پستِ جدیدی اومده یا نه. برایِ هر کانال یعنی
۱۹۲۰ درخواست در روز، همه از یک «اکانتِ ناظرِ» ثابت، با فاصله‌ی دقیقاً یکنواخت،
۲۴ ساعته - حتی وقتی کانال هفته‌ها هیچ پستی نداشت. اون اکانت عملاً محکوم به فریز
بود، و وقتی می‌سوخت کد یه اکانتِ تصادفیِ دیگه رو جاش می‌ذاشت تا اون هم بسوزه.

راه‌حل: خودِ ربات ادمینِ کاناله، پس تلگرام پستِ جدید رو *بهش پوش می‌کنه*
(آپدیتِ channel_post). دیگه هیچ‌کس لازم نیست چیزی رو پول کنه. مصرفِ اکانت‌ها
برایِ تشخیصِ پستِ جدید می‌شه صفر.

این ماژول عمداً هیچ ایمپورتی از bot.py یا auto_view.py نداره تا حلقه‌ی وابستگی
نسازه؛ هر دو طرف فقط ازش استفاده می‌کنن.
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# چند پستِ اخیرِ هر کانال در حافظه نگه داشته بشه. فقط شناسه و تاریخ ذخیره می‌شه،
# نه خودِ پیام - پس حتی با صدها کانال هم ناچیزه.
MAX_POSTS_PER_CHANNEL = int(os.getenv("CHANNEL_WATCH_BUFFER", "300"))
# نتیجه‌ی «ربات این کانال رو می‌بینه؟» چند ثانیه معتبره
WATCHABLE_TTL_SECONDS = float(os.getenv("CHANNEL_WATCH_PROBE_TTL", "1800"))
# اگه پروبِ Bot API بیشتر از این طول کشید، بی‌خیالش شو (به فالبک می‌افتیم)
PROBE_TIMEOUT = float(os.getenv("CHANNEL_WATCH_PROBE_TIMEOUT", "15"))


class SeenPost:
    """
    کمینه‌ترین چیزی که _check_one_channel لازم داره: شناسه و تاریخ.

    عمداً شکلِ همون آبجکتِ پیامِ Telethon رو تقلید می‌کنه (`.id` و `.date`) تا
    مسیرِ پایین‌دستی اصلاً نفهمه پست از کجا اومده.
    """

    __slots__ = ("id", "date")

    def __init__(self, message_id: int, date: datetime):
        self.id = int(message_id)
        self.date = date

    def __repr__(self):
        return f"SeenPost(id={self.id}, date={self.date})"


# username (بدونِ @ و lowercase) -> {message_id: SeenPost}
_posts: dict = {}
# username -> (نتیجه‌ی بولی، مهرِ زمانِ monotonic)
_watchable: dict = {}
# chat_id عددی -> username، تا آپدیتِ کانالی که فقط با آیدی شناخته می‌شه هم جا بیفته
_id_to_username: dict = {}
_bot = None
_lock = asyncio.Lock()

try:  # ثبت در پاک‌سازیِ حافظه (اختیاری - نبودش نباید چیزی رو بشکنه)
    import memory_hygiene as _mh

    # ⚠️ clearable=False عمدیه: این بافر *تنها* جاییه که پستِ رسیده‌ولی‌هنوز-
    # زمان‌بندی‌نشده نگه داشته می‌شه. اگه پاک‌سازیِ اضطراریِ رم خالیش می‌کرد، اون
    # پست برایِ همیشه گم می‌شد (تلگرام دوباره نمی‌فرستدش). حجمش هم ناچیزه: فقط
    # شناسه و تاریخ، حداکثر MAX_POSTS_PER_CHANNEL تا برایِ هر کانال.
    _mh.register_cache("دیده‌بانِ کانال · پست‌های دیده‌شده", _posts,
                       soft_max=5000, clearable=False)
    _mh.register_cache("دیده‌بانِ کانال · قابلِ‌دیده‌شدن", _watchable,
                       soft_max=5000, clearable=True)
    _mh.register_cache("دیده‌بانِ کانال · نگاشتِ آیدی به یوزرنیم", _id_to_username,
                       soft_max=5000, clearable=True)
except Exception:
    pass


def norm(username: str) -> str:
    """یوزرنیم رو یکدست می‌کنه: بدونِ @، بدونِ فاصله، حروفِ کوچک."""
    if not username:
        return ""
    u = str(username).strip()
    if u.startswith("https://t.me/"):
        u = u[len("https://t.me/"):]
    elif u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/").lower()


def set_bot(bot) -> None:
    """نمونه‌ی Botِ python-telegram-bot رو ثبت می‌کنه (از bot.py صدا زده می‌شه)."""
    global _bot
    _bot = bot


def has_bot() -> bool:
    return _bot is not None


# ---------------------------------------------------------------------------
# ورودی: آپدیتِ پستِ کانال از سمتِ ربات
# ---------------------------------------------------------------------------
def record_post(chat_id, username: str, message_id: int, date: datetime) -> None:
    """
    یه پستِ تازه‌رسیده رو ثبت می‌کنه. از هندلرِ channel_post در bot.py صدا زده می‌شه.

    عمداً هیچ استثنایی به بیرون نمی‌ده: این تابع داخلِ مسیرِ آپدیتِ رباته و یه خطایِ
    کوچیک این‌جا نباید کلِ پردازشِ آپدیت رو بخوابونه.
    """
    try:
        key = norm(username)
        if not key:
            return
        if date is None:
            date = datetime.now(timezone.utc)
        if chat_id is not None:
            _id_to_username[int(chat_id)] = key

        bucket = _posts.setdefault(key, {})
        bucket[int(message_id)] = SeenPost(message_id, date)

        # بافرِ کرانه‌دار: فقط جدیدترین‌ها می‌مونن
        if len(bucket) > MAX_POSTS_PER_CHANNEL:
            for old in sorted(bucket)[:-MAX_POSTS_PER_CHANNEL]:
                bucket.pop(old, None)

        # رسیدنِ پست خودش قطعی‌ترین مدرکه که ربات این کانال رو می‌بینه -
        # دیگه لازم نیست از Bot API بپرسیم
        _watchable[key] = (True, time.monotonic())
    except Exception:
        pass


def note_bot_status(username: str, status: str) -> None:
    """
    آپدیتِ my_chat_member: ربات همین الان در این کانال ادمین شد یا حذف شد.
    کشِ is_watchable همون لحظه درست می‌شه تا منتظرِ TTL نمونه.
    """
    try:
        key = norm(username)
        if key:
            _watchable[key] = (str(status) in ADMIN_STATUSES, time.monotonic())
    except Exception:
        pass


def new_posts_since(username: str, since_id: int) -> list:
    """
    پست‌هایِ جدیدترِ از since_id که ربات دیده - قدیمی‌ترین اول.
    دقیقاً همون شکلِ خروجیِ check_new_posts، تا مسیرِ پایین‌دستی تغییری نخواد.
    """
    bucket = _posts.get(norm(username))
    if not bucket:
        return []
    since = int(since_id or 0)
    return [bucket[i] for i in sorted(bucket) if i > since]


def latest_seen_id(username: str) -> int:
    """بزرگ‌ترین شناسه‌ای که ربات از این کانال دیده (۰ یعنی هیچ‌چی)."""
    bucket = _posts.get(norm(username))
    return max(bucket) if bucket else 0


def forget_channel(username: str) -> None:
    """وقتی کانال از لیستِ بازدیدِ خودکار حذف شد، بافرش هم آزاد بشه."""
    key = norm(username)
    _posts.pop(key, None)
    _watchable.pop(key, None)
    for cid, uname in list(_id_to_username.items()):
        if uname == key:
            _id_to_username.pop(cid, None)


def prune_to(usernames) -> int:
    """هر کانالی که دیگه ثبت‌شده نیست رو از حافظه پاک می‌کنه. تعدادِ پاک‌شده‌ها."""
    live = {norm(u) for u in (usernames or []) if u}
    gone = [u for u in list(_posts) if u not in live]
    for u in gone:
        forget_channel(u)
    for u in [u for u in list(_watchable) if u not in live]:
        _watchable.pop(u, None)
    return len(gone)


# ---------------------------------------------------------------------------
# «آیا ربات این کانال رو می‌بینه؟»
# ---------------------------------------------------------------------------
# وضعیت‌هایی که یعنی «ربات آپدیتِ پستِ این کانال رو می‌گیره».
# در کانال، ربات فقط به‌صورتِ ادمین اضافه می‌شه - «member» عملاً پیش نمیاد و
# تضمینی هم برایِ دریافتِ آپدیت نیست، پس عمداً نمی‌پذیریمش.
ADMIN_STATUSES = ("administrator", "creator")


async def bot_admin_status(username: str) -> dict:
    """
    با جزئیات می‌گه ربات ادمینِ این کاناله یا نه - برایِ مرحله‌ی *ثبتِ* کانال.

    برخلافِ is_watchable که فقط بله/خیر می‌ده و کش می‌شه، این تابع همیشه تازه
    می‌پرسه و علتِ دقیق رو برمی‌گردونه تا بشه به ادمین پیامِ درست نشون داد.

    خروجی: {"ok": bool, "reason": str, "status": str|None}
      reason ∈ no_bot | bad_username | not_found | not_admin | error
    """
    key = norm(username)
    if not key:
        return {"ok": False, "reason": "bad_username", "status": None}
    if _bot is None:
        return {"ok": False, "reason": "no_bot", "status": None}

    try:
        me = await asyncio.wait_for(_bot.get_me(), timeout=PROBE_TIMEOUT)
        member = await asyncio.wait_for(
            _bot.get_chat_member(chat_id=f"@{key}", user_id=me.id),
            timeout=PROBE_TIMEOUT,
        )
    except Exception as e:
        text = str(e).lower()
        if "not found" in text or "chat not found" in text or "invalid" in text:
            return {"ok": False, "reason": "not_found", "status": None}
        # «ربات عضوِ چت نیست» هم از همین‌جا میاد
        return {"ok": False, "reason": "not_admin", "status": None,
                "detail": str(e)}

    status = getattr(member, "status", None)
    status = str(getattr(status, "value", status))
    ok = status in ADMIN_STATUSES
    # نتیجه رو در کشِ is_watchable هم بنشون تا اسکنِ بعدی دوباره نپرسه
    _watchable[key] = (ok, time.monotonic())
    return {"ok": ok, "reason": "" if ok else "not_admin", "status": status}


async def is_watchable(username: str) -> bool:
    """
    آیا ربات آپدیتِ پستِ این کانال رو دریافت می‌کنه؟

    تلگرام آپدیتِ channel_post رو فقط برایِ کانال‌هایی می‌فرسته که ربات عضو یا
    ادمینشونه. با getChatMember مستقیم می‌پرسیم - این یه فراخوانیِ Bot API ـه و
    *هیچ* هزینه‌ای برایِ اکانت‌ها نداره.

    نتیجه کش می‌شه، و اگه پستی از این کانال رسیده باشه اصلاً پروب نمی‌زنیم
    (رسیدنِ پست قطعی‌تر از هر پروبیه).
    """
    key = norm(username)
    if not key:
        return False

    cached = _watchable.get(key)
    if cached and (time.monotonic() - cached[1]) < WATCHABLE_TTL_SECONDS:
        return cached[0]

    if _bot is None:
        return False

    async with _lock:
        # ممکنه همین‌الان یه کوروتینِ دیگه جوابو گرفته باشه
        cached = _watchable.get(key)
        if cached and (time.monotonic() - cached[1]) < WATCHABLE_TTL_SECONDS:
            return cached[0]

        ok = False
        try:
            me = await asyncio.wait_for(_bot.get_me(), timeout=PROBE_TIMEOUT)
            member = await asyncio.wait_for(
                _bot.get_chat_member(chat_id=f"@{key}", user_id=me.id),
                timeout=PROBE_TIMEOUT,
            )
            status = getattr(member, "status", None)
            status = getattr(status, "value", status)   # Enum یا رشته
            ok = str(status) in ADMIN_STATUSES
        except Exception as e:
            # ربات عضو نیست، کانال خصوصیه، یا شبکه قطعه - همه یعنی «فعلاً نه»
            logger.debug("پروبِ دیده‌بانِ کانال برایِ @%s ناموفق: %s", key, e)
            ok = False

        _watchable[key] = (ok, time.monotonic())
        return ok


def snapshot() -> dict:
    """نمایِ خلاصه برایِ پنلِ وضعیت."""
    return {
        "bot_attached": _bot is not None,
        "channels_buffered": len(_posts),
        "posts_buffered": sum(len(b) for b in _posts.values()),
        "watchable": {u: v[0] for u, v in _watchable.items()},
    }
