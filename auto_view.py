import asyncio
import collections
import os
import math
import random
import time
from datetime import datetime, timedelta, timezone

from database import (
    get_auto_view_channels, get_auto_view_channel, update_auto_view_last_id,
    get_all_accounts, get_account,
    set_auto_view_watcher,
    add_scheduled_views, get_due_scheduled_views, cleanup_old_scheduled_views,
    stop_views_for_deleted_posts, FINISHED_ORDER_RETENTION_HOURS,
    purge_finished_orders, finished_order_cutoff_iso,
    update_auto_view_check_status,
    get_membership_map, set_membership, count_recent_joins,
    add_auto_view_cycle_state, get_due_auto_view_cycle_states, get_running_auto_view_cycle_states,
    finish_auto_view_cycle,
    cleanup_old_auto_view_cycle_states, cleanup_old_auto_view_message_viewers,
    get_done_account_ids_for_view_messages, apply_scheduled_view_results,
    get_pending_scheduled_view_counts, get_last_scheduled_view_fire_ats,
    increment_account_activity_bulk,
    claim_auto_view_cycle, release_auto_view_cycle_claim, postpone_auto_view_cycle,
    get_scheduled_view_counts_for_posts, purge_auto_view_message_viewers_for_post,
    purge_auto_view_post_history, get_finished_auto_view_cycle_states,
    get_active_auto_view_posts, purge_auto_view_posts, purge_orphan_auto_view_rows,
    set_account_frozen,
)
import telethon_handler
import resource_monitor
import memory_hygiene
# نگهبانِ فعالیتِ اکانت‌ها: قرنطینه‌یِ سشنِ تازه، دوره‌یِ گرم‌کردن و سقفِ فعالیتِ روزانه.
# ⚠️ این ایمپورت *نبود* و همین بزرگ‌ترین شکافِ «🎭 رفتارِ اکانت‌ها» بود: سنگین‌ترین
# مصرف‌کننده‌ی اکانت‌ها (همین موتور) نه سقفِ روزانه را چک می‌کرد و نه شمارنده را پر
# می‌کرد - پس سقف رویِ آمارِ ناقص حساب می‌شد و یه اکانتِ تازه می‌توانست همان روزِ
# اول صدها بازدید بزند. نگاه کن به _fire_due_views.
import activity_guard
from telethon_handler import (
    # ⚠️ check_new_posts عمداً حذف شد: پولینگِ پستِ جدید دیگه *اصلاً* از اکانت‌ها
    # انجام نمی‌شه. تشخیصِ پستِ جدید فقط از آپدیتِ رباته (channel_watch) و ثبتِ
    # کانال بدونِ ادمین‌بودنِ ربات ممکن نیست. اگه روزی دوباره این‌جا ایمپورت شد،
    # یعنی همون مسیرِ فریزکننده برگشته.
    view_post, join_channel_by_username, check_membership_by_username,
    check_posts_exist,
)
from telethon.errors import FloodWaitError
# دیده‌بانِ کانال از سمتِ ربات - جایگزینِ پولینگِ اکانت‌ها
import channel_watch
from scheduling import IMMEDIATE_GAP_MIN, IMMEDIATE_GAP_MAX, ACCOUNT_ACTION_TIMEOUT, utcnow

# هر چند ثانیه یه‌بار کانال‌های ثبت‌شده رو برای پستِ جدید چک کنه
POLL_INTERVAL_SECONDS = 45
# ⚠️ رندوم‌سازیِ فاصله‌ی حلقه‌ها (درصد). قبلاً هر حلقه دقیقاً هر ۴۵ ثانیه بیدار
# می‌شد - بدونِ ذره‌ای نوسان. یه اکانتِ واقعی هیچ‌وقت با دقتِ ساعتِ اتمی کار
# نمی‌کنه؛ همین یکنواختیِ کامل به‌تنهایی یه الگویِ ماشین‌خوانه. با ۰.۳۵ یعنی
# فاصله بینِ ~۲۹ تا ~۶۱ ثانیه می‌چرخه بدونِ اینکه میانگین عوض بشه.
POLL_JITTER_RATIO = float(os.getenv("AUTOVIEW_POLL_JITTER", "0.35"))


def _jittered_poll(base: float = None) -> float:
    """فاصله‌ی خوابِ حلقه با نوسانِ رندوم - نگاه کن به POLL_JITTER_RATIO."""
    base = POLL_INTERVAL_SECONDS if base is None else base
    spread = base * max(0.0, min(POLL_JITTER_RATIO, 0.9))
    return max(5.0, random.uniform(base - spread, base + spread))
# چندتا کانال هم‌زمان برایِ پستِ جدید چک بشن. با ۵۰ کانالِ ثبت‌شده، چکِ ترتیبی
# می‌تونه چند دقیقه طول بکشه و کلِ دورِ اسکن رو از POLL_INTERVAL عقب بندازه.
SCAN_CONCURRENCY = 6
# حداکثر زمانی که منتظرِ چکِ یک کانال می‌مونیم - بدونِ این، یه کانالِ کند/خراب
# می‌تونه یه اسلاتِ اسکن رو برایِ همیشه اشغال کنه
CHANNEL_SCAN_TIMEOUT = 60
# هر چند ثانیه یه‌بار سراغِ بازدیدهایی بره که موعدشون رسیده. قبلاً ۲۰ ثانیه بود، یعنی
# هر بازدید تا ۲۰ ثانیه دیرتر از موعدش اجرا می‌شد و رویِ دوره‌هایِ چنددقیقه‌ای این خطا
# جمع می‌شد؛ ۴ ثانیه عملاً خطا رو حذف می‌کنه.
FIRE_LOOP_INTERVAL_SECONDS = 4
# در هر پاس حداکثر چندتا بازدیدِ سررسیده از دیتابیس خونده بشه. قبلاً رویِ پیش‌فرضِ ۲۰۰
# بود؛ با ۵۰ کانال و دوره‌هایِ بزرگ، خودِ این سقف گلوگاه می‌شد و صف خالی نمی‌شد.
DUE_FETCH_LIMIT = 800
# هم‌زمانیِ اجرایِ بازدید - بینِ این دو حد، خودکار و بر اساسِ فشارِ واقعیِ سرور
# تنظیم می‌شه (نگاه کن به _current_concurrency). قبلاً کاملاً ترتیبی بود: هر بازدید
# تا ۴۵ ثانیه تایم‌اوت داشت، پس صدها بازدیدِ سررسیده ساعت‌ها طول می‌کشید.
# پایه = وقتی صف عقب نیست. سقف = وقتی صف عقب افتاده (مثلاً ۵۰ کانال که هم‌زمان پست
# می‌ذارن). هر بازدید از یه اکانتِ *متفاوت* زده می‌شه، پس نرخِ هر اکانت پایین می‌مونه و
# بالا بردنِ این عدد ریسکِ FloodWaitِ هر اکانت رو زیاد نمی‌کنه - فقط باید زیرِ سقفِ
# کانکشن‌هایِ کش‌شده (MAX_CACHED_CLIENTS) بمونه که با اختلافِ زیاد می‌مونه.
FIRE_CONCURRENCY_BASE = 12
FIRE_CONCURRENCY_CEILING = 48
FIRE_CONCURRENCY_MIN = 2
# صفِ سررسیده‌ی بیشتر از این تعداد یعنی «عقب افتادیم» و هم‌زمانی باید بره بالا
BACKLOG_SCALE_AT = 50
BACKLOG_FULL_AT = 1000
# آستانه‌هایِ «کِی سرور تحتِ فشاره» (تاخیرِ تیک، نسبتِ کانکشن‌ها، رم) دیگه این‌جا
# تکرار نمی‌شن: هر سه در resource_monitor.py و در *یک* جا تعریف شدن و همین ماژول
# هم همون عدد رو می‌خونه. سه‌تا آستانه‌ی موازی در سه فایل، دقیقاً همون چیزی بود که
# باعث می‌شد پنل «۸۱۱ مگابایت» نشون بده ولی موتورِ بازدید رم رو اصلاً نبینه.
# اگه یه اکانت برایِ یه بازدید شکست خورد، حداکثر چندتا اکانتِ دیگه برایِ همون بازدید
# امتحان بشه (قبلاً سخت‌کدشده ۲ بود)
MAX_VIEW_ATTEMPTS_PER_SLOT = 4
# بازدیدهایِ هر دوره داخلِ چند درصدِ ابتداییِ همون دوره پخش می‌شن (و مقدارِ رندومیِ
# فاصله‌ها) - این تضمین می‌کنه دوره‌ها هیچ‌وقت رویِ هم نیفتن و «هر N دقیقه X بازدید»
# دقیق دربیاد.
#
# ⚠️ این دو عدد از view_window_planner ایمپورت می‌شن، نه این‌جا تعریف. همون ماژول
# برایِ ادمین حساب می‌کنه «کلِ بازدید در چند ساعت تمام می‌شه»؛ اگه دو نسخه‌ی جدا
# داشتن، عددی که به ادمین نشون داده می‌شد با کاری که واقعاً اجرا می‌شه فرق می‌کرد.
from view_window_planner import ROUND_FILL_RATIO, ROUND_JITTER_RATIO
# اگه همه‌ی اکانت‌هایِ آزاد در این پاس مصرف شدن ولی اکانتِ واجدِ شرایط وجود داره،
# بازدید این‌قدر (ثانیه، رندوم) عقب می‌افته - «شکست‌خورده» علامت نمی‌خوره
RETRY_POSTPONE_MIN_SECONDS = 15
RETRY_POSTPONE_MAX_SECONDS = 45
# یه چرخه حداقل این‌قدر ثانیه باید در حالِ اجرا بوده باشه تا «تمام‌شده» حساب بشه -
# جلویِ باگی رو می‌گیره که چرخه بلافاصله بعدِ ساختن «تمام» اعلام می‌شد و چرخه‌ی بعدی
# از زمانِ غلط برنامه‌ریزی می‌شد
MIN_CYCLE_RUN_SECONDS = 45
# لیستِ اکانت‌ها و نقشه‌یِ عضویت این‌قدر ثانیه کش می‌شن. بدونِ کش، حلقه‌یِ اجرا هر ۴
# ثانیه کلِ جدولِ اکانت‌ها (و یه نقشه‌یِ عضویت به‌ازایِ هر کانال) رو از دیسک می‌خوند -
# با ۵۰ کانال یعنی صدها کوئریِ تکراری در دقیقه، فقط برایِ داده‌ای که تقریباً ثابته.
ACCOUNTS_CACHE_TTL = 30
MEMBERSHIP_CACHE_TTL = 60
# در هر اسکن، حداکثر چندتا پستِ *جدید* از یه کانال پردازش بشه. بدونِ این سقف، کانالی
# که یک‌دفعه ۱۰۰ پست منتشر کرده (یا کانالی که تازه ثبت شده و last_id عقبه) در یک دور
# ۱۰۰×۲۰۰ = ۲۰٬۰۰۰ ردیفِ بازدید می‌ساخت و کلِ ربات رو قفل می‌کرد. بقیه‌ی پست‌ها در
# دورهایِ بعدی پردازش می‌شن، پس چیزی از دست نمی‌ره - فقط بار پخش می‌شه.
MAX_NEW_POSTS_PER_SCAN = 5
# ===================== نگهبانِ خودترمیم =====================
# هر حلقه در هر دور «ضربانِ قلب» ثبت می‌کنه. اگه یه حلقه بیشتر از سقفِ خودش ضربان
# نزنه، یعنی جایی گیر کرده (مثلاً یه await که هیچ‌وقت برنمی‌گرده) - نگهبان همون یک
# حلقه رو کنسل و از نو راه می‌ندازه، بدونِ اینکه به بقیه‌ی ربات دست بزنه.
# این همون چیزیه که جلویِ «مجبورم Railway رو ری‌استارت کنم» رو می‌گیره.
WATCHDOG_INTERVAL_SECONDS = 30
# سقفِ بی‌ضربانیِ هر حلقه (ثانیه) - سخاوتمندانه انتخاب شدن تا یه دورِ کندِ طبیعی
# اشتباهاً «گیرکرده» تشخیص داده نشه
HEARTBEAT_LIMIT = {
    "scan": 60 * 6,     # اسکن: ۵۰ کانال حتی با کندی خیلی زیر این تمام می‌شه
    "fire": 60 * 3,     # اجرا: تیکش ۴ ثانیه‌ست، پس ۳ دقیقه یعنی قطعاً گیر کرده
    "warmup": 60 * 10,  # گرم‌کردن: کندترین و کم‌اهمیت‌ترین
    "deleted": 60 * 8,  # پاسبانِ پستِ پاک‌شده: چندتا کوئریِ سبک + چکِ کانال‌ها
}
# ردیف‌هایِ بازدیدِ خودکار فقط یه شبکه‌یِ ایمنیِ زمانی لازم دارن: پاک‌سازیِ *اصلی*
# پست‌محوره و به‌محضِ تمام‌شدنِ دوره‌ی همون پست انجام می‌شه (نگاه کن به
# _purge_finished_posts). این عدد فقط برایِ ردیف‌هایِ یتیم است.
RETENTION_DAYS = 10
CLEANUP_INTERVAL_SECONDS = 3600
# نگه‌داریِ کش‌هایِ درون‌حافظه‌ای بارها بیشتر از پاکسازیِ دیتابیس اجرا می‌شه: کش‌ها
# بینِ دو پاکسازیِ ساعتی هم رشد می‌کنن، و اگه رم به سقفِ نرم برسه باید *همون موقع*
# واکنش نشان داد نه یه ساعت بعد. این تابع ارزونه (چیدنِ چند دیکشنری + malloc_trim).
MEMORY_SWEEP_INTERVAL_SECONDS = int(os.getenv("MEMORY_SWEEP_INTERVAL_SECONDS", "300"))
# ---------------------------------------------------------------------------
# سیاستِ بازدیدِ هر پست: فقط و فقط **یک دوره‌ی ۱۰۰٪**
# ---------------------------------------------------------------------------
# ⚠️ چیزی که کاملاً حذف شد: «نردبانِ کاهشیِ بازدید». قبلاً بعدِ تمام‌شدنِ بازدیدهایِ
# یه پست، دورِ دیگه‌ای با ۲۰٪ بازدیدِ کمتر اجرا می‌شد (۱۰۰٪ → ۸۰٪ → ۶۰٪ → ۴۰٪ →
# ۲۰٪ → صفر) و یه ماژولِ جدا (view_cycle_planner) فاصله‌ی این دوره‌ها را حساب
# می‌کرد: پایه‌ی رشدی، رندومی، پرهیز از ساعاتِ سحر، اجرایِ زودترِ فرصت‌طلبانه...
# کلِ آن سیستم - کد، ماژول، تنظیمات و متن‌هایش - برداشته شد.
#
# قانونِ فعلی، کلِ منطقِ بازدیدِ پست در سه خط:
#   ۱) هر پستِ جدید *یک* دوره می‌گیرد: ۱۰۰٪ سفارش (همون عددی که ادمین ثبت کرده).
#   ۲) به‌محضِ تمام‌شدنِ ردیف‌هایِ همون یک دوره، پست **کنار گذاشته می‌شود**: وضعیتش
#      finished می‌شود و از آن لحظه به بعد هیچ بازدیدی برایش ثبت نمی‌شود.
#   ۳) تاریخچه‌یِ «کدام اکانت این پست را دید» همان لحظه پاک می‌شود، چون دوره‌ی
#      بعدی‌ای وجود ندارد که به آن نیاز داشته باشد.
#
# چیزی که *نگه* داشته شد: هماهنگی با «📊 منابعِ سرور». اگه لحظه‌ی شروعِ همین یک
# دوره سرور اشباع باشد، شروعش کمی عقب می‌افتد (بسته نمی‌شود) - نگاه کن به
# DEFER_AT_PRESSURE. این تنها جایی است که فشارِ سرور در زمان‌بندیِ پست دخالت می‌کند.
# ---------------------------------------------------------------------------

# اگه *هیچ* اکانتِ سالمی موجود نباشه (همه لاگ‌اوت/فریز)، دوره بسته نمی‌شه؛ این‌قدر
# دقیقه بعد دوباره تلاش می‌شه. بدونِ این، یه قطعیِ موقتیِ اکانت‌ها بازدیدِ پست رو
# برایِ همیشه می‌کشت.
VIEW_CYCLE_NO_ACCOUNT_RETRY_MINUTES = 15

# ⚠️ باگِ رفع‌شده - «یه صفرِ گذرا کلِ بازدیدِ پست را می‌کشت»: اگه در لحظه‌ی شروع همه‌ی
# اکانت‌هایِ واجدِ شرایط این پست را دیده باشن (ظرفیتِ تازه = صفر)، پست فوراً کنار
# گذاشته نمی‌شود؛ چند بار با فاصله صبر می‌کند، چون صفر می‌تواند کاملاً گذرا باشد
# (موجِ FloodWait، ری‌استارتِ سرور، پاکسازیِ سشن).
VIEW_CYCLE_STARVED_MAX_RETRIES = int(os.getenv("VIEW_CYCLE_STARVED_MAX_RETRIES", "4"))
VIEW_CYCLE_STARVED_RETRY_MINUTES = int(os.getenv("VIEW_CYCLE_STARVED_RETRY_MINUTES", "45"))

# اگه ساختنِ ردیف‌هایِ دوره شکست بخوره، این‌قدر ثانیه بعد از نو تلاش می‌شه
VIEW_CYCLE_CLAIM_RETRY_SECONDS = 60

# ---------------------------------------------------------------------------
# عقب انداختنِ شروع وقتی سرور اشباع است - تنها اتصالِ زمان‌بندی به منابعِ سرور
# ---------------------------------------------------------------------------
# فشار عددِ ۰..۱ از resource_monitor است؛ *همون* عددی که پنلِ «📊 منابعِ سرور»
# نشان می‌دهد. اگه همین لحظه سرور اشباع باشد، ساختنِ چند هزار ردیفِ جدید رویِ صفِ
# سرریزشده فقط اوضاع را بدتر می‌کند - پس شروعِ دوره کمی عقب می‌افتد.
DEFER_AT_PRESSURE = float(os.getenv("VIEW_CYCLE_DEFER_AT_PRESSURE", "0.85"))
DEFER_MIN_MINUTES = int(os.getenv("VIEW_CYCLE_DEFER_MIN_MINUTES", "20"))
DEFER_MAX_MINUTES = int(os.getenv("VIEW_CYCLE_DEFER_MAX_MINUTES", "75"))
# محافظِ ضدِ «برایِ همیشه عقب انداختن»: بعد از این تعداد عقب‌اندازیِ پشتِ‌سرِهم،
# دوره هرچه باشد اجرا می‌شود. بدونِ این، یه سرورِ همیشه-شلوغ می‌توانست بازدیدِ یه
# پست را تا ابد معلق نگه دارد.
DEFER_MAX_STREAK = int(os.getenv("VIEW_CYCLE_DEFER_MAX_STREAK", "6"))

# بعدِ تمام‌شدنِ دوره‌ی یه پست، گزارشش این‌قدر ساعت نگه داشته می‌شه (تا در پنل دیده
# بشه) و بعد کلِ ردِ پایش - ردیف‌هایِ زمان‌بندی + خودِ ردیفِ وضعیت - پاک می‌شه.
# تاریخچه‌یِ اکانت‌ها *بلافاصله* بعدِ پایان پاک می‌شه، چون دیگه هیچ دوره‌ای در راه نیست.
# هم‌تراز با FINISHED_ORDER_RETENTION_HOURSِ دیتابیس (۲۴ ساعت): گزارشِ پستِ
# تمام‌شده تا همون مرز در «🗂 سفارش‌ها ← ✅ تمام‌شده» دیده می‌شه و بعد می‌ره.
# ⚠️ قبلاً ۶ ساعت بود - یعنی سفارشِ بازدیدِ تمام‌شده ۱۸ ساعت *زودتر* از
# ری‌اکشن/شیر ناپدید می‌شد و سه بخش سه عمرِ متفاوت داشتن.
VIEW_CYCLE_PURGE_GRACE_HOURS = float(
    os.getenv("VIEW_CYCLE_PURGE_GRACE_HOURS", str(FINISHED_ORDER_RETENTION_HOURS)))
# شبکه‌یِ ایمنیِ نهایی: ردیفِ پست‌هایِ تمام‌شده بعد از این چند روز حتماً پاک می‌شن
VIEW_CYCLE_RETENTION_DAYS = 10

# ===========================================================================
#                       پاسبانِ «پستِ پاک‌شده»
# ---------------------------------------------------------------------------
# مسئله: بازدیدِ هر پست می‌تونه صدها ردیفِ زمان‌بندی‌شده داشته باشه که تا ساعت‌ها بعد
# اجرا می‌شن. اگه ادمینِ کانال همون پست رو پاک کنه، ربات تا حالا هیچ‌وقت نمی‌فهمید:
#   • بازدیدها روی یه پستِ ناموجود ادامه پیدا می‌کرد (بی‌فایده)
#   • هر تلاش از سقفِ فعالیتِ روزانه‌ی یه اکانتِ سالم خرج می‌شد
#   • صفِ سررسیده الکی شلوغ می‌موند - و چون طولِ صف یکی از مؤلفه‌هایِ فشارِ سرور
#     در resource_monitor ـه، مستقیم بقیه‌ی بخش‌ها رو هم کند می‌کرد
#   • ردیف‌هاش (زمان‌بندی + تاریخچه‌ی بازدیدکننده‌ها + ردیفِ وضعیت) تا روزها در
#     دیتابیس می‌موندن و در کش‌هایِ حافظه هم جا می‌گرفتن
#
# راه‌حل: یه حلقه‌ی مستقل (زیرِ نظرِ همون نگهبانِ خودترمیم) که پست‌هایِ *فعال* رو
# دوره‌ای از خودِ تلگرام می‌پرسه و به‌محضِ تاییدِ پاک‌شدن، بازدیدشون رو متوقف و کلِ
# ردِ پایشون رو از دیتابیس حذف می‌کنه.
#
# ⚠️ اصلِ طراحی: «پاک‌شده» یه حکمِ *حذفیه*، پس فقط با شاهدِ قطعی صادر می‌شه -
# دو مشاهده‌ی پشتِ‌سرِهمِ «نبود» (DELETED_CONFIRM_STRIKES). خطایِ شبکه، FloodWait،
# اکانتِ ناظرِ خراب یا خروجیِ ناتراز هیچ‌وقت «پاک‌شده» ترجمه نمی‌شن.
# ===========================================================================
# هر چند ثانیه یه دورِ چک انجام بشه
DELETED_CHECK_INTERVAL_SECONDS = int(os.getenv("VIEW_DELETED_CHECK_INTERVAL", "90"))
# هر پستِ سالم بعدِ این‌قدر ثانیه دوباره چک می‌شه (چکِ پستِ سالم عجله‌ای نداره)
DELETED_RECHECK_SECONDS = int(os.getenv("VIEW_DELETED_RECHECK_SECONDS", "300"))
# پستی که یه بار «نبود» دیده شده، این‌قدر ثانیه بعد دوباره چک می‌شه - یعنی عملاً
# همون دورِ بعد. تاییدِ سریع مهمه چون تا تاییدِ نهایی بازدیدها ادامه دارن.
DELETED_SUSPECT_RECHECK_SECONDS = int(os.getenv("VIEW_DELETED_SUSPECT_RECHECK", "20"))
# سقفِ پستِ چک‌شده در هر دور - جلویِ یه دورِ غول رو می‌گیره
DELETED_CHECK_MAX_POSTS = int(os.getenv("VIEW_DELETED_CHECK_MAX_POSTS", "400"))
# چند کانال هم‌زمان چک بشن (هر کانال یه درخواستِ سبکِ خواندنی به‌ازایِ هر ۱۰۰ پست)
DELETED_CHECK_CONCURRENCY = int(os.getenv("VIEW_DELETED_CHECK_CONCURRENCY", "3"))
# سقفِ زمانیِ چکِ یه کانال
DELETED_CHECK_TIMEOUT = 60
# چند مشاهده‌یِ پشتِ‌سرِهمِ «نبود» لازمه تا حکمِ پاک‌شدن قطعی بشه
DELETED_CONFIRM_STRIKES = max(1, int(os.getenv("VIEW_DELETED_CONFIRM_STRIKES", "2")))
# هماهنگی با «📊 منابعِ سرور»: اگه سرور همین الان اشباع باشه، این دور رد می‌شه.
# چکِ پاک‌شدن یه کارِ *بهینه‌سازیه*، نه سفارشِ مشتری - پس اولین چیزیه که در مضیقه
# عقب می‌کشه. یه دور دیرتر انجام‌شدنش هیچ چیزی رو خراب نمی‌کنه.
DELETED_SKIP_AT_PRESSURE = float(os.getenv("VIEW_DELETED_SKIP_AT_PRESSURE", "0.9"))
# کلیدِ پستِ تاییدشده این‌قدر ثانیه در حافظه می‌مونه - فقط برایِ ردکردنِ ردیف‌هایِ
# «در پرواز» (اونایی که همون لحظه از دیتابیس خونده شده بودن). بعدش دیگه نه ردیفی
# مونده نه وضعیتی، پس نگه‌داشتنش فقط حافظه می‌گیره.
DELETED_MEMORY_TTL_SECONDS = int(os.getenv("VIEW_DELETED_MEMORY_TTL", "900"))
# سقفِ کلیدهایِ در-حافظه‌ی این بخش (در پاکسازیِ حافظه هم اعمال می‌شه)
DELETED_MEMORY_MAX_KEYS = int(os.getenv("VIEW_DELETED_MEMORY_MAX_KEYS", "2000"))

# ===================== تنظیماتِ سرعتِ جوین =====================
# این اعداد محافظه‌کارانه انتخاب شدن. تلگرام سقفِ دقیقِ جوین رو منتشر نکرده، پس این‌ها
# بر اساسِ رفتارِ ایمنِ شناخته‌شده تنظیم شدن، نه یه عددِ رسمی. اگه بازم FloodWait گرفتی،
# JOIN_GAP رو بیشتر کن؛ اگه خیلی کند بود، کمترش کن.
#
# فاصله‌ی بینِ دو جوینِ پشتِ‌سرِهم (کلِ ربات، نه فقط یه کانال) - رندوم تا الگو یکنواخت نباشه.
# میانگینِ ~۳۷ ثانیه یعنی حدودِ ۹۵ جوین در ساعت در کلِ ربات.
JOIN_GAP_MIN = 25
JOIN_GAP_MAX = 50
# سقفِ جوین در ساعت برایِ **هر کانال** - جلویِ هجومِ ناگهانیِ عضو به یه کانال رو می‌گیره
# (چیزی که برایِ تلگرام مشکوک‌ترین الگوئه)
JOIN_MAX_PER_HOUR_PER_CHANNEL = 60
# بعدِ گرفتنِ FloodWait، علاوه بر ثانیه‌هایی که تلگرام گفته، این‌قدر هم اضافه صبر می‌کنیم
JOIN_FLOOD_EXTRA_SECONDS = 120
# هر دورِ حلقه حداکثر چندتا «چکِ عضویت» انجام بشه (اینا سبک‌ترن، فقط خواندنی) - قبلاً
# فقط برایِ کانال‌هایی که join_before_view روشن بود انجام می‌شد؛ حالا برایِ *همه‌ی*
# کانال‌ها انجام می‌شه (چون as_follower/other همیشه باید بر اساسِ عضویتِ واقعی باشه)،
# پس عددش رو کمی بیشتر کردیم تا همگرایی برایِ کانال‌هایِ بیشتر هم معقول بمونه
MEMBERSHIP_CHECKS_PER_CYCLE = 5
# وضعیتِ عضویتِ ذخیره‌شده بعدِ چند ساعت دوباره چک بشه.
# ⚠️ از ۱۲ ساعت به یک هفته رفت. عضویتِ یه اکانت در یه کانال چیزی نیست که هر ۱۲
# ساعت عوض بشه، ولی همین عدد باعث می‌شد کلِ استخر دو بار در روز مجبور به
# getParticipant بشه: با ۵۰۰۰ اکانت یعنی ۱۰٬۰۰۰ چک در روز در برابرِ ظرفیتِ
# ۹٬۶۰۰ - حلقه‌ی گرم‌کردن هیچ‌وقت بی‌کار نمی‌شد. این تغییر به‌تنهایی بار رو
# ۱۴ برابر کم می‌کنه.
MEMBERSHIP_RECHECK_HOURS = float(os.getenv("MEMBERSHIP_RECHECK_HOURS", "168"))
# اگه یه اکانت چند بار پشتِ‌سرِهم توی جوین‌شدن شکست خورد، دیگه سراغش نریم
MAX_JOIN_FAILURES = 3

# اگه همه‌ی اکانت‌هایِ واجدِ شرایط به سقفِ روزانه‌شون رسیده باشن، بازدیدِ سررسیده
# «شکست‌خورده» علامت نمی‌خوره - فقط این‌قدر (رندوم، به دقیقه) عقب می‌افته تا بعداً،
# یا فردا که شمارنده صفر می‌شه، انجام بشه. نگاه کن به postpone_scheduled_view.
CAP_POSTPONE_MIN_MINUTES = 30
CAP_POSTPONE_MAX_MINUTES = 75

_running = False
# رویدادِ توقف - تا نگهبان و حلقه‌ها *فوراً* به دستورِ خاموشی جواب بدن، نه بعدِ تمام
# شدنِ خوابِ فعلی‌شون. بدونِ این، خاموش‌شدنِ ربات می‌تونست تا یه تیکِ کاملِ نگهبان
# (۳۰ ثانیه) طول بکشه - که موقعِ دیپلویِ Railway یعنی kill شدنِ اجباریِ پردازه، و
# همون چیزیه که کارِ نیمه‌کاره و وضعیتِ خراب جا می‌ذاره.
_stop_event: "asyncio.Event | None" = None


def _get_stop_event() -> asyncio.Event:
    global _stop_event
    if _stop_event is None:
        _stop_event = asyncio.Event()
    return _stop_event


async def _sleep_or_stop(seconds: float) -> None:
    """خوابِ قابلِ قطع: به‌محضِ درخواستِ توقف بیدار می‌شه."""
    try:
        await asyncio.wait_for(_get_stop_event().wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass

# زمانِ آخرین جوین و فاصله‌ی رندومِ انتخاب‌شده برایِ جوینِ بعدی (در-حافظه، کلِ ربات).
# بعدِ ری‌استارت صفر می‌شه، ولی سقفِ ساعتیِ هر کانال (که از دیتابیس خونده می‌شه) همچنان
# جلویِ هجوم رو می‌گیره.
_last_join_at = 0.0
_next_join_gap = 0.0
# تا این لحظه (time.monotonic) هیچ جوینی نزن - بعدِ FloodWait ست می‌شه
_global_join_pause_until = 0.0


def _join_slot_available() -> bool:
    """آیا الان نوبتِ جوینِ بعدی رسیده؟ (فاصله‌ی رندوم + مکثِ بعدِ FloodWait)"""
    now = time.monotonic()
    if now < _global_join_pause_until:
        return False
    return (now - _last_join_at) >= _next_join_gap


def _consume_join_slot():
    """بعدِ هر جوین صدا زده می‌شه - فاصله‌ی بعدی رو رندوم انتخاب می‌کنه"""
    global _last_join_at, _next_join_gap
    _last_join_at = time.monotonic()
    _next_join_gap = random.uniform(JOIN_GAP_MIN, JOIN_GAP_MAX)


def _pause_joins(seconds: float):
    """بعدِ FloodWait، کلِ جوین‌ها رو برایِ این مدت متوقف می‌کنه"""
    global _global_join_pause_until
    _global_join_pause_until = max(_global_join_pause_until, time.monotonic() + seconds)


def _needs_recheck(record: dict) -> bool:
    """آیا وضعیتِ عضویتِ ذخیره‌شده کهنه شده و باید دوباره چک بشه؟"""
    if not record or not record.get("checked_at"):
        return True
    try:
        checked = datetime.fromisoformat(record["checked_at"])
    except Exception:
        return True
    return (utcnow() - checked) > timedelta(hours=MEMBERSHIP_RECHECK_HOURS)


def _in_flood_cooldown(record: dict) -> bool:
    """آیا این اکانت هنوز توی دوره‌ی مکثِ بعدِ FloodWait هست؟"""
    if not record or not record.get("flood_until"):
        return False
    try:
        return utcnow() < datetime.fromisoformat(record["flood_until"])
    except Exception:
        return False


async def _warmup_joins():
    """
    فرآیندِ «گرم‌کردن»: دو کارِ کاملاً جدا انجام می‌ده -

    ۱) چکِ عضویت (سبک، فقط خواندنی): برایِ **همه‌ی** کانال‌هایِ بازدیدِ خودکار (فارغ از
       اینکه join_before_view روشنه یا نه) - چون as_follower/other که موقعِ ثبتِ بازدید
       استفاده می‌شه (نگاه کن به _fire_due_views) باید همیشه بر اساسِ عضویتِ *واقعیِ*
       اکانت باشه، نه فقط برایِ کانال‌هایی که این سوییچِ خاص روشنه.
    ۲) جوینِ خودکار: فقط برایِ کانال‌هایی که join_before_view روشنه - این همون رفتارِ
       قبلیه، دست‌نخورده.

    ⚠️ اینجا بزرگ‌ترین منبعِ فریزِ کلِ پروژه بود. نسخه‌ی قبلی صریحاً می‌گفت «این مسیر
    کاملاً از سقفِ فعالیتِ روزانه جدا شده» - و همین دقیقاً مشکل بود. یعنی چکِ عضویت
    از *هیچ‌کدوم* از محافظ‌ها رد نمی‌شد: نه دروازه‌ی فاصله‌ی حداقلی، نه سقفِ روزانه،
    نه ساعاتِ سکوتِ شبانه، نه قرنطینه، نه مکثِ بعدِ FloodWait.

    نتیجه‌اش این بود: به‌محضِ ثبتِ *یک* کانال - حتی بدونِ یک پست - این حلقه شروع
    می‌کرد به بیدارکردنِ کلِ استخرِ اکانت‌ها، وصل‌کردنشون به تلگرام و پرسیدنِ
    «عضوِ این کانالم؟»، شبانه‌روزی. با ۵۰۰۰ اکانت و بازچکِ ۱۲ ساعته، حلقه هیچ‌وقت
    بی‌کار نمی‌شد. دقیقاً همون چیزی که دیده شده بود: بدونِ کانال هیچ مشکلی نیست،
    با یه کانالِ خالی اکانت‌ها فریز می‌شن.

    حالا چکِ عضویت هم یه «کارِ شمارش‌شدنی»ه مثلِ هر کارِ دیگه‌ای و کاملاً از
    activity_guard رد می‌شه.
    """
    channels = await get_auto_view_channels()
    if not channels:
        return

    # کانال‌هایی که دیگه ثبت نیستن، بافرِ دیده‌بانشون هم آزاد بشه
    try:
        channel_watch.prune_to([c.get("channel_username") for c in channels])
    except Exception:
        pass

    accounts = _eligible_from(await _cached_accounts())
    if not accounts:
        return

    guard_settings = await activity_guard.get_settings()
    guard_usage = await activity_guard.daily_usage_map()

    # ساعاتِ سکوت: کلِ گرم‌کردن تعطیل. یه کاربرِ واقعی ساعتِ ۴ صبح بیدار نمی‌شه
    # که ببینه عضوِ یه کاناله یا نه.
    if activity_guard.in_quiet_hours(guard_settings):
        return

    # ---- مرحله‌ی ۱: چکِ عضویت برایِ همه‌ی کانال‌ها ----
    check_targets = list(channels)
    random.shuffle(check_targets)
    checks_left = MEMBERSHIP_CHECKS_PER_CYCLE

    for ch in check_targets:
        if checks_left <= 0:
            break
        username = ch["channel_username"]
        country_code = ch.get("country_code")
        if country_code:
            pool = [a for a in accounts if a["country_code"] == country_code]
        else:
            pool = list(accounts)
        if not pool:
            continue

        membership = await _cached_membership(username)
        unknown = [
            a for a in pool
            if _needs_recheck(membership.get(a["id"])) and not _in_flood_cooldown(membership.get(a["id"]))
            # ⚠️ دروازه‌ی اصلی: اکانتی که نوبتش نرسیده، سقفش پر شده، در قرنطینه‌ست
            # یا کولدان داره اصلاً انتخاب نمی‌شه
            and activity_guard.is_ready(a, guard_settings, guard_usage)
        ]
        random.shuffle(unknown)
        for acc in unknown[:checks_left]:
            # وضعیت می‌تونه وسطِ همین حلقه عوض شده باشه (چکِ قبلی FloodWait خورده)
            if not activity_guard.is_ready(acc, guard_settings, guard_usage):
                continue
            res = await check_membership_by_username(acc["session_file"], username)
            # چکِ عضویت هم یه فعالیتِ واقعیه: هم دروازه‌ی فاصله بسته بشه، هم از
            # سقفِ روزانه کم بشه. بدونِ این، همون حلقه‌ی بی‌وقفه‌ی قبلی برمی‌گرده.
            activity_guard.mark_used(acc["id"], weight=1, acc=acc, settings=guard_settings)
            await activity_guard.note_actions(acc["id"], 1, guard_usage)
            if res.get("status") == "flood":
                activity_guard.report_flood(acc["id"], res.get("seconds", 0) or 0)
            if res["status"] == "member":
                await set_membership(acc["id"], username, "member")
            elif res["status"] == "not_member":
                await set_membership(acc["id"], username, "not_member")
            elif res["status"] == "flood":
                until = (utcnow() + timedelta(seconds=res.get("seconds", 60))).isoformat(timespec="seconds")
                await set_membership(acc["id"], username, "unknown", flood_until=until)
            # کشِ عضویت باید فوراً باطل بشه، وگرنه تا TTL بعدی همین اکانت‌ها دوباره
            # «نامعلوم» دیده می‌شن و بی‌دلیل دوباره چک می‌شن
            _invalidate_membership(username)
            checks_left -= 1
            if checks_left <= 0:
                break

    # ---- مرحله‌ی ۲: جوینِ خودکار - فقط برایِ کانال‌هایی که join_before_view روشنه ----
    join_targets = [c for c in channels if c.get("join_before_view")]
    if not join_targets:
        return
    random.shuffle(join_targets)

    for ch in join_targets:
        username = ch["channel_username"]
        country_code = ch.get("country_code")
        if country_code:
            pool = [a for a in accounts if a["country_code"] == country_code]
        else:
            pool = list(accounts)
        if not pool:
            continue

        if not _join_slot_available():
            continue

        # سقفِ ساعتیِ همین کانال
        since = (utcnow() - timedelta(hours=1)).isoformat(timespec="seconds")
        if await count_recent_joins(username, since) >= JOIN_MAX_PER_HOUR_PER_CHANNEL:
            continue

        membership = await _cached_membership(username)
        follower_percent = _follower_share(ch)
        target_total = int(ch.get("view_count_max") or ch.get("view_count") or 0)
        target_members = math.ceil(target_total * follower_percent / 100)
        current_members = sum(
            1 for a in pool
            if (membership.get(a["id"]) or {}).get("status") == "member"
        )
        # بیشتر از سهمِ Followers عضو نکن؛ وگرنه استخرِ Other از بین می‌رود.
        if current_members >= target_members:
            continue
        candidates = [
            a for a in pool
            if (membership.get(a["id"]) or {}).get("status") == "not_member"
            and not _in_flood_cooldown(membership.get(a["id"]))
            and (membership.get(a["id"]) or {}).get("fail_count", 0) < MAX_JOIN_FAILURES
            # ⚠️ جوین خطرناک‌ترین عملیاتِ کلِ پروژه‌ست (منبعِ اصلیِ PEER_FLOOD) و
            # دقیقاً همونی بود که هیچ محافظی نداشت
            and activity_guard.is_ready(a, guard_settings, guard_usage)
        ]
        if not candidates:
            continue

        acc = random.choice(candidates)
        res = await join_channel_by_username(acc["session_file"], username)
        # جوین یه فعالیتِ سنگینه - وزنش بیشتر از یه چکِ ساده حساب می‌شه
        activity_guard.mark_used(acc["id"], weight=1, acc=acc, settings=guard_settings)
        await activity_guard.note_actions(acc["id"], 1, guard_usage)
        if res.get("status") == "flood":
            activity_guard.report_flood(acc["id"], res.get("seconds", 0) or 0)
        elif res.get("status") == "peer_flood" or telethon_handler.is_peer_flood(
                res.get("message", "")):
            activity_guard.report_peer_flood(acc["id"])

        _invalidate_membership(username)
        if res["status"] in ("joined", "already"):
            await set_membership(acc["id"], username, "member", mark_joined=(res["status"] == "joined"))
            _consume_join_slot()
        elif res["status"] == "flood":
            secs = res.get("seconds", 300) + JOIN_FLOOD_EXTRA_SECONDS
            until = (utcnow() + timedelta(seconds=secs)).isoformat(timespec="seconds")
            await set_membership(acc["id"], username, "not_member", flood_until=until)
            # کلِ جوین‌ها رو عقب بکش - این یعنی داریم به سقفِ تلگرام نزدیک می‌شیم
            _pause_joins(secs)
            print(f"⏱️  FloodWait روی جوینِ @{username} - جوین‌ها {int(secs)} ثانیه متوقف شد")
        else:
            await set_membership(acc["id"], username, "not_member", bump_fail=True)
            _consume_join_slot()

        # فقط یه جوین در هر دور، در کلِ همه‌ی کانال‌ها
        break




# ===================== وضعیتِ زنده (برایِ «📊 منابعِ سرور») =====================
# این بخش تنها منبعِ حقیقتِ «بازدیدِ خودکار الان در چه حالیه» است و مستقیم توسطِ
# پنلِ منابعِ سرور در bot.py خونده می‌شه - تا وقتی ادمین می‌پرسه «چرا گیر کرده؟»
# جواب با عدد و مدرک باشه، نه حدس.
_runtime = {
    "started_at": None,
    "last_scan_at": None,
    "last_scan_seconds": 0.0,
    "last_scan_channels": 0,
    "scan_errors": {},          # channel_username -> پیامِ خطا
    "last_fire_at": None,
    "tick_lag": 0.0,
    "tick_lag_max": 0.0,
    "concurrency": FIRE_CONCURRENCY_BASE,
    "last_batch": 0,
    "backlog_capped": False,
    "total_done": 0,
    "total_failed": 0,
    "total_postponed": 0,
    "capacity_notes": {},       # "@channel #msg" -> متنِ کمبودِ اکانت
    "deferred_posts": {},       # channel -> چندتا پست به دورِ بعد موکول شد
    "bot_watched": set(),       # کانال‌هایی که ربات خودش می‌بینه (بدونِ مصرفِ اکانت)
    "loops_alive": {"scan": False, "fire": False, "warmup": False, "deleted": False},
    "heartbeat": {},            # اسمِ حلقه -> آخرین time.monotonic
    "restarts": {},             # اسمِ حلقه -> چند بار خودترمیم شده
    "last_restart_at": None,
    # ---- شمارنده‌هایِ بازدیدِ پست‌ها (هر پست فقط یک دوره‌ی ۱۰۰٪) ----
    "posts_started": 0,         # چند پست دوره‌ی ۱۰۰٪شون شروع شده
    "posts_finished": 0,        # چند پست کامل شد و کنار گذاشته شد
    "posts_purged": 0,          # چند پست ردِ پایش از دیتابیس پاک شد
    "cycle_last_error": None,
    # ---- هماهنگی با «📊 منابعِ سرور» ----
    "pressure": 0.0,            # فشارِ سرور در آخرین تصمیم (از resource_monitor)
    "pressure_worst": None,     # کدام مؤلفه فشار را ساخته
    "cycles_deferred": 0,       # چند پست به‌خاطرِ فشارِ سرور دیرتر شروع شد
    "cycles_starved": 0,        # چند پست به‌خاطرِ کمبودِ اکانتِ تازه عقب افتاد
    "cap_postponed": 0,         # چند بازدید به‌خاطرِ سقفِ فعالیتِ روزانه عقب افتاد
    # ---- تفکیکِ دقیقِ «چرا عقب افتاد» (تمدیدِ هوشمندِ سفارش) ----
    "postpone_reasons": {},     # علت -> تعداد (cap/hourly/paced/quiet/flood/...)
    "resolve_deferred": 0,      # چند بازدید به‌خاطرِ سهمیه‌یِ resolve عقب افتاد
    "flood_events": 0,          # چند بار FloodWait خوردیم و اکانت استراحت داده شد
    "frozen_detected": 0,       # چند اکانت وسطِ کار «فریز» تشخیص داده شد
    # ---- پاسبانِ پستِ پاک‌شده ----
    "posts_deleted": 0,          # چند پست «پاک‌شده» تشخیص داده شد و بازدیدش متوقف شد
    "deleted_views_cancelled": 0,  # چند بازدیدِ باقی‌مونده‌ی همون پست‌ها لغو شد
    "deleted_rows_freed": 0,     # مجموعِ ردیف‌هایِ دیتابیسی که آزاد شد
    "deleted_checks": 0,         # چند پست تا حالا چک شد
    "deleted_rounds": 0,         # چند دورِ چک انجام شد
    "deleted_last_check_at": None,
    "deleted_last_seconds": 0.0,
    "deleted_last_batch": 0,     # در آخرین دور چندتا پست چک شد
    "deleted_skipped_pressure": 0,  # چند دور به‌خاطرِ فشارِ سرور رد شد
    "deleted_check_errors": {},  # channel_username -> پیامِ خطا
    "orphan_rows_purged": 0,     # ردیف‌هایِ یتیمِ پاک‌شده در پاکسازیِ ساعتی
    # ---- کانالی که از لیست حذف شده ----
    "revoked_channels": 0,       # چند کانال از لیست حذف و فوراً متوقف شد
    "revoked_views_blocked": 0,  # چند بازدیدِ سررسیده‌ی همون کانال‌ها شلیک *نشد*
    "revoked_cycles_blocked": 0, # چند پستِ همون کانال‌ها دوره نگرفت
}


def _beat(name: str) -> None:
    """ضربانِ قلبِ یه حلقه - نگهبان از همین می‌فهمه حلقه زنده‌ست یا گیر کرده."""
    _runtime["heartbeat"][name] = time.monotonic()


# ---------------------------------------------------------------------------
# اتصال به سیستمِ یگانه‌ی منابع
# ---------------------------------------------------------------------------
# موتورِ بازدید دو سنجه‌ی حیاتی دارد که *هیچ‌کسِ دیگری* نمی‌تواند اندازه بگیرد:
# تاخیرِ تیکِ حلقه‌ی اجرا، و طولِ صفِ سررسیده. این‌ها را به resource_monitor
# می‌دهد و در عوض، فشارِ کلِ سرور (شاملِ رم، که خودش نمی‌بیند) را از آن می‌گیرد.
resource_monitor.register_provider("tick_lag", lambda: _runtime.get("tick_lag", 0.0))
resource_monitor.register_provider("backlog", lambda: _runtime.get("last_batch", 0))

_accounts_cache = {"at": 0.0, "rows": []}
_accounts_cache_lock = asyncio.Lock()
_membership_cache = {}          # username -> (monotonic, map)
_membership_cache_lock = asyncio.Lock()
# سقفِ تعدادِ کانالی که نقشه‌ی عضویتشون در حافظه می‌مونه. هر نقشه به‌اندازه‌ی تعدادِ
# اکانت‌هاست (با ۱۷۷۱ اکانت، هر نقشه چند صد کیلوبایت) - پس ۲۰۰ کانال یعنی ده‌ها
# مگابایت. با متغیرِ محیطی قابلِ کم‌کردنه اگه رم تنگ بود.
MEMBERSHIP_CACHE_MAX_CHANNELS = int(os.getenv("MEMBERSHIP_CACHE_MAX_CHANNELS", "60"))

# ---------------------------------------------------------------------------
# ثبتِ کش‌ها در دفترِ حافظه
# ---------------------------------------------------------------------------
# چرا: نقشه‌ی عضویت سنگین‌ترین ساختارِ درون‌حافظه‌ایِ این ماژول است (تعدادِ کانال ×
# تعدادِ اکانت). قبلاً هیچ‌کس نمی‌دانست چقدر جا گرفته و در مضیقه‌ی رم هم کسی
# نمی‌ریختش. حالا در پنلِ منابعِ سرور دیده می‌شود و در پاکسازیِ اضطراری اولین
# چیزی است که خالی می‌شود - چون خالی‌شدنش فقط یه بازخوانیِ ارزان هزینه دارد،
# هیچ حالتی از دست نمی‌رود.
memory_hygiene.register_cache(
    "بازدیدِ خودکار · نقشه‌ی عضویتِ کانال‌ها", _membership_cache,
    soft_max=MEMBERSHIP_CACHE_MAX_CHANNELS, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · خطاهایِ اسکن", _runtime["scan_errors"],
    soft_max=200, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · یادداشتِ ظرفیتِ پست‌ها", _runtime["capacity_notes"],
    soft_max=60, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · پست‌هایِ موکول‌شده", _runtime["deferred_posts"],
    soft_max=200, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · خطاهایِ چکِ پستِ پاک‌شده", _runtime["deleted_check_errors"],
    soft_max=200, clearable=True,
)

# ---------------------------------------------------------------------------
# حافظه‌ی پاسبانِ پستِ پاک‌شده
# ---------------------------------------------------------------------------
# هر سه‌تا کوچک و کاملاً قابلِ بازسازی‌ان: بدترین اثرِ خالی‌شدنشون یه دورِ چکِ اضافه‌ست،
# هیچ حالتی از دست نمی‌ره - پس هم سقفِ نرم دارن هم در پاکسازیِ اضطراری خالی می‌شن.
#   _deleted_posts   = پست‌هایِ تاییدشده‌ی پاک‌شده (تا TTL) - برایِ ردکردنِ ردیف‌هایی
#                      که همون لحظه از دیتابیس خونده شده بودن ولی پستشون پاک شده
#   _delete_strikes  = چند بار پشتِ‌سرِهم «نبود» دیده شده (حکم با ۲ مشاهده صادر می‌شه)
#   _deleted_next_check = این پست از این لحظه‌ی monotonic به بعد دوباره چک بشه
_deleted_posts = {}
_delete_strikes = {}
_deleted_next_check = {}

memory_hygiene.register_cache(
    "بازدیدِ خودکار · پست‌هایِ پاک‌شده (کوتاه‌مدت)", _deleted_posts,
    soft_max=DELETED_MEMORY_MAX_KEYS, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · سرنخِ پستِ پاک‌شده", _delete_strikes,
    soft_max=DELETED_MEMORY_MAX_KEYS, clearable=True,
)
memory_hygiene.register_cache(
    "بازدیدِ خودکار · موعدِ چکِ پستِ پاک‌شده", _deleted_next_check,
    soft_max=DELETED_MEMORY_MAX_KEYS, clearable=True,
)


async def _cached_accounts(force: bool = False) -> list:
    """
    لیستِ اکانت‌ها با کشِ کوتاه‌مدت. اکانت‌ها هر ۴ ثانیه عوض نمی‌شن، پس خوندنِ
    مکررشون فقط event loop و دیسک رو بی‌دلیل مشغول می‌کنه.
    """
    async with _accounts_cache_lock:
        now = time.monotonic()
        if force or (now - _accounts_cache["at"]) > ACCOUNTS_CACHE_TTL:
            _accounts_cache["rows"] = await get_all_accounts()
            _accounts_cache["at"] = now
        return _accounts_cache["rows"]


async def _cached_membership(username: str) -> dict:
    """نقشه‌یِ عضویتِ یه کانال با کشِ کوتاه‌مدت."""
    async with _membership_cache_lock:
        now = time.monotonic()
        hit = _membership_cache.get(username)
        if hit and (now - hit[0]) <= MEMBERSHIP_CACHE_TTL:
            return hit[1]
    # کوئری بیرونِ قفل زده می‌شه تا یه کانالِ کند بقیه رو بلاک نکنه
    data = await get_membership_map(username)
    async with _membership_cache_lock:
        # ⚠️ اول pop بعد ست: dictهایِ پایتون ترتیبِ *درج* رو نگه می‌دارن، پس بدونِ
        # این pop، کانالی که مرتب استفاده می‌شه همیشه «قدیمی‌ترین» می‌مونه و همون
        # قربانیِ تخلیه می‌شه - یعنی کشِ داغ‌ترین کانال مرتب دور ریخته می‌شد.
        _membership_cache.pop(username, None)
        _membership_cache[username] = (time.monotonic(), data)
        # جلوگیری از رشدِ بی‌نهایتِ کش وقتی کانال‌ها حذف/اضافه می‌شن.
        # قبلاً این‌جا هر بار کلِ کش sort می‌شد (O(n log n) زیرِ قفل، از یه مسیرِ
        # داغ). حالا چون ترتیبِ درج = ترتیبِ قدمت، فقط از ابتدا برداشته می‌شه.
        while len(_membership_cache) > MEMBERSHIP_CACHE_MAX_CHANNELS:
            _membership_cache.pop(next(iter(_membership_cache)), None)
    return data


def _invalidate_membership(username: str) -> None:
    _membership_cache.pop(username, None)


# ===========================================================================
#            کانالی که از لیستِ بازدیدِ خودکار حذف می‌شه: توقفِ *فوری*
# ---------------------------------------------------------------------------
# مسئله: حذفِ کانال از دیتابیس تنها نصفِ کار است. در همان لحظه ممکنه:
#   • حلقه‌ی اجرا یه دسته‌ی سررسیده رو چند میلی‌ثانیه قبل خونده باشه و در حالِ
#     اجراش باشه (بازدیدهایی که «همین حالا» زده می‌شن)
#   • یه پستِ همین کانال منتظرِ شروع باشه و تیکِ بعدی صدها ردیفِ تازه براش بسازه
#   • نتایجِ نیمه‌کاره‌ی همان دسته بعداً نوشته بشه و ردیفِ یتیم بسازه
#
# پس حذف دو لایه دارد: لایه‌ی *پایدار* (دیتابیس - database.delete_auto_view که
# همه‌ی ردیف‌هایِ کانال رو یک‌جا می‌بره) و لایه‌ی *لحظه‌ای* (همین‌جا). bot.py قبل
# از حذفِ دیتابیسی، forget_auto_view را صدا می‌زند تا از همان میلی‌ثانیه هیچ
# بازدیدِ جدیدی برایِ این کانال شلیک یا ثبت نشود.
#
# چرا یه ست با زمانِ انقضا (و نه ست بی‌نهایت): بعد از چند دقیقه دیگه هیچ دسته‌ی
# در-پروازی از اون کانال نمی‌مونه و خودِ دیتابیس هم پاک شده - نگه‌داشتنِ ابدیِ
# آیدی‌ها فقط حافظه‌ی الکیه (همون چیزی که در کلِ این ماژول از آن پرهیز شده).
# ===========================================================================
REVOKED_MEMORY_SECONDS = float(os.getenv("AUTO_VIEW_REVOKED_MEMORY_SECONDS", "900"))
_revoked_auto_views = {}     # auto_view_id -> time.monotonic()

memory_hygiene.register_cache(
    "بازدیدِ خودکار · کانال‌هایِ حذف‌شده (کوتاه‌مدت)", _revoked_auto_views,
    soft_max=200, clearable=True,
)


def _prune_revoked() -> None:
    now = time.monotonic()
    for key, at in list(_revoked_auto_views.items()):
        if (now - at) > REVOKED_MEMORY_SECONDS:
            _revoked_auto_views.pop(key, None)


def forget_auto_view(auto_view_id, channel_username: str = None) -> None:
    """
    «این کانال دیگه در لیست نیست» - از همین لحظه.

    bot.py این را *قبل* از حذفِ دیتابیسی صدا می‌زند. بعد از این فراخوان:
      • هیچ پستِ این کانال دوره نمی‌گیرد (نگاه کن به _activate_due_cycle_states)
      • هیچ ردیفِ سررسیده‌ی این کانال شلیک نمی‌شود (نگاه کن به _fire_due_views)
      • هیچ نتیجه/تاریخچه‌ای برایش نوشته نمی‌شود (فیلترِ پایانِ همان تابع)
    """
    try:
        _revoked_auto_views[int(auto_view_id)] = time.monotonic()
    except (TypeError, ValueError):
        return
    _channel_locks.pop(int(auto_view_id), None)
    _prune_revoked()
    if channel_username:
        name = str(channel_username).lstrip("@")
        _invalidate_membership(name)
        _runtime["scan_errors"].pop(name, None)
        _runtime["deferred_posts"].pop(name, None)
        _runtime["deleted_check_errors"].pop(name, None)
        prefix = f"@{name} #"
        for key in [k for k in _runtime["capacity_notes"] if k.startswith(prefix)]:
            _runtime["capacity_notes"].pop(key, None)
    _runtime["revoked_channels"] += 1


def is_auto_view_revoked(auto_view_id) -> bool:
    """آیا این ردیفِ بازدیدِ خودکار همین اواخر حذف شده؟"""
    if not _revoked_auto_views:
        return False
    try:
        return int(auto_view_id) in _revoked_auto_views
    except (TypeError, ValueError):
        return False


def _current_concurrency(backlog: int = 0) -> int:
    """
    هم‌زمانیِ امنِ همین لحظه - دو نیرویِ مخالف رو با هم می‌سنجه:

      • **بالا بردن** وقتی صف عقب افتاده: با ۵۰ کانال که هم‌زمان پست می‌ذارن، صدها
        بازدید در یک لحظه سررسید می‌شه. با هم‌زمانیِ ثابتِ پایین، صف هیچ‌وقت جبران
        نمی‌شه و بازدیدها ساعت‌ها دیر می‌افتن - همون «دقیق ثبت نمی‌شه».
      • **پایین آوردن** وقتی خودِ سرور تحتِ فشاره: اگه event loop عقب افتاده یا
        کانکشن‌هایِ تلگرام به سقف نزدیکن، فشارِ بیشتر فقط اوضاع رو بدتر می‌کنه و به
        بقیه‌ی بخش‌هایِ ربات هم اختلال می‌زنه.

    فشار همیشه بر عقب‌افتادگی اولویت داره - یعنی ربات هیچ‌وقت خودش رو خفه نمی‌کنه.
    """
    level = FIRE_CONCURRENCY_BASE
    if backlog > BACKLOG_SCALE_AT:
        span = max(BACKLOG_FULL_AT - BACKLOG_SCALE_AT, 1)
        ratio = min(1.0, (backlog - BACKLOG_SCALE_AT) / span)
        level = int(FIRE_CONCURRENCY_BASE +
                    ratio * (FIRE_CONCURRENCY_CEILING - FIRE_CONCURRENCY_BASE))

    # ---- عقب کشیدن بر اساسِ فشارِ *یگانه*ی سرور ----
    # ⚠️ قبلاً این‌جا فشار از نو و با معیارِ خودش حساب می‌شد (فقط tick_lag و تعدادِ
    # کانکشن) و **رم را اصلاً نمی‌دید**. یعنی پنل «۸۱۱ مگابایت» نشان می‌داد و در
    # همان لحظه موتورِ بازدید با ۴۸ بازدیدِ هم‌زمان به پیش می‌رفت تا پردازه OOM شود.
    # حالا همان عددی خوانده می‌شود که پنلِ منابعِ سرور نشان می‌دهد - یک منبعِ حقیقت
    # برایِ هر دو. نگاه کن به resource_monitor.py.
    snap = resource_monitor.snapshot()
    pressure = float(snap["pressure"])
    if pressure >= 0.85:
        level = FIRE_CONCURRENCY_MIN
    elif pressure >= 0.55:
        level = max(FIRE_CONCURRENCY_MIN, level // 3)
    elif pressure >= 0.30:
        level = max(FIRE_CONCURRENCY_MIN, level // 2)

    # حالتِ سختِ حافظه، حرفِ آخر رو می‌زنه: هیچ مقدار عقب‌افتادگیِ صف ارزشِ OOM شدن
    # رو نداره، چون OOM یعنی از دست رفتنِ *کلِ* کارِ در جریان.
    if snap["memory_state"] == "hard":
        level = FIRE_CONCURRENCY_MIN

    level = max(FIRE_CONCURRENCY_MIN, min(level, FIRE_CONCURRENCY_CEILING))
    _runtime["concurrency"] = level
    _runtime["pressure"] = pressure
    _runtime["pressure_worst"] = snap["worst"]
    return level


def _note_capacity(username: str, message_id: int, note: str, kind: str = None) -> None:
    """
    یادداشتِ وضعیتِ یه پست رو ثبت می‌کنه تا در پنلِ «منابعِ سرور» دیده بشه.

    ⚠️ kind چرا لازم شد: قبلاً کلیدِ یادداشت فقط «@کانال #پست» بود، پس یادداشتِ
    *پایانِ* بازدید، یادداشتِ «ظرفیتِ اکانت کم بود» رو پاک می‌کرد - یعنی مهم‌ترین
    توضیحی که ادمین لازم داشت گم می‌شد. حالا هر نوع یادداشت جایِ خودش رو داره.
    """
    notes = _runtime["capacity_notes"]
    key = f"@{username} #{message_id}"
    if kind:
        key = f"{key} · {kind}"
    notes[key] = note
    if len(notes) > 60:
        for k in list(notes)[:-60]:
            notes.pop(k, None)


def get_runtime_snapshot() -> dict:
    """کپیِ سطحیِ وضعیتِ زنده - برایِ خواندن از bot.py بدونِ ریسکِ تغییرِ ناخواسته."""
    snap = dict(_runtime)
    snap["scan_errors"] = dict(_runtime["scan_errors"])
    snap["capacity_notes"] = dict(_runtime["capacity_notes"])
    snap["deferred_posts"] = dict(_runtime["deferred_posts"])
    snap["deleted_check_errors"] = dict(_runtime["deleted_check_errors"])
    snap["restarts"] = dict(_runtime["restarts"])
    now = time.monotonic()
    snap["heartbeat_age"] = {
        n: round(now - t, 1) for n, t in _runtime["heartbeat"].items()
    }
    snap["loops_alive"] = dict(_runtime["loops_alive"])
    snap["running"] = _running
    # فشارِ کانکشن‌ها از منبعِ یگانه خونده می‌شه، نه با یه محاسبه‌ی موازیِ محلی
    snap["client_pressure"] = round(float(resource_monitor.snapshot()["client_ratio"]), 2)
    snap["cached_accounts"] = len(_accounts_cache["rows"])
    snap["cached_membership_channels"] = len(_membership_cache)
    # فشار از منبعِ یگانه خونده می‌شه، نه از نو حساب - تا پنل و موتور هیچ‌وقت دو
    # عددِ متفاوت نگن
    snap["resources"] = resource_monitor.snapshot()
    # مشخصاتِ سیاستِ بازدید - تا پنل و موتور هیچ‌وقت دو روایتِ متفاوت نگن
    snap["view_cycle"] = {
        "rounds_per_post": 1,
        "round_percent": 100,
        "defer_at_pressure": DEFER_AT_PRESSURE,
        "defer_max_streak": DEFER_MAX_STREAK,
        "starved_max_retries": VIEW_CYCLE_STARVED_MAX_RETRIES,
        "purge_grace_hours": VIEW_CYCLE_PURGE_GRACE_HOURS,
    }
    # وضعیتِ پاسبانِ پستِ پاک‌شده - همون اعدادی که پنلِ «📊 منابعِ سرور» نشون می‌ده
    snap["deleted_watch"] = {
        "interval_seconds": DELETED_CHECK_INTERVAL_SECONDS,
        "recheck_seconds": DELETED_RECHECK_SECONDS,
        "confirm_strikes": DELETED_CONFIRM_STRIKES,
        "skip_at_pressure": DELETED_SKIP_AT_PRESSURE,
        "max_posts_per_round": DELETED_CHECK_MAX_POSTS,
        "tracked": len(_deleted_posts),      # پست‌هایِ تاییدشده‌ی در حافظه (کوتاه‌مدت)
        "suspects": len(_delete_strikes),    # پست‌هایی که یه بار «نبود» دیده شدن
        "watched": len(_deleted_next_check), # پست‌هایی که موعدِ چکشون ثبت شده
    }
    return snap


def _build_view_schedule(view_count: int, base_time: datetime, start_delay_minutes: float,
                          round_interval_minutes: float, views_per_round: int = None) -> list:
    """
    view_count تا لحظه‌ی fire_at می‌سازه - از base_time (زمانِ خودِ پست، نه لحظه‌ای که ربات
    پست رو دید) به‌اضافه‌یِ تأخیرِ شروع.

    مدلِ *دوره‌ای*: دوره‌ی nاُم دقیقاً در start_delay + n×round_interval شروع می‌شه و
    بازدیدهایِ همون دوره **یکنواخت داخلِ خودِ همون دوره** پخش می‌شن (با کمی رندومی)،
    پس هیچ‌وقت از مرزِ دوره بیرون نمی‌زنن.

    ⚠️ باگی که اینجا رفع شد: قبلاً بازدیدهایِ داخلِ هر دوره با فاصله‌هایِ تجمعیِ
    ۲.۵ تا ۵ ثانیه‌ای چیده می‌شدن. برایِ ۲۰۰ بازدید این یعنی ~۱۲.۵ دقیقه، در حالی که
    خودِ دوره ۵ دقیقه بود - نتیجه‌اش این بود که دوره‌ها رویِ هم می‌افتادن، صف مرتب
    عقب می‌موند و «هر ۵ دقیقه ۲۰۰ بازدید» هیچ‌وقت درست درنمی‌اومد. حالا گامِ بینِ
    بازدیدها از خودِ طولِ دوره حساب می‌شه (interval×fill ÷ تعدادِ بازدیدِ دوره).
    """
    if view_count <= 0:
        return []
    start_seconds = max(start_delay_minutes or 0, 0) * 60

    if views_per_round and views_per_round > 0:
        interval_seconds = max(round_interval_minutes or 1, 1) * 60
        offsets = []
        remaining = view_count
        round_index = 0
        while remaining > 0:
            this_round = min(views_per_round, remaining)
            round_start = start_seconds + (round_index * interval_seconds)
            # گامِ یکنواخت داخلِ همین دوره - همه‌ی بازدیدها قبلِ شروعِ دوره‌ی بعد تمام می‌شن
            usable = interval_seconds * ROUND_FILL_RATIO
            step = usable / this_round
            jitter = step * ROUND_JITTER_RATIO
            round_deadline = round_start + interval_seconds - 1
            for i in range(this_round):
                t = round_start + (i * step) + random.uniform(0, jitter)
                offsets.append(min(t, round_deadline))
            remaining -= this_round
            round_index += 1
        offsets.sort()
        return [base_time + timedelta(seconds=o) for o in offsets]

    # رفتارِ ساده: بدونِ دوره‌بندی - همه پشتِ‌سرِهم، فقط بعدِ تأخیرِ شروع
    offsets = []
    acc = start_seconds
    for i in range(view_count):
        if i > 0:
            acc += random.uniform(IMMEDIATE_GAP_MIN, IMMEDIATE_GAP_MAX)
        offsets.append(acc)
    return [base_time + timedelta(seconds=o) for o in offsets]


def _eligible_from(accounts: list, country_code: str = None) -> list:
    """اکانت‌هایی که می‌تونن بازدید بزنن (لاگین، غیرِ فریز، هم‌کشور)."""
    return [
        a for a in accounts
        if a.get("is_logged_in") and not a.get("is_frozen")
        and (not country_code or a.get("country_code") == country_code)
    ]


# ===========================================================================
#                    بازدیدِ هر پست: یک دوره‌ی ۱۰۰٪، و تمام
# ---------------------------------------------------------------------------
# قانون: هر پستِ جدید *یک* دوره بازدید می‌گیرد (۱۰۰٪ سفارش). به‌محضِ تمام‌شدنِ
# ردیف‌هایِ همان یک دوره، پست کنار گذاشته می‌شود و هیچ بازدیدِ دیگری برایش ثبت
# نمی‌شود. نردبانِ کاهشی (۱۰۰→۸۰→۶۰→۴۰→۲۰٪) و برنامه‌ریزِ فاصله‌ی دوره‌ها حذف شده‌اند.
#
# دو حقیقتی که کلِ طراحی رویشان بنا شده:
#   ۱) تلگرام برایِ هر پست از هر اکانت فقط **یک** بازدید می‌شمارد. پس سقفِ واقعیِ
#      هر پست = تعدادِ اکانتِ سالمِ همان کشور، و همین یک دوره حداکثر همان تعداد را
#      تحویل می‌دهد. (قبلاً نردبانِ کامل ۳ برابرِ سفارش اکانتِ یکتا می‌خواست؛ حالا
#      دقیقاً به‌اندازه‌ی خودِ سفارش.)
#   ۲) تاریخچه‌یِ «کدام اکانت این پست را دیده» فقط تا پایانِ همین یک دوره لازم است -
#      تا داخلِ خودِ دوره اکانتِ تکراری انتخاب نشود - و بلافاصله بعدش پاک می‌شود.
# ===========================================================================

def _round_view_count(base_view_count: int) -> int:
    """تعدادِ بازدیدِ تنها دوره‌ی این پست = خودِ سفارش (۱۰۰٪)، بدونِ هیچ کاهشی."""
    return max(0, int(base_view_count or 0))


def _pressure_defer_needed(pressure: float, defer_streak: int = 0) -> bool:
    """
    آیا شروعِ دوره‌ی این پست باید عقب بیفتد چون سرور همین الان اشباع است؟

    defer_streak جلویِ عقب‌اندازیِ ابدی را می‌گیرد: بعد از DEFER_MAX_STREAK بار،
    دوره هرچه باشد اجرا می‌شود.
    """
    if int(defer_streak or 0) >= DEFER_MAX_STREAK:
        return False
    return float(pressure or 0.0) >= DEFER_AT_PRESSURE


def _pressure_defer_until(now: datetime) -> datetime:
    """چند دقیقه‌ی *رندومِ* بعد - تا چند پستِ عقب‌مانده هم‌زمان شلیک نشوند."""
    return now + timedelta(minutes=random.uniform(DEFER_MIN_MINUTES, DEFER_MAX_MINUTES))


def _as_naive_utc(value) -> datetime:
    """رشته/تاریخِ ورودی را به datetimeِ UTC-naive تبدیل می‌کند (تنها فرمتِ داخلیِ ربات)."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# فعال‌سازیِ دوره‌ی پست‌ها فقط و فقط یکی‌یکی انجام می‌شود.
#
# ⚠️ این قفل باگِ «هر پست دو برابرِ سفارش بازدید می‌گرفت» را می‌بندد:
# _activate_due_cycle_states هم از حلقه‌ی اسکنِ پست صدا زده می‌شود هم از حلقه‌ی اجرا.
# چون بینِ «خواندنِ پست‌هایِ سررسیده» و «علامت‌زدنشان» چند await وجود دارد، هر دو
# حلقه یک پست را می‌دیدند و *هر دو* برایش ردیفِ بازدید می‌ساختند - یعنی دقیقاً دو
# برابرِ تعدادِ سفارش، و استخرِ اکانت‌ها دو برابر سریع‌تر ته می‌کشید.
# (در تستِ واقعی: ۲۰ ردیف برایِ یک سفارشِ ۱۰ بازدیدی.)
# قفلِ درون-پردازه‌ای + قاپِ اتمیِ دیتابیس، هر دو لایه را می‌بندد.
_cycle_activation_lock = asyncio.Lock()


async def _finish_post_views(state: dict, now_iso: str, reason: str = None) -> None:
    """
    بازدیدِ یه پست را برایِ همیشه می‌بندد و تاریخچه‌یِ اکانت‌هایش را *همان لحظه* پاک
    می‌کند - یعنی همون «این پست کنار گذاشته شد».

    از این لحظه به بعد هیچ دوره‌ای برایِ این پست ساخته نمی‌شود: ردیفِ وضعیتش
    finished است، پس نه در سررسیده‌ها می‌آید و نه در در-حالِ-اجراها.

    تاریخچه‌یِ اکانت‌ها فقط تا وقتی ارزش داشت که همین یک دوره در جریان بود؛ بعدش فقط
    حافظه اشغال می‌کند. خودِ ردیفِ وضعیت و ردیف‌هایِ زمان‌بندی‌شده تا
    VIEW_CYCLE_PURGE_GRACE_HOURS نگه داشته می‌شوند تا گزارشِ پایانی در پنل دیده شود،
    و بعد در پاک‌سازیِ ساعتی کاملاً حذف می‌شوند.
    """
    await finish_auto_view_cycle(state["id"], now_iso)
    try:
        removed = await purge_auto_view_message_viewers_for_post(
            state["auto_view_id"], state["message_id"]
        )
    except Exception as e:
        removed = 0
        _runtime["cycle_last_error"] = f"purge viewers: {type(e).__name__}: {e}"
    _runtime["posts_finished"] += 1
    delivered = state.get("views_delivered")
    note = reason or "دوره‌ی ۱۰۰٪ کامل شد"
    _note_capacity(
        state["channel_username"], state["message_id"],
        f"✅ بازدیدِ این پست بسته شد ({note}) - بازدیدِ تحویل‌شده: "
        f"{delivered if delivered is not None else '?'}"
        f"، تاریخچه‌یِ {removed} اکانت پاک شد. این پست کنار گذاشته شد؛ "
        f"دوره‌ی دیگری برایش اجرا نمی‌شود.",
        kind="پایان"
    )


async def _activate_due_cycle_states(now_iso: str):
    """
    پست‌هایی که زمانِ شروعِ بازدیدشان رسیده را به ردیف‌هایِ واقعیِ بازدید تبدیل می‌کند.

    ترتیبِ کارها عمدی و مهم است:
      ۱) ظرفیتِ *واقعیِ* اکانت‌هایِ تازه برایِ این پست حساب می‌شود
      ۲) اگه سرور همین الان اشباع باشد، شروع کمی عقب می‌افتد (بسته نمی‌شود)
      ۳) دوره **اتمی قاپ زده می‌شود** - و تنها بعدِ آن ردیف‌ها ساخته می‌شوند
      ۴) اگر ساختنِ ردیف‌ها شکست خورد، قاپ برگردانده می‌شود تا پست گم نشود
    """
    async with _cycle_activation_lock:
        await _activate_due_cycle_states_locked(now_iso)


async def _activate_due_cycle_states_locked(now_iso: str):
    now_dt = _as_naive_utc(now_iso)

    # فقط و فقط سررسیده‌ها.
    # ⚠️ قبلاً تا چند ساعتِ آینده هم خوانده می‌شد تا برنامه‌ریز بتواند دوره را
    # «زودتر» اجرا کند (وقتی سرور خلوت بود). آن منطق همراهِ نردبان حذف شد: تنها
    # زمانِ شروعِ معتبر، همون «تأخیرِ شروعِ» ثبت‌شده‌ی خودِ ادمین است.
    due_states = await get_due_auto_view_cycle_states(now_iso)
    if not due_states:
        return

    accounts = await _cached_accounts()
    # «چه اکانت‌هایی این پست را دیده‌اند» برایِ همه‌ی پست‌ها با یک کوئری
    used_map = await get_done_account_ids_for_view_messages(
        [(st["auto_view_id"], st["message_id"]) for st in due_states]
    )
    snap = resource_monitor.snapshot()
    pressure = float(snap["pressure"])

    for state in due_states:
        key = (state["auto_view_id"], state["message_id"])
        # ⚠️ پستی که همین چند لحظه پیش «پاک‌شده» تایید شده، حتی اگه ردیفِ وضعیتش
        # هنوز در این لیستِ خونده‌شده باشه، *نباید* دوره بگیره - وگرنه دقیقاً بعدِ
        # پاک‌سازی، صدها ردیفِ تازه برایِ یه پستِ ناموجود ساخته می‌شد.
        if _is_known_deleted(state["auto_view_id"], state["message_id"]):
            continue
        # ⚠️ کانالی که همین چند لحظه پیش از لیستِ بازدیدِ خودکار حذف شده *نباید*
        # دوره بگیره. ردیف‌هاش از دیتابیس رفته‌ن، ولی این لیستِ سررسیده‌ها ممکنه
        # میلی‌ثانیه‌ای قبلِ حذف خونده شده باشه - و ساختنِ صدها ردیفِ تازه برایِ
        # کانالی که ادمین همین الان حذفش کرد، بدترین شکلِ «ادامه دادنِ کارِ لغوشده»ست.
        if is_auto_view_revoked(state["auto_view_id"]):
            _runtime["revoked_cycles_blocked"] += 1
            continue
        planned_at = _as_naive_utc(state["next_cycle_at"])
        defer_streak = int(state.get("defer_streak") or 0)

        pool = _eligible_from(accounts, state.get("country_code"))
        already_viewed = used_map.get(key, set())
        fresh = [a for a in pool if a["id"] not in already_viewed]
        capacity = len(fresh)

        # ---- حالتِ «الان هیچ اکانتِ سالمی نیست» با «استخر تمام شد» فرق دارد ----
        if not pool:
            retry_at = now_dt + timedelta(minutes=VIEW_CYCLE_NO_ACCOUNT_RETRY_MINUTES)
            await postpone_auto_view_cycle(
                state["id"], retry_at.isoformat(timespec="seconds"),
                reason="هیچ اکانتِ سالمی موجود نیست",
            )
            _note_capacity(
                state["channel_username"], state["message_id"],
                f"هیچ اکانتِ سالمی موجود نیست (همه لاگ‌اوت/فریز) - بازدیدِ این پست "
                f"بسته نشد، {VIEW_CYCLE_NO_ACCOUNT_RETRY_MINUTES} دقیقه بعد دوباره "
                f"تلاش می‌شود"
            )
            continue

        ordered = int(state["base_view_count"] or 0)
        round_views = _round_view_count(ordered)
        if round_views <= 0:
            await _finish_post_views(
                state, now_iso,
                reason=f"سفارشِ این پست صفر است ({ordered} بازدید)"
            )
            continue

        # ---- سرور اشباع است؟ شروع عقب می‌افتد، پست بسته نمی‌شود ----
        if _pressure_defer_needed(pressure, defer_streak):
            retry_at = _pressure_defer_until(now_dt)
            await postpone_auto_view_cycle(
                state["id"], retry_at.isoformat(timespec="seconds"),
                reason=f"فشارِ سرور {int(pressure * 100)}٪ ({snap['worst']})",
            )
            _runtime["cycles_deferred"] += 1
            _note_capacity(
                state["channel_username"], state["message_id"],
                f"شروعِ بازدیدِ این پست عقب افتاد چون سرور اشباعه "
                f"(فشار {int(pressure * 100)}٪ - بیشترین سهم: {snap['worst']}) - "
                f"تلاشِ بعدی {retry_at.strftime('%H:%M')} UTC "
                f"[{defer_streak + 1} از {DEFER_MAX_STREAK}]",
                kind="فشار"
            )
            continue

        # ---- اکانتِ تازه تمام شده؟ اول چند بار صبر کن، بعد ببند ----
        # ⚠️ یه صفرِ گذرا (موجِ FloodWait، ری‌استارت) نباید بازدیدِ پست را بکشد.
        if capacity <= 0:
            if defer_streak < VIEW_CYCLE_STARVED_MAX_RETRIES:
                retry_at = now_dt + timedelta(minutes=VIEW_CYCLE_STARVED_RETRY_MINUTES)
                await postpone_auto_view_cycle(
                    state["id"], retry_at.isoformat(timespec="seconds"),
                    reason="هیچ اکانتِ تازه‌ای نمانده",
                )
                _runtime["cycles_starved"] += 1
                _note_capacity(
                    state["channel_username"], state["message_id"],
                    f"همه‌ی {len(pool)} اکانتِ واجدِ شرایط این پست را دیده‌اند - "
                    f"منتظرِ اکانتِ جدید (تلاشِ {defer_streak + 1} از "
                    f"{VIEW_CYCLE_STARVED_MAX_RETRIES}، بعدی "
                    f"{retry_at.strftime('%m-%d %H:%M')} UTC)",
                    kind="ظرفیت"
                )
                continue
            await _finish_post_views(
                state, now_iso,
                reason=f"بعد از {VIEW_CYCLE_STARVED_MAX_RETRIES} تلاش هیچ اکانتِ تازه‌ای "
                       f"پیدا نشد (همه‌ی {len(pool)} اکانت این پست را دیده‌اند)"
            )
            continue

        if round_views > capacity:
            _note_capacity(
                state["channel_username"], state["message_id"],
                f"{round_views} بازدید سفارش داده شده ولی فقط {capacity} اکانتِ تازه "
                f"موجوده - رویِ {capacity} تنظیم شد. تلگرام از هر اکانت فقط ۱ بازدید "
                f"برایِ هر پست می‌شماره، پس سقفِ این پست همون تعدادِ اکانته."
            )
            round_views = capacity

        views_per_round = state.get("views_per_round") or 0
        if views_per_round and views_per_round > round_views:
            views_per_round = round_views

        # ---- قاپِ اتمی: از این خط به بعد، این دوره فقط مالِ همین فراخوان است ----
        if not await claim_auto_view_cycle(state["id"], now_iso):
            continue  # یکی دیگه زودتر قاپیده - هیچ ردیفی نساز

        # ⚠️ اگه ربات خواب/خاموش بوده و موعدِ شروع گذشته، پخش از *همین الان* شروع
        # می‌شه نه از یه زمانِ گذشته - وگرنه کلِ دوره یک‌جا و در یک لحظه شلیک می‌شد
        # که هم غیرطبیعیه هم صف رو منفجر می‌کنه.
        base_time = max(planned_at, now_dt)

        schedule = _build_view_schedule(
            round_views, base_time, 0,
            state.get("spread_minutes") or 0, views_per_round,
        )
        follower_percent = _follower_share(state)
        follower_count = round(round_views * follower_percent / 100)
        source_types = (["follower"] * follower_count
                        + ["other"] * (round_views - follower_count))
        random.shuffle(source_types)
        rows = [
            {
                "auto_view_id": state["auto_view_id"],
                "message_id": state["message_id"],
                "channel_username": state["channel_username"],
                "country_code": state.get("country_code"),
                "fire_at": fire_at.isoformat(timespec="seconds"),
                "source_type": source_type,
            }
            for fire_at, source_type in zip(schedule, source_types)
        ]
        try:
            inserted = await add_scheduled_views(rows)
        except Exception as e:
            inserted = 0
            _runtime["cycle_last_error"] = f"add_scheduled_views: {type(e).__name__}: {e}"

        if not inserted:
            # ردیفی ساخته نشد - قاپ رو برگردون تا این پست گم نشه
            retry_at = now_dt + timedelta(seconds=VIEW_CYCLE_CLAIM_RETRY_SECONDS)
            await release_auto_view_cycle_claim(
                state["id"], retry_at.isoformat(timespec="seconds")
            )
            continue

        _runtime["posts_started"] += 1
        print(f"🔁 @{state['channel_username']} #{state['message_id']}: بازدید شروع شد "
              f"({round_views} بازدید - تنها دوره‌ی این پست، ظرفیتِ تازه: {capacity})")



async def _close_completed_post_rounds(now_iso: str):
    """
    هر پستی که *همه‌ی ردیف‌هایِ بازدیدش* تمام شده را می‌بندد و کنار می‌گذارد.

    اینجا دیگر «دوره‌ی بعدی» وجود ندارد: نه فاصله‌ای حساب می‌شود، نه دوره‌ی ۸۰٪ای
    ساخته می‌شود. پست تمام است، تاریخچه‌اش همان لحظه پاک می‌شود.

    ⚠️ باگِ رفع‌شده (اعلامِ زودهنگامِ پایان): یه دوره‌ی تازه‌ساخته‌شده می‌تواند برایِ
    چند لحظه صفر ردیفِ pending داشته باشد. پس دو محافظ داریم: حداقل زمانِ اجرا
    (MIN_CYCLE_RUN_SECONDS) و شرطِ «کلاً ردیفی ساخته شده باشد».
    """
    now_dt = _as_naive_utc(now_iso)
    running_states = await get_running_auto_view_cycle_states()
    if not running_states:
        return

    pairs = [(st["auto_view_id"], st["message_id"]) for st in running_states]
    stats = await get_scheduled_view_counts_for_posts(pairs)

    for state in running_states:
        info = stats.get(
            (state["auto_view_id"], state["message_id"]),
            {"pending": 0, "total": 0, "last_fire_at": None},
        )

        # محافظِ «تازه ساخته شده» - جلویِ اعلامِ زودهنگامِ پایان
        started_at = state.get("cycle_started_at")
        run_seconds = None
        if started_at:
            try:
                run_seconds = (now_dt - _as_naive_utc(started_at)).total_seconds()
            except Exception:
                run_seconds = None
        if run_seconds is not None and run_seconds < MIN_CYCLE_RUN_SECONDS:
            continue

        # ---- دوره‌ای که هیچ ردیفی ندارد: کرش/خطا بینِ قاپ و ساختِ ردیف‌ها ----
        # این پست را «تمام‌شده» حساب نمی‌کنیم (وگرنه یه پست کامل بی‌بازدید می‌موند)؛
        # قاپ را برمی‌گردانیم تا ردیف‌هایش از نو ساخته شوند.
        if info["total"] == 0:
            await release_auto_view_cycle_claim(
                state["id"], now_dt.isoformat(timespec="seconds")
            )
            _note_capacity(
                state["channel_username"], state["message_id"],
                "دوره‌ی بازدیدِ این پست بدونِ ردیف مانده بود - از نو ساخته می‌شود"
            )
            continue

        if info["pending"] > 0:
            continue  # این پست هنوز بازدیدِ نکرده دارد

        # ---- تمام شد: پست کنار گذاشته می‌شود ----
        await _finish_post_views(state, now_iso)
        delivered = state.get("views_delivered")
        print(f"✅ @{state['channel_username']} #{state['message_id']}: بازدیدِ این پست "
              f"کامل شد ({delivered if delivered is not None else '?'} بازدید) - "
              f"کنار گذاشته شد، دوره‌ی بعدی‌ای وجود ندارد")


async def _purge_finished_posts(now_dt: datetime) -> int:
    """
    ردِ پایِ پست‌هایی که بازدیدشان تمام شده و مهلتِ گزارششان هم گذشته را کاملاً
    پاک می‌کند - تا دیتابیس با پست‌هایِ قدیمی ورم نکند.
    """
    cutoff = (now_dt - timedelta(hours=VIEW_CYCLE_PURGE_GRACE_HOURS)).isoformat(timespec="seconds")
    try:
        finished = await get_finished_auto_view_cycle_states(cutoff)
    except Exception as e:
        _runtime["cycle_last_error"] = f"finished list: {type(e).__name__}: {e}"
        return 0
    purged = 0
    for st in finished:
        try:
            await purge_auto_view_post_history(
                st["auto_view_id"], st["message_id"], drop_state=True
            )
            purged += 1
            # ردِ پایِ این پست در حافظه هم پاک بشه، نه فقط در دیتابیس - وگرنه
            # یادداشت‌هایِ پنل با هر پستِ تمام‌شده یه کلیدِ مرده بیشتر می‌گیرن.
            prefix = f"@{st.get('channel_username')} #{st['message_id']}"
            for k in [k for k in _runtime["capacity_notes"] if k.startswith(prefix)]:
                _runtime["capacity_notes"].pop(k, None)
        except Exception as e:
            _runtime["cycle_last_error"] = f"purge: {type(e).__name__}: {e}"
    if purged:
        _runtime["posts_purged"] += purged
    return purged


# ===========================================================================
#            پاسبانِ پستِ پاک‌شده - تشخیص، توقفِ بازدید، و آزادسازیِ حافظه
# ===========================================================================


def _follower_share(row) -> int:
    """
    سهمِ Followers (۰..۱۰۰) از یه ردیفِ کانال یا وضعیتِ پست.

    ⚠️ پیش‌فرض ۱۰۰ ـه، نه ۰: ردیف‌هایی که *قبلِ* اضافه‌شدنِ این قابلیت ثبت شدن
    مقدارِ NULL دارن و رفتارِ تاریخی‌شون «همه بازدیدها از فالوور» بوده. قبلاً یه
    جا ۰ و یه جا ۱۰۰ پیش‌فرض گرفته می‌شد - یعنی پنل یه چیز نشون می‌داد و موتور
    یه چیزِ دیگه اجرا می‌کرد.
    """
    try:
        value = row.get("follower_percent")
    except AttributeError:
        value = None
    if value is None:
        return 100
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 100


def _post_key(auto_view_id, message_id) -> tuple:
    return (int(auto_view_id), int(message_id))


def _is_known_deleted(auto_view_id, message_id) -> bool:
    """آیا همین حالا مطمئنیم این پست پاک شده؟ (برایِ ردکردنِ ردیف‌هایِ در پرواز)"""
    try:
        return _post_key(auto_view_id, message_id) in _deleted_posts
    except Exception:
        return False


def _forget_post_watch(key: tuple) -> None:
    """کلیدهایِ کمکیِ یه پست رو از حافظه پاک می‌کنه (بعدِ حکمِ نهایی لازمشون نداریم)."""
    _delete_strikes.pop(key, None)
    _deleted_next_check.pop(key, None)


def _prune_deleted_memory() -> None:
    """
    حافظه‌ی این بخش رو کوچک نگه می‌داره: کلیدهایِ منقضی‌شده می‌رن و سقفِ نرم اعمال می‌شه.

    چرا خودش این کار رو می‌کنه و منتظرِ پاکسازیِ عمومی نمی‌مونه: پاکسازیِ عمومی هر
    ۵ دقیقه اجرا می‌شه و فقط سقف رو اعمال می‌کنه؛ اینجا TTL هم داریم (کلیدِ پستِ
    پاک‌شده بعد از چند دقیقه دیگه هیچ ارزشی نداره چون نه ردیفی مونده نه وضعیتی).
    """
    now = time.monotonic()
    for key in [k for k, at in list(_deleted_posts.items())
                if (now - at) > DELETED_MEMORY_TTL_SECONDS]:
        _deleted_posts.pop(key, None)
    for store in (_deleted_posts, _delete_strikes, _deleted_next_check):
        while len(store) > DELETED_MEMORY_MAX_KEYS:
            store.pop(next(iter(store)), None)


def _flag_delete_suspect(auto_view_id, message_id) -> int:
    """
    یه سرنخِ «شاید این پست پاک شده» ثبت می‌کنه (از مسیرِ واکنشی: خودِ بازدید با خطایِ
    «آی‌دیِ پیام نامعتبر» برگشت) و پست رو برایِ چکِ فوریِ دورِ بعد نشان می‌کنه.

    ⚠️ این تابع هیچ‌وقت خودش حکمِ پاک‌شدن صادر نمی‌کنه - حتی اگه سرنخ‌ها زیاد بشن.
    حکم فقط وقتی صادر می‌شه که خودِ *پاسبان* هم با پرسیدن از تلگرام «نبود» ببینه.
    """
    key = _post_key(auto_view_id, message_id)
    if key in _deleted_posts:
        return DELETED_CONFIRM_STRIKES
    _delete_strikes[key] = _delete_strikes.get(key, 0) + 1
    _deleted_next_check[key] = 0.0   # یعنی «همین دورِ بعد چکش کن»
    return _delete_strikes[key]


async def _purge_deleted_posts(states: list, reason: str) -> int:
    """
    بازدیدِ این پست‌ها رو برایِ همیشه متوقف و کلِ ردِ پایشون رو از دیتابیس حذف می‌کنه.

    سه چیز هم‌زمان اتفاق می‌افته:
      ۱) ردیف‌هایِ بازدیدِ باقی‌مونده حذف می‌شن → نه بازدیدِ دیگه‌ای زده می‌شه، نه
         سهمیه‌ی روزانه‌ی اکانتی خرج می‌شه، نه صف الکی شلوغ می‌مونه
      ۲) تاریخچه‌ی بازدیدکننده‌ها و خودِ ردیفِ وضعیت حذف می‌شن → حجمِ دیتابیس آزاد
      ۳) کلیدِ پست تا چند دقیقه در حافظه می‌مونه تا ردیف‌هایِ «در پرواز» (اونایی که
         همون لحظه از دیتابیس خونده شده بودن) هم اجرا نشن

    ⚠️ پستِ پاک‌شده عمداً به‌جایِ «finished» علامت‌خوردن، *حذف* می‌شه: ردیفِ finished
    باید تا VIEW_CYCLE_PURGE_GRACE_HOURS برایِ گزارش نگه داشته بشه، ولی گزارشِ یه
    پستِ ناموجود به هیچ‌کس کمکی نمی‌کنه - فقط جا می‌گیره. توضیحش در یادداشت‌هایِ
    پنل (که سقف‌دار و در-حافظه‌ست) می‌مونه.
    """
    if not states:
        return 0
    pairs = [(st["auto_view_id"], st["message_id"]) for st in states]
    try:
        # ⚠️ قبلاً این‌جا purge_auto_view_posts بود، یعنی ردیف‌ها *حذف* می‌شدن و
        # سفارشِ بازدیدی که وسطِ کار بود بی‌هیچ توضیحی از «🗂 سفارش‌ها» ناپدید
        # می‌شد. ری‌اکشن و شیر هر دو ردیف رو با status='deleted' نگه می‌داشتن تا
        # معلوم باشه «نسوخت - پست پاک شد». حالا هر سه یک رفتار دارن و ردیف بعدِ
        # ۲۴ ساعت با پاکسازیِ مشترک می‌ره.
        report = await stop_views_for_deleted_posts(pairs)
    except Exception as e:
        _runtime["cycle_last_error"] = f"purge deleted: {type(e).__name__}: {e}"
        return 0

    cancelled = int(report.get("scheduled", 0) or 0)
    freed = cancelled + int(report.get("viewers", 0) or 0) + int(report.get("states", 0) or 0)
    _runtime["posts_deleted"] += len(pairs)
    _runtime["deleted_views_cancelled"] += cancelled
    _runtime["deleted_rows_freed"] += freed

    now = time.monotonic()
    for st in states:
        key = _post_key(st["auto_view_id"], st["message_id"])
        _deleted_posts[key] = now
        _forget_post_watch(key)
        username = st.get("channel_username") or "?"
        # یادداشت‌هایِ قبلیِ همین پست (ظرفیت، فشار، …) دیگه بی‌معنی‌ن
        prefix = f"@{username} #{st['message_id']}"
        for k in [k for k in _runtime["capacity_notes"] if k.startswith(prefix)]:
            _runtime["capacity_notes"].pop(k, None)
        _note_capacity(
            username, st["message_id"],
            f"🗑 این پست از کانال پاک شده ({reason}) - بازدیدش متوقف شد و کلِ "
            f"اطلاعاتش از دیتابیس حذف شد (هیچ حافظه‌ای الکی نگه داشته نمی‌شه)",
            kind="پاک‌شده",
        )
    print(f"🗑 {len(pairs)} پستِ پاک‌شده تشخیص داده شد ({reason}) - بازدیدشون متوقف شد، "
          f"{cancelled} بازدیدِ باقی‌مونده لغو و {freed} ردیفِ دیتابیس آزاد شد")
    _prune_deleted_memory()
    return len(pairs)


async def _check_channel_posts_deleted(row: dict, items: list, lock: asyncio.Lock,
                                        confirmed: list) -> None:
    """
    پست‌هایِ فعالِ یک کانال رو از خودِ تلگرام می‌پرسه و نتیجه رو در حافظه ثبت می‌کنه.

    قواعدِ سخت‌گیرانه‌ی این تابع (هر سه برایِ جلوگیری از حذفِ اشتباهی‌ان):
      • خطا/FloodWait/تایم‌اوت → *هیچ* حکمی صادر نمی‌شه (فقط در پنل ثبت می‌شه)
      • پستی که «هست» دیده بشه، سرنخ‌هایِ قبلی‌اش پاک می‌شن (شروعِ دوباره از صفر)
      • حکم فقط با DELETED_CONFIRM_STRIKES مشاهده‌ی پشتِ‌سرِهمِ «نبود»
    """
    username = row["channel_username"]
    watcher = await _pick_watcher(row)
    if not watcher:
        _runtime["deleted_check_errors"][username] = (
            "هیچ اکانتِ سالمی برایِ چکِ پاک‌شدنِ پست‌ها نیست (همه لاگ‌اوت/فریز)"
        )
        return

    ids = [st["message_id"] for st in items]
    try:
        res = await asyncio.wait_for(
            check_posts_exist(watcher["session_file"], ids, username=username),
            timeout=DELETED_CHECK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        _runtime["deleted_check_errors"][username] = (
            f"چکِ پاک‌شدنِ پست‌ها بیش از {DELETED_CHECK_TIMEOUT} ثانیه طول کشید"
        )
        return
    except Exception as e:
        # خطا باید به کولدانِ واقعیِ اکانت تبدیل بشه، نه فقط یه رشته‌ی نمایشی -
        # وگرنه همین اکانت دورِ بعد دوباره همون درخواست رو می‌زنه
        _note_watcher_failure(watcher, e)
        _runtime["deleted_check_errors"][username] = f"{type(e).__name__}: {e}"
        return

    # چکِ پاک‌شدن هم یه فعالیتِ واقعیه و باید شمرده بشه
    activity_guard.mark_used(watcher["id"], weight=1, acc=watcher)

    status = res.get("status")
    if status == "ok":
        _runtime["deleted_check_errors"].pop(username, None)
    elif status == "flood":
        activity_guard.report_flood(watcher["id"], res.get("seconds", 0) or 0)
        _runtime["deleted_check_errors"][username] = (
            f"FloodWait {res.get('seconds', '?')} ثانیه - چکِ این کانال به دورِ بعد افتاد"
        )
    else:
        _runtime["deleted_check_errors"][username] = str(res.get("message", "خطا"))[:120]

    existing = set(res.get("existing") or ())
    # ⚠️ «نبود»ها فقط از یه چکِ *سالم* پذیرفته می‌شن. در حالتِ خطا/FloodWait حتی اگه
    # نتیجه‌ی جزئی داشته باشیم، فقط «هست»ها رو باور می‌کنیم.
    missing = set(res.get("missing") or ()) if status == "ok" else set()

    stamp = time.monotonic()
    async with lock:
        for st in items:
            mid = int(st["message_id"])
            key = _post_key(st["auto_view_id"], mid)
            if mid in existing:
                _runtime["deleted_checks"] += 1
                _deleted_next_check[key] = stamp + DELETED_RECHECK_SECONDS
                _delete_strikes.pop(key, None)
            elif mid in missing:
                _runtime["deleted_checks"] += 1
                strikes = _delete_strikes.get(key, 0) + 1
                _delete_strikes[key] = strikes
                if strikes >= DELETED_CONFIRM_STRIKES:
                    confirmed.append(st)
                else:
                    # مشاهده‌ی اول - دورِ بعد دوباره و سریع چک می‌شه
                    _deleted_next_check[key] = stamp + DELETED_SUSPECT_RECHECK_SECONDS
            # نه در «هست» نه در «نبود» (چکِ نیمه‌کاره/خطا) → دست‌نخورده می‌مونه


async def _scan_deleted_posts() -> int:
    """
    یه دورِ کاملِ «کدوم پستِ فعال از کانال پاک شده؟». خروجی: چند پست پاک‌شده تایید شد.

    ترتیبِ کارها:
      ۱) اگه سرور اشباعه، این دور کامل رد می‌شه (هماهنگی با «📊 منابعِ سرور»)
      ۲) پست‌هایی که کانالشون دیگه ثبت‌شده نیست، بی‌چون‌وچرا پاک می‌شن (یتیمِ قطعی)
      ۳) بقیه بر اساسِ موعدِ چک و سرنخ‌ها اولویت‌بندی و گروه‌بندیِ کانالی می‌شن
    """
    snap = resource_monitor.snapshot()
    pressure = float(snap["pressure"])
    if pressure >= DELETED_SKIP_AT_PRESSURE or snap["memory_state"] == "hard":
        _runtime["deleted_skipped_pressure"] += 1
        _runtime["deleted_last_batch"] = 0
        return 0

    _prune_deleted_memory()

    try:
        states = await get_active_auto_view_posts()
    except Exception as e:
        _runtime["cycle_last_error"] = f"deleted list: {type(e).__name__}: {e}"
        return 0
    if not states:
        _runtime["deleted_check_errors"].clear()
        _runtime["deleted_last_batch"] = 0
        return 0

    try:
        channel_rows = await get_auto_view_channels()
    except Exception as e:
        _runtime["cycle_last_error"] = f"deleted channels: {type(e).__name__}: {e}"
        return 0
    channels = {c["id"]: c for c in channel_rows}
    live_channels = set(channels)

    # خطاهایِ کانال‌هایی که دیگه ثبت نیستن، در پنل هم نمونن
    for stale in [u for u in _runtime["deleted_check_errors"]
                  if u not in {c.get("channel_username") for c in channel_rows}]:
        _runtime["deleted_check_errors"].pop(stale, None)

    deleted_total = 0

    # ---- ۱) پستِ یتیم: کانالش از بازدیدِ خودکار حذف شده ----
    # ⚠️ باگِ واقعیِ حجم: delete_auto_view فقط scheduled_views رو پاک می‌کرد و
    # ردیفِ وضعیت + تاریخچه‌ی بازدیدکننده‌ها برایِ همیشه می‌موندن. این ردیف‌ها
    # هیچ‌وقت دیگه خونده نمی‌شن، چون کانالشون وجود نداره.
    orphans = [st for st in states if st["auto_view_id"] not in live_channels]
    if orphans:
        deleted_total += await _purge_deleted_posts(
            orphans, "کانالش دیگه در بازدیدِ خودکار ثبت نیست"
        )

    now = time.monotonic()
    due = []
    for st in states:
        if st["auto_view_id"] not in live_channels:
            continue
        key = _post_key(st["auto_view_id"], st["message_id"])
        if key in _deleted_posts:
            continue
        next_at = _deleted_next_check.get(key)
        if next_at is not None and now < next_at:
            continue
        due.append(st)

    if not due:
        _runtime["deleted_last_batch"] = 0
        return deleted_total

    # اولویت: اول پست‌هایی که سرنخِ «شاید پاک شده» دارن (تاییدشون فوریه)، بعد
    # پست‌هایی که خیلی وقته چک نشدن
    def _priority(st):
        key = _post_key(st["auto_view_id"], st["message_id"])
        return (-_delete_strikes.get(key, 0), _deleted_next_check.get(key, 0.0))

    due.sort(key=_priority)
    batch = due[:DELETED_CHECK_MAX_POSTS]
    _runtime["deleted_last_batch"] = len(batch)

    groups = {}
    for st in batch:
        groups.setdefault(st["auto_view_id"], []).append(st)

    sem = asyncio.Semaphore(max(1, DELETED_CHECK_CONCURRENCY))
    lock = asyncio.Lock()
    confirmed = []

    async def guarded(av_id, items):
        row = channels.get(av_id)
        if not row:
            return
        async with sem:
            if not _running:
                return
            try:
                await _check_channel_posts_deleted(row, items, lock, confirmed)
            except Exception as e:
                _runtime["deleted_check_errors"][row.get("channel_username", "?")] = (
                    f"{type(e).__name__}: {e}"
                )

    await asyncio.gather(*(guarded(av, items) for av, items in groups.items()),
                         return_exceptions=True)

    if confirmed:
        deleted_total += await _purge_deleted_posts(
            confirmed,
            f"تاییدشده با {DELETED_CONFIRM_STRIKES} چکِ پشتِ‌سرِهم"
        )
    return deleted_total


async def _pick_watcher(row: dict) -> dict:
    """
    اکانتِ ناظرِ سالم برایِ این کانال - و اگه ناظرِ ثبت‌شده خراب بود، خودکار یکی دیگه
    انتخاب و *ثبت* می‌کنه.

    ⚠️ باگی که اینجا رفع شد: ناظر موقعِ ثبتِ کانال همیشه اولین اکانتِ استخر بود، پس
    همه‌ی کانال‌ها یک ناظرِ مشترک داشتن. با فریز/لاگ‌اوتِ همون یک اکانت، *همه‌ی*
    کانال‌ها برایِ همیشه کور می‌شدن و هیچ پستِ جدیدی زمان‌بندی نمی‌شد - دقیقاً همون
    «پستِ جدید رو نمی‌خونه». حالا ربات خودش رو تعمیر می‌کنه و نیازی به ری‌استارت نیست.
    """
    # ⚠️ این تابع قبلاً فقط is_logged_in و is_frozen رو چک می‌کرد - یعنی دروازه‌ی
    # فاصله‌ی حداقلی، سقفِ روزانه، ساعاتِ سکوت، قرنطینه و مکثِ بعدِ FloodWait هیچ‌کدوم
    # روی اکانتِ ناظر اعمال نمی‌شد. نتیجه: یه اکانت روزی ۱۹۲۰ درخواست می‌زد بدونِ
    # اینکه هیچ محافظی جلوشو بگیره. حالا ناظر هم مثلِ هر مصرف‌کننده‌ی دیگه‌ای
    # از activity_guard رد می‌شه.
    settings = await activity_guard.get_settings()
    usage = await activity_guard.daily_usage_map()

    watcher_id = row.get("watcher_account_id")
    if watcher_id:
        watcher = await get_account(watcher_id)
        if watcher and activity_guard.is_ready(watcher, settings, usage):
            return watcher
        # ناظر سالمه ولی همین الان نوبتش نیست (فاصله/سقف/سکوت): *عوضش نکن*.
        # جایگزینیِ ناظر فقط برایِ خرابیِ واقعیه - وگرنه هر بار که دروازه بسته بود
        # یه اکانتِ سالمِ دیگه رو هم وارد چرخه‌ی سوختن می‌کردیم.
        if watcher and activity_guard.is_usable(watcher):
            return None

    # ناظر واقعاً خرابه (لاگ‌اوت/فریز) - از بینِ اکانت‌هایِ *آماده* یکی انتخاب کن
    accounts = await _cached_accounts()
    pool = _eligible_from(accounts, row.get("country_code")) or _eligible_from(accounts)
    pool = [a for a in pool if activity_guard.is_ready(a, settings, usage)]
    if not pool:
        return None

    replacement = random.choice(pool)
    try:
        await set_auto_view_watcher(row["id"], replacement["id"])
    except Exception:
        pass  # ثبتش نشد، ولی همین دور رو با این اکانت جلو می‌ریم
    print(f"🔁 اکانتِ ناظرِ @{row['channel_username']} خراب بود - "
          f"خودکار به اکانتِ {replacement['id']} سوییچ شد")
    return replacement


def _note_watcher_failure(watcher: dict, exc: Exception) -> None:
    """
    خطایِ اکانتِ ناظر رو به activity_guard گزارش می‌کنه تا کولدانِ درست بخوره.

    قبلاً خطا فقط توی یه رشته‌ی نمایشی ثبت می‌شد: یه اکانت می‌تونست FloodWait
    بخوره و حلقه ۴۵ ثانیه بعد دقیقاً همون درخواست رو دوباره از همون اکانت بزنه.
    """
    aid = watcher.get("id")
    if aid is None:
        return
    try:
        if telethon_handler.is_frozen_response(exc):
            activity_guard.report_frozen(aid)
        elif isinstance(exc, FloodWaitError):
            activity_guard.report_flood(aid, getattr(exc, "seconds", 0) or 0)
        elif telethon_handler.is_peer_flood(exc):
            activity_guard.report_peer_flood(aid)
    except Exception:
        pass


# هر ردیفِ بازدیدِ خودکار یه قفل: حالا دو مسیر به _check_one_channel می‌رسن (اسکنِ
# دوره‌ای و تریگرِ فوریِ notify_channel_post). بدونِ قفل، هر دو با یه last_idِ کهنه
# همون پست رو هم‌زمان برمی‌داشتن و ممکن بود last_id به عقب برگرده.
_channel_locks: dict = {}
# کانال‌هایی که تریگرِ فوری‌شون در صفه (برایِ جمع‌کردنِ چند پستِ یه آلبوم در یک چک)
_instant_pending: set = set()
_instant_tasks: set = set()
# پستِ آلبوم چند آپدیتِ جدا پشتِ‌سرِهمه - کمی صبر کن تا همه برسن، بعد یک‌جا چک کن
INSTANT_DEBOUNCE_SECONDS = 2.0
# پستی که این‌قدر (ثانیه) قبل از ثبتِ کانال منتشر شده، «قدیمی» حساب می‌شه و بازدید
# نمی‌گیره - حاشیه برایِ اختلافِ ساعتِ سرورِ تلگرام و دیتابیس
OLD_POST_GRACE_SECONDS = 120


def _channel_lock(auto_view_id) -> asyncio.Lock:
    lock = _channel_locks.get(auto_view_id)
    if lock is None:
        lock = asyncio.Lock()
        _channel_locks[auto_view_id] = lock
    return lock


def notify_channel_post(username: str) -> None:
    """
    bot.py همون لحظه‌ای که آپدیتِ channel_post رسید صداش می‌زنه.

    قبلاً پستِ رسیده فقط در بافرِ channel_watch می‌موند تا اسکنِ دوره‌ای (هر ~۴۵
    ثانیه) برش داره. حالا همون لحظه زمان‌بندی می‌شه؛ اسکنِ دوره‌ای فقط تورِ ایمنیه.
    هیچ اکانتی درگیر نمی‌شه - فقط بافرِ ربات و دیتابیس.
    """
    if not _running:
        return
    key = channel_watch.norm(username)
    if not key or key in _instant_pending:
        return
    _instant_pending.add(key)
    try:
        task = asyncio.get_running_loop().create_task(_instant_check(key))
    except RuntimeError:
        _instant_pending.discard(key)
        return
    _instant_tasks.add(task)
    task.add_done_callback(_instant_tasks.discard)


async def _instant_check(key: str) -> None:
    try:
        await asyncio.sleep(INSTANT_DEBOUNCE_SECONDS)
    finally:
        # از همین لحظه پستِ تازه یه تریگرِ جدید می‌سازه - چیزی جا نمی‌مونه
        _instant_pending.discard(key)
    try:
        rows = await get_auto_view_channels()
        for r in rows:
            if channel_watch.norm(r.get("channel_username")) == key:
                await asyncio.wait_for(_check_one_channel(r), timeout=CHANNEL_SCAN_TIMEOUT)
    except Exception as e:
        _runtime["scan_errors"][key] = f"{type(e).__name__}: {e}"


async def _check_one_channel(row: dict):
    """چکِ یک کانالِ ثبت‌شده - اگر پستِ جدید داشت، برای هر پست یک چرخه‌ی مستقل می‌سازد."""
    async with _channel_lock(row["id"]):
        # ردیف تازه خونده می‌شه: ممکنه مسیرِ دیگه (اسکن/تریگرِ فوری) همین الان
        # last_message_id رو جلو برده باشه
        fresh = await get_auto_view_channel(row["id"])
        if not fresh:
            return
        await _check_one_channel_locked(fresh)


async def _check_one_channel_locked(row: dict):
    username = row["channel_username"]
    # کانالی که وسطِ همین دورِ اسکن از لیست حذف شد: چکِ پستِ جدیدش بی‌معنیه
    if is_auto_view_revoked(row["id"]):
        return
    since_id = row["last_message_id"] or 0

    # ========================================================================
    # مسیرِ اول: خوراکِ ربات (بدونِ هیچ مصرفی از اکانت‌ها)
    # ------------------------------------------------------------------------
    # ⚠️ این‌جا بزرگ‌ترین منبعِ فریزِ کلِ پروژه بود. قبلاً همین حلقه هر ۴۵ ثانیه
    # یه اکانتِ واقعی رو مجبور می‌کرد getHistory بزنه: روزی ۱۹۲۰ درخواست به‌ازایِ
    # هر کانال، همه از یک اکانتِ ناظرِ ثابت، با فاصله‌ی دقیقاً یکنواخت، ۲۴ ساعته -
    # *حتی وقتی کانال هیچ پستی نداشت*. هیچ‌کدوم از محافظ‌هایِ activity_guard هم
    # روی این مسیر اعمال نمی‌شد. وقتی ناظر می‌سوخت، کد یه اکانتِ تصادفیِ دیگه رو
    # جاش می‌ذاشت تا اون هم بسوزه - یه زنجیره‌ی فرسایشی.
    #
    # چون ربات ادمینِ کاناله، تلگرام پستِ جدید رو خودش بهش پوش می‌کنه. پس اون
    # پولینگ از اساس زائد بود.
    # ========================================================================
    if await channel_watch.is_watchable(username):
        new_messages = channel_watch.new_posts_since(username, since_id)
        _runtime["scan_errors"].pop(username, None)
        _runtime["bot_watched"].add(username)
        await update_auto_view_check_status(row["id"], error=None)
        if not new_messages:
            return
        await _schedule_new_posts(row, new_messages)
        return

    # ========================================================================
    # ربات ادمینِ این کانال نیست -> هیچ کاری انجام نمی‌شه
    # ------------------------------------------------------------------------
    # ⚠️ اینجا قبلاً فالبکِ اکانتی بود: یه اکانتِ واقعی هر ۴۵ ثانیه getHistory
    # می‌زد. روزی ۱۹۲۰ درخواست از یک اکانتِ ثابت، حتی روی کانالی که هیچ پستی
    # نداشت - و وقتی اون اکانت می‌سوخت، یه اکانتِ تصادفیِ دیگه جاش می‌اومد تا
    # اون هم بسوزه. همون زنجیره‌ای که باعثِ فریزِ دسته‌جمعی می‌شد.
    #
    # حالا ثبتِ کانال بدونِ ادمین‌بودنِ ربات اصلاً ممکن نیست، پس رسیدن به اینجا
    # یعنی ربات *بعد از ثبت* از کانال حذف شده. جوابِ درست توقف و اطلاع‌دادن به
    # ادمینه، نه سوزوندنِ اکانت.
    # ========================================================================
    msg = ("ربات دیگه ادمینِ این کانال نیست، پس پستِ جدید تشخیص داده نمی‌شه. "
           "ربات رو دوباره ادمینِ کانال کن - تا اون موقع هیچ بازدیدِ جدیدی "
           "زمان‌بندی نمی‌شه (هیچ اکانتی هم درگیر نمی‌شه).")
    _runtime["scan_errors"][username] = msg
    await update_auto_view_check_status(row["id"], error=msg)


async def _schedule_new_posts(row: dict, new_messages: list) -> None:
    """
    برایِ هر پستِ جدید یک چرخه‌ی بازدید می‌سازد.

    از _check_one_channel جدا شد چون حالا دو منبعِ کاملاً متفاوت به اینجا می‌رسن -
    خوراکِ ربات و فالبکِ اکانتی - و منطقِ زمان‌بندی باید برایِ هر دو *دقیقاً* یکی
    باشه. کپی‌کردنش یعنی یه روز یکیشون از قلم می‌افته.
    """
    username = row["channel_username"]
    country_code = row.get("country_code")
    view_min = int(row.get("view_count_min") or row["view_count"])
    view_max = int(row.get("view_count_max") or row["view_count"])
    if view_min > view_max:
        view_min, view_max = view_max, view_min
    start_delay_minutes = row.get("start_delay_minutes") or 0
    round_interval_minutes = row.get("spread_minutes") or 0
    views_per_round = row.get("views_per_round")

    # سقفِ پستِ هر اسکن - بقیه در دورهایِ بعدی (چیزی گم نمی‌شه، فقط بار پخش می‌شه)
    if len(new_messages) > MAX_NEW_POSTS_PER_SCAN:
        _runtime["deferred_posts"][username] = len(new_messages) - MAX_NEW_POSTS_PER_SCAN
        new_messages = new_messages[:MAX_NEW_POSTS_PER_SCAN]
    else:
        _runtime["deferred_posts"].pop(username, None)

    # زمانِ ثبتِ کانال (UTC، از CURRENT_TIMESTAMPِ SQLite). پستی که قبل از ثبت
    # منتشر شده هیچ‌وقت نباید بازدید بگیره - نگاه کن به _finalize_autoview در bot.py:
    # نقطه‌ی شروع دیگه با اکانت از تاریخچه‌ی کانال خونده نمی‌شه.
    registered_at = None
    try:
        if row.get("created_at"):
            registered_at = datetime.fromisoformat(str(row["created_at"]))
            if registered_at.tzinfo is not None:
                registered_at = registered_at.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        registered_at = None

    for msg in new_messages:
        # چکِ دوباره داخلِ حلقه: خودِ چکِ کانال تا ۶۰ ثانیه طول می‌کشه و ادمین
        # می‌تونه دقیقاً همون وسط کانال رو حذف کنه
        if is_auto_view_revoked(row["id"]):
            return
        msg_date = getattr(msg, "date", None) or datetime.now(timezone.utc)
        if msg_date.tzinfo is not None:
            msg_date = msg_date.astimezone(timezone.utc).replace(tzinfo=None)

        if registered_at is not None and \
                msg_date < registered_at - timedelta(seconds=OLD_POST_GRACE_SECONDS):
            await update_auto_view_last_id(row["id"], msg.id)
            continue

        # عددِ هر پست یک‌بار از بازه انتخاب و در state ذخیره می‌شود.
        view_count = random.randint(view_min, view_max)
        # تنها دوره‌ی این پست: ۱۰۰٪ سفارش، با همان شروعِ تأخیردارِ ثبت‌شده.
        first_cycle_start = msg_date + timedelta(minutes=max(start_delay_minutes, 0))
        cycle_state = await add_auto_view_cycle_state(
            auto_view_id=row["id"],
            message_id=msg.id,
            channel_username=username,
            country_code=country_code,
            base_view_count=view_count,
            next_cycle_at_iso=first_cycle_start.isoformat(timespec="seconds"),
            start_delay_minutes=start_delay_minutes,
            spread_minutes=round_interval_minutes,
            views_per_round=views_per_round,
            follower_percent=_follower_share(row),
        )
        if cycle_state.get("status") != "success":
            # ⚠️ اینجا دو حالتِ کاملاً متفاوت وجود داره که قبلاً یکسان رفتار می‌شدن:
            #
            #   ۱) وضعیتِ این پست از قبل ثبت شده (خطایِ UNIQUE). این یعنی کار *قبلاً*
            #      انجام شده - مثلاً ربات وسطِ کار ری‌استارت شده و last_id آپدیت نشده.
            #      قبلاً کد اینجا continue می‌زد و last_id رو جلو نمی‌برد، پس ربات تا
            #      ابد هر ۴۵ ثانیه همون پست رو دوباره می‌گرفت و همون خطا رو می‌خورد -
            #      یه چرخه‌یِ بی‌پایانِ کارِ الکی که خودش هم تمام نمی‌شد. حالا چون کار
            #      واقعاً انجام شده، last_id جلو می‌ره و از این پست رد می‌شیم.
            #
            #   ۲) خطایِ واقعیِ دیتابیس. اینجا واقعاً باید متوقف شیم تا دورِ بعد
            #      دوباره تلاش بشه (وگرنه پست بی‌بازدید رد می‌شه).
            err = str(cycle_state.get("message", ""))
            already_exists = "UNIQUE" in err.upper() or "constraint" in err.lower()
            if already_exists:
                await update_auto_view_last_id(row["id"], msg.id)
                continue
            await update_auto_view_check_status(row["id"], error=err or "ثبتِ چرخه شکست خورد")
            return

        # last_message_id فقط بعدِ ثبتِ *موفقِ* چرخه جلو می‌ره، پس هیچ پستی از قلم نمی‌افته
        await update_auto_view_last_id(row["id"], msg.id)


async def _view_with_timeout(session_file: str, link_info: dict, as_follower: bool = False) -> dict:
    try:
        return await asyncio.wait_for(
            view_post(session_file, link_info, as_follower=as_follower),
            timeout=ACCOUNT_ACTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        return {"status": "error", "message": f"بیش از {ACCOUNT_ACTION_TIMEOUT} ثانیه طول کشید"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class _AccountAllocator:
    """
    هر اکانت در هر پاسِ اجرا حداکثر به یک بازدید داده می‌شود.

    چرا لازمه: قبلاً هر ردیفِ سررسیده جداگانه از کلِ استخر یه اکانتِ رندوم برمی‌داشت.
    وقتی صدها بازدید هم‌زمان سررسید می‌شدن، چند ردیف همون اکانت رو انتخاب می‌کردن و
    چون تلگرام بازدیدِ تکراریِ یه اکانت رو نمی‌شماره، بازدید عملاً هدر می‌رفت.
    """

    def __init__(self, ordered: list):
        self._queue = collections.deque(ordered)
        self._lock = asyncio.Lock()

    async def take(self):
        async with self._lock:
            return self._queue.popleft() if self._queue else None


def _note_activity(sink: dict, guard: dict, account_id, acc: dict = None) -> None:
    """
    یه «فعالیتِ شمارش‌شدنی» برایِ این اکانت ثبت می‌کنه.

    دو جا هم‌زمان: در نقشه‌ی درون-حافظه‌ی guard["usage"] (تا سقفِ روزانه *داخلِ
    همین پاس* هم رعایت بشه، نه فقط پاسِ بعد) و در sink (تا در پایانِ پاس همه با
    یه کوئری نوشته بشن - نگاه کن به increment_account_activity_bulk).
    """
    if account_id is None:
        return
    usage = guard.get("usage")
    if isinstance(usage, dict):
        usage[account_id] = usage.get(account_id, 0) + 1
    sink["activity"][account_id] = sink["activity"].get(account_id, 0) + 1
    # ⚠️ لایه‌یِ ضرب‌آهنگ *همین لحظه* مهر می‌خوره، نه در پایانِ پاس. بدونِ این، چند
    # تسکِ موازیِ همین پاس همه یه اکانتِ «آزاد» می‌دیدن و پشتِ‌سرِهم ازش استفاده
    # می‌کردن - یعنی دقیقاً همون رگبارِ ماشینی که باعثِ فریز می‌شه.
    activity_guard.mark_used(account_id, weight=1, acc=acc,
                             settings=guard.get("settings"))


def _retry_at(reason: str, item: dict, guard: dict, **kw) -> str:
    """
    موعدِ تلاشِ بعدیِ یه ردیف بر اساسِ *علتِ* اجرا‌نشدنش (تمدیدِ هوشمندِ سفارش).

    ⚠️ جایگزینِ _cap_retry_atِ قبلی که هر علتی رو یک‌کاسه تا فردا عقب می‌انداخت:
    یه کمبودِ دو دقیقه‌ای هم ۲۴ ساعت جریمه می‌گرفت. حالا تصمیم در یک جا و برایِ
    بازدید/ری‌اکشن/شیر یکسان گرفته می‌شه - نگاه کن به activity_guard.retry_plan.
    """
    return activity_guard.retry_plan(
        reason, guard.get("settings"),
        extended_seconds=activity_guard.row_extension_seconds(item),
        pressure=_runtime.get("pressure") or 0.0,
        **kw
    )


async def _run_scheduled_view(item: dict, link_info: dict, allocator: "_AccountAllocator",
                              member_ids: set, sem: asyncio.Semaphore, sink: dict,
                              guard: dict) -> str:
    """
    یک ردیفِ بازدید را اجرا می‌کند و نتیجه را در sink جمع می‌کند (بدونِ نوشتن در
    دیتابیس). نوشتنِ همه‌ی نتایج یک‌جا و در پایانِ پاس انجام می‌شه - نگاه کن به
    apply_scheduled_view_results. خروجی: done | failed | no_account | capped

    guard = {"settings": تنظیماتِ رفتارِ اکانت‌ها، "usage": شمارنده‌ی امروز}. سقفِ
    روزانه *همین‌جا* هم دوباره چک می‌شه (نه فقط موقعِ ساختنِ استخر): یه اکانت
    می‌تونه در یه پاس به چند پستِ مختلف داده بشه، و بدونِ این چک همون اکانت از
    سقفش رد می‌شد.
    """
    async with sem:
        attempted = False
        blocked_reason = None
        for _ in range(MAX_VIEW_ATTEMPTS_PER_SLOT):
            candidate = await allocator.take()
            if candidate is None:
                # ⚠️ باگِ رفع‌شده: قبلاً این ردیف بدونِ هیچ علامتی رها می‌شد، پس با
                # fire_atِ گذشته هر ۴ ثانیه دوباره انتخاب می‌شد و حلقه‌ی اجرا رو
                # بی‌دلیل داغ نگه می‌داشت. حالا صریحاً چند ثانیه موکول می‌شه.
                sink["postponed"].append((item["id"], _retry_at("no_account", item, guard)))
                _runtime["total_postponed"] += 1
                return "no_account"
            # ⚠️ دوباره‌چکِ کاملِ رفتارِ اکانت (نه فقط سقفِ روزانه): بینِ ساختنِ استخر و
            # رسیدنِ نوبتِ این ردیف، ممکنه همین اکانت در همین پاس چند بار استفاده شده
            # باشه، به سقفِ ساعتی خورده باشه، یا FloodWait گرفته باشه.
            ok, reason = activity_guard.readiness(candidate, guard["settings"], guard["usage"])
            if not ok:
                blocked_reason = reason
                continue
            attempted = True
            is_member = candidate["id"] in member_ids
            result = await _view_with_timeout(
                candidate["session_file"], link_info, as_follower=is_member
            )
            status = result.get("status")
            # ⚠️ «deferred» یعنی هیچ درخواستی به تلگرام نرفت (سهمیه‌ی resolve). پس
            # نباید از سهمیه‌ی فعالیتِ اکانت هم کم بشه - فقط ردیف عقب می‌افته.
            if status == "deferred":
                sink["postponed"].append((
                    item["id"],
                    _retry_at("resolve", item, guard,
                              retry_after=result.get("retry_after") or 0.0),
                ))
                _runtime["total_postponed"] += 1
                _runtime["resolve_deferred"] = _runtime.get("resolve_deferred", 0) + 1
                _note_capacity(
                    item["channel_username"], item["message_id"],
                    "سهمیه‌یِ resolveِ روزانه‌ی اکانت‌ها پر شده؛ بازدید عقب افتاد "
                    "(این محافظِ اصلیِ ضدِ فریزه)", kind="resolve",
                )
                return "deferred"
            # درخواست واقعاً به تلگرام رفت → از سهمیه‌ی امروزِ این اکانت کم می‌شه،
            # موفق یا ناموفق (همون قانونی که ری‌اکشنِ خودکار دارد)
            _note_activity(sink, guard, candidate["id"], acc=candidate)
            if status in ("flood", "peer_flood"):
                # ⚠️ قبلاً FloodWait مثلِ یه خطایِ معمولی بود و موتور فوراً اکانتِ بعدی
                # رو امتحان می‌کرد - یعنی همون فشاری که فلود رو ساخته بود ادامه پیدا
                # می‌کرد. حالا اکانت واقعاً استراحت می‌کنه و سقفِ امروزش هم جریمه می‌شه.
                if status == "flood":
                    wait = activity_guard.report_flood(
                        candidate["id"], result.get("seconds") or 0)
                else:
                    wait = activity_guard.report_peer_flood(candidate["id"])
                _runtime["flood_events"] = _runtime.get("flood_events", 0) + 1
                sink["postponed"].append((
                    item["id"], _retry_at("flood", item, guard, flood_seconds=wait,
                                         account_id=candidate["id"]),
                ))
                _runtime["total_postponed"] += 1
                _note_capacity(
                    item["channel_username"], item["message_id"],
                    f"اکانت FloodWait گرفت؛ {int(wait)} ثانیه استراحت داده شد و "
                    "سقفِ امروزش کم شد", kind="flood",
                )
                return "flood"
            if status == "frozen":
                # اکانت فریز شده: همون لحظه از کلِ چرخه بیرون می‌ره (هم در دیتابیس هم
                # در حافظه) تا حتی یه درخواستِ دیگه هم براش فرستاده نشه
                activity_guard.report_frozen(candidate["id"])
                try:
                    await set_account_frozen(candidate["id"], True)
                except Exception:
                    pass
                await _cached_accounts(force=True)
                _runtime["frozen_detected"] = _runtime.get("frozen_detected", 0) + 1
                sink["postponed"].append((item["id"], _retry_at("no_account", item, guard)))
                _runtime["total_postponed"] += 1
                return "frozen"
            if status == "deleted":
                # ⚠️ این «شکستِ اکانت» نیست: خودِ تلگرام گفته این آی‌دیِ پیام وجود
                # نداره. نه ردیف رو failed می‌کنیم (که یعنی بازدیدِ سوخته) و نه
                # اکانتِ بعدی رو امتحان می‌کنیم (همه همین جواب رو می‌گیرن). فقط
                # سرنخ ثبت و ردیف کمی موکول می‌شه؛ پاسبانِ پستِ پاک‌شده در دورِ
                # بعدی با پرسیدن از کانال حکمِ نهایی رو می‌ده و - اگه واقعاً پاک
                # شده باشه - همین ردیف‌ها رو حذف می‌کنه.
                _flag_delete_suspect(item["auto_view_id"], item["message_id"])
                retry_at = utcnow() + timedelta(
                    seconds=random.randint(RETRY_POSTPONE_MIN_SECONDS,
                                           RETRY_POSTPONE_MAX_SECONDS)
                )
                sink["postponed"].append((item["id"], retry_at.isoformat(timespec="seconds")))
                _runtime["total_postponed"] += 1
                return "deleted"
            if status == "success":
                sink["done"].append((item["id"], candidate["id"]))
                # تاریخچه‌یِ دقیق: کدام اکانت، کدام پست را دید (تا در همین دوره
                # دوباره انتخاب نشود)
                sink["viewers"].append(
                    (item["auto_view_id"], item["message_id"], candidate["id"])
                )
                _runtime["total_done"] += 1
                return "done"
            # اکانتِ خراب برنمی‌گرده به صف - سراغِ اکانتِ بعدی می‌ریم
        if not attempted:
            # هیچ تلاشِ واقعی‌ای انجام نشد - قوانینِ «🎭 رفتارِ اکانت‌ها» مانع بودن.
            # این «شکست» نیست، «الان نه»ـه؛ علامت‌زدنش به‌عنوانِ failed یعنی خوردنِ
            # بی‌صدایِ بازدیدِ سفارشِ مشتری. مدتِ عقب‌افتادن از *علتِ واقعی* می‌آد:
            # نوبتِ ضرب‌آهنگ → چند دقیقه، سقفِ روزانه → تا نیمه‌شبِ تهران،
            # ساعاتِ سکوت → تا صبح.
            reason = blocked_reason or "no_capacity"
            sink["postponed"].append((item["id"], _retry_at(reason, item, guard)))
            _runtime["total_postponed"] += 1
            _runtime["cap_postponed"] += 1
            _runtime["postpone_reasons"][reason] = \
                _runtime["postpone_reasons"].get(reason, 0) + 1
            return "capped"
        sink["failed"].append(item["id"])
        _runtime["total_failed"] += 1
        return "failed"


async def _fire_due_views():
    """
    بازدیدهایی که موعدشون رسیده (fire_at <= الان) رو واقعاً اجرا می‌کند.

    ⚠️ باگ‌هایی که اینجا رفع شدن: این تابع قبلاً کاملاً **ترتیبی** بود - برایِ هر ردیف
    یه کوئریِ used_ids و یه کوئریِ membership می‌زد و بعد منتظرِ خودِ بازدید می‌موند (تا
    ۴۵ ثانیه تایم‌اوت). با ۵۰ کانال این یعنی هزاران کوئری و ساعت‌ها زمان، پس صف مرتب
    عقب می‌موند و زمان‌بندی به‌هم می‌ریخت. حالا:
      • ردیف‌ها بر اساسِ پست گروه می‌شن و used_ids/membership یک‌بار برایِ هر پست خونده می‌شه
      • لیستِ اکانت‌ها کش می‌شه (نه یه کوئریِ کاملِ جدول در هر تیک)
      • بازدیدها با هم‌زمانیِ *تطبیقی* اجرا می‌شن (بر اساسِ فشارِ واقعیِ سرور)
      • هر اکانت در هر پاس فقط به یک بازدید تخصیص داده می‌شه
      • اگه اکانتِ آزاد ته کشید ولی اکانتِ واجدِ شرایط وجود داره، بازدید عقب می‌افته
        (نه اینکه «شکست‌خورده» علامت بخوره)
    """
    now_iso = utcnow().isoformat(timespec="seconds")
    due = await get_due_scheduled_views(now_iso, limit=DUE_FETCH_LIMIT)
    if not due:
        # ⚠️ باگِ رفع‌شده «فشارِ سرور بی‌دلیل بالا می‌موند»: قبلاً همین‌جا فقط return
        # می‌شد و last_batch رویِ *آخرین دسته‌ی شلوغ* جا می‌موند. همون عدد به‌عنوانِ
        # سنجه‌ی backlog به resource_monitor گزارش می‌شد، پس یه صفِ کاملاً خالی هم تا
        # ابد مثلِ یه صفِ ۸۰۰ ردیفیِ عقب‌افتاده خونده می‌شد: فشار مصنوعی بالا می‌رفت،
        # هم‌زمانی به کف می‌چسبید، و شروعِ بازدیدِ پست‌هایِ جدید بی‌هیچ دلیلِ واقعی عقب
        # می‌افتاد. صفِ خالی = فشارِ صفر.
        if _runtime["last_batch"] or _runtime["backlog_capped"]:
            _runtime["last_batch"] = 0
            _runtime["backlog_capped"] = False
            _current_concurrency(backlog=0)   # عددِ پنل هم همون لحظه واقعی بشه
        return

    accounts_cache = await _cached_accounts()
    # نگهبانِ فعالیت - یه‌بار برایِ کلِ این پاس (نه به‌ازایِ هر ردیف). تنظیمات کشِ
    # ۱۰ ثانیه‌ای دارن، پس تغییرِ سقف از منویِ «🎭 رفتارِ اکانت‌ها» تقریباً فوری
    # اثر می‌کنه.
    guard = {
        "settings": await activity_guard.get_settings(),
        "usage": await activity_guard.daily_usage_map(),
    }
    # طولِ همین دسته، معیارِ «چقدر عقبیم» - اگه به سقفِ خواندن خورده، یعنی صف بلندتره
    _runtime["last_batch"] = len(due)
    _runtime["backlog_capped"] = len(due) >= DUE_FETCH_LIMIT
    sem = asyncio.Semaphore(_current_concurrency(backlog=len(due)))

    # گروه‌بندی بر اساسِ پست - همه‌ی ردیف‌هایِ یک پست استخر و عضویتِ مشترک دارن
    groups = collections.OrderedDict()
    for item in due:
        groups.setdefault((item["auto_view_id"], item["message_id"]), []).append(item)

    # همه‌ی «چه اکانت‌هایی این پست را دیده‌اند» با یک کوئری، نه یکی به‌ازایِ هر پست
    used_map = await get_done_account_ids_for_view_messages(list(groups.keys()))

    # نتایجِ این پاس اینجا جمع می‌شن و در پایان یک‌جا نوشته می‌شن
    sink = {"done": [], "failed": [], "postponed": [], "viewers": [], "activity": {}}

    tasks = []
    for (auto_view_id, message_id), items in groups.items():
        # پستِ تاییدشده‌ی پاک‌شده: ردیف‌هاش همین حالا از دیتابیس حذف شدن، ولی این
        # دسته چند میلی‌ثانیه قبلِ حذف خونده شده بود. هیچ اکانتی براش خرج نمی‌کنیم.
        if _is_known_deleted(auto_view_id, message_id):
            continue
        # کانالی که ادمین همین حالا از لیست حذف کرد: این دسته چند میلی‌ثانیه قبلِ
        # حذف خونده شده بود. هیچ بازدیدی براش زده نمی‌شه - «همون موقع استپ».
        if is_auto_view_revoked(auto_view_id):
            _runtime["revoked_views_blocked"] += len(items)
            continue
        username = items[0]["channel_username"]
        country_code = items[0].get("country_code")
        ensure_member = bool(items[0].get("join_before_view"))

        used_ids = used_map.get((auto_view_id, message_id), set())
        membership = await _cached_membership(username)

        # لایه‌ی اول (ساختاری): لاگین، غیرِ فریز، هم‌کشور، و این پست را ندیده
        structural_pool = [
            a for a in _eligible_from(accounts_cache, country_code)
            if a["id"] not in used_ids
        ]
        # لایه‌ی دوم (رفتارِ اکانت‌ها): قرنطینه‌یِ سشنِ تازه + سقفِ روزانه/گرم‌کردن.
        # این‌ها اکانت را *موقتاً* کنار می‌گذارند، پس ردیفشان «شکست» نیست - عقب
        # می‌افتد (پایین‌تر).
        # ⚠️ readiness به‌جایِ is_ready: هم جواب می‌ده هم *علت*. علت لازمه تا ردیفِ
        # عقب‌افتاده موعدِ درست بگیره - «نوبتش نرسیده» با «سقفش پر شده» زمین تا
        # آسمون فرق دارن و قبلاً هر دو یک‌کاسه تا فردا عقب می‌افتادن.
        base_pool = []
        hold_reasons: dict = {}
        for a in structural_pool:
            ok, why = activity_guard.readiness(a, guard["settings"], guard["usage"])
            if ok:
                base_pool.append(a)
            else:
                hold_reasons[why] = hold_reasons.get(why, 0) + 1
        member_ids = {
            a["id"] for a in base_pool
            if (membership.get(a["id"]) or {}).get("status") == "member"
        }

        link_info = {
            "is_private": False, "username": username,
            "channel_id": None, "message_id": message_id,
        }

        # چند اکانت فقط به‌خاطرِ قوانینِ «🎭 رفتارِ اکانت‌ها» کنار موندن؟
        held_back = len(structural_pool) - len(base_pool)
        if held_back > 0:
            labels = {
                "cap": "سقفِ روزانه پر شده",
                "hourly": "سقفِ ساعتی پر شده",
                "paced": "نوبتِ فعالیتِ بعدی‌شون نرسیده",
                "quiet": "ساعاتِ سکوتِ شبانه",
                "flood": "در حالِ استراحت بعدِ FloodWait",
                "frozen": "فریز",
                "quarantine": "قرنطینه‌یِ سشنِ تازه",
                "usable": "لاگین نیستن",
            }
            detail = "، ".join(
                f"{n} {labels.get(k, k)}"
                for k, n in sorted(hold_reasons.items(), key=lambda kv: -kv[1])
            )
            _note_capacity(
                username, message_id,
                f"{held_back} اکانت این دور استراحت دادیم ({detail}) - "
                "سفارش طولانی‌تر می‌شه ولی اکانت‌ها سالم می‌مونن "
                "(از «🎭 رفتارِ اکانت‌ها» قابلِ تغییره)",
                kind="cap",
            )
            for k, n in hold_reasons.items():
                _runtime["postpone_reasons"][k] = \
                    _runtime["postpone_reasons"].get(k, 0) + n

        members = [a for a in base_pool if a["id"] in member_ids]
        # Other فقط عضونبودنِ تاییدشده است؛ unknown ممکن است واقعاً عضو باشد.
        others = [
            a for a in base_pool
            if (membership.get(a["id"]) or {}).get("status") == "not_member"
        ]
        random.shuffle(members)
        random.shuffle(others)

        targeted = any((i.get("source_type") or "any") != "any" for i in items)
        # ⚠️ عمداً source_groups، نه groups: نامِ groups همون دیکشنریِ پست‌هاست که
        # همین حلقه داره رویش می‌چرخه. بازنویسیِ اسمش وسطِ حلقه (که قبلاً اتفاق
        # می‌افتاد) فقط به‌خاطرِ جزئیاتِ پیاده‌سازیِ iteratorِ پایتون بی‌خطر بود؛ هر
        # کدِ بعدی که داخلِ حلقه به groups به‌عنوانِ دیکشنری دست می‌زد، AttributeError
        # می‌گرفت. یه مینِ زمانی که حالا خنثی شد.
        if targeted:
            source_groups = [
                ("Followers", [i for i in items if i.get("source_type") == "follower"], members),
                ("Other", [i for i in items if i.get("source_type") == "other"], others),
            ]
        else:
            ordered = list(base_pool)
            random.shuffle(ordered)
            if ensure_member:
                ordered.sort(key=lambda a: a["id"] not in member_ids)
            source_groups = [("بازدید", items, ordered)]

        for source_label, source_items, source_pool in source_groups:
            if not source_items:
                continue
            runnable = source_items[:len(source_pool)]
            leftover = source_items[len(source_pool):]
            allocator = _AccountAllocator(source_pool)
            for item in runnable:
                tasks.append(_run_scheduled_view(
                    item, link_info, allocator, member_ids, sem, sink, guard
                ))
            for item in leftover:
                # منبعِ اشتباه جایگزین نمی‌شود؛ نسبتِ نمودار مهم‌تر از تمام‌کردنِ
                # ظاهریِ صف است. بعد از گرم‌شدن عضویت یا آزادشدن سهمیه دوباره تلاش می‌شود.
                if source_pool:
                    retry_reason = "source"
                else:
                    # هیچ اکانتِ آماده‌ای نیست: موعدِ بعدی از *غالب‌ترین علت* می‌آد،
                    # نه از یه عددِ ثابت. اگه همه سقفشون پر شده، تا نیمه‌شبِ تهران؛
                    # اگه فقط نوبتشون نرسیده، چند دقیقه.
                    retry_reason = max(hold_reasons, key=hold_reasons.get) \
                        if hold_reasons else "no_capacity"
                sink["postponed"].append((item["id"], _retry_at(retry_reason, item, guard)))
                _runtime["postpone_reasons"][retry_reason] = \
                    _runtime["postpone_reasons"].get(retry_reason, 0) + 1
                _runtime["total_postponed"] += 1
                _note_capacity(
                    username, message_id,
                    f"برای سهمِ {source_label} فعلاً اکانتِ آماده کافی نیست؛ "
                    "ردیف نگه داشته شد تا نسبت نمودار خراب نشه",
                    kind="source",
                )

    try:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # ⚠️ نتیجه‌ی پست‌هایی که وسطِ همین پاس «پاک‌شده» تایید شدن نباید نوشته بشه:
        # ردیفِ وضعیت و ردیف‌هاشون حذف شدن، پس نوشتنِ تاریخچه‌ی بازدیدکننده فقط یه
        # ردیفِ یتیمِ تازه می‌ساخت - یعنی همون حافظه‌ی الکی که داریم حذفش می‌کنیم.
        # ⚠️ و همین حکم برایِ کانالی که *وسطِ همین پاس* از لیست حذف شد هم صادق است:
        # ردیف‌هاش رفته‌ن، پس نوشتنِ تاریخچه فقط یه ردیفِ یتیمِ تازه می‌ساخت.
        if _deleted_posts or _revoked_auto_views:
            sink["viewers"] = [
                v for v in sink["viewers"]
                if not _is_known_deleted(v[0], v[1]) and not is_auto_view_revoked(v[0])
            ]
        # ⚠️ حتی اگه وسطِ کار خطایی بده، نتایجِ به‌دست‌آمده *باید* نوشته بشن - وگرنه
        # همون بازدیدها دفعه‌یِ بعد دوباره زده می‌شن و اکانت‌ها بی‌دلیل مصرف می‌شن
        await apply_scheduled_view_results(
            sink["done"], sink["failed"], sink["postponed"], sink["viewers"]
        )
        # شمارنده‌یِ «سقفِ فعالیتِ روزانه» - یه کوئری برایِ کلِ پاس. بدونِ این،
        # سقف رویِ آمارِ ناقص حساب می‌شد (بازدیدهایِ خودکار هیچ‌وقت شمرده نمی‌شدن).
        if sink["activity"]:
            await increment_account_activity_bulk(
                sink["activity"], activity_guard.tehran_day_key()
            )
    _runtime["last_fire_at"] = utcnow().isoformat(timespec="seconds")


async def _post_scan_loop():
    """
    فقط دنبالِ پستِ جدید می‌گردد و دوره‌ی بازدیدِ آن را (همون یک دوره) ثبت می‌کند.

    ⚠️ چرا از حلقه‌ی اجرا جدا شد: قبلاً هر دو کار در یک حلقه بودن. چکِ پستِ کانال‌ها
    شبکه‌ایه و با ۵۰ کانال می‌تونه دقیقه‌ها طول بکشه؛ در تمامِ اون مدت هیچ بازدیدی
    اجرا نمی‌شد و بازدیدهایِ سررسیده دیر می‌افتادن. حالا این دو مستقل کار می‌کنن و
    خودِ چکِ کانال‌ها هم موازیه (با سقفِ SCAN_CONCURRENCY).
    """
    _runtime["loops_alive"]["scan"] = True
    try:
        while _running:
            _beat("scan")
            t0 = time.monotonic()
            try:
                rows = await get_auto_view_channels()
                sem = asyncio.Semaphore(SCAN_CONCURRENCY)

                async def guarded(r):
                    async with sem:
                        if not _running:
                            return
                        try:
                            await _check_one_channel(r)
                        except Exception as e:
                            _runtime["scan_errors"][r.get("channel_username", "?")] = f"{type(e).__name__}: {e}"

                if rows:
                    await asyncio.gather(*(guarded(r) for r in rows), return_exceptions=True)

                # کانال‌هایی که دیگه ثبت‌شده نیستن، خطایِ کهنه‌شون هم پاک بشه
                live = {r.get("channel_username") for r in rows}
                for stale in [u for u in _runtime["scan_errors"] if u not in live]:
                    _runtime["scan_errors"].pop(stale, None)

                _beat("scan")
                _runtime["last_scan_channels"] = len(rows)
                _runtime["last_scan_at"] = utcnow().isoformat(timespec="seconds")
                _runtime["last_scan_seconds"] = round(time.monotonic() - t0, 1)

                # چرخه‌هایی که همین الان سررسیدن، بدونِ انتظار برایِ تیکِ بعدی فعال بشن
                await _activate_due_cycle_states(utcnow().isoformat(timespec="seconds"))
            except Exception as e:
                print(f"⚠️  خطا در اسکنِ پست‌هایِ بازدیدِ خودکار: {e}")
            await _sleep_or_stop(_jittered_poll())
    finally:
        _runtime["loops_alive"]["scan"] = False


async def _fire_loop():
    """پست‌هایِ سررسیده را فعال و بازدیدهایِ سررسیده را اجرا می‌کند - با تیکِ کوتاه."""
    _runtime["loops_alive"]["fire"] = True
    last_cleanup = utcnow()
    last_memory_sweep = utcnow()
    expected_wake = time.monotonic()
    try:
        while _running:
            _beat("fire")
            # تاخیرِ واقعیِ همین تیک - ورودیِ تصمیمِ هم‌زمانیِ تطبیقی و پنلِ منابعِ سرور
            lag = max(0.0, time.monotonic() - expected_wake)
            _runtime["tick_lag"] = round(lag, 2)
            if lag > _runtime["tick_lag_max"]:
                _runtime["tick_lag_max"] = round(lag, 2)

            try:
                now = utcnow()
                await _activate_due_cycle_states(now.isoformat(timespec="seconds"))
                await _fire_due_views()
                await _close_completed_post_rounds(utcnow().isoformat(timespec="seconds"))
                _beat("fire")

                if (now - last_memory_sweep).total_seconds() >= MEMORY_SWEEP_INTERVAL_SECONDS:
                    try:
                        await memory_hygiene.routine_maintenance()
                    except Exception as e:
                        _runtime["cycle_last_error"] = f"memory: {type(e).__name__}: {e}"
                    last_memory_sweep = now

                if (now - last_cleanup).total_seconds() >= CLEANUP_INTERVAL_SECONDS:
                    # ۱) پاک‌سازیِ *پست‌محور* - اصلِ کار: به‌محضِ اینکه دوره‌ی یه پست
                    #    تمام شد (و مهلتِ دیدنِ گزارشش گذشت)، کلِ ردِ پایش می‌ره.
                    #    این همون چیزیه که جلویِ ورمِ دیتابیس رو می‌گیره، بدونِ اینکه
                    #    به پست‌هایِ در حالِ اجرا (که تاریخچه‌شون *لازمه*) دست بزنه.
                    await _purge_finished_posts(now)
                    # ۲) شبکه‌یِ ایمنیِ زمانی برایِ ردیف‌هایِ یتیم (پستی که ردیفِ وضعیتش
                    #    به هر دلیل گم شده). حالا که هر پست فقط یک دوره دارد و ظرفِ
                    #    چند ساعت تمام می‌شود، این بازه فقط یه شبکه‌ی ایمنیِ سخاوتمنده.
                    cutoff = (now - timedelta(days=RETENTION_DAYS)).isoformat()
                    await cleanup_old_scheduled_views(cutoff)
                    # سفارشِ تمام‌شده: ۲۴ ساعت و بعد کاملاً از دیتابیس
                    try:
                        await purge_finished_orders("view", finished_order_cutoff_iso())
                    except Exception as e:
                        print(f"⚠️  پاکسازیِ سفارشِ بازدیدِ تمام‌شده: {e}")
                    await cleanup_old_auto_view_message_viewers(cutoff)
                    cycle_cutoff = (now - timedelta(days=VIEW_CYCLE_RETENTION_DAYS)).isoformat()
                    await cleanup_old_auto_view_cycle_states(cycle_cutoff)
                    # ۳) ردیف‌هایِ *یتیم*: بازدیدها/تاریخچه‌ای که ردیفِ وضعیتِ پستشون
                    #    دیگه وجود نداره - مثلاً پستِ پاک‌شده‌ای که پاک‌سازیش وسطِ
                    #    کرش نصفه مونده، یا کانالی که با delete_auto_view حذف شده و
                    #    تاریخچه‌ی بازدیدکننده‌هاش جا مونده. فاصله‌ی یک‌ساعته تضمین
                    #    می‌کنه دوره‌ای که همین الان در حالِ ساخته‌شدنه قربانی نشه.
                    try:
                        orphan = await purge_orphan_auto_view_rows(
                            (now - timedelta(hours=1)).isoformat(timespec="seconds")
                        )
                        _runtime["orphan_rows_purged"] += int(orphan.get("total", 0) or 0)
                    except Exception as e:
                        _runtime["cycle_last_error"] = f"orphan purge: {type(e).__name__}: {e}"
                    last_cleanup = now
            except Exception as e:
                print(f"⚠️  خطا در حلقه‌ی اجرایِ بازدیدِ خودکار: {e}")

            expected_wake = time.monotonic() + FIRE_LOOP_INTERVAL_SECONDS
            await _sleep_or_stop(FIRE_LOOP_INTERVAL_SECONDS)
    finally:
        _runtime["loops_alive"]["fire"] = False


async def _warmup_loop():
    """گرم‌کردن (چکِ عضویت + جوینِ کنترل‌شده) - مستقل از مسیرِ اجرایِ بازدید."""
    _runtime["loops_alive"]["warmup"] = True
    try:
        while _running:
            _beat("warmup")
            try:
                await _warmup_joins()
            except Exception as e:
                print(f"⚠️  خطا در گرم‌کردنِ بازدیدِ خودکار: {e}")
            await _sleep_or_stop(_jittered_poll())
    finally:
        _runtime["loops_alive"]["warmup"] = False


async def _deleted_scan_loop():
    """
    پاسبانِ پستِ پاک‌شده - مستقل از اسکن و اجرا، زیرِ نظرِ همون نگهبانِ خودترمیم.

    ⚠️ چرا حلقه‌ی جدا و نه چند خط داخلِ حلقه‌ی اجرا: تیکِ حلقه‌ی اجرا ۴ ثانیه‌ست و
    باید کوتاه بمونه، ولی چکِ پاک‌شدن شبکه‌ایه و می‌تونه ده‌ها ثانیه طول بکشه. اگه
    داخلِ همون حلقه بود، هر دورِ چک بازدیدهایِ سررسیده رو عقب می‌انداخت - یعنی
    درست کردنِ یه مشکل با ساختنِ یه مشکلِ بدتر.
    """
    _runtime["loops_alive"]["deleted"] = True
    try:
        while _running:
            _beat("deleted")
            t0 = time.monotonic()
            try:
                await _scan_deleted_posts()
                _runtime["deleted_rounds"] += 1
                _runtime["deleted_last_check_at"] = utcnow().isoformat(timespec="seconds")
                _runtime["deleted_last_seconds"] = round(time.monotonic() - t0, 1)
            except Exception as e:
                print(f"⚠️  خطا در پاسبانِ پستِ پاک‌شده: {e}")
            _beat("deleted")
            await _sleep_or_stop(DELETED_CHECK_INTERVAL_SECONDS)
    finally:
        _runtime["loops_alive"]["deleted"] = False


async def _supervisor(factories: dict):
    """
    نگهبانِ خودترمیم: ضربانِ هر حلقه رو می‌پاید و هر حلقه‌ای که گیر کرده یا مرده رو
    تک‌به‌تک از نو راه می‌ندازه.

    چرا این مهم‌ترین بخشه: قبلاً هر سه حلقه با یه asyncio.gather اجرا می‌شدن. اگه یکی
    از اون‌ها رویِ یه await برایِ همیشه معلق می‌شد (مثلاً یه درخواستِ تلگرام که نه
    جواب می‌داد نه خطا)، اون حلقه تا ابد ساکت می‌موند و **هیچ‌کس نمی‌فهمید** - نتیجه‌اش
    همون «پستِ جدید خونده نمی‌شه» یا «بازدید ثبت نمی‌شه» بود که فقط با ری‌استارتِ کلِ
    سرور درست می‌شد. حالا ربات خودش تعمیرش می‌کنه و ری‌استارتِ Railway لازم نیست.

    اگه گرفتنِ خودِ نگهبان هم خطا بدهد، بی‌صدا رد می‌شود و دورِ بعد دوباره تلاش می‌کند -
    نگهبان هیچ‌وقت نباید خودش عاملِ توقف باشد.
    """
    tasks = {name: asyncio.create_task(factory()) for name, factory in factories.items()}
    for name in factories:
        _beat(name)

    try:
        while _running:
            await _sleep_or_stop(WATCHDOG_INTERVAL_SECONDS)
            if not _running:
                break
            now = time.monotonic()

            for name, factory in factories.items():
                task = tasks.get(name)
                limit = HEARTBEAT_LIMIT.get(name, 600)
                age = now - _runtime["heartbeat"].get(name, now)

                dead = task is None or task.done()
                stalled = age > limit

                if not (dead or stalled):
                    continue

                if dead and task is not None:
                    # چرا مرد؟ اگه استثنا بوده باید دیده بشه، نه اینکه بی‌صدا گم بشه
                    exc = task.exception() if not task.cancelled() else None
                    reason = f"با خطا تمام شد: {exc}" if exc else "بی‌دلیل تمام شد"
                else:
                    reason = f"{int(age)} ثانیه ضربان نزد (گیر کرده)"

                print(f"🔧 حلقه‌یِ «{name}» {reason} - در حالِ راه‌اندازیِ مجدد")

                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=5)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass

                tasks[name] = asyncio.create_task(factory())
                _beat(name)
                _runtime["restarts"][name] = _runtime["restarts"].get(name, 0) + 1
                _runtime["last_restart_at"] = utcnow().isoformat(timespec="seconds")
    finally:
        for task in tasks.values():
            if task and not task.done():
                task.cancel()
        # منتظرِ بسته‌شدنِ واقعی بمون تا تسکِ سرگردان جا نمونه
        pending = [t for t in tasks.values() if t]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def auto_view_loop():
    """
    حلقه‌ی اصلیِ بازدیدِ خودکار - چهار کارِ مستقل، زیرِ نظرِ یک نگهبانِ خودترمیم:
    ۱) اسکنِ موازیِ پست‌های جدید و ثبتِ دوره‌ی بازدیدشان
    ۲) شروعِ دوره‌هایِ سررسیده، اجرای بازدیدهای رسیده، و بستنِ پست‌هایِ تمام‌شده
    ۳) گرم‌کردن (چکِ عضویت و جوینِ کنترل‌شده)
    ۴) پاسبانِ پستِ پاک‌شده: پستی که از کانال حذف شده، بازدیدش متوقف و کلِ ردِ پایش
       از دیتابیس پاک می‌شود

    هر کدام مستقل راه می‌افتد؛ اگر یکی گیر کند یا بمیرد، فقط همان یکی از نو اجرا
    می‌شود و بقیه‌ی ربات هیچ اثری نمی‌بیند.
    """
    global _running
    if _running:
        return
    _running = True
    _get_stop_event().clear()
    _runtime["started_at"] = utcnow().isoformat(timespec="seconds")
    try:
        await _supervisor({
            "scan": _post_scan_loop,
            "fire": _fire_loop,
            "warmup": _warmup_loop,
            "deleted": _deleted_scan_loop,
        })
    finally:
        _running = False


def stop_auto_view_loop():
    """توقف - حلقه‌ها و نگهبان فوراً (نه بعدِ خوابِ فعلی‌شون) بیدار و بسته می‌شن."""
    global _running
    _running = False
    try:
        _get_stop_event().set()
    except Exception:
        pass
