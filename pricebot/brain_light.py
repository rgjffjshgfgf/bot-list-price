"""What the bot process needs to know about the brain without loading it (its memory alone is ~60 MB)."""
from __future__ import annotations

from . import config


def enabled() -> bool:
    return config.BRAIN_MODE != "off"


def mode() -> str:
    return config.BRAIN_MODE if config.BRAIN_MODE in ("auto", "teacher", "local") else "auto"


def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def page_note(info: dict) -> str:
    """One line about how the brain handled a file (for the analysis message)."""
    how = info.get("how")
    tpl = info.get("format") or {}
    name = tpl.get("name") or "بدون عنوان"
    need = config.BRAIN_TRUST_AFTER
    streak = min(tpl.get("streak", 0), need)
    if how == "local":
        if info.get("by") == "layout":
            return (f"🧠 قالب آشنا «{name}»: جای همه قیمت‌ها را ربات خودش از روی جدول پیدا کرد "
                    "و فقط ارقام هر قیمت، بریده و بزرگ‌شده، خوانده شد")
        if info.get("by") == "format":
            return f"🧠 تشخیص با هوش خود ربات، بدون Gemini (قالب آشنا: «{name}»)"
        return "🧠 تشخیص با هوش خود ربات، بدون Gemini (قالب تازه، با تجربه‌ای که از لیست‌های قبلی دارد)"
    if how == "fallback":
        return ("🧠 Gemini در دسترس نبود؛ هوش خود ربات (هنوز در حال یادگیری) بررسی کرد. "
                "اگر کادرهای سبز درست است 👍 بزن تا یاد بگیرد.")
    if how == "teacher":
        guess = info.get("guess_ok")
        g = "" if guess is None else (" — حدس خود ربات: درست ✅" if guess else " — حدس خود ربات: اشتباه ❌")
        if info.get("new_format"):
            return f"🎓 قالب جدید «{name}»: Gemini بررسی کرد و ربات یاد گرفت{g}"
        return f"🎓 Gemini بررسی کرد، ربات یاد گرفت (قالب «{name}»: {_fa(streak)} از {_fa(need)} تأیید){g}"
    return ""
