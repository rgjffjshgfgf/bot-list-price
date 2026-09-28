"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader so local runs work without extra packages."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


BASE_DIR = Path(__file__).resolve().parent.parent

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# --- Google Gemini (free tier) -------------------------------------------------
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
# Smarter model, small free quota: photos / scans and free-form commands.
GEMINI_MODEL = _str("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_RPM = _int("GEMINI_RPM", 10)
GEMINI_RPD = _int("GEMINI_RPD", 20)
# Lighter model, large free quota: text PDFs, Excel, re-reads; takes over when Flash runs out.
GEMINI_LITE_MODEL = _str("GEMINI_LITE_MODEL", "gemini-flash-lite-latest")
GEMINI_LITE_RPM = _int("GEMINI_LITE_RPM", 15)
GEMINI_LITE_RPD = _int("GEMINI_LITE_RPD", 500)
# 1 = read every page with both models and merge (more accurate, uses twice the quota).
AI_CONSENSUS = _bool("AI_CONSENSUS", False)

AI_EFFORT_IMAGE = _str("AI_EFFORT_IMAGE", "high")      # photos / scans
AI_EFFORT_TEXT = _str("AI_EFFORT_TEXT", "low")         # PDFs with a text layer
AI_EFFORT_COMMAND = _str("AI_EFFORT_COMMAND", "medium")
AI_EFFORT_TABLE = _str("AI_EFFORT_TABLE", "low")       # Excel transcription
# Independent re-read of every price that came from pixels (scans/photos).
AI_VERIFY_IMAGE_PRICES = _bool("AI_VERIFY_IMAGE_PRICES", True)
AI_PARALLEL_PAGES = max(1, _int("AI_PARALLEL_PAGES", 3))

# --- bot ----------------------------------------------------------------------
# Comma separated Telegram numeric user ids allowed to use the bot. Empty = everyone.
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x.isdigit()
}
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 60)
SESSION_TTL_MINUTES = _int("SESSION_TTL_MINUTES", 180)
IMAGE_EXPORT_DPI = _int("IMAGE_EXPORT_DPI", 200)
WORK_DIR = Path(os.environ["WORK_DIR"]) if os.environ.get("WORK_DIR") else Path(tempfile.gettempdir()) / "pricebot"
# Lasting data (learned knowledge, quota counter). A Railway volume mounted at /data
# is used automatically; without one everything learned is lost on every redeploy.
DATA_DIR = Path(_str("DATA_DIR", "/data" if os.path.isdir("/data") and os.access("/data", os.W_OK)
                     else str(WORK_DIR)))
USAGE_FILE = Path(_str("USAGE_FILE", str(DATA_DIR / "gemini_usage.json")))
# False = files live in the container only and vanish on every redeploy
DATA_PERSISTENT = bool(os.environ.get("DATA_DIR") or os.environ.get("BRAIN_DIR")) or str(DATA_DIR) == "/data"

# --- the bot's own AI ("brain") --------------------------------------------------
# auto    = handle known formats itself, ask Gemini (the teacher) about the rest and learn
# teacher = always ask Gemini, but keep learning in the background
# local   = never ask Gemini to find prices (uses what it has learned so far)
# off     = do not use the brain at all (old behaviour)
BRAIN_MODE = _str("BRAIN_MODE", "auto").lower()
BRAIN_DIR = Path(_str("BRAIN_DIR", str(DATA_DIR / "brain")))
# Checked answers in a row a list format needs before the bot handles it alone.
BRAIN_TRUST_AFTER = _int("BRAIN_TRUST_AFTER", 2)
# Whole pages in a row the price model must get right before it handles unseen formats alone.
BRAIN_GENERAL_AFTER = _int("BRAIN_GENERAL_AFTER", 20)
# Tesseract OCR runs allowed at the same time (each needs a few hundred MB on big pages).
OCR_PARALLEL = _int("OCR_PARALLEL", 2)

# Extra font folder (put B Nazanin, IRANSans, ... TTF files here for better matches on images).
FONTS_DIR = Path(os.environ.get("FONTS_DIR", str(BASE_DIR / "fonts")))

LOG_LEVEL = _str("LOG_LEVEL", "INFO")


def ai_enabled() -> bool:
    return bool(GEMINI_API_KEY)
