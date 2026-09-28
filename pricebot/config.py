"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
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
        return int(os.environ.get(name, default))
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

# --- AI providers -----------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = _str("CLAUDE_MODEL", "claude-opus-5")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = _str("OPENAI_MODEL", "gpt-6-astra")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()     # optional gateway / proxy
OPENAI_IMAGE_DETAIL = _str("OPENAI_IMAGE_DETAIL", "original")       # keeps pixel coordinates exact

# Preference order; the first available one leads, the other verifies / takes over.
AI_PROVIDERS = [p.strip().lower() for p in _str("AI_PROVIDERS", "claude,openai").split(",") if p.strip()]
# When both keys are set: analyse every page with both models and merge the answers.
AI_CONSENSUS = _bool("AI_CONSENSUS", True)

AI_EFFORT_IMAGE = _str("AI_EFFORT_IMAGE", "high")      # photos / scans
AI_EFFORT_TEXT = _str("AI_EFFORT_TEXT", "medium")      # PDFs with a text layer
AI_EFFORT_COMMAND = _str("AI_EFFORT_COMMAND", "medium")
AI_EFFORT_TABLE = _str("AI_EFFORT_TABLE", "medium")    # Excel transcription
# Independent re-read of every price that came from pixels (scans/photos).
AI_VERIFY_IMAGE_PRICES = _bool("AI_VERIFY_IMAGE_PRICES", True)
AI_PARALLEL_PAGES = max(1, _int("AI_PARALLEL_PAGES", 4))

# --- bot ----------------------------------------------------------------------
# Comma separated Telegram numeric user ids allowed to use the bot. Empty = everyone.
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x.isdigit()
}
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 60)
SESSION_TTL_MINUTES = _int("SESSION_TTL_MINUTES", 180)
IMAGE_EXPORT_DPI = _int("IMAGE_EXPORT_DPI", 200)
WORK_DIR = Path(os.environ["WORK_DIR"]) if os.environ.get("WORK_DIR") else None

# Extra font folder (put B Nazanin, IRANSans, ... TTF files here for better matches on images).
FONTS_DIR = Path(os.environ.get("FONTS_DIR", str(BASE_DIR / "fonts")))

LOG_LEVEL = _str("LOG_LEVEL", "INFO")


def ai_enabled() -> bool:
    return bool(ANTHROPIC_API_KEY) or bool(OPENAI_API_KEY)
