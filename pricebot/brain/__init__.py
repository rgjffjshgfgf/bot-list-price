"""The bot's own AI.

Gemini is the teacher: while a list format is new, Gemini finds the prices and
the bot learns from its answer (and from the user's 👍 / 👎). Once the bot's own
answer has matched the checked answer enough times in a row, it handles that
format alone, with no Gemini request at all. After enough formats, the price
model also handles formats it has never seen.
"""
from __future__ import annotations

import time
from pathlib import Path

from .. import config
from . import groups, layout, ocr
from .layout import PageDoc
from .learner import Brain, Decision

BRAIN = Brain(config.BRAIN_DIR, config.BRAIN_TRUST_AFTER, config.BRAIN_GENERAL_AFTER)
# Lessons taught and checked by hand, shipped with the code (see tools/teach.py).
SEED_DIR = Path(__file__).resolve().parent / "seed"
if config.BRAIN_MODE != "off":
    BRAIN.merge_seed(SEED_DIR)

__all__ = ["BRAIN", "Decision", "PageDoc", "enabled", "mode", "groups", "layout", "ocr", "column_names",
           "report", "page_note", "truth_from_boxes"]


def enabled() -> bool:
    return config.BRAIN_MODE != "off"


def mode() -> str:
    return config.BRAIN_MODE if config.BRAIN_MODE in ("auto", "teacher", "local") else "auto"


def _inside(t, boxes) -> bool:
    return any(layout.center_in(t.box, b, 0.2 * max(1.0, t.box[3] - t.box[1])) for b in boxes)


def _hidden_header(doc: PageDoc, c: int) -> list:
    """The header of column c among all words, those the file does not name included:
    the nearest line of words right above the column's first number."""
    x0, x1 = doc.col_span(c)
    top = min(doc.nums[i].box[1] for i in doc.columns[c])
    pad = 0.4 * doc.line_h
    above = [w for w in doc.words + doc.hidden
             if w.box[2] >= x0 - pad and w.box[0] <= x1 + pad and w.box[3] <= top + 0.3 * doc.line_h
             and top - w.box[1] <= 4 * doc.line_h]
    if not above:
        return []
    first = max(w.box[3] for w in above)
    return [w for w in above if first - w.box[3] <= 1.6 * doc.line_h][:8]


def header_text(doc: PageDoc, c: int, skip: list = (), word_text=None) -> str:
    """The name of column c; `skip`: boxes of group titles (not part of any header)."""
    words = doc.header(c)
    if word_text is not None and doc.hidden:
        words = _hidden_header(doc, c) or words
    words = [w for w in words if not _inside(w, skip)]
    rtl = doc.rtl()
    words = sorted(words, key=lambda w: (round(w.box[1] / max(1.0, doc.line_h)), -w.xc if rtl else w.xc))
    return " ".join((word_text(w) if word_text else w.text) for w in words)


def column_names(doc: PageDoc, skip: list = (), word_text=None) -> dict[int, str]:
    return {c + 1: header_text(doc, c, skip, word_text) for c in range(len(doc.columns))}


def truth_from_boxes(doc: PageDoc, boxes: list[tuple[float, float, float, float]],
                     rough: list[tuple[float, float, float, float]] = ()) -> set[int]:
    """Numbers of `doc` that sit on one of the checked price boxes (`rough`:
    approximate boxes, only matched when their centre falls on the number)."""
    out = set()
    for i, n in enumerate(doc.nums):
        pad = 0.3 * max(1.0, n.h)
        if any(layout.box_iou(n.box, b) >= 0.3 or layout.center_in(n.box, b, pad) or layout.center_in(b, n.box, pad)
               for b in boxes) or any(layout.center_in(b, n.box, pad) for b in rough):
            out.add(i)
    return out


# ============================================================== reporting ==

def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def page_note(info: dict) -> str:
    """One line about how the brain handled a file (for the analysis message)."""
    how = info.get("how")
    tpl = info.get("format") or {}
    name = tpl.get("name") or "بدون عنوان"
    need = BRAIN.trust_after
    streak = min(tpl.get("streak", 0), need)
    if how == "local":
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


def report() -> str:
    if not enabled():
        return "🧠 هوش ربات خاموش است (BRAIN_MODE=off)."
    b = BRAIN
    with b.lock:
        tpls = sorted(b.templates, key=lambda t: -t.get("updated", 0))
        stats = b.stats
        n_samples = len(b.samples)
    trusted = [t for t in tpls if t.get("streak", 0) >= b.trust_after]
    lines = ["🧠 هوش خود ربات"]
    mode_names = {"auto": "خودکار (قالب‌های آشنا بدون Gemini، بقیه با کمک Gemini و یادگیری)",
                  "teacher": "همیشه با Gemini، یادگیری در پس‌زمینه",
                  "local": "فقط هوش خود ربات (بدون Gemini)"}
    lines.append(f"حالت: {mode_names[mode()]}")
    lines.append(f"\n📚 قالب‌های یادگرفته: {_fa(len(tpls))} — مستقل: {_fa(len(trusted))}")
    for t in tpls[:8]:
        st = min(t.get("streak", 0), b.trust_after)
        kind = "عکس" if t.get("source") == "ocr" else "PDF"
        if t.get("streak", 0) >= b.trust_after:
            ready = "✅ مستقل" if t.get("source") != "ocr" or b.reads_ready(t) else "⏳ خواندن عکس در حال یادگیری"
        else:
            ready = f"⏳ {_fa(st)} از {_fa(b.trust_after)}"
        lines.append(f"• {t.get('name', '')[:40]} ({kind}) — {ready}")
    if len(tpls) > 8:
        lines.append(f"• … و {_fa(len(tpls) - 8)} قالب دیگر")

    lines.append("\n🎯 مدل تشخیص قیمت")
    for src, label in (("pdf", "PDF"), ("ocr", "عکس")):
        h = stats.get("hist", {}).get(src, [])
        if not h:
            continue
        recent = h[-b.general_after:]
        state = "✅ آماده برای قالب‌های ناآشنا" if b.general_ready(src) else \
            f"⏳ {_fa(sum(recent))} از {_fa(b.general_after)} صفحه درست پشت سر هم لازم است"
        lines.append(f"• {label}: دقت صفحه‌ای اخیر {_fa(round(100 * sum(recent) / len(recent)))}٪ — {state}")
    chk = stats.get("checked", {})
    if chk.get("tokens"):
        lines.append(f"• دقت روی {_fa(chk['tokens'])} عدد بررسی‌شده: "
                     f"{_fa(round(100 * chk.get('tokens_ok', 0) / chk['tokens'], 1))}٪")
    lines.append(f"• نمونه‌های آموزشی: {_fa(n_samples)}")
    with b.lock:
        n_group = len(b.gsamples)
    if n_group:
        taught = stats.get("groups_learned", 0)
        lines.append(f"🗂 تشخیص گروه‌های لیست: {_fa(n_group)} درس"
                     + (f" — {_fa(taught)} بار یادگیری از Gemini" if taught else ""))
    r = stats.get("reads", {})
    if ocr.available():
        if r.get("n"):
            lines.append(f"👁 خواندن عدد از عکس: {_fa(r.get('ok', 0))} درست از {_fa(r['n'])} "
                         f"({_fa(round(100 * r.get('ok', 0) / r['n'], 1))}٪)")
        n_glyphs = len(b.glyphs)
        if n_glyphs:
            g = stats.get("glyph_reads", {})
            acc = (f" — دقت {_fa(round(100 * g.get('ok', 0) / g['n'], 1))}٪ روی {_fa(g['n'])} عدد"
                   if g.get("n") else "")
            state = "✅ مستقل" if b.glyph_trusted() else "⏳ در حال یادگیری"
            lines.append(f"🔠 کتابخانه شکل رقم‌ها: {_fa(n_glyphs)} شکل{acc} ({state})")
    else:
        lines.append("👁 OCR (Tesseract) نصب نیست؛ عکس‌ها فقط با Gemini خوانده می‌شوند.")
    c = stats.get("counts", {})
    if c.get("day") == time.strftime("%Y-%m-%d"):
        lines.append(f"\n📈 امروز: {_fa(c.get('local', 0))} صفحه بدون Gemini، "
                     f"{_fa(c.get('teacher', 0))} صفحه با Gemini (یادگیری)")
    tot = stats.get("totals", {})
    if tot:
        lines.append(f"📈 کل: {_fa(tot.get('local', 0))} صفحه بدون Gemini، {_fa(tot.get('teacher', 0))} صفحه با Gemini")
    fb = stats.get("feedback")
    if fb:
        lines.append(f"👍 {_fa(fb.get('good', 0))}   👎 {_fa(fb.get('bad', 0))}")
    if not config.DATA_PERSISTENT:
        lines.append("\n⚠️ حافظه دائمی وصل نیست: با هر Redeploy در Railway، هرچه ربات یاد گرفته پاک می‌شود. "
                     "یک Volume با مسیر /data به سرویس اضافه کن.")
    return "\n".join(lines)
