"""Everything heavy the bot does. Only the engine process imports this module (see pricebot.engine).

Each job loads only what it needs: a batch of screenshots never loads the brain or PyMuPDF, a price list
loads them on its first page. What goes back to the bot process is plain data: price lists come back
`pack`ed (what only this process can read rides along as `Analysis.heavy`, opaque to the bot) and are
`unpack`ed when they come back for a preview, an output file or a lesson.
"""
from __future__ import annotations

import dataclasses
import io
import logging
import pickle
import time
import zlib
from decimal import Decimal
from pathlib import Path, PurePath

from . import config
from .models import Analysis, ExportJob
from .shot import values as shotv

log = logging.getLogger("service")
PROTOCOL = pickle.HIGHEST_PROTOCOL
BACKUP_DIR = config.DATA_DIR / "brain_backups"
HD_SIDE = 2560      # Telegram keeps photos up to this long side as "HD"; 1280 and below show as SD


def prepare(kind: str) -> None:
    """Load what a job of this kind needs while its file is still downloading."""
    if kind == "price":
        from . import outputs, pipeline  # noqa: F401
    elif kind == "shot":
        from .shot import detect, render  # noqa: F401


# ======================================================== light price lists ==

_PLAIN = (str, int, float, bool, type(None), bytes, Decimal)


class _NotPlain(Exception):
    pass


def _plain(v):
    """`v` as data the bot process can hold without numpy, PyMuPDF or the brain (numpy numbers become
    Python numbers); raises _NotPlain for anything else."""
    t = type(v)
    if t in _PLAIN or isinstance(v, PurePath):
        return v
    if t in (list, tuple, set, frozenset):
        return t(_plain(x) for x in v)
    if t is dict:
        return {(_plain(k) if type(k) not in (str, int) else k): _plain(x) for k, x in v.items()}
    if t.__module__ == "numpy" and getattr(v, "ndim", None) == 0:
        return v.item()
    raise _NotPlain(t)


def _plain_or(v, default):
    try:
        return _plain(v)
    except _NotPlain:
        return default


def pack(a: Analysis) -> Analysis:
    """Move what only this process can read into `a.heavy`; the rest is plain data for the bot process."""
    heavy = {"payload": {}, "brain": {}, "cache": {}}
    for it in a.items:
        if it.payload is not None:
            heavy["payload"][it.id] = it.payload
            it.payload = None
        it.page = int(it.page)
        it.bbox = tuple(float(v) for v in it.bbox)
        if type(it.fmt.decimals) is not int:        # frozen: a plain copy instead of a numpy number
            it.fmt = dataclasses.replace(it.fmt, decimals=int(it.fmt.decimals))
    for p in a.pages:
        p.index, p.width, p.height, p.zoom = int(p.index), int(p.width), int(p.height), float(p.zoom)
    a.page_count = int(a.page_count)
    missing = object()
    brain = {}
    for page, info in a.brain.items():
        light = {}
        for k, v in info.items():
            p = _plain_or(v, missing)
            if p is missing:
                heavy["brain"].setdefault(int(page), {})[k] = v
            else:
                light[k] = p
        brain[int(page)] = light
    a.brain = brain
    cache = {}
    for k, v in a.cache.items():
        p = _plain_or(v, missing)
        if p is not missing:
            cache[k] = p
        elif not callable(v):           # the progress callback of the job is not kept
            heavy["cache"][k] = v
    a.cache = cache
    a.warnings = [str(w) for w in a.warnings]
    a.heavy = zlib.compress(pickle.dumps(heavy, PROTOCOL), 1)
    return a


def unpack(a: Analysis) -> Analysis:
    if not a.heavy:
        return a
    heavy = pickle.loads(zlib.decompress(a.heavy))
    for it in a.items:
        if it.id in heavy["payload"]:
            it.payload = heavy["payload"][it.id]
    for page, extra in heavy["brain"].items():
        a.brain.setdefault(page, {}).update(extra)
    a.cache.update(heavy["cache"])
    a.heavy = b""
    return a


# =============================================================== price lists ==

def analyze(path: Path, filename: str, workdir: Path, progress=None, force_teacher: bool = False,
            stage=None) -> Analysis:
    from . import pipeline
    return pack(pipeline.analyze(path, filename, workdir, progress, force_teacher, stage))


def file_info(path: Path) -> tuple[str | None, int | None]:
    """(kind, pages) of a received file; pages is None when a PDF cannot be opened."""
    from . import pipeline
    kind = pipeline.detect_kind(path)
    if kind != "pdf":
        return kind, 1
    import pymupdf
    try:
        with pymupdf.open(path) as doc:
            return kind, doc.page_count
    except Exception:  # noqa: BLE001 - the analysis step reports broken files
        return kind, None


def preview(a: Analysis, page: int) -> bytes:
    """The page with every detected price boxed in green, as a JPEG."""
    from . import pipeline
    im = pipeline.preview(unpack(a), page)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def export(job: ExportJob, fmt: str, subtitle: str = "") -> tuple[list[Path], ExportJob]:
    """Build one output format; the job comes back with the work done so far (for the next format)."""
    from .outputs import Exporter
    if job.state:
        exporter = pickle.loads(zlib.decompress(job.state))
    else:
        exporter = Exporter([(unpack(a), v) for a, v in job.entries], job.out_dir, job.label, job.summary)
    paths = exporter.build(fmt, subtitle)
    job.state = zlib.compress(pickle.dumps(exporter, PROTOCOL), 1)
    job.entries = []
    job.warnings = list(exporter.warnings)
    return paths, job


def interpret_command(text: str, listing: list[dict], currency: str) -> dict:
    from . import ai
    return ai.interpret_command(text, listing, currency)


def self_test() -> str:
    from . import ai
    return ai.self_test()


def learn_from_user(a: Analysis, good: bool) -> Analysis:
    """👍: every page becomes a checked example. 👎: formats lose their trust."""
    from . import brain
    a = unpack(a)
    ids: set[str] = set()
    wrong_models: list[str] = []
    for info in a.brain.values():
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
    return pack(a)


# ===================================================================== brain ==

def brain_report() -> str:
    from . import brain
    return brain.report()


def status_lines() -> tuple[str, str]:
    """(the brain, OCR) lines of /status."""
    from . import brain
    if not brain.enabled():
        short = "هوش ربات: خاموش"
    else:
        with brain.BRAIN.lock:
            tpls = list(brain.BRAIN.templates)
        ready = sum(1 for t in tpls if t.get("streak", 0) >= brain.BRAIN.trust_after)
        short = f"هوش ربات: {_fa(len(tpls))} قالب یادگرفته، {_fa(ready)} مستقل — جزئیات: /brain"
    if not brain.ocr.available():
        ocr = "OCR (Tesseract): نصب نیست — عکس‌ها فقط با Gemini"
    else:
        langs = sorted(brain.ocr._LANGS or [])
        ocr = (f"OCR (Tesseract): فعال — زبان‌ها: {', '.join(langs)}"
               + ("" if "fas" in langs else " ⚠️ فارسی نصب نیست"))
    return short, ocr


def brain_summary() -> dict:
    from . import brain
    return brain.BRAIN.summary()


def auto_backup(reason: str) -> Path:
    """A safety copy kept on the server before the memory is replaced or wiped (last 5 kept)."""
    from . import brain
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"auto-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.zip"
    brain.BRAIN.export_zip(path)
    for old in sorted(BACKUP_DIR.glob("auto-*.zip"))[:-5]:
        old.unlink(missing_ok=True)
    return path


def backup_export(path: Path) -> dict:
    from . import brain
    return brain.BRAIN.export_zip(path)


def backup_info(path: Path) -> dict:
    """What a received backup holds; raises when the file is not a backup of this bot."""
    from . import brain
    info, _, _ = brain.BRAIN.read_backup(path)
    return info


def backup_restore(path: Path, merge: bool) -> tuple[int, dict, Path]:
    """(formats added or brought back, the memory now, the safety copy taken before)."""
    from . import brain
    safety = auto_backup("before-restore")
    _, data, samples = brain.BRAIN.read_backup(path)
    glyphs = brain.BRAIN.backup_glyphs(path)
    n = brain.BRAIN.restore(data, samples, merge, glyphs)
    return n, brain.BRAIN.summary(), safety


def backup_undo(path: Path) -> dict:
    from . import brain
    _, data, samples = brain.BRAIN.read_backup(path)
    glyphs = brain.BRAIN.backup_glyphs(path)
    brain.BRAIN.restore(data, samples, False, glyphs)
    return brain.BRAIN.summary()


def brain_reset() -> Path:
    from . import brain
    safety = auto_backup("before-reset")
    brain.BRAIN.reset()
    brain.BRAIN.merge_seed(brain.SEED_DIR)
    return safety


def _fa(n) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


# =============================================================== screenshots ==

def _read_image(data: bytes):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _photo_bytes(img) -> bytes:
    """The result as a normal Telegram photo, always at HD size.

    A forwarded photo is usually SD (1280 px), so the finished picture is enlarged as a whole: the new numbers
    were drawn at the photo's own sharpness and grow with everything around them."""
    import cv2
    h, w = img.shape[:2]
    f = HD_SIDE / max(h, w)
    if abs(f - 1) > 1e-3:
        img = cv2.resize(img, (round(w * f), round(h * f)),
                         interpolation=cv2.INTER_LANCZOS4 if f > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _build(an, values: dict[int, str]) -> tuple[bytes, list[str]]:
    from .shot import render
    result = render.apply_views(an, values)
    return _photo_bytes(result.img), list(result.notes)


def shot_process(data: bytes, caption: str) -> shotv.ShotOut:
    """Read a screenshot. Numbers in the caption: the finished photo comes back. None: a boxed preview, and
    what is needed to build it once the numbers are known (the photo itself, not its decoded pixels)."""
    import cv2
    from .shot import detect, render
    an = detect.analyze(_read_image(data))
    out = shotv.ShotOut([f.views_text if f.readable else "" for f in an.footers], list(an.warnings))
    raw, out.question = shotv.read_values(caption, len(an.footers), strict=True)
    if raw is not None:
        out.photo, out.notes = _build(an, {i: shotv.parse_value(v) for i, v in sorted(raw.items())})
        return out
    ok, buf = cv2.imencode(".jpg", render.preview(an), [cv2.IMWRITE_JPEG_QUALITY, 85])
    out.preview = buf.tobytes() if ok else None
    an.img = None
    out.blob = zlib.compress(pickle.dumps((data, an), PROTOCOL), 1)
    return out


def shot_build(blob: bytes, values: dict[int, str]) -> tuple[bytes, list[str]]:
    """Build a screenshot read earlier by `shot_process`, once its numbers (as Telegram shows them) are known."""
    data, an = pickle.loads(zlib.decompress(blob))
    an.img = _read_image(data)
    return _build(an, values)
