"""analyze(file) -> Analysis: every price in the file, located exactly.

Text-layer PDFs: numbers come from the PDF itself; the bot's own AI (brain)
decides which of them are prices when it knows the format, otherwise Gemini
does and the brain learns from the answer.
Photos / scans: read from pixels (by the brain's OCR once proven, otherwise by
Gemini); the rough boxes are then snapped to the real ink, and every read is
double-checked.
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import ai, brain, config, fonts, raster
from .brain import layout as blayout
from .brain import ocr as bocr
from .models import Analysis, PageInfo, PriceItem
from .numfmt import looks_like_price, parse_number, to_latin_digits
from .pdftext import TextToken, compute_layout, extract_tokens, page_chars

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]
AI_TEXT_EDGE = 2000


class UserError(Exception):
    """An error whose message (Persian) can be shown to the user as is."""


def detect_kind(path: Path) -> str | None:
    head = path.read_bytes()[:1024]
    if b"%PDF" in head:
        return "pdf"
    try:
        with Image.open(path) as im:
            im.verify()
        return "image"
    except Exception:  # noqa: BLE001 - not an image
        return None


def analyze(path: Path, filename: str, workdir: Path, progress: Progress | None = None,
            force_teacher: bool = False) -> Analysis:
    """force_teacher: ask Gemini even for formats the brain already knows (after a 👎)."""
    kind = detect_kind(path)
    if kind is None:
        raise UserError("این فایل PDF یا عکس نیست. لطفاً فایل PDF یا عکس (JPG/PNG/WEBP) بفرست.")
    workdir.mkdir(parents=True, exist_ok=True)
    analysis = Analysis(source=path, filename=filename, kind=kind, workdir=workdir, used_ai=ai.enabled())
    analysis.cache["force_teacher"] = force_teacher
    currencies = _analyze_pdf(analysis, progress) if kind == "pdf" else _analyze_image(analysis, progress)
    currencies = [c for c in currencies if c]
    analysis.currency = max(set(currencies), key=currencies.count) if currencies else ""
    return analysis


# ==================================================================== PDF ==

def _analyze_pdf(analysis: Analysis, progress: Progress | None) -> list[str]:
    try:
        doc = pymupdf.open(analysis.source)
    except Exception as exc:  # noqa: BLE001
        raise UserError("فایل PDF باز نشد (ممکن است خراب باشد).") from exc
    if doc.needs_pass:
        raise UserError("این PDF رمز دارد. لطفاً نسخه بدون رمز را بفرست.")
    n = doc.page_count
    doc.close()
    if n > config.MAX_PDF_PAGES:
        raise UserError(f"این PDF {n} صفحه دارد؛ حداکثر {config.MAX_PDF_PAGES} صفحه پشتیبانی می‌شود.")
    analysis.page_count = n
    results: list = [None] * n
    done = 0
    workers = config.AI_PARALLEL_PAGES if ai.enabled() or brain.enabled() else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_analyze_pdf_page, analysis, i): i for i in range(n)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad page must not sink the file
                log.exception("page %d failed", i + 1)
                results[i] = (PageInfo(i, "empty"), [], [f"صفحه {i + 1} بررسی نشد: {exc}"], "")
            done += 1
            if progress:
                progress(done, n)
    currencies = []
    for info, items, warns, currency in results:
        analysis.pages.append(info)
        analysis.items.extend(items)
        analysis.warnings.extend(warns)
        currencies.append(currency)
    return currencies


def _raster_zoom(page: pymupdf.Page) -> float:
    """Render scale matching the resolution of the page's main picture."""
    best, best_area = None, 0.0
    for info in page.get_image_info():
        full = pymupdf.Rect(info["bbox"])
        bb = full & page.rect
        area = bb.width * bb.height
        if area > best_area and info.get("width") and info.get("height") and full.width and full.height:
            best_area = area
            best = math.sqrt(info["width"] * info["height"] / abs(full.width * full.height))
    return 3.0 if best is None else min(max(best, 2.0), 5.0)


def _image_coverage(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height or 1.0
    total = 0.0
    for info in page.get_image_info():
        bb = pymupdf.Rect(info["bbox"]) & page.rect
        total += bb.width * bb.height
    return total / area


def render_page(page: pymupdf.Page, zoom: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def _px_box(r: pymupdf.Rect) -> tuple[int, int, int, int]:
    return int(r.x0), int(r.y0), int(math.ceil(r.x1)), int(math.ceil(r.y1))


def _brain_mode(analysis: Analysis) -> str:
    return "teacher" if analysis.cache.get("force_teacher") else brain.mode()


def _analyze_pdf_page(analysis: Analysis, index: int) -> tuple[PageInfo, list[PriceItem], list[str], str]:
    doc = pymupdf.open(analysis.source)
    try:
        page = doc[index]
        chars = page_chars(page)
        tokens = extract_tokens(page, index)
        cands = [t for t in tokens if not t.attached and len(_digits(t.text)) >= 2]
        coverage = _image_coverage(page)
        warnings: list[str] = []
        info = PageInfo(index, "empty")

        if not cands:
            # a scan / picture page: read it from pixels
            if coverage > 0.02 and (ai.enabled() or (brain.enabled() and bocr.available())):
                zoom_r = _raster_zoom(page)
                rgb = render_page(page, zoom_r)
                items, warns, currency = _raster_page(analysis, rgb, index, 1)
                if items:
                    info = PageInfo(index, "raster", analysis.workdir / f"page_{index + 1}.png", zoom_r,
                                    rgb.shape[1], rgb.shape[0])
                    Image.fromarray(rgb).save(info.raster_path)
                return info, items, warnings + warns, currency
            if coverage > 0.3:
                warnings.append(f"صفحه {index + 1} عکس/اسکن است و برای خواندن آن کلید Gemini یا OCR لازم است.")
            return info, [], warnings, ""

        selected: list[tuple[TextToken, str, int]] = []   # token, label, column
        image_prices: list[dict] = []
        currency = ""
        columns: dict[int, str] = {}
        mode = _brain_mode(analysis)
        bdoc = dec = None
        if brain.enabled():
            bdoc = blayout.from_pdf(page, cands)
            dec = brain.BRAIN.decide(bdoc, mode)
            # prices drawn inside pictures are only seen by the teacher
            if dec.trusted and mode != "local" and coverage > 0.15 and (dec.template or {}).get("img", True):
                dec.trusted = False

        data = None
        if ai.enabled() and not (dec is not None and dec.trusted):
            zoom_ai = AI_TEXT_EDGE / max(page.rect.width, page.rect.height)
            ai_img, s = ai.fit_for_ai(Image.fromarray(render_page(page, zoom_ai)))
            zoom_ai *= s
            to_px = page.rotation_matrix * pymupdf.Matrix(zoom_ai, zoom_ai)
            boxes = [_px_box(t.bbox * to_px) for t in cands]
            listing = [(k, t.text, boxes[k]) for k, t in enumerate(cands)]
            try:
                data = _merge_page(ai.analyze_page_all(ai_img, listing, False), cands, boxes)
            except Exception as exc:  # noqa: BLE001
                log.warning("AI failed on page %d: %s", index + 1, exc)
                who = "هوش خود ربات" if dec is not None else "تشخیص ساده"
                if isinstance(exc, ai.QuotaExceeded):
                    warnings.append(f"صفحه {index + 1}: سهمیه رایگان Gemini تمام شده بود؛ با {who} بررسی شد.")
                else:
                    warnings.append(f"صفحه {index + 1}: Gemini در دسترس نبود؛ با {who} بررسی شد.")
                data = None
            if data is not None:
                currency = data.get("currency", "") or ""
                columns = {c["column_id"]: c["header"] for c in data.get("columns", [])}
                for p in data.get("text_prices", []):
                    k = p.get("id")
                    if isinstance(k, int) and 0 <= k < len(cands):
                        selected.append((cands[k], p.get("label", ""), int(p.get("column_id", 0))))
                for p in data.get("image_prices", []):
                    image_prices.append({**p, "bbox": [v / zoom_ai for v in p["bbox"]]})  # -> points
                if bdoc is not None:
                    truth = {k for k, t in enumerate(cands) if any(t is s[0] for s in selected)}
                    outcome = brain.BRAIN.learn(bdoc, truth, dec, "gemini", {"img": bool(image_prices)})
                    brain.BRAIN.count("teacher")
                    analysis.brain[index] = {"how": "teacher", "doc": bdoc, "decision": dec, "taught": True,
                                             "selected": truth, "format": outcome.get("format"),
                                             "new_format": outcome.get("new_format", False),
                                             "guess_ok": outcome.get("model_ok")}
        if data is None:
            if dec is not None:
                for i in sorted(dec.selected):
                    selected.append((cands[i], bdoc.label(i), bdoc.col_of[i] + 1))
                columns = brain.column_names(bdoc)
                currency = bdoc.currency()
                how = "local" if dec.trusted else "fallback"
                if how == "local":
                    brain.BRAIN.count("local")
                analysis.brain[index] = {"how": how, "doc": bdoc, "decision": dec, "taught": False,
                                         "selected": set(dec.selected), "format": dec.template, "by": dec.how}
            else:
                selected = [(t, "", 0) for t in _heuristic_select(cands)]

        seen: set[int] = set()
        selected = [s for s in selected if not (id(s[0]) in seen or seen.add(id(s[0])))]
        compute_layout(page, [s[0] for s in selected], chars)
        items: list[PriceItem] = []
        for tok, label, col in sorted(selected, key=lambda s: (s[0].bbox.y0, -s[0].bbox.x1)):
            items.append(PriceItem(
                id=f"p{index + 1}-{len(items) + 1}", page=index, kind="text", value=tok.parsed.value,
                text=tok.text, fmt=tok.parsed.fmt, bbox=tuple(tok.bbox), label=label,
                column=columns.get(col, str(col) if col else ""), payload=tok))
        if items:
            info.mode = "text"

        if image_prices:
            zoom_r = _raster_zoom(page)
            rgb = render_page(page, zoom_r)
            info.mode = "mixed" if items else "raster"
            info.zoom = zoom_r
            info.height, info.width = rgb.shape[:2]
            info.raster_path = analysis.workdir / f"page_{index + 1}.png"
            Image.fromarray(rgb).save(info.raster_path)
            for p in image_prices:
                p["bbox"] = [v * zoom_r for v in p["bbox"]]
            r_items, r_warn, _, _ = _raster_items(rgb, image_prices, index, len(items) + 1, columns)
            items.extend(r_items)
            warnings.extend(r_warn)
        return info, items, warnings, currency
    finally:
        doc.close()


def _digits(s: str) -> str:
    return "".join(ch for ch in to_latin_digits(s) if ch.isdigit())


def _heuristic_select(tokens: list[TextToken]) -> list[TextToken]:
    """No-AI fallback: grouped numbers >= 1000 that sit in a real price column."""
    hits = [t for t in tokens if looks_like_price(t.parsed) and not t.attached]
    if not hits:
        return []
    cols: list[list[TextToken]] = []
    for t in sorted(hits, key=lambda t: t.bbox.x0):
        for col in cols:
            ref = col[0].bbox
            if min(ref.x1, t.bbox.x1) - max(ref.x0, t.bbox.x0) >= 0.3 * min(ref.width, t.bbox.width):
                col.append(t)
                break
        else:
            cols.append([t])
    biggest = max(len(c) for c in cols)
    return [t for c in cols if len(c) >= max(2, 0.25 * biggest) or biggest < 3 for t in c]


# ============================================================ consensus ====

def _xoverlap(a, b) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0


def _merge_page(results: list[tuple[str, dict]], cands: list[TextToken], boxes: list[tuple]) -> dict:
    """Combine the answers of the two providers (preferred one first).

    Text-layer prices: both agree -> kept. Only one picked it -> kept when it is
    formatted like a price and sits in a column the two agree on.
    Pixel prices: paired by position. Same digits from both -> trusted as is;
    anything else goes to an extra independent read and a majority vote.
    """
    def clean_img(d: dict, who: str) -> list[dict]:
        out = []
        for p in d.get("image_prices", []):
            bb = p.get("bbox") or []
            if len(bb) == 4 and bb[2] > bb[0] and bb[3] > bb[1]:
                out.append({**p, "votes": [(who, p.get("text", ""))], "agreed": False})
        return out

    if len(results) == 1:
        who, d = results[0]
        return {**d, "image_prices": clean_img(d, who)}

    (na, a), (nb, b) = results[0], results[1]
    merged = {"currency": a.get("currency") or b.get("currency", ""),
              "columns": a.get("columns") or b.get("columns") or [], "notes": ""}

    def valid(d: dict) -> dict[int, dict]:
        return {p["id"]: p for p in d.get("text_prices", [])
                if isinstance(p.get("id"), int) and 0 <= p["id"] < len(cands)}

    ta, tb = valid(a), valid(b)
    agreed = ta.keys() & tb.keys()
    if agreed:
        chosen = set(agreed)
        ref = [boxes[i] for i in agreed]
        for i in ta.keys() ^ tb.keys():
            if looks_like_price(cands[i].parsed) and any(_xoverlap(boxes[i], r) for r in ref):
                chosen.add(i)
        dropped = (ta.keys() | tb.keys()) - chosen
        if dropped:
            log.info("consensus dropped text ids %s", sorted(dropped))
    else:
        chosen = set(ta if len(ta) >= len(tb) else tb)
    merged["text_prices"] = [ta.get(i) or tb.get(i) for i in sorted(chosen)]

    ia, ib = clean_img(a, na), clean_img(b, nb)
    used_b: set[int] = set()
    out: list[dict] = []
    unmatched_a = []
    for pa in ia:
        ba = pa["bbox"]
        ha = ba[3] - ba[1]
        best, best_d = None, None
        for j, pb in enumerate(ib):
            if j in used_b:
                continue
            bb = pb["bbox"]
            dy = abs((ba[1] + ba[3]) / 2 - (bb[1] + bb[3]) / 2)
            if dy <= 0.6 * max(ha, bb[3] - bb[1]) and _xoverlap(ba, bb) and (best_d is None or dy < best_d):
                best, best_d = j, dy
        if best is None:
            unmatched_a.append(pa)
            continue
        used_b.add(best)
        pb = ib[best]
        same = bool(_digits(pa["text"])) and _digits(pa["text"]) == _digits(pb["text"])
        out.append({**pa, "votes": pa["votes"] + pb["votes"], "agreed": same})
    unmatched = unmatched_a + [pb for j, pb in enumerate(ib) if j not in used_b]
    if out:
        out += [p for p in unmatched if any(_xoverlap(p["bbox"], q["bbox"]) for q in out)]
    else:
        out = ia if len(ia) >= len(ib) else ib
    merged["image_prices"] = out
    return merged


# ============================================================ raster work ==

def _ocr_text_at(bdoc: blayout.PageDoc, box) -> str:
    best, best_iou = "", 0.3
    for n in bdoc.nums:
        iou = blayout.box_iou(n.box, tuple(box))
        if iou > best_iou:
            best, best_iou = n.text, iou
    return best


def _raster_items(rgb: np.ndarray, prices: list[dict], page: int, start: int, columns: dict[int, str],
                  bdoc: blayout.PageDoc | None = None, local_only: bool = False,
                  mode: str = "auto") -> tuple[list[PriceItem], list[str], tuple[int, int], int]:
    """Returns (items, warnings, (local reads checked, local reads right), unresolved).

    With `bdoc` (the page's OCR), every price is also read locally; a local read
    that matches Gemini's saves the Gemini re-read. `local_only`: the prices came
    from the brain, so the local reads are the answer (Gemini only settles the
    ones the local reads disagree on, when available)."""
    warnings: list[str] = []
    approx = [tuple(p["bbox"]) for p in prices]
    texts = [p.get("text", "") for p in prices]
    heights = [b[3] - b[1] for b in approx if b[3] > b[1]]
    stat, unresolved = (0, 0), 0
    if not heights:
        return [], warnings, stat, unresolved
    ink = raster.InkMap(rgb, float(np.median(heights)))
    targets = raster.locate(ink, approx, texts)
    raster.style_columns(ink, targets)

    keep: list[tuple[raster.Target, dict]] = []
    for t, p in zip(targets, prices):
        if t is None:
            warnings.append(f"صفحه {page + 1}: قیمت «{p.get('text', '')}» ({p.get('label', '')}) دقیق پیدا نشد و تغییر نمی‌کند.")
        elif t.style is None:
            warnings.append(f"صفحه {page + 1}: فونت مناسب برای «{t.text}» پیدا نشد.")
        else:
            keep.append((t, p))

    if keep:
        local = None
        if bdoc is not None and bocr.available():
            first = [p.get("ocr") or _ocr_text_at(bdoc, t.box) for t, p in keep]
            try:
                local = bocr.read_targets(ink, [t for t, _ in keep], first)
            except Exception:  # noqa: BLE001 - the local reader is an extra, never a blocker
                log.exception("local reading failed")
        if local_only:
            finals = [r.text for r in local] if local else [None] * len(keep)
            pending = [k for k, f in enumerate(finals) if f is None]
            if pending and ai.enabled() and config.AI_VERIFY_IMAGE_PRICES and mode != "local":
                sub = [[("ocr", first[k])] if first[k] else [] for k in pending]
                for k, r in zip(pending, _verify(rgb, [keep[k][0] for k in pending], sub, [False] * len(pending))):
                    finals[k] = r
            unresolved = sum(1 for f in finals if f is None)
        else:
            votes = [list(p.get("votes") or [("?", p.get("text", ""))]) for _, p in keep]
            agreed = [bool(p.get("agreed")) for _, p in keep]
            if local:
                for k, r in enumerate(local):
                    if not r.text:
                        continue
                    if not agreed[k] and _digits(votes[k][0][1]) == _digits(r.text) and parse_number(votes[k][0][1]):
                        agreed[k] = True        # Gemini and the independent local read agree
                    votes[k].append(("ocr", r.text))
            if config.AI_VERIFY_IMAGE_PRICES and ai.enabled():
                finals = _verify(rgb, [t for t, _ in keep], votes, agreed)
            else:
                finals = [v[0][1] for v in votes]
            if local:
                done = [(r.text, f) for r, f in zip(local, finals) if r.text and f]
                stat = (len(done), sum(1 for a, b in done if _digits(a) == _digits(b)))
        checked = []
        for (t, p), text in zip(keep, finals):
            if text is None:
                warnings.append(f"صفحه {page + 1}: عدد «{t.text}» ({p.get('label', '')}) با اطمینان خوانده نشد و تغییر نمی‌کند.")
                continue
            t.text = text
            checked.append((t, p))
        keep = checked

    items = []
    for t, p in sorted(keep, key=lambda tp: (tp[0].box[1], -tp[0].box[2])):
        parsed = parse_number(t.text)
        if parsed is None:
            warnings.append(f"صفحه {page + 1}: «{t.text}» عدد معتبری نیست.")
            continue
        col = int(p.get("column_id", 0))
        items.append(PriceItem(
            id=f"p{page + 1}-{start + len(items)}", page=page, kind="raster", value=parsed.value, text=t.text,
            fmt=parsed.fmt, bbox=tuple(float(v) for v in t.box), label=p.get("label", ""),
            column=columns.get(col, str(col) if col else ""), payload=t))
    return items, warnings, stat, unresolved


def _label_font(size: int) -> ImageFont.ImageFont:
    for path in fonts.candidate_fonts():
        if fonts.supports(path, "0123456789"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _contact_sheets(rgb: np.ndarray, targets: list[raster.Target], strip_h: int) -> list[tuple[Image.Image, list[int]]]:
    """Crops of every target, stacked with an index tag, split into sheets."""
    H, W = rgb.shape[:2]
    crops = []
    for t in targets:
        x0, y0, x1, y1 = t.box
        m = max(3, int(0.35 * (y1 - y0)))
        crop = Image.fromarray(rgb[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)])
        s = strip_h / crop.height
        crops.append(crop.resize((max(1, round(crop.width * s)), strip_h), Image.LANCZOS))
    per_sheet = max(1, (ai.MAX_IMAGE_EDGE - 20) // (strip_h + 16))
    tag_w = int(strip_h * 1.6)
    font = _label_font(int(strip_h * 0.55))
    sheets = []
    for start in range(0, len(crops), per_sheet):
        chunk = crops[start:start + per_sheet]
        width = min(ai.MAX_IMAGE_EDGE, tag_w + max(c.width for c in chunk) + 30)
        height = 10 + len(chunk) * (strip_h + 16)
        sheet = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(sheet)
        idxs = []
        for k, crop in enumerate(chunk):
            y = 10 + k * (strip_h + 16)
            idx = start + k + 1
            idxs.append(idx)
            d.rectangle([4, y, tag_w - 8, y + strip_h], fill=(200, 30, 30))
            d.text(((tag_w - 4) / 2, y + strip_h / 2), str(idx), fill="white", font=font, anchor="mm")
            if crop.width > width - tag_w - 10:
                crop = crop.resize((width - tag_w - 10, strip_h), Image.LANCZOS)
            sheet.paste(crop, (tag_w, y))
            d.line([0, y + strip_h + 8, width, y + strip_h + 8], fill=(180, 180, 180))
        sheets.append((sheet, idxs))
    return sheets


def _read_all(rgb: np.ndarray, targets: list[raster.Target], strip_h: int, provider: ai.Provider) -> dict[int, str]:
    reads: dict[int, str] = {}
    for sheet, idxs in _contact_sheets(rgb, targets, strip_h):
        try:
            got = ai.verify_reads(sheet, idxs, provider)
        except Exception as exc:  # noqa: BLE001
            log.warning("verification by %s failed: %s", provider.name, exc)
            continue
        reads.update({idx: got[idx] for idx in idxs if idx in got})
    return reads


def _majority(votes: list[str]) -> str | None:
    digits = [_digits(v) for v in votes]
    for v, d in zip(votes, digits):
        if d and digits.count(d) >= 2 and parse_number(v):
            return v
    return None


def _verify(rgb: np.ndarray, targets: list[raster.Target], votes: list[list[tuple[str, str]]],
            agreed: list[bool]) -> list[str | None]:
    """Independent re-reads of each crop until two reads agree.

    A read from a model that has not voted on an item yet is preferred
    (Flash-Lite checks Flash and vice versa); the second round uses bigger crops.
    """
    provs = ai.providers()
    result: list[str | None] = [votes[i][0][1] if agreed[i] else None for i in range(len(targets))]
    pending = [i for i in range(len(targets)) if not agreed[i]]
    for strip_h in (56, 90):
        if not pending or not provs:
            break
        groups: dict[ai.Provider, list[int]] = {}
        for i in pending:
            voted = [name for name, _ in votes[i]]
            fresh = [p for p in provs if p.name not in voted]
            p = fresh[0] if fresh else min(provs, key=lambda q: voted.count(q.name))
            groups.setdefault(p, []).append(i)
        for p, idxs in groups.items():
            reads = _read_all(rgb, [targets[i] for i in idxs], strip_h, p)
            for k, i in enumerate(idxs):
                votes[i].append((p.name, reads.get(k + 1, "")))
        still = []
        for i in pending:
            winner = _majority([t for _, t in votes[i]])
            if winner:
                result[i] = winner
            else:
                still.append(i)
        pending = still
    return result


# ================================================================= images ==

def load_image(path: Path) -> tuple[Image.Image, str, Image.Image | None]:
    im = Image.open(path)
    fmt = im.format or "PNG"
    im = ImageOps.exif_transpose(im)
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=alpha)
        im = bg
    else:
        im = im.convert("RGB")
    return im, fmt, alpha


def _analyze_image(analysis: Analysis, progress: Progress | None) -> list[str]:
    im, fmt, alpha = load_image(analysis.source)
    analysis.image_format = fmt
    analysis.page_count = 1
    rgb = np.asarray(im).copy()
    info = PageInfo(0, "raster", analysis.workdir / "image.png", 1.0, im.width, im.height)
    im.save(info.raster_path)
    if alpha is not None:
        alpha.save(analysis.workdir / "alpha.png")
    analysis.pages.append(info)
    if not ai.enabled() and not (brain.enabled() and bocr.available()):
        raise UserError("برای خواندن قیمت از روی عکس، کلید Gemini (GEMINI_API_KEY) یا OCR لازم است.")
    items, warns, currency = _raster_page(analysis, rgb, 0, 1)
    analysis.items = items
    analysis.warnings.extend(warns)
    if progress:
        progress(1, 1)
    return [currency]


def _local_raster(analysis: Analysis, rgb: np.ndarray, bdoc: blayout.PageDoc, dec: brain.Decision,
                  index: int, start: int, mode: str) -> tuple[list[PriceItem], list[str], int]:
    prices = [{"text": bdoc.nums[i].text, "bbox": list(bdoc.nums[i].box), "label": bdoc.label(i),
               "column_id": bdoc.col_of[i] + 1, "ocr": bdoc.nums[i].text} for i in sorted(dec.selected)]
    items, warns, _, unresolved = _raster_items(rgb, prices, index, start, brain.column_names(bdoc), bdoc,
                                                local_only=True, mode=mode)
    return items, warns, unresolved


def _raster_page(analysis: Analysis, rgb: np.ndarray, index: int, start: int) -> tuple[list[PriceItem], list[str], str]:
    """Prices of one photo / scanned page (pixel coordinates of `rgb`)."""
    mode = _brain_mode(analysis)
    bdoc = dec = None
    if brain.enabled() and bocr.available():
        try:
            bdoc = blayout.from_ocr(bocr.page_words(rgb), rgb.shape[1], rgb.shape[0])
            dec = brain.BRAIN.decide(bdoc, mode)
        except Exception:  # noqa: BLE001 - OCR problems must not stop Gemini
            log.exception("page OCR failed")
            bdoc = dec = None

    def local(how: str, extra_warn: str = "") -> tuple[list[PriceItem], list[str], str]:
        items, warns, _ = _local_raster(analysis, rgb, bdoc, dec, index, start, mode)
        if how == "local":
            brain.BRAIN.count("local")
        analysis.brain[index] = {"how": how, "doc": bdoc, "decision": dec, "taught": False,
                                 "selected": set(dec.selected),
                                 "format": dec.template, "by": dec.how}
        return items, ([extra_warn] if extra_warn else []) + warns, bdoc.currency()

    if dec is not None and dec.selected and (dec.trusted or not ai.enabled()):
        items, warns, unresolved = _local_raster(analysis, rgb, bdoc, dec, index, start, mode)
        if not ai.enabled() or mode == "local" or unresolved <= max(1, 0.1 * len(dec.selected)):
            how = "local" if dec.trusted else "fallback"
            if how == "local":
                brain.BRAIN.count("local")
            analysis.brain[index] = {"how": how, "doc": bdoc, "decision": dec, "taught": False,
                                     "selected": set(dec.selected),
                                     "format": dec.template, "by": dec.how}
            return items, warns, bdoc.currency()
        log.info("page %d: %d local reads unsure, asking the teacher", index + 1, unresolved)

    small, s = ai.fit_for_ai(Image.fromarray(rgb))
    try:
        data = _merge_page(ai.analyze_page_all(small, [], raster_only=True), [], [])
    except Exception as exc:  # noqa: BLE001
        if dec is not None and dec.selected:
            log.warning("AI failed on page %d, using the brain: %s", index + 1, exc)
            quota = isinstance(exc, ai.QuotaExceeded)
            why = "سهمیه Gemini تمام شده بود" if quota else "Gemini در دسترس نبود"
            return local("fallback", f"صفحه {index + 1}: {why}؛ با هوش خود ربات بررسی شد.")
        raise
    columns = {c["column_id"]: c["header"] for c in data.get("columns", [])}
    prices = [{**p, "bbox": [v / s for v in p["bbox"]]} for p in data.get("image_prices", [])]
    items, warns, reads, _ = _raster_items(rgb, prices, index, start, columns, bdoc)
    if bdoc is not None:
        truth = brain.truth_from_boxes(bdoc, [it.bbox for it in items], [tuple(p["bbox"]) for p in prices])
        outcome = brain.BRAIN.learn(bdoc, truth, dec, "gemini", {"reads": reads, "img": False})
        brain.BRAIN.count("teacher")
        analysis.brain[index] = {"how": "teacher", "doc": bdoc, "decision": dec, "taught": True,
                                 "selected": truth, "format": outcome.get("format"),
                                 "new_format": outcome.get("new_format", False),
                                 "guess_ok": outcome.get("model_ok")}
    return items, warns, data.get("currency", "")


# ============================================== page images (preview/excel) ==

@dataclass
class PageView:
    image: Image.Image
    boxes: dict[str, tuple[int, int, int, int]]   # item id -> pixel box in `image`


def page_view(analysis: Analysis, page: int, max_edge: int) -> PageView:
    """The page as an image (long edge <= max_edge) plus every price's box on it."""
    items = [it for it in analysis.items if it.page == page]
    boxes: dict[str, tuple[int, int, int, int]] = {}
    if analysis.kind == "pdf":
        doc = pymupdf.open(analysis.source)
        try:
            pg = doc[page]
            zoom = max_edge / max(pg.rect.width, pg.rect.height)
            im = Image.fromarray(render_page(pg, zoom))
            to_px = pg.rotation_matrix * pymupdf.Matrix(zoom, zoom)
            for it in items:
                if it.kind == "text":
                    boxes[it.id] = _px_box(pymupdf.Rect(it.bbox) * to_px)
                else:
                    s = zoom / analysis.pages[page].zoom
                    boxes[it.id] = tuple(int(round(v * s)) for v in it.bbox)
        finally:
            doc.close()
    else:
        im = Image.open(analysis.pages[0].raster_path).convert("RGB")
        s = min(1.0, max_edge / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        boxes = {it.id: tuple(int(round(v * s)) for v in it.bbox) for it in items}
    return PageView(im, boxes)


def preview(analysis: Analysis, page: int = 0, max_edge: int = 1600) -> Image.Image:
    """The page with every detected price boxed in green."""
    view = page_view(analysis, page, max_edge)
    overlay = Image.new("RGBA", view.image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x0, y0, x1, y1 in view.boxes.values():
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(0, 200, 80, 60), outline=(0, 160, 60, 255), width=2)
    return Image.alpha_composite(view.image.convert("RGBA"), overlay).convert("RGB")
