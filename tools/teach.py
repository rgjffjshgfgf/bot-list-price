"""Teach the bot's brain by hand (a person or Claude acting as the teacher).

    python tools/teach.py show  FILE              # columns of every page + preview images
    python tools/teach.py learn FILE --cols "*:3,4" [--skip "1405,1404"] [--trust] [--reads-ok]

--cols     price columns per page: "*:3,4" = columns 3 and 4 on every page,
           "1:3;2:2,3" = column 3 on page 1, columns 2 and 3 on page 2.
--skip     numbers inside those columns that are NOT prices (exact text).
--trust    the answer was checked carefully: the format is handled without Gemini at once.
--reads-ok (photos) the OCR reads printed by `show` were checked and are all correct.
--photos   (PDFs) also teach photo-like versions of every page (screenshot, phone photo,
           small image), so a photo or screenshot of the same list is recognised too.
           The answer comes from the PDF; how well OCR reads each version is measured.

Lessons go to pricebot/brain/seed/ (shipped with the code; merged into the
bot's memory on start). Set BRAIN_DIR to write elsewhere.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("BRAIN_DIR", str(ROOT / "pricebot" / "brain" / "seed"))
os.environ["GEMINI_API_KEY"] = ""
os.environ.pop("GOOGLE_API_KEY", None)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pricebot import brain, pipeline  # noqa: E402
from pricebot.brain import layout, ocr  # noqa: E402
from pricebot.pdftext import extract_tokens  # noqa: E402

B = brain.BRAIN


def pages(path: Path):
    """(page number, PageDoc, rgb image, pixels-per-unit) for every page."""
    kind = pipeline.detect_kind(path)
    if kind == "image":
        im, _, _ = pipeline.load_image(path)
        rgb = np.asarray(im).copy()
        yield 1, layout.from_ocr(ocr.glyph_words(rgb, ocr.page_words(rgb), B.glyphs), rgb.shape[1], rgb.shape[0]), rgb, 1.0
        return
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        cands = [t for t in extract_tokens(page, i) if not t.attached and len(pipeline._digits(t.text)) >= 2]
        if cands:
            zoom = 1600 / max(page.rect.width, page.rect.height)
            yield i + 1, layout.from_pdf(page, cands), pipeline.render_page(page, zoom), zoom
        else:
            zoom = pipeline._raster_zoom(page)
            rgb = pipeline.render_page(page, zoom)
            yield i + 1, layout.from_ocr(ocr.page_words(rgb), rgb.shape[1], rgb.shape[0]), rgb, 1.0


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def show(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pno, doc, rgb, z in pages(path):
        dec = B.decide(doc, "auto")
        print(f"\n=== page {pno} ({doc.source}) — {len(doc.nums)} numbers, format match: "
              f"{(dec.template or {}).get('name', '-')} ({dec.similarity:.2f}) ===")
        im = Image.fromarray(rgb).convert("RGB")
        d = ImageDraw.Draw(im)
        f = _font(max(14, int(im.height / 70)))
        for c, idx in enumerate(doc.columns):
            vals = [doc.nums[i].text for i in idx]
            p = np.mean([dec.probs[i] for i in idx]) if dec.probs else 0
            guess = sum(1 for i in idx if i in dec.selected)
            print(f"col {c + 1:>2}: n={len(idx):>3}  guess={guess:>3}  p={p:.2f}  header=«{brain.header_text(doc, c)}»  "
                  f"values={vals[:6]}{' …' if len(vals) > 6 else ''}")
            for i in idx:
                x0, y0, x1, y1 = (v * z for v in doc.nums[i].box)
                d.rectangle([x0, y0, x1, y1], outline=(0, 170, 0) if i in dec.selected else (220, 0, 0), width=2)
            x0, y0 = doc.nums[idx[0]].box[0] * z, doc.nums[idx[0]].box[1] * z
            d.text((x0, max(0, y0 - f.size - 4)), str(c + 1), fill=(0, 0, 255), font=f)
        out = out_dir / f"{path.stem}_p{pno}.png"
        im.save(out)
        print(f"preview: {out}  (green = current guess, blue numbers = column ids)")


def _parse_cols(spec: str) -> dict:
    out = {}
    for part in spec.split(";"):
        page, _, cols = part.partition(":")
        out[page.strip()] = {int(c) for c in cols.split(",") if c.strip()}
    return out


def learn(path: Path, cols: str, skip: str, trust: bool, reads_ok: bool) -> None:
    spec = _parse_cols(cols)
    skip_set = {s.strip() for s in skip.split(",") if s.strip()}
    for pno, doc, _, _ in pages(path):
        want = spec.get(str(pno), spec.get("*", set()))
        truth = {i for c, idx in enumerate(doc.columns) if c + 1 in want for i in idx
                 if doc.nums[i].text not in skip_set}
        bad = [c for c in want if not 1 <= c <= len(doc.columns)]
        if bad or (want and not truth):
            print(f"page {pno}: columns {sorted(want)} do not match this page ({len(doc.columns)} columns) — skipped")
            continue
        dec = B.decide(doc, "auto")
        extra = {"img": False}
        if doc.source == "ocr" and reads_ok:
            extra["reads"] = (len(truth), len(truth))
        outcome = B.learn(doc, truth, dec, "claude", extra)
        tpl = outcome.get("format")
        if trust and tpl is not None:
            tpl["streak"] = max(tpl.get("streak", 0), B.trust_after)
            tpl["taught_by"] = "claude"
            B.save()
        print(f"page {pno}: {len(truth)} prices learned — own guess was "
              f"{'right' if outcome.get('model_ok') else 'wrong'}; format «{(tpl or {}).get('name', '-')}» "
              f"streak {(tpl or {}).get('streak', 0)}")


# (name, extra scale, blur radius, JPEG quality or None)
PHOTO_VARIANTS = [("screenshot", 1.0, 0.0, None), ("phone", 0.8, 0.7, 60), ("small", 0.6, 0.0, 75),
                  ("blurry", 0.7, 1.0, 50), ("large", 1.25, 0.5, 70)]


def learn_photos(path: Path, cols: str, skip: str, trust: bool) -> None:
    """Teach photo-like renderings of a text PDF, labelled from the PDF itself.

    For every page, all versions are first read with what the bot knows so far
    (scored honestly: this page's digits are not in the library yet), then the
    digit shapes of the page are added to the lasting library."""
    import io
    from PIL import ImageFilter
    from pricebot import raster
    spec = _parse_cols(cols)
    skip_set = {s.strip() for s in skip.split(",") if s.strip()}
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        pno = i + 1
        cands = [t for t in extract_tokens(page, i) if not t.attached and len(pipeline._digits(t.text)) >= 2]
        if not cands:
            continue
        pdoc = layout.from_pdf(page, cands)
        want = spec.get(str(pno), spec.get("*", set()))
        truth = [pdoc.nums[k] for c, idx in enumerate(pdoc.columns) if c + 1 in want for k in idx
                 if pdoc.nums[k].text not in skip_set]
        if not truth:
            continue
        z = 2400 / max(page.rect.width, page.rect.height)          # like a 1-2 MP screenshot / photo
        base = Image.fromarray(pipeline.render_page(page, z))
        to_harvest = []
        for name, s, blur, q in PHOTO_VARIANTS:
            im = base.resize((round(base.width * s), round(base.height * s)), Image.LANCZOS) if s != 1 else base
            if blur:
                im = im.filter(ImageFilter.GaussianBlur(blur))
            if q:
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=q)
                im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            rgb = np.asarray(im).copy()
            odoc = layout.from_ocr(ocr.glyph_words(rgb, ocr.page_words(rgb), B.glyphs), rgb.shape[1], rgb.shape[0])
            f = z * s
            boxes = [tuple(v * f for v in t.box) for t in truth]
            texts = [t.text for t in truth]
            found = brain.truth_from_boxes(odoc, boxes)
            ink = raster.InkMap(rgb, float(np.median([b[3] - b[1] for b in boxes])))
            targets = raster.locate(ink, boxes, texts)
            # only numbers found where the PDF has them (a target snapped to another row is not scored)
            pairs = [(t, x) for t, x, b in zip(targets, texts, boxes)
                     if t is not None and layout.center_in(t.box, b, 0.2 * (b[3] - b[1]))]
            first = [pipeline._ocr_text_at(odoc, t.box) for t, _ in pairs]
            reads = ocr.read_targets(ink, [t for t, _ in pairs], first, B.glyphs, True)
            acc = [(r.text, x) for r, (_, x) in zip(reads, pairs) if r.text]
            right = sum(1 for a, b in acc if layout.digits_of(a) == layout.digits_of(b))
            if os.environ.get("TEACH_DEBUG"):
                for r, (_, x) in zip(reads, pairs):
                    if r.text and layout.digits_of(r.text) != layout.digits_of(x):
                        print(f"   WRONG READ: {r.text} (real {x}) via {r.how}, votes {r.votes}")
            sure = [(r.glyph, x) for r, (_, x) in zip(reads, pairs) if r.glyph and r.glyph_sure]
            B.record_glyph_reads(len(sure), sum(1 for a, b in sure if layout.digits_of(a) == layout.digits_of(b)))
            to_harvest.append((ink, pairs))

            dec = B.decide(odoc, "auto")
            outcome = B.learn(odoc, found, dec, "claude", {"img": False, "reads": (len(acc), right)})
            tpl = outcome.get("format")
            if trust and tpl is not None:
                tpl["streak"] = max(tpl.get("streak", 0), B.trust_after)
                tpl["taught_by"] = "claude"
                B.save()
            print(f"page {pno} as {name:10}: prices located {len(found)}/{len(truth)}, "
                  f"read {len(acc)}/{len(pairs)} ({right} right, {len(acc) - right} wrong), "
                  f"shape library {len(B.glyphs)} — guess {'right' if outcome.get('model_ok') else 'wrong'}")
        for ink, pairs in to_harvest:
            ocr.harvest(ink, [t for t, _ in pairs], [x for _, x in pairs], B.glyphs)
        B.save_glyphs()
        B.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["show", "learn", "report"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--cols", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--trust", action="store_true")
    ap.add_argument("--reads-ok", action="store_true")
    ap.add_argument("--photos", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "teach_preview"))
    a = ap.parse_args()
    if a.action == "report":
        print(brain.report())
    elif a.action == "show":
        show(Path(a.file), Path(a.out))
    else:
        learn(Path(a.file), a.cols, a.skip, a.trust, a.reads_ok)
        if a.photos:
            learn_photos(Path(a.file), a.cols, a.skip, a.trust)


if __name__ == "__main__":
    main()
