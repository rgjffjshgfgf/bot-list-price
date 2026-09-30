"""Writing the result: the edited document itself, or it converted to PDF /
images / Excel."""
from __future__ import annotations

import io
import logging
from decimal import Decimal
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from . import config, raster
from .models import Analysis, PriceItem
from .pdftext import FontBank, Replacement, apply_page

log = logging.getLogger(__name__)

FORMATS = ("pdf", "xlsx", "image", "arizon", "arizon_image")
ARIZON_IMAGE_DPI = 170
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def original_format(analysis: Analysis) -> str:
    return "pdf" if analysis.kind == "pdf" else "image"


# ============================================================ edit in place ==

def _edit_raster(rgb: np.ndarray, items: list[tuple[PriceItem, Decimal]], soften: bool) -> list[tuple[int, int, int, int]]:
    targets = [(it.payload, it.fmt.format(v)) for it, v in items]
    if not targets:
        return []
    ink = raster.InkMap(rgb, float(np.median([t.box[3] - t.box[1] for t, _ in targets])))
    # erase everything first, so a neighbour's erase can never clip a freshly drawn price
    erased = [raster.erase(rgb, ink, t) for t, _ in targets]
    rects = []
    for (t, text), er in zip(targets, erased):
        drawn = raster.draw(rgb, t, text, soften=soften)
        if drawn:
            er = (min(er[0], drawn[0]), min(er[1], drawn[1]), max(er[2], drawn[2]), max(er[3], drawn[3]))
        rects.append(er)
    return rects


def _merge_rects(rects: list[tuple[int, int, int, int]], pad: int = 2) -> list[tuple[int, int, int, int]]:
    rs = [(r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad) for r in rects]
    merged = True
    while merged:
        merged = False
        out: list[tuple[int, int, int, int]] = []
        for r in rs:
            for k, o in enumerate(out):
                if r[0] <= o[2] and o[0] <= r[2] and r[1] <= o[3] and o[1] <= r[3]:
                    out[k] = (min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3]))
                    merged = True
                    break
            else:
                out.append(r)
        rs = out
    return rs


def apply_pdf(analysis: Analysis, new_values: dict[str, Decimal], out_path: Path) -> list[str]:
    warnings: list[str] = []
    doc = pymupdf.open(analysis.source)
    bank = FontBank(doc)
    by_page: dict[int, list[PriceItem]] = {}
    for it in analysis.items:
        if it.id in new_values:
            by_page.setdefault(it.page, []).append(it)
    for pno, items in sorted(by_page.items()):
        page = doc[pno]
        reps = [Replacement(it.payload, it.fmt.format(new_values[it.id])) for it in items if it.kind == "text"]
        warnings += apply_page(page, reps, bank)
        rast = [(it, new_values[it.id]) for it in items if it.kind == "raster"]
        if not rast:
            continue
        info = analysis.pages[pno]
        rgb = np.asarray(Image.open(info.raster_path).convert("RGB")).copy()
        rects = _edit_raster(rgb, rast, soften=True)
        H, W = rgb.shape[:2]
        for x0, y0, x1, y1 in _merge_rects(rects):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            buf = io.BytesIO()
            Image.fromarray(rgb[y0:y1, x0:x1]).save(buf, "PNG")
            z = info.zoom
            r = pymupdf.Rect(x0 / z, y0 / z, x1 / z, y1 / z)
            if page.rotation:
                r = r * page.derotation_matrix
            page.insert_image(r, stream=buf.getvalue(), keep_proportion=False, rotate=page.rotation)
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    return warnings


def apply_image(analysis: Analysis, new_values: dict[str, Decimal], out_path: Path) -> list[str]:
    info = analysis.pages[0]
    rgb = np.asarray(Image.open(info.raster_path).convert("RGB")).copy()
    items = [(it, new_values[it.id]) for it in analysis.items if it.id in new_values]
    fmt = image_save_format(analysis)
    _edit_raster(rgb, items, soften=fmt in {"JPEG", "WEBP"})
    out = Image.fromarray(rgb)
    alpha_path = analysis.workdir / "alpha.png"
    if alpha_path.exists() and fmt in {"PNG", "WEBP"}:
        out = out.convert("RGBA")
        out.putalpha(Image.open(alpha_path))
    kwargs: dict = {}
    if fmt == "JPEG":
        kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
    elif fmt == "WEBP":
        kwargs = {"quality": 95, "method": 6}
    elif fmt == "GIF":
        out = out.convert("P", palette=Image.ADAPTIVE)
    out.save(out_path, fmt, **kwargs)
    return []


def image_save_format(analysis: Analysis) -> str:
    fmt = (analysis.image_format or "PNG").upper()
    return fmt if fmt in {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"} else "PNG"


def image_suffix(analysis: Analysis) -> str:
    ext = Path(analysis.filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return ext
    return {"JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif", "GIF": ".gif"}.get(
        image_save_format(analysis), ".png")


# ============================================================== converters ==

def image_to_pdf(image_path: Path, pdf_path: Path) -> None:
    """One page per image, at the image's own resolution (assumes 150 dpi when unknown)."""
    with Image.open(image_path) as im:
        dpi = im.info.get("dpi", (150, 150))[0] or 150
        dpi = dpi if 50 <= dpi <= 1200 else 150
        w, h = im.size
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
            rgb.save(buf, "JPEG", quality=95, subsampling=0)
        else:
            rgb.save(buf, "PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(pdf_path, garbage=3, deflate=True)
    doc.close()


class Exporter:
    """Builds any output format for one plan; the edited document is made once
    and reused by every format."""

    def __init__(self, entries: list[tuple[Analysis, dict[str, Decimal]]], out_dir: Path, label: str,
                 summary: str):
        self.entries = entries
        self.out_dir = out_dir
        self.label = label
        self.summary = summary
        self.warnings: list[str] = []
        self._native: dict[int, Path] = {}
        self._arizon: dict[tuple[int, str], Path] = {}     # (file, line under the title) -> pdf
        self._arizon_dirs: dict[str, Path] = {}             # line under the title -> its folder
        out_dir.mkdir(parents=True, exist_ok=True)

    def _stem(self, analysis: Analysis) -> str:
        return f"{Path(analysis.filename).stem} - {self.label}"

    def native(self, i: int) -> Path:
        """The document in its own format with the new prices."""
        if i not in self._native:
            analysis, values = self.entries[i]
            if analysis.kind == "pdf":
                path = self.out_dir / f"{self._stem(analysis)}.pdf"
                self.warnings += apply_pdf(analysis, values, path)
            else:
                path = self.out_dir / f"{self._stem(analysis)}{image_suffix(analysis)}"
                self.warnings += apply_image(analysis, values, path)
            self._native[i] = path
        return self._native[i]

    def pdf(self) -> list[Path]:
        out = []
        for i, (analysis, _) in enumerate(self.entries):
            src = self.native(i)
            if analysis.kind == "pdf":
                out.append(src)
            else:
                path = self.out_dir / f"{self._stem(analysis)}.pdf"
                image_to_pdf(src, path)
                out.append(path)
        return out

    def images(self) -> list[Path]:
        out = []
        for i, (analysis, _) in enumerate(self.entries):
            src = self.native(i)
            if analysis.kind == "pdf":
                pages_dir = self.out_dir / f"img_{i + 1}"
                pages_dir.mkdir(exist_ok=True)
                out += pdf_to_images(src, pages_dir, self._stem(analysis))
            else:
                out.append(src)
        return out

    def excel(self) -> Path:
        from . import excel   # heavy import, only when asked for

        rows = []
        for analysis, values in self.entries:
            tables = analysis.cache.get("tables")
            if tables is None:
                tables, source = excel.extract_tables(analysis)
                analysis.cache["tables"] = tables
                log.info("excel tables for %s from %s", analysis.filename, source)
            rows.append((analysis, values, tables))
        first = Path(self.entries[0][0].filename).stem if len(self.entries) == 1 else "price-lists"
        path = self.out_dir / f"{first} - {self.label}.xlsx"
        excel.write_workbook(rows, path, self.summary)
        return path

    def arizon(self, as_images: bool = False, subtitle: str = "") -> list[Path]:
        """The list rebuilt in the Arizon template (PDF, or one PNG per page);
        subtitle: the line under «لیست قیمت محصولات», empty for none."""
        from . import arizon   # heavy import, only when asked for

        out = []
        for i, (analysis, values) in enumerate(self.entries):
            # the supplier's file name is not reused: the template carries the Arizon name only
            stem = f"لیست قیمت آریزون {arizon.render.today_fa()[0].replace('/', '-')}"
            if len(self.entries) > 1:
                stem += f" ({i + 1})"
            key = (i, subtitle)
            if key not in self._arizon:
                # every line under the title makes its own documents (same file names, own folder)
                if subtitle not in self._arizon_dirs:
                    self._arizon_dirs[subtitle] = self.out_dir / f"arizon_v{len(self._arizon_dirs) + 1}"
                folder = self._arizon_dirs[subtitle]
                folder.mkdir(exist_ok=True)
                self._arizon[key] = arizon.build(analysis, values, lambda: self.native(i), folder, stem, subtitle)
            pdf = self._arizon[key]
            if as_images:
                pages_dir = pdf.parent / f"pages_{i + 1}"
                pages_dir.mkdir(exist_ok=True)
                out += pdf_to_images(pdf, pages_dir, stem, dpi=ARIZON_IMAGE_DPI)
            else:
                out.append(pdf)
        return out

    def build(self, fmt: str, subtitle: str = "") -> list[Path]:
        """subtitle: the Arizon template's line under its title (other formats ignore it)."""
        if fmt == "arizon":
            return self.arizon(subtitle=subtitle)
        if fmt == "arizon_image":
            return self.arizon(as_images=True, subtitle=subtitle)
        if fmt == "pdf":
            return self.pdf()
        if fmt == "image":
            return self.images()
        if fmt == "xlsx":
            return [self.excel()]
        raise ValueError(fmt)


def pdf_to_images(pdf_path: Path, out_dir: Path, stem: str, dpi: int | None = None) -> list[Path]:
    dpi = dpi or config.IMAGE_EXPORT_DPI
    out = []
    doc = pymupdf.open(pdf_path)
    try:
        width = len(str(doc.page_count))
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            name = f"{stem}.png" if doc.page_count == 1 else f"{stem} - {page.number + 1:0{width}d}.png"
            path = out_dir / name
            pix.save(path)
            out.append(path)
    finally:
        doc.close()
    return out
