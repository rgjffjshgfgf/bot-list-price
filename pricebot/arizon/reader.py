"""The whole price list, page by page, as content for the Arizon template.

Every page takes the most exact road that works for it:
1. text PDFs: the PDF's own text layer (exact words, no AI, milliseconds);
2. photos, scans and pages whose text layer is gibberish: Gemini transcribes
   the table (the prices are referenced by id, so the new values are exact);
3. anything that still cannot be rebuilt reliably: the page itself, with the
   new prices already written on it, is placed in the Arizon frame.
Nothing is ever dropped: a price that would not appear in the rebuilt table
sends its page down to the next road.
"""
from __future__ import annotations

import concurrent.futures
import io
import logging
from pathlib import Path
from typing import Callable

import pymupdf
from PIL import Image

from .. import ai, config
from ..models import Analysis
from . import extract
from .model import Cell, Content, Frame, Row

log = logging.getLogger(__name__)

MAX_UNREADABLE = 0.25     # share of gibberish text cells above which a page is not rebuilt from its text
FRAME_DPI = 150


def _good(r: extract.PageRead | None) -> bool:
    """The page's own text makes a complete, readable table. With Gemini at hand
    even a few unreadable cells send the page to it; without, they are shown as
    pictures of the original text."""
    if r is None or r.total == 0 or r.placed + r.in_title != r.total:
        return False
    return r.unreadable == 0 or (not ai.enabled() and r.unreadable <= MAX_UNREADABLE)


def _ai_page(analysis: Analysis, page: int) -> list[extract.RawTable] | None:
    """Gemini's transcription of the page, when every price found its cell."""
    from ..excel import _ai_page as transcribe   # heavy import, only when needed

    try:
        tables = transcribe(analysis, page)
    except Exception as exc:  # noqa: BLE001 - quota, network: the page falls back to a frame
        log.warning("arizon: AI transcription failed on page %d: %s", page + 1, exc)
        return None
    wanted = {it.id for it in analysis.items if it.page == page}
    placed = {c.price_id for t in tables for r in t.rows for c in r if c.price_id}
    if not tables or not wanted <= placed:
        log.info("arizon: AI table of page %d misses %d prices", page + 1, len(wanted - placed))
        return None
    raws = []
    for t in tables:
        n = max([len(t.headers)] + [len(r) for r in t.rows])
        rows = []
        for r in t.rows:
            filled = [c for c in r if c.text.strip() or c.price_id]
            if len(filled) == 1 and not filled[0].price_id and n > 2:
                rows.append(Row([Cell(extract.clean(filled[0].text))], "group"))
                continue
            cells = [Cell(extract.clean(c.text), [c.price_id] if c.price_id else []) for c in r]
            rows.append(Row(cells + [Cell() for _ in range(n - len(cells))], "item"))
        headers = [extract.clean(h) for h in t.headers] + [""] * (n - len(t.headers))
        raws.append(extract.RawTable(headers, rows, t.direction != "ltr"))
    return raws


def frame(analysis: Analysis, native: Path, page: int) -> Frame:
    """The page with its new prices, as a picture."""
    if analysis.kind == "pdf":
        doc = pymupdf.open(native)
        try:
            pix = doc[page].get_pixmap(dpi=FRAME_DPI, alpha=False)
            data = pix.tobytes("jpeg", jpg_quality=90)
            return Frame(data, pix.width, pix.height, page)
        finally:
            doc.close()
    with Image.open(native) as im:
        rgb = im.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=92)
        return Frame(buf.getvalue(), rgb.width, rgb.height, page)


def read(analysis: Analysis) -> Content:
    """Content of the whole list. Pages kept as pictures are placeholders
    (Frame without picture): their picture depends on the new prices and is
    filled in by fill_frames()."""
    pages = sorted({it.page for it in analysis.items})
    got: dict[int, object] = {}
    how: dict[int, str] = {}
    need_ai: list[int] = []
    second: dict[int, extract.PageRead] = {}    # usable (with pictures for unreadable cells) if Gemini fails
    for p in pages:
        info = analysis.pages[p] if p < len(analysis.pages) else None
        if analysis.kind == "pdf" and info is not None and info.mode in ("text", "mixed") \
                and all(it.kind == "text" for it in analysis.items if it.page == p):
            try:
                r = extract.read_pdf_page(analysis, p)
            except Exception:  # noqa: BLE001 - one odd page must not sink the file
                log.exception("arizon: reading page %d failed", p + 1)
                r = None
            if _good(r):
                got[p], how[p] = r, "pdf"
                continue
            if r is not None and r.total and r.placed + r.in_title == r.total and r.unreadable <= MAX_UNREADABLE:
                second[p] = r
        need_ai.append(p)
    if need_ai and ai.enabled():
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.AI_PARALLEL_PAGES) as pool:
            for p, raws in zip(need_ai, pool.map(lambda q: _ai_page(analysis, q), need_ai)):
                if raws:
                    got[p], how[p] = raws, "ai"
    for p in pages:
        if p in got:
            continue
        if p in second:
            got[p], how[p] = second[p], "pdf"
        else:
            got[p], how[p] = Frame(b"", 0, 0, p), "frame"
    content = extract.assemble([(p, got[p]) for p in pages])
    content.how = how
    log.info("arizon: %s", {k: sum(1 for v in how.values() if v == k) for k in ("pdf", "ai", "frame")})
    return content


def fill_frames(content: Content, analysis: Analysis, native: Callable[[], Path]) -> Content:
    """A copy of the content whose page pictures show the current new prices."""
    if not any(isinstance(b, Frame) for b in content.blocks):
        return content
    path = native()
    blocks = [frame(analysis, path, b.page) if isinstance(b, Frame) else b for b in content.blocks]
    return Content(blocks, content.subject, content.notes, content.titles, content.currency, content.how)
