"""Arizon template: a price list rebuilt as a branded Arizon document.

    build(analysis, values, native, out_dir, stem) -> [pdf]

The list is read into tables (see reader.py), the prices are replaced by the
new values and everything is drawn in the Arizon design (see render.py).
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Callable

from ..models import Analysis
from . import reader, render
from .model import Content, Frame

log = logging.getLogger(__name__)


def content_of(analysis: Analysis) -> Content:
    """Read once per file (Gemini is asked at most once); kept in the analysis
    for later exports with other prices."""
    content = analysis.cache.get("arizon")
    if content is None:
        t = time.time()
        content = reader.read(analysis)
        analysis.cache["arizon"] = content
        log.info("arizon: %s read in %.1fs", analysis.filename, time.time() - t)
    return content


def build(analysis: Analysis, values: dict[str, Decimal], native: Callable[[], Path], out_dir: Path,
          stem: str) -> Path:
    """native() gives the list with its new prices applied (made only when a
    page has to be shown as a picture)."""
    content = reader.fill_frames(content_of(analysis), analysis, native)
    framed = {b.page for b in content.blocks if isinstance(b, Frame)}
    count = content.item_count + sum(1 for it in analysis.items if it.page in framed)
    meta = render.Meta(currency=content.currency, item_count=count)
    path = out_dir / f"{stem}.pdf"
    t = time.time()
    pages = render.render(content, {it.id: it for it in analysis.items}, values, path, meta)
    log.info("arizon: %s drawn, %d pages in %.1fs", path.name, pages, time.time() - t)
    return path
