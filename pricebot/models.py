from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .numfmt import NumberFormat


@dataclass
class PriceItem:
    id: str                      # stable id, e.g. "p1-7"
    page: int                    # 0-based page index (0 for single images)
    kind: str                    # "text" (PDF text layer) | "raster" (pixels)
    value: Decimal
    text: str                    # exactly as displayed
    fmt: NumberFormat
    bbox: tuple[float, float, float, float]  # text: PDF points, raster: page-image pixels
    label: str = ""              # row / product description
    column: str = ""             # column header
    payload: Any = None          # TextToken or raster.Target


@dataclass
class PageInfo:
    index: int
    mode: str                    # "text" | "raster" | "mixed" | "empty"
    raster_path: Path | None = None
    zoom: float = 1.0            # raster pixels per PDF point
    width: int = 0
    height: int = 0


@dataclass
class Analysis:
    source: Path
    filename: str
    kind: str                    # "pdf" | "image"
    workdir: Path
    items: list[PriceItem] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)
    currency: str = ""
    warnings: list[str] = field(default_factory=list)
    used_ai: bool = False
    image_format: str | None = None   # PIL format name for images
    page_count: int = 1
    cache: dict = field(default_factory=dict)   # e.g. transcribed tables for Excel

    def item(self, item_id: str) -> PriceItem | None:
        return next((it for it in self.items if it.id == item_id), None)
