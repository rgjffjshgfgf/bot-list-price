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
    group: str = ""              # the list's group (section) the row belongs to, e.g. «گروه پژو 405»


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
    brain: dict = field(default_factory=dict)   # page -> how the bot's own AI handled it
    # what only the engine process can read (price payloads, the brain's page documents...), packed by
    # pricebot.service so the bot process keeps the rest as plain data
    heavy: bytes = b""

    def item(self, item_id: str) -> PriceItem | None:
        return next((it for it in self.items if it.id == item_id), None)

    def groups(self) -> list[tuple[str, int]]:
        """The list's groups in order, with how many prices each holds."""
        out: dict[str, int] = {}
        for it in self.items:
            if it.group:
                out[it.group] = out.get(it.group, 0) + 1
        return list(out.items())


@dataclass
class ExportJob:
    """The output files of one price change. Built in the engine process; `state` carries the finished work
    (the edited document) from one format to the next."""
    entries: list                     # [(Analysis, {item id: new value})] until the first build
    out_dir: Path
    label: str
    summary: str
    warnings: list[str] = field(default_factory=list)
    state: bytes = b""
