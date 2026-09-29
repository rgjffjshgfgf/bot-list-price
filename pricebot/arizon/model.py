"""What the Arizon template shows: the price list as tables, independent of
where it was read from (PDF text layer, Gemini, OCR)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cell:
    text: str = ""
    price_ids: list[str] = field(default_factory=list)   # prices shown in this cell (drawn with the new value)
    image: bytes | None = None                           # a product photo (PNG/JPEG)

    def empty(self) -> bool:
        return not (self.text.strip() or self.price_ids or self.image)


@dataclass
class Row:
    cells: list[Cell]
    kind: str = "item"            # "item" | "group" (a group title spanning the table)


@dataclass
class Table:
    headers: list[str]
    rows: list[Row]
    title: str = ""
    rtl: bool = True
    roles: list[str] = field(default_factory=list)   # per column: row|code|name|price|qty|unit|image|text

    @property
    def ncols(self) -> int:
        return max([len(self.headers)] + [len(r.cells) for r in self.rows if r.kind == "item"] + [1])


@dataclass
class Frame:
    """A page kept as a picture (with the new prices already written on it),
    used when its table cannot be rebuilt reliably."""
    png: bytes
    width: int
    height: int
    page: int = 0


@dataclass
class Content:
    blocks: list[Table | Frame] = field(default_factory=list)
    subject: str = ""                    # what the list is about, e.g. «ترموستات‌ها»
    notes: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)   # the original list's own title lines
    currency: str = ""
    how: dict[int, str] = field(default_factory=dict)  # page -> "pdf" | "ai" | "ocr" | "frame"

    @property
    def item_count(self) -> int:
        return sum(1 for b in self.blocks if isinstance(b, Table) for r in b.rows if r.kind == "item")
