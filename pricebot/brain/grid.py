"""Photos of lists printed as ruled tables: the format is the grid of lines.

A list photographed (or exported) again next month has the same table lines in
the same places, relative to the whole grid - only the numbers in the price
cells change. The bot remembers such a list as its grid plus the cell of every
price (with the price's group, row label and column). A new photo with the same
grid is the same list: every price is found in its cell, without OCR (which
cannot read many Persian fonts) and without asking Gemini to find prices on the
whole page (small print on a photo is where it errs most); only the digits of
each price still need reading, cut out and enlarged.

Every check is strict: a grid that differs anywhere, a price cell that is empty
or holds more than one number, a cell with a number the format does not know,
or a row whose text changed - and the page goes the ordinary way (Gemini)."""
from __future__ import annotations

import base64
import logging
import math
import statistics
from dataclasses import dataclass

import cv2
import numpy as np

from .. import raster
from ..numfmt import digit_script

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]

LINE_T = 30           # darkness of a ruling line against the paper around it
INK_T = 40            # darkness of text (as raster.DARK_T)
MAX_EDGE = 2400       # bigger photos are looked at scaled down (lines are long and clear)
POS_TOL = 0.006       # where a line is, as a share of the grid's height / width
SPAN_TOL = 0.025      # where a line starts / ends
MATCH = 0.95          # share of lines that must match, both ways
ROW_SAME = 0.8        # a row's text (all but the price) must look this much alike
SIG_W, SIG_H = 96, 8  # size of a row's text picture
MIN_DIGIT_H = 5.0     # prices smaller than this (pixels) lose their zeros to the paper
WORK_DIGIT_H = 14.0   # smaller prices are looked at enlarged to this height


@dataclass
class Seg:
    pos: float        # y of a horizontal line, x of a vertical one (image pixels)
    a: float          # start along the line
    b: float          # end
    w: float          # thickness


@dataclass
class Cell:
    x0: float         # centres of the lines around it
    y0: float
    x1: float
    y1: float
    pad: float        # half the thickest of its lines, +1 px: the inside starts after it
    top: Seg          # the line above it (its span = the width of its table row)

    @property
    def inner(self) -> Box:
        return (self.x0 + self.pad, self.y0 + self.pad, self.x1 - self.pad, self.y1 - self.pad)

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


class Grid:
    """The ruling lines of one page (positions in image pixels)."""

    def __init__(self, width: int, height: int, h: list[Seg], v: list[Seg], dark: np.ndarray,
                 lines: np.ndarray, scale: float):
        self.width, self.height = width, height
        self.h, self.v = h, v
        self._dark = dark             # text darkness and line pixels, at `scale`
        self._lines = lines
        self._scale = scale
        self._fp: dict | None = None
        xs = [s.a for s in h] + [s.b for s in h] + [s.pos for s in v]
        ys = [s.pos for s in h] + [s.a for s in v] + [s.b for s in v]
        self.frame = (min(xs), min(ys), max(xs), max(ys)) if h and v else (0.0, 0.0, 1.0, 1.0)

    @property
    def fw(self) -> float:
        return max(1.0, self.frame[2] - self.frame[0])

    @property
    def fh(self) -> float:
        return max(1.0, self.frame[3] - self.frame[1])

    def ok(self) -> bool:
        """Enough of a table to be a list's fingerprint."""
        return len(self.h) >= 4 and len(self.v) >= 2 and self.fw * self.fh >= 0.1 * self.width * self.height

    def to_norm(self, box: Box) -> list[float]:
        x0, y0 = self.frame[:2]
        return [round((box[0] - x0) / self.fw, 5), round((box[1] - y0) / self.fh, 5),
                round((box[2] - x0) / self.fw, 5), round((box[3] - y0) / self.fh, 5)]

    def to_px(self, nbox: list[float]) -> Box:
        x0, y0 = self.frame[:2]
        return (x0 + nbox[0] * self.fw, y0 + nbox[1] * self.fh, x0 + nbox[2] * self.fw, y0 + nbox[3] * self.fh)

    # ---- cells ----
    def cell_at(self, x: float, y: float) -> Cell | None:
        """The closed cell around a point: the nearest lines above, below, left and right
        that pass by it."""
        top = max((s for s in self.h if s.pos < y and s.a - 2 <= x <= s.b + 2), key=lambda s: s.pos, default=None)
        bot = min((s for s in self.h if s.pos > y and s.a - 2 <= x <= s.b + 2), key=lambda s: s.pos, default=None)
        lef = max((s for s in self.v if s.pos < x and s.a - 2 <= y <= s.b + 2), key=lambda s: s.pos, default=None)
        rig = min((s for s in self.v if s.pos > x and s.a - 2 <= y <= s.b + 2), key=lambda s: s.pos, default=None)
        if None in (top, bot, lef, rig):
            return None
        pad = max(top.w, bot.w, lef.w, rig.w) / 2 + 1
        return Cell(lef.pos, top.pos, rig.pos, bot.pos, pad, top)

    def column_runs(self, cell: Cell) -> list[list[Cell]]:
        """The cells of the table column a cell is in (same lines left and right), in
        unbroken runs from top to bottom (a title row across the table ends a run)."""
        xc = cell.xc
        ys = sorted({round(s.pos, 1) for s in self.h if s.a - 2 <= xc <= s.b + 2})
        runs: list[list[Cell]] = []
        prev = None
        for ya, yb in zip(ys, ys[1:]):
            c = self.cell_at(xc, (ya + yb) / 2)
            same = c is not None and abs(c.x0 - cell.x0) <= 3 and abs(c.x1 - cell.x1) <= 3
            if not same:
                prev = None
                continue
            if prev is None or abs(prev.y1 - c.y0) > 3:
                runs.append([])
            runs[-1].append(c)
            prev = c
        return runs

    # ---- what is written in a row ----
    def row_signature(self, cell: Cell) -> np.ndarray:
        """A small picture of the text of the cell's table row, the cell itself left out
        (the price changes, the product names do not)."""
        s = self._scale
        x0, x1 = cell.top.a, cell.top.b
        y0, y1 = cell.y0 + cell.pad, cell.y1 - cell.pad
        band = (self._dark[int(y0 * s):max(int(y0 * s) + 1, int(y1 * s)),
                           int(x0 * s):max(int(x0 * s) + 1, int(x1 * s))] > INK_T).astype(np.float32)
        lines = self._lines[int(y0 * s):max(int(y0 * s) + 1, int(y1 * s)),
                            int(x0 * s):max(int(x0 * s) + 1, int(x1 * s))]
        band[lines > 0] = 0
        cx0, cx1 = int((cell.x0 - x0 - cell.pad) * s), int((cell.x1 - x0 + cell.pad) * s) + 1
        band[:, max(0, cx0):max(0, cx1)] = 0
        if not band.size:
            return np.zeros(SIG_W * SIG_H, np.float32)
        v = cv2.resize(band, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).flatten()
        n = float(np.linalg.norm(v))
        return v / n if n else v


def row_likeness(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two rows' text pictures are, compared coarsely (a small or soft photo
    draws the same words a little differently)."""
    def coarse(v: np.ndarray) -> np.ndarray:
        c = v.reshape(SIG_H, SIG_W).reshape(SIG_H // 2, 2, SIG_W // 4, 4).mean(axis=(1, 3)).flatten()
        c = cv2.GaussianBlur(c.reshape(SIG_H // 2, SIG_W // 4).astype(np.float32), (3, 1), 0).flatten()
        n = float(np.linalg.norm(c))
        return c / n if n else c
    return float(np.dot(coarse(a), coarse(b)))


def _merge(segs: list[Seg], pos_tol: float, gap: float) -> list[Seg]:
    """Pieces of one line (broken by noise) joined."""
    out: list[Seg] = []
    for s in sorted(segs, key=lambda s: (round(s.pos / max(1.0, pos_tol)), s.a)):
        for o in out:
            if abs(o.pos - s.pos) <= pos_tol and s.a <= o.b + gap and s.b >= o.a - gap:
                total = (o.b - o.a) + (s.b - s.a)
                o.pos = (o.pos * (o.b - o.a) + s.pos * (s.b - s.a)) / max(1.0, total)
                o.a, o.b, o.w = min(o.a, s.a), max(o.b, s.b), max(o.w, s.w)
                break
        else:
            out.append(Seg(s.pos, s.a, s.b, s.w))
    return sorted(out, key=lambda s: (s.pos, s.a))


def find(rgb: np.ndarray) -> Grid:
    """The table lines of a page: long thin straight runs of dark pixels."""
    H, W = rgb.shape[:2]
    s = min(1.0, MAX_EDGE / max(H, W))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sh, sw = gray.shape
    k = min(max(int(min(sh, sw) * 0.03) | 1, 15), 255)
    dark = cv2.subtract(cv2.medianBlur(gray, k), gray)
    src = (dark > LINE_T).astype(np.uint8) * 255
    found: dict[str, list[Seg]] = {}
    masks = []
    for orient, size in (("h", (max(25, int(0.05 * sw)), 1)), ("v", (1, max(25, int(0.035 * sh))))):
        m = cv2.morphologyEx(src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        masks.append(m)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        segs = []
        for i in range(1, n):
            x, y, bw, bh, _ = (int(v) for v in st[i])
            if orient == "h":
                segs.append(Seg((y + bh / 2) / s, x / s, (x + bw) / s, max(1.0, bh - 2) / s))
            else:
                segs.append(Seg((x + bw / 2) / s, y / s, (y + bh) / s, max(1.0, bw - 2) / s))
        along = W if orient == "h" else H
        found[orient] = _merge(segs, max(2.0, 0.003 * along), 0.004 * along)
    lines = cv2.dilate(cv2.bitwise_or(*masks), np.ones((3, 3), np.uint8))
    hs, vs = _ruled(found["h"], found["v"], max(4.0, 0.01 * max(W, H)))
    return Grid(W, H, hs, vs, dark, lines, s)


def _ruled(h: list[Seg], v: list[Seg], tol: float) -> tuple[list[Seg], list[Seg]]:
    """Only lines that are part of a table: each one meets at least two lines across
    it. A line of text smeared by blur, or an underline, meets none - and would
    move the grid's frame."""
    def meets(a: Seg, b: Seg) -> bool:
        return a.a - tol <= b.pos <= a.b + tol and b.a - tol <= a.pos <= b.b + tol
    for _ in range(3):
        h2 = [s for s in h if sum(1 for o in v if meets(s, o)) >= 2]
        v2 = [s for s in v if sum(1 for o in h2 if meets(s, o)) >= 2]
        if len(h2) == len(h) and len(v2) == len(v):
            break
        h, v = h2, v2
    return h, v


# ============================================================ fingerprint ==

def fingerprint(grid: Grid) -> dict:
    if grid._fp is not None:
        return grid._fp
    x0, y0 = grid.frame[:2]
    grid._fp = {
        "aspect": round(grid.fw / grid.fh, 4),
        "h": [[round((s.pos - y0) / grid.fh, 4), round((s.a - x0) / grid.fw, 4), round((s.b - x0) / grid.fw, 4)]
              for s in grid.h],
        "v": [[round((s.pos - x0) / grid.fw, 4), round((s.a - y0) / grid.fh, 4), round((s.b - y0) / grid.fh, 4)]
              for s in grid.v]}
    return grid._fp


def _pair(old: list[list[float]], new: list[list[float]]) -> int:
    """How many lines of `old` have their own counterpart in `new`."""
    used: set[int] = set()
    hits = 0
    for o in old:
        best, best_d = None, None
        for j, n in enumerate(new):
            if j in used or abs(n[0] - o[0]) > POS_TOL or abs(n[1] - o[1]) > SPAN_TOL or abs(n[2] - o[2]) > SPAN_TOL:
                continue
            d = abs(n[0] - o[0]) + 0.2 * (abs(n[1] - o[1]) + abs(n[2] - o[2]))
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def similarity(fp: dict, grid: Grid) -> float:
    """0..1: the share of lines the two grids have in common (the worse way round)."""
    if not grid.ok() or not fp.get("h") or not fp.get("v"):
        return 0.0
    if abs(math.log((grid.fw / grid.fh) / max(1e-6, fp["aspect"]))) > 0.03:
        return 0.0
    new = fingerprint(grid)
    score = 1.0
    for o in ("h", "v"):
        hits = _pair(fp[o], new[o])
        score = min(score, hits / len(fp[o]), hits / len(new[o]))
    return score


# ================================================================ prices ==

def _encode(v: np.ndarray) -> str:
    return base64.b64encode((np.clip(v * 255 / (float(v.max()) or 1.0), 0, 255)).astype(np.uint8).tobytes()).decode()


def _decode(s: str) -> np.ndarray:
    v = np.frombuffer(base64.b64decode(s), dtype=np.uint8).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _cell_ink(ink: raster.InkMap, cell: Cell, pol: str) -> tuple[int, int, int, int] | None:
    """The box of all the writing inside a cell. The threshold follows the cell's own
    darkest ink: on a small or soft photo a dot-shaped Persian zero is far fainter
    than the digits, and a fixed threshold loses it (the price would look shorter)."""
    H, W = ink.gray.shape
    x0, y0 = max(0, math.ceil(cell.inner[0])), max(0, math.ceil(cell.inner[1]))
    x1, y1 = min(W, int(cell.inner[2])), min(H, int(cell.inner[3]))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    strength = (ink.dark if pol == "dark" else ink.light)[y0:y1, x0:x1]
    peak = float(strength.max())
    if peak < INK_T:
        return None
    bw = ((strength > max(12.0, 0.15 * peak)) & ~(ink.lines[y0:y1, x0:x1] > 0)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, a = (int(v) for v in st[k])
        if a < 2 or x == 0 or y == 0 or x + w >= x1 - x0 or y + h >= y1 - y0:
            continue                                # speck, or the edge of a ruling line
        boxes.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    if not boxes:
        return None
    # the number is one line of glyphs close together: grow it from its biggest glyph,
    # leaving out specks a soft JPEG puts along the cell's lines
    main = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    h = main[3] - main[1]
    line = [b for b in boxes if main[1] - 0.6 * h <= (b[1] + b[3]) / 2 <= main[3] + 0.6 * h]
    x0, x1 = main[0], main[2]
    grown = True
    while grown:
        grown = False
        for b in line:
            if b[0] < x0 - 1e-6 or b[2] > x1:
                if b[2] >= x0 - 1.6 * h and b[0] <= x1 + 1.6 * h:
                    x0, x1 = min(x0, b[0]), max(x1, b[2])
                    grown = True
    keep = [b for b in line if b[0] >= x0 and b[2] <= x1]
    return (x0, min(b[1] for b in keep), x1, max(b[3] for b in keep))


def _numberlike(box: tuple[int, int, int, int] | None, h: float, cell: Cell) -> bool:
    """Ink that may be a price: a run of glyphs of about the prices' height."""
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return 0.6 * h <= y1 - y0 <= 1.7 * h and 0.8 * h <= x1 - x0 <= cell.x1 - cell.x0


def _same_cell(a: list[float], b: list[float]) -> bool:
    return all(abs(p - q) <= 2 * POS_TOL for p, q in zip(a, b))


def _ink_map(rgb: np.ndarray, price_h: float) -> raster.InkMap:
    return raster.InkMap(rgb, max(6.0, price_h))


def describe(rgb: np.ndarray, grid: Grid, prices: list[dict]) -> dict | None:
    """The format record of a checked page: `prices` are dicts with box (pixels),
    pol, label, group, column (header) and text. None when the page is no clean
    ruled table: a price outside a closed cell, two in one cell, a cell holding
    more than the price, or a number in a price column that is not among the
    prices (the answer missed it)."""
    if not grid.ok() or not prices:
        return None
    hs = [p["box"][3] - p["box"][1] for p in prices]
    price_h = float(statistics.median(hs))
    ink = _ink_map(rgb, price_h)
    cells: list[Cell] = []
    for p in prices:
        b = p["box"]
        c = grid.cell_at((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if c is None:
            return None
        ix0, iy0, ix1, iy1 = c.inner
        if not (b[0] >= ix0 - 2 and b[2] <= ix1 + 2 and b[1] >= iy0 - 2 and b[3] <= iy1 + 2):
            return None                         # the price crosses a line: no cell of its own
        u = _cell_ink(ink, c, p["pol"])
        if u is None:
            return None
        m = 0.5 * price_h
        if u[0] < b[0] - m or u[2] > b[2] + m or u[1] < b[1] - m or u[3] > b[3] + m:
            return None                         # more than the price in its cell (a unit, a note)
        if any(abs(c.xc - o.xc) < 1 and abs(c.yc - o.yc) < 1 for o in cells):
            return None
        cells.append(c)
    if _unknown_numbers(grid, ink, cells, [p["pol"] for p in prices], price_h):
        return None
    # price columns numbered from the right (as Gemini numbers them)
    centres: list[float] = []
    for x in sorted((c.xc for c in cells), reverse=True):
        if not centres or centres[-1] - x > 0.02 * grid.fw:
            centres.append(x)
    # reading order: table by table from the right, row by row, right to left in a row
    order = sorted(range(len(cells)), key=lambda k: (-round(cells[k].top.b / (0.02 * grid.fw)),
                                                      round(cells[k].yc), -cells[k].xc))
    out = []
    for p, c in ((prices[k], cells[k]) for k in order):
        col = 1 + min(range(len(centres)), key=lambda k: abs(centres[k] - c.xc))
        out.append({"cell": grid.to_norm((c.x0, c.y0, c.x1, c.y1)),
                    "h": round((p["box"][3] - p["box"][1]) / grid.fh, 5),
                    "w": round((p["box"][2] - p["box"][0]) / grid.fw, 5),
                    "dx": round(((p["box"][0] + p["box"][2]) / 2 - c.xc) / grid.fw, 5),
                    "pol": p["pol"], "text": p.get("text", ""), "label": p.get("label", ""),
                    "group": p.get("group", ""), "column": p.get("column", ""),
                    "column_id": col, "row": _encode(grid.row_signature(c))})
    scripts = [digit_script(p.get("text", "")) for p in prices if p.get("text")]
    return {"grid": fingerprint(grid), "prices": out,
            "script": max(set(scripts), key=scripts.count) if scripts else ""}


def _unknown_numbers(grid: Grid, ink: raster.InkMap, cells: list[Cell], pols: list[str], price_h: float) -> bool:
    """A number in a price column that is not one of the prices: between the first
    and the last price of a run of the column, or below the last one (headers
    above the first price are not numbers)."""
    columns: dict[tuple[int, int], tuple[Cell, str]] = {}
    for cell, pol in zip(cells, pols):
        columns.setdefault((round(cell.x0), round(cell.x1)), (cell, pol))
    for cell, pol in columns.values():
        for run in grid.column_runs(cell):
            known = [k for k, c in enumerate(run) if any(abs(c.yc - p.yc) < 2 and abs(c.xc - p.xc) < 2 for p in cells)]
            if not known:
                continue
            for k in range(known[0] + 1, len(run)):
                if k in known:
                    continue
                if _numberlike(_cell_ink(ink, run[k], pol), price_h, run[k]):
                    return True
    return False


@dataclass
class Placed:
    box: tuple[int, int, int, int]
    rec: dict


def place(layout: dict, grid: Grid, rgb: np.ndarray) -> tuple[raster.InkMap, list[Placed]] | None:
    """Every price of a known format on this page, found in its cell. None if the
    page does not fit the format exactly. A small photo is looked at enlarged (its
    prices a comfortable size), the boxes given back in the photo's own pixels."""
    price_h = float(statistics.median(r["h"] for r in layout["prices"])) * grid.fh
    if price_h < MIN_DIGIT_H:
        log.info("known list: prices %.1f px tall, too small to find every zero", price_h)
        return None
    f = min(3.0, WORK_DIGIT_H / price_h)
    if f <= 1.05:
        found = _place(layout, grid, rgb)
        return None if found is None else (found[0], found[1])
    big = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    found = _place(layout, find(big), big)
    if found is None:
        return None
    H, W = rgb.shape[:2]
    out = [Placed((max(0, int(p.box[0] / f)), max(0, int(p.box[1] / f)),
                   min(W, math.ceil(p.box[2] / f)), min(H, math.ceil(p.box[3] / f))), p.rec) for p in found[1]]
    return _ink_map(rgb, price_h), out


def _place(layout: dict, grid: Grid, rgb: np.ndarray) -> tuple[raster.InkMap, list[Placed]] | None:
    recs = layout["prices"]
    price_h = float(statistics.median(r["h"] for r in recs)) * grid.fh
    ink = _ink_map(rgb, price_h)
    out: list[Placed] = []
    cells: list[Cell] = []
    unsure = 0
    for r in recs:
        x0, y0, x1, y1 = grid.to_px(r["cell"])
        c = grid.cell_at((x0 + x1) / 2, (y0 + y1) / 2)
        hard = soft = ""
        u = None
        if c is None or not _same_cell(grid.to_norm((c.x0, c.y0, c.x1, c.y1)), r["cell"]):
            hard = "its cell is not where it was"
        else:
            u = _cell_ink(ink, c, r["pol"])
            row = row_likeness(grid.row_signature(c), _decode(r["row"]))
            if row < ROW_SAME:
                hard = f"the text of its row changed (likeness {row:.2f})"
            elif u is None or (u[3] - u[1]) < 0.4 * r["h"] * grid.fh:
                hard = "its cell is empty"
            elif not _numberlike(u, r["h"] * grid.fh, c):
                soft = f"its ink does not look like the price did ({u})"
            elif u[2] - u[0] < 0.7 * r.get("w", 0.0) * grid.fw:
                soft = f"far narrower than it was ({u[2] - u[0]} < {0.7 * r['w'] * grid.fw:.1f} px)"
        if hard:
            log.info("known list «%s» does not fit: price «%s» - %s", layout.get("name"), r.get("label"), hard)
            return None
        if soft:
            # faint or smudged on this photo: the whole inside of the cell is the price's place
            # (a price cell holds nothing else), so no faint zero is left out
            log.info("known list «%s»: price «%s» - %s; using its whole cell", layout.get("name"), r.get("label"), soft)
            unsure += 1
            ix0, iy0, ix1, iy1 = c.inner
            u = (math.ceil(ix0) + 1, math.ceil(iy0) + 1, int(ix1) - 1, int(iy1) - 1)
        if "dx" in r:
            # never narrower than the price was, where it was: a zero too faint to see on
            # this photo is still inside the box (erased, and shown when the digits are read)
            half = r["w"] * grid.fw / 2
            xc = c.xc + r["dx"] * grid.fw
            ix0, iy0, ix1, iy1 = c.inner
            grow = max(0.0, r["h"] * grid.fh - (u[3] - u[1])) / 2 + 1     # the faint tip of a tall digit
            u = (max(math.ceil(ix0), min(u[0], int(xc - half))), max(math.ceil(iy0), int(u[1] - grow)),
                 min(int(ix1), max(u[2], math.ceil(xc + half))), min(int(iy1), math.ceil(u[3] + grow)))
        cells.append(c)
        out.append(Placed(u, r))
    if unsure > max(2, 0.15 * len(recs)):
        log.info("known list «%s» does not fit: %d prices unclear", layout.get("name"), unsure)
        return None
    if _unknown_numbers(grid, ink, cells, [r["pol"] for r in recs], price_h):
        log.info("known list «%s» does not fit: a number in a price column it does not know", layout.get("name"))
        return None
    return ink, out


def same_cells(a: dict, b: dict) -> bool:
    """Two records of one format hold their prices in the same cells."""
    pa, pb = a.get("prices", []), b.get("prices", [])
    if len(pa) != len(pb):
        return False
    left = list(pb)
    for p in pa:
        k = next((i for i, q in enumerate(left) if _same_cell(p["cell"], q["cell"])), None)
        if k is None:
            return False
        left.pop(k)
    return True
