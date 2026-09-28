"""A photo/scan's table rebuilt from OCR words (for the Excel export without Gemini)."""
from __future__ import annotations

import cv2
import numpy as np

from .layout import PageDoc, Tok, center_in


def _vertical_lines(rgb: np.ndarray, line_h: float) -> list[float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(25, int(line_h * 4))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    cols = (vl > 0).sum(axis=0)
    xs = np.nonzero(cols >= max(k, 0.25 * cols.max() if cols.max() else 1))[0]
    out: list[float] = []
    for x in xs:
        if out and x - out[-1] <= 3:
            out[-1] = (out[-1] + x) / 2
        else:
            out.append(float(x))
    return out


def _gap_bounds(toks: list[Tok], width: float, line_h: float) -> list[float]:
    cover = np.zeros(int(width) + 2, bool)
    for t in toks:
        cover[max(0, int(t.box[0])):int(t.box[2]) + 1] = True
    bounds, run = [], 0
    for x, c in enumerate(cover):
        if not c:
            run += 1
        else:
            if run >= 1.2 * line_h:
                bounds.append(x - run / 2)
            run = 0
    return bounds


def build(doc: PageDoc, rgb: np.ndarray | None, prices: list[tuple[str, tuple]]) -> dict | None:
    """prices: (item id, pixel box). Returns {"title", "direction", "headers", "rows"}
    with rows as lists of (text, price_id), or None when no table shape is found."""
    toks = doc.words + doc.nums
    if not toks or not prices:
        return None
    lh = doc.line_h
    bounds = _vertical_lines(rgb, lh) if rgb is not None else []
    if len(bounds) < 3:
        bounds = _gap_bounds(toks, doc.width, lh)
    edges = [0.0] + sorted(bounds) + [doc.width + 1]
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= 0.8 * lh]
    if len(spans) < 2:
        return None

    # rows: tokens grouped by vertical position
    rows: list[list[Tok]] = []
    for t in sorted(toks, key=lambda t: t.yc):
        if rows and abs(t.yc - np.mean([r.yc for r in rows[-1]])) <= 0.55 * lh:
            rows[-1].append(t)
        else:
            rows.append([t])

    rtl = doc.rtl()
    order = sorted(range(len(spans)), key=lambda k: -spans[k][0] if rtl else spans[k][0])
    grid: list[list[tuple[str, str]]] = []
    for row in rows:
        cells = []
        for k in order:
            a, b = spans[k]
            inside = [t for t in row if a <= t.xc < b]
            inside.sort(key=lambda t: -t.xc if rtl else t.xc)
            pid = ""
            for t in inside:
                if t.num is not None:
                    pid = next((i for i, box in prices if center_in(t.box, box, 0.3 * lh)), pid)
            cells.append((" ".join(t.text for t in inside), pid))
        grid.append(cells)

    first = next((r for r, cells in enumerate(grid) if any(pid for _, pid in cells)), None)
    if first is None:
        return None
    keep = [k for k in range(len(order)) if any(grid[r][k][0].strip() for r in range(len(grid)))]
    grid = [[cells[k] for k in keep] for cells in grid]
    head = first - 1 if first >= 1 else None
    headers = [t for t, _ in grid[head]] if head is not None else [""] * len(keep)
    title = " ".join(" ".join(t for t, _ in cells if t) for cells in grid[:max(0, head or 0)]).strip()
    return {"title": title[:100], "direction": "rtl" if rtl else "ltr", "headers": headers,
            "rows": [cells for cells in grid[first:] if any(t.strip() or p for t, p in cells)]}
