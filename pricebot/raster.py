"""Pixel-level work: finding the exact ink of each price, erasing it, and
drawing the new price so it looks like it was always there.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import fonts
from .fonts import FontStyle
from .numfmt import digit_script

log = logging.getLogger(__name__)

DARK_T = 40      # min darkness vs. local background to count as ink
LINE_T = 22      # lower threshold used only for detecting table lines


@dataclass
class Word:
    x0: int
    y0: int
    x1: int
    y1: int
    polarity: str
    comps: list[int] = field(default_factory=list)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


class InkMap:
    """Ink masks, table lines and word boxes of one raster page."""

    def __init__(self, rgb: np.ndarray, text_h: float):
        self.rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.text_h = float(max(6.0, text_h))
        k = int(self.text_h * 3) | 1
        k = min(max(k, 15), 255)
        bg = cv2.medianBlur(self.gray, k)
        dark = cv2.subtract(bg, self.gray)
        light = cv2.subtract(self.gray, bg)

        line_src = ((dark > LINE_T) | (light > 3 * LINE_T)).astype(np.uint8) * 255
        hk = max(15, int(self.text_h * 2.4))
        vk = max(15, int(self.text_h * 2.0))
        hl = cv2.morphologyEx(line_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
        vl = cv2.morphologyEx(line_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
        self.hlines = hl
        self.vlines = vl
        self.lines = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
        not_lines = cv2.bitwise_not(self.lines)

        self.dark = dark
        self.light = light
        self.ink = {
            "dark": cv2.bitwise_and((dark > DARK_T).astype(np.uint8) * 255, not_lines),
            "light": cv2.bitwise_and((light > DARK_T + 20).astype(np.uint8) * 255, not_lines),
        }
        self.words: list[Word] = []
        self._comp_stats: dict[str, np.ndarray] = {}
        for pol in ("dark", "light"):
            self.words += self._words(pol)

    def _words(self, pol: str) -> list[Word]:
        ink = self.ink[pol]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        self._comp_stats[pol] = stats
        if n <= 1:
            return []
        kx = max(2, int(round(self.text_h * 0.32)))
        ky = max(1, int(round(self.text_h * 0.12)))
        dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        dil[self.lines > 0] = 0
        dil = cv2.bitwise_or(dil, ink)
        _, glab = cv2.connectedComponents(dil, connectivity=8)
        grp = np.zeros(n, dtype=np.int32)
        sel = lab > 0
        np.maximum.at(grp, lab[sel], glab[sel])
        groups: dict[int, list[int]] = {}
        for comp in range(1, n):
            groups.setdefault(int(grp[comp]), []).append(comp)
        words = []
        max_h = self.text_h * 4
        for comps in groups.values():
            xs0 = [stats[c, 0] for c in comps]
            ys0 = [stats[c, 1] for c in comps]
            xs1 = [stats[c, 0] + stats[c, 2] for c in comps]
            ys1 = [stats[c, 1] + stats[c, 3] for c in comps]
            area = sum(int(stats[c, 4]) for c in comps)
            w = Word(min(xs0), min(ys0), max(xs1), max(ys1), pol, comps)
            if area < 4 or w.h > max_h:
                continue
            words.append(w)
        return words

    def comp_boxes(self, pol: str, comps: list[int]) -> list[tuple[int, int, int, int, int]]:
        """(x0, y0, x1, y1, area) of each connected component."""
        st = self._comp_stats[pol]
        return [(int(st[c, 0]), int(st[c, 1]), int(st[c, 0] + st[c, 2]), int(st[c, 1] + st[c, 3]), int(st[c, 4]))
                for c in comps]


@dataclass
class Target:
    """One price on a raster page, located to the pixel."""
    box: tuple[int, int, int, int]
    polarity: str
    text: str                 # as displayed
    bounds: tuple[int, int]   # free horizontal space (cell) around it
    align: str = "center"
    column: int = 0
    style: FontStyle | None = None
    color: tuple[int, int, int] = (0, 0, 0)
    baseline: float = 0.0     # y of the text baseline in page pixels
    size_scale: float = 1.0   # this price is set larger/smaller than the rest of its column


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _cluster_columns(boxes: list[tuple[float, float, float, float]]) -> list[int]:
    """Group boxes into columns by horizontal overlap. Returns column index per box."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2]) / 2)
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            bi, bj = boxes[i], boxes[j]
            ov = _overlap(bi[0], bi[2], bj[0], bj[2])
            if ov >= 0.3 * min(bi[2] - bi[0], bj[2] - bj[0]):
                parent[find(i)] = find(j)
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(boxes))]


def _align_dp(items: list[tuple[float, float, float, float]], cands: list[Word], h: float) -> list[int | None]:
    """Order-preserving assignment of approximate AI boxes (sorted by y) to words."""
    n, m = len(items), len(cands)
    unmatched = 3.0
    inf = float("inf")

    def cost(i: int, j: int) -> float:
        x0, y0, x1, y1 = items[i]
        c = cands[j]
        dy = abs((y0 + y1) / 2 - c.yc) / h
        if dy > 2.5:
            return inf
        wa = max(1.0, x1 - x0)
        wr = abs(math.log(max(1.0, c.w) / wa))
        dx = abs((x0 + x1) / 2 - c.xc) / max(wa, h)
        return dy + 0.6 * min(wr, 2.0) + 0.4 * min(dx, 3.0)

    D = np.full((n + 1, m + 1), inf)
    B = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 skip cand, 1 item unmatched, 2 match
    D[0, :] = 0.0
    for i in range(1, n + 1):
        D[i, 0] = i * unmatched
        B[i, 0] = 1
        for j in range(1, m + 1):
            best, move = D[i, j - 1], 0
            if D[i - 1, j] + unmatched < best:
                best, move = D[i - 1, j] + unmatched, 1
            c = cost(i - 1, j - 1)
            if c < inf and D[i - 1, j - 1] + c < best:
                best, move = D[i - 1, j - 1] + c, 2
            D[i, j], B[i, j] = best, move
    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0:
        mv = B[i, j]
        if j == 0 or mv == 1:
            i -= 1
        elif mv == 0:
            j -= 1
        else:
            out[i - 1] = j - 1
            i -= 1
            j -= 1
    return out


def locate(ink: InkMap, approx: list[tuple[float, float, float, float]], texts: list[str]) -> list[Target | None]:
    """Snap approximate boxes (e.g. from the vision model) to real ink."""
    if not approx:
        return []
    h_med = float(np.median([b[3] - b[1] for b in approx]))
    cols = _cluster_columns(approx)
    results: list[Target | None] = [None] * len(approx)
    for col in set(cols):
        idx = sorted([i for i in range(len(approx)) if cols[i] == col], key=lambda i: approx[i][1] + approx[i][3])
        bx0 = min(approx[i][0] for i in idx) - 0.3 * h_med
        bx1 = max(approx[i][2] for i in idx) + 0.3 * h_med
        by0 = min(approx[i][1] for i in idx) - 4 * h_med
        by1 = max(approx[i][3] for i in idx) + 4 * h_med
        cands = [w for w in ink.words
                 if _overlap(w.x0, w.x1, bx0, bx1) >= 0.4 * max(1, w.w)
                 and 0.35 * h_med <= w.h <= 2.6 * h_med and w.y1 >= by0 and w.y0 <= by1]
        cands.sort(key=lambda w: w.yc)
        assign = _align_dp([approx[i] for i in idx], cands, h_med)
        for k, i in enumerate(idx):
            seed = cands[assign[k]] if assign[k] is not None else None
            results[i] = _refine(ink, approx[i], texts[i], seed, h_med, col)
    _set_alignment(ink, results)
    return results


def _refine(ink: InkMap, box, text: str, seed: Word | None, h_med: float, col: int) -> Target | None:
    ax0, ay0, ax1, ay1 = box
    if seed is None:
        # No word matched in order; fall back to whatever ink sits under the box.
        near = [w for w in ink.words
                if _overlap(w.y0, w.y1, ay0, ay1) >= 0.5 * min(w.h, ay1 - ay0)
                and _overlap(w.x0, w.x1, ax0, ax1) >= 0.3 * max(1, w.w)]
        if not near:
            return None
        seed = max(near, key=lambda w: _overlap(w.x0, w.x1, ax0, ax1))
    pol = seed.polarity
    h_ref = float(seed.h)
    min_area = max(3.0, 0.006 * h_ref * h_ref)
    n_chars = max(1, sum(1 for ch in text if not ch.isspace()))

    def glyphs(bxs) -> int:
        return sum(1 for b in bxs if b[4] >= min_area)

    seed_glyphs = glyphs(ink.comp_boxes(pol, seed.comps))
    # expected width from the seed's own glyph pitch (font independent)
    w_exp = seed.w * n_chars / seed_glyphs if seed_glyphs else float(ax1 - ax0)
    reach = 2.5 * w_exp + h_ref

    # Components on the same text line near the seed, grouped into tight segments.
    line_words = [w for w in ink.words if w.polarity == pol
                  and _overlap(w.y0, w.y1, seed.y0, seed.y1) >= 0.5 * min(w.h, seed.h)
                  and w.h <= 1.7 * h_ref
                  and w.x1 >= seed.x0 - reach and w.x0 <= seed.x1 + reach]
    if seed not in line_words:
        line_words.append(seed)
    boxes = sorted(ink.comp_boxes(pol, [c for w in line_words for c in w.comps]), key=lambda b: b[0])
    segs: list[list[tuple[int, int, int, int]]] = []
    right = -10 ** 9
    for b in boxes:
        if segs and b[0] - right < 0.2 * h_ref:
            segs[-1].append(b)
        else:
            segs.append([b])
        right = max(right, b[2])
    spans = [(min(b[0] for b in s), max(b[2] for b in s)) for s in segs]
    core = [k for k, (s0, s1) in enumerate(spans) if _overlap(s0, s1, seed.x0, seed.x1) > 0]
    if not core:
        return None

    best, best_score = None, float("inf")
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            if not any(i <= k <= j for k in core):
                continue
            if any(spans[k + 1][0] - spans[k][1] > 0.9 * h_ref or
                   _vline_between(ink, spans[k][1], spans[k + 1][0], seed.y0, seed.y1)
                   for k in range(i, j)):
                continue
            x0, x1 = spans[i][0], spans[j][1]
            iou = _overlap(x0, x1, ax0, ax1) / max(1.0, max(x1, ax1) - min(x0, ax0))
            n = glyphs([b for s in segs[i:j + 1] for b in s])
            score = (1.5 * abs(n - n_chars) / n_chars
                     + 0.5 * abs(math.log(max(1.0, x1 - x0) / max(1.0, w_exp)))
                     + 0.4 * (1 - iou) + 0.02 * (j - i))
            if score < best_score:
                best, best_score = (i, j), score
    if best is None:
        return None
    sel = [b for s in segs[best[0]:best[1] + 1] for b in s]
    x0, y0 = min(b[0] for b in sel), min(b[1] for b in sel)
    x1, y1 = max(b[2] for b in sel), max(b[3] for b in sel)
    left, right = _free_bounds(ink, (x0, y0, x1, y1), pol)
    return Target((x0, y0, x1, y1), pol, text, (left, right), column=col)


def _vline_between(ink: InkMap, xa: int, xb: int, y0: int, y1: int) -> bool:
    if xb <= xa:
        return False
    band = ink.vlines[max(0, y0):max(0, y1), xa:xb]
    return bool(band.size) and bool(band.max() > 0)


def _free_bounds(ink: InkMap, box, pol: str) -> tuple[int, int]:
    """Horizontal space available around a box before hitting a line or other text."""
    x0, y0, x1, y1 = box
    H, W = ink.gray.shape
    yc0, yc1 = max(0, y0), min(H, y1)
    band = ink.vlines[yc0:yc1, :]
    col_hits = np.nonzero(band.max(axis=0) > 0)[0] if band.size else np.array([], dtype=int)
    left = max([c for c in col_hits if c < x0], default=0)
    right = min([c for c in col_hits if c > x1], default=W - 1)
    for w in ink.words:
        if _overlap(w.y0, w.y1, y0, y1) < 0.4 * min(w.h, y1 - y0):
            continue
        if w.x1 <= x0 and w.x1 > left:
            left = w.x1
        elif w.x0 >= x1 and w.x0 < right:
            right = w.x0
    return int(left), int(right)


def _set_alignment(ink: InkMap, targets: list[Target | None]) -> None:
    by_col: dict[int, list[Target]] = {}
    for t in targets:
        if t is not None:
            by_col.setdefault(t.column, []).append(t)
    for col, ts in by_col.items():
        align = None
        widths = [t.box[2] - t.box[0] for t in ts]
        if len(ts) >= 2 and max(widths) - min(widths) > 0.3 * ink.text_h:
            sx0 = np.std([t.box[0] for t in ts])
            sx1 = np.std([t.box[2] for t in ts])
            sxc = np.std([(t.box[0] + t.box[2]) / 2 for t in ts])
            best = min((sx1, "right"), (sxc, "center"), (sx0, "left"))
            if best[0] < 0.25 * ink.text_h:
                align = best[1]
        for t in ts:
            if align:
                t.align = align
                continue
            gl = t.box[0] - t.bounds[0]
            gr = t.bounds[1] - t.box[2]
            span = t.bounds[1] - t.bounds[0]
            if abs(gl - gr) <= max(3, 0.12 * span):
                t.align = "center"
            else:
                t.align = "right" if gr < gl else "left"


# ---------------------------------------------------------------- styling --

def ink_mask(ink: InkMap, t: Target, relative: bool = False) -> np.ndarray:
    """Glyph pixels inside the target box. `relative` thresholds at half the local
    contrast, which matches how a rendered glyph is binarised (stroke weight)."""
    x0, y0, x1, y1 = t.box
    base = ink.ink[t.polarity][y0:y1, x0:x1] > 0
    if not relative or base.sum() < 3:
        return base
    strength = (ink.dark if t.polarity == "dark" else ink.light)[y0:y1, x0:x1]
    peak = float(np.percentile(strength[base], 95))
    return base & (strength >= 0.5 * peak)


def ink_coverage(ink: InkMap, t: Target) -> np.ndarray:
    """Anti-aliased glyph coverage (0..1) inside the target box."""
    x0, y0, x1, y1 = t.box
    base = ink.ink[t.polarity][y0:y1, x0:x1] > 0
    strength = (ink.dark if t.polarity == "dark" else ink.light)[y0:y1, x0:x1].astype(np.float32)
    if base.sum() < 3:
        return base.astype(np.float32)
    peak = float(np.percentile(strength[base], 95)) or 1.0
    near = cv2.dilate(base.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.clip(strength / peak, 0, 1) * near


def text_color(ink: InkMap, t: Target) -> tuple[int, int, int]:
    x0, y0, x1, y1 = t.box
    strength = (ink.dark if t.polarity == "dark" else ink.light)[y0:y1, x0:x1]
    mask = ink_mask(ink, t)
    if mask.sum() < 3:
        return (0, 0, 0) if t.polarity == "dark" else (255, 255, 255)
    thr = np.percentile(strength[mask], 80)
    core = mask & (strength >= thr)
    px = ink.rgb[y0:y1, x0:x1][core]
    return tuple(int(v) for v in np.median(px, axis=0))


def style_columns(ink: InkMap, targets: list[Target | None]) -> None:
    """Choose one font per column (so all rewritten prices look consistent)."""
    by_col: dict[int, list[Target]] = {}
    for t in targets:
        if t is not None:
            by_col.setdefault(t.column, []).append(t)
    # Columns of one list set in the same size are almost always the same font:
    # match them together (more samples, consistent result).
    groups: list[list[list[Target]]] = []
    for ts in by_col.values():
        h = float(np.median([_digit_height(ink_mask(ink, t, relative=True)) for t in ts]))
        for g in groups:
            gh = float(np.median([_digit_height(ink_mask(ink, t, relative=True)) for t in g[0]]))
            if g[0][0].polarity == ts[0].polarity and abs(h / gh - 1) <= 0.12:
                g.append(ts)
                break
        else:
            groups.append([ts])
    styles: dict[int, FontStyle | None] = {}
    ref_h: dict[int, float] = {}
    for g in groups:
        pool = [t for ts in g for t in ts]
        samples = sorted(pool, key=lambda t: -(t.box[2] - t.box[0]))[:5]
        scripts = [digit_script(t.text) for t in samples]
        style = fonts.match_font([(ink_coverage(ink, t), t.text) for t in samples],
                                 prefer_script=max(set(scripts), key=scripts.count))
        smp_h = float(np.median([_digit_height(ink_mask(ink, t, relative=True)) for t in samples]))
        for ts in g:
            ref_h[id(ts)] = smp_h
            styles[id(ts)] = None if style is None else FontStyle(style.path, style.size_px, style.hscale,
                                                                  style.script, style.score)
    for ts in by_col.values():
        style = styles.get(id(ts))
        if style is None:
            continue
        # One size and one colour for the whole column: per-item estimates wobble
        # with JPEG noise, and a price column is uniform in the original anyway.
        med_all = float(np.median([_digit_height(ink_mask(ink, t, relative=True)) for t in ts]))
        med_smp = ref_h[id(ts)]
        if med_smp > 0:
            style.size_px *= med_all / med_smp
        colors = np.array([text_color(ink, t) for t in ts], dtype=np.float32)
        med_color = np.median(colors, axis=0)
        for t, c in zip(ts, colors):
            t.style = style
            own = np.abs(c - med_color).max() > 60          # e.g. one price highlighted in red
            t.color = tuple(int(v) for v in (c if own else med_color))
            t.baseline = _baseline(ink, t, style)
            ratio = _digit_height(ink_mask(ink, t, relative=True)) / med_all if med_all else 1.0
            t.size_scale = ratio if abs(ratio - 1) > 0.18 else 1.0   # e.g. a bold/large total row


def _digit_rows(mask: np.ndarray) -> tuple[float, float] | None:
    """(top, bottom) of the digit bodies inside a glyph mask, ignoring commas
    and dots that hang below or float in the middle."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    hs = st[1:, 3]
    tall = st[1:][hs >= 0.6 * hs.max()]
    return float(np.median(tall[:, 1])), float(np.median(tall[:, 1] + tall[:, 3]))


def _digit_height(mask: np.ndarray) -> float:
    rows = _digit_rows(mask)
    return rows[1] - rows[0] if rows else float(mask.shape[0])


def _baseline(ink: InkMap, t: Target, style: FontStyle) -> float:
    """Baseline in page pixels: where the original digits sit, corrected by how
    far the chosen font's digits sit from its own baseline."""
    x0, y0, x1, y1 = t.box
    rows = _digit_rows(ink_mask(ink, t, relative=True))
    if rows is None:
        return float(y1)
    variant = dict(fonts._script_variants(t.text)).get(style.script, t.text)
    r = fonts.render(style.path, variant, style.size_px, style.hscale, ss=2)
    if r is None:
        return y0 + rows[1]
    rx0, ry0, rx1, ry1 = r.ink
    rrows = _digit_rows(r.alpha[ry0:ry1, rx0:rx1] > 0.5)
    overshoot = (ry0 + rrows[1] - r.baseline) if rrows else 0.0
    return y0 + rows[1] - overshoot


# ------------------------------------------------------------ erase/draw --

def _dominant_color(px: np.ndarray) -> tuple[np.ndarray, float]:
    if len(px) == 0:
        return np.array([255, 255, 255], dtype=np.float32), 0.0
    q = (px // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    sel = px[keys == top].astype(np.float32)
    color = sel.mean(axis=0)
    near = np.abs(px.astype(np.float32) - color).max(axis=1) <= 14
    return color, float(near.mean())


def erase(img: np.ndarray, ink: InkMap, t: Target) -> tuple[int, int, int, int]:
    """Remove the old number, restoring the cell background. Returns the erased rect."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = t.box
    h = y1 - y0
    pad = max(2, int(round(0.12 * h)))
    ex0, ey0 = max(0, x0 - pad), max(0, y0 - pad)
    ex1, ey1 = min(W, x1 + pad), min(H, y1 + pad)
    r = max(3, int(round(0.35 * h)))
    rx0, ry0, rx1, ry1 = max(0, ex0 - r), max(0, ey0 - r), min(W, ex1 + r), min(H, ey1 + r)

    ring = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
    ring[ey0 - ry0:ey1 - ry0, ex0 - rx0:ex1 - rx0] = False
    any_ink = (ink.ink["dark"] | ink.ink["light"])[ry0:ry1, rx0:rx1] > 0
    any_ink = cv2.dilate(any_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring &= ~any_ink & ~(ink.lines[ry0:ry1, rx0:rx1] > 0)
    color, flat = _dominant_color(img[ry0:ry1, rx0:rx1][ring])

    lines = ink.lines[ey0:ey1, ex0:ex1] > 0
    if flat >= 0.7:
        region = img[ey0:ey1, ex0:ex1]
        region[~lines] = color.astype(np.uint8)
    else:
        # Textured/gradient background: inpaint just the glyph pixels, locally.
        mask = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
        local = (ink.ink[t.polarity][ey0:ey1, ex0:ex1] > 0).astype(np.uint8) * 255
        local = cv2.dilate(local, np.ones((3, 3), np.uint8), iterations=2)
        local[lines] = 0
        mask[ey0 - ry0:ey1 - ry0, ex0 - rx0:ex1 - rx0] = local
        crop = cv2.cvtColor(np.ascontiguousarray(img[ry0:ry1, rx0:rx1]), cv2.COLOR_RGB2BGR)
        fixed = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        img[ry0:ry1, rx0:rx1] = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    return ex0, ey0, ex1, ey1


def draw(img: np.ndarray, t: Target, new_text: str, soften: bool = False) -> tuple[int, int, int, int] | None:
    """Draw `new_text` in the target's style. Returns the painted rect."""
    style = t.style
    if style is None:
        return None
    H, W = img.shape[:2]
    text = dict(fonts._script_variants(new_text)).get(style.script, new_text)
    if not fonts.supports(style.path, text):
        return None
    size, hscale = style.size_px * t.size_scale, style.hscale
    x0, y0, x1, y1 = t.box
    orig_h = y1 - y0

    avail = (t.bounds[1] - t.bounds[0]) - max(2, 0.25 * orig_h)
    r = fonts.render(style.path, text, size, hscale)
    if r is None:
        return None
    width = r.ink[2] - r.ink[0]
    if 0 < avail < width:
        # Too wide for the cell: condense a little first, then shrink the size.
        shrink = avail / width
        new_h = max(hscale * shrink, 0.82 * style.hscale)
        rest = shrink * hscale / new_h
        hscale = new_h
        if rest < 1:
            size *= rest
        r = fonts.render(style.path, text, size, hscale) or r
    width = r.ink[2] - r.ink[0]
    if t.align == "right":
        left = x1 - width
    elif t.align == "left":
        left = float(x0)
    else:
        left = (x0 + x1) / 2 - width / 2
    left = min(max(left, t.bounds[0] + 1), t.bounds[1] - 1 - width)
    top = t.baseline - r.baseline  # alpha row 0 in page coords

    # re-render with the sub-pixel remainder so placement is exact
    ox = left - r.ink[0]
    fx, fy = ox - math.floor(ox), top - math.floor(top)
    r = fonts.render(style.path, text, size, hscale, frac=(fx, fy)) or r
    ox, oy = int(math.floor(ox)), int(math.floor(top))
    alpha = r.alpha
    if soften and orig_h >= 18:
        # match the slight blur of JPEG text; small text would just turn grey
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.4)
    ah, aw = alpha.shape
    px0, py0 = max(0, ox), max(0, oy)
    px1, py1 = min(W, ox + aw), min(H, oy + ah)
    if px1 <= px0 or py1 <= py0:
        return None
    a = alpha[py0 - oy:py1 - oy, px0 - ox:px1 - ox][..., None]
    region = img[py0:py1, px0:px1].astype(np.float32)
    col = np.array(t.color, dtype=np.float32)
    img[py0:py1, px0:px1] = np.clip(region * (1 - a) + col * a, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(a[..., 0] > 0.02)
    if len(xs) == 0:
        return None
    return px0 + int(xs.min()), py0 + int(ys.min()), px0 + int(xs.max()) + 1, py0 + int(ys.max()) + 1
