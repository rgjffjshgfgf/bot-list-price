"""Find the footers (eye icon, view count, time) of Telegram posts in a screenshot and read their characters."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .bank import EYE, Bank, load_bank

TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):(?:[0-5]\d?)?$")    # minutes may be hidden under a button
VIEWS_RE = re.compile(r"^(?:\d{1,3}|\d{1,3}(?:\.\d)?[KM])$")    # 999, 1K, 1.2K, 24.6K, 1.5M
SS = 3                      # supersampling used when a layer is moved to the screenshot's pixel grid
MIN_SCORE = 0.70            # a character whose best match is weaker than this is not trusted ('?')
MIN_MARGIN = 0.04           # ... nor one whose runner-up is this close
MIN_DOT = 0.40              # the decimal point is checked by its place instead (see _parse_footer)


class ShotError(Exception):
    """A problem worth telling the user about (the message is meant for them)."""


@dataclass
class Comp:
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int          # exclusive
    kind: str = "glyph"          # glyph | colon | part (a piece of a blob that JPEG noise had joined)
    char: str = "?"
    score: float = 0.0
    guess: str = "?"             # the best match even when it was not sure enough to be trusted as `char`
    cx: float = 0.0              # centre of mass in the screenshot (px)
    cy: float = 0.0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class Footer:
    eye: tuple[int, int, int, int]               # ink box of the eye icon (x0, y0, x1, y1)
    views: list[Comp]
    time: list[Comp]                             # the five characters of "hh:mm"
    sign: int = 1                                # +1 light text on dark, -1 dark text on light
    contrast: float = 1.0
    bg: float = 0.0
    band: tuple[int, int, int, int] = (0, 0, 0, 0)
    tx: float = 0.0                              # screenshot x = s * native x + tx
    ty: float = 0.0
    rms: float = 0.0                             # position error of the fit (px)
    notes: list[str] = field(default_factory=list)

    @property
    def views_text(self) -> str:
        return "".join(c.char for c in self.views)

    @property
    def views_guess(self) -> str:
        return "".join(c.guess for c in self.views)

    @property
    def time_text(self) -> str:
        return "".join(c.char for c in self.time)

    @property
    def time_ok(self) -> bool:
        return TIME_RE.match(self.time_text) is not None

    @property
    def readable(self) -> bool:
        return VIEWS_RE.match(self.views_text) is not None and self.time_ok


@dataclass
class Analysis:
    img: np.ndarray                  # BGR uint8
    scale: float                     # screenshot px per native px
    footers: list[Footer]
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0          # 0..1, how well the learned shapes explain this screenshot


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _warp(canvas: np.ndarray, cx0: float, cy0: float, s: float, tx: float, ty: float,
          win: tuple[int, int, int, int]) -> np.ndarray:
    """Move a native-resolution layer (index coords cx0.., cy0..) into the screenshot window (x0, y0, x1, y1).

    Native index i lands on screenshot position s*i + t (continuous), pixel j covers [j, j + 1]."""
    X0, Y0, X1, Y1 = win
    w, h = X1 - X0, Y1 - Y0
    bx = s * cx0 + tx - 0.5 - X0
    by = s * cy0 + ty - 0.5 - Y0
    if abs(s - 1.0) < 1e-3 and abs(bx - round(bx)) < 1e-2 and abs(by - round(by)) < 1e-2:
        out = np.zeros((h, w), np.float32)
        ox, oy = int(round(bx)), int(round(by))
        ch, cw = canvas.shape
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        ww, hh = min(cw - sx0, w - dx0), min(ch - sy0, h - dy0)
        if ww > 0 and hh > 0:
            out[dy0:dy0 + hh, dx0:dx0 + ww] = canvas[sy0:sy0 + hh, sx0:sx0 + ww]
        return out
    M = np.array([[s * SS, 0, SS * (bx + 0.5) - 0.5], [0, s * SS, SS * (by + 0.5) - 0.5]], np.float32)
    big = cv2.warpAffine(canvas, M, (w * SS, h * SS), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- the eye --

def _eye_template(bank: Bank) -> np.ndarray:
    g = bank.glyphs[EYE].a
    m = 3
    return np.pad(g, m).astype(np.float32), m


def _match(gray: np.ndarray, tpl: np.ndarray, s: float) -> np.ndarray:
    t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if t.shape[0] < 7 or t.shape[1] < 7 or t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1]:
        return np.zeros((1, 1), np.float32)
    r = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
    return np.abs(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


def _peaks(r: np.ndarray, thr: float, size: int) -> list[tuple[float, int, int]]:
    if r.max() < thr:
        return []
    k = max(3, size | 1)
    local = r == cv2.dilate(r, np.ones((k, k), np.uint8))
    ys, xs = np.where(local & (r >= thr))
    return [(float(r[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]


def find_eyes(gray: np.ndarray, bank: Bank, lo: float | None = None, hi: float | None = None
              ) -> tuple[float, list[tuple[float, int, int]]]:
    """Return (scale, [(score, x, y)]) of the eye icons; x, y is the template corner.

    A coarse search over the scales lo..hi (default: every scale that is worth trying) runs on a half-size
    copy of big screenshots, then every candidate is refined at full size."""
    tpl, m = _eye_template(bank)
    f = 0.5 if gray.shape[1] > 900 else 1.0
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1.0 else gray
    cands: list[tuple[float, float, int, int]] = []
    s = 0.40 if f == 1.0 else 0.56
    lo, hi = (lo or 0.0) / 1.045, min(hi or 1.50, 1.50) * 1.045    # one step of slack on either side
    while s <= 1.50:
        if not lo <= s <= hi:        # the same grid of sizes either way; a narrower range only skips some
            s *= 1.045
            continue
        r = _match(small, tpl, s * f)
        for sc, x, y in _peaks(r, 0.72, int(tpl.shape[0] * s * f)):
            cands.append((sc, s, int(round(x / f)), int(round(y / f))))
        s *= 1.045
    if not cands:
        return 1.0, []
    cands.sort(reverse=True)
    top = [c for c in cands if c[0] >= max(0.72, cands[0][0] - 0.06)][:6]
    scale = float(np.median([c[1] for c in top]))
    # distinct candidate positions near the consensus scale, refined at full size
    pos: list[tuple[int, int]] = []
    size = tpl.shape[0] * scale
    for sc, ss, x, y in cands:
        if abs(ss / scale - 1) > 0.12:
            continue
        if all(abs(x - px) > size * 0.6 or abs(y - py) > size * 0.6 for px, py in pos):
            pos.append((x, y))
    found: list[tuple[float, int, int, float]] = []
    for px, py in pos:
        best = None
        for fs in (0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04):
            ss = scale * fs
            t = cv2.resize(tpl, None, fx=ss, fy=ss, interpolation=cv2.INTER_AREA)
            th, tw = t.shape
            x0, y0 = max(0, px - 6), max(0, py - 6)
            x1, y1 = min(gray.shape[1], px + tw + 6), min(gray.shape[0], py + th + 6)
            if x1 - x0 < tw or y1 - y0 < th:
                continue
            r = np.abs(np.nan_to_num(cv2.matchTemplate(gray[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)))
            iy, ix = np.unravel_index(int(r.argmax()), r.shape)
            if best is None or r[iy, ix] > best[0]:
                best = (float(r[iy, ix]), x0 + int(ix), y0 + int(iy), ss)
        if best and best[0] >= 0.78:
            found.append(best)
    if not found:
        return scale, []
    found.sort(reverse=True)
    keep: list[tuple[float, int, int, float]] = []
    for sc, x, y, ss in found:
        if all(abs(x - k[1]) > size * 0.6 or abs(y - k[2]) > size * 0.6 for k in keep):
            keep.append((sc, x, y, ss))
    scales = [k[3] for k in keep if k[0] >= 0.88] or [k[3] for k in keep]
    return float(np.median(scales)), [(sc, x, y) for sc, x, y, _ in keep]


# ------------------------------------------------------------- characters --

def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), int(st[i, 4])) for i in range(1, n)]


def _alpha(gray: np.ndarray, box: tuple[int, int, int, int], bg: float, sign: int, contrast: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.clip(sign * (gray[y0:y1, x0:x1] - bg) / max(contrast, 1.0), 0, 1.3)


def _centroid(gray: np.ndarray, c: Comp, bg: float, sign: int, contrast: float) -> tuple[float, float]:
    H, W = gray.shape
    box = (max(0, c.x0 - 1), max(0, c.y0 - 1), min(W, c.x1 + 1), min(H, c.y1 + 1))
    a = _alpha(gray, box, bg, sign, contrast)
    a = np.where(a > 0.06, a, 0)
    tot = float(a.sum())
    if tot <= 0:
        return (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2
    ys, xs = np.mgrid[box[1]:box[3], box[0]:box[2]]
    return float((a * xs).sum() / tot) + 0.5, float((a * ys).sum() / tot) + 0.5


def read_char(gray: np.ndarray, c: Comp, bank: Bank, s: float, sign: int, hd: float
              ) -> tuple[str, float, float]:
    """Best learned character for one component: (char, score, runner-up score).

    Every candidate is drawn the way the screenshot itself was made: the native shape is placed on the
    component's centre of mass and resampled to the screenshot's scale (with small sub-pixel shifts and a
    little blur for JPEG / resized copies), then compared pixel by pixel. This stays exact on photos that
    Telegram shrank to less than half size, where the digits are only ~7 px wide."""
    H, W = gray.shape
    pad = 2
    X0, Y0, X1, Y1 = max(0, c.x0 - pad), max(0, c.y0 - pad), min(W, c.x1 + pad), min(H, c.y1 + pad)
    P = gray[Y0:Y1, X0:X1].astype(np.float32)
    P = P - P.mean()
    pn = float(np.sqrt((P * P).sum())) or 1.0
    small = c.h < 0.5 * hd
    native = abs(s - 1.0) < 2e-3
    sigmas = (0.0,) if native else (0.0, 0.45)

    def score(ch: str, shifts: tuple[float, ...]) -> float:
        g = bank.glyphs[ch]
        can = np.pad(g.a, 2)
        best = -1.0
        for dx in shifts:
            for dy in shifts:
                tx = c.cx + dx - s * (g.cx + 2)
                ty = c.cy + dy - s * (g.cy + 2)
                R0 = _warp(can, 0, 0, s, tx, ty, (X0, Y0, X1, Y1))
                for sg in sigmas:
                    R = cv2.GaussianBlur(R0, (0, 0), sg) if sg else R0
                    R = R - R.mean()
                    rn = float(np.sqrt((R * R).sum())) or 1.0
                    best = max(best, sign * float((P * R).sum()) / (pn * rn))
        return best

    chars = [ch for ch, g in bank.glyphs.items()
             if ch not in (EYE, ":") and (ch == ".") == small and abs(g.h * s - c.h) <= max(2.5, 0.35 * c.h)]
    if not chars:
        return "?", 0.0, 0.0
    # every candidate centred, then the sub-pixel search only for the few that came close
    scores = sorted(((score(ch, (0.0,)), ch) for ch in chars), reverse=True)
    scores = sorted([(score(ch, (-0.35, 0.0, 0.35)), ch) for _, ch in scores[:4]] + scores[4:], reverse=True)
    second = scores[1][0] if len(scores) > 1 else 0.0
    return scores[0][1], scores[0][0], second


def _parse_footer(gray: np.ndarray, bank: Bank, s: float, ex: int, ey: int, tw: int, th: int) -> Footer | None:
    H, W = gray.shape
    # ink box of the eye inside the template window
    wx0, wy0 = max(0, ex), max(0, ey)
    win = gray[wy0:min(H, ey + th), wx0:min(W, ex + tw)]
    if win.size == 0:
        return None
    border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    bg0 = float(np.median(border))
    dev = np.abs(win - bg0)
    contrast = float(np.percentile(dev, 99.5))
    if contrast < 12:
        return None
    sign = 1 if (win - bg0).max() >= -(win - bg0).min() else -1
    inkm = dev > 0.30 * contrast
    ys, xs = np.where(inkm)
    if len(xs) < 8:
        return None
    eye = (wx0 + int(xs.min()), wy0 + int(ys.min()), wx0 + int(xs.max()) + 1, wy0 + int(ys.max()) + 1)
    hd = bank.cap_h * s
    by0 = max(0, int(eye[1] - 0.45 * hd))
    by1 = min(H, int(eye[3] + 0.45 * hd))
    bx0 = min(W - 1, eye[2] + 1)
    bx1 = min(W, int(eye[2] + 340 * s))
    if bx1 - bx0 < 20 * s or by1 - by0 < 6:
        return None
    band = gray[by0:by1, bx0:bx1]
    # the bubble colour right next to the eye (the band may reach past the bubble edge onto the wallpaper)
    near = gray[by0:by1, bx0:min(W, bx0 + max(3, int(6 * s)))]
    bg = float(np.median(near)) if abs(float(np.median(near)) - bg0) < 0.5 * contrast else bg0
    ink = sign * (band - bg)

    def marks(mask: np.ndarray, x0: int, y0: int) -> list[Comp]:
        return [Comp(x0 + x, y0 + y, x0 + x + w, y0 + y + h)
                for x, y, w, h, area in sorted(_components(mask)) if area >= 2]

    def split() -> tuple[list[Comp], list[Comp]] | None:
        """The view count and the time as separate marks, or None when the band does not read as a footer."""
        items = marks(ink > 0.30 * contrast, bx0, by0)
        # keep the chain of neighbours that starts next to the eye
        chain: list[Comp] = []
        last_x = eye[2]
        while items:
            c = items.pop(0)
            if c.h > 1.45 * hd or c.w > 1.4 * hd:
                if chain and c.kind == "glyph":
                    # JPEG noise can join the time's last digit to whatever sits right after it (a tick, a
                    # button edge): look at just this blob with a stricter threshold, once
                    box = ink[c.y0 - by0:c.y1 - by0, c.x0 - bx0:c.x1 - bx0]
                    parts = marks(box > 0.5 * contrast, c.x0, c.y0)
                    for part in parts:
                        part.kind = "part"      # never split twice
                    if parts:
                        items[:0] = parts
                        continue
                if chain:
                    break
                continue
            c.kind = "glyph"
            if not chain and c.x0 - last_x > 14 * s * 1.6:
                return None
            if chain and c.x0 - last_x > 24 * s:
                break
            chain.append(c)
            last_x = max(last_x, c.x1)
        if len(chain) < 6:
            return None
        # merge the two dots of a colon
        merged: list[Comp] = []
        i = 0
        while i < len(chain):
            c = chain[i]
            if i + 1 < len(chain):
                n = chain[i + 1]
                small = c.h < 0.34 * hd and n.h < 0.34 * hd
                overlap = min(c.x1, n.x1) - max(c.x0, n.x0)
                if small and overlap >= 0.5 * min(c.w, n.w) and abs(c.y0 - n.y0) > 0.15 * hd:
                    merged.append(Comp(min(c.x0, n.x0), min(c.y0, n.y0), max(c.x1, n.x1), max(c.y1, n.y1), "colon", ":"))
                    i += 2
                    continue
            merged.append(c)
            i += 1
        colon = next((k for k, c in enumerate(merged) if c.kind == "colon"), None)
        if colon is None or colon < 3:
            return None
        time = merged[colon - 2: colon + 3]
        views = merged[:colon - 2]
        # the minutes are often half hidden under the round scroll-down button, or joined to its rim: the time
        # only anchors the layout (by its first digit), so doubtful marks at its end are left out (a sliver of a
        # covered digit can even pass for a second colon)
        while len(time) > 3 and (time[-1].h < 0.7 * hd or time[-1].kind in ("part", "colon")):
            time.pop()
        tall = [c for c in time if c.kind != "colon"]
        if any(c.h < 0.7 * hd for c in tall) or not views:
            return None
        return views, time

    parts = split()
    if parts is None:
        return None
    views, time = parts
    # the views/time gap must be the widest "space" in front of the time
    foot = Footer(eye, views, time, sign=sign, contrast=contrast, band=(bx0, by0, bx1, by1))
    for c in views + time:
        c.cx, c.cy = _centroid(gray, c, bg, sign, contrast)
        if c.kind == "colon":
            c.score, c.guess = 1.0, ":"
        else:
            c.char, c.score, second = read_char(gray, c, bank, s, sign, hd)
            c.guess = c.char
            if c.char == ".":
                continue
            if c.score < MIN_SCORE or c.score - second < MIN_MARGIN:
                c.char = "?"
    while len(time) > 3 and time[-1].char == "?":       # a minute digit the button covers
        time.pop()
    # on shrunk photos the point is a 2x2 px blob that no shape comparison is sure of; it is the only small
    # mark a view count has, so it is trusted when it sits on the baseline of the digits around it
    bottoms = [c.y1 for c in views if c.char not in (".", "?")]
    for c in views:
        if c.char == "." and (c.score < MIN_DOT or not bottoms
                              or abs(c.y1 - float(np.median(bottoms))) > max(1.5, 0.15 * hd)):
            c.char = "?"
    foot.bg = bg
    return foot


# ------------------------------------------------------------ layout fit --

def native_layout(bank: Bank, views: str, time: str, eye: bool = True) -> dict:
    """Native-pixel left edges of the eye, each views character and each time character.

    The first time character sits at x = 0."""
    xs_time = [0.0]
    for a, b in zip(time, time[1:]):
        xs_time.append(xs_time[-1] + bank.distance(a, b))
    xs_views: list[float] = []
    nxt_x, nxt_c, space = xs_time[0], time[0], True
    for ch in reversed(views):
        x = nxt_x - bank.distance(ch, nxt_c, space)
        xs_views.append(x)
        nxt_x, nxt_c, space = x, ch, False
    xs_views.reverse()
    out = {"time": xs_time, "views": xs_views}
    if eye:
        first_x = xs_views[0] if xs_views else xs_time[0]
        first_c = views[0] if views else time[0]
        out["eye"] = first_x - bank.distance(EYE, first_c)
    return out


def _points(foot: Footer, bank: Bank) -> np.ndarray:
    """(native x, native y, screenshot x, screenshot y) of every character centre; only the time when the
    view count could not be read."""
    lay = native_layout(bank, foot.views_text if foot.readable else "", foot.time_text, eye=False)
    pts = []
    groups = ((foot.views, lay["views"]), (foot.time, lay["time"])) if foot.readable else ((foot.time, lay["time"]),)
    for comps, xs in groups:
        for c, x in zip(comps, xs):
            g = bank.glyphs[c.char]
            pts.append((x + g.cx, g.dy + g.cy, c.cx, c.cy))
    return np.array(pts)


def fit_scale(foot: Footer, bank: Bank) -> float | None:
    """Screenshot pixels per native pixel, from the spread of the character centres."""
    if not foot.readable:
        return None
    P = _points(foot, bank)
    if len(P) < 5:
        return None
    keep = np.ones(len(P), bool)
    sol = np.array([1.0, 0.0])
    for _ in range(3):
        A = np.c_[P[keep, 0], np.ones(int(keep.sum()))]
        sol = np.linalg.lstsq(A, P[keep, 2], rcond=None)[0]
        res = np.abs(P[:, 0] * sol[0] + sol[1] - P[:, 2])
        keep = res < max(1.2, 2.5 * float(np.median(res)))
        if keep.sum() < 5:
            break
    return float(sol[0])


def fit_offsets(foot: Footer, bank: Bank, s: float) -> tuple[float, float, float] | None:
    """With the scale known: where native (0, 0) lands in the screenshot, and the leftover error (px)."""
    if not foot.time_ok:
        return None
    P = _points(foot, bank)
    if len(P) < 4:
        return None
    tx = float(np.median(P[:, 2] - s * P[:, 0]))
    ty = float(np.median(P[:, 3] - s * P[:, 1]))
    err = np.hypot(P[:, 2] - (s * P[:, 0] + tx), P[:, 3] - (s * P[:, 1] + ty))
    return tx, ty, float(np.sqrt(np.mean(err ** 2)))


# ----------------------------------------------------------------- entry --

def analyze(img_bgr: np.ndarray, bank: Bank | None = None, scale: float | None = None) -> Analysis:
    """Find and read every footer. `scale` (screenshot px per native px), when already known, skips the search
    over sizes; otherwise the sizes an iPhone screenshot of this width can have are tried first, then all."""
    bank = bank or load_bank()
    if not bank.ready():
        raise ShotError("نمونه‌های فونت هنوز روی ربات نصب نشده‌اند (فایل‌های pricebot/shot/assets).")
    gray = _gray(img_bgr)
    W = gray.shape[1]
    tpl, m = _eye_template(bank)
    if scale:
        tries = [(0.96 * scale, 1.04 * scale)]
    else:
        # iPhones are 1080..1320 px wide at 3x (2x models come out the same); a cropped screenshot falls through
        tries = [(0.93 * W / 1320, 1.07 * W / 1080), (None, None)]
    seen_eyes = False
    footers: list[Footer] = []
    for lo, hi in tries:
        s, eyes = find_eyes(gray, bank, lo, hi)
        if not eyes:
            continue
        seen_eyes = True
        tw = int(round(tpl.shape[1] * s))
        th = int(round(tpl.shape[0] * s))
        eyes = sorted(eyes, key=lambda e: (e[2], e[1]))

        def read_all(at: float) -> list[Footer]:
            found = (_parse_footer(gray, bank, at, x, y, tw, th) for sc, x, y in eyes)
            return [f for f in found if f is not None]

        footers = read_all(s)
        if footers:
            break
    if not seen_eyes:
        raise ShotError("هیچ «تعداد بازدید» (علامت چشم کنار ساعت پیام) در این عکس پیدا نشد.\n"
                        "این قابلیت برای اسکرین‌شات تلگرام روی آیفون است و پیام‌های کانال را می‌خواند.")
    if not footers:
        raise ShotError("علامت چشم پیدا شد ولی عدد بازدید و ساعت کنارش خوانده نشد.\n"
                        "اسکرین‌شات کامل و بدون برش بفرست و دوباره امتحان کن.")
    scales = [r for r in (fit_scale(f, bank) for f in footers) if r]
    s_fit = float(np.median(scales)) if scales else s
    if abs(s_fit - s) / s > 0.05:
        s_fit = s
    if abs(s_fit - s) / s > 0.003:
        # the eye only gives a rough scale; read once more with the one the characters agree on
        again = read_all(s_fit)
        if sum(f.readable for f in again) >= sum(f.readable for f in footers):
            footers = again
    for f in footers:
        r = fit_offsets(f, bank, s_fit)
        if r:
            f.tx, f.ty, f.rms = r
    warnings: list[str] = []
    unread = [f for f in footers if not f.readable]
    if unread:
        fixable = sum(f.time_ok for f in unread)
        warnings.append(f"{len(unread)} مورد از بازدیدها را نتوانستم دقیق بخوانم (فونت یا اندازه‌ی متفاوت)"
                        + ("؛ جایش از روی ساعت معلوم است و باز هم عوض می‌شود." if fixable == len(unread) else "."))
    good = [f for f in footers if f.readable]
    conf = 0.0
    if good:
        conf = float(np.clip(1.0 - (np.mean([f.rms for f in good]) / (0.9 * max(s_fit, 0.3))), 0, 1))
        conf *= len(good) / len(footers)
    return Analysis(img_bgr, s_fit, footers, warnings, conf)
