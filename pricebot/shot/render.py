"""Redraw a footer with another view count: calibrate on the real time stamp, erase, draw, verify."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .bank import EYE, Bank, load_bank
from .detect import VIEWS_RE, Analysis, Comp, Footer, ShotError, _gray, _warp, analyze, native_layout

_FA = {**{ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}, **{ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")},
       ord("٫"): ".", ord("٬"): "", ord(","): "", ord("،"): ""}


# ------------------------------------------------------------- the number --

def format_views(n: int) -> str:
    """The way Telegram prints a view count: 999, 1K, 1.2K, 27K, 24.6K (one decimal, cut not rounded)."""
    if n < 0:
        raise ShotError("عدد بازدید نمی‌تواند منفی باشد.")
    if n < 1000:
        return str(n)
    for div, suffix in ((1_000_000, "M"), (1000, "K")):
        if n >= div:
            tenths = n * 10 // div
            if tenths >= 10000:      # 1000K -> 1M
                continue
            whole, frac = divmod(tenths, 10)
            return f"{whole}{suffix}" if frac == 0 else f"{whole}.{frac}{suffix}"
    return str(n)


def parse_value(text: str) -> str:
    """'2', '۲', '2500', '2.5K' -> the text to draw ('2', '2', '2.5K', '2.5K')."""
    t = (text or "").strip().translate(_FA).replace(" ", "").upper()
    t = t.replace("هزار", "K")
    m = re.fullmatch(r"(\d+(?:\.\d)?)([KM]?)", t)
    if not m:
        raise ShotError("عدد را درست نفهمیدم. مثلاً بنویس: 2 یا 2500 یا 2.5K")
    num, suffix = m.groups()
    if suffix:
        return num + suffix
    return format_views(int(num))


# --------------------------------------------------------- pixel plumbing --

def _fill(img: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Rebuild the background under `mask` (inside box): interpolate along every row, then smooth."""
    x0, y0, x1, y1 = box
    out = img.astype(np.float32).copy()
    H, W = mask.shape
    for y in range(y0, y1):
        row = mask[y]
        x = x0
        while x < x1:
            if not row[x]:
                x += 1
                continue
            a = x
            while x < x1 and row[x]:
                x += 1
            b = x                      # run is [a, b)
            la, rb = max(0, a - 1), min(W - 1, b)
            left, right = out[y, la], out[y, rb]
            n = b - a
            for k in range(n):
                t = (k + 1) / (n + 1)
                out[y, a + k] = left * (1 - t) + right * t
    _relax(out, mask, box)
    return out


def _relax(out: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], steps: int = 400) -> None:
    """Smooth the rebuilt pixels in place: each becomes the mean of its four neighbours (the real pixels around
    stay fixed). On JPEG photos the row ends pick up compression noise that row interpolation stretches into
    streaks; this spreads it out in both directions instead (a membrane over the hole)."""
    H, W = mask.shape
    x0, y0, x1, y1 = max(1, box[0]), max(1, box[1]), min(W - 1, box[2]), min(H - 1, box[3])
    m = mask[y0:y1, x0:x1]
    if not m.any():
        return
    sub = out[y0 - 1:y1 + 1, x0 - 1:x1 + 1]
    for _ in range(steps):
        avg = (sub[:-2, 1:-1] + sub[2:, 1:-1] + sub[1:-1, :-2] + sub[1:-1, 2:]) * 0.25
        inner = sub[1:-1, 1:-1]
        inner[m] = avg[m]


def _canvas(items: list[tuple[np.ndarray, int, int]]) -> tuple[np.ndarray, int, int]:
    """Paste glyph bitmaps (alpha, native x, native y) on one native canvas."""
    x0 = min(x for _, x, _ in items) - 3
    y0 = min(y for _, _, y in items) - 3
    x1 = max(x + a.shape[1] for a, x, _ in items) + 3
    y1 = max(y + a.shape[0] for a, _, y in items) + 3
    can = np.zeros((y1 - y0, x1 - x0), np.float32)
    for a, x, y in items:
        sub = can[y - y0:y - y0 + a.shape[0], x - x0:x - x0 + a.shape[1]]
        np.maximum(sub, a, out=sub)
    return can, x0, y0


# ------------------------------------------------------------------ result --

@dataclass
class Result:
    img: np.ndarray
    changed: int = 0
    verified: int = 0
    unchecked: int = 0           # edited footers the automatic re-read could not find again (e.g. faded under a header)
    notes: list[str] = field(default_factory=list)
    old_texts: list[str] = field(default_factory=list)
    fit: float = 0.0             # how closely the learned shapes reproduce the real time stamps (0..1)


# ------------------------------------------------------------------ editing --

def _native_lefts(foot: Footer, bank: Bank, s: float) -> tuple[list[int], list[int]]:
    """Integer native left edges of the views / time characters, recovered from where they really are."""
    def one(c: Comp) -> int:
        g = bank.glyphs[c.char]
        return int(round((c.cx - foot.tx) / s - g.cx))
    return [one(c) for c in foot.views if c.char in bank.glyphs], [one(c) for c in foot.time]


def _calibrate(img: np.ndarray, bgfill: np.ndarray, foot: Footer, bank: Bank, s: float,
               time_lefts: list[int]) -> tuple[np.ndarray, float, float, float, float]:
    """Find the colour (and for scaled screenshots the sub-pixel shift and softness) by redrawing the real
    time stamp and comparing. Returns (colour BGR, dx, dy, sigma, explained 0..1)."""
    items = []
    for c, x in zip(foot.time, time_lefts):
        g = bank.glyphs[c.char]
        items.append((g.a, int(x), int(round(g.dy))))
    canvas, cx0, cy0 = _canvas(items)
    bx0, by0, bx1, by1 = foot.band
    tx0 = max(bx0, foot.time[0].x0 - 4)
    tx1 = min(img.shape[1], foot.time[-1].x1 + 4)
    win = (tx0, by0, tx1, by1)
    P = img[by0:by1, tx0:tx1].astype(np.float32)
    B = bgfill[by0:by1, tx0:tx1]
    native = abs(s - 1.0) < 2e-3
    shifts = [0.0] if native else [-0.5, -0.25, 0.0, 0.25, 0.5]
    sigmas = [0.0] if native else [0.0, 0.3, 0.5, 0.7]
    best = None
    total = float(((P - B) ** 2).sum()) or 1.0
    for dx in shifts:
        for dy in shifts:
            base = _warp(canvas, cx0, cy0, s, foot.tx + dx, foot.ty + dy, win)
            for sg in sigmas:
                R = cv2.GaussianBlur(base, (0, 0), sg) if sg > 0 else base
                Rr = R[..., None]
                num = ((P - B * (1 - Rr)) * Rr).sum(axis=(0, 1))
                den = float((R * R).sum()) or 1.0
                fg = num / den
                res = float(((P - B * (1 - Rr) - fg * Rr) ** 2).sum())
                if best is None or res < best[0]:
                    best = (res, fg, dx, dy, sg)
    res, fg, dx, dy, sg = best
    explained = float(np.clip(1 - res / total, 0, 1))
    return np.clip(fg, 0, 255), dx, dy, sg, explained


def _ink_mask(gray: np.ndarray, foot: Footer, boxes: list[tuple[int, int, int, int]], pad: int, thr: float
              ) -> np.ndarray:
    H, W = gray.shape
    m = np.zeros((H, W), bool)
    dev = foot.sign * (gray - foot.bg) / max(foot.contrast, 1.0)
    for x0, y0, x1, y1 in boxes:
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        m[Y0:Y1, X0:X1] |= dev[Y0:Y1, X0:X1] > thr
    return m


def _edit_footer(img: np.ndarray, out: np.ndarray, foot: Footer, text: str, bank: Bank, s: float) -> Result | None:
    """Draw `text` as the view count of one footer into `out` (the working copy of the screenshot)."""
    H, W = img.shape[:2]
    gray = _gray(img)
    bx0, by0, bx1, by1 = foot.band
    time_x0 = foot.time[0].x0
    views_l, time_l = _native_lefts(foot, bank, s)
    # the old count only decides how far the eye moves; an unsure digit is close enough for that
    old_text = foot.views_text if foot.readable else foot.views_guess if VIEWS_RE.match(foot.views_guess) else ""
    anchor = time_l[0]

    # --- the new layout, in native pixels, hanging on the real first character of the time
    new_lay = native_layout(bank, text, foot.time_text)
    shift = anchor - new_lay["time"][0]
    new_x = [int(round(x + shift)) for x in new_lay["views"]]
    if old_text:
        old_lay = native_layout(bank, old_text, foot.time_text)
        d_eye = new_lay["eye"] - old_lay["eye"]
    else:
        d_eye = new_lay["eye"] + shift - (foot.eye[0] - foot.tx) / s
    d_eye_px = s * d_eye
    move_eye = abs(d_eye_px) >= 0.35

    # --- colour / softness from the real time stamp
    ink_all = [c for c in foot.views + foot.time]
    boxes_all = [(c.x0, c.y0, c.x1, c.y1) for c in ink_all] + [foot.eye]
    pad = max(1, int(round(1.5 * s)))
    mask_all = _ink_mask(gray, foot, boxes_all, pad, 0.10)
    mask_all = cv2.dilate(mask_all.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    wx0 = max(0, foot.eye[0] - 6 - int(np.ceil(abs(d_eye_px))))
    bgfill = _fill(img, mask_all, (wx0, by0, min(W, bx1), by1))
    fg, dx, dy, sigma, explained = _calibrate(img, bgfill, foot, bank, s, time_l)

    # --- erase the old numbers (and the eye when it has to move)
    erase_boxes = [(c.x0, c.y0, c.x1, c.y1) for c in foot.views]
    if move_eye:
        erase_boxes.append(foot.eye)
    emask = _ink_mask(gray, foot, erase_boxes, pad, 0.10)
    if not move_eye:
        ex0, ey0, ex1, ey1 = foot.eye
        emask[:, :ex1 + 1] = False
    emask[:, time_x0 - 1:] = False
    emask = cv2.dilate(emask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    emask[:, time_x0 - 1:] = False
    if not move_eye:
        emask[:, :foot.eye[2] + 1] = False
    base = bgfill.copy()
    work = out.astype(np.float32)
    work[emask] = base[emask]

    # --- the new characters (and the eye, redrawn from the learned shape so it stays as sharp as the digits)
    items = []
    for ch, x in zip(text, new_x):
        g = bank.glyphs[ch]
        items.append((g.a, x, int(round(g.dy))))
    if move_eye:
        eg = bank.glyphs[EYE]
        items.append((eg.a, int(round(new_lay["eye"] + shift)), int(round(eg.dy))))
    canvas, cx0, cy0 = _canvas(items)
    win = (wx0, by0, min(W, time_x0), by1)
    R = _warp(canvas, cx0, cy0, s, foot.tx + dx, foot.ty + dy, win)
    if sigma > 0:
        R = cv2.GaussianBlur(R, (0, 0), sigma)
    layers = [R]
    A = np.clip(sum(layers), 0, 1)[..., None]
    reg = work[win[1]:win[3], win[0]:win[2]]
    work[win[1]:win[3], win[0]:win[2]] = reg * (1 - A) + fg.reshape(1, 1, 3) * A
    out[:] = np.clip(work + 0.5, 0, 255).astype(np.uint8)
    res = Result(out)
    res.fit = explained
    return res


def apply_views(an: Analysis, text: str | dict[int, str], bank: Bank | None = None, only: list[int] | None = None,
                force: bool = False) -> Result:
    """Set view counts: `text` for every footer (or for `only`, 0-based), or {footer index: text} for a
    different number on each one."""
    bank = bank or load_bank()
    if isinstance(text, dict):
        targets = dict(text)
    else:
        targets = {i: text for i in range(len(an.footers)) if only is None or i in only}
    for t in targets.values():
        if not t:
            raise ShotError("عدد جدید را بگو.")
        miss = bank.missing(t)
        if miss:
            raise ShotError("این نویسه‌ها هنوز در نمونه‌های من نیست: " + " ".join(miss)
                            + "\nفقط رقم، نقطه و K را می‌توانم بنویسم؛ برای بقیه چند اسکرین‌شات نمونه بفرست.")
    out = an.img.copy()
    result = Result(out)
    fits = []
    edited: dict[int, str] = {}
    for i, foot in enumerate(an.footers):
        if i not in targets:
            continue
        t = targets[i]
        if not foot.time_ok:
            result.notes.append(f"ساعت کنار بازدید شماره {i + 1} خوانده نشد؛ دست نخورد.")
            continue
        if foot.readable and foot.views_text == t and not force:
            continue
        r = _edit_footer(an.img, out, foot, t, bank, an.scale)
        if r is None:
            continue
        result.old_texts.append(foot.views_text)
        fits.append(r.fit)
        for n in r.notes:
            if n not in result.notes:
                result.notes.append(n)
        edited[i] = t
    result.changed = len(edited)
    result.fit = float(np.mean(fits)) if fits else 0.0
    poor = sum(1 for x in fits if x < 0.80)
    if fits and (result.fit < 0.85 or poor / len(fits) > 0.3):
        result.notes.append("شکل ارقام این عکس با نمونه‌ها کمی فرق دارد؛ نتیجه را از نزدیک نگاه کن.")
    if edited:
        try:
            again = analyze(out, bank)
        except ShotError:
            again = None
        bad = []
        for i, t in edited.items():
            if again is None:
                result.unchecked += 1
                continue
            foot = an.footers[i]
            near = [g for g in again.footers if abs(g.eye[1] - foot.eye[1]) <= 6]
            if not near:
                result.unchecked += 1
            elif near[0].readable and near[0].views_text == t:
                result.verified += 1
            else:
                bad.append(i + 1)
        if bad:
            result.notes.append(f"بعد از ساخت، بازدید شماره {'، '.join(map(str, bad))} درست خوانده نشد؛ "
                                "خروجی را یک بار نگاه کن.")
    return result


def preview(an: Analysis) -> np.ndarray:
    """The screenshot with a green box and a number around every view count that was found."""
    img = an.img.copy()
    for i, f in enumerate(an.footers, 1):
        x0 = f.views[0].x0 - 3
        x1 = f.views[-1].x1 + 3
        y0 = min(c.y0 for c in f.views) - 3
        y1 = max(c.y1 for c in f.views) + 3
        color = (60, 200, 60) if f.readable else (40, 160, 240)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        label = f"{f.views_text or '?'}"
        cv2.putText(img, str(i), (max(0, x0 - 4), max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img
