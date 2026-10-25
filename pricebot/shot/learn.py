"""Learn the footer font from real screenshots (iPhone Telegram, native resolution).

Every footer gets its text read once with Tesseract; after that the characters are cut out, averaged and stored
together with the spacing between neighbours. Nothing here runs while the bot answers users."""
from __future__ import annotations

import collections
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import detect
from .bank import ASSETS, EYE, Bank, make_glyph

LABEL_RE = re.compile(r"^(\d{1,3}(?:\.\d)?[KM]?)(\d\d:\d\d)")
SUPPORT = 0.05


def _seed_bank() -> Bank:
    seed = cv2.imread(str(ASSETS / "eye_seed.png"), 0)
    if seed is None:
        raise SystemExit("assets/eye_seed.png is missing")
    a = seed.astype(np.float32)
    a = (a - a.min()) / max(1.0, float(a.max() - a.min()))
    if a[0, 0] > 0.5:       # light background: invert so the ink is high
        a = 1 - a
    b = Bank()
    b.glyphs[EYE] = make_glyph(a, 0.0)
    return b


def load_images(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []

    def add(name: str, data: bytes) -> None:
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out.append((name, im))

    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    add(f.name, f.read_bytes())
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in sorted(z.namelist()):
                    if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and "__MACOSX" not in n:
                        add(Path(n).name, z.read(n))
        elif p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            add(p.name, p.read_bytes())
    return out


def _ocr(gray: np.ndarray, foot: detect.Footer) -> str:
    import pytesseract
    x0 = foot.views[0].x0 - 3
    x1 = foot.time[-1].x1 + 4
    y0 = min(c.y0 for c in foot.views + foot.time) - 4
    y1 = max(c.y1 for c in foot.views + foot.time) + 4
    crop = gray[max(0, y0):y1, max(0, x0):x1]
    dev = np.clip(foot.sign * (crop - foot.bg) / max(foot.contrast, 1.0), 0, 1)
    img = (255 - 255 * dev).astype(np.uint8)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)
    txt = pytesseract.image_to_string(img, config="--psm 7 -c tessedit_char_whitelist=0123456789.:KM")
    return txt.replace(" ", "").strip()


def _tight(a: np.ndarray) -> tuple[np.ndarray, int, int]:
    ys, xs = np.where(a > SUPPORT)
    if len(xs) == 0:
        return a, 0, 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return a[y0:y1, x0:x1], int(x0), int(y0)


def learn(paths: list[Path], base: Bank | None = None, log=print) -> Bank:
    images = load_images(paths)
    if not images:
        raise SystemExit("no screenshots found")
    probe = base if (base and base.glyphs.get(EYE) is not None) else _seed_bank()
    inst: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    tops: dict[str, list[float]] = collections.defaultdict(list)
    pair_obs: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    sp_obs: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    used = skipped = 0
    for name, img in images:
        gray = detect._gray(img)
        s, eyes = detect.find_eyes(gray, probe)
        if not eyes:
            log(f"  {name}: no eye icon — skipped")
            skipped += 1
            continue
        if not 0.94 <= s <= 1.06:
            log(f"  {name}: scale {s:.2f} is not native — skipped (only full-size iPhone screenshots are learned)")
            skipped += 1
            continue
        tpl, m = detect._eye_template(probe)
        tw, th = int(round(tpl.shape[1] * s)), int(round(tpl.shape[0] * s))
        got = 0
        for sc, x, y in eyes:
            foot = detect._parse_footer(gray, probe, s, x, y, tw, th)
            if foot is None:
                continue
            text = _ocr(gray, foot)
            mt = LABEL_RE.match(text)
            comps = foot.views + foot.time
            if not mt:
                log(f"  {name}: unreadable footer ({text!r})")
                continue
            label = mt.group(1) + mt.group(2)
            if len(label) != len(comps):
                log(f"  {name}: {label!r} does not match {len(comps)} marks — skipped")
                continue
            # support boxes with the alpha of every mark
            marks = []
            for ch, c in zip(label, comps):
                box = (max(0, c.x0 - 2), max(0, c.y0 - 2), min(gray.shape[1], c.x1 + 2), min(gray.shape[0], c.y1 + 2))
                a = detect._alpha(gray, box, foot.bg, foot.sign, foot.contrast)
                tight, ox, oy = _tight(a)
                marks.append((ch, tight, box[0] + ox, box[1] + oy))
            ex0, ey0, ex1, ey1 = foot.eye
            ebox = (max(0, ex0 - 2), max(0, ey0 - 2), min(gray.shape[1], ex1 + 2), min(gray.shape[0], ey1 + 2))
            ea = detect._alpha(gray, ebox, foot.bg, foot.sign, foot.contrast)
            et, eox, eoy = _tight(ea)
            marks.insert(0, (EYE, et, ebox[0] + eox, ebox[1] + eoy))
            ref = [mk[3] for mk in marks if mk[0] in "1K47" and mk[1].size]
            if not ref:
                continue
            ref_top = float(np.median(ref))
            for ch, tight, left, top in marks:
                if tight.size == 0:
                    continue
                inst[ch].append(tight)
                tops[ch].append(top - ref_top)
            n_views = len(mt.group(1))
            for i in range(len(marks) - 1):
                a, b = marks[i], marks[i + 1]
                d = int(b[2] - a[2])
                key = f"{a[0]}|{b[0]}"
                # i indexes marks (eye first): views are marks[1..n_views], the time starts at n_views + 1
                if i == n_views:
                    sp_obs[key][d] += 1
                else:
                    pair_obs[key][d] += 1
            got += 1
        used += 1
        log(f"  {name}: {got} footer(s)")
    new = Bank()
    new.images = (base.images if base else 0) + used
    # the average mark per character (marks are drawn pixel-snapped, so equal marks have equal shapes)
    for ch, lst in inst.items():
        shapes = collections.Counter(a.shape for a in lst)
        shape = shapes.most_common(1)[0][0]
        same = np.stack([a for a in lst if a.shape == shape])
        med = np.median(same, axis=0)
        dev = float(np.mean([np.abs(a - med).mean() for a in same]))
        if dev > 0.03:
            log(f"  ! '{ch}': the {len(same)} samples disagree (mean difference {dev:.3f}) — check the labels")
        new.glyphs[ch] = make_glyph(med, float(np.median(tops[ch])))
        new.counts[ch] = len(lst)
    if base:
        for ch, g in base.glyphs.items():
            if ch not in new.glyphs:
                new.glyphs[ch] = g
                new.counts[ch] = base.counts.get(ch, 0)
            else:
                new.counts[ch] = new.counts.get(ch, 0) + base.counts.get(ch, 0)
        for key, d in base.pair_obs.items():
            for dist, n in d.items():
                pair_obs[key][int(dist)] += n
        for key, d in base.sp_obs.items():
            for dist, n in d.items():
                sp_obs[key][int(dist)] += n
    new.pair_obs = {k: {str(d): n for d, n in v.items()} for k, v in pair_obs.items()}
    new.sp_obs = {k: {str(d): n for d, n in v.items()} for k, v in sp_obs.items()}
    one = new.glyphs.get("1")
    new.cap_h = float(one.h) if one else 23.0
    new.sp_samples = list(base.sp_samples) if base else []
    _fit_spacing(new)
    log(f"learned from {used} screenshot(s), skipped {skipped}; characters: "
        + " ".join(f"{c}×{new.counts.get(c, 0)}" for c in sorted(new.glyphs)))
    return new


def _fit_spacing(bank: Bank) -> None:
    """u[first] + v[second] (+ sp across the views/time space) for the distance between left ink edges."""
    rows: list[tuple[str, str, bool, float, int]] = []
    for table, space in ((bank.pair_obs, False), (bank.sp_obs, True)):
        for key, d in table.items():
            a, b = key.split("|")
            for dist, n in d.items():
                rows.append((a, b, space, float(dist), int(n)))
    firsts = sorted({r[0] for r in rows})
    seconds = sorted({r[1] for r in rows})
    ref = "0" if "0" in seconds else seconds[0]
    vs = [c for c in seconds if c != ref]
    names = [("u", c) for c in firsts] + [("v", c) for c in vs] + [("sp", "")]
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(rows), len(names)))
    y = np.zeros(len(rows))
    w = np.zeros(len(rows))
    for i, (a, b, space, dist, n) in enumerate(rows):
        A[i, idx[("u", a)]] = 1
        if b != ref:
            A[i, idx[("v", b)]] = 1
        if space:
            A[i, idx[("sp", "")]] = 1
        y[i] = dist
        w[i] = np.sqrt(n)
    sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    bank.u = {c: float(sol[idx[("u", c)]]) for c in firsts}
    bank.v = {c: float(sol[idx[("v", c)]]) for c in vs}
    bank.v[ref] = 0.0
    bank.sp = float(sol[idx[("sp", "")]])
    # a mark that is only ever followed by a space (K, M) cannot be told apart from the space itself:
    # only u + sp is known for it. Move the width of the space to the measured value when there is one.
    if len(bank.sp_samples) >= 5:
        sp_new = float(np.median(bank.sp_samples))
        followed = {key.split("|")[0] for key in bank.pair_obs}
        for c in firsts:
            if c not in followed:
                bank.u[c] += bank.sp - sp_new
        bank.sp = sp_new
    res = A @ sol - y
    log_err = float(np.sqrt(np.mean(res ** 2)))
    bank.counts["_spacing_rms"] = int(round(log_err * 100))


def refine_space(bank: Bank, images: list[tuple[str, np.ndarray]], log=print) -> int:
    """Measure the width of the space between the view count and the time on screenshots of any scale
    (native screenshots never show a digit in front of the space, only K)."""
    got = 0
    for name, img in images:
        try:
            an = detect.analyze(img, bank)
        except detect.ShotError:
            continue
        for f in an.footers:
            if not f.readable:
                continue
            c1, c2 = f.views[-1], f.time[0]
            if c1.char not in "0123456789":
                continue
            g1, g2 = bank.glyphs[c1.char], bank.glyphs[c2.char]
            d = (c2.cx - c1.cx) / an.scale - (g2.cx - g1.cx)
            bank.sp_samples.append(round(float(d - bank.u[c1.char] - bank.v[c2.char]), 2))
            got += 1
    if got:
        bank.sp_samples = bank.sp_samples[-400:]
        _fit_spacing(bank)
        log(f"space between views and time: {bank.sp:.2f} native px (from {got} footers)")
    return got
