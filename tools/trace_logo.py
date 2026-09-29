"""Trace the Arizon logo (pricebot/arizon/assets/arizon-logo.jpg) into vector
shapes, so the template can draw it sharp at any size and on any background.

    python tools/trace_logo.py

Writes pricebot/arizon/assets/logo.json: for each part ("emblem", "diamond",
"word") a list of closed polygons in logo units (1 unit = 1 source pixel),
filled with the even-odd rule so the holes of letters stay open.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "pricebot" / "arizon" / "assets"
SCALE = 3          # trace on an upscaled copy: smoother edges from the JPEG
EPSILON = 0.9      # polygon simplification, in upscaled pixels


def _polys(mask: np.ndarray) -> list[list[list[float]]]:
    big = cv2.resize(mask.astype(np.uint8) * 255, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (5, 5), 0)
    _, bw = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 40 * SCALE * SCALE:
            continue
        c = cv2.approxPolyDP(c, EPSILON, True)
        out.append([[round(float(p[0][0]) / SCALE, 2), round(float(p[0][1]) / SCALE, 2)] for p in c])
    return out


def main() -> None:
    rgb = np.asarray(Image.open(ASSETS / "arizon-logo.jpg").convert("RGB")).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yellow = (r > 150) & (g > 150) & (b < 140)
    red = (r > 150) & (g < 130) & (b < 130)
    rows = np.nonzero(yellow.any(axis=1))[0]
    # the emblem and the word are separated by an empty band of rows
    gaps = [y for y in range(rows[0], rows[-1]) if not yellow[y].any() and not red[y].any()]
    split = int(np.median(gaps)) if gaps else rgb.shape[0] // 2
    emblem, word = yellow.copy(), yellow.copy()
    emblem[split:] = False
    word[:split] = False
    diamond = red.copy()
    diamond[split:] = False
    parts = {"emblem": _polys(emblem), "diamond": _polys(diamond), "word": _polys(word)}
    boxes = {}
    for name, polys in parts.items():
        pts = np.array([p for poly in polys for p in poly])
        boxes[name] = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    data = {"source": "arizon-logo.jpg", "size": [rgb.shape[1], rgb.shape[0]],
            "colors": {"yellow": "#FFF112", "red": "#EC3237"}, "boxes": boxes, "parts": parts}
    (ASSETS / "logo.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print({k: len(v) for k, v in parts.items()}, {k: [round(x) for x in v] for k, v in boxes.items()})


if __name__ == "__main__":
    main()
