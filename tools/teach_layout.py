"""Teach the brain a photo list that is printed as a ruled table (a person or
Claude acting as the teacher).

    python tools/teach_layout.py tools/layouts/edward_lent.json           # teach + check
    python tools/teach_layout.py tools/layouts/edward_lent.json --check   # check only

The JSON holds the checked answer for one photo of the list: the photo's file
name (next to the JSON), and every price - row number, product, price exactly
as printed, and its box in the photo's pixels - grouped by the list's own
sections. See tools/layouts/edward_lent.json.

The lesson goes to the seed memory (pricebot/brain/seed/brain.json, «layouts»),
shipped with the code and taken into the bot's memory on its next start. From
then on a photo of the same list - any size, a screenshot, re-compressed by
Telegram - is recognised by its table lines: every price is found in its cell,
with its group and product, and only its digits are read.

The check shows, for the photo and for altered copies of it (smaller, larger,
blurred, JPEG), whether the list is recognised and every price found exactly
where the answer says.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "pricebot" / "brain" / "seed" / "brain.json"
os.environ["BRAIN_MODE"] = "off"            # the seed is edited here directly, not through the bot's memory
os.environ["GEMINI_API_KEY"] = ""
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from pricebot import config, pipeline  # noqa: E402
from pricebot.brain import grid as G  # noqa: E402
from pricebot.brain import layout as L  # noqa: E402

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
# (name, scale, blur radius, JPEG quality or None)
VARIANTS = [("as taught", 1.0, 0.0, None), ("telegram", 1.185, 0.0, 85), ("phone", 1.5, 0.6, 75),
            ("large", 2.0, 0.7, 80), ("small", 0.8, 0.0, 70), ("blurry", 1.2, 1.0, 60)]


def load(path: Path) -> tuple[dict, np.ndarray, list[dict]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    im, _, _ = pipeline.load_image(path.parent / spec["image"])
    prices = []
    for g in spec["groups"]:
        for row, name, price, box in g["rows"]:
            prices.append({"box": tuple(float(v) for v in box), "pol": "dark", "text": price,
                           "label": f"{str(row).translate(FA)} - {name}", "group": g["name"],
                           "column": spec.get("column", "")})
    return spec, np.asarray(im).copy(), prices


def variant(rgb: np.ndarray, scale: float, blur: float, quality: int | None) -> np.ndarray:
    im = Image.fromarray(rgb)
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if quality:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(im).copy()


def check(layout: dict, rgb: np.ndarray, prices: list[dict]) -> bool:
    good = True
    for name, scale, blur, q in VARIANTS:
        img = variant(rgb, scale, blur, q)
        t = time.time()
        grid = G.find(img)
        sim = G.similarity(layout["grid"], grid)
        placed = G.place(layout, grid, img) if sim >= G.MATCH else None
        right = 0
        if placed is not None:
            truth = {(p["label"], p["group"]): tuple(v * scale for v in p["box"]) for p in prices}
            for got in placed[1]:
                want = truth.get((got.rec["label"], got.rec["group"]))
                if want is not None and L.box_iou(got.box, want) >= 0.6:
                    right += 1
        ok = placed is not None and right == len(prices)
        good &= ok
        print(f"  {name:10} {img.shape[1]:>4}×{img.shape[0]:<4} grid match {sim:.3f}  "
              f"prices found in their cells: {right}/{len(prices)}  ({time.time() - t:.2f}s)  {'✓' if ok else '✗'}")
    return good


def teach(path: Path, only_check: bool) -> int:
    spec, rgb, prices = load(path)
    grid = G.find(rgb)
    print(f"{spec['name']}: {len(prices)} prices in {len(spec['groups'])} groups; "
          f"table lines {len(grid.h)} across, {len(grid.v)} down")
    record = G.describe(rgb, grid, prices)
    if record is None:
        print("✗ not a clean ruled table: a price outside its own cell, a cell holding more than "
              "its price, or a number in a price column the answer does not list")
        return 1
    layout = {"id": spec["id"], "source": "photo", "name": spec["name"], "created": time.time(),
              "updated": time.time(), "streak": config.BRAIN_TRUST_AFTER, "checks": 1, "agree": 1, "seen": 1,
              "taught_by": "claude", "currency": spec.get("currency", ""), **record}
    print("  reading order: " + " | ".join(dict.fromkeys(p["group"] for p in layout["prices"])))
    print("check:")
    ok = check(layout, rgb, prices)
    if not ok:
        print("✗ the check failed: not written")
        return 1
    if only_check:
        return 0
    data = json.loads(SEED.read_text(encoding="utf-8"))
    data["layouts"] = [x for x in data.get("layouts", []) if x.get("id") != layout["id"]] + [layout]
    SEED.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"✓ written to {SEED.relative_to(ROOT)} as «{layout['id']}» ({len(layout['prices'])} prices)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--check", action="store_true", help="only check, do not write the seed")
    a = ap.parse_args()
    sys.exit(teach(a.file, a.check))


if __name__ == "__main__":
    main()
