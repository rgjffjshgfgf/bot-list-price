"""Teach the bot the footer font of Telegram screenshots (view counts and times).

    python tools/learn_shots.py PATH [PATH ...]            # folders, zip files or images; adds to what is known
    python tools/learn_shots.py PATH --fresh                # forget the old shapes and learn only from PATH
    python tools/learn_shots.py --check PATH                # redraw every footer's own text and report the error

PATH may hold any number of iPhone screenshots of Telegram channels at full size (the more different
numbers and times they show, the better). Screenshots of another size are not used for the shapes, only to
measure the space between the view count and the time.

Needs Tesseract once (it reads the text of each footer so the characters can be labelled).
The result goes to pricebot/shot/assets/ and ships with the code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pricebot.shot import analyze, apply_views, learn  # noqa: E402
from pricebot.shot.bank import Bank  # noqa: E402


def check(paths: list[Path], bank: Bank) -> None:
    errs = []
    for name, img in learn.load_images(paths):
        try:
            an = analyze(img, bank)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: {exc}")
            continue
        for i, f in enumerate(an.footers):
            if not f.readable:
                print(f"  {name}: footer {i + 1} not readable")
                continue
            res = apply_views(an, f.views_text, bank, only=[i], force=True)
            x0, x1 = f.eye[0] - 8, f.time[0].x0
            y0, y1 = f.band[1], f.band[3]
            d = np.abs(res.img[y0:y1, x0:x1].astype(float) - img[y0:y1, x0:x1].astype(float))
            errs.append(float(d.mean() / f.contrast))
    if errs:
        print(f"{len(errs)} footers redrawn: mean error {np.mean(errs) * 100:.2f}% of the text contrast, "
              f"worst {max(errs) * 100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--fresh", action="store_true", help="do not keep the shapes that are already known")
    ap.add_argument("--check", action="store_true", help="only measure how well the known shapes redraw these screenshots")
    args = ap.parse_args()
    old = Bank.load()
    if args.check:
        check(args.paths, old)
        return
    bank = learn.learn(args.paths, None if args.fresh else old)
    scaled = [(n, im) for n, im in learn.load_images(args.paths)]
    learn.refine_space(bank, scaled)
    bank.save()
    print("saved to pricebot/shot/assets/")


if __name__ == "__main__":
    main()
