"""Teach the bot's brain the groups (sections) of price lists.

    python tools/teach_groups.py            # teach from the lists in «pdf new.zip», check, write the seed
    python tools/teach_groups.py --check    # only measure: the current seed, and on list formats held out
    python tools/teach_groups.py --photos 4 # also check photos: up to 4 pages of every list as screenshots, read by OCR

The checked answers - which group every price of every list belongs to - are in
tools/group_truth.json. They were made by the bot's two independent readings
(the table grid of the Arizon template and the brain's own lines), every place
where the two disagreed settled by looking at the page itself.

Each line of a page that holds no price is a lesson: a title of the group the
next price belongs to (1) or not (0). The model learns from all of them; the
check also trains without one family of lists at a time and tests on it, to
see how the brain does on list formats it has never seen.

Only the PDFs teach. Photos are checked, not taught: lessons from lines as the
OCR reads them made the brain worse on PDFs of formats it had not seen (100% ->
96.7%) for a small gain on photos (75% -> 77.5%).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "pricebot" / "brain" / "seed"
WORK = Path(tempfile.mkdtemp(prefix="teach_groups_"))
# the lists are read with a copy of the seed brain, never the seed itself, and without Gemini
shutil.copytree(SEED, WORK / "brain")
os.environ["BRAIN_DIR"] = str(WORK / "brain")
os.environ["DATA_DIR"] = str(WORK / "data")
os.environ["GEMINI_API_KEY"] = ""
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.setdefault("LOG_LEVEL", "ERROR")
sys.path.insert(0, str(ROOT))

from pricebot import brain, pipeline  # noqa: E402
from pricebot.brain import groups as G  # noqa: E402
from pricebot.brain import layout as L  # noqa: E402
from pricebot.brain import ocr  # noqa: E402

TRUTH = ROOT / "tools" / "group_truth.json"
ZIP = ROOT / "pdf new.zip"


def _lists() -> dict[str, Path]:
    """The PDFs of the zip, by their real (Persian) file names."""
    out = {}
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(".pdf"):
                continue
            name = info.filename
            try:
                name = name.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            name = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), Path(name).name)
            path = WORK / "lists" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(info))
            out[name] = path
    return out


def _key(page: int, bbox) -> tuple[int, int, int]:
    return page, round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)


class Page:
    """One list read once: its prices, the pages as the group finder sees them, the truth."""

    def __init__(self, name: str, path: Path, truth: dict):
        t = time.time()
        self.name = name
        self.family = truth["family"]
        self.analysis = pipeline.analyze(path, name, WORK / "work" / str(abs(hash(name))))
        names = truth["groups"]
        want = {(p, x, y): (names[g] if g >= 0 else "") for p, x, y, g in truth["prices"]}
        self.want = {it.id: want.get(_key(it.page, it.bbox)) for it in self.analysis.items}
        missing = sum(1 for v in self.want.values() if v is None)
        print(f"read {name[:50]:50s} {len(self.analysis.items):4d} prices ({missing} not in the truth) "
              f"{time.time() - t:.1f}s", flush=True)

    def outs(self, model: G.GroupModel):
        return pipeline.group_pages(self.analysis, model)

    def samples(self, model: G.GroupModel) -> list[tuple[list[str], int]]:
        if hasattr(self, "_samples"):
            return self._samples
        ins, outs = self.outs(model)
        by_page: dict[int, list[tuple[float, str]]] = {}
        for it in self.analysis.items:
            w = self.want.get(it.id)
            by_page.setdefault(it.page, []).append(((it.bbox[1] + it.bbox[3]) / 2, "?" if w is None else w))
        for v in by_page.values():
            v.sort()
        out = []
        for k, o in enumerate(outs):
            def names_below(line, k=k):
                for j in range(k, len(outs)):
                    for y, g in by_page.get(j, []):
                        if j == k and y <= line.y1:
                            continue
                        return None if g == "?" else g
                return None
            out += G.labels(o, names_below)
        self._samples = out
        return out

    def score(self, model: G.GroupModel) -> tuple[int, int, list]:
        ins, outs = self.outs(model)
        ok = n = 0
        wrong = []
        for it in self.analysis.items:
            want = self.want.get(it.id)
            if want is None or outs[it.page].ctx is None:
                continue
            got = outs[it.page].group_of(it.bbox)
            n += 1
            if (not want and not got) or (want and got and G.same_group(want, got)):
                ok += 1
            else:
                wrong.append((it.page + 1, it.text, want, got))
        return ok, n, wrong


class Photo:
    """Pages of one list as screenshots read by OCR - the prices taken where the list has
    them (the photo reading of groups is taught and checked, not the reading of prices)."""

    def __init__(self, page: Page, path: Path, n: int):
        import numpy as np
        from PIL import Image

        self.name, self.family = page.name, page.family
        self.pages = []
        doc = pymupdf.open(path)
        items = page.analysis.items
        for p in sorted({it.page for it in items})[:n]:
            z = 2000 / max(doc[p].rect.width, doc[p].rect.height)
            rgb = np.asarray(Image.fromarray(pipeline.render_page(doc[p], z)).convert("RGB")).copy()
            pdoc = L.from_ocr(ocr.glyph_words(rgb, ocr.page_words(rgb), brain.BRAIN.glyphs), rgb.shape[1], rgb.shape[0])
            where = [((it.bbox[0] + it.bbox[2]) / 2 * z, (it.bbox[1] + it.bbox[3]) / 2 * z, page.want.get(it.id))
                     for it in items if it.page == p]
            sel, want = set(), {}
            for i, t in enumerate(pdoc.nums):
                cx, cy = t.xc, t.yc
                best = min(where, key=lambda w: (w[0] - cx) ** 2 + (w[1] - cy) ** 2)
                if (best[0] - cx) ** 2 + (best[1] - cy) ** 2 <= (6 * z) ** 2 and best[2] is not None:
                    sel.add(i)
                    want[i] = best[2]
            self.pages.append((pdoc, sel, want, rgb))
        print(f"photos {self.name[:50]:50s} {len(self.pages)} pages, {sum(len(p[1]) for p in self.pages)} prices",
              flush=True)

    def outs(self, model: G.GroupModel):
        if not hasattr(self, "_ins"):
            # (each line of a photo is read once, however often the pages are looked at)
            def cached(rgb):
                namer, memo = pipeline._photo_namer(rgb), {}

                def read(box, side=0.3):
                    k = (tuple(round(v, 1) for v in box), side)
                    if k not in memo:
                        memo[k] = namer(box, side)
                    return memo[k]
                return read
            self._ins = [(d, sel, G.image_graphics(rgb), cached(rgb)) for d, sel, _, rgb in self.pages]
        return G.find([G.PageIn(d, sel, g, n) for d, sel, g, n in self._ins], model)

    def score(self, model: G.GroupModel) -> tuple[int, int, list]:
        """Right when the price is in its group, the name read well enough to tell (or no group)."""
        outs = self.outs(model)
        ok = n = 0
        wrong = []
        for o, (d, sel, want, _) in zip(outs, self.pages):
            for i in sel:
                got, w = o.group_of(d.nums[i].box), want[i]
                n += 1
                if (not w and not got) or (w and got and G.likeness(w, got) >= 0.55):
                    ok += 1
                else:
                    wrong.append((0, d.nums[i].text, w, got))
        return ok, n, wrong


def train(pages: list, epochs: int = 40) -> tuple[G.GroupModel, list]:
    random.seed(1)
    model = G.GroupModel()
    data = [s for p in pages for s in p.samples(model)]
    model.train(data, epochs=epochs)
    return model, data


def report(pages: list[Page], model: G.GroupModel, title: str) -> float:
    ok = tot = 0
    for p in pages:
        o, n, wrong = p.score(model)
        ok += o
        tot += n
        if wrong:
            print(f"   {p.name[:50]:50s} {o}/{n}", *(f"\n      p{w[0]} {w[1]}: want «{w[2]}» got «{w[3]}»" for w in wrong[:4]))
    print(f"{title}: {ok}/{tot} prices in the right group ({100 * ok / max(1, tot):.2f}%)")
    return ok / max(1, tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="measure only, do not write the seed")
    ap.add_argument("--no-cv", action="store_true", help="skip the held-out check")
    ap.add_argument("--photos", type=int, default=0, help="also check photos: this many pages of every list")
    a = ap.parse_args()
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    files = _lists()
    pages = [Page(name, files[name], t) for name, t in truth.items() if name in files]
    if a.photos:
        ocr.available()
        pages += [Photo(p, files[p.name], a.photos) for p in list(pages)]
    print(f"{len(pages)} lists read\n")

    pdfs = [p for p in pages if isinstance(p, Page)]
    photos = [p for p in pages if isinstance(p, Photo)]
    kinds = [("PDF", pdfs)] + ([("photos", photos)] if photos else [])

    seed = json.loads((SEED / "brain.json").read_text(encoding="utf-8")).get("groups") or {}
    for kind, ps in kinds:
        report(ps, G.GroupModel(seed.get("w"), seed.get("g2")), f"seed brain now ({kind})")
    if not a.no_cv:
        score = {kind: [0, 0] for kind, _ in kinds}
        for fam in sorted({p.family for p in pages}):
            model, _ = train([p for p in pdfs if p.family != fam])
            line = []
            for kind, ps in kinds:
                fo = fn = 0
                for p in ps:
                    if p.family == fam:
                        o, n, _ = p.score(model)
                        fo, fn = fo + o, fn + n
                score[kind][0] += fo
                score[kind][1] += fn
                line.append(f"{kind} {fo}/{fn}")
            print(f"   held out {fam:10s}: {'  '.join(line)}")
        for kind, (ok, tot) in score.items():
            print(f"formats never seen, {kind} (each family held out): {ok}/{tot} ({100 * ok / max(1, tot):.2f}%)")
    model, data = train(pdfs)
    print(f"\n{len(data)} lessons ({sum(y for _, y in data)} group titles)")
    for kind, ps in kinds:
        report(ps, model, f"taught brain ({kind})")
    if a.check:
        return
    brain = json.loads((SEED / "brain.json").read_text(encoding="utf-8"))
    brain["groups"] = {"w": model.w, "g2": model.g2,
                       "edition": f"{time.strftime('%Y-%m-%d')}-{len(pdfs)}lists-{len(data)}"}
    (SEED / "brain.json").write_text(json.dumps(brain, ensure_ascii=False), encoding="utf-8")
    (SEED / "group_samples.jsonl").write_text(
        "".join(json.dumps({"f": f, "y": y}, ensure_ascii=False) + "\n" for f, y in data), encoding="utf-8")
    print(f"seed written: {SEED / 'brain.json'} (groups edition {brain['groups']['edition']})")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
