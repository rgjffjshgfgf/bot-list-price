"""Build the glyph memory of the Arizon template (pricebot/arizon/assets/glyphs.json)
from price lists whose PDFs name their glyphs correctly.

    python tools/learn_glyphs.py lists/*.pdf [more.zip ...]
    python tools/learn_glyphs.py --context lists.zip      # also from the words they sit in

Every glyph outline that the PDFs always name the same way is remembered, so a
PDF that leaves the same glyph unnamed (the text layer reads "�") can be
read anyway. Run it again with more lists to grow the memory; entries are
merged with the existing file.

--context: a glyph no PDF names is worked out from the words it stands in:
the letter that turns those words («تی_و», «سان_», «بلبرین_») into words the
lists write elsewhere («تیگو», «سانگ», «بلبرینگ»). Only a clear winner, seen
in at least two different words, is taken.
"""
from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pricebot.arizon import glyphs  # noqa: E402


def _docs(paths: list[str]) -> list[pymupdf.Document]:
    out = []
    for p in paths:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.lower().endswith(".pdf"):
                        out.append(pymupdf.open("pdf", io.BytesIO(z.read(name))))
        else:
            out.append(pymupdf.open(p))
    return out


_LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئآ"
_SAME = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه"})
_AR = re.compile("[\u0600-\u06ff]")


def _unnamed_keys(page: pymupdf.Page) -> dict[tuple[float, float], str]:
    """Origin of every glyph the page does not name -> its outline key."""
    xrefs = glyphs.page_fonts(page)
    fonts: dict[str, object] = {}
    keys: dict[tuple, str | None] = {}
    out = {}
    for sp in page.get_texttrace():
        for c in sp["chars"]:
            if not (c[0] == 0xFFFD or 0xE000 <= c[0] <= 0xF8FF):
                continue
            name = sp["font"]
            if name not in fonts:
                xref = xrefs.get(name) or xrefs.get(glyphs._font_name(name))
                fonts[name] = glyphs.load_font(page.parent, xref) if xref else None
            if fonts[name] is None:
                continue
            k = (name, c[1])
            if k not in keys:
                keys[k] = glyphs.outline_key(fonts[name], c[1])
            if keys[k]:
                out[(round(c[2][0], 1), round(c[2][1], 1))] = keys[k]
    return out


def _words(page: pymupdf.Page, mem: dict[str, str]) -> list[list]:
    """Words of a page in reading order; a glyph nobody names is ("?", outline key)."""
    from pricebot.pdftext import _lines, page_chars

    known = _unnamed_keys(page)
    out: list[list] = []
    for line in _lines(page_chars(page)):
        run: list = []

        def flush() -> None:
            if run:
                persian = any(isinstance(x, tuple) or _AR.match(x) for x in run)
                out.append(run[::-1] if persian else list(run))      # visual order -> reading order
            run.clear()
        prev = None
        for ch in line:
            if not ch.c.strip():
                flush()
                prev = None
                continue
            if prev is not None and ch.bbox.x0 - prev.bbox.x1 > 0.25 * ch.size:
                flush()
            if ch.c == "\ufffd" or "\ue000" <= ch.c <= "\uf8ff":
                k = known.get((round(ch.origin[0], 1), round(ch.origin[1], 1)))
                if k and k in mem:
                    run.append(unicodedata.normalize("NFKC", mem[k]).translate(_SAME))
                elif k and ch.bbox.width >= 0.08 * ch.size:
                    run.append(("?", k))
                elif not k:
                    run.append("\ufffd")
            else:
                run.append(unicodedata.normalize("NFKC", ch.c).translate(_SAME))
            prev = ch
        flush()
    return out


def learn_from_context(docs: list[pymupdf.Document], mem: dict[str, str]) -> dict[str, str]:
    vocab: Counter = Counter()
    pending = []
    for doc in docs:
        for page in doc:
            for w in _words(page, mem):
                if all(isinstance(x, str) for x in w):
                    s = "".join(w)
                    if "\ufffd" not in s and _AR.search(s) and len(s) >= 2:
                        vocab[s] += 1
                elif any(isinstance(x, tuple) for x in w):
                    pending.append(w)
    learned: dict[str, str] = {}
    while True:
        by_key: dict[str, set] = defaultdict(set)
        for w in pending:
            ks = {x[1] for x in w if isinstance(x, tuple) and x[1] not in learned}
            if len(ks) == 1 and "\ufffd" not in [x for x in w if isinstance(x, str)]:
                by_key[ks.pop()].add(tuple(w))
        new = {}
        for k, words in by_key.items():
            score: Counter = Counter()
            for w in words:
                for letter in _LETTERS:
                    if "".join(x if isinstance(x, str) else (learned.get(x[1]) or letter) for x in w) in vocab:
                        score[letter] += 1
            top = score.most_common(2) + [(None, 0), (None, 0)]
            (best, n), (_, n2) = top[0], top[1]
            if best and n >= 2 and n >= 3 * n2:
                new[k] = best
                ex = ["".join(x if isinstance(x, str) else best for x in w) for w in list(words)[:4]]
                print(f"   {k} -> {best}  ({n} words, e.g. {', '.join(ex)})")
        if not new:
            return learned
        learned.update(new)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--context"]
    if not args:
        sys.exit(__doc__)
    docs = _docs(args)
    learned = glyphs.learn(docs)
    old = glyphs.memory()
    if "--context" in sys.argv:
        learned.update(learn_from_context(docs, {**old, **learned}))
    merged = {**old, **learned}
    glyphs.MEMORY.write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                             encoding="utf-8")
    print(f"{len(learned)} outlines learned, {len(merged)} in memory ({len(merged) - len(old)} new)")


if __name__ == "__main__":
    main()
