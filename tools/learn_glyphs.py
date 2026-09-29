"""Build the glyph memory of the Arizon template (pricebot/arizon/assets/glyphs.json)
from price lists whose PDFs name their glyphs correctly.

    python tools/learn_glyphs.py lists/*.pdf [more.zip ...]

Every glyph outline that the PDFs always name the same way is remembered, so a
PDF that leaves the same glyph unnamed (the text layer reads "�") can be
read anyway. Run it again with more lists to grow the memory; entries are
merged with the existing file.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
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


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    learned = glyphs.learn(_docs(sys.argv[1:]))
    old = glyphs.memory()
    merged = {**old, **learned}
    glyphs.MEMORY.write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                             encoding="utf-8")
    print(f"{len(learned)} outlines learned, {len(merged)} in memory ({len(merged) - len(old)} new)")


if __name__ == "__main__":
    main()
