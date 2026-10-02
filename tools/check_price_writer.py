"""Check that every changed price really lands in the edited PDF.

    python tools/check_price_writer.py                 # the lists of «pdf new.zip» + made-up font cases
    python tools/check_price_writer.py list1.pdf ...   # these PDFs (and the made-up cases)

Each list is read as the bot reads it, every price raised 15% and the PDF
written as the bot writes it; then each new price is looked up in the written
page. The made-up cases are PDFs whose embedded fonts lack the digits the new
prices need (Excel's Calibri, DejaVu, a bold serif, Persian digits), and one
whose font is not embedded at all - the situations in which a price was once
erased without the new one being written.
Exit code 1 when any price is missing.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(tempfile.mkdtemp(prefix="check_writer_"))
os.environ["BRAIN_DIR"] = str(WORK / "brain")
os.environ["DATA_DIR"] = str(WORK / "data")
os.environ["GEMINI_API_KEY"] = ""
os.environ.setdefault("LOG_LEVEL", "ERROR")
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from pricebot import commands, fonts, pipeline  # noqa: E402
from pricebot.numfmt import to_latin_digits  # noqa: E402
from pricebot.outputs import Exporter  # noqa: E402

ASSETS = ROOT / "pricebot" / "arizon" / "assets"


def _made_up() -> list[Path]:
    """Lists written with few digits, so that the embedded subsets lack the new prices' digits."""
    lat = ["1,250,000", "2,150,000", "5,200,000", "1,000,500", "2,500,000", "1,520,000", "5,000,000", "2,020,500"]
    fa = [p.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")).replace(",", "٬") for p in lat]
    cases = [("calibri-like", fonts.twin_font("Calibri", False, "0"), lat),
             ("dejavu", fonts.twin_font("Tahoma", False, "0"), lat),
             ("serif-bold", fonts.twin_font("TimesNewRoman-Bold", True, "0"), lat),
             ("persian-digits", str(ASSETS / "Vazirmatn-Regular.ttf"), fa),
             ("not-embedded", "helv", lat)]
    out = []
    for name, font, prices in cases:
        if font is None:
            print(f"(made-up case {name} skipped: its font is not installed)")
            continue
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="LB", fontfile=str(ASSETS / "Vazirmatn-Regular.ttf"))
        pf = font
        if font != "helv":
            page.insert_font(fontname="PF", fontfile=font)
            pf = "PF"
        page.insert_text((250, 80), "لیست قیمت", fontname="LB", fontsize=14)
        page.insert_text((60, 110), "قیمت", fontname="LB", fontsize=10)
        page.insert_text((400, 110), "شرح کالا", fontname="LB", fontsize=10)
        for k, p in enumerate(prices):
            page.insert_text((400, 135 + 22 * k), f"کالای شماره {k + 1}", fontname="LB", fontsize=10)
            page.insert_text((60, 135 + 22 * k), p, fontname=pf, fontsize=10)
        doc.subset_fonts()
        path = WORK / "made-up" / f"{name}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path)
        out.append(path)
    return out


def _zip_lists() -> list[Path]:
    z = ROOT / "pdf new.zip"
    if not z.exists():
        return []
    out = []
    with zipfile.ZipFile(z) as zf:
        for k, info in enumerate(i for i in zf.infolist() if i.filename.lower().endswith(".pdf")):
            name = info.filename
            try:
                name = name.encode("cp437").decode("utf-8")     # Persian names in an old-style zip
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            path = WORK / "lists" / f"{k + 1:02d} {Path(name).name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zf.read(info))
            out.append(path)
    return out


def check(path: Path) -> int:
    """Missing new prices in the written PDF of one list."""
    work = WORK / "work" / path.stem
    a = pipeline.analyze(path, path.name, work)
    values = commands.compute(commands.parse_local("15 درصد افزایش").plan, a.items, "")
    ex = Exporter([(a, values)], work / "out", "t", "check")
    doc = pymupdf.open(ex.build("pdf")[0])
    missing = []
    for it in a.items:
        if it.id not in values:
            continue
        new = to_latin_digits(it.fmt.format(values[it.id]))
        if it.kind == "raster":
            continue                       # (a price inside a picture has no text to look up)
        near = to_latin_digits(doc[it.page].get_text("text", clip=pymupdf.Rect(it.bbox) + (-40, -3, 40, 3)))
        if new not in near:
            missing.append(f"p{it.page + 1} {it.text} -> {new}")
    status = "ok" if not missing else f"{len(missing)} MISSING"
    print(f"{path.name[:44]:44s} {len(values):4d} prices changed  {status}"
          + "".join(f"\n      {m}" for m in missing[:5])
          + "".join(f"\n      ⚠️ {w}" for w in list(dict.fromkeys(ex.warnings))[:3]))
    return len(missing)


def main() -> None:
    lists = [Path(p) for p in sys.argv[1:]] or _zip_lists()
    total = 0
    for path in lists + _made_up():
        total += check(path)
    print(f"\n{'every price written' if not total else f'{total} prices missing'}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
