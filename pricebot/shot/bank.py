"""The learned shapes of the footer characters (digits, '.', ':', 'K', the eye icon) and their spacing."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
BANK_NPZ = "footer_bank.npz"
BANK_JSON = "footer_model.json"
EYE = "E"
CHARS = "0123456789.:KM"


@dataclass
class Glyph:
    a: np.ndarray        # alpha 0..1, cropped tight to the ink (native pixels)
    dy: float            # top of the ink relative to the top of the digit '1' (native pixels)
    cx: float            # centre of mass inside `a`
    cy: float

    @property
    def h(self) -> int:
        return self.a.shape[0]

    @property
    def w(self) -> int:
        return self.a.shape[1]


def make_glyph(alpha: np.ndarray, dy: float) -> Glyph:
    a = np.clip(alpha, 0, 1).astype(np.float32)
    tot = float(a.sum()) or 1.0
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return Glyph(a, float(dy), float((a * xs).sum() / tot), float((a * ys).sum() / tot))


class Bank:
    """Shapes + a tiny layout model: the distance between the left edges of two neighbours is
    u[first] + v[second]; the space between views and time adds `sp`."""

    def __init__(self) -> None:
        self.glyphs: dict[str, Glyph] = {}
        self.u: dict[str, float] = {}
        self.v: dict[str, float] = {}
        self.sp: float = 0.0
        self.cap_h: float = 23.0
        self.pair_obs: dict[str, dict[str, int]] = {}    # "a|b" -> {distance: count}
        self.sp_obs: dict[str, int] = {}                 # views -> time distances minus the model part
        self.sp_samples: list[float] = []                # width of the space, measured on scaled screenshots
        self.counts: dict[str, int] = {}
        self.images: int = 0

    # --------------------------------------------------------------- queries --
    def ready(self) -> bool:
        return EYE in self.glyphs and all(c in self.glyphs for c in "0123456789:") and bool(self.u)

    def has(self, ch: str) -> bool:
        return ch in self.glyphs

    def missing(self, text: str) -> list[str]:
        return sorted({c for c in text if c not in self.glyphs})

    def distance(self, a: str, b: str, space: bool = False) -> float:
        """Native-pixel distance between the left ink edges of neighbours a and b."""
        seen = (self.sp_obs if space else self.pair_obs).get(f"{a}|{b}")
        if seen:
            n = sum(seen.values())
            if n >= 2:
                return sum(int(d) * c for d, c in seen.items()) / n
        return self.u.get(a, 0.0) + self.v.get(b, 0.0) + (self.sp if space else 0.0)

    # ------------------------------------------------------------------- io --
    def save(self, folder: Path = ASSETS) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {f"g_{ord(k)}": (np.clip(g.a, 0, 1) * 255).round().astype(np.uint8) for k, g in self.glyphs.items()}
        np.savez_compressed(folder / BANK_NPZ, **arrays)
        meta = {
            "glyphs": {k: {"dy": g.dy} for k, g in self.glyphs.items()},
            "u": self.u, "v": self.v, "sp": self.sp, "cap_h": self.cap_h,
            "pair_obs": self.pair_obs, "sp_obs": self.sp_obs, "sp_samples": self.sp_samples,
            "counts": self.counts, "images": self.images,
        }
        (folder / BANK_JSON).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path = ASSETS) -> "Bank":
        bank = cls()
        npz, js = folder / BANK_NPZ, folder / BANK_JSON
        if not (npz.exists() and js.exists()):
            return bank
        meta = json.loads(js.read_text(encoding="utf-8"))
        with np.load(npz) as data:
            for k, info in meta.get("glyphs", {}).items():
                name = f"g_{ord(k)}"
                if name in data:
                    bank.glyphs[k] = make_glyph(data[name].astype(np.float32) / 255.0, info["dy"])
        bank.u = {k: float(v) for k, v in meta.get("u", {}).items()}
        bank.v = {k: float(v) for k, v in meta.get("v", {}).items()}
        bank.sp = float(meta.get("sp", 0.0))
        bank.cap_h = float(meta.get("cap_h", 23.0))
        bank.pair_obs = meta.get("pair_obs", {})
        bank.sp_obs = meta.get("sp_obs", {})
        bank.sp_samples = [float(x) for x in meta.get("sp_samples", [])]
        bank.counts = meta.get("counts", {})
        bank.images = int(meta.get("images", 0))
        return bank


_LOCK = threading.Lock()
_BANK: Bank | None = None


def load_bank(reload: bool = False) -> Bank:
    global _BANK
    with _LOCK:
        if _BANK is None or reload:
            _BANK = Bank.load()
        return _BANK
