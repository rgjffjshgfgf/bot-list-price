"""Edit the numbers in a Telegram screenshot (first target: the view counts under channel posts).

The digits are not typed with a look-alike font. Their exact shapes and spacing are learned from real
screenshots of the same app (`tools/learn_shots.py`) and redrawn pixel for pixel.

The image code (OpenCV, numpy) loads on first use, so the bot process can import `pricebot.shot.values`
alone.
"""
from __future__ import annotations

import importlib

_WHERE = {"Bank": "bank", "load_bank": "bank", "ShotError": "values", "Analysis": "detect", "analyze": "detect",
          "apply_views": "render", "format_views": "values", "parse_value": "values", "preview": "render"}

__all__ = list(_WHERE)


def __getattr__(name: str):
    if name in _WHERE:
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(name)
