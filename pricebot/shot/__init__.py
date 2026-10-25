"""Edit the numbers in a Telegram screenshot (first target: the view counts under channel posts).

The digits are not typed with a look-alike font. Their exact shapes and spacing are learned from real
screenshots of the same app (`tools/learn_shots.py`) and redrawn pixel for pixel.
"""
from .bank import Bank, load_bank
from .detect import ShotError, Analysis, analyze
from .render import apply_views, format_views, parse_value, preview

__all__ = ["Bank", "load_bank", "ShotError", "Analysis", "analyze", "apply_views", "format_views",
           "parse_value", "preview"]
