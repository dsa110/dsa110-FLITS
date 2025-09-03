"""Compatibility façade for the FRB scintillation simulator."""

from geometry import _DA, _array2  # re-exported for backwards compatibility
from instrument import InstrumentalCfg
from screen import ScreenCfg, Screen
from engine import SimCfg, FRBScintillator

__all__ = [
    "FRBScintillator",
    "SimCfg",
    "ScreenCfg",
    "Screen",
    "InstrumentalCfg",
    "_DA",
    "_array2",
]
