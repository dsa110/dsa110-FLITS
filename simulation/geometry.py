from __future__ import annotations

"""Helpers for cosmological distances and vector utilities."""

from typing import Tuple

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo


def _DA(z1: float, z2: float = 0.0) -> u.Quantity:
    """Angular-diameter distance between two redshifts."""
    if z2 < z1:
        z1, z2 = z2, z1
    return cosmo.angular_diameter_distance_z1z2(z1, z2)


def _array2(vec: Tuple[float, float] | np.ndarray | None, unit: u.Unit) -> np.ndarray:
    """Ensure a 2-vector has correct shape and units."""
    if vec is None:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("Velocity / offset vectors must be 2-element tuples.")
    return (arr * unit).to(unit).value
