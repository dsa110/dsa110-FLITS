#!/usr/bin/env python3
"""
Shared configuration for galaxy catalog query scripts.

This module centralizes:
- Target sight-lines (RA, Dec, z_max)
- Cosmology and precomputed lookup tables
- Query parameters and constants
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18

# =============================================================================
# TARGET DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class Target:
    """A single FRB sight-line target."""
    ra: str
    dec: str
    z_max: float
    name: str = ""
    
    @property
    def coord(self) -> SkyCoord:
        """Return SkyCoord for this target."""
        return SkyCoord(self.ra, self.dec, frame="icrs")
    
    def __iter__(self):
        """Allow tuple unpacking: ra, dec, z = target"""
        return iter((self.ra, self.dec, self.z_max))


# Master target list - single source of truth
# Names: person name (TNS name)
TARGETS: List[Target] = [
    Target("20h40m47.886s", "+72d52m56.378s", 0.0430, "Zach (FRB20220207A)"),
    Target("08h58m52.92s",  "+73d29m27.0s",   0.4790, "Whitney (FRB20220310A)"),
    Target("21h12m10.760s", "+72d49m38.20s",  0.3005, "Oran (FRB20220506A)"),
    Target("04h45m38.64s",  "+70d18m26.6s",   0.2505, "Isha (FRB20221113A)"),
    Target("21h00m31.09s",  "+72d02m15.22s",  0.5100, "Wilhelm (FRB20221203A)"),
    Target("11h51m07.52s",  "+71d41m44.3s",   0.2710, "Phineas (FRB20230307A)"),
    Target("05h52m45.12s",  "+74d12m01.7s",   1.0000, "Freya (FRB20230325A)"),
    Target("20h20m08.92s",  "+70d47m33.96s",  0.3024, "Hamilton (FRB20230913A)"),
    Target("02h39m03.96s",  "+71d01m04.3s",   1.0000, "Mahi (FRB20240122A)"),
    Target("20h50m28.59s",  "+73d54m00.0s",   0.0740, "Chromatica (FRB20240203A)"),
    Target("11h19m56.05s",  "+70d40m34.4s",   0.2870, "Casey (FRB20240229A)"),
    Target("22h23m53.94s",  "+73d01m33.26s",  1.0000, "JohnDoeII (FRB20230814A)"),
]

# Legacy tuple format for backward compatibility
TARGETS_TUPLE: List[Tuple[str, str, float]] = [
    (t.ra, t.dec, t.z_max) for t in TARGETS
]


# =============================================================================
# COSMOLOGY & LOOKUP TABLES
# =============================================================================

COSMO = Planck18

# Precomputed angular diameter distance lookup table for speed
# Covers z=0.001 to z=2.0 with 1000 points (interpolation error < 0.1%)
_Z_GRID = np.linspace(0.001, 2.0, 1000)
_DA_GRID = COSMO.angular_diameter_distance(_Z_GRID).to(u.Mpc).value  # Mpc


def angular_diameter_distance_fast(z: np.ndarray) -> np.ndarray:
    """
    Fast angular diameter distance lookup via linear interpolation.
    
    Parameters
    ----------
    z : array-like
        Redshift(s) to compute D_A for.
    
    Returns
    -------
    d_a : np.ndarray
        Angular diameter distance in Mpc.
    """
    z = np.atleast_1d(z)
    return np.interp(z, _Z_GRID, _DA_GRID)


def theta_for_impact(z: float, impact_kpc: float = 100.0) -> float:
    """
    Angular radius (arcmin) subtending impact_kpc at redshift z.
    
    Uses the fast lookup table for D_A.
    """
    d_a_mpc = angular_diameter_distance_fast(np.array([z]))[0]
    theta_rad = (impact_kpc / 1000.0) / d_a_mpc  # kpc -> Mpc
    return np.degrees(theta_rad) * 60  # rad -> arcmin


# =============================================================================
# QUERY CONFIGURATION
# =============================================================================

# Physical parameters
R_PHYS_KPC = 100  # Maximum impact parameter in kpc
R_PHYS = R_PHYS_KPC * u.kpc

# Network/retry settings
MAX_TRIES = 5
BASE_DELAY = 2  # seconds
PAUSE = 0.5  # seconds between queries
NED_TIMEOUT = 180  # seconds

# Parallelization
MAX_WORKERS = 8  # for ThreadPoolExecutor

# Caching
CACHE_DIR = Path(".cache/galaxy_queries")
CACHE_EXPIRY_HOURS = 24

# Output paths
OUTPUT_DIR = Path("results/galaxies")


# =============================================================================
# CATALOG COLUMN MAPPINGS
# =============================================================================

# Standard output columns
STD_COLUMNS = ["name", "ra", "dec", "z", "Mstar", "Rproj_kpc", "catalog"]

# Vizier catalog IDs
VIZIER_CATALOGS = {
    "DESI_EDR": "VII/292",
    "DESI_AJ165": "J/AJ/165/144",
    "DESI_AJ164": "J/AJ/164/207",
    "DES_DR2": "II/371/des-dr2",
    "LS_DR9": "II/368/ls-dr9",
    "WISE_PHOTOZ": "J/ApJS/234/23/galaxies",
    "GLADE1": "VII/275/glade1",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_target_coords() -> List[SkyCoord]:
    """Return list of SkyCoord objects for all targets."""
    return [t.coord for t in TARGETS]


def get_target_by_index(idx: int) -> Target:
    """Get target by 1-indexed ID."""
    return TARGETS[idx - 1]


def filter_targets(indices: List[int] = None) -> List[Target]:
    """Filter targets by 1-indexed list, or return all if None."""
    if indices is None:
        return TARGETS
    return [TARGETS[i - 1] for i in indices]
