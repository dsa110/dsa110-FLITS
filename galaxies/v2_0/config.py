"""Configuration and target list for galaxy search v2.0."""

from typing import List, Tuple
from astropy.cosmology import Planck18

# Cosmology to use throughout the module
COSMO = Planck18

# Default search parameters
DEFAULT_IMPACT_KPC = 100.0
DEFAULT_Z_EPS = 0.01  # Redshift buffer for foreground search

# The 12 FRB sightlines in our sample
# Format: (name, RA, Dec, z_frb)
TARGETS: List[Tuple[str, str, str, float]] = [
    ("Zach",       "20h40m47.886s", "+72d52m56.378s", 0.0430),
    ("Whitney",    "08h58m52.92s",  "+73d29m27.0s",   0.4790),
    ("Oran",       "21h12m10.760s", "+72d49m38.20s",  0.3005),
    ("Isha",       "04h45m38.64s",  "+70d18m26.6s",   0.2505),
    ("Wilhelm",    "21h00m31.09s",  "+72d02m15.22s",  0.5100),
    ("Phineas",    "11h51m07.52s",  "+71d41m44.3s",   0.2710),
    ("Freya",      "05h52m45.12s",  "+74d12m01.7s",   1.0000),
    ("Hamilton",   "20h20m08.92s",  "+70d47m33.96s",  0.3024),
    ("Mahi",       "02h39m03.96s",  "+71d01m04.3s",   1.0000),
    ("Chromatica", "20h50m28.59s",  "+73d54m00.0s",   0.0740),
    ("Casey",      "11h19m56.05s",  "+70d40m34.4s",   0.2870),
    ("Johndoeii",  "22h23m53.94s",  "+73d01m33.26s",  1.0000),
]

# Catalog identifiers for Vizier
VIZIER_CATALOGS = {
    "GLADE+": "VII/281/glade2", # Includes 2MPZ, 2MASS XSC, SDSS-DR16Q
    "SDSS_DR12": "V/147/sdss12", # Stable spectroscopic/photometric catalog
}
