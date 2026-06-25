"""Configuration and target list for foreground galaxy search."""

from astropy.cosmology import Planck18

# Cosmology to use throughout the module
COSMO = Planck18

# Default search parameters
DEFAULT_IMPACT_KPC = 100.0
DEFAULT_CLUSTER_IMPACT_KPC = (
    5000.0  # 5 Mpc: clusters intercept sightlines far beyond galaxy-scale impacts
)
DEFAULT_Z_EPS = 0.01  # Redshift buffer for foreground search
MIN_Z_SEARCH = 0.005  # Minimum redshift cutoff to avoid infinite search cone
# Minimum credible photometric redshift: DESI Legacy photo-z (VII/292/north) pin
# unreliable sources near z~0 (z=0.001-0.008 floor noise). Applies only to rows
# carrying a photo-z error; spec-z catalogs (NED, GLADE+) are exempt so genuine
# nearby galaxies (e.g. UGC 06371, z=0.009) survive.
FOREGROUND_PHOTOZ_FLOOR = 0.01
# Velocity-offset window (km/s) below which a spec-z neighbour is host/local
# ambiguous, not a clean foreground. dv = c*(z_frb - z_gal)/(1 + z_frb); a galaxy
# inside this window sits at the host's recession velocity (group member / host
# peculiar velocity ~200-500 km/s) and cannot be cleanly attributed as an
# intervening system. Set at the upper edge of typical group dispersions.
FOREGROUND_AMBIGUITY_KMS = 500.0
SPEED_OF_LIGHT_KMS = 299792.458
# Spec-z catalogs: redshift is spectroscopic / distance-derived, so any z_err they
# carry (e.g. GLADE+'s generic 0.015 floor) is NOT a per-object photo-z and must
# not gate via the photo-z error path. Matched on the `catalog` column substring;
# DESI VII/292/north is the lone photo-z catalog.
SPEC_Z_CATALOG_SUBSTRINGS = ("ned", "vii/291", "glade", "sdss", "desi_dr1")
PHOTO_Z_CATALOG_SUBSTRINGS = ("vii/292",)
MAX_SEARCH_RADIUS_DEG = (
    2.0  # Cap angular query radius (low-z clusters otherwise blow up Vizier query cones)
)

# The 12 FRB sightlines in our sample
# Format: (name, RA, Dec, z_frb)
TARGETS: list[tuple[str, str, str, float]] = [
    ("Zach", "20h40m47.886s", "+72d52m56.378s", 0.0430),
    ("Whitney", "08h58m52.92s", "+73d29m27.0s", 0.4790),
    ("Oran", "21h12m10.760s", "+72d49m38.20s", 0.3005),
    ("Isha", "04h45m38.64s", "+70d18m26.6s", 0.2505),
    ("Wilhelm", "21h00m31.09s", "+72d02m15.22s", 0.5100),
    ("Phineas", "11h51m07.52s", "+71d41m44.3s", 0.2710),
    ("Freya", "05h52m45.12s", "+74d12m01.7s", 1.0000),
    ("Hamilton", "20h20m08.92s", "+70d47m33.96s", 0.3024),
    ("Mahi", "02h39m03.96s", "+71d01m04.3s", 1.0000),
    ("Chromatica", "20h50m28.59s", "+73d54m00.0s", 0.0740),
    ("Casey", "11h19m56.05s", "+70d40m34.4s", 0.2870),
    ("Johndoeii", "22h23m53.94s", "+73d01m33.26s", 1.0000),
]

# Catalog identifiers for Vizier
VIZIER_CATALOGS = {
    "GLADE+": "VII/291/gladep",  # GLADE+ (2022); Vizier renamed table glade -> gladep
    "DESI_DR8_NORTH": "VII/292/north",  # High-Dec northern sky coverage
    "SDSS_DR12": "V/147/sdss12",  # Stable spectroscopic/photometric catalog
}

# Opt-in extra search engines (TAP-backed). Disabled by default so run_search()
# output is byte-for-byte unchanged unless explicitly enabled.
EXTRA_SEARCH_ENGINES = {"DESI_DR1": "desi_dr1.zpix"}

# Opt-in enricher catalogs (cross-matched photometry, not foreground search).
ENRICHER_CATALOGS = {
    "DESI_LS_DR10": "ls_dr10.tractor",
    "ALLWISE": "II/328/allwise",
    "GALEX_AIS": "II/335/galex_ais",
    "2MASS_XSC": "VII/233/xsc",
}

ENABLE_EXTRA_ENGINES = False
ENABLE_ENRICHERS = False

# All-sky galaxy-cluster catalogs. Only all-sky catalogs reach the sample's
# declination (+70..+74); SDSS-based WHL/redMaPPer do not (see
# docs/rse/specs/research-foreground-galaxies-sightlines.md). PSZ2 reports M500 as
# MSZ (1e14 Msun); MCXC/MCXC-II report M500 (1e14) + R500 (Mpc).
CLUSTER_VIZIER_CATALOGS = {
    "PSZ2": "J/A+A/594/A27/psz2",
    "MCXC": "J/A+A/534/A109/mcxc",
    "MCXC_II": "J/A+A/688/A187/mcxcii",  # MCXC-II (A&A 688, A187; arXiv:2402.01538), live-confirmed
}
# Keep a cluster when impact <= this multiple of its own r200 (research: meaningful
# cluster DM needs the sightline within ~1-2 r200).
CLUSTER_R200_FACTOR = 2.0
# M200/M500 conversion for r200 + halo-mass injection (order-of-magnitude; typical
# NFW c500 ~ 1.5 gives M200/M500 ~ 1.3).
CLUSTER_M500_TO_M200 = 1.3
ENABLE_CLUSTER_ENGINE = True
