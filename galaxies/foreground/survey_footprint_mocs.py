"""Exact survey footprints via CDS MOCServer MOCs (cached on disk)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import Latitude, Longitude
from astropy.utils.data import download_file
from mocpy import MOC

# CDS MocServer IDs for catalogs queried in run_search (see config.VIZIER_CATALOGS).
CDS_MOC_IDS: dict[str, str] = {
    "GLADE+": "CDS/VII/291/gladep",
    "DESI_DR8_NORTH": "CDS/VII/292/north",
    "SDSS_DR12": "CDS/V/147/sdss12",
}

# Fallback if MocServer is unreachable.
VIZIER_MOC_TABLES: dict[str, str] = {
    "GLADE+": "VII/291/gladep",
    "DESI_DR8_NORTH": "VII/292/north",
    "SDSS_DR12": "V/147/sdss12",
}

# NED TAP + PSZ2/MCXC/MCXC-II cluster compendia have no spatial footprint limit.
ALL_SKY_SURVEYS: frozenset[str] = frozenset({"NED", "CLUSTERS"})

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "survey_footprints"
DEFAULT_RASTER_ORDER = 6


def survey_display_names() -> list[str]:
    return ["NED", "GLADE+", "DESI_DR8_NORTH", "SDSS_DR12", "CLUSTERS"]


def _cache_path(cache_dir: Path, survey: str) -> Path:
    return cache_dir / f"{survey.lower()}.moc.fits"


def _full_sky_moc(order: int = DEFAULT_RASTER_ORDER) -> MOC:
    return MOC.from_cone(lon=0 * u.deg, lat=0 * u.deg, radius=180 * u.deg, max_depth=order)


def _mocserver_url(cds_id: str) -> str:
    return (
        "http://alasky.cds.unistra.fr/MocServer/query?"
        f"get=moc&fmt=fits&ID={quote(cds_id, safe='')}"
    )


def _fetch_moc(survey: str) -> MOC:
    cds_id = CDS_MOC_IDS[survey]
    url = _mocserver_url(cds_id)
    try:
        tmp = download_file(url, cache=False, show_progress=False)
        return MOC.load(tmp)
    except Exception:
        return MOC.from_vizier_table(VIZIER_MOC_TABLES[survey])


def load_survey_moc(survey: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> MOC:
    """Load (or fetch and cache) the CDS MOC for a foreground-search survey."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if survey in ALL_SKY_SURVEYS:
        path = _cache_path(cache_dir, survey)
        if path.is_file():
            return MOC.load(str(path))
        moc = _full_sky_moc()
        moc.save(str(path), "fits")
        return moc

    if survey not in CDS_MOC_IDS:
        raise KeyError(f"No MOC source configured for survey {survey!r}")

    path = _cache_path(cache_dir, survey)
    if path.is_file():
        return MOC.load(str(path))

    moc = _fetch_moc(survey)
    moc.save(str(path), "fits")
    return moc


def rasterize_moc(moc: MOC, order: int = DEFAULT_RASTER_ORDER) -> np.ndarray:
    """HEALPix map (1=in footprint, UNSEEN=out) at fixed order for mollview."""
    if moc.max_order > order:
        moc = moc.degrade_to_order(order)
    nside = 2**order
    ipix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, ipix)
    ra = np.rad2deg(phi)
    dec = 90.0 - np.rad2deg(theta)
    inside = moc.contains_lonlat(Longitude(ra * u.deg), Latitude(dec * u.deg))
    return np.where(inside, 1.0, hp.UNSEEN)


def moc_sky_area_deg2(moc: MOC) -> float:
    return float(moc.sky_fraction * 41253.0)


def prefetch_all_survey_mocs(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> dict[str, Path]:
    """Fetch/cache every survey MOC; returns survey -> cache path."""
    paths: dict[str, Path] = {}
    for survey in survey_display_names():
        load_survey_moc(survey, cache_dir)
        paths[survey] = _cache_path(Path(cache_dir), survey)
    return paths
