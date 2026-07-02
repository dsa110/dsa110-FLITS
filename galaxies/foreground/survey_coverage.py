"""Survey query coverage: which catalogs were queried per sightline and footprint status."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from astropy.coordinates import SkyCoord

# Nominal sky footprints (geometry-only; independent of cone yield).
# all_sky: NED, GLADE+, cluster compendia
# desi_north: DESI Legacy DR8 North photo-z (VII/292/north)
# sdss_ngc: SDSS imaging North Galactic Cap (Dec >= 1.26 deg)
# desi_dr1: DESI DR1 zpix spectroscopic footprint (northern Legacy; approximate)
FOOTPRINT_RULES: dict[str, str] = {
    "NED": "all_sky",
    "GLADE+": "all_sky",
    "DESI_DR8_NORTH": "desi_north",
    "SDSS_DR12": "sdss_ngc",
    "CLUSTERS": "all_sky",
    "DESI_DR1": "desi_dr1",
}


def survey_in_footprint(survey_key: str, coord: SkyCoord) -> bool:
    """Return whether the sightline lies inside the catalog's nominal sky footprint."""
    rule = FOOTPRINT_RULES.get(survey_key, "all_sky")
    dec = coord.dec.deg
    if rule == "all_sky":
        return True
    if rule == "desi_north":
        return dec >= -20.0
    if rule == "sdss_ngc":
        return dec >= 1.26
    if rule == "desi_dr1":
        return dec >= -20.0
    return True


def engine_survey_key(engine: Any) -> str:
    """Stable survey label for coverage tables (matches search log names)."""
    from .engines import VizierEngine
    from .engines_extra import ClusterEngine, DesiDr1Engine, NedTapEngine

    if isinstance(engine, NedTapEngine):
        return "NED"
    if isinstance(engine, ClusterEngine):
        return "CLUSTERS"
    if isinstance(engine, DesiDr1Engine):
        return "DESI_DR1"
    if isinstance(engine, VizierEngine):
        from .config import VIZIER_CATALOGS

        for label, cat_id in VIZIER_CATALOGS.items():
            if cat_id == engine.catalog_id:
                return label
        return engine.catalog_id
    return engine.__class__.__name__


def classify_coverage(
    *,
    in_footprint: bool,
    raw_count: int,
    foreground_count: int,
) -> str:
    if not in_footprint:
        return "no_footprint"
    if foreground_count > 0:
        return "foreground"
    if raw_count > 0:
        return "catalog_hits"
    return "footprint_empty"


def write_survey_coverage_csv(rows: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "survey_coverage.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
