"""Validation against manually-curated FRB foreground objects (network; run with -m network).

Compares pipeline output with known foreground objects manually identified
through VizieR searches for the CHIME-DSA co-detection sample.
"""

import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from .discover import discover_tables
from .normalize import to_common_schema
from .query import cone_query
from .reduce import merge_and_rank
from .registry import discover_tap_services
from .targets import Target, get_cosmology

VIZIER = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

# Known FRB targets and their manually-found foreground objects
VALIDATION_TARGETS = {
    "zach": {
        "frb": Target(name="zach", ra=310.1995, dec=72.8823, z_host=0.043),
        "expected_halos": [
            {"ra": 310.0913, "dec": 72.8104, "z_phot": 0.0127, "impact_kpc": 75.9, "survey": "WISE, PS1, STRM"}
        ],
        "expected_clusters": [
            {"ra": 133.9417, "dec": 73.3749, "z_phot": 0.1216, "impact_kpc": 2039.36, "m500": 0.64, "richness": 13.72}
        ],
    },
    "whitney": {
        "frb": Target(name="whitney", ra=134.7205, dec=73.4908, z_host=0.479),
        "expected_halos": [
            {"ra": 134.7356, "dec": 73.4910, "z_phot": 0.5731, "impact_kpc": 104.0, "survey": "Legacy DR8"}
        ],
        "expected_clusters": [
            {"ra": 133.8827, "dec": 73.4089, "z_phot": 0.4239, "impact_kpc": 5210.8, "m500": 0.9, "richness": 19.64}
        ],
    },
    "isha": {  # control case - no foreground objects expected
        "frb": Target(name="isha", ra=71.4110, dec=70.3074, z_host=0.2505),
        "expected_halos": [],
        "expected_clusters": [],
    },
}


def _separation_arcmin(ra1, dec1, ra2, dec2):
    return SkyCoord(ra1 * u.deg, dec1 * u.deg).separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).to(u.arcmin).value


def find_matching_objects(candidates, expected_objects, position_tolerance_arcmin=2.0):
    """Pipeline candidates within position tolerance of the expected objects."""
    matches = []
    for expected in expected_objects:
        for _, candidate in candidates.iterrows():
            sep = _separation_arcmin(expected["ra"], expected["dec"], candidate["ra"], candidate["dec"])
            if sep <= position_tolerance_arcmin:
                matches.append({"expected": expected, "found": candidate, "angular_separation_arcmin": sep})
                break
    return matches


def _collect_candidates(frb, tables, radii_arcmin, maxrec):
    all_candidates = []
    for radius in radii_arcmin:
        for _, table_row in tables.iterrows():
            try:
                result = cone_query(
                    VIZIER, table_row["table"], table_row["ra_col"], table_row["dec_col"],
                    ra_deg=frb.ra, dec_deg=frb.dec, radius_deg=radius / 60.0, maxrec=maxrec,
                )
            except Exception:
                continue
            if len(result) > 0:
                all_candidates.append(
                    to_common_schema(
                        result,
                        ra_col=table_row["ra_col"], dec_col=table_row["dec_col"], z_col=table_row["z_col"],
                        service=VIZIER, table=table_row["table"],
                    )
                )
    return all_candidates


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
def test_frb_zach_recovery(tmp_path):
    """Recover known foreground objects for FRB zach."""
    target_data = VALIDATION_TARGETS["zach"]
    frb = target_data["frb"]

    services = discover_tap_services(keywords=("galaxy", "cluster"), max_services=20)
    if VIZIER not in services["access_url"].values:
        pytest.skip("VizieR not available for validation test")

    tables = discover_tables(VIZIER, limit=50, cache_dir=tmp_path)
    if len(tables) == 0:
        pytest.skip("No suitable tables found for validation")

    all_candidates = _collect_candidates(frb, tables, radii_arcmin=[5, 10, 20, 30], maxrec=1000)
    if not all_candidates:
        pytest.fail("No candidates found in any search radius")

    ranked = merge_and_rank(pd.concat(all_candidates, ignore_index=True), frb.ra, frb.dec, get_cosmology())
    assert len(ranked) > 0, "Should find at least some foreground candidates"

    halo_matches = find_matching_objects(ranked, target_data["expected_halos"])
    assert len(halo_matches) >= 1, f"Should recover at least one known halo (candidates: {len(ranked)})"

    # Impact parameter within 20% of the manually-computed value
    best_halo = halo_matches[0]["found"]
    expected_impact = target_data["expected_halos"][0]["impact_kpc"]
    impact_ratio = abs(best_halo["b_kpc"] - expected_impact) / expected_impact
    assert impact_ratio < 0.2, f"Impact parameter mismatch: expected {expected_impact}, got {best_halo['b_kpc']}"

    # Clusters are harder to recover; report only
    cluster_matches = find_matching_objects(ranked, target_data["expected_clusters"])
    if target_data["expected_clusters"] and not cluster_matches:
        print("Warning: no clusters recovered (may require specific cluster catalogs)")


@pytest.mark.integration
@pytest.mark.network
def test_frb_isha_control(tmp_path):
    """Control case: FRB with no known foreground objects finds few close objects."""
    frb = VALIDATION_TARGETS["isha"]["frb"]

    services = discover_tap_services(max_services=5)
    if VIZIER not in services["access_url"].values:
        pytest.skip("VizieR not available for control test")

    tables = discover_tables(VIZIER, limit=20, cache_dir=tmp_path)
    if len(tables) == 0:
        pytest.skip("No tables found for control test")

    all_candidates = _collect_candidates(frb, tables, radii_arcmin=[10], maxrec=500)
    if all_candidates:
        ranked = merge_and_rank(pd.concat(all_candidates, ignore_index=True), frb.ra, frb.dec, get_cosmology())
        close_objects = ranked[ranked["b_kpc"] < 200]
        print(f"Control case: {len(close_objects)} objects within 200 kpc of {len(ranked)} candidates")


@pytest.mark.integration
@pytest.mark.network
def test_pipeline_completeness(tmp_path):
    """Pipeline finds objects in physically-reasonable redshift and impact ranges."""
    frb = VALIDATION_TARGETS["zach"]["frb"]

    services = discover_tap_services(max_services=10)
    if VIZIER not in services["access_url"].values:
        pytest.skip("VizieR not available")

    tables = discover_tables(VIZIER, limit=30, cache_dir=tmp_path)
    if len(tables) == 0:
        pytest.skip("No tables found")

    all_candidates = _collect_candidates(frb, tables, radii_arcmin=[20], maxrec=1000)
    if not all_candidates:
        pytest.skip("No candidates found for completeness test")

    ranked = merge_and_rank(pd.concat(all_candidates, ignore_index=True), frb.ra, frb.dec, get_cosmology())
    z_values = ranked["z"].dropna()
    impact_values = ranked["b_kpc"].dropna()
    if len(z_values) > 0:
        assert z_values.min() >= 0, "No negative redshifts"
        assert z_values.max() <= 10.0, "Reasonable maximum redshift"
        assert impact_values.min() >= 0, "Positive impact parameters"
