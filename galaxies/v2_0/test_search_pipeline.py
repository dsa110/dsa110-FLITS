import math

import pandas as pd

from galaxies.v2_0.config import COSMO
from galaxies.v2_0.engines import _add_desi_stellar_mass
from galaxies.v2_0.search import _deduplicate_matches, _foreground_mask


def test_foreground_mask_uses_two_sigma_photo_z_error():
    df = pd.DataFrame(
        {
            "z": [0.545, 0.610, 0.545],
            "z_phot_err": [0.057, 0.057, math.nan],
            "impact_kpc": [50.0, 50.0, 50.0],
        }
    )

    mask = _foreground_mask(df, z_frb=0.479, z_eps=0.01, impact_kpc=100.0)

    assert mask.tolist() == [True, False, False]


def test_deduplicate_matches_requires_close_redshift_and_prefers_better_source():
    matches = pd.DataFrame(
        {
            "name": ["desi_photo", "ned_spec", "high_z_pair"],
            "ra": [10.0, 10.0 + 1.0 / 3600.0, 10.0 + 2.0 / 3600.0],
            "dec": [20.0, 20.0, 20.0],
            "z": [0.300, 0.301, 0.420],
            "z_phot_err": [0.05, math.nan, 0.05],
            "catalog": ["VII/292/north", "NED", "VII/292/north"],
        }
    )

    deduped = _deduplicate_matches(matches)

    assert deduped["name"].tolist() == ["ned_spec", "high_z_pair"]


def test_deduplicate_matches_prefers_smaller_redshift_error():
    matches = pd.DataFrame(
        {
            "name": ["large_error", "small_error"],
            "ra": [10.0, 10.0 + 1.0 / 3600.0],
            "dec": [20.0, 20.0],
            "z": [0.300, 0.301],
            "z_phot_err": [0.05, 0.01],
            "catalog": ["VII/292/north", "VII/292/north"],
        }
    )

    deduped = _deduplicate_matches(matches)

    assert deduped["name"].tolist() == ["small_error"]


def test_deduplicate_matches_ignores_already_dropped_entries():
    matches = pd.DataFrame(
        {
            "name": ["kept_first", "dropped_middle", "kept_third"],
            "ra": [10.0, 10.0 + 8.0 / 3600.0, 10.0 + 16.0 / 3600.0],
            "dec": [20.0, 20.0, 20.0],
            "z": [0.300, 0.300, 0.300],
            "catalog": ["NED", "VII/292/north", "VII/292/north"],
        }
    )

    deduped = _deduplicate_matches(matches)

    assert deduped["name"].tolist() == ["kept_first", "kept_third"]


def test_add_desi_stellar_mass_from_fluxes_and_redshift():
    df = pd.DataFrame(
        {
            "flux_g": [10.0],
            "flux_r": [20.0],
            "flux_z": [30.0],
            "z": [0.1],
        }
    )

    result = _add_desi_stellar_mass(df.copy(), "VII/292/north")

    g_mag = 22.5 - 2.5 * math.log10(10.0)
    r_mag = 22.5 - 2.5 * math.log10(20.0)
    d_l_pc = COSMO.luminosity_distance(0.1).to("pc").value
    distance_modulus = 5.0 * math.log10(d_l_pc / 10.0)
    absolute_r = r_mag - distance_modulus
    expected_log_mass = -0.68 + 0.70 * (g_mag - r_mag) - 0.4 * absolute_r

    assert result.loc[0, "mass_source"] == "photometric"
    assert math.isclose(result.loc[0, "M_star"], expected_log_mass, rel_tol=1e-10)


def test_glade_mass_source_is_catalog_when_mass_present():
    df = pd.DataFrame({"M_star": [10.2], "z": [0.1]})

    result = _add_desi_stellar_mass(df.copy(), "VII/291/glade")

    assert result.loc[0, "mass_source"] == "catalog"
