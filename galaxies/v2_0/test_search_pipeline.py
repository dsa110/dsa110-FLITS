import math

import numpy as np
import pandas as pd

from galaxies.v2_0 import search as search_mod
from galaxies.v2_0.config import COSMO
from galaxies.v2_0.engines import _add_desi_stellar_mass
from galaxies.v2_0.search import (
    _deduplicate_matches,
    _enrich_with_ps1_photometry,
    _foreground_mask,
)


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


def test_enrich_with_ps1_only_fills_rows_missing_catalog_mass(monkeypatch):
    # Row 0: DESI-style, no catalog mass -> should be PS1-queried and filled.
    # Row 1: GLADE-style, has a catalog mass -> must be skipped (not queried).
    matches = pd.DataFrame(
        {
            "ra": [177.78, 310.20],
            "dec": [71.70, 72.87],
            "z": [0.241, 0.043],
            "M_star": [math.nan, 9.51],
        }
    )

    queried = []

    def fake_ps1(coord, match_radius=None):
        queried.append((float(coord.ra.deg), float(coord.dec.deg)))
        return 20.18, 18.84, 0.01  # g_Kron, i_Kron, sep_arcsec

    monkeypatch.setattr(search_mod, "query_ps1_gi_mags", fake_ps1)

    result = _enrich_with_ps1_photometry(matches)

    # Only the mass-less DESI row triggered a PS1 query.
    assert len(queried) == 1
    assert queried[0][0] == 177.78
    # DESI row got photometry; GLADE row stayed NaN.
    assert math.isclose(result.loc[0, "gmag"], 20.18)
    assert math.isclose(result.loc[0, "imag"], 18.84)
    assert math.isclose(result.loc[0, "ps1_sep_arcsec"], 0.01)
    assert np.isnan(result.loc[1, "gmag"])
    assert np.isnan(result.loc[1, "ps1_sep_arcsec"])


def test_enrich_with_ps1_handles_no_match(monkeypatch):
    # Faint high-z galaxy below PS1 depth -> no match -> columns stay NaN.
    matches = pd.DataFrame({"ra": [88.20], "dec": [74.20], "z": [0.965]})

    monkeypatch.setattr(search_mod, "query_ps1_gi_mags", lambda coord, match_radius=None: (None, None, None))

    result = _enrich_with_ps1_photometry(matches)

    assert np.isnan(result.loc[0, "gmag"])
    assert np.isnan(result.loc[0, "imag"])
    assert np.isnan(result.loc[0, "ps1_sep_arcsec"])
