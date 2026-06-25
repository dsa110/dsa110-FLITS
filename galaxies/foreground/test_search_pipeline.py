import math

import numpy as np
import pandas as pd

from galaxies.foreground import scattering_predict as scat
from galaxies.foreground import search as search_mod
from galaxies.foreground.config import COSMO
from galaxies.foreground.engines import _add_desi_stellar_mass
from galaxies.foreground.search import (
    _cluster_impact_limit_kpc,
    _deduplicate_matches,
    _enrich_with_ps1_photometry,
    _foreground_mask,
)


def test_foreground_mask_photoz_point_estimate_not_rescued_by_error():
    # Per-sightline adjudication (PI decision): a DESI VII/292 photo-z is foreground
    # iff its POINT ESTIMATE is below z_frb. The 1-sigma error never rescues a
    # background point estimate (the old capped-2-sigma cut did). z=0.46 < z_frb is
    # kept even though z+e_zphot > z_frb (Wilhelm); z=0.545 > z_frb is dropped.
    df = pd.DataFrame(
        {
            "z": [0.46, 0.545],
            "e_zphot": [0.057, 0.057],
            "catalog": ["VII/292/north", "VII/292/north"],
            "impact_kpc": [35.0, 35.0],
        }
    )

    mask = _foreground_mask(df, z_frb=0.510, z_eps=0.01, impact_kpc=100.0)

    assert mask.tolist() == [True, False]


def test_foreground_mask_drops_photoz_floor_keeps_spec_lowz():
    # DESI photo-z floor junk (z~0.001, carries e_zphot) is dropped; a genuine
    # nearby spec-z galaxy at the same redshift (no photo-z error) is kept.
    df = pd.DataFrame(
        {
            "z": [0.001, 0.001, 0.30],
            "z_phot_err": [0.8, math.nan, 0.04],
            "impact_kpc": [50.0, 50.0, 50.0],
        }
    )
    mask = _foreground_mask(df, z_frb=0.40, z_eps=0.01, impact_kpc=100.0)
    assert mask.tolist() == [False, True, True]


def test_foreground_mask_photoz_background_leak_rejected():
    # A z=0.9 DESI photo-z behind a z=0.27 FRB carries an absurd e_zphot (sigma=1.0)
    # that the old capped-2-sigma cut would rescue; the point-estimate rule rejects
    # it outright. A genuine foreground photo-z (z=0.20) is kept.
    df = pd.DataFrame(
        {
            "z": [0.90, 0.20],
            "e_zphot": [1.0, 0.04],
            "catalog": ["VII/292/north", "VII/292/north"],
            "impact_kpc": [50.0, 50.0],
        }
    )
    mask = _foreground_mask(df, z_frb=0.27, z_eps=0.01, impact_kpc=100.0)
    assert mask.tolist() == [False, True]


def test_foreground_mask_specz_velocity_offset_adjudication():
    # Spec-z (GLADE+) rows are adjudicated by velocity offset, NOT the photo-z error
    # path, even though GLADE+ carries a generic 0.015 z-error floor. Reproduces the
    # Zach sightline (z_frb=0.043): the dv=667 km/s neighbour is a clean foreground;
    # the dv=76 km/s neighbour is host-velocity ambiguous and dropped; a background
    # spec-z (Chromatica-like, z>z_frb) is dropped.
    df = pd.DataFrame(
        {
            "z": [0.04068, 0.042735, 0.0764],
            "z_phot_err": [0.015, 0.015, 0.015],
            "catalog": ["VII/291/gladep", "VII/291/gladep", "VII/291/gladep"],
            "impact_kpc": [19.0, 24.0, 53.0],
        }
    )
    mask = _foreground_mask(df, z_frb=0.043, z_eps=0.01, impact_kpc=100.0)
    assert mask.tolist() == [True, False, False]


def test_foreground_mask_applies_cluster_impact_threshold():
    # Same impact (1000 kpc): a galaxy is rejected (>100 kpc) but a cluster is
    # kept (<5000 kpc). Cluster flagged via NED 'GClstr' / SDSS 'ClG' classifications.
    df = pd.DataFrame(
        {
            "z": [0.10, 0.10, 0.10],
            "impact_kpc": [1000.0, 1000.0, 1000.0],
            "classification": ["G", "GClstr", "ClG"],
        }
    )

    mask = _foreground_mask(
        df, z_frb=0.479, z_eps=0.01, impact_kpc=100.0, cluster_impact_kpc=5000.0
    )

    assert mask.tolist() == [False, True, True]


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

    result = _add_desi_stellar_mass(df.copy(), "VII/291/gladep")

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

    monkeypatch.setattr(
        search_mod, "query_ps1_gi_mags", lambda coord, match_radius=None: (None, None, None)
    )

    result = _enrich_with_ps1_photometry(matches)

    assert np.isnan(result.loc[0, "gmag"])
    assert np.isnan(result.loc[0, "imag"])
    assert np.isnan(result.loc[0, "ps1_sep_arcsec"])


def test_cluster_impact_limit_uses_r200_when_mass_present():
    # A 5e14 Msun cluster at z=0.1: limit = 2 * r200(M200=1.3*M500).
    df = pd.DataFrame(
        {"classification": ["cluster"], "z": [0.1], "m500_msun": [5.0e14], "impact_kpc": [1.0]}
    )
    r200 = scat.r_delta_kpc(1.3 * 5.0e14, 0.1, 200)
    limit = _cluster_impact_limit_kpc(df, impact_kpc=100.0, r200_factor=2.0, fallback_kpc=5000.0)
    assert abs(limit.iloc[0] - 2.0 * r200) / (2.0 * r200) < 1e-9


def test_cluster_impact_limit_falls_back_without_mass():
    df = pd.DataFrame({"classification": ["GClstr"], "z": [0.1], "impact_kpc": [1.0]})
    limit = _cluster_impact_limit_kpc(df, impact_kpc=100.0, r200_factor=2.0, fallback_kpc=5000.0)
    assert limit.iloc[0] == 5000.0  # NED-Type cluster, no catalog mass


def test_foreground_mask_r200_rejects_far_cluster_keeps_near():
    # r200(1.3*5e14, z=0.1) ~ 2 Mpc; 2*r200 ~ 4 Mpc. 3 Mpc kept, 6 Mpc rejected.
    df = pd.DataFrame(
        {
            "classification": ["cluster", "cluster"],
            "z": [0.1, 0.1],
            "m500_msun": [5.0e14, 5.0e14],
            "impact_kpc": [3000.0, 6000.0],
        }
    )
    mask = _foreground_mask(df, z_frb=0.5, z_eps=0.01, impact_kpc=100.0)
    assert mask.tolist() == [True, False]
