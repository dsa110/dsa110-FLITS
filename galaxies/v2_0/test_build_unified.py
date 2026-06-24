import math

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from galaxies.v2_0 import scattering_predict as scat
from galaxies.v2_0.build_unified import build_for_target, build_unified_records


def _sightline(ra_str, dec_str):
    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return coord.ra.deg, coord.dec.deg


def test_glade_unified_columns_and_mass():
    # GLADE+ (gladep) rows carry a catalog stellar mass -> mass_source
    # 'glade_catalog'. Frozen inline fixture so the test is independent of the
    # regenerated results/ snapshot (upstream catalog refreshes change it).
    sight_ra, sight_dec = _sightline("20h40m47.886s", "+72d52m56.378s")
    matches = pd.DataFrame(
        {
            "ra": [sight_ra + 0.002, sight_ra - 0.002],
            "dec": [sight_dec + 0.001, sight_dec - 0.001],
            "z": [0.040, 0.042],
            "M_star": [9.5, 9.1],
            "catalog": ["VII/291/gladep", "VII/291/gladep"],
            "impact_kpc": [30.0, 45.0],
        }
    )

    df = build_unified_records(matches, 0.30, sight_ra, sight_dec, enrich=False)

    expected_columns = {
        "logM_best",
        "mass_source",
        "mass_method",
        "M_halo",
        "logM_halo",
        "R_vir_kpc",
        "r_s",
        "c",
        "b_over_rvir",
        "intersects_rvir",
        "pred_tau_scat_ms_1GHz",
        "pred_scint_bw_khz",
        "scattering_rank",
        "dm_halo",
        "dm_cool",
        "cool_fc",
        "pred_mgii_wr",
        "cgm_extractable_flags",
        "is_star_forming",
        "metallicity_12logOH",
    }
    assert expected_columns.issubset(df.columns)
    assert list(df["mass_source"]) == ["glade_catalog", "glade_catalog"]
    np.testing.assert_allclose(df["logM_best"], matches["M_star"])
    assert (df["intersects_rvir"] == (df["impact_kpc"] <= df["R_vir_kpc"])).all()

    for flags in df["cgm_extractable_flags"]:
        assert isinstance(flags, dict)
        assert flags["stellar_mass"] == "MEASURED"
        assert flags["dm_halo"] == "PREDICTED"
        assert flags["desi_spectro"] == "NOT_AVAILABLE"

    assert np.isfinite(df["pred_tau_scat_ms_1GHz"]).any()
    assert (df["pred_tau_scat_ms_1GHz"].dropna() >= 0).all()
    assert int(df["scattering_rank"].min()) == 1


def test_unified_mass_sources_ps1_and_assumed():
    # Frozen inline fixture (independent of results/): row 0 carries PS1 g/i mags
    # -> ps1_taylor; rows 1-2 have no photometry/catalog mass -> assumed default.
    # Row 0 is foreground (z<z_frb); rows 1-2 are background (z>z_frb).
    sight_ra, sight_dec = _sightline("11h51m07.52s", "+71d41m44.3s")
    matches = pd.DataFrame(
        {
            "ra": [sight_ra + 0.0008, sight_ra + 0.0016, sight_ra - 0.0008],
            "dec": [sight_dec + 0.0008, sight_dec - 0.0008, sight_dec + 0.0016],
            "z": [0.241, 0.519, 0.965],
            "catalog": ["VII/292/north", "VII/292/north", "VII/292/north"],
            "gmag": [20.177, math.nan, math.nan],
            "imag": [18.844, math.nan, math.nan],
            "impact_kpc": [30.0, 50.0, 60.0],
        }
    )

    df = build_unified_records(matches, 0.271, sight_ra, sight_dec, enrich=False)

    assert df.loc[0, "mass_source"] == "ps1_taylor"
    assert "taylor" in df.loc[0, "mass_method"]
    assert math.isfinite(df.loc[0, "logM_best"])
    assert list(df.loc[1:, "mass_source"]) == ["assumed", "assumed"]
    assert list(df.loc[1:, "logM_best"]) == [10.0, 10.0]
    assert df.loc[1, "cgm_extractable_flags"]["stellar_mass"] == "PREDICTED"
    assert df.loc[2, "cgm_extractable_flags"]["stellar_mass"] == "PREDICTED"

    assert df["dm_halo"].map(math.isfinite).all()
    # Row 0 (z=0.241) is foreground -> finite tau prediction. Rows 1-2
    # (z=0.519, 0.965) are background to z_frb=0.271, so the intervening-screen
    # model has no leverage and tau is "not predictable" (NaN), not a literal 0.
    assert math.isfinite(df.loc[0, "pred_tau_scat_ms_1GHz"])
    assert df.loc[0, "cgm_extractable_flags"]["tau_scat"] == "PREDICTED"
    assert df.loc[1:, "pred_tau_scat_ms_1GHz"].isna().all()
    for idx in (1, 2):
        assert df.loc[idx, "cgm_extractable_flags"]["tau_scat"] == "NOT_PREDICTABLE"
    assert ((df["cool_fc"] >= 0.0) & (df["cool_fc"] <= 1.0)).all()
    for flags in df["cgm_extractable_flags"]:
        assert flags["desi_spectro"] == "NOT_AVAILABLE"
        assert flags["wise"] == "NOT_AVAILABLE"


def test_build_for_target_uses_name_lower_path(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ra": 310.2033,
                "dec": 72.8711,
                "z": 0.04068,
                "impact_kpc": 19.4,
                "catalog": "VII/291/glade",
                "M_star": 10.0,
            }
        ]
    ).to_csv(results_dir / "zach_galaxies.csv", index=False)

    df = build_for_target(
        "Zach",
        "20h40m47.886s",
        "+72d52m56.378s",
        0.043,
        results_dir=str(results_dir),
        enrich=False,
    )

    assert (results_dir / "zach_unified.csv").exists()
    assert len(df) == 1
    assert df.loc[0, "mass_source"] == "glade_catalog"

    missing = build_for_target(
        "Missing",
        "20h40m47.886s",
        "+72d52m56.378s",
        0.043,
        results_dir=str(results_dir),
        enrich=False,
    )
    assert missing.empty
    assert not (results_dir / "missing_unified.csv").exists()


def test_empty_input_returns_empty():
    df = build_unified_records(
        pd.DataFrame(columns=["ra", "dec", "z", "impact_kpc", "catalog"]),
        0.5,
        100.0,
        70.0,
        enrich=False,
    )

    assert df.empty


def test_intersects_rvir_logic():
    matches = pd.DataFrame(
        [
            {
                "ra": 10.0,
                "dec": 10.0,
                "z": 0.05,
                "impact_kpc": 10.0,
                "catalog": "VII/291/glade",
                "M_star": 11.0,
            },
            {
                "ra": 10.1,
                "dec": 10.1,
                "z": 0.05,
                "impact_kpc": 5000.0,
                "catalog": "VII/291/glade",
                "M_star": 9.0,
            },
        ]
    )

    df = build_unified_records(matches, 0.5, 10.0, 10.0, enrich=False)

    assert bool(df.loc[0, "intersects_rvir"])
    assert not bool(df.loc[1, "intersects_rvir"])
    assert (df["intersects_rvir"] == (df["impact_kpc"] <= df["R_vir_kpc"])).all()


def test_background_galaxy_tau_is_not_predictable_nan_not_zero():
    # Two galaxies on one sightline: one foreground (z < z_frb) and one
    # background (z >= z_frb). The intervening-screen model has no leverage for
    # a background galaxy (g_scatt == 0), so its predicted scattering must be
    # NaN ("not predictable"), NOT 0.0 ("no scattering"). The foreground galaxy
    # keeps a finite prediction even with an assumed stellar mass.
    z_frb = 0.40
    matches = pd.DataFrame(
        {
            "ra": [150.0, 150.0],
            "dec": [2.0, 2.0],
            "z": [0.20, 0.50],  # foreground, background
            "impact_kpc": [30.0, 30.0],
            "catalog": ["TEST", "TEST"],
        }
    )

    df = build_unified_records(matches, z_frb, sight_ra=150.001, sight_dec=2.0, enrich=False)
    fg, bg = df.iloc[0], df.iloc[1]

    # Foreground: real intervening screen -> finite, positive tau, flagged PREDICTED.
    assert fg["g_scatt"] > 0.0
    assert np.isfinite(fg["pred_tau_scat_ms_1GHz"])
    assert fg["pred_tau_scat_ms_1GHz"] > 0.0
    assert fg["cgm_extractable_flags"]["tau_scat"] == "PREDICTED"

    # Background: no leverage -> tau (and band + scint bw) are NaN, flagged NOT_PREDICTABLE.
    assert bg["g_scatt"] == 0.0
    assert np.isnan(bg["pred_tau_scat_ms_1GHz"])
    assert np.isnan(bg["pred_tau_scat_ms_1GHz_lo"])
    assert np.isnan(bg["pred_tau_scat_ms_1GHz_hi"])
    assert np.isnan(bg["pred_scint_bw_khz"])
    assert bg["cgm_extractable_flags"]["tau_scat"] == "NOT_PREDICTABLE"


def test_unified_tau_is_two_phase_hot_plus_cool():
    # Foreground star-forming galaxy at small impact -> both hot and cool
    # scattering contributions; total tau = hot + cool, and the cool clumpy
    # phase makes the total exceed the hot-only term.
    matches = pd.DataFrame(
        {
            "ra": [150.0],
            "dec": [2.0],
            "z": [0.20],
            "impact_kpc": [25.0],
            "catalog": ["TEST"],
        }
    )
    df = build_unified_records(matches, 0.45, sight_ra=150.0008, sight_dec=2.0, enrich=False)
    r = df.iloc[0]

    for col in ("pred_tau_hot_ms_1GHz", "pred_tau_cool_ms_1GHz", "pred_tau_scat_ms_1GHz"):
        assert col in df.columns
    assert np.isfinite(r["pred_tau_scat_ms_1GHz"]) and r["pred_tau_scat_ms_1GHz"] > 0.0
    assert np.isfinite(r["pred_tau_hot_ms_1GHz"])
    # Total is the sum of the two phases.
    assert r["pred_tau_scat_ms_1GHz"] == pytest.approx(
        r["pred_tau_hot_ms_1GHz"] + np.nan_to_num(r["pred_tau_cool_ms_1GHz"]), rel=1e-6
    )
    # Cool component is present (>=0) and pushes the total at/above hot-only.
    assert r["pred_tau_scat_ms_1GHz"] >= r["pred_tau_hot_ms_1GHz"]


def test_unified_background_two_phase_tau_is_nan():
    matches = pd.DataFrame(
        {"ra": [150.0], "dec": [2.0], "z": [0.60], "impact_kpc": [25.0], "catalog": ["TEST"]}
    )
    df = build_unified_records(matches, 0.45, sight_ra=150.0008, sight_dec=2.0, enrich=False)
    r = df.iloc[0]
    assert np.isnan(r["pred_tau_scat_ms_1GHz"])
    assert np.isnan(r["pred_tau_hot_ms_1GHz"])
    assert np.isnan(r["pred_tau_cool_ms_1GHz"])
    assert r["cgm_extractable_flags"]["tau_scat"] == "NOT_PREDICTABLE"


def test_cluster_row_uses_catalog_mass_and_beta_model_dm():
    # A catalog cluster bypasses the stellar-mass ladder: M_halo = 1.3*M500,
    # dm_halo from the beta-model ICM, and dm_cool / tau zeroed (DM not scattering).
    matches = pd.DataFrame(
        {
            "ra": [312.5],
            "dec": [72.1],
            "z": [0.10],
            "impact_kpc": [800.0],
            "classification": ["cluster"],
            "m500_msun": [5.0e14],
            "r500_kpc": [1300.0],
            "catalog": ["MCXC"],
        }
    )
    out = build_unified_records(matches, 0.5, 312.4, 72.0, enrich=False)
    row = out.iloc[0]
    m200 = 1.3 * 5.0e14
    assert abs(row["M_halo"] - m200) / m200 < 1e-9  # catalog mass, not stellar-derived
    assert row["mass_source"] == "cluster_catalog"
    dm_ref = scat.dm_cluster_beta_model(5.0e14, 0.10, 800.0, r500_kpc=1300.0)
    assert abs(row["dm_halo"] - dm_ref) / dm_ref < 1e-9  # beta-model, not mNFW
    assert row["dm_cool"] == 0.0  # no cool-CGM phase for clusters
    assert row["pred_tau_scat_ms_1GHz"] == 0.0  # clusters negligible scatterers


def test_galaxy_row_unchanged_by_cluster_path():
    matches = pd.DataFrame(
        {"ra": [312.5], "dec": [72.1], "z": [0.10], "impact_kpc": [50.0], "catalog": ["NED"]}
    )
    out = build_unified_records(matches, 0.5, 312.4, 72.0, enrich=False)
    assert out.iloc[0]["mass_source"] != "cluster_catalog"  # galaxy path intact
