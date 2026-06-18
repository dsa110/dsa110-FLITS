import math
import os

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from galaxies.v2_0.build_unified import build_for_target, build_unified_records


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)


def _sightline(ra_str, dec_str):
    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return coord.ra.deg, coord.dec.deg


def test_zach_unified_columns_and_glade_mass():
    sight_ra, sight_dec = _sightline("20h40m47.886s", "+72d52m56.378s")
    matches = pd.read_csv(os.path.join(RESULTS_DIR, "zach_galaxies.csv"))

    df = build_unified_records(matches, 0.043, sight_ra, sight_dec, enrich=False)

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


def test_phineas_unified_mass_sources_and_predictions():
    sight_ra, sight_dec = _sightline("11h51m07.52s", "+71d41m44.3s")
    matches = pd.read_csv(os.path.join(RESULTS_DIR, "phineas_galaxies.csv"))

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
            "z": [0.20, 0.50],          # foreground, background
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
