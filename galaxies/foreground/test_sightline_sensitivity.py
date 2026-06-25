import math

import numpy as np
import pandas as pd
import pytest

from galaxies.foreground import sightline_sensitivity as ss


def test_default_prior_families_have_expected_names_and_bounds():
    families = ss.default_prior_families()
    assert set(families) == {
        "fiducial_literature",
        "conservative_low_cgm",
        "aggressive_cgm_scattering",
    }
    for family in families.values():
        assert family.dm_mw_halo_mean > 0.0
        assert family.dm_mw_halo_sigma > 0.0
        assert 0.0 < family.f_igm_min < family.f_igm_max < 1.0
        assert family.measured_mass_sigma_dex > 0.0
        assert family.assumed_logmstar_min < family.assumed_logmstar_max
        assert family.cool_boost_min > 0.0
        assert family.cool_boost_min < family.cool_boost_max


def test_sample_nuisance_draws_are_reproducible_and_finite():
    family = ss.default_prior_families()["fiducial_literature"]
    a = ss.sample_nuisance_draws(family, n=5, seed=123)
    b = ss.sample_nuisance_draws(family, n=5, seed=123)
    assert a.equals(b)
    assert len(a) == 5
    for col in (
        "dm_mw_halo",
        "f_igm",
        "mass_shift_measured_dex",
        "mass_shift_assumed_dex",
        "f_hot",
        "cool_dm_factor",
        "cool_boost",
        "cosmic_scatter",
        "placeholder_z",
    ):
        assert col in a.columns
        assert np.isfinite(a[col]).all()
    assert ((a["f_igm"] > 0.0) & (a["f_igm"] < 1.0)).all()
    assert (a["cool_boost"] > 0.0).all()


def test_apply_draw_to_budget_with_measured_redshift_keeps_real_z():
    budget = {
        "name": "Aaa",
        "z_frb": 0.30,
        "z_is_placeholder": False,
        "dm_obs": 400.0,
        "dm_mw_ism": 80.0,
        "dm_intervening_capped": 50.0,
        "dm_intervening": 60.0,
        "tau_intervening_ms": 0.02,
        "tau_intervening_hi": 0.06,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": None,
        "dm_intervening_regime": "CGM",
        "intervening_mass_confidence": "measured",
    }
    draw = ss.sample_nuisance_draws(ss.default_prior_families()["fiducial_literature"], n=1, seed=1).iloc[0]
    row = ss.apply_draw_to_budget(budget, draw)
    assert row["name"] == "Aaa"
    assert row["z_used"] == pytest.approx(0.30)
    assert row["z_status"] == "measured"
    expected_cosmic = ss.scaled_dm_cosmic(0.30, draw["f_igm"], draw["cosmic_scatter"])
    assert row["dm_cosmic"] == pytest.approx(expected_cosmic)
    expected_host = 400.0 - 80.0 - draw["dm_mw_halo"] - expected_cosmic - row["dm_intervening_capped"]
    assert row["dm_host_capped"] == pytest.approx(expected_host)


def test_apply_draw_to_budget_with_placeholder_redshift_is_hypothetical():
    budget = {
        "name": "Freya",
        "z_frb": 1.0,
        "z_is_placeholder": True,
        "dm_obs": 912.0,
        "dm_mw_ism": 68.0,
        "dm_intervening_capped": 4.0,
        "dm_intervening": 5.0,
        "tau_intervening_ms": 0.001,
        "tau_intervening_hi": 0.003,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": "FAIL",
        "dm_intervening_regime": "CGM",
        "intervening_mass_confidence": "assumed",
    }
    draw = ss.sample_nuisance_draws(ss.default_prior_families()["fiducial_literature"], n=1, seed=2).iloc[0]
    row = ss.apply_draw_to_budget(budget, draw)
    assert row["z_status"] == "placeholder_hypothetical"
    assert row["z_used"] == pytest.approx(draw["placeholder_z"])
    assert math.isfinite(row["dm_cosmic"])
    assert row["hypothetical_placeholder_z"] is True
    assert row["prior_dominated"] is True


def test_run_sensitivity_returns_draws_for_each_family():
    budgets = [
        {
            "name": "Aaa",
            "z_frb": 0.30,
            "z_is_placeholder": False,
            "dm_obs": 400.0,
            "dm_mw_ism": 80.0,
            "dm_intervening_capped": 50.0,
            "dm_intervening": 60.0,
            "tau_intervening_ms": 0.02,
            "tau_intervening_hi": 0.06,
            "tau_obs_ms": math.nan,
            "tau_obs_quality": None,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ]
    draws = ss.run_sensitivity(budgets, n_per_family=4, seed=99)
    assert len(draws) == 12
    assert set(draws["prior_family"]) == set(ss.default_prior_families())
    assert set(draws["name"]) == {"Aaa"}


def test_summarize_sensitivity_labels_placeholder_and_prior_dominated():
    rows = []
    for i in range(10):
        rows.append({
            "name": "Freya",
            "prior_family": "fiducial_literature",
            "dm_host_capped": -10.0 if i < 9 else 20.0,
            "host_negative": i < 9,
            "interv_dm_gt_100": False,
            "tau_gt_0p1ms": False,
            "tau_gt_obs_over_10": False,
            "hypothetical_placeholder_z": True,
            "prior_dominated": True,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "assumed",
        })
    summary = ss.summarize_sensitivity(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["name"] == "Freya"
    assert row["p_host_negative"] == pytest.approx(0.9)
    assert row["host_budget_label"] == "robust_negative_host"
    assert bool(row["placeholder_z_hypothetical"]) is True
    assert bool(row["prior_dominated"]) is True


def test_format_summary_markdown_declares_prior_predictive():
    summary = pd.DataFrame([
        {
            "name": "Aaa",
            "n_draws": 30,
            "p_host_negative": 0.91,
            "p_interv_dm_gt_100": 0.2,
            "p_tau_gt_0p1ms": 0.0,
            "p_tau_gt_obs_over_10": 0.0,
            "dm_host_cap_median": -12.0,
            "dm_host_cap_p16": -30.0,
            "dm_host_cap_p84": 5.0,
            "dm_interv_cap_median": 70.0,
            "tau_interv_median_ms": 0.02,
            "host_budget_label": "robust_negative_host",
            "placeholder_z_hypothetical": False,
            "prior_dominated": False,
            "robustness_label": "robust_negative_host",
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ])
    md = ss.format_summary_markdown(summary)
    assert "prior-predictive" in md.lower()
    assert "Aaa" in md
    assert "robust_negative_host" in md


def test_write_sensitivity_artifacts_creates_expected_files(tmp_path):
    draws = pd.DataFrame([
        {
            "name": "Aaa",
            "prior_family": "fiducial_literature",
            "draw_id": 0,
            "dm_host_capped": -1.0,
            "host_negative": True,
            "interv_dm_gt_100": False,
            "tau_gt_0p1ms": False,
            "tau_gt_obs_over_10": False,
            "dm_intervening_capped": 10.0,
            "tau_intervening_ms": 0.001,
            "hypothetical_placeholder_z": False,
            "prior_dominated": False,
            "dm_intervening_regime": "CGM",
            "intervening_mass_confidence": "measured",
        }
    ])
    summary = ss.summarize_sensitivity(draws)
    paths = ss.write_sensitivity_artifacts(draws, summary, output_dir=tmp_path)
    for key in ("draws_csv", "summary_csv", "summary_md", "priors_yaml"):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0


def test_knob_sweep_varies_one_parameter_for_iconic_sightline():
    budget = {
        "name": "Whitney",
        "z_frb": 0.479,
        "z_is_placeholder": False,
        "dm_obs": 462.0,
        "dm_mw_ism": 46.0,
        "dm_intervening_capped": 200.0,
        "dm_intervening": 364.0,
        "tau_intervening_ms": 0.27,
        "tau_intervening_hi": 0.96,
        "tau_obs_ms": math.nan,
        "tau_obs_quality": None,
        "dm_intervening_regime": "GALAXY_INTERIOR",
        "intervening_mass_confidence": "assumed",
    }
    sweep = ss.one_parameter_sweep(budget, parameter="dm_mw_halo", values=[20.0, 40.0, 80.0])
    assert list(sweep["parameter_value"]) == [20.0, 40.0, 80.0]
    assert sweep["dm_host_capped"].iloc[0] > sweep["dm_host_capped"].iloc[-1]
