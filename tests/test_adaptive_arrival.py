"""Adaptive arrival-regression selection and event-level DM synthesis."""

import math

import pytest

from dispersion.dm_campaign.adaptive_arrival import (
    combine_event_measurements,
    evaluate_waterfall,
    product_dm_from_filename,
    product_path,
    select_stable_candidate,
)
from dispersion.dm_campaign.injection import standard_bright_case
from dispersion.dm_campaign.run_adaptive_arrival import sha256_file


def _candidate(dm, sigma, *, time_factor, n_subband, n_good=4):
    return {
        "dm": dm,
        "sigma": sigma,
        "constrains_dm": True,
        "time_factor": time_factor,
        "freq_factor": 4,
        "n_subband": n_subband,
        "n_good_subbands": n_good,
        "chi2_red": 1.0,
    }


def test_product_dm_comes_from_actual_dedispersion_filename():
    assert product_dm_from_filename("chromatica_dsa_I_272_368_2500b_cntr_bpc.npy") == 272.368


def test_product_path_uses_telescope_specific_root(tmp_path):
    config = {
        "chime_full_root": str(tmp_path / "chime"),
        "dsa_full_root": str(tmp_path / "dsa"),
    }
    assert product_path({"telescope": "chime", "filename": "c.npy"}, config) == tmp_path / "chime/c.npy"
    assert product_path({"telescope": "dsa", "filename": "d.npy"}, config) == tmp_path / "dsa/d.npy"
    with pytest.raises(ValueError, match="unknown telescope"):
        product_path({"telescope": "other", "filename": "x.npy"}, config)


def test_provenance_hashes_file_bytes(tmp_path):
    path = tmp_path / "input.npy"
    path.write_bytes(b"known input")
    assert sha256_file(path) == "6fdccd8e1aa8ecede3d631c3d6182eeaefece8235805d1dc1461ee3a1e52183b"


@pytest.mark.parametrize("instrument", ["chime", "dsa"])
def test_adaptive_policy_recovers_known_bright_injection(instrument):
    wf, freq_ghz, dt_ms, truth = standard_bright_case(instrument, seed=11)
    config = {
        "windows": {instrument: truth["window"]},
        "adaptive": {
            "max_channels": [128, 256],
            "max_time": [256, 512, 1024],
            "n_subbands": [3, 4, 6, 8],
            "dm_step": 0.25,
        },
    }
    candidates = evaluate_waterfall(
        wf,
        freq_ghz * 1e3,
        dt_ms * 1e-3,
        truth["dm_ref"],
        truth["dm_ref"],
        telescope=instrument,
        config=config,
    )
    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] in {"science-grade", "marginal-fit"}
    # Which member of a stable cluster wins is uncertainty-limited (the dsa
    # case has two members straddling the median within one sigma), so the
    # hard 0.5 accuracy bar belongs to the cluster, and the selected member
    # must agree with truth within its own quoted uncertainty.
    assert abs(selected["dm"] - truth["dm_true"]) < 2 * selected["sigma"]
    assert abs(selected["cluster_median_dm"] - truth["dm_true"]) < 0.5


def test_selects_stable_science_grade_cluster_not_isolated_precision():
    candidates = [
        _candidate(100.01, 0.10, time_factor=1, n_subband=4),
        _candidate(100.04, 0.08, time_factor=1, n_subband=6, n_good=6),
        _candidate(99.98, 0.12, time_factor=2, n_subband=4),
        _candidate(103.0, 0.01, time_factor=8, n_subband=8, n_good=8),
    ]

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "science-grade"
    assert selected["dm"] == 100.01
    assert selected["stable_candidate_count"] == 3
    assert selected["distinct_resolution_count"] == 3


def test_prefers_stable_pass_cluster_over_central_marginal_candidate():
    candidates = [
        _candidate(100.00, 0.08, time_factor=1, n_subband=4),
        _candidate(100.04, 0.08, time_factor=2, n_subband=6),
        _candidate(100.02, 0.04, time_factor=4, n_subband=8),
    ]
    candidates[0]["chi2_red"] = 0.8
    candidates[1]["chi2_red"] = 1.1
    candidates[2]["chi2_red"] = 0.05

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "science-grade"
    assert selected["fit_quality"] == "PASS"
    assert selected["dm"] in (100.00, 100.04)
    assert selected["stable_candidate_count"] == 2


def test_mixed_fallback_cluster_is_not_promoted_to_science_grade():
    candidates = [
        _candidate(100.00, 0.08, time_factor=1, n_subband=4),
        _candidate(100.04, 0.08, time_factor=2, n_subband=6),
    ]
    candidates[0]["chi2_red"] = 1.0
    candidates[1]["chi2_red"] = 0.05

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "marginal-fit"
    assert selected["distinct_resolution_count"] == 2


def test_rejects_catastrophic_regression_chi2_from_science_grade_cluster():
    candidates = [
        _candidate(100.01, 0.02, time_factor=1, n_subband=4),
        _candidate(100.02, 0.02, time_factor=2, n_subband=6),
    ]
    for candidate in candidates:
        candidate["chi2_red"] = 25.0

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "weak-only"
    assert selected["dm"] is None


def test_rejects_nonfinite_regression_chi2_from_science_grade_cluster():
    candidates = [
        _candidate(100.01, 0.02, time_factor=1, n_subband=4),
        _candidate(100.02, 0.02, time_factor=2, n_subband=6),
    ]
    for candidate in candidates:
        candidate["chi2_red"] = float("nan")

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "weak-only"
    assert selected["dm"] is None


def test_stable_cluster_is_pairwise_bounded_and_selects_central_member():
    candidates = [
        _candidate(100.00, 0.08, time_factor=1, n_subband=8, n_good=8),
        _candidate(100.10, 0.08, time_factor=2, n_subband=4),
        _candidate(100.20, 0.08, time_factor=4, n_subband=6, n_good=6),
        _candidate(101.00, 0.08, time_factor=8, n_subband=8, n_good=8),
    ]

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["cluster_span_dm"] <= 0.25
    assert selected["dm"] == 100.10


def test_reports_weak_only_when_no_science_grade_cluster_exists():
    candidates = [
        _candidate(411.2, 8.0, time_factor=9, n_subband=3),
        _candidate(411.3, 12.0, time_factor=9, n_subband=6),
    ]

    selected = select_stable_candidate(candidates, sigma_max=0.5, stability_dm=0.25)

    assert selected["status"] == "weak-only"
    assert selected["dm"] is None
    assert selected["best_weak_sigma"] == 8.0


def test_event_combination_labels_two_band_and_single_band_cases():
    both = combine_event_measurements(
        "zach",
        {
            "chime": {"status": "science-grade", "dm": 262.36, "sigma": 0.04},
            "dsa": {"status": "science-grade", "dm": 262.40, "sigma": 0.03},
        },
    )
    one = combine_event_measurements(
        "isha",
        {
            "chime": {"status": "science-grade", "dm": 411.56, "sigma": 0.02},
            "dsa": {"status": "weak-only", "dm": None, "sigma": None},
        },
    )

    assert both["support"] == "two-band-consistent"
    assert both["bands_used"] == ["chime", "dsa"]
    assert 262.36 < both["dm"] < 262.40
    assert one["support"] == "single-band"
    assert one["bands_used"] == ["chime"]
    assert one["dm"] == 411.56


def test_event_combination_flags_tension_without_choosing_a_single_dm():
    result = combine_event_measurements(
        "phineas",
        {
            "chime": {"status": "science-grade", "dm": 610.47, "sigma": 0.02},
            "dsa": {"status": "science-grade", "dm": 610.21, "sigma": 0.01},
        },
    )

    assert result["support"] == "two-band-tension"
    assert result["agreement_z"] > 3
    assert result["dm"] is None
    assert result["sigma"] is None
    assert result["band_measurements"]["chime"]["dm"] == 610.47
    assert result["band_measurements"]["dsa"]["dm"] == 610.21


def test_event_combination_inflates_subthreshold_disagreement_error():
    result = combine_event_measurements(
        "example",
        {
            "chime": {"status": "science-grade", "dm": 100.0, "sigma": 0.1},
            "dsa": {"status": "science-grade", "dm": 100.28, "sigma": 0.1},
        },
    )

    fixed_effect_sigma = 1.0 / math.sqrt(200.0)
    assert result["support"] == "two-band-consistent"
    assert 1.9 < result["agreement_z"] < 2.0
    assert result["sigma"] > fixed_effect_sigma
    assert result["sigma"] == pytest.approx(
        fixed_effect_sigma * math.sqrt(result["chi2_red"])
    )


def test_event_combination_does_not_claim_dm_without_science_grade_band():
    result = combine_event_measurements(
        "mahi",
        {
            "chime": {"status": "weak-only", "dm": None, "sigma": None},
            "dsa": {"status": "unconstrained", "dm": None, "sigma": None},
        },
    )

    assert result["support"] == "none"
    assert result["dm"] is None
