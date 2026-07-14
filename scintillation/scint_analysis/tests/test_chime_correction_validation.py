from __future__ import annotations

import numpy as np

from scintillation.scint_analysis.chime_correction_validation import (
    adjudicate_chime_result,
    combine_science_status,
    injection_recovery_summary,
    kernel_crosscheck,
)
from scintillation.scint_analysis.chime_product import simulate_alignment_streak


def _passing_checks():
    return {
        "manifest_verification": {"pass": True},
        "off_pulse_null": {"pass": True},
        "injection_recovery": {"pass": True},
        "low_lag_stability": {"pass": True},
        "fit_window_stability": {"pass": True},
        "split_time_stability": {"pass": True},
        "split_band_stability": {"pass": True},
        "comb_residual": {"pass": True},
        "kernel_crosscheck": {"pass": True},
        "manual_review": {"pass": True},
    }


def test_adjudication_distinguishes_measurement_upper_limit_and_diagnostic():
    measured = adjudicate_chime_result(_passing_checks(), fitted_dnu_mhz=0.021)
    assert measured["product_correction_status"] == "pass"
    assert measured["science_status"] == "measurement"

    upper = adjudicate_chime_result(_passing_checks(), fitted_dnu_mhz=None)
    assert upper["product_correction_status"] == "pass"
    assert upper["science_status"] == "upper_limit"

    checks = _passing_checks()
    checks["off_pulse_null"] = {"pass": False}
    failed = adjudicate_chime_result(checks, fitted_dnu_mhz=0.021)
    assert failed["product_correction_status"] == "fail"
    assert failed["science_status"] == "diagnostic_only"
    assert failed["failed_checks"] == ["off_pulse_null"]

    checks = _passing_checks()
    checks["manual_review"] = {"pass": None}
    pending = adjudicate_chime_result(checks, fitted_dnu_mhz=0.021)
    assert pending["product_correction_status"] == "inconclusive"
    assert pending["science_status"] == "diagnostic_only"


def test_injection_recovery_uses_bias_and_coverage_gates():
    summary = injection_recovery_summary(
        injected_mhz=np.array([0.02, 0.04, 0.08, 0.12]),
        recovered_mhz=np.array([0.0205, 0.039, 0.081, 0.118]),
        lower_mhz=np.array([0.018, 0.035, 0.075, 0.105]),
        upper_mhz=np.array([0.023, 0.044, 0.087, 0.131]),
        channel_width_mhz=0.006103515625,
        min_trials_for_coverage=4,
        coverage_tolerance=0.35,
    )
    assert summary["pass"] is True
    assert summary["max_fractional_bias"] < 0.1
    assert summary["coverage_68"] == 1.0

    biased = injection_recovery_summary(
        injected_mhz=np.array([0.02, 0.04]),
        recovered_mhz=np.array([0.03, 0.06]),
        lower_mhz=np.array([0.029, 0.059]),
        upper_mhz=np.array([0.031, 0.061]),
        channel_width_mhz=0.006103515625,
        min_trials_for_coverage=2,
    )
    assert biased["pass"] is False


def test_kernel_crosscheck_compares_corrected_acf_to_independent_prediction():
    raw = np.array([1.0, 0.30, 0.18, 0.10])
    kernel = np.array([0.0, 0.20, 0.10, 0.04])
    corrected = np.array([1.0, 0.101, 0.079, 0.061])
    verdict = kernel_crosscheck(raw, corrected, kernel, uncertainty=np.full(4, 0.01))
    assert verdict["pass"] is True
    assert verdict["max_abs_z"] < 1.0


def test_forward_model_reproduces_and_removes_alignment_streak():
    result = simulate_alignment_streak(seed=20260712)
    assert result["pre_low_lag_correlation"] > 0.08
    assert result["post_low_lag_correlation"] < 0.05
    assert result["post_low_lag_correlation"] < 0.1 * result["pre_low_lag_correlation"]
    assert result["temporal_correlation"][0] > result["temporal_correlation"][4]


def test_correction_cannot_promote_failed_provenance():
    correction = {"product_correction_status": "pass", "science_status": "measurement"}
    combined = combine_science_status(
        artifact_status="diagnostic_only", correction_status=correction
    )
    assert combined == "diagnostic_only"

    assert (
        combine_science_status(artifact_status="measurement", correction_status=correction)
        == "measurement"
    )
