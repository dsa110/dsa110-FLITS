"""Tests for verify_fits labeling (pure, no pipeline import needed)."""

import json
import math

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_fits as vf


def test_quality_from_chi2_bands():
    assert vf._quality_from_chi2(1.0) == "PASS"
    assert vf._quality_from_chi2(1.36) == "PASS"
    assert vf._quality_from_chi2(3.9) == "MARGINAL"
    assert vf._quality_from_chi2(0.1) == "MARGINAL"
    assert vf._quality_from_chi2(69.0) == "FAIL"
    assert vf._quality_from_chi2(float("nan")) == "FAIL"


def test_delta_logz_prefers_scattering():
    allres = {
        "M0": {"log_evidence": 9938.0},
        "M1": {"log_evidence": 10763.0},
        "M3": {"log_evidence": 11697.0},
    }
    # M3 - max(M0,M1) = 11697 - 10763 = 934
    assert vf._delta_logz_scattering(allres) == pytest.approx(934.0)


def test_label_detection_when_pass_and_decisive_evidence():
    data = {
        "best_model": "M3",
        "best_params_percentiles": {"tau_1ghz": {"median": 0.194, "err_minus": 0.02, "err_plus": 0.03}},
        "goodness_of_fit": {"chi2_reduced": 1.36, "quality_flag": "PASS"},
        "all_results": {"M0": {"log_evidence": 9938.0}, "M3": {"log_evidence": 11697.0}},
    }
    out = vf.label_fit(data)
    assert out["label"] == "DETECTION"
    assert out["locked_in"] is True
    assert out["tau_ms"] == pytest.approx(0.194)


def test_label_upper_limit_when_pass_but_no_decisive_scattering():
    data = {
        "best_model": "M3",
        "best_params_percentiles": {"tau_1ghz": {"median": 0.01, "err_minus": 0.005, "err_plus": 0.005}},
        "goodness_of_fit": {"chi2_reduced": 1.1, "quality_flag": "PASS"},
        "all_results": {"M0": {"log_evidence": 100.0}, "M3": {"log_evidence": 101.0}},  # dlogZ=1 < 5
    }
    out = vf.label_fit(data)
    assert out["label"] == "UPPER-LIMIT"
    assert out["locked_in"] is False


def test_label_unfittable_on_fail():
    data = {
        "best_model": "M3",
        "goodness_of_fit": {"chi2_reduced": 69.0, "quality_flag": "FAIL"},
        "best_params": {"tau_1ghz": 0.019},
    }
    out = vf.label_fit(data)
    assert out["label"] == "UNFITTABLE"
    assert out["locked_in"] is False


def test_label_uses_chi2_when_flag_absent():
    data = {
        "best_model": "M3",
        "goodness_of_fit": {"chi2_reduced": 3.9},  # no stored flag
        "best_params": {"tau_1ghz": 0.1},
        "all_results": {"M0": {"log_evidence": 1.0}, "M3": {"log_evidence": 50.0}},
    }
    out = vf.label_fit(data)
    assert out["quality_flag"] == "MARGINAL"
    assert out["label"] == "MARGINAL"


def test_summarize_dir(tmp_path):
    (tmp_path / "aaa_fit_results.json").write_text(json.dumps({
        "best_model": "M3",
        "best_params_percentiles": {"tau_1ghz": {"median": 0.2, "err_minus": 0.01, "err_plus": 0.01}},
        "goodness_of_fit": {"chi2_reduced": 1.2, "quality_flag": "PASS"},
        "all_results": {"M0": {"log_evidence": 0.0}, "M3": {"log_evidence": 20.0}},
    }))
    rows = vf.summarize_dir(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["burst"] == "aaa"
    assert rows[0]["label"] == "DETECTION"
