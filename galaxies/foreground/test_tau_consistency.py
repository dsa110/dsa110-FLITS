"""Tests for tau_consistency JSON loading and refit runner errors."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from galaxies.foreground.run_tau_consistency_refits import run_burst
from galaxies.foreground.tau_consistency import (
    _joint_fit_scalar,
    _posterior_median,
    find_allexp_joint_json,
    load_joint_free_alpha,
    tau_consistency_from_refit,
)


def test_posterior_median_dict_and_scalar():
    assert _posterior_median({"median": 0.5}) == 0.5
    assert _posterior_median(0.061) == 0.061
    assert np.isnan(_posterior_median(None))


def test_joint_fit_scalar_ppc_payload():
    payload = {"tau_1ghz": 0.06086799947757, "alpha": 2.396}
    assert _joint_fit_scalar(payload, "tau_1ghz") == 0.06086799947757
    assert _joint_fit_scalar(payload, "alpha") == 2.396


def test_tau_consistency_from_refit_scalar():
    row = tau_consistency_from_refit({"tau_1ghz": 0.1, "alpha": 4.0})
    assert row["tau_consistency_1ghz_ms"] == 0.1
    assert row["refit_status"] == "alpha4_joint_complete"


def test_load_joint_free_alpha_scalar_json(tmp_path, monkeypatch):
    ppc = {
        "burst": "fake",
        "tau_1ghz": 0.42,
        "alpha": 3.1,
        "suffix": "_ppc",
    }
    fit = {
        "burst": "fake",
        "percentiles": {
            "tau_1ghz": {"median": 0.99},
            "alpha": {"median": 4.0},
        },
    }
    root = tmp_path / "fits"
    root.mkdir()
    (root / "fake_joint_ppc_multi_pbf-exp-exp.json").write_text(json.dumps(ppc))
    (root / "fake_joint_fit_sharedzeta_pbf-exp-exp.json").write_text(json.dumps(fit))
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.JOINT_GATE_CSV",
        tmp_path / "missing_gate.csv",
    )
    monkeypatch.setattr(
        "galaxies.foreground.tau_consistency.ALLEXP_FITS_DIR",
        root,
    )
    assert find_allexp_joint_json("fake") == root / "fake_joint_fit_sharedzeta_pbf-exp-exp.json"
    loaded = load_joint_free_alpha("fake")
    assert loaded["tau_joint_1ghz_ms"] == 0.99
    assert loaded["alpha_joint_free"] == 4.0


def test_run_burst_raises_when_joint_output_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "galaxies.foreground.run_tau_consistency_refits.TAU_CONSISTENCY_DIR",
        tmp_path / "tau_consistency",
    )
    monkeypatch.setenv("FLITS_RUNS", str(tmp_path / "runs"))
    with patch("galaxies.foreground.run_tau_consistency_refits.subprocess.run"):
        with pytest.raises(FileNotFoundError, match="expected output missing"):
            run_burst("casey")
