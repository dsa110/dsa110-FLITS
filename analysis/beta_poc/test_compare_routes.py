"""Wiring tests for the #105 Route A-vs-B comparison driver.

The comparator itself is #100's pure function with its own test suite
(test_posterior_compare.py); these pin the driver's #105-specific contract:
physics-params selection, stop-condition semantics, and artifact bookkeeping.
Synthetic dict artifacts only -- no fit outputs required.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load("compare_routes", _HERE / "compare_routes.py")


def _triplet(median, err=0.01):
    return {"median": median, "err_minus": err, "err_plus": err}


def _poc_artifact(**medians):
    # run_beta_poc.py summary layout: {"median": {param: triplet}}
    return {"median": {k: _triplet(v) for k, v in medians.items()}}


def _joint_artifact(**medians):
    # run_joint_fit.py summary layout: {"percentiles": {param: triplet}}
    return {"percentiles": {k: _triplet(v) for k, v in medians.items()}}


_BASE = dict(
    beta=3.684, tau_1ghz=0.1144, zeta_1ghz=0.0855, x_zeta=-0.752, t0_C=17.26, delta_dm_C=0.012
)


def test_physics_params_only():
    a = _poc_artifact(**_BASE)
    # t0_C deliberately shifted wildly in B: must NOT affect the verdict.
    b = _joint_artifact(**{**_BASE, "t0_C": 99.0})
    result = cr.compare(a, b)
    assert sorted(result["params"]) == sorted(cr.PHYSICS_PARAMS)
    assert "t0_C" not in result["params"]
    assert result["verdict"] == "agree"
    assert result["stop_condition_triggered"] is False


def test_stop_condition_on_shift():
    a = _poc_artifact(**_BASE)
    b = _joint_artifact(**{**_BASE, "beta": _BASE["beta"] + 0.1})  # ~7 sigma at err=0.01*sqrt2
    result = cr.compare(a, b)
    assert result["params"]["beta"]["verdict"] == "incompatible"
    assert result["verdict"] == "incompatible"
    assert result["stop_condition_triggered"] is True


def test_stop_condition_on_widened():
    a = _poc_artifact(**_BASE)
    b = _joint_artifact(**_BASE)
    # triple B's beta width: ratio 1/3 < 1/width_ratio_max -> widened is a breach
    b["percentiles"]["beta"] = _triplet(_BASE["beta"], err=0.03)
    result = cr.compare(a, b)
    assert result["params"]["beta"]["verdict"] == "widened"
    assert result["verdict"] == "widened"
    assert result["stop_condition_triggered"] is True


def test_bookkeeping_fields():
    a = _poc_artifact(**_BASE)
    b = _joint_artifact(**_BASE)
    result = cr.compare(a, b)
    assert result["issue"] == "dsa110-FLITS#105"
    assert set(result["routes"]) == {"a", "b"}
    assert result["thresholds"]["shift_sigma_max"] == 2.0  # comparator defaults ARE the tolerance


@pytest.mark.slow
def test_real_artifacts_roundtrip():
    # Application test: only meaningful once both fit artifacts exist on disk.
    if not (cr.ROUTE_A_JSON.exists() and cr.ROUTE_B_NPZ.exists()):
        pytest.skip("real freya fit artifacts not present")
    result = cr.compare()
    assert sorted(result["params"]) == sorted(cr.PHYSICS_PARAMS)
    for p in result["params"].values():
        assert p["shift_sigma"] >= 0.0
