"""Tests for the #106 verdict builder: bar-evaluation and claim-band logic on
synthetic inputs (deterministic, no data), plus a slow real-artifact roundtrip.
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


bv = _load("build_verdict", _HERE / "build_verdict.py")


def test_unrailed_quantification():
    r = bv.unrailed({"median": 3.684, "err_minus": 0.0136, "err_plus": 0.0128})
    assert r["unrailed"] is True
    assert r["sigma_from_lower"] == pytest.approx((3.684 - 3.0) / 0.0136)
    assert r["sigma_from_upper"] == pytest.approx((4.0 - 3.684) / 0.0128)


def test_railed_at_upper_edge():
    r = bv.unrailed({"median": 3.99, "err_minus": 0.01, "err_plus": 0.005})
    assert r["unrailed"] is False  # 2 sigma from beta=4: railed
    assert r["sigma_from_upper"] == pytest.approx(2.0)


def test_grade_pass():
    assert bv.grade(1.18, 1.03, "agree", True) == "PASS"


def test_grade_marginal_per_kernel_contract():
    # kernel contract: finite 1.5 < chi2 <= 10 is MARGINAL, not FAIL
    assert bv.grade(1.8, 1.03, "agree", True) == "MARGINAL"
    assert bv.grade(4.6, 1.03, "agree", True) == "MARGINAL"
    assert bv.grade(10.0, 1.03, "agree", True) == "MARGINAL"
    assert bv.grade(0.2, 1.03, "agree", True) == "MARGINAL"  # suspiciously low side


def test_grade_band_edges_pass():
    assert bv.grade(0.3, 1.5, "agree", True) == "PASS"  # both edges inclusive


def test_grade_fail_closed():
    assert bv.grade(10.1, 1.03, "agree", True) == "FAIL"  # catastrophic chi2
    assert bv.grade(float("nan"), 1.03, "agree", True) == "FAIL"  # non-finite
    assert bv.grade(float("inf"), 1.03, "agree", True) == "FAIL"
    assert bv.grade(None, 1.03, "agree", True) == "FAIL"
    assert bv.grade(1.18, 1.03, "shifted", True) == "FAIL"  # A-vs-B stop verdict
    assert bv.grade(1.18, 1.03, "widened", True) == "FAIL"  # widened IS a breach (#105)
    assert bv.grade(1.18, 1.03, "agree", False) == "FAIL"  # railed beta


def test_chi2_flag_parity_with_kernel():
    # pin the mirror against the real classify_fit_quality on a boundary grid
    sys.path.insert(0, str(bv.REPO / "scattering"))
    from scat_analysis.burstfit import classify_fit_quality

    grid = [0.05, 0.2, 0.3, 0.9, 1.5, 1.6, 4.6, 10.0, 10.1, 50.0, float("nan"), float("inf")]
    for c in grid:
        kernel_flag, _ = classify_fit_quality(c)
        assert bv._chi2_flag(c) == kernel_flag, c


@pytest.mark.slow
def test_real_artifacts_roundtrip():
    needed = (bv.ROUTE_B_JSON, bv.ROUTE_B_NPZ, bv.ROUTE_A_JSON, bv.A_VS_B_JSON, bv.EXP_ERA_JSON)
    if not all(p.exists() for p in needed):
        pytest.skip("DAG artifacts not present")
    v = bv.build(preflight=False)  # skip the CHIME prep; data may be absent
    assert v["provisional_citable_bar"]["grade"] == "PASS"
    assert v["provisional_citable_bar"]["provisional_citable"] is True
    assert v["a_vs_b"]["verdict"] == "agree"
    assert v["exp_era_comparison"]["within_claim_band"] is True
    assert v["exp_era_comparison"]["comparator"]["verdict"] in ("agree", "widened")
    assert abs(v["x_zeta_beta_corr"]) < 0.1
    md = bv.render_md({**v, "tail_coverage_at_fit": None})
    assert "SENSITIVITY-REGIME CAVEAT" in md
    assert "beta-table row candidate" in md
