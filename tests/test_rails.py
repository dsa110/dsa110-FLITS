"""Tests for the unified prior-rail classifier (ADR-0008 Gate 3).

Correctness criteria:
- A posterior with >=30% mass within 0.05 of a bound is railed (posterior-mass
  path). This is the canonical test; it catches a tight posterior pinned at the
  bound that the median-only test misses.
- A median within 3 sigma of a bound is railed (summary-only fallback), labeled
  method="summary_3sigma" so it is never mistaken for the mass test.
- An interior posterior is not railed.
- The mass test catches the failure mode the median-only test misses: a tight
  posterior pinned at the bound whose *median* is >0.1 from it (gate_joint_committed's
  RAIL_EDGE=0.1 median-only test would false-pass this).
"""

import numpy as np

from flits.fitting.rails import EDGE_MASS_FRAC, classify_rail


def test_posterior_mass_railed_hi():
    # 40% of mass in the top 0.05 window of [3.0, 4.0], rest well inside => railed-hi
    s = np.array([3.3] * 30 + [3.97] * 40 + [3.5] * 30, dtype=float)
    v = classify_rail(lo=3.0, hi=4.0, samples=s)
    assert v.railed is True
    assert v.railed_hi is True
    assert v.railed_lo is False
    assert v.cls == "railed-hi"
    assert v.method == "posterior_mass"
    assert v.edge_mass_hi >= EDGE_MASS_FRAC


def test_interior_not_railed():
    rng = np.random.default_rng(0)
    s = rng.normal(3.5, 0.1, 2000)
    s = np.clip(s, 3.0 + 0.1, 4.0 - 0.1)  # well inside [3.0, 4.0]
    v = classify_rail(lo=3.0, hi=4.0, samples=s)
    assert v.railed is False
    assert v.cls == "interior"


def test_mass_test_catches_tight_pin_that_median_only_misses():
    # 35% of mass at 3.96 (in the top 0.05 window), 65% around 3.5 => median ~3.5,
    # which is >0.1 from the 4.0 bound, so gate_joint_committed's RAIL_EDGE=0.1
    # median-only test would NOT flag this. The mass test must flag it.
    s = np.array([3.96] * 35 + [3.5] * 65, dtype=float)
    v = classify_rail(lo=3.0, hi=4.0, samples=s)
    assert v.railed is True
    assert v.railed_hi is True
    assert v.edge_mass_hi >= EDGE_MASS_FRAC
    # The median is >0.1 from the bound, so the median-only test fails here.
    assert abs(np.median(s) - 4.0) > 0.1


def test_summary_fallback_railed():
    # median + 3*err_plus reaches the hi bound => railed-hi, summary path
    v = classify_rail(lo=3.0, hi=4.0, median=3.9, err_minus=0.05, err_plus=0.05)
    assert v.railed is True
    assert v.railed_hi is True
    assert v.method == "summary_3sigma"
    assert v.edge_mass_hi is None


def test_summary_fallback_interior():
    v = classify_rail(lo=3.0, hi=4.0, median=3.5, err_minus=0.1, err_plus=0.1)
    assert v.railed is False
    assert v.cls == "interior"
    assert v.method == "summary_3sigma"


def test_weights_respected():
    # 5 samples, 1 in the top window => 20% unweighted => not railed.
    s = np.array([3.0, 3.2, 3.4, 3.6, 3.97], dtype=float)
    v = classify_rail(lo=3.0, hi=4.0, samples=s)
    assert v.railed is False
    # Weight the 3.97 sample to 50% => railed.
    w = np.array([0.125, 0.125, 0.125, 0.125, 0.5])
    v_w = classify_rail(lo=3.0, hi=4.0, samples=s, weights=w)
    assert v_w.railed is True
    assert v_w.railed_hi is True


def test_unconstrained_both_bounds():
    s = np.array([3.02] * 40 + [3.98] * 40 + [3.5] * 20, dtype=float)
    v = classify_rail(lo=3.0, hi=4.0, samples=s)
    assert v.railed_hi is True
    assert v.railed_lo is True
    assert v.cls == "unconstrained"


def test_missing_inputs_raises():
    import pytest

    with pytest.raises(ValueError):
        classify_rail(lo=3.0, hi=4.0)  # neither samples nor summary
    with pytest.raises(ValueError):
        classify_rail(lo=3.0, hi=4.0, median=3.5)  # partial summary
