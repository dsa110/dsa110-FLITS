"""Tests for the A1 escalation-trigger evidence engine (limb i).

Nested-sampling model comparison: single Lorentzian (M1) vs the physical
two-screen form lor1 + lor2 + lor1*lor2 + c (M2, width-ordered via the
gamma2 = f*gamma1 prior). Synthetic truth both directions, plus the
prior-edge rail guard. See docs/rse/specs/plan-a1-trigger-calibration.md
(Faber2026) Phase 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

from scint_analysis.acf_evidence import (  # noqa: E402
    compare_acf_evidence,
    lorentzian_1,
    two_screen_model,
)


def _synth_acf(rng, gammas=(0.4,), mods=(0.8,), c=0.0, n=120, dlag=0.05, sig=0.02):
    """One-sided synthetic ACF with iid Gaussian noise (lag-0 excluded)."""
    lags = dlag * np.arange(1, n + 1)
    if len(gammas) == 1:
        acf = lorentzian_1(lags, gammas[0], mods[0] ** 2, c)
    else:
        f = gammas[1] / gammas[0]
        acf = two_screen_model(lags, gammas[0], mods[0] ** 2, f, mods[1] ** 2, c)
    return lags, acf + rng.normal(0.0, sig, n), np.full(n, sig)


def test_single_screen_prefers_m1():
    rng = np.random.default_rng(11)
    lags, acf, err = _synth_acf(rng)
    res = compare_acf_evidence(
        lags, acf, cov=np.diag(err**2),
        channel_width_mhz=0.05, band_width_mhz=6.0,
        nlive=300, dlogz=0.5, seed=11,
    )
    assert res["dlnz"] < 3.0  # no strong two-screen preference on 1-screen truth
    assert res["m1"]["logz_err"] < 0.5


def test_two_screen_prefers_m2():
    rng = np.random.default_rng(12)
    lags, acf, err = _synth_acf(rng, gammas=(0.15, 1.8), mods=(0.7, 0.6))
    res = compare_acf_evidence(
        lags, acf, cov=np.diag(err**2),
        channel_width_mhz=0.05, band_width_mhz=6.0,
        nlive=300, dlogz=0.5, seed=12,
    )
    assert res["dlnz"] > 5.0
    g = res["m2"]["params_median"]
    assert g["gamma2"] > 3.0 * g["gamma1"]  # width ordering held by the f prior


def test_rail_flag_field_present_and_valid():
    rng = np.random.default_rng(13)
    lags, acf, err = _synth_acf(rng)  # single-screen truth
    res = compare_acf_evidence(
        lags, acf, cov=np.diag(err**2),
        channel_width_mhz=0.05, band_width_mhz=6.0,
        nlive=300, dlogz=0.5, seed=13,
    )
    assert "rail_flags" in res["m2"]
    assert set(res["m2"]["rail_flags"]) <= {"gamma1", "f", "m2_1", "m2_2", "c"}


def test_edge_mass_flags_fire_on_edge_pile_and_stay_quiet_interior():
    # Rail criterion tested directly (no sampler): posterior mass piled
    # within EDGE_WIDTH_FRAC of a prior edge above EDGE_MASS_FRAC flags the
    # parameter; interior mass does not. (A truth-outside-prior nested run
    # exercises the same code path but costs ~1e5 likelihood-ratio dynamic
    # range in dynesty — deterministic unit test instead.)
    from scint_analysis.acf_evidence import _edge_mass_flags

    rng = np.random.default_rng(14)
    n = 4000
    g_lo, g_hi = 0.025, 1.5
    # gamma piled hard against the upper (log) edge; c interior
    gamma = g_hi * np.exp(rng.uniform(-0.005, 0.0, n) * np.log(g_hi / g_lo))
    c = rng.normal(0.0, 0.05, n)
    samples = np.column_stack([gamma, c])
    weights = np.full(n, 1.0 / n)
    bounds = [("gamma", g_lo, g_hi, True), ("c", -0.5, 0.5, False)]

    flags = _edge_mass_flags(samples, weights, bounds)
    assert flags == ["gamma"]

    # interior posterior: nothing flagged
    samples_ok = np.column_stack([
        np.exp(rng.uniform(np.log(0.2), np.log(0.6), n)), c,
    ])
    assert _edge_mass_flags(samples_ok, weights, bounds) == []
