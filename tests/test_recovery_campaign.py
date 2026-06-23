"""Quantitative tau-recovery validation (B.5).

The ensemble-averaged sim->fit campaign recovers tau *linearly*: recovered tau is
proportional to injected tau with a constant ratio across ~80x in tau. Asserting
linearity (constant ratio), not ratio==1, is deliberate -- a narrow single band
carries a constant nu0^-alpha normalisation offset (see recovery_campaign and the
sim_fit_bridge.roundtrip caveat); the science that must hold is relative recovery.

Slow: many simulator realisations + an MCMC fit per grid point.
"""

import os
import sys

import numpy as np
import pytest

_SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)


@pytest.mark.slow
def test_recovery_is_linear():
    pytest.importorskip("emcee")
    import logging

    logging.disable(logging.WARNING)
    from recovery_campaign import recovery_curve

    df = recovery_curve(scale_L=(1.0, 3.0, 9.0), n_real=12, n_steps=600)

    assert df["tau_fit_ms"].gt(0).all() and np.isfinite(df["ratio"]).all()
    # injected tau must span a wide range for "linear" to mean something
    assert df["tau_true_ms"].max() / df["tau_true_ms"].min() > 20
    # constant ratio across the grid == recovered tau tracks injected tau
    r = df["ratio"].to_numpy()
    assert r.std() / r.mean() < 0.15, f"recovery not linear: ratios={r}"


@pytest.mark.slow
def test_dnu_recovery_is_linear():
    import logging

    logging.disable(logging.WARNING)
    from recovery_campaign import dnu_recovery_curve

    df = dnu_recovery_curve(host_L=(1.0, 1.5, 2.2), n_real=12)

    assert df["dnu_fit_MHz"].gt(0).all() and np.isfinite(df["ratio"]).all()
    # injected Dnu must span a useful range for "linear" to mean something
    assert df["dnu_true_MHz"].max() / df["dnu_true_MHz"].min() > 3
    r = df["ratio"].to_numpy()
    assert r.std() / r.mean() < 0.15, f"dnu recovery not linear: ratios={r}"
