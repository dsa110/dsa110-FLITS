"""DM-phase (structure-max) estimator must recover a known injected DM — at PHYSICAL scale.

Gate for the CHIME-side pillar-2 extraction: before trusting `DMPhaseEstimator` on real
bursts, prove it recovers the DM of a synthetic dispersed multi-component waterfall AND that
its coherence curve actually *peaks* sharply (a near-flat curve is not a detection).

Physical-scale guard: the injection uses the real dispersion law
``delay[s] = K_DM * DM * (1/f^2 - 1/f_ref^2)`` with K_DM in seconds (MHz^2 pc^-1 cm^3 s) and
f in MHz — the SAME constant the estimator uses (dmphasev2.py:57). A regression that rescales
the estimator's delay (e.g. the old spurious 1e-3 factor, 1000x too small) breaks de-dispersion:
the curve goes flat and recovery fails, so this test catches it. Pure host test (no docker).

Sign convention: the estimator de-disperses with ``exp(-2j pi f * DM * delay_sec)``
(dmphasev2.py:77), so a positive trial-DM grid recovers ``dm`` for an injection whose
per-channel delay carries the estimator's sign, ``-K_DM*DM*(1/f^2 - 1/f_ref^2)``.
"""

import numpy as np
import pytest

from dispersion.dm_phase_analysis import measure_dm_phase
from dispersion.dmphasev2 import DMPhaseEstimator, dmphase_trial_to_physical_residual_dm
from flits.common.constants import K_DM


def _disperse(n_t, freqs, dt, dm, comps):
    """Synthesize an intensity waterfall (n_t, n_ch) with sharp PHYSICALLY-dispersed sub-pulses."""
    ref = freqs.max()
    delay = -K_DM * (1.0 / freqs**2 - 1.0 / ref**2) * dm  # seconds, physical scale
    t = np.arange(n_t) * dt
    wf = np.zeros((n_t, freqs.size))
    for t0, amp, width in comps:
        wf += amp * np.exp(-0.5 * ((t[:, None] - t0 - delay[None, :]) / width) ** 2)
    return wf


def test_dmphase_recovers_known_dm():
    # cost ~ n_grid*n_t*n_ch*n_boot; kept modest. dm_true=20 -> ~0.39 s sweep across 400-800 MHz,
    # comfortably inside the n_t*dt = 1.024 s window.
    rng = np.random.default_rng(0)
    freqs = np.linspace(400.0, 800.0, 96)
    dt, n_t, dm_true = 1.0e-3, 1024, 20.0
    comps = [(0.45, 1.0, 2.0e-3), (0.462, 0.7, 2.0e-3)]  # two sharp sub-pulses, ~12 ms apart
    wf = _disperse(n_t, freqs, dt, dm_true, comps)
    wf += 0.05 * rng.standard_normal((n_t, freqs.size))

    grid = np.arange(dm_true - 5.0, dm_true + 5.0, 0.2)
    est = DMPhaseEstimator(wf, freqs, dt, grid, ref="top", n_boot=40, random_state=1)
    curve = est.result()["dm_curve"]

    # peak must be interior to the grid (a railed edge result is not a recovery)
    i_pk = int(np.argmax(curve))
    assert 0 < i_pk < len(grid) - 1, f"DM-structure peak railed to grid edge (idx {i_pk})"

    # the curve must actually PEAK: a working physical-scale estimator gives a sharp coherence
    # peak; the 1000x-too-small delay regression gives a near-flat curve (ratio ~ 1).
    flat_ratio = float(curve.max() / curve.min())
    assert flat_ratio > 2.0, f"DM-phase curve is near-flat (ratio {flat_ratio:.2f}) — no real peak"

    # robust recovery: argmax of the mean curve within ~2 grid steps of truth
    dm_best = float(grid[i_pk])
    assert abs(dm_best - dm_true) < 2.0 * (grid[1] - grid[0]), f"dm_best={dm_best} != {dm_true}"


def test_dmphase_trial_sign_converts_to_physical_residual_dm():
    rng = np.random.default_rng(2)
    freqs = np.linspace(400.0, 800.0, 96)
    dt, n_t, residual_true = 1.0e-3, 1400, 8.0
    ref = freqs.max()
    delay = K_DM * residual_true * (1.0 / freqs**2 - 1.0 / ref**2)
    t = np.arange(n_t) * dt
    wf = np.exp(-0.5 * ((t[:, None] - 0.5 - delay[None, :]) / 2.0e-3) ** 2)
    wf += 0.02 * rng.standard_normal(wf.shape)

    grid = np.arange(-12.0, 12.1, 0.25)
    est = DMPhaseEstimator(wf, freqs, dt, grid, ref="top", n_boot=20, random_state=3)
    physical_residual = dmphase_trial_to_physical_residual_dm(est.dm_best)

    assert est.dm_best == pytest.approx(-residual_true, abs=0.5)
    assert physical_residual == pytest.approx(residual_true, abs=0.5)


def test_dmphase_analysis_reports_absolute_candidate_with_physical_sign():
    rng = np.random.default_rng(4)
    freqs = np.linspace(400.0, 800.0, 64)
    dt, n_t, residual_true, dm_ref = 1.0e-3, 1024, 4.0, 500.0
    ref = freqs.max()
    delay = K_DM * residual_true * (1.0 / freqs**2 - 1.0 / ref**2)
    t = np.arange(n_t) * dt
    wf = np.exp(-0.5 * ((t[:, None] - 0.45 - delay[None, :]) / 2.0e-3) ** 2).T
    wf += 0.02 * rng.standard_normal(wf.shape)

    result = measure_dm_phase(
        wf,
        freqs,
        dt,
        dm_ref,
        np.arange(-6.0, 6.1, 0.5),
        f_cut_hz=(20.0, 800.0),
        n_boot=20,
        random_state=5,
    )

    assert result["physical_residual_dm"] == pytest.approx(residual_true, abs=0.75)
    assert result["candidate_dm"] == pytest.approx(dm_ref + residual_true, abs=0.75)
    assert result["candidate_sigma"] >= 0.25
