"""DM-phase (structure-max) estimator must recover a known injected DM.

Gate for the CHIME-side pillar-2 extraction: before trusting `DMPhaseEstimator`
on real bursts, prove it recovers the DM of a synthetic dispersed multi-component
waterfall to within its bootstrap uncertainty, with the peak *interior* to the grid
(a railed edge result must FAIL). Pure host test (no docker).

Sign convention: `DMPhaseEstimator` de-disperses with phase
``exp(-2j pi f * DM * delay_sec)`` (dmphasev2.py:77), so its coherent power peaks at
positive DM for an injection whose per-channel delay carries the SAME sign as the
estimator's ``delay_sec`` correction (i.e. ``-K_DM*DM*(1/f^2 - 1/f_ref^2)`` here).
This fixes the test's injection orientation; the matching orientation for *real*
(physically dispersed) CHIME waterfalls is asserted empirically in the extraction
(it requires an interior peak and flips the time axis otherwise).
"""

import numpy as np

from dispersion.dmphasev2 import DMPhaseEstimator
from flits.common.constants import K_DM


def _disperse(n_t, freqs, dt, dm, comps):
    """Synthesize an intensity waterfall (n_t, n_ch) with sharp dispersed sub-pulses.

    Delay uses the estimator's de-dispersion sign so a positive trial-DM grid recovers ``dm``.
    """
    ref = freqs.max()
    delay = -1e-3 * K_DM * (1.0 / freqs**2 - 1.0 / ref**2) * dm  # seconds, per channel
    t = np.arange(n_t) * dt
    wf = np.zeros((n_t, freqs.size))
    for t0, amp, width in comps:
        wf += amp * np.exp(-0.5 * ((t[:, None] - t0 - delay[None, :]) / width) ** 2)
    return wf


def test_dmphase_recovers_known_dm():
    # kept small on purpose: DM-phase allocates ~(n_grid, n_t, n_ch) per bootstrap,
    # so cost ~ n_grid*n_t*n_ch*n_boot — fine here, OOM-thrashes at notebook sizes.
    rng = np.random.default_rng(0)
    freqs = np.linspace(400.0, 800.0, 96)
    dt, n_t, dm_true = 5.0e-4, 768, 500.0
    comps = [(0.18, 1.0, 1.0e-3), (0.186, 0.7, 1.0e-3)]  # two sharp sub-pulses, ~6 ms apart
    wf = _disperse(n_t, freqs, dt, dm_true, comps)
    wf += 0.05 * rng.standard_normal((n_t, freqs.size))

    grid = np.arange(dm_true - 6.0, dm_true + 6.0, 0.2)
    est = DMPhaseEstimator(wf, freqs, dt, grid, ref="top", n_boot=40, random_state=1)
    dm_best, dm_sigma = est.get_dm()

    # peak must be interior to the grid (a railed edge result is not a recovery)
    i_pk = int(np.argmax(est.result()["dm_curve"]))
    assert 0 < i_pk < len(grid) - 1, f"DM-structure peak railed to grid edge (idx {i_pk})"
    assert dm_sigma > 0
    assert abs(dm_best - dm_true) < max(3.0 * dm_sigma, 2.0 * (grid[1] - grid[0]))
