"""Boundary behavior of the thin-screen beta-family kernel (ADR-0006).

Phase-1 gate of the beta-coherent thin-screen campaign
(docs/rse/specs/plan-beta-coherent-thin-screen-campaign.md): the power-law
kernel must hand off to the closed-form exponential at the BETA_EXP_EPS
switch, stay area-normalized across the sampled beta range, and grow tail
mass monotonically as beta drops. The switch also carries a small, bounded
derived-alpha discontinuity (the price of the discrete shape handoff) that
is pinned here so a future BETA_EXP_EPS change re-prices it consciously.
"""

import numpy as np

from scattering.scat_analysis.burstfit import (
    analytic_gaussian_exp_convolution,
    gaussian_powerlaw_convolution,
)
from scattering.scat_analysis.turbulence import (
    BETA_EXP_EPS,
    BETA_THIN_SCREEN_MAX,
    alpha_from_beta,
)

# tau=1 ms on a 60 ms grid: the exponential regime is resolved to ~55 tau.
# sig = 20*DT so the FFT path's sub-sample registration offset (t0-degenerate,
# removed here only to integer precision) contributes < 0.5% on the rise.
T_MAX, DT = 60.0, 0.01
MU, SIG, TAU = 5.0, 0.2, 1.0


def _grid():
    t = np.arange(0.0, T_MAX, DT)
    one = np.array([[1.0]])
    return t, MU * one, SIG * one, TAU * one


def _aligned_max_diff(a, b):
    # The FFT path leaves a SUB-sample time-registration offset that is fully
    # degenerate with the free t0 (gaussian_powerlaw_convolution docstring):
    # integer xcorr alignment first, then profile the fractional remainder the
    # way a fit's free t0 would, by minimizing over interpolated shifts.
    n = len(b)
    x = np.arange(n, dtype=float)
    shift = int(np.argmax(np.correlate(a, b, mode="full"))) - (n - 1)
    a = np.roll(a, -shift)
    peak = float(np.max(b))
    return min(
        float(np.max(np.abs(np.interp(x + frac, x, a) - b)) / peak)
        for frac in np.linspace(-1.0, 1.0, 81)
    )


def test_powerlaw_kernel_matches_exponential_at_the_switch():
    t, mu, sig, tau = _grid()
    beta_switch = BETA_THIN_SCREEN_MAX - BETA_EXP_EPS
    pl = gaussian_powerlaw_convolution(t, mu, sig, tau, beta_switch - 1e-6)[0]
    ex = analytic_gaussian_exp_convolution(t, mu, sig, tau)[0]
    # s_c = 2 ln(2/eps) ~ 9.2 at the switch: residual tail mass ~ e^-9.2, so
    # the two kernels must agree to well under a percent of peak.
    assert _aligned_max_diff(pl, ex) < 0.01


def test_convolution_stays_area_normalized_across_beta():
    t, mu, sig, tau = _grid()
    for beta, tol in [(2.1, 0.10), (2.5, 0.05), (3.0, 0.01), (11.0 / 3.0, 0.01), (3.9, 0.01)]:
        total = float(gaussian_powerlaw_convolution(t, mu, sig, tau, beta)[0].sum() * DT)
        # Loss is grid truncation of the power-law tail only, so it grows as
        # beta drops; the campaign prior lives at beta >= 3 where it is ~1%.
        assert 1.0 - tol < total <= 1.001, f"beta={beta}: integral {total}"


def test_tail_mass_grows_monotonically_as_beta_drops():
    t, mu, sig, tau = _grid()
    cut = t > (MU + 5.0 * TAU)
    fracs = []
    for beta in (3.95, 11.0 / 3.0, 3.0, 2.5):
        prof = gaussian_powerlaw_convolution(t, mu, sig, tau, beta)[0]
        fracs.append(float(prof[cut].sum() / prof.sum()))
    assert all(a < b for a, b in zip(fracs, fracs[1:])), fracs


def test_alpha_discontinuity_at_the_shape_switch_is_pinned():
    # Inside the eps window the shape snaps to the exponential member, so the
    # derived scaling snaps to alpha = 4 with it (ADR-0006: shape and scaling
    # must assert the same beta). Just below the window the closure gives
    # 2b/(b-2); the jump is the documented price of the discrete handoff and
    # must stay small vs the campaign's alpha error bars (~0.02-0.2).
    b = BETA_THIN_SCREEN_MAX - BETA_EXP_EPS
    jump = 2.0 * b / (b - 2.0) - 4.0
    assert 0.0 < jump < 0.025
    assert alpha_from_beta(b) == 4.0
    assert abs(alpha_from_beta(b - 1e-9) - (4.0 + jump)) < 1e-6
