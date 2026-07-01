"""Turbulence-spectrum helpers for the beta-coherent scattering model.

The fundamental parameter is the electron-density fluctuation spectral index beta
(P_n(q) propto q^{-beta}). For a thin screen with 2 < beta <= 4, both the
pulse-broadening function shape and the frequency-scaling index follow from beta:

    alpha = 2*beta / (beta - 2)          (tau propto nu^{-alpha})
    PBF tail ~ t^{-beta/2}  (via gaussian_powerlaw_convolution)

At beta -> 4 the power-law PBF reduces to the thin-screen exponential (alpha = 4).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KOLMOGOROV_BETA",
    "BETA_THIN_SCREEN_MIN",
    "BETA_THIN_SCREEN_MAX",
    "BETA_EXP_EPS",
    "alpha_from_beta",
    "beta_from_alpha_thin_screen",
    "beta_bounds_from_alpha_bounds",
    "default_joint_beta_bounds",
]

KOLMOGOROV_BETA = 11.0 / 3.0
BETA_THIN_SCREEN_MIN = 2.01
BETA_THIN_SCREEN_MAX = 4.0
# Use the closed-form exponential PBF when beta is within this of 4.
BETA_EXP_EPS = 0.02


def alpha_from_beta(beta: float) -> float:
    """Frequency-scaling index alpha from turbulence spectral index beta (thin screen, beta <= 4)."""
    beta = float(beta)
    if beta >= BETA_THIN_SCREEN_MAX - BETA_EXP_EPS:
        return 4.0
    if beta <= 2.0:
        raise ValueError(f"beta must be > 2 for thin-screen alpha mapping, got {beta}")
    return 2.0 * beta / (beta - 2.0)


def beta_from_alpha_thin_screen(alpha: float) -> float:
    """Inverse of alpha_from_beta for alpha >= 4 (thin-screen branch)."""
    alpha = float(alpha)
    if alpha <= 2.0:
        raise ValueError(f"alpha must be > 2 for thin-screen beta mapping, got {alpha}")
    if np.isclose(alpha, 4.0):
        return BETA_THIN_SCREEN_MAX
    return 2.0 * alpha / (alpha - 2.0)


def beta_bounds_from_alpha_bounds(
    alpha_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Map legacy alpha prior bounds to beta bounds on the thin-screen branch.

    Only alpha >= 4 is reachable with beta in (2, 4]; sub-Kolmogorov alpha < 4
    requires beta > 4 (extended-medium physics) and is not mapped here.
    """
    lo_a, hi_a = float(alpha_bounds[0]), float(alpha_bounds[1])
    # Higher alpha -> lower beta; clamp to the integrable PBF regime.
    beta_lo = beta_from_alpha_thin_screen(max(hi_a, 4.0 + 1e-6))
    beta_hi = beta_from_alpha_thin_screen(max(lo_a, 4.0))
    beta_lo = max(beta_lo, BETA_THIN_SCREEN_MIN)
    beta_hi = min(beta_hi, BETA_THIN_SCREEN_MAX)
    if beta_lo >= beta_hi:
        beta_lo = BETA_THIN_SCREEN_MIN
        beta_hi = BETA_THIN_SCREEN_MAX
    return (beta_lo, beta_hi)


def default_joint_beta_bounds() -> tuple[float, float]:
    """Default joint-fit beta bounds (~ alpha in [4, 6] on the thin-screen branch)."""
    return beta_bounds_from_alpha_bounds((4.0, 6.0))
