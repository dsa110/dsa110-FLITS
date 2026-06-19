"""Physical-parameter derivations for scintillation analysis.

Scaling relations migrated from the legacy ``scint_pipeline_funcs`` module:
timescale from bandwidth, effective transverse velocity, scattering-screen
distance from arc curvature, and a weighted mean/standard deviation helper.

These four functions have no equivalent elsewhere in ``scint_analysis``; the
related ``interpret_modulation_index``, ``estimate_emission_region_size`` and
``two_screen_coherence_constraint`` live in :mod:`scintillation.scint_analysis.analysis`.

References
----------
- Cordes & Rickett 1998, ApJ 505, 315 (effective velocity).
- Stinebring et al. 2001, ApJ 549, L97 (arc curvature η = λ^2 D_eff / 2c).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import scipy.constants as cons
from numpy.typing import NDArray

__all__ = [
    "scintillation_bandwidth_to_timescale",
    "effective_velocity",
    "screen_distance_from_curvature",
    "weighted_avg_and_std",
]

C_MPS: float = cons.c  # speed of light, m/s
PARSEC_M: float = cons.parsec  # parsec, m


def scintillation_bandwidth_to_timescale(
    delta_nu_d_hz: float,
    freq_mhz: float,
    alpha: float = 4.0,
    coefficient: float = 1.0,
) -> float:
    r"""Estimate the scintillation timescale from the bandwidth.

    Uses the uncertainty relation ``2\pi * \Delta\nu_d * \tau_d \approx C`` (default C = 1.0).

    Parameters
    ----------
    delta_nu_d_hz : float
        Characteristic scintillation bandwidth in Hz.
    freq_mhz : float
        Observing frequency in MHz (contextual; not used in the C relation).
    alpha : float, optional
        Frequency-scaling index \Delta\nu_d \propto \nu^\alpha (contextual). Default 4.0.
    coefficient : float, optional
        Uncertainty relation scaling constant C. Default 1.0. Typically C ≈ 1.16 for a
        thin Kolmogorov screen, or C ≈ 0.72 for a thick Kolmogorov screen.

    Returns
    -------
    float
        Estimated scintillation timescale \tau_d in seconds, or NaN if
        ``delta_nu_d_hz <= 0``.

    Notes
    -----
    This is a scaling *estimate*; the constant C depends on the structure
    function (e.g. C \approx 1.16 for a Kolmogorov thin screen).
    """
    if delta_nu_d_hz <= 0:
        return float("nan")
    return coefficient / (2.0 * np.pi * delta_nu_d_hz)


def effective_velocity(
    lens_dist_kpc: float,
    source_dist_kpc: float,
    tau_d_ms: float,
    delta_nu_d_khz: float,
    freq_ghz: float,
    is_earth_term_dominant: bool = False,
) -> float:
    """Estimate the effective transverse velocity from scintillation.

    Uses ``V_eff = r_F / τ_d`` with Fresnel scale
    ``r_F = sqrt(λ D_eff / 2π)`` and effective distance
    ``D_eff = D_L D_LS / D_S``.

    Parameters
    ----------
    lens_dist_kpc : float
        Distance to the scattering screen (lens) in kpc.
    source_dist_kpc : float
        Distance to the source in kpc.
    tau_d_ms : float
        Scintillation timescale in milliseconds.
    delta_nu_d_khz : float
        Scintillation bandwidth in kHz (retained for interface compatibility).
    freq_ghz : float
        Observing frequency in GHz.
    is_earth_term_dominant : bool, optional
        If True, take ``D_eff = D_L`` (screen near the observer). Default False.

    Returns
    -------
    float
        Effective transverse velocity in km/s, or NaN if inputs are
        non-physical (any non-positive input, or ``D_L >= D_S``).
    """
    if any(
        val <= 0
        for val in (lens_dist_kpc, source_dist_kpc, tau_d_ms, delta_nu_d_khz, freq_ghz)
    ):
        return float("nan")

    d_source = source_dist_kpc * 1000 * PARSEC_M
    d_lens = lens_dist_kpc * 1000 * PARSEC_M
    d_lens_source = d_source - d_lens
    if d_lens_source <= 0:
        print("Warning: Lens distance >= Source distance. Check inputs.")
        return float("nan")

    d_eff = d_lens if is_earth_term_dominant else (d_lens * d_lens_source) / d_source
    lambda_m = C_MPS / (freq_ghz * 1e9)
    tau_d_s = tau_d_ms / 1000.0

    r_fresnel = np.sqrt(lambda_m * d_eff / (2.0 * np.pi))
    v_eff_mps = r_fresnel / tau_d_s
    return v_eff_mps / 1000.0


def screen_distance_from_curvature(
    curvature: float,
    freq_ghz: float,
    source_dist_mpc: Optional[float] = None,
    v_eff_kms: Optional[float] = None,
    select_host_root: bool = False,
) -> float:
    r"""Estimate screen distance from the scintillation-arc curvature.

    Inverts ``\eta = \lambda^2 D_eff / (2 c V_eff^2)`` to get
    ``D_eff = 2 c V_eff^2 \eta \nu^2 / c^2 = 2 \eta V_eff^2 \nu^2 / c``.
    If a source distance is given, solves the quadratic ``D_eff = D_L (1 - D_L / D_S)``
    for the lens distance.

    Parameters
    ----------
    curvature : float
        Arc curvature \eta in s^3 (i.e. s / Hz^2).
    freq_ghz : float
        Central observing frequency in GHz.
    source_dist_mpc : float, optional
        Source distance in Mpc. If given, return the lens distance D_L;
        otherwise return the effective distance D_eff.
    v_eff_kms : float, optional
        Effective transverse velocity in km/s. If not provided, assumes V_eff = 100 km/s
        as a representative physical velocity. If set to 1.0/1000.0 (i.e. 1 m/s), recovers
        the unscaled legacy behavior.
    select_host_root : bool, optional
        If True and source_dist_mpc is given, return the larger root for D_L (screen
        close to the host galaxy / source) rather than the smaller root (screen close to
        the observer / Milky Way). Default False.

    Returns
    -------
    float
        Effective distance D_eff (pc) or lens distance D_L (pc), or NaN if the
        inputs are non-physical or no valid lens solution exists.
    """
    if curvature <= 0 or freq_ghz <= 0:
        return float("nan")

    v_eff = 100.0 if v_eff_kms is None else v_eff_kms
    v_eff_mps = v_eff * 1000.0

    freq_hz = freq_ghz * 1e9
    d_eff_m = 2.0 * curvature * (v_eff_mps ** 2) * (freq_hz ** 2) / C_MPS
    d_eff_pc = d_eff_m / PARSEC_M

    if source_dist_mpc is None:
        return d_eff_pc

    # Solve (1/D_S) D_L^2 - D_L + D_eff = 0 for a physical 0 < D_L < D_S.
    d_source_pc = source_dist_mpc * 1e6
    a = 1.0 / d_source_pc
    b = -1.0
    c_quad = d_eff_pc
    discriminant = b ** 2 - 4 * a * c_quad
    if discriminant < 0:
        print("Warning: No real solution for D_L (discriminant < 0).")
        return float("nan")

    roots = [
        (-b + np.sqrt(discriminant)) / (2 * a),
        (-b - np.sqrt(discriminant)) / (2 * a),
    ]
    physical = [d for d in roots if 0 < d < d_source_pc]
    if len(physical) == 1:
        return physical[0]
    if not physical:
        print("Warning: No physical solution for D_L found (0 < D_L < D_S).")
        return float("nan")
    print(f"Warning: Two possible solutions for D_L: {roots[0]:.2f}, {roots[1]:.2f} pc.")
    if select_host_root:
        return max(physical)
    return min(physical)


def weighted_avg_and_std(
    values: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> Tuple[float, float]:
    """Return the weighted mean and (population) weighted standard deviation.

    Parameters
    ----------
    values : ndarray
        Values to average.
    weights : ndarray
        Weights, same shape as ``values``.

    Returns
    -------
    average : float
        Weighted mean.
    std : float
        Weighted population standard deviation.

    Raises
    ------
    ValueError
        If shapes mismatch or the weights sum to zero.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("Shapes of values and weights must match.")
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero.")

    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return float(average), math.sqrt(variance)
