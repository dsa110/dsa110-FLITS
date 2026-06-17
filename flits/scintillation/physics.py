"""Physical-parameter derivations for scintillation analysis.

Scaling relations migrated from the legacy `scint_pipeline_funcs` module:
timescale from bandwidth, effective transverse velocity, scattering-screen
distance from arc curvature, and a weighted mean/standard deviation helper.

References
----------
- Cordes & Rickett 1998, ApJ 505, 315 (effective velocity).
- Stinebring et al. 2001, ApJ 549, L97 (arc curvature \u03b7 = \u03bb^2 D_eff / 2c).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.constants as cons
from numpy.typing import NDArray

__all__ = [
    "scintillation_bandwidth_to_timescale",
    "effective_velocity",
    "screen_distance_from_curvature",
    "weighted_avg_and_std",
    "interpret_modulation_index",
    "estimate_emission_region_size",
    "two_screen_coherence_constraint",
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

    Uses ``V_eff = r_F / \u03c4_d`` with Fresnel scale
    ``r_F = sqrt(\u03bb D_eff / 2\u03c0)`` and effective distance
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


def interpret_modulation_index(m: float, m_err: float = 0.0) -> Dict[str, Any]:
    r"""Interpret modulation index based on Nimmo et al. (2025) framework.

    The modulation index m is defined as \sigma_I / <I>, where \sigma_I is the standard
    deviation of the intensity and <I> is the mean. For the ACF, the peak
    amplitude equals m^2.

    Physical interpretation:
    - m \approx 1: Point source (unresolved emission region)
    - m < 1: Emission region partially resolved by scattering screen
    - m \ll 1 (0.1-0.3): Weak scintillation regime

    Parameters
    ----------
    m : float
        Measured modulation index from ACF fit.
    m_err : float, optional
        Uncertainty on m.

    Returns
    -------
    dict
        Dictionary containing measured value, error, interpretation, and resolution regime.

    References
    ----------
    Nimmo et al. 2025, FRB 20221022A scintillation analysis
    Rickett 1990, ARA&A, 28, 561 (scintillation theory)
    """
    result = {
        "m": m,
        "m_err": m_err,
        "interpretation": "",
        "emission_resolved": False,
        "resolution_regime": "unknown",
    }

    if not np.isfinite(m):
        result["interpretation"] = "Invalid modulation index measurement"
        return result

    # Use error to define tolerance for "consistent with 1"
    tol = max(2.0 * m_err, 0.05) if m_err > 0 else 0.05

    if m > 1.0 + tol:
        result["interpretation"] = (
            f"Super-unity modulation (m = {m:.2f} \u00b1 {m_err:.2f}) - "
            "may indicate calibration issues, RFI contamination, or "
            "intrinsic intensity variations beyond scintillation"
        )
        result["resolution_regime"] = "anomalous"
    elif abs(m - 1.0) <= tol or m > 0.95:
        result["interpretation"] = (
            f"Point source / unresolved emission (m = {m:.2f} \u00b1 {m_err:.2f}) - "
            "emission region is smaller than the diffractive scale of the "
            "scattering screen (Nimmo et al. 2025)"
        )
        result["resolution_regime"] = "unresolved"
        result["emission_resolved"] = False
    elif 0.7 < m <= 0.95:
        result["interpretation"] = (
            f"Marginally resolved emission region (m = {m:.2f} \u00b1 {m_err:.2f}) - "
            "emission region is comparable to or slightly larger than the "
            "diffractive scale; consistent with magnetospheric emission "
            "(Nimmo et al. 2025 Fig. 3)"
        )
        result["resolution_regime"] = "marginally_resolved"
        result["emission_resolved"] = True
    elif 0.3 < m <= 0.7:
        result["interpretation"] = (
            f"Partially resolved emission (m = {m:.2f} \u00b1 {m_err:.2f}) - "
            "emission region significantly resolved by scattering screen; "
            "may constrain emission mechanism and/or screen distance"
        )
        result["resolution_regime"] = "partially_resolved"
        result["emission_resolved"] = True
    else:  # m <= 0.3
        result["interpretation"] = (
            f"Heavily suppressed modulation (m = {m:.2f} \u00b1 {m_err:.2f}) - "
            "either weak scintillation regime or very extended emission region; "
            "check if observation is in strong scintillation regime"
        )
        result["resolution_regime"] = "weak_or_resolved"
        result["emission_resolved"] = True

    return result


def estimate_emission_region_size(
    m: float,
    delta_nu_dc_mhz: float,
    d_source_screen_pc: float,
    freq_mhz: float,
    m_err: float = 0.0,
    delta_nu_err_mhz: float = 0.0,
) -> Dict[str, Any]:
    r"""Estimate lateral emission region size from modulation index.

    Uses the relationship between modulation index and source resolution
    from Nimmo et al. (2025) Eq. 22-23:

        m = 1 / sqrt(1 + 4(R_obs/chi)^2)

        chi = (1/nu) * sqrt(c * d * \Delta\nu_dc / 2\pi)  [screen resolution]

    Solving for R_obs:
        R_obs = sqrt((c * d * \Delta\nu_dc) / (8\pi \nu^2) * (1/m^2 - 1))

    Parameters
    ----------
    m : float
        Measured modulation index (0 < m \le 1 for resolved sources).
    delta_nu_dc_mhz : float
        Decorrelation bandwidth in MHz.
    d_source_screen_pc : float
        Distance from source to scattering screen in parsecs.
    freq_mhz : float
        Observing frequency in MHz.
    m_err : float, optional
        Uncertainty on modulation index.
    delta_nu_err_mhz : float, optional
        Uncertainty on decorrelation bandwidth.

    Returns
    -------
    dict
        Dictionary containing estimated emission region size, diffractive scale, and context.

    References
    ----------
    Nimmo et al. 2025, Eq. 21-23
    Kumar et al. 2024, MNRAS, 527, 457 (FRB scintillation constraints)
    """
    c_m_s = C_MPS  # speed of light in m/s
    pc_to_m = PARSEC_M  # parsec to meters

    # Convert units
    delta_nu_hz = delta_nu_dc_mhz * 1e6
    freq_hz = freq_mhz * 1e6
    d_m = d_source_screen_pc * pc_to_m

    result = {
        "R_obs_km": np.nan,
        "R_obs_err_km": np.nan,
        "chi_km": np.nan,
        "is_upper_limit": False,
        "physical_context": "",
    }

    # Calculate screen diffractive scale \chi (Nimmo Eq. 21)
    # \chi = (1/\nu) * sqrt(c * d * \Delta\nu / 2\pi)
    chi_m = (1.0 / freq_hz) * np.sqrt(c_m_s * d_m * delta_nu_hz / (2.0 * np.pi))
    chi_km = chi_m / 1e3
    result["chi_km"] = chi_km

    # Handle edge cases
    if m >= 1.0:
        # Unresolved: R_obs < \chi (upper limit)
        result["R_obs_km"] = chi_km
        result["is_upper_limit"] = True
        result["physical_context"] = (
            f"Unresolved (m \u2265 1): R_obs < \u03c7 = {chi_km:.1f} km (upper limit)"
        )
        return result

    if m <= 0.0:
        result["physical_context"] = "Invalid modulation index (m \u2264 0)"
        return result

    # Calculate R_obs from Nimmo Eq. 23:
    factor = (c_m_s * d_m * delta_nu_hz) / (8.0 * np.pi * freq_hz ** 2)
    m_factor = (1.0 / m ** 2) - 1.0

    if m_factor <= 0:
        result["R_obs_km"] = 0.0
        result["physical_context"] = "Point source (m_factor \u2264 0)"
        return result

    R_obs_m = np.sqrt(factor * m_factor)
    R_obs_km = R_obs_m / 1e3
    result["R_obs_km"] = R_obs_km

    # Error propagation (simplified, assumes dominant error from m)
    if m_err > 0:
        dm_factor = np.sqrt(factor) * (m ** (-3)) / np.sqrt(m_factor)
        R_obs_err_km = abs(dm_factor * m_err) / 1e3
        result["R_obs_err_km"] = R_obs_err_km

    # Physical context
    context_parts = [f"Estimated R_obs = {R_obs_km:.1f} km"]

    # Compare to known scales
    if R_obs_km < 100:
        context_parts.append("consistent with pulsar emission (~10-100 km)")
    elif R_obs_km < 1000:
        context_parts.append("consistent with pulsar/magnetar magnetosphere (~100-1000 km)")
    elif R_obs_km < 1e4:
        context_parts.append("consistent with neutron star light cylinder (~1000-10,000 km)")
    elif R_obs_km < 1e5:
        context_parts.append("larger than typical magnetosphere; may indicate shock emission")
    else:
        context_parts.append("very large; likely non-magnetospheric origin")

    result["physical_context"] = "; ".join(context_parts)
    return result


def two_screen_coherence_constraint(
    delta_nu_1_mhz: float,
    delta_nu_2_mhz: float,
    freq_mhz: float,
    d_source_mpc: float,
    C1: float = 1.0,
    C2: float = 1.0,
) -> Dict[str, Any]:
    r"""Calculate two-screen coherence constraint from Nimmo et al. (2025).

    When two scintillation scales are observed, mutual coherence requires:

        \Delta\nu_s1 * \Delta\nu_s2 >= C1 * C2 * \nu^2 * (d_s1★ * d_s2★ * d_⊕s1) / (d_⊕★^2 * d_⊕s2)

    For an extragalactic source with one Galactic screen (s1) and one
    host-galaxy screen (s2), this simplifies to (Nimmo Eq. 10):

        d_⊕s1 * d_s2★ <= \Delta\nu_s1 * \Delta\nu_s2 * d_⊕★^2 / (C1 * C2 * \nu^2)

    Parameters
    ----------
    delta_nu_1_mhz : float
        Decorrelation bandwidth of screen 1 (closest to observer) in MHz.
    delta_nu_2_mhz : float
        Decorrelation bandwidth of screen 2 (closest to source) in MHz.
    freq_mhz : float
        Observing frequency in MHz.
    d_source_mpc : float
        Distance to source in Mpc.
    C1, C2 : float, optional
        Geometry constants, default 1.0.

    Returns
    -------
    dict
        Dictionary containing the upper limit on the screen distance product in kpc^2.

    References
    ----------
    Nimmo et al. 2025, Eq. 7-11
    """
    # Convert to Hz
    delta_nu_1_hz = delta_nu_1_mhz * 1e6
    delta_nu_2_hz = delta_nu_2_mhz * 1e6
    freq_hz = freq_mhz * 1e6

    # Convert distance to meters then to kpc for result
    d_source_kpc = d_source_mpc * 1e3  # Mpc to kpc

    # Constraint: d_⊕s1 * d_s2★ <= \Delta\nu1 * \Delta\nu2 * d_⊕★^2 / (C1 * C2 * \nu^2)
    numerator = delta_nu_1_hz * delta_nu_2_hz * (d_source_kpc ** 2)
    denominator = C1 * C2 * (freq_hz ** 2)

    d_product_kpc2 = numerator / denominator

    result = {
        "d_product_kpc2": d_product_kpc2,
        "example_constraints": {},
    }

    # Example scenarios
    galactic_distances = [0.1, 0.3, 0.64, 1.0, 3.0]  # kpc
    for d_gal in galactic_distances:
        d_host_max = d_product_kpc2 / d_gal
        result["example_constraints"][f"d_gal_{d_gal}kpc"] = {
            "d_galactic_kpc": d_gal,
            "max_d_host_kpc": d_host_max,
        }

    return result
