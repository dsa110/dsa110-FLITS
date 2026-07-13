from __future__ import annotations

import warnings

import numpy as np

INJECTION_SIGMA68 = {
    "chime": ((12.0, 0.008121986545538374), (25.0, 0.003694335302605563), (50.0, 0.0018114759909541658)),
    "dsa": ((50.0, 0.11162548245180161),),
}


def calibrated_injection_sigma(telescope: str, profile_snr: float) -> float:
    """Frozen held-out 68% error scale at the nearest supported S/N floor."""
    surface = INJECTION_SIGMA68[telescope]
    eligible = [sigma for floor, sigma in surface if profile_snr >= floor]
    if not eligible:
        return float("nan")
    return float(eligible[-1])


def robust_profile_snr(waterfall: np.ndarray) -> float:
    """Peak band-summed S/N using an off-peak MAD noise estimate."""
    wf = np.asarray(waterfall, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nansum(wf - np.nanmedian(wf, axis=1)[:, None], axis=0)
    if profile.size >= 9:
        profile = np.convolve(profile, np.ones(9) / 9.0, mode="same")
    peak = int(np.nanargmax(profile))
    guard = max(8, profile.size // 20)
    off = np.ones(profile.size, dtype=bool)
    off[max(0, peak - guard) : min(profile.size, peak + guard + 1)] = False
    median = float(np.nanmedian(profile[off]))
    sigma = 1.4826 * float(np.nanmedian(np.abs(profile[off] - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float((profile[peak] - median) / sigma)


def peak_z_score(score: np.ndarray) -> float:
    values = np.asarray(score, dtype=float)
    peak = int(np.nanargmax(values))
    exclusion = max(1, values.size // 10)
    use = np.ones(values.size, dtype=bool)
    use[max(0, peak - exclusion) : min(values.size, peak + exclusion + 1)] = False
    baseline = values[use]
    median = float(np.nanmedian(baseline))
    sigma = 1.4826 * float(np.nanmedian(np.abs(baseline - median)))
    return float((values[peak] - median) / sigma) if sigma > 0 else float("inf")


def bootstrap_central_fraction(
    peaks: np.ndarray,
    point_peak: float,
    sigma: float,
    *,
    minimum_radius: float = 0.04,
) -> float:
    """Fraction of bootstrap peaks in the point estimate's central mode."""
    values = np.asarray(peaks, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(sigma):
        return 0.0
    radius = max(float(minimum_radius), 2.0 * float(sigma))
    return float(np.mean(np.abs(values - float(point_peak)) <= radius))


def eligibility_reasons(
    *,
    profile_snr: float,
    coherence_peak_z: float,
    edge_peak: bool,
    bootstrap_success_fraction: float,
    cutoff_stable: bool,
    minimum_profile_snr: float = 8.0,
    minimum_peak_z: float = 4.0,
    minimum_bootstrap_success_fraction: float = 0.90,
) -> tuple[str, ...]:
    reasons = []
    if not np.isfinite(profile_snr) or profile_snr < minimum_profile_snr:
        reasons.append("profile_snr")
    if not np.isfinite(coherence_peak_z) or coherence_peak_z < minimum_peak_z:
        reasons.append("coherence_significance")
    if edge_peak:
        reasons.append("edge_peak")
    if bootstrap_success_fraction < minimum_bootstrap_success_fraction:
        reasons.append("bootstrap_success")
    if not cutoff_stable:
        reasons.append("cutoff_stability")
    return tuple(reasons)
