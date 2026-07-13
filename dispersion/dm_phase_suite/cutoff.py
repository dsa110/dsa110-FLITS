from __future__ import annotations

import numpy as np


def estimate_fwhm_s(waterfall: np.ndarray, sample_time_s: float) -> float:
    """Estimate the contiguous half-maximum width of the band-summed pulse."""
    wf = np.asarray(waterfall, dtype=float)
    profile = np.nansum(wf - np.nanmedian(wf, axis=1)[:, None], axis=0)
    smooth_bins = max(1, int(round(1e-4 / sample_time_s)))
    if smooth_bins > 1:
        profile = np.convolve(profile, np.ones(smooth_bins) / smooth_bins, mode="same")
    baseline = float(np.nanmedian(profile))
    profile = profile - baseline
    peak = int(np.nanargmax(profile))
    half = 0.5 * float(profile[peak])
    left = peak
    while left > 0 and profile[left - 1] >= half:
        left -= 1
    right = peak
    while right < profile.size - 1 and profile[right + 1] >= half:
        right += 1
    return max(float(sample_time_s), (right - left + 1) * float(sample_time_s))


def width_derived_cutoffs(
    waterfall: np.ndarray,
    sample_time_s: float,
    *,
    minimum_hz: float = 50.0,
    maximum_hz: float = 1500.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return lower/base/upper adjacent policies from the observed pulse width.

    For a Gaussian, useful phase structure is concentrated below roughly
    1/(2 pi sigma) = 0.375/FWHM. Adjacent policies test cutoff sensitivity.
    """
    fwhm = estimate_fwhm_s(waterfall, sample_time_s)
    base = float(np.clip(0.375 / fwhm, 2.0 * minimum_hz, maximum_hz))
    highs = [float(np.clip(scale * base, 2.0 * minimum_hz, maximum_hz)) for scale in (0.75, 1.0, 1.25)]
    return tuple((float(minimum_hz), high) for high in highs)  # type: ignore[return-value]
