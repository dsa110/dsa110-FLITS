from __future__ import annotations

import numpy as np

from .model import BootstrapResult
from .search import search_dm


def channel_bootstrap(
    *,
    waterfall: np.ndarray,
    frequencies_mhz: np.ndarray,
    sample_time_s: float,
    reference_dm: float,
    coarse_grid: np.ndarray,
    fine_step: float,
    f_cut_hz: tuple[float, float],
    n_bootstrap: int,
    random_seed: int,
) -> BootstrapResult:
    """Repeat the complete search on channel-resampled waterfalls."""
    wf = np.asarray(waterfall, dtype=float)
    freq = np.asarray(frequencies_mhz, dtype=float)
    rng = np.random.default_rng(random_seed)
    peaks = []
    for _ in range(int(n_bootstrap)):
        index = rng.choice(wf.shape[0], size=wf.shape[0], replace=True)
        result = search_dm(
            waterfall=wf[index],
            frequencies_mhz=freq[index],
            sample_time_s=sample_time_s,
            reference_dm=reference_dm,
            coarse_grid=coarse_grid,
            fine_step=fine_step,
            f_cut_hz=f_cut_hz,
        )
        if not result.edge_peak and np.isfinite(result.residual_dm):
            peaks.append(result.residual_dm)
    values = np.asarray(peaks, dtype=float)
    success = float(values.size / max(1, int(n_bootstrap)))
    sigma = float(np.std(values, ddof=1)) if values.size >= 2 else float("nan")
    if np.isfinite(sigma):
        sigma = max(sigma, 0.5 * float(fine_step))
    return BootstrapResult(values, sigma, success)
