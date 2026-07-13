from __future__ import annotations

import numpy as np

from .model import CoherenceCurve
from .shifts import residual_delay_s


def _normalise_channels(waterfall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wf = np.asarray(waterfall, dtype=float)
    if wf.ndim != 2 or min(wf.shape) < 2:
        raise ValueError("waterfall must be a non-empty (frequency,time) array")
    finite = np.isfinite(wf)
    med = np.nanmedian(wf, axis=1)
    mad = np.nanmedian(np.abs(wf - med[:, None]), axis=1)
    sigma = 1.4826 * mad
    valid = (finite.mean(axis=1) >= 0.9) & np.isfinite(sigma) & (sigma > 0)
    out = np.zeros_like(wf, dtype=float)
    out[valid] = np.nan_to_num(
        (wf[valid] - med[valid, None]) / sigma[valid, None],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return out, valid


def coherence_curve(
    waterfall: np.ndarray,
    frequencies_mhz: np.ndarray,
    sample_time_s: float,
    residual_dm_grid: np.ndarray,
    *,
    f_cut_hz: tuple[float, float],
    channel_weights: np.ndarray | None = None,
) -> CoherenceCurve:
    wf, valid = _normalise_channels(waterfall)
    freq = np.asarray(frequencies_mhz, dtype=float)
    grid = np.asarray(residual_dm_grid, dtype=float)
    if freq.shape != (wf.shape[0],):
        raise ValueError("one frequency is required per waterfall channel")
    if grid.ndim != 1 or grid.size < 3 or np.any(~np.isfinite(grid)):
        raise ValueError("residual_dm_grid must contain at least three finite values")
    order = np.argsort(freq)
    wf, freq, valid = wf[order], freq[order], valid[order]
    if valid.sum() < 2:
        raise ValueError("fewer than two valid channels")
    if channel_weights is None:
        weights = valid.astype(float)
    else:
        weights = np.asarray(channel_weights, dtype=float)[order]
        weights = np.where(valid & np.isfinite(weights) & (weights >= 0), weights, 0.0)
    weights /= weights.sum()

    maximum_delay_s = float(
        np.max(np.abs(residual_delay_s(freq, float(np.max(np.abs(grid))))))
    )
    pad = int(np.ceil(maximum_delay_s / float(sample_time_s))) + 8
    padded = np.pad(wf, ((0, 0), (pad, pad)), mode="constant")
    fluctuation = np.fft.rfftfreq(padded.shape[1], float(sample_time_s))
    low, high = map(float, f_cut_hz)
    if not (0 <= low < high):
        raise ValueError("f_cut_hz must be an increasing non-negative pair")
    use = (fluctuation >= low) & (fluctuation <= high) & (fluctuation > 0)
    if use.sum() < 2:
        raise ValueError("cutoff leaves fewer than two positive fluctuation bins")

    spectrum = np.fft.rfft(padded, axis=1)[:, use]
    amplitude = np.abs(spectrum)
    phase_only = np.divide(
        spectrum,
        amplitude,
        out=np.zeros_like(spectrum),
        where=amplitude > np.finfo(float).tiny,
    )
    f_used = fluctuation[use]
    delay_per_dm = residual_delay_s(freq, 1.0)
    power = np.empty((grid.size, int(use.sum())), dtype=float)
    for index, residual_dm in enumerate(grid):
        correction = np.exp(
            2j
            * np.pi
            * delay_per_dm[:, None]
            * float(residual_dm)
            * f_used[None, :]
        )
        coherent = np.sum(phase_only * correction * weights[:, None], axis=0)
        power[index] = np.abs(coherent) ** 2
    score = np.sum(power * f_used[None, :] ** 2, axis=1)
    return CoherenceCurve(grid, score, f_used, power, int(valid.sum()))
