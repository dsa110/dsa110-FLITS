from __future__ import annotations

import numpy as np

from .model import ResolutionEvaluation

FACTORS = {
    "chime": {
        "frequency": (1, 2, 4, 8, 16),
        "time": (1, 2, 4, 8, 16, 32, 64, 128, 256),
    },
    "dsa": {
        "frequency": (1, 2, 4, 8, 16, 32, 64),
        "time": (1, 2, 4, 8, 16),
    },
}


def resolution_factors(telescope: str) -> list[tuple[int, int]]:
    if telescope not in FACTORS:
        raise ValueError(f"unsupported telescope: {telescope}")
    pairs = [
        (frequency, time)
        for frequency in FACTORS[telescope]["frequency"]
        for time in FACTORS[telescope]["time"]
    ]
    return sorted(pairs, key=lambda pair: (np.log2(pair[0]) + np.log2(pair[1]), pair[1], pair[0]))


def block_average(
    waterfall: np.ndarray,
    frequencies_mhz: np.ndarray,
    frequency_factor: int,
    time_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware power-of-two averaging; incomplete trailing blocks are omitted."""
    wf = np.asarray(waterfall, dtype=float)
    freq = np.asarray(frequencies_mhz, dtype=float)
    ff, tf = int(frequency_factor), int(time_factor)
    if ff < 1 or tf < 1 or ff & (ff - 1) or tf & (tf - 1):
        raise ValueError("resolution factors must be positive powers of two")
    nf, nt = (wf.shape[0] // ff) * ff, (wf.shape[1] // tf) * tf
    if nf == 0 or nt == 0:
        raise ValueError("resolution factors exceed waterfall dimensions")
    blocks = wf[:nf, :nt].reshape(nf // ff, ff, nt // tf, tf)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    reduced = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=float),
        where=count > 0,
    )
    reduced_freq = np.nanmean(freq[:nf].reshape(nf // ff, ff), axis=1)
    return reduced, reduced_freq


def select_resolution(
    evaluations: list[ResolutionEvaluation],
    *,
    stability_floor: float = 0.10,
) -> ResolutionEvaluation | None:
    """Choose minimum-information-loss eligible result with a stable neighbour."""
    eligible = [row for row in evaluations if row.eligible and row.residual_dm is not None]
    eligible.sort(
        key=lambda row: (
            row.information_loss,
            row.time_factor,
            row.frequency_factor,
            -row.coherence_peak_z,
            float("inf") if row.sigma is None else row.sigma,
        )
    )
    for candidate in eligible:
        for neighbour in eligible:
            if neighbour is candidate:
                continue
            sigma_candidate = 0.0 if candidate.sigma is None else candidate.sigma
            sigma_neighbour = 0.0 if neighbour.sigma is None else neighbour.sigma
            tolerance = max(
                stability_floor,
                float(np.hypot(sigma_candidate, sigma_neighbour)),
            )
            if abs(candidate.residual_dm - neighbour.residual_dm) <= tolerance:
                return candidate
    return None
