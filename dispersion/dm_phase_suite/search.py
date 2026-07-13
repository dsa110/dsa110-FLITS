from __future__ import annotations

import numpy as np

from .coherence import coherence_curve
from .model import DMSearchResult


def _bounded_parabolic_peak(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, bool]:
    if index <= 0 or index >= x.size - 1:
        return float(x[index]), False
    xs = x[index - 1 : index + 2]
    ys = y[index - 1 : index + 2]
    coeff = np.polyfit(xs - xs[1], ys, 2)
    if not np.all(np.isfinite(coeff)) or coeff[0] >= 0:
        return float(xs[1]), False
    peak = float(xs[1] - coeff[1] / (2.0 * coeff[0]))
    if not xs[0] <= peak <= xs[-1]:
        return float(xs[1]), False
    return peak, True


def search_dm(
    *,
    waterfall: np.ndarray,
    frequencies_mhz: np.ndarray,
    sample_time_s: float,
    reference_dm: float,
    coarse_grid: np.ndarray,
    fine_step: float,
    f_cut_hz: tuple[float, float],
) -> DMSearchResult:
    coarse_grid = np.asarray(coarse_grid, dtype=float)
    if coarse_grid.size < 5 or np.any(np.diff(coarse_grid) <= 0):
        raise ValueError("coarse_grid must have at least five strictly increasing points")
    if not np.isfinite(fine_step) or fine_step <= 0:
        raise ValueError("fine_step must be positive")
    coarse = coherence_curve(
        waterfall,
        frequencies_mhz,
        sample_time_s,
        coarse_grid,
        f_cut_hz=f_cut_hz,
    )
    coarse_index = int(np.argmax(coarse.score))
    coarse_spacing = float(np.min(np.diff(coarse_grid)))
    centre = float(coarse_grid[coarse_index])
    fine_grid = np.arange(
        centre - coarse_spacing,
        centre + coarse_spacing + 0.5 * fine_step,
        fine_step,
    )
    fine = coherence_curve(
        waterfall,
        frequencies_mhz,
        sample_time_s,
        fine_grid,
        f_cut_hz=f_cut_hz,
    )
    fine_index = int(np.argmax(fine.score))
    peak, interpolated = _bounded_parabolic_peak(fine_grid, fine.score, fine_index)
    edge = coarse_index in (0, coarse_grid.size - 1) or fine_index in (0, fine_grid.size - 1)
    return DMSearchResult(
        reference_dm=float(reference_dm),
        residual_dm=peak,
        absolute_dm=float(reference_dm + peak),
        coarse=coarse,
        fine=fine,
        grid_peak_dm=float(fine_grid[fine_index]),
        edge_peak=bool(edge),
        interpolation_used=interpolated,
    )
