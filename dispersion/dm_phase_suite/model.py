from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoherenceCurve:
    residual_dm_grid: np.ndarray
    score: np.ndarray
    fluctuation_frequency_hz: np.ndarray
    coherent_power: np.ndarray
    valid_channel_count: int


@dataclass(frozen=True)
class DMSearchResult:
    reference_dm: float
    residual_dm: float
    absolute_dm: float
    coarse: CoherenceCurve
    fine: CoherenceCurve
    grid_peak_dm: float
    edge_peak: bool
    interpolation_used: bool
