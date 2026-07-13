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


@dataclass(frozen=True)
class BootstrapResult:
    peaks: np.ndarray
    sigma: float
    success_fraction: float


@dataclass(frozen=True)
class ResolutionEvaluation:
    frequency_factor: int
    time_factor: int
    shape: tuple[int, int]
    residual_dm: float | None
    sigma: float | None
    profile_snr: float
    coherence_peak_z: float
    bootstrap_success_fraction: float
    edge_peak: bool
    cutoff_stable: bool
    eligible: bool
    failure_reasons: tuple[str, ...]

    @property
    def information_loss(self) -> float:
        return float(np.log2(self.time_factor) + np.log2(self.frequency_factor))
