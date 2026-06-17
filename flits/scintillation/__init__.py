"""Scintillation analysis subpackage for dsa110-FLITS.

Migrated from the legacy `scint_pipeline` / `scint_pipeline_funcs` scripts into
cohesive modules:

- :mod:`~flits.scintillation.preprocessing` -- scrunching and upchannelization.
- :mod:`~flits.scintillation.acf` -- autocorrelation-function estimation.
- :mod:`~flits.scintillation.fitting` -- Lorentzian / power-law models and fitters.
- :mod:`~flits.scintillation.secondary` -- secondary (conjugate) spectrum.
- :mod:`~flits.scintillation.physics` -- physical-parameter derivations.
- :mod:`~flits.scintillation.analyser` -- the :class:`ScintillationAnalyser` class.
"""

from __future__ import annotations

from .acf import calculate_acf_1d, calculate_acf_2d
from .analyser import ScintillationAnalyser
from .fitting import (
    double_lorentzian_with_const,
    fit_lorentzian_acf,
    fit_scint_bandwidth_freq_relation,
    lorentzian,
    lorentzian_with_const,
    power_law,
)
from .physics import (
    effective_velocity,
    estimate_emission_region_size,
    interpret_modulation_index,
    scintillation_bandwidth_to_timescale,
    screen_distance_from_curvature,
    two_screen_coherence_constraint,
    weighted_avg_and_std,
)
from .preprocessing import scrunch, upchannelize
from .secondary import calculate_secondary_spectrum

__all__ = [
    "ScintillationAnalyser",
    "scrunch",
    "upchannelize",
    "calculate_acf_1d",
    "calculate_acf_2d",
    "lorentzian",
    "lorentzian_with_const",
    "double_lorentzian_with_const",
    "power_law",
    "fit_lorentzian_acf",
    "fit_scint_bandwidth_freq_relation",
    "calculate_secondary_spectrum",
    "scintillation_bandwidth_to_timescale",
    "effective_velocity",
    "screen_distance_from_curvature",
    "weighted_avg_and_std",
    "interpret_modulation_index",
    "estimate_emission_region_size",
    "two_screen_coherence_constraint",
]
