"""BurstFit scattering pipeline (modular split of the former burstfit_pipeline monolith).

Public API re-exported here so callers can use a single import path:

    from scattering.scat_analysis.pipeline import BurstPipeline, BurstDataset
"""

from .core import BurstPipeline, build_safe_results
from .io import BurstDataset
from .optimization import refine_initial_guess_mle, auto_burn_thin
from .diagnostics import (
    BurstDiagnostics,
    create_four_panel_plot,
    create_fit_summary_plot,
)

__all__ = [
    "BurstPipeline",
    "build_safe_results",
    "BurstDataset",
    "refine_initial_guess_mle",
    "auto_burn_thin",
    "BurstDiagnostics",
    "create_four_panel_plot",
    "create_fit_summary_plot",
]
