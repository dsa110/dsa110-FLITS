# ==============================================================================
# File: scint_analysis/scint_analysis/__init__.py
# ==============================================================================
# This makes the core classes, main pipeline controller, and key utilities
# easily importable from the top level of the package.

# Core data structures from core.py
from .core import DynamicSpectrum, ACF

# Configuration loading from config.py
from .config import load_config

# Noise modeling and synthesis tools from noise_model.py
from .noise import NoiseDescriptor, estimate_noise_descriptor

# Main pipeline controller from pipeline.py
# Now, when this is imported, the noise_model is already known to the package.
from .pipeline import ScintillationAnalysis

# Key plotting functions from plotting.py
from .plotting import (
    plot_analysis_overview, 
    plot_noise_distribution,
    plot_2d_fit_overview,
    plot_gamma_scaling,
    plot_2d_acf_grid,
)

# 2D fitting module
from .fitting_2d import (
    Scintillation2DModel,
    Scintillation2DResult,
    fit_2d_scintillation,
    compare_1d_vs_2d,
)

# Interpretation functions based on Bhat et al. (2004) and Nimmo et al. (2025)
from .analysis import (
    interpret_modulation_index,
    estimate_emission_region_size,
    two_screen_coherence_constraint,
    scattering_scintillation_consistency,
)

# Interactive widgets for manual analysis (requires ipywidgets)
try:
    from . import widgets
except ImportError:
    widgets = None  # widgets module unavailable without ipywidgets