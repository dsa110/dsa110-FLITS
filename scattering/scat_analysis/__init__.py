from .burstfit import FRBFitter, FRBModel, FRBParams
from .pipeline.core import BurstPipeline
from .pipeline.diagnostics import BurstDiagnostics

# Refactored pipeline components
from .pipeline.io import BurstDataset

# from .burstfit_interactive import InitialGuessWidget # Requires ipywidgets
from .visualization import plot_scattering_diagnostic

__all__ = [
    "FRBModel",
    "FRBParams",
    "FRBFitter",
    "BurstDataset",
    "BurstPipeline",
    "BurstDiagnostics",
    # "InitialGuessWidget",
    "plot_scattering_diagnostic",
]

from .burstfit import build_priors

# Model selection (BIC-based)
from .burstfit_modelselect import fit_models_bic
from .dm_preprocessing import refine_dm_init

# Nested sampling (evidence-based model selection)
try:
    from .burstfit_nested import (
        NestedSamplingResult,
        fit_models_evidence,
    )
except ImportError:
    # dynesty not installed
    fit_models_evidence = None
    NestedSamplingResult = None

# Physical priors from NE2001
try:
    from .priors_physical import (
        PhysicalPriors,
        build_physical_priors,
        get_burst_priors_from_catalog,
        get_ne2001_scattering,
    )
except ImportError:
    # mwprop not installed
    build_physical_priors = None
    get_ne2001_scattering = None
    PhysicalPriors = None
    get_burst_priors_from_catalog = None

# Robustness diagnostics
# Data-driven initial guess estimation
from .burstfit_init import (
    InitialGuessResult,
    data_driven_initial_guess,
    estimate_pulse_width,
    estimate_scattering_from_tail,
    estimate_spectral_index,
    quick_initial_guess,
)
from .burstfit_robust import (
    dm_optimization_check,
    fit_subband_profiles,
    leave_one_out_influence,
    subband_consistency,
)
