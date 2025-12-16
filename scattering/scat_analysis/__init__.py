from .burstfit_pipeline import BurstPipeline, BurstDataset, BurstDiagnostics
from .burstfit import FRBModel, FRBFitter, FRBParams, build_priors
from .dm_preprocessing import refine_dm_init

# Model selection (BIC-based)
from .burstfit_modelselect import fit_models_bic

# Nested sampling (evidence-based model selection)
try:
    from .burstfit_nested import (
        fit_models_evidence,
        fit_single_model_nested,
        NestedSamplingResult,
        interpret_bayes_factor,
    )
except ImportError:
    # dynesty not installed
    fit_models_evidence = None
    fit_single_model_nested = None
    NestedSamplingResult = None
    interpret_bayes_factor = None

# Physical priors from NE2001
try:
    from .priors_physical import (
        build_physical_priors,
        get_ne2001_scattering,
        PhysicalPriors,
        get_burst_priors_from_catalog,
    )
except ImportError:
    # mwprop not installed
    build_physical_priors = None
    get_ne2001_scattering = None
    PhysicalPriors = None
    get_burst_priors_from_catalog = None

# Robustness diagnostics
from .burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    dm_optimization_check,
    fit_subband_profiles,
)

# Data-driven initial guess estimation
from .burstfit_init import (
    data_driven_initial_guess,
    quick_initial_guess,
    estimate_spectral_index,
    estimate_pulse_width,
    estimate_scattering_from_tail,
    InitialGuessResult,
)
