"""
burstfit_nested.py
==================

Nested sampling implementation for rigorous Bayesian model comparison.

Uses dynesty to compute the Bayesian evidence (marginal likelihood) for each
model in the FRB dynamic spectrum model family (M0-M3). This enables
model comparison via Bayes factors instead of BIC.

Key advantages over BIC:
- Rigorous marginalization over parameter space
- No asymptotic approximations
- Proper handling of prior volume
- Uncertainty estimates on evidence

Usage
-----
```python
from burstfit_nested import fit_models_evidence

best_model, results = fit_models_evidence(
    model=frb_model,
    init=initial_params,
    model_keys=["M0", "M1", "M2", "M3"],
)

# Bayes factor between M3 and M2
ln_bf = results["M3"]["log_evidence"] - results["M2"]["log_evidence"]
print(f"ln(BF) = {ln_bf:.1f}")  # >5 is strong evidence for M3
```

Bayes Factor Interpretation (Jeffreys' scale):
- |ln(BF)| < 1: Inconclusive
- 1 < |ln(BF)| < 2.5: Weak
- 2.5 < |ln(BF)| < 5: Moderate  
- |ln(BF)| > 5: Strong
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from .burstfit import FRBModel, FRBParams, build_priors

log = logging.getLogger(__name__)

__all__ = [
    "fit_models_evidence",
    "fit_single_model_nested",
    "quick_evidence_comparison",
    "NestedSamplingResult",
    "interpret_bayes_factor",
]

# Model parameter definitions
_PARAM_KEYS = {
    "M0": ("c0", "t0", "gamma"),
    "M1": ("c0", "t0", "gamma", "zeta"),
    "M2": ("c0", "t0", "gamma", "tau_1ghz"),
    "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
}


@dataclass
class NestedSamplingResult:
    """Container for nested sampling results."""
    
    log_evidence: float
    log_evidence_err: float
    samples: NDArray[np.floating]
    weights: NDArray[np.floating]
    param_names: Tuple[str, ...]
    model_key: str
    nlive: int
    ncall: int
    
    # Derived quantities
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fixed_params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute parameter percentiles from weighted samples."""
        if len(self.percentiles) == 0:
            self._compute_percentiles()
    
    def _compute_percentiles(self):
        """Compute median and 68% CI for each parameter."""
        for i, name in enumerate(self.param_names):
            samples = self.samples[:, i]
            # Weighted percentiles
            sorted_idx = np.argsort(samples)
            sorted_samples = samples[sorted_idx]
            sorted_weights = self.weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            cumsum /= cumsum[-1]
            
            p16 = sorted_samples[np.searchsorted(cumsum, 0.16)]
            p50 = sorted_samples[np.searchsorted(cumsum, 0.50)]
            p84 = sorted_samples[np.searchsorted(cumsum, 0.84)]
            
            self.percentiles[name] = {
                "median": p50,
                "lower": p16,
                "upper": p84,
                "err_minus": p50 - p16,
                "err_plus": p84 - p50,
            }
    
    def get_best_params(self) -> FRBParams:
        """Return FRBParams at posterior median."""
        full_names = _PARAM_KEYS[self.model_key]
        theta = []
        for name in full_names:
            if name in self.fixed_params:
                theta.append(self.fixed_params[name])
            elif name in self.percentiles:
                theta.append(self.percentiles[name]["median"])
            else:
                # Should not happen if logic is correct
                theta.append(0.0) 
        return FRBParams.from_sequence(theta, self.model_key)
    
    def __repr__(self) -> str:
        return (
            f"NestedSamplingResult(\n"
            f"  model={self.model_key}\n"
            f"  log_evidence={self.log_evidence:.2f} ± {self.log_evidence_err:.2f}\n"
            f"  nlive={self.nlive}, ncall={self.ncall}\n"
            f")"
        )


def interpret_bayes_factor(ln_bf: float, model1: str = "first model", model2: str = "second model") -> str:
    """Interpret log Bayes factor on Jeffreys' scale.
    
    Parameters
    ----------
    ln_bf : float
        Natural log of Bayes factor = ln(Z_1/Z_2)
    model1 : str
        Name of first model (positive BF favors this)
    model2 : str
        Name of second model (negative BF favors this)
    """
    abs_bf = abs(ln_bf)
    if abs_bf < 1:
        strength = "Inconclusive"
    elif abs_bf < 2.5:
        strength = "Weak"
    elif abs_bf < 5:
        strength = "Moderate"
    else:
        strength = "Strong"
    
    direction = f"favors {model1}" if ln_bf > 0 else f"favors {model2}"
    return f"{strength} evidence {direction} (ln(BF) = {ln_bf:.2f})"


def _build_prior_transform(
    priors: Dict[str, Tuple[float, float]],
    param_names: Tuple[str, ...],
    log_params: Tuple[str, ...] = ("c0", "tau_1ghz", "zeta"),
):
    """Build prior transform function for dynesty.
    
    Maps unit cube [0,1]^D to parameter space.
    For log-scale parameters, uses log-uniform prior.
    """
    def prior_transform(u: NDArray[np.floating]) -> NDArray[np.floating]:
        theta = np.zeros_like(u)
        for i, name in enumerate(param_names):
            lo, hi = priors[name]
            if name in log_params and lo > 0 and hi > 0:
                # Log-uniform prior
                log_lo, log_hi = np.log(lo), np.log(hi)
                theta[i] = np.exp(log_lo + u[i] * (log_hi - log_lo))
            else:
                # Uniform prior
                theta[i] = lo + u[i] * (hi - lo)
        return theta
    
    return prior_transform


def _build_log_likelihood(
    model: FRBModel,
    model_key: str,
    param_names: Tuple[str, ...],
    alpha_prior: Optional[Tuple[float, float]] = None,
    tau_prior: Optional[Tuple[float, float]] = None,
    likelihood_kind: str = "gaussian",
    student_nu: float = 5.0,
    fixed_params: Optional[Dict[str, float]] = None,
):
    """Build log-likelihood function for dynesty.
    
    Optionally includes priors on alpha and tau (as part of log-likelihood
    to avoid issues with nested sampling prior volume calculation).
    
    Parameters
    ----------
    alpha_prior : tuple, optional
        (mu, sigma) for Gaussian prior on α
    tau_prior : tuple, optional
        (mu, sigma) for log-normal prior on τ (in log10 space)
    """
    full_param_names = _PARAM_KEYS[model_key]

    def log_likelihood(theta: NDArray[np.floating]) -> float:
        # Reconstruct full theta vector if we have fixed params
        if fixed_params:
            full_theta = []
            # We need to map the current sub-theta to names
            # But theta only has values. We assume theta follows 'param_names' order.
            # param_names is the REDUCED list.
            theta_ptr = 0
            for name in full_param_names:
                if name in fixed_params:
                    full_theta.append(fixed_params[name])
                else:
                    full_theta.append(theta[theta_ptr])
                    theta_ptr += 1
            params = FRBParams.from_sequence(full_theta, model_key)
        else:
            # Standard path
            params = FRBParams.from_sequence(theta, model_key)
        
        # Base likelihood
        if likelihood_kind == "gaussian":
            ll = model.log_likelihood(params, model_key)
        else:
            ll = model.log_likelihood_student_t(params, model_key, nu=student_nu)
        
        # Guard against NaN/Inf likelihoods
        if not np.isfinite(ll):
            return -1e100  # Very negative but finite
        
        # Add Gaussian prior on alpha if specified
        if alpha_prior is not None and "alpha" in param_names:
            mu, sigma = alpha_prior
            alpha = params.alpha
            ll += -0.5 * ((alpha - mu) / sigma) ** 2
        
        # Add log-normal prior on tau if specified (in log10 space)
        if tau_prior is not None and "tau_1ghz" in param_names:
            mu_log10, sigma_log10 = tau_prior
            tau = params.tau_1ghz
            if tau > 0:
                log10_tau = np.log10(tau)
                ll += -0.5 * ((log10_tau - mu_log10) / sigma_log10) ** 2
            else:
                ll = -1e100
        
        return ll
    
    return log_likelihood


def fit_single_model_nested(
    *,
    model: FRBModel,
    init: FRBParams,
    model_key: str = "M3",
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    nlive: int = 500,
    dlogz: float = 0.1,
    alpha_prior: Optional[Tuple[float, float]] = None,
    alpha_fixed: Optional[float] = None,
    tau_prior: Optional[Tuple[float, float]] = None,
    likelihood_kind: str = "gaussian",
    student_nu: float = 5.0,
    sample: str = "rwalk",
    verbose: bool = True,
    **dynesty_kwargs,
) -> NestedSamplingResult:
    """Run nested sampling for a single model.
    
    Parameters
    ----------
    model : FRBModel
        The burst model with data loaded
    init : FRBParams
        Initial parameter estimates (used for prior scaling)
    model_key : str
        Model to fit ("M0", "M1", "M2", "M3")
    priors : dict, optional
        Prior bounds. If None, auto-built from init
    nlive : int
        Number of live points
    dlogz : float
        Stopping criterion (evidence tolerance)
    alpha_prior : tuple, optional
        (mu, sigma) for Gaussian prior on alpha
    likelihood_kind : str
        "gaussian" or "student_t"
    student_nu : float
        Degrees of freedom for Student-t likelihood
    sample : str
        Dynesty sampling method ("rwalk", "slice", "auto")
    verbose : bool
        Print progress
    **dynesty_kwargs
        Passed to NestedSampler
        
    Returns
    -------
    NestedSamplingResult
        Evidence and posterior samples
    
    Notes
    -----
    alpha_prior and tau_prior are included as part of the log-likelihood
    (rather than the prior transform) to avoid issues with evidence
    calculation when using informative priors.
    """
    try:
        from dynesty import NestedSampler
    except ImportError:
        raise ImportError(
            "dynesty is required for nested sampling. "
            "Install with: pip install dynesty"
        )
    
    param_names = _PARAM_KEYS[model_key]
    
    # Remove alpha from param list if fixed
    if alpha_fixed is not None and "alpha" in param_names:
        param_names = tuple(p for p in param_names if p != "alpha")
        # Set alpha in init to fixed value
        init = FRBParams(**{**init.__dict__, "alpha": alpha_fixed})
    
    ndim = len(param_names)
    
    # Build priors if not provided
    if priors is None:
        full_priors, _ = build_priors(init, scale=6.0, log_weight_pos=True)
        priors = {k: full_priors[k] for k in param_names if k in full_priors}
    
    # Build callable functions
    prior_transform = _build_prior_transform(priors, param_names)
    log_likelihood = _build_log_likelihood(
        model, model_key, param_names, alpha_prior, tau_prior, likelihood_kind, student_nu,
        fixed_params={"alpha": alpha_fixed} if alpha_fixed is not None and "alpha" in _PARAM_KEYS[model_key] else None
    )
    
    # Run nested sampling
    if verbose:
        log.info(f"Running nested sampling for {model_key} with nlive={nlive}")
    
    sampler = NestedSampler(
        log_likelihood,
        prior_transform,
        ndim,
        nlive=nlive,
        sample=sample,
        **dynesty_kwargs,
    )
    
    sampler.run_nested(dlogz=dlogz, print_progress=verbose)
    results = sampler.results
    
    # Extract weights
    weights = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()
    
    return NestedSamplingResult(
        log_evidence=results.logz[-1],
        log_evidence_err=results.logzerr[-1],
        samples=results.samples,
        weights=weights,
        param_names=param_names,
        model_key=model_key,
        nlive=nlive,
        ncall=results.ncall,
        fixed_params={"alpha": alpha_fixed} if alpha_fixed is not None and "alpha" in _PARAM_KEYS[model_key] else {},
    )


def fit_models_evidence(
    *,
    model: FRBModel,
    init: FRBParams,
    model_keys: Sequence[str] = ("M0", "M1", "M2", "M3"),
    nlive: int = 500,
    dlogz: float = 0.1,
    alpha_prior: Optional[Tuple[float, float]] = None,
    alpha_fixed: Optional[float] = None,
    tau_prior: Optional[Tuple[float, float]] = None,
    likelihood_kind: str = "gaussian",
    verbose: bool = True,
    **dynesty_kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Compare models using Bayesian evidence from nested sampling.
    
    This is the main entry point for evidence-based model selection.
    
    Parameters
    ----------
    model : FRBModel
        The burst model with data loaded
    init : FRBParams
        Initial parameter estimates
    model_keys : sequence of str
        Models to compare (default: all M0-M3)
    nlive : int
        Number of live points per model
    dlogz : float
        Evidence tolerance for stopping
    alpha_prior : tuple, optional
        (mu, sigma) for Gaussian prior on alpha
    likelihood_kind : str
        "gaussian" or "student_t"
    verbose : bool
        Print progress and results
    **dynesty_kwargs
        Passed to NestedSampler
        
    Returns
    -------
    best_model : str
        Model with highest evidence
    results : dict
        Dictionary with NestedSamplingResult for each model plus Bayes factors
        
    Examples
    --------
    >>> best, results = fit_models_evidence(model=model, init=p0)
    >>> print(f"Best model: {best}")
    >>> print(f"Evidence: {results[best].log_evidence:.1f}")
    """
    if model.data is None:
        raise ValueError("FRBModel must have data loaded for fitting.")
    
    results = {}
    
    for key in model_keys:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fitting model {key}")
            print(f"{'='*50}")
        
        result = fit_single_model_nested(
            model=model,
            init=init,
            model_key=key,
            nlive=nlive,
            dlogz=dlogz,
            alpha_prior=alpha_prior,
            alpha_fixed=alpha_fixed,
            tau_prior=tau_prior,
            likelihood_kind=likelihood_kind,
            verbose=verbose,
            **dynesty_kwargs,
        )
        
        results[key] = result
        
        if verbose:
            print(f"[{key}] log(Z) = {result.log_evidence:.2f} ± {result.log_evidence_err:.2f}")
    
    # Find best model
    best_key = max(results, key=lambda k: results[k].log_evidence)
    
    # Compute all pairwise Bayes factors
    bayes_factors = {}
    for k1, k2 in combinations(model_keys, 2):
        ln_bf = results[k1].log_evidence - results[k2].log_evidence
        bayes_factors[f"ln_BF_{k1}_vs_{k2}"] = ln_bf
    
    results["bayes_factors"] = bayes_factors
    
    if verbose:
        print(f"\n{'='*50}")
        print("Model Comparison Summary")
        print(f"{'='*50}")
        for key in model_keys:
            r = results[key]
            delta = r.log_evidence - results[best_key].log_evidence
            marker = "← BEST" if key == best_key else f"(ΔlnZ = {delta:.1f})"
            print(f"{key}: log(Z) = {r.log_evidence:8.2f} ± {r.log_evidence_err:.2f}  {marker}")
        
        print(f"\n→ Best model by evidence: {best_key}")
        
        # Report relevant Bayes factors
        print("\nBayes factors:")
        for (k1, k2) in [("M3", "M2"), ("M3", "M1"), ("M2", "M1")]:
            if k1 in model_keys and k2 in model_keys:
                bf_key = f"ln_BF_{k1}_vs_{k2}"
                if bf_key in bayes_factors:
                    ln_bf = bayes_factors[bf_key]
                elif f"ln_BF_{k2}_vs_{k1}" in bayes_factors:
                    ln_bf = -bayes_factors[f"ln_BF_{k2}_vs_{k1}"]
                else:
                    ln_bf = results[k1].log_evidence - results[k2].log_evidence # Fallback calc

                interp = interpret_bayes_factor(ln_bf)
                print(f"  {k1} vs {k2}: {interp}")
    
    return best_key, results


# Convenience function for quick evidence comparison
def quick_evidence_comparison(
    model: FRBModel,
    init: FRBParams,
    nlive: int = 200,
) -> Tuple[str, float]:
    """Quick evidence comparison with fewer live points.
    
    For rapid prototyping; use fit_models_evidence() for publication.
    """
    best, results = fit_models_evidence(
        model=model,
        init=init,
        model_keys=("M2", "M3"),  # Just the scattering models
        nlive=nlive,
        dlogz=0.5,  # Looser tolerance
        verbose=False,
    )
    
    ln_bf = results["M3"].log_evidence - results["M2"].log_evidence
    return best, ln_bf
