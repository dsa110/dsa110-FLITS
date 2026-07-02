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
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from .burstfit import FRBModel, FRBParams, build_priors
from .turbulence import beta_from_alpha_thin_screen

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
    "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "beta", "delta_dm"),
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


class _PriorTransform:
    """Picklable prior transform (unit cube -> parameter space) for dynesty.

    A module-level callable instance, not a closure, so it can be pickled to
    multiprocessing/dynesty.pool workers.
    """

    def __init__(
        self,
        priors: Dict[str, Tuple[float, float]],
        param_names: Tuple[str, ...],
        log_params: Tuple[str, ...] = ("c0", "tau_1ghz", "zeta"),
    ):
        self.priors = priors
        self.param_names = param_names
        self.log_params = log_params

    def __call__(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        theta = np.zeros_like(u)
        for i, name in enumerate(self.param_names):
            lo, hi = self.priors[name]
            if name in self.log_params and lo > 0 and hi > 0:
                log_lo, log_hi = np.log(lo), np.log(hi)
                theta[i] = np.exp(log_lo + u[i] * (log_hi - log_lo))
            else:
                theta[i] = lo + u[i] * (hi - lo)
        return theta


def _build_prior_transform(
    priors: Dict[str, Tuple[float, float]],
    param_names: Tuple[str, ...],
    log_params: Tuple[str, ...] = ("c0", "tau_1ghz", "zeta"),
):
    """Return a picklable prior-transform callable for dynesty."""
    return _PriorTransform(priors, param_names, log_params)


class _LogLikelihood:
    """Picklable log-likelihood callable for dynesty.

    Module-level callable (not a closure) so dynesty.pool can ship it to
    worker processes; FRBModel holds only numpy arrays + scalars and pickles.
    Optional Gaussian priors on alpha / log10(tau) are folded into the
    log-likelihood (dynesty's prior_transform handles the rest).
    """

    def __init__(
        self,
        model: FRBModel,
        model_key: str,
        param_names: Tuple[str, ...],
        beta_prior: Optional[Tuple[float, float]] = None,
        alpha_prior: Optional[Tuple[float, float]] = None,
        tau_prior: Optional[Tuple[float, float]] = None,
        likelihood_kind: str = "gaussian",
        student_nu: float = 5.0,
        fixed_params: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.model_key = model_key
        self.param_names = param_names
        self.full_param_names = _PARAM_KEYS[model_key]
        self.beta_prior = beta_prior
        if self.beta_prior is None and alpha_prior is not None:
            mu_a, sigma_a = alpha_prior
            if sigma_a is None or sigma_a <= 0.0:
                # (mu, None) encodes "fixed alpha"; fixing is handled by
                # fixed_params / the prior transform, not a Gaussian factor.
                self.beta_prior = None
            else:
                # Jacobian-convert the width: sigma_beta = sigma_alpha *
                # 4/(alpha-2)^2 (same transform as apply_physical_priors).
                self.beta_prior = (
                    beta_from_alpha_thin_screen(mu_a),
                    sigma_a * 4.0 / (mu_a - 2.0) ** 2,
                )
        self.tau_prior = tau_prior
        self.likelihood_kind = likelihood_kind
        self.student_nu = student_nu
        self.fixed_params = fixed_params

    def __call__(self, theta: NDArray[np.floating]) -> float:
        if self.fixed_params:
            full_theta = []
            theta_ptr = 0
            for name in self.full_param_names:
                if name in self.fixed_params:
                    full_theta.append(self.fixed_params[name])
                else:
                    full_theta.append(theta[theta_ptr])
                    theta_ptr += 1
            params = FRBParams.from_sequence(full_theta, self.model_key)
        else:
            params = FRBParams.from_sequence(theta, self.model_key)

        if self.likelihood_kind == "gaussian":
            ll = self.model.log_likelihood(params, self.model_key)
        else:
            ll = self.model.log_likelihood_student_t(
                params, self.model_key, nu=self.student_nu
            )

        if not np.isfinite(ll):
            return -1e100

        if self.beta_prior is not None and "beta" in self.param_names:
            mu, sigma = self.beta_prior
            ll += -0.5 * ((params.beta - mu) / sigma) ** 2

        if self.tau_prior is not None and "tau_1ghz" in self.param_names:
            mu_log10, sigma_log10 = self.tau_prior
            tau = params.tau_1ghz
            if tau > 0:
                ll += -0.5 * ((np.log10(tau) - mu_log10) / sigma_log10) ** 2
            else:
                ll = -1e100

        return ll


def _build_log_likelihood(
    model: FRBModel,
    model_key: str,
    param_names: Tuple[str, ...],
    beta_prior: Optional[Tuple[float, float]] = None,
    alpha_prior: Optional[Tuple[float, float]] = None,
    tau_prior: Optional[Tuple[float, float]] = None,
    likelihood_kind: str = "gaussian",
    student_nu: float = 5.0,
    fixed_params: Optional[Dict[str, float]] = None,
):
    """Return a picklable log-likelihood callable for dynesty."""
    return _LogLikelihood(
        model, model_key, param_names, beta_prior, alpha_prior, tau_prior,
        likelihood_kind, student_nu, fixed_params,
    )


def fit_single_model_nested(
    *,
    model: FRBModel,
    init: FRBParams,
    model_key: str = "M3",
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    nlive: int = 500,
    dlogz: float = 0.1,
    beta_prior: Optional[Tuple[float, float]] = None,
    alpha_prior: Optional[Tuple[float, float]] = None,
    beta_fixed: Optional[float] = None,
    alpha_fixed: Optional[float] = None,
    tau_prior: Optional[Tuple[float, float]] = None,
    likelihood_kind: str = "gaussian",
    student_nu: float = 5.0,
    sample: str = "rwalk",
    nproc: Optional[int] = None,
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
    
    if beta_fixed is None and alpha_fixed is not None:
        beta_fixed = beta_from_alpha_thin_screen(alpha_fixed)

    # Remove beta from param list if fixed
    if beta_fixed is not None and "beta" in param_names:
        param_names = tuple(p for p in param_names if p != "beta")
        init = FRBParams(**{**init.__dict__, "beta": beta_fixed})
    
    ndim = len(param_names)
    
    # Build priors if not provided
    if priors is None:
        # Nested sampling explores the prior, so the prior must be init-independent
        # (otherwise the evidence — and thus M0–M3 selection — depends on the init).
        full_priors, _ = build_priors(
            init, scale=6.0, log_weight_pos=True, absolute_bounds=True
        )
        priors = {k: full_priors[k] for k in param_names if k in full_priors}
    
    # Build callable functions
    prior_transform = _build_prior_transform(priors, param_names)
    log_likelihood = _build_log_likelihood(
        model, model_key, param_names, beta_prior, alpha_prior, tau_prior, likelihood_kind, student_nu,
        fixed_params={"beta": beta_fixed} if beta_fixed is not None and "beta" in _PARAM_KEYS[model_key] else None
    )
    
    # Run nested sampling
    if verbose:
        log.info(f"Running nested sampling for {model_key} with nlive={nlive}")
    
    if nproc is not None and nproc > 1:
        # Parallel: dynesty.pool.Pool ships the (picklable) loglike/ptform to
        # workers once, then evaluates queue_size live-point proposals at a time.
        # Force 'fork' so workers inherit memory instead of re-importing __main__
        # (the 'spawn' default re-imports the entry script and crashes).
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        from dynesty import pool as dypool

        with dypool.Pool(int(nproc), log_likelihood, prior_transform) as pool:
            sampler = NestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                nlive=nlive,
                sample=sample,
                pool=pool,
                queue_size=int(nproc),
                **dynesty_kwargs,
            )
            sampler.run_nested(dlogz=dlogz, print_progress=verbose)
            results = sampler.results
    else:
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
        fixed_params={"beta": beta_fixed} if beta_fixed is not None and "beta" in _PARAM_KEYS[model_key] else {},
    )


def fit_models_evidence(
    *,
    model: FRBModel,
    init: FRBParams,
    model_keys: Sequence[str] = ("M0", "M1", "M2", "M3"),
    nlive: int = 500,
    dlogz: float = 0.1,
    beta_prior: Optional[Tuple[float, float]] = None,
    alpha_prior: Optional[Tuple[float, float]] = None,
    beta_fixed: Optional[float] = None,
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
            beta_prior=beta_prior,
            alpha_prior=alpha_prior,
            beta_fixed=beta_fixed,
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
