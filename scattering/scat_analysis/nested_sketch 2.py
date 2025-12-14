"""
nested_sketch.py
================

Placeholder for future nested sampling integration (dynesty/ultranest).

This module sketches how to wire the existing FRBModel + likelihood into a
nested sampler without changing the core model/priors interface.

Notes:
- Keep BIC/emcee default; nested is earmarked only.
- Prior transform maps unit cube u in [0,1]^D to param space using the same
  top-hat priors used by the emcee path (and Gaussian prior for alpha via
  reparam or log-prior addition inside loglik).
- The loglik function should reuse model.log_likelihood or .log_likelihood_student_t.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

def prior_transform(u: np.ndarray, priors: Dict[str, Tuple[float, float]], names: Tuple[str, ...]) -> np.ndarray:
    """Map unit-cube vector u to parameter vector theta using top-hat priors.

    For alpha Gaussian prior, use a broad top-hat here and add the Gaussian term
    inside the log-likelihood (to avoid truncation bias).
    """
    theta = []
    for ui, name in zip(u, names):
        lo, hi = priors[name]
        theta.append(lo + (hi - lo) * ui)
    return np.asarray(theta, dtype=float)

def loglik_theta(theta: np.ndarray, names: Tuple[str, ...], model, model_key: str,
                 alpha_prior: Tuple[float, float] | None = None,
                 likelihood_kind: str = "gaussian", student_nu: float = 5.0) -> float:
    """Compute log-likelihood (plus Gaussian alpha prior term) for nested sampling.

    This mirrors the emcee path but expects full-precision theta (no log-space
    transforms). For multi-component, names encodes shared + per-component params
    and should build the summed model before comparing to data.
    """
    from .burstfit import FRBParams

    if model_key == "M3":
        params = FRBParams.from_sequence(theta, model_key)
        if alpha_prior is not None and alpha_prior[1] is not None:
            mu, sigma = alpha_prior
            a = params.alpha
            logp_a = -0.5 * ((a - mu)/sigma)**2 - np.log(sigma * np.sqrt(2*np.pi))
        else:
            logp_a = 0.0
        if likelihood_kind == "gaussian":
            return logp_a + model.log_likelihood(params, model_key)
        else:
            return logp_a + model.log_likelihood_student_t(params, model_key, nu=student_nu)
    else:
        # Multi-component sketch (build summed model like pipeline does)
        # Intentionally omitted for brevity in this earmark.
        return -np.inf

def run_nested_fit():
    """Earmark: dynesty/ultranest driver (not implemented).

    Pseudocode (dynesty):
        from dynesty import NestedSampler
        ns = NestedSampler(loglik, prior_transform, ndim, logl_args=(...), ptform_args=(...))
        ns.run_nested()
        res = ns.results
        return res.logz, res.samples, res.weights
    """
    raise NotImplementedError("Nested sampling integration is earmarked; not implemented yet.")


