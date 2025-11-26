"""
burstfit_modelselect.py
=======================

Sequential evidence scanner for the FRB dynamic‐spectrum model family
M0 → M3.  Each model is fitted with a *short* MCMC run, its maximum
log‑likelihood extracted, and the Bayesian Information Criterion (BIC)
computed:

\[\mathrm{BIC}= -2\log L_{\max} + k\ln n\]

The model with the smallest BIC is considered the preferred description
of the data.  The user can supply any subset of model keys; the order of
`model_keys` dictates fit order.

Typical usage
-------------
```
python
from burstfit_modelselect import fit_models_bic
best_key, res = fit_models_bic(
    data=ds, freq=f, time=t,
    dm_init=0.0, init=p0,
    n_steps=1500, pool=None,
)
print("Winner:", best_key)
# sampler, bic, logL_max for the best model
sampler = res[best_key][0]
```

Returned structure
------------------
```
results[key] = (sampler, bic_value, logL_max)
```
This allows re‑running a longer chain after selecting the best model.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    compute_bic,
)

__all__ = ["fit_models_bic"]

_PARAM_KEYS = {
    "M0": ("c0", "t0", "gamma"),
    "M1": ("c0", "t0", "gamma", "zeta"),
    "M2": ("c0", "t0", "gamma", "tau_1ghz"),
    "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz", "alpha", "delta_dm"),
}

# ---------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------

def _restrict_priors(pri: Dict[str, Tuple[float, float]], key: str):
    """Selects only the priors needed for a given model key."""
    return {k: pri[k] for k in _PARAM_KEYS[key]}

# ---------------------------------------------------------------------
# public driver
# ---------------------------------------------------------------------

def fit_models_bic(
    *,
    model: FRBModel,
    init: FRBParams,
    model_keys: Sequence[str] = ("M0", "M1", "M2", "M3"),
    n_steps: int = 1500,
    pool=None,
    **fitter_kwargs
) -> tuple[str, dict[str, tuple["emcee.EnsembleSampler", float, float]]]:

    if model.data is None:
        raise ValueError("The FRBModel instance must contain data for fitting.")

    n_obs = model.data.size
    results: dict[str, tuple] = {}

    # 1. global prior dict + Jeffreys flag
    full_priors, use_logw = build_priors(
        init, scale=6.0, log_weight_pos=True
    )

    for key in model_keys:
        # 2. keep only the params relevant for this model
        pri_subset = _restrict_priors(full_priors, key)

        fitter = FRBFitter(
            model,
            pri_subset,
            n_steps=n_steps,
            pool=pool,
            log_weight_pos=use_logw,
            **fitter_kwargs,
        )
        sampler = fitter.sample(init, model_key=key)

        logL_max = float(np.nanmax(sampler.get_log_prob(flat=True)))
        bic_val  = compute_bic(logL_max,
                               k=len(_PARAM_KEYS[key]),
                               n=n_obs)
        results[key] = (sampler, bic_val, logL_max)
        print(f"[Model {key}]  logL_max = {logL_max:9.1f} | BIC = {bic_val:9.1f}")

    best_key = min(results, key=lambda k: results[k][1])
    print(f"\n→ Best model by BIC: {best_key}")
    return best_key, results