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

from burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    compute_bic,
)

__all__ = ["fit_models_bic"]

# Map of parameter names for each model key ---------------------------
_PARAM_KEYS = {
    "M0": ("c0", "t0", "gamma"),
    "M1": ("c0", "t0", "gamma", "zeta"),
    "M2": ("c0", "t0", "gamma", "tau_1ghz"),
    "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
}

# ---------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------

def _restrict_params(p: FRBParams, key: str) -> FRBParams:
    d = p.__dict__.copy()
    keep = {k: d.get(k, 0.0) for k in _PARAM_KEYS[key]}
    # ensure optional fields exist
    keep.setdefault("zeta", 0.0)
    keep.setdefault("tau_1ghz", 0.0)
    return FRBParams(**keep)  # type: ignore[arg-type]


def _restrict_priors(pri: Dict[str, Tuple[float, float]], key: str):
    return {k: pri[k] for k in _PARAM_KEYS[key]}

# ---------------------------------------------------------------------
# public driver
# ---------------------------------------------------------------------

def fit_models_bic(
    *,
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm_init: float,
    init: FRBParams,
    model_keys: Sequence[str] = ("M0", "M1", "M2", "M3"),
    n_steps: int = 1500,
    pool = None,
) -> Tuple[str, Dict[str, Tuple["emcee.EnsembleSampler", float, float]]]:
    """Fit each model, compute BIC, return best.

    Parameters
    ----------
    data, freq, time, dm_init
        Dynamic spectrum and axes.
    init
        Initial parameter guess (full 5‑param set).  Will be projected
        onto simpler models automatically.
    model_keys
        Iterable subset of {"M0", "M1", "M2", "M3"}.
    n_steps
        Chain length **per** model.  Keep modest for evidence scan.
    pool
        Passed to :class:`burstfit.FRBFitter`.

    Returns
    -------
    best_key
        Key with lowest BIC.
    results
        Dict mapping key → (sampler, bic, logL_max).
    """
    n_obs = data.size
    base_model = FRBModel(time=time, freq=freq, data=data, dm_init=dm_init)

    results: Dict[str, Tuple] = {}

    for key in model_keys:
        p0 = _restrict_params(init, key)
        pri = _restrict_priors(build_priors(init, scale=3.0), key)

        fitter = FRBFitter(base_model, pri, n_steps=n_steps, pool=pool)
        sampler = fitter.sample(p0, model_key=key)

        logL_max = float(np.nanmax(sampler.get_log_prob(flat=True)))
        bic_val = compute_bic(logL_max, k=len(_PARAM_KEYS[key]), n=n_obs)
        results[key] = (sampler, bic_val, logL_max)

        print(f"[Model {key}]  logL_max = {logL_max:8.1f} | BIC = {bic_val:8.1f}")

    best_key = min(results, key=lambda k: results[k][1])
    print(f"\n→ Best model by BIC: {best_key}")
    return best_key, results
