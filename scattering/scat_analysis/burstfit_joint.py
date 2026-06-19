"""
burstfit_joint.py
=================

Joint two-telescope scattering fit (CHIME ~0.6 GHz + DSA ~1.4 GHz).

Single-band M3 fits with alpha fixed = 4 give *inconsistent* tau_1ghz between
CHIME and DSA for the same sightline (observed up to ~15x), which is only
possible if the scattering index alpha != 4. A single band cannot separate
alpha from tau (the tau-alpha degeneracy): tau(nu) = tau_1ghz * nu^-alpha, and
one band fixes the product at its own frequency, not the slope.

This module fits both bands simultaneously with a *shared* (tau_1ghz, alpha) and
*per-telescope* (c0, t0, gamma, zeta, delta_dm). The ~1 GHz lever arm between
the two bands measures alpha directly: the ratio tau_C/tau_D pins the slope,
the shared tau_1ghz pins the normalization at 1 GHz.

12-parameter vector (M3 both bands, alpha free):

    [tau_1ghz, alpha | c0_C, t0_C, gamma_C, zeta_C, ddm_C | c0_D, t0_D, gamma_D, zeta_D, ddm_D]
     ^shared sightline   ^CHIME intrinsic/timing            ^DSA intrinsic/timing

Independent noise -> the joint log-likelihood is the sum of the two single-band
Gaussian log-likelihoods (reuses the nch^2-fixed FRBModel.log_likelihood).

zeta is kept per-telescope (not shared): intrinsic width is achromatic in
principle, but the measured zeta also absorbs unmodelled per-band structure, so
the conservative choice is to let each band fit its own. The alpha lever arm
comes from the shared tau, independent of how zeta is treated.

Usage
-----
```python
from burstfit_joint import fit_joint_scattering
res = fit_joint_scattering(
    model_C=model_chime, init_C=init_chime,
    model_D=model_dsa,   init_D=init_dsa,
    alpha_bounds=(2.0, 6.0), nlive=600, nproc=8,
)
print(res["percentiles"]["alpha"], res["percentiles"]["tau_1ghz"])
```
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from .burstfit import FRBModel, FRBParams, build_priors

log = logging.getLogger(__name__)

__all__ = ["fit_joint_scattering", "JOINT_PARAM_NAMES"]

# Joint 12-vector layout. First two are the shared sightline params; the rest
# are per-telescope (suffix _C = CHIME, _D = DSA).
JOINT_PARAM_NAMES: Tuple[str, ...] = (
    "tau_1ghz", "alpha",
    "c0_C", "t0_C", "gamma_C", "zeta_C", "delta_dm_C",
    "c0_D", "t0_D", "gamma_D", "zeta_D", "delta_dm_D",
)
# Positive params sampled log-uniform (Jeffreys), mirroring burstfit_nested.
_LOG_NAMES = frozenset({"tau_1ghz", "c0_C", "zeta_C", "c0_D", "zeta_D"})


def _joint_prior_spec(
    init_C: FRBParams,
    init_D: FRBParams,
    alpha_bounds: Tuple[float, float],
) -> List[Tuple[str, Tuple[float, float], bool]]:
    """Assemble per-index (name, (lo, hi), is_log) from the single-band priors.

    Reuses build_priors(absolute_bounds=True) per telescope so the joint prior is
    init-independent (required for a global sampler) except t0, whose window is
    anchored on each band's data profile-peak estimate. alpha is widened to
    alpha_bounds to allow shallower-than-Kolmogorov slopes (the whole point).
    """
    pC, _ = build_priors(init_C, absolute_bounds=True)
    pD, _ = build_priors(init_D, absolute_bounds=True)
    # tau_1ghz bound is the absolute WIDTH_MIN..WIDTH_MAX (identical in pC/pD).
    by_name = {
        "tau_1ghz": pC["tau_1ghz"], "alpha": tuple(alpha_bounds),
        "c0_C": pC["c0"], "t0_C": pC["t0"], "gamma_C": pC["gamma"],
        "zeta_C": pC["zeta"], "delta_dm_C": pC["delta_dm"],
        "c0_D": pD["c0"], "t0_D": pD["t0"], "gamma_D": pD["gamma"],
        "zeta_D": pD["zeta"], "delta_dm_D": pD["delta_dm"],
    }
    return [(n, by_name[n], n in _LOG_NAMES) for n in JOINT_PARAM_NAMES]


class _JointPriorTransform:
    """Picklable unit-cube -> parameter transform for the 12-vector.

    Module-level callable (not a closure) so dynesty.pool can ship it to workers.
    Log-uniform on the flagged positive params, uniform on the rest.
    """

    def __init__(self, spec: List[Tuple[str, Tuple[float, float], bool]]):
        self.lo = np.array([s[1][0] for s in spec], dtype=float)
        self.hi = np.array([s[1][1] for s in spec], dtype=float)
        # only log-sample where flagged AND both bounds strictly positive
        self.is_log = np.array(
            [bool(s[2] and s[1][0] > 0 and s[1][1] > 0) for s in spec]
        )
        # precompute log-bounds with safe placeholders (log(1)=0) on linear axes
        self._loglo = np.log(np.where(self.is_log, self.lo, 1.0))
        self._loghi = np.log(np.where(self.is_log, self.hi, 1.0))

    def __call__(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        lin = self.lo + u * (self.hi - self.lo)
        logu = np.exp(self._loglo + u * (self._loghi - self._loglo))
        return np.where(self.is_log, logu, lin)


class _JointLogLikelihood:
    """Picklable joint log-likelihood: ll_CHIME(pC) + ll_DSA(pD).

    Two FRBModels sharing (tau_1ghz, alpha); independent noise -> additive.
    Both FRBModels hold only numpy arrays + scalars, so this pickles.
    """

    def __init__(self, model_C: FRBModel, model_D: FRBModel):
        self.model_C = model_C
        self.model_D = model_D

    def __call__(self, theta: NDArray[np.floating]) -> float:
        tau, alpha = float(theta[0]), float(theta[1])
        pC = FRBParams(
            c0=theta[2], t0=theta[3], gamma=theta[4], zeta=theta[5],
            tau_1ghz=tau, alpha=alpha, delta_dm=theta[6],
        )
        pD = FRBParams(
            c0=theta[7], t0=theta[8], gamma=theta[9], zeta=theta[10],
            tau_1ghz=tau, alpha=alpha, delta_dm=theta[11],
        )
        ll = (
            self.model_C.log_likelihood(pC, "M3")
            + self.model_D.log_likelihood(pD, "M3")
        )
        return ll if np.isfinite(ll) else -1e100


def _weighted_percentiles(
    samples: NDArray[np.floating], weights: NDArray[np.floating]
) -> Dict[str, Dict[str, float]]:
    """Weighted 16/50/84 percentiles per column (mirror NestedSamplingResult)."""
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(JOINT_PARAM_NAMES):
        s = samples[:, i]
        idx = np.argsort(s)
        ss, sw = s[idx], weights[idx]
        cdf = np.cumsum(sw)
        cdf /= cdf[-1]
        p16, p50, p84 = ss[np.searchsorted(cdf, [0.16, 0.50, 0.84])]
        out[name] = {
            "median": float(p50), "lower": float(p16), "upper": float(p84),
            "err_minus": float(p50 - p16), "err_plus": float(p84 - p50),
        }
    return out


def fit_joint_scattering(
    *,
    model_C: FRBModel,
    init_C: FRBParams,
    model_D: FRBModel,
    init_D: FRBParams,
    alpha_bounds: Tuple[float, float] = (2.0, 6.0),
    nlive: int = 600,
    dlogz: float = 0.5,
    nproc: Optional[int] = None,
    sample: str = "rwalk",
    verbose: bool = True,
    **dynesty_kwargs,
) -> Dict[str, Any]:
    """Run the joint CHIME+DSA nested fit; return posterior summary.

    Parameters
    ----------
    model_C, model_D : FRBModel
        CHIME and DSA burst models, each with data + noise loaded.
    init_C, init_D : FRBParams
        Per-band data-driven inits (used only to anchor the t0 prior window and
        scale-free absolute bounds).
    alpha_bounds : (lo, hi)
        Uniform prior on the shared scattering index. Default (2, 6) is wide
        enough to detect shallow (sub-Kolmogorov) slopes.
    nlive, dlogz, nproc, sample
        dynesty knobs (12-dim -> nlive ~600+ recommended).

    Returns
    -------
    dict with keys: param_names, percentiles, log_evidence, log_evidence_err,
    samples, weights, alpha_bounds.
    """
    from dynesty import NestedSampler

    if model_C.data is None or model_D.data is None:
        raise ValueError("both FRBModels must have data loaded")

    spec = _joint_prior_spec(init_C, init_D, alpha_bounds)
    ndim = len(spec)
    ptform = _JointPriorTransform(spec)
    loglike = _JointLogLikelihood(model_C, model_D)

    if verbose:
        log.info(f"Joint CHIME+DSA fit: ndim={ndim}, nlive={nlive}, alpha~U{alpha_bounds}")

    if nproc is not None and nproc > 1:
        # fork so workers inherit memory instead of re-importing __main__ (spawn
        # default crashes); identical pattern to burstfit_nested.
        import multiprocessing as _mp
        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        from dynesty import pool as dypool

        with dypool.Pool(int(nproc), loglike, ptform) as pool:
            sampler = NestedSampler(
                pool.loglike, pool.prior_transform, ndim,
                nlive=nlive, sample=sample, pool=pool, queue_size=int(nproc),
                **dynesty_kwargs,
            )
            sampler.run_nested(dlogz=dlogz, print_progress=verbose)
            results = sampler.results
    else:
        sampler = NestedSampler(
            loglike, ptform, ndim, nlive=nlive, sample=sample, **dynesty_kwargs
        )
        sampler.run_nested(dlogz=dlogz, print_progress=verbose)
        results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()

    return {
        "param_names": list(JOINT_PARAM_NAMES),
        "percentiles": _weighted_percentiles(results.samples, weights),
        "log_evidence": float(results.logz[-1]),
        "log_evidence_err": float(results.logzerr[-1]),
        "samples": results.samples,
        "weights": weights,
        "alpha_bounds": tuple(alpha_bounds),
        "ncall": int(np.sum(results.ncall)),  # dynesty .ncall is per-iteration, sum for total
    }


def demo() -> None:
    """Self-check: the shared-tau likelihood must prefer the true alpha.

    Builds two synthetic single-band bursts (CHIME 0.6 / DSA 1.4 GHz) scattered
    with a common tau_1ghz, alpha_true, then verifies the joint log-likelihood
    (profiled crudely over the per-band amplitudes only) peaks at alpha_true and
    rejects a wrong alpha. No sampler -- a fast logic gate, not a fit.
    """
    rng = np.random.default_rng(0)
    tau_true, alpha_true = 1.0, 4.0
    truth = dict(c0=20.0, gamma=0.0, zeta=0.3, tau_1ghz=tau_true, alpha=alpha_true)

    def make(fmin, fmax, nch):
        freq = np.linspace(fmin, fmax, nch)
        time = np.arange(220) * 0.05
        m = FRBModel(time=time, freq=freq, data=np.zeros((nch, time.size)),
                     dm_init=0.0)
        p = FRBParams(t0=time.mean(), delta_dm=0.0, **truth)
        clean = m(p, "M3")
        noisy = clean + rng.normal(0, 0.05 * clean.max(), clean.shape)
        return FRBModel(time=time, freq=freq, data=noisy, dm_init=0.0), p

    mC, pC = make(0.40, 0.80, 16)
    mD, pD = make(1.31, 1.50, 16)
    ll = _JointLogLikelihood(mC, mD)

    def vec(alpha):
        # [tau, alpha | c0_C,t0_C,g_C,z_C,dd_C | c0_D,t0_D,g_D,z_D,dd_D]
        return np.array([tau_true, alpha,
                         pC.c0, pC.t0, pC.gamma, pC.zeta, 0.0,
                         pD.c0, pD.t0, pD.gamma, pD.zeta, 0.0])

    ll_true = ll(vec(alpha_true))
    ll_wrong = ll(vec(alpha_true + 1.5))
    assert ll_true > ll_wrong, f"true alpha not preferred: {ll_true} <= {ll_wrong}"
    # and the shared-tau model with a wrong tau is worse too
    bad = vec(alpha_true); bad[0] = tau_true * 5
    assert ll_true > ll(bad), "true tau not preferred"
    print(f"demo OK: ll(alpha={alpha_true})={ll_true:.0f} > "
          f"ll(alpha={alpha_true+1.5})={ll_wrong:.0f}")


if __name__ == "__main__":
    demo()
