#!/usr/bin/env python
"""Relaxed-alpha A/B driver: joint CHIME+DSA fit with alpha sampled FREE, decoupled
from the beta-native thin-screen tie alpha=2 beta/(beta-2).

Diagnostic only (casey + wilhelm). Reuses the EXACT production preparation
(``run_joint_fit.prepare_joint`` -> mask-aware S/N-driven resolution + robust common
window) and the production shared-zeta prior spec, inserting a free alpha in [2,6]
after beta (beta prior kept at [3,4], NOT widened). The likelihood is
``JointLogLikelihoodSharedZetaFreeAlpha`` -- byte-identical to the production
shared-zeta gain-marginal path except beta sets only the PBF shape while the sampled
alpha sets the tau(nu) exponent (see relaxalpha_loglike for the physics rationale).

Writes a SEPARATE artifact (ab_<burst>_relaxalpha_joint_fit.json + _samples.npz), never
the production table.

  python run_joint_fit_relaxalpha.py <burst> [nlive=400] [nproc=8]
"""

import json
import os
import sys

REPO = os.environ.get("FLITS_REPO", "/home/ubuntu/worktrees/joint-tf-fits")
RUNS = os.environ.get("FLITS_RUNS", "/home/ubuntu/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scat_analysis.burstfit_joint import (
    _JointPriorTransform,
    _joint_prior_spec_gain_shared_zeta,
    _weighted_percentiles,
)
from scat_analysis.turbulence import default_joint_beta_bounds
from dynesty import NestedSampler

from relaxalpha_loglike import JointLogLikelihoodSharedZetaFreeAlpha
from run_joint_fit import prepare_joint

ALPHA_PRIOR = (2.0, 6.0)


def main():
    burst = sys.argv[1]
    nlive = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    cfg_dir = f"{RUNS}/configs"
    out_dir = f"{RUNS}/data/joint"
    os.makedirs(out_dir, exist_ok=True)
    cC = f"{cfg_dir}/{burst}_chime_run.yaml"
    cD = f"{cfg_dir}/{burst}_dsa_run.yaml"
    for c in (cC, cD):
        if not os.path.exists(c):
            sys.exit(f"missing config: {c}")

    print(f"[{burst}] RELAX-ALPHA: preparing CHIME + DSA models ...", flush=True)
    model_C, init_C, model_D, init_D = prepare_joint(cC, cD, burst, out_dir)
    print(
        f"[{burst}] CHIME init: tau={init_C.tau_1ghz:.3g} | DSA init: tau={init_D.tau_1ghz:.3g}",
        flush=True,
    )

    # Production shared-zeta spec (8-vector), then insert a FREE alpha after beta.
    beta_bounds = default_joint_beta_bounds()  # (3.0, 4.0) -- unchanged, NOT widened
    spec = _joint_prior_spec_gain_shared_zeta(init_C, init_D, beta_bounds)
    assert spec[1][0] == "beta", f"unexpected spec layout: {[s[0] for s in spec]}"
    spec.insert(2, ("alpha", tuple(ALPHA_PRIOR), False))
    names = [s[0] for s in spec]
    assert names[:3] == ["tau_1ghz", "beta", "alpha"], names

    ptform = _JointPriorTransform(spec)
    loglike = JointLogLikelihoodSharedZetaFreeAlpha(model_C, model_D)
    ndim = len(spec)
    print(
        f"[{burst}] ndim={ndim} names={names} nlive={nlive} nproc={nproc} "
        f"beta~U{beta_bounds} alpha~U{ALPHA_PRIOR} (FREE)",
        flush=True,
    )

    run_kwargs = {"dlogz": 0.5, "print_progress": True}
    if nproc and nproc > 1:
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        from dynesty import pool as dypool

        with dypool.Pool(int(nproc), loglike, ptform) as pool:
            sampler = NestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                nlive=nlive,
                sample="rwalk",
                pool=pool,
                queue_size=int(nproc),
            )
            sampler.run_nested(**run_kwargs)
            results = sampler.results
    else:
        sampler = NestedSampler(loglike, ptform, ndim, nlive=nlive, sample="rwalk")
        sampler.run_nested(**run_kwargs)
        results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()
    pct = _weighted_percentiles(results.samples, weights, tuple(names))

    def med(n):
        d = pct[n]
        return d["median"], d["err_minus"], d["err_plus"]

    a_m, a_lo, a_hi = med("alpha")
    b_m, b_lo, b_hi = med("beta")
    t_m, t_lo, t_hi = med("tau_1ghz")

    summary = {
        "burst": burst,
        "variant": "relax_alpha",
        "free_alpha": True,
        "alpha_prior": list(ALPHA_PRIOR),
        "beta_bounds": list(beta_bounds),
        "shared_zeta": True,
        "alpha": {"median": a_m, "err_minus": a_lo, "err_plus": a_hi},
        "beta": {"median": b_m, "err_minus": b_lo, "err_plus": b_hi},
        "tau_1ghz": {"median": t_m, "err_minus": t_lo, "err_plus": t_hi},
        "log_evidence": float(results.logz[-1]),
        "log_evidence_err": float(results.logzerr[-1]),
        "components_C": 1,
        "components_D": 1,
        "percentiles": pct,
        "ncall": int(np.sum(results.ncall)),
        "param_names": names,
    }
    out = f"{out_dir}/ab_{burst}_relaxalpha_joint_fit.json"
    json.dump(summary, open(out, "w"), indent=2)

    np.savez_compressed(
        f"{out_dir}/ab_{burst}_relaxalpha_joint_samples.npz",
        samples=results.samples,
        weights=weights,
        param_names=np.array(names, dtype=object),
        alpha_prior=np.array(ALPHA_PRIOR, dtype=float),
        beta_bounds=np.array(beta_bounds, dtype=float),
        freq_C=model_C.freq,
        freq_D=model_D.freq,
    )

    # Flag whether the FREE alpha posterior sits at/above the nu^-4 line or below.
    lo95 = float(np.percentile(_resample(results.samples[:, 2], weights), 5))
    verdict = (
        "alpha>=4 data-driven (nu^-4 supported)"
        if lo95 >= 3.95
        else ("alpha<4: PBF-shape/bounded-screen signature" if a_m < 3.98 else "straddles 4")
    )
    edge_a = " [ALPHA AT PRIOR EDGE]" if (a_m - 3 * a_lo <= ALPHA_PRIOR[0] or a_m + 3 * a_hi >= ALPHA_PRIOR[1]) else ""
    edge_b = " [BETA AT PRIOR EDGE]" if (b_m - 3 * b_lo <= beta_bounds[0] or b_m + 3 * b_hi >= beta_bounds[1]) else ""
    print(
        f"\n[{burst}] RELAX-ALPHA  alpha = {a_m:.3f} (+{a_hi:.3f}/-{a_lo:.3f}){edge_a}"
        f"   beta = {b_m:.3f} (+{b_hi:.3f}/-{b_lo:.3f}){edge_b}"
        f"   tau_1GHz = {t_m:.3g} ms   lnZ = {results.logz[-1]:.1f}",
        flush=True,
    )
    print(f"[{burst}] alpha 5th pct = {lo95:.3f} -> {verdict}", flush=True)
    print(f"[{burst}] wrote {out}", flush=True)


def _resample(col, weights, n=40000):
    """Weighted bootstrap of one posterior column for a robust percentile."""
    idx = np.random.default_rng(0).choice(len(col), size=n, p=weights)
    return col[idx]


if __name__ == "__main__":
    main()
