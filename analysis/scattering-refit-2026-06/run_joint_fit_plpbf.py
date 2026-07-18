#!/usr/bin/env python
"""PL-PBF real-data joint fit driver: joint CHIME+DSA fit with the inner-scale power-law
PBF kernel (chromatic s_i), alpha TIED to beta. The third leg of the shape-model
comparison for casey + wilhelm (EMG production vs free-alpha wedge vs PL-PBF).

Reuses the EXACT production preparation (``run_joint_fit.prepare_joint`` -> mask-aware
S/N-driven resolution + robust common window) and the production shared-zeta prior spec,
inserting log10 s_i in [SI_PRIOR] after beta. beta prior kept at [3,4] (NOT widened); the
ONLY model change vs production is the PBF tail shape. Likelihood is
``JointLogLikelihoodSharedZetaPLPBF`` (upgrades the prepared models to FRBModelPLPBF so the
inner-scale kernel is used at the convolution seam; dispersion / smearing / gain-marginal
are byte-identical to production).

Writes a SEPARATE artifact (plpbf_<burst>_joint_fit.json + _samples.npz), never the
production table. Compare its log_evidence against the production and relaxed-alpha fits.

  python run_joint_fit_plpbf.py <burst> [nlive=400] [nproc=8]
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

from plpbf_loglike import JointLogLikelihoodSharedZetaPLPBF
from run_joint_fit import prepare_joint

# log10 s_i prior. Lower edge -1 = inner scale inside the core (-> pure exponential / EMG
# limit); upper edge 4 = inner scale far out (-> production PL-PBF, tail below noise). The
# fit pins s_i where the near tail (t/tau ~ 3-15) demands, or rails to an edge = one of the
# nested limits (flagged below). Matches plpbf_inject PL_LO/PL_HI on log10 s_i, +1 dex head.
SI_PRIOR = (-1.0, 4.0)


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

    print(f"[{burst}] PL-PBF: preparing CHIME + DSA models ...", flush=True)
    model_C, init_C, model_D, init_D = prepare_joint(cC, cD, burst, out_dir)
    print(
        f"[{burst}] CHIME init: tau={init_C.tau_1ghz:.3g} | DSA init: tau={init_D.tau_1ghz:.3g}",
        flush=True,
    )

    # Production shared-zeta spec (8-vector), then insert log10 s_i after beta.
    beta_bounds = default_joint_beta_bounds()  # (3.0, 4.0) -- unchanged, NOT widened
    spec = _joint_prior_spec_gain_shared_zeta(init_C, init_D, beta_bounds)
    assert spec[1][0] == "beta", f"unexpected spec layout: {[s[0] for s in spec]}"
    spec.insert(2, ("log10_s_i", tuple(SI_PRIOR), False))
    names = [s[0] for s in spec]
    assert names[:3] == ["tau_1ghz", "beta", "log10_s_i"], names

    ptform = _JointPriorTransform(spec)
    loglike = JointLogLikelihoodSharedZetaPLPBF(model_C, model_D)
    ndim = len(spec)
    print(
        f"[{burst}] ndim={ndim} names={names} nlive={nlive} nproc={nproc} "
        f"beta~U{beta_bounds} log10_s_i~U{SI_PRIOR} (alpha TIED)",
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

    si_m, si_lo, si_hi = med("log10_s_i")
    b_m, b_lo, b_hi = med("beta")
    t_m, t_lo, t_hi = med("tau_1ghz")

    summary = {
        "burst": burst,
        "variant": "plpbf_innerscale",
        "inner_scale": True,
        "alpha_tied": True,
        "log10_si_prior": list(SI_PRIOR),
        "beta_bounds": list(beta_bounds),
        "shared_zeta": True,
        "log10_s_i": {"median": si_m, "err_minus": si_lo, "err_plus": si_hi},
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
    out = f"{out_dir}/plpbf_{burst}_joint_fit.json"
    json.dump(summary, open(out, "w"), indent=2)

    np.savez_compressed(
        f"{out_dir}/plpbf_{burst}_joint_samples.npz",
        samples=results.samples,
        weights=weights,
        param_names=np.array(names, dtype=object),
        log10_si_prior=np.array(SI_PRIOR, dtype=float),
        beta_bounds=np.array(beta_bounds, dtype=float),
        freq_C=model_C.freq,
        freq_D=model_D.freq,
    )

    # Where did s_i land? Upper rail -> inner scale unconstrained (nests to production PL,
    # tail below noise); lower rail -> pure-exp / EMG limit; interior -> a measured inner scale.
    lo, hi = SI_PRIOR
    if si_m + 3 * si_hi >= hi:
        verdict = "s_i AT UPPER RAIL -> inner scale unconstrained (~production PL-PBF; tail below noise)"
    elif si_m - 3 * si_lo <= lo:
        verdict = "s_i AT LOWER RAIL -> pure-exp / EMG limit (no resolved power-law tail)"
    else:
        verdict = f"s_i INTERIOR -> resolved inner scale s_i(1GHz)~10^{si_m:.2f}={10 ** si_m:.1f}"
    print(
        f"\n[{burst}] PL-PBF  log10_s_i = {si_m:.3f} (+{si_hi:.3f}/-{si_lo:.3f})"
        f"   beta = {b_m:.3f} (+{b_hi:.3f}/-{b_lo:.3f})"
        f"   tau_1GHz = {t_m:.3g} ms   lnZ = {results.logz[-1]:.1f}",
        flush=True,
    )
    print(f"[{burst}] {verdict}", flush=True)
    print(f"[{burst}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
