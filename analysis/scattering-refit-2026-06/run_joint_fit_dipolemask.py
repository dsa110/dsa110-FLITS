#!/usr/bin/env python
"""Dipole-mask free-alpha wedge driver (task #13 discriminant).

Re-runs the relaxed-alpha wedge (free alpha in [2,6], beta in [3,4]) on a single
burst after MASKING the frequency-coherent peak dipole in ONE band. The question:
is the sub-4 wedge (casey alpha~2.43, wilhelm alpha~2.57) driven by the sharp
peak-shape systematic (the +/-26..32 sigma single-bin dipole at the pulse peak) or
by a distributed chromatic tail?

  * alpha -> ~4 after masking  => the wedge was PEAK-SHAPE-DRIVEN (dipole is a
    non-scattering, frequency-coherent systematic; the underlying scattering index
    is consistent with the nu^-4 thin-screen line).
  * alpha stays < 4            => the sub-4 index is DISTRIBUTED chromatic scaling,
    not localized at the peak; the anomaly survives the dipole excision.

Masking mechanism (driver-only, NO canonical edit, exact):
  The gain-marginal per-band likelihood (FRBModel.log_likelihood_gain_marginal)
  forms S_dd=sum_t d^2, S_dk=sum_t d K, S_kk=sum_t K^2 over time, with K=self(...)
  the unit kernel. Scaling BOTH the data column d[:,j] and the model output K[:,j]
  by sqrt(w_j) multiplies every one of those three sums by exactly w_j at bin j:
      w_j = 0        -> bin excluded from all sums (HARD mask, exact)
      w_j = 1/f^2    -> per-bin variance inflated by f^2 (DOWN-WEIGHT, sigma->f*sigma)
  Implemented as a per-INSTANCE gated wrapper on FRBModel.__call__ (dunder is looked
  up on the type, so the wrapper lives on the class but only fires for the tagged
  target-band instance via getattr(self,'_dipole_sqrtw')). The non-target band and
  every other caller are byte-identical. Per-channel noise_std and valid are
  untouched. The absolute lnZ carries a different data normalization once bins are
  removed, so masked-vs-unmasked lnZ is NOT a Bayes factor -- we report the alpha
  posterior (shape-driven, normalization-independent), which is the discriminant.

Writes a SEPARATE artifact:
  ab_<burst>_dipolemask_<band>_<mode>_joint_fit.json  (+ _samples.npz)

  python run_joint_fit_dipolemask.py <burst> <band C|D> <mode hard|soft> \
      [halfwin=2] [inflate=10.0] [nlive=400] [nproc=8] [--smoke]
"""

import json
import os
import sys

REPO = os.environ.get("FLITS_REPO", "/home/ubuntu/worktrees/joint-tf-fits")
RUNS = os.environ.get("FLITS_RUNS", "/home/ubuntu/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scat_analysis.burstfit as _bf
from scat_analysis.burstfit_joint import (
    _JointLogLikelihoodGainSharedZeta,
    _JointPriorTransform,
    _joint_prior_spec_gain_shared_zeta,
    _weighted_percentiles,
)
from scat_analysis.turbulence import default_joint_beta_bounds
from dynesty import NestedSampler

from relaxalpha_loglike import JointLogLikelihoodSharedZetaFreeAlpha
from run_joint_fit import prepare_joint

ALPHA_PRIOR = (2.0, 6.0)

# ---------------------------------------------------------------------------
# Per-instance gated time-weight wrapper on FRBModel.__call__.
# A no-op for any instance lacking `_dipole_sqrtw` (so prepare_joint, the CHIME/
# DSA non-target band, and all other callers are unaffected).
# ---------------------------------------------------------------------------
_ORIG_CALL = _bf.FRBModel.__call__


def _dipole_masked_call(self, p, model_key="M3", freq_subset=None):
    out = _ORIG_CALL(self, p, model_key, freq_subset)
    sw = getattr(self, "_dipole_sqrtw", None)
    if sw is not None:
        out = out * sw[None, :]  # (F, T) * (1, T): scale target time bins by sqrt(w)
    return out


_bf.FRBModel.__call__ = _dipole_masked_call


def _apply_dipole_mask(model, halfwin, mode, inflate):
    """Tag `model` with a sqrt(w) time-weight zeroing/deweighting the peak dipole.

    Peak = argmax of the band-summed on-pulse profile over valid channels (found on
    the ORIGINAL data, before scaling). Masks peak +/- halfwin bins. Returns a diag
    dict. Mutates model.data (copy) and sets model._dipole_sqrtw.
    """
    valid = model.valid
    prof = np.nansum(np.asarray(model.data)[valid], axis=0)  # (T,)
    peak = int(np.nanargmax(prof))
    T = prof.size
    lo = max(0, peak - halfwin)
    hi = min(T, peak + halfwin + 1)
    bins = np.arange(lo, hi)

    if mode == "hard":
        wv = 0.0
    elif mode == "soft":
        wv = 1.0 / float(inflate) ** 2  # per-bin variance inflated by inflate^2
    else:
        raise SystemExit(f"unknown mode {mode!r} (hard|soft)")

    sqrt_w = np.ones(T, dtype=float)
    sqrt_w[bins] = np.sqrt(wv)

    data = np.array(model.data, dtype=float, copy=True)
    data[:, bins] *= np.sqrt(wv)
    model.data = data
    model._dipole_sqrtw = sqrt_w

    tot = float(np.nansum(prof))
    frac = float(np.nansum(prof[bins])) / tot if tot > 0 else float("nan")
    dt_ms = float(model.time[1] - model.time[0])
    return {
        "peak_bin": peak,
        "masked_bins": [int(b) for b in bins],
        "n_masked": int(bins.size),
        "mask_span_ms": float(bins.size * dt_ms),
        "peak_time_ms": float(model.time[peak]),
        "signal_frac_masked": frac,
        "w_value": float(wv),
        "mode": mode,
        "inflate": float(inflate) if mode == "soft" else None,
        "halfwin": int(halfwin),
    }


def _wq(x, w, q):
    """Weighted quantile (q in [0,1])."""
    order = np.argsort(x)
    xs = np.asarray(x)[order]
    cw = np.cumsum(np.asarray(w)[order])
    cw /= cw[-1]
    return float(np.interp(q, cw, xs))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    burst = args[0]
    band = args[1].upper()  # C or D
    mode = args[2].lower()  # hard or soft
    halfwin = int(args[3]) if len(args) > 3 else 2
    inflate = float(args[4]) if len(args) > 4 else 10.0
    nlive = int(args[5]) if len(args) > 5 else 400
    nproc = int(args[6]) if len(args) > 6 else 8
    smoke = "--smoke" in flags
    tied = "--tied" in flags  # production alpha-tied EMG partner (for the wedge delta-lnZ)
    if band not in ("C", "D"):
        raise SystemExit("band must be C or D")

    cfg_dir = f"{RUNS}/configs"
    out_dir = f"{RUNS}/data/joint"
    os.makedirs(out_dir, exist_ok=True)
    cC = f"{cfg_dir}/{burst}_chime_run.yaml"
    cD = f"{cfg_dir}/{burst}_dsa_run.yaml"
    for c in (cC, cD):
        if not os.path.exists(c):
            sys.exit(f"missing config: {c}")

    tag = f"{band}_{mode}" + ("_tied" if tied else "")
    print(f"[{burst}] DIPOLE-MASK {tag}: preparing CHIME + DSA models ...", flush=True)
    model_C, init_C, model_D, init_D = prepare_joint(cC, cD, burst, out_dir)
    target = model_C if band == "C" else model_D

    mask_diag = _apply_dipole_mask(target, halfwin, mode, inflate)
    print(
        f"[{burst}] mask band={band} mode={mode} peak_bin={mask_diag['peak_bin']} "
        f"t_peak={mask_diag['peak_time_ms']:.3f}ms bins={mask_diag['masked_bins']} "
        f"span={mask_diag['mask_span_ms']:.3f}ms "
        f"signal_frac_masked={mask_diag['signal_frac_masked']:.3f} w={mask_diag['w_value']:g}",
        flush=True,
    )

    beta_bounds = default_joint_beta_bounds()
    spec = _joint_prior_spec_gain_shared_zeta(init_C, init_D, beta_bounds)
    assert spec[1][0] == "beta", f"unexpected spec layout: {[s[0] for s in spec]}"
    if tied:
        # Production alpha-tied EMG on the SAME masked data. Same 8-vector spec,
        # same sampler config -> a valid shared-data partner: lnZ(free) - lnZ(tied)
        # on masked data IS the wedge Bayes factor recomputed under excision.
        loglike = _JointLogLikelihoodGainSharedZeta(model_C, model_D)
        names = [s[0] for s in spec]
        assert names[:2] == ["tau_1ghz", "beta"], names
    else:
        spec.insert(2, ("alpha", tuple(ALPHA_PRIOR), False))
        loglike = JointLogLikelihoodSharedZetaFreeAlpha(model_C, model_D)
        names = [s[0] for s in spec]
        assert names[:3] == ["tau_1ghz", "beta", "alpha"], names
    ptform = _JointPriorTransform(spec)
    ndim = len(spec)

    if smoke:
        theta = ptform(np.full(ndim, 0.5))
        ll_masked = float(loglike(theta))
        # Verify the hard mask actually zeros the target columns in BOTH d and K.
        p_chk = init_C if band == "C" else init_D  # valid FRBParams from prepare_joint
        K = target(p_chk, "M3", freq_subset=target.valid)
        bins = mask_diag["masked_bins"]
        kmax_in = float(np.max(np.abs(K[:, bins]))) if bins else 0.0
        dmax_in = float(np.max(np.abs(np.asarray(target.data)[target.valid][:, bins]))) if bins else 0.0
        # Reference: remove the tag and re-eval to confirm no-op equivalence path.
        saved = target._dipole_sqrtw
        saved_data = target.data
        del target._dipole_sqrtw
        K_ref = target(p_chk, "M3", freq_subset=target.valid)
        kmax_ref_in = float(np.max(np.abs(K_ref[:, bins]))) if bins else 0.0
        target._dipole_sqrtw = saved
        target.data = saved_data
        print(
            f"[SMOKE {burst} {tag}] ndim={ndim} ll_masked={ll_masked:.3f}\n"
            f"  masked-K max|.| in bins = {kmax_in:.3e} (mode={mode}); "
            f"UNmasked-K max|.| in same bins = {kmax_ref_in:.3e}\n"
            f"  masked-data max|.| in bins = {dmax_in:.3e}\n"
            f"  => hard-mask columns zero? "
            f"{'PASS' if (mode!='hard' or (kmax_in < 1e-12 and dmax_in < 1e-12)) else 'FAIL'} ; "
            f"wrapper active (unmasked-K nonzero there)? "
            f"{'PASS' if kmax_ref_in > 0 else 'FAIL'}",
            flush=True,
        )
        return

    print(
        f"[{burst}] ndim={ndim} names={names} nlive={nlive} nproc={nproc} "
        f"alpha~U{ALPHA_PRIOR} beta~U{beta_bounds} (dipole-mask {tag})",
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
                pool.loglike, pool.prior_transform, ndim,
                nlive=nlive, sample="rwalk", pool=pool, queue_size=int(nproc),
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

    b_m, b_lo, b_hi = med("beta")
    t_m, t_lo, t_hi = med("tau_1ghz")
    if tied:
        # alpha is the derived thin-screen tie (clamped at 4), not a sampled param.
        a_m = min(2.0 * b_m / (b_m - 2.0), 4.0) if b_m > 2.0 else 4.0
        a_lo = a_hi = 0.0
        a_lo95 = a_hi95 = a_m
        alpha_block = {"median": a_m, "err_minus": 0.0, "err_plus": 0.0,
                       "derived_tied": True}
    else:
        a_m, a_lo, a_hi = med("alpha")
        ai = names.index("alpha")
        a_lo95 = _wq(results.samples[:, ai], weights, 0.05)
        a_hi95 = _wq(results.samples[:, ai], weights, 0.95)
        alpha_block = {"median": a_m, "err_minus": a_lo, "err_plus": a_hi,
                       "q05": a_lo95, "q95": a_hi95}

    summary = {
        "burst": burst,
        "variant": "dipole_mask_tied" if tied else "dipole_mask",
        "tied_alpha": tied,
        "band_masked": band,
        "mask_mode": mode,
        "mask": mask_diag,
        "free_alpha": not tied,
        "alpha_prior": list(ALPHA_PRIOR),
        "beta_bounds": list(beta_bounds),
        "shared_zeta": True,
        "alpha": alpha_block,
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
    out = f"{out_dir}/ab_{burst}_dipolemask_{tag}_joint_fit.json"
    json.dump(summary, open(out, "w"), indent=2)
    np.savez_compressed(
        f"{out_dir}/ab_{burst}_dipolemask_{tag}_joint_samples.npz",
        samples=results.samples, weights=weights,
        param_names=np.array(names, dtype=object),
        alpha_prior=np.array(ALPHA_PRIOR, dtype=float),
        beta_bounds=np.array(beta_bounds, dtype=float),
        freq_C=model_C.freq, freq_D=model_D.freq,
    )

    edge_b = " [BETA AT PRIOR EDGE]" if (b_m - 3 * b_lo <= beta_bounds[0] or b_m + 3 * b_hi >= beta_bounds[1]) else ""
    if tied:
        print(
            f"[{burst}] DONE dipole-mask {tag} (TIED-alpha partner): alpha_tied={a_m:.3f} "
            f"beta={b_m:.3f}{edge_b}  tau={t_m:.4g}  lnZ={results.logz[-1]:.2f}\n"
            f"  (wedge delta-lnZ = lnZ(free,masked) - THIS; computed in postprocess)\n  wrote {out}",
            flush=True,
        )
    else:
        edge_a = " [ALPHA AT PRIOR EDGE]" if (a_m - 3 * a_lo <= ALPHA_PRIOR[0] or a_m + 3 * a_hi >= ALPHA_PRIOR[1]) else ""
        verdict = (
            "alpha->4: wedge was PEAK-SHAPE-DRIVEN (dipole systematic)"
            if a_lo95 >= 3.90
            else ("alpha stays <4: DISTRIBUTED chromatic (survives dipole excision)"
                  if a_m < 3.90 else "straddles 4")
        )
        print(
            f"[{burst}] DONE dipole-mask {tag}: alpha={a_m:.3f} (-{a_lo:.3f}/+{a_hi:.3f}) "
            f"[q05={a_lo95:.3f} q95={a_hi95:.3f}]{edge_a}  beta={b_m:.3f}{edge_b}  "
            f"tau={t_m:.4g}  lnZ={results.logz[-1]:.2f}\n  VERDICT: {verdict}\n  wrote {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
