#!/usr/bin/env python3
"""Chunked (checkpoint/resume) version of refit_runner for the 45-s sandbox
bash limit. Each invocation advances the sampler and checkpoints every 8 s;
call repeatedly until it prints DONE.

Usage: python3 refit_chunk.py <burst> [--nlive 160] [--nproc 4] [--dlogz 0.5]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/sessions/youthful-amazing-pasteur/mnt/outputs")
import refit_runner as R  # noqa: E402

from dynesty import NestedSampler  # noqa: E402
from scat_analysis.burstfit_joint import (  # noqa: E402
    _JointLogLikelihoodGainMulti,
    _append_derived_alpha_percentiles,
    _weighted_percentiles,
    alpha_from_beta,
)

RUNS = R.RUNS


def finalize(burst, s, names, rows, res, n_C, n_D, model_C, model_D, t0):
    suffix = R.suffix_for(burst)
    out = RUNS / "data" / "joint"
    out.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(res.samples)
    weights = np.exp(np.asarray(res.logwt) - float(res.logz[-1]))
    pct = _weighted_percentiles(samples, weights, names)
    pct = _append_derived_alpha_percentiles(pct, samples, weights, names)
    fit = {
        "burst": burst,
        "fit_note": s["note"] + " Sandbox chunked refit 2026-07-07 (bad-fit remediation).",
        "marginalize_gain": False,
        "marginalize_gain_gp": False,
        "shared_zeta": False,
        "beta": pct["beta"],
        "beta_bounds": [3.0, 4.0],
        "alpha": pct["alpha"],
        "tau_1ghz": pct["tau_1ghz"],
        "log_evidence": float(res.logz[-1]),
        "log_evidence_err": float(res.logzerr[-1]),
        "alpha_bounds": [alpha_from_beta(4.0), alpha_from_beta(3.0)],
        "components_C": n_C,
        "components_D": n_D,
        "component_windows": {name: list(bounds) for name, bounds, _ in rows},
        "gain_s2": None,
        "percentiles": {name: pct[name] for name in names + ("alpha",)},
        "ncall": int(np.sum(res.ncall)),
        "runtime_s": round(_time.time() - t0, 1),
    }
    fit_path = out / f"{burst}_joint_fit{suffix}.json"
    fit_path.write_text(json.dumps(fit, indent=2))
    np.savez_compressed(
        out / f"{burst}_joint_samples{suffix}.npz",
        samples=samples,
        weights=weights,
        param_names=np.asarray(names, dtype=object),
        alpha_bounds=np.asarray(fit["alpha_bounds"]),
        freq_C=np.asarray(model_C.freq),
        freq_D=np.asarray(model_D.freq),
    )
    print(f"DONE wrote {fit_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("burst", choices=sorted(R.SPEC))
    ap.add_argument("--nlive", type=int, default=160)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--dlogz", type=float, default=0.5)
    args = ap.parse_args()

    burst = args.burst
    s = R.SPEC[burst]
    n_C, n_D = len(s["C"]["comps"]), len(s["D"]["comps"])
    for band in ("chime", "dsa"):
        R.write_config(burst, band, s)
    model_C = R.prepare(burst, "chime", s)
    model_D = R.prepare(burst, "dsa", s)
    names, rows, ptform = R.build_spec(burst)
    loglike = _JointLogLikelihoodGainMulti(model_C, model_D, n_C=n_C, n_D=n_D, s2=None)

    # /tmp: sandbox-local, persists across bash calls, and (unlike the mounted
    # outputs dir) allows deletion for stale-checkpoint resets.
    ckpt = Path(f"/tmp/{burst}{R.suffix_for(burst)}_dynesty.save")
    t0 = _time.time()
    with mp.Pool(args.nproc) as pool:
        if ckpt.exists():
            sampler = NestedSampler.restore(str(ckpt), pool=pool)
            resume = True
            try:
                print(f"RESUME iter={int(sampler.it)} ncall={int(sampler.ncall)}", flush=True)
            except Exception:
                pass
        else:
            sampler = NestedSampler(
                loglike, ptform, ndim=len(names), nlive=args.nlive,
                sample="rwalk", queue_size=args.nproc, pool=pool,
            )
            resume = False
        # No KeyboardInterrupt handling: if the outer `timeout` kills us the
        # checkpoint (every 8 s) carries the state; finalize NEVER runs on a
        # partial chunk.
        sampler.run_nested(
            dlogz=args.dlogz,
            print_progress=False,
            checkpoint_file=str(ckpt),
            checkpoint_every=8,
            resume=resume,
        )
        res = sampler.results
        print(f"CONVERGED iter={int(res.niter)} logz={float(res.logz[-1]):.1f}", flush=True)
    # run_nested returns only when the dlogz target is met -> converged.
    finalize(burst, s, names, rows, res, n_C, n_D, model_C, model_D, t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
