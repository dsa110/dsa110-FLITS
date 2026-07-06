#!/usr/bin/env python
"""Sim-injection gate for the beta-coherent thin-screen campaign (Phase 3).

Three synthetic two-band injections routed through the PRODUCTION joint path
(fit_joint_scattering, shared-zeta gain-marginal, campaign-default beta prior),
not a private likelihood, so the gate exercises exactly the code the fleet
will run (plan: docs/rse/specs/plan-beta-coherent-thin-screen-campaign.md):

  interior-3.7  beta_true=3.70  -> recover within 3 sigma, un-railed
  interior-3.3  beta_true=3.30  -> recover within 3 sigma, un-railed
                (heavier t^-1.65 tail; wider CHIME window so the tail shape,
                not its truncation, drives beta -- run_beta_poc.py rationale)
  rail-4.0      beta_true=4.00  -> the forward model dispatches the analytic
                exponential member; the posterior must RAIL at the beta=4
                bound (ADR-0004: median within 3 sigma of the prior edge)

Injection physics (grids, gains, noise) is reused from the freya POC
(analysis/beta_poc/run_beta_poc.py) so truth construction stays single-source.

  conda run -n flits python analysis/beta_campaign/sim_gate.py [--nlive N]

Exit 0 iff all three verdicts are correct; the fleet must not launch otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scattering.scat_analysis.burstfit_init import data_driven_initial_guess  # noqa: E402
from scattering.scat_analysis.burstfit_joint import fit_joint_scattering  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_beta_poc", REPO / "analysis" / "beta_poc" / "run_beta_poc.py"
)
_poc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_poc)

CASES = [
    # (name, beta_true, tau_1ghz_ms, chime_t_max_ms, expect_rail)
    ("interior-3.7", 3.70, 0.05, 32.0, False),
    ("interior-3.3", 3.30, 0.03, 48.0, False),
    ("rail-4.0", 4.00, 0.05, 32.0, True),
]
BETA_BOUNDS = (3.0, 4.0)  # campaign default (alpha in [4, 6])


def _make_bands(beta: float, tau: float, t_max_C: float, seed: int):
    rng = np.random.default_rng(seed)
    n_time_C = int(448 * t_max_C / 32.0)  # keep the POC's dt when widening the window
    m_C0 = _poc._build_band(_poc.CHIME, n_freq=48, t_max=t_max_C, n_time=n_time_C)
    m_D0 = _poc._build_band(_poc.DSA, n_freq=48, t_max=6.0, n_time=320)
    m_C = _poc._inject(m_C0, tau, beta, _poc.ZETA1_TRUE, _poc.X_ZETA_TRUE, _poc.T0_C_TRUE, rng)
    m_D = _poc._inject(m_D0, tau, beta, _poc.ZETA1_TRUE, _poc.X_ZETA_TRUE, _poc.T0_D_TRUE, rng)
    return m_C, m_D


def _init_for(m):
    return data_driven_initial_guess(
        data=m.data, freq=m.freq, time=m.time, dm=0.0, verbose=False
    ).params


def _railed(med: float, err_minus: float, err_plus: float) -> bool:
    lo, hi = BETA_BOUNDS
    return (med - 3.0 * err_minus <= lo) or (med + 3.0 * err_plus >= hi)


def run_case(name, beta_true, tau, t_max_C, expect_rail, nlive, nproc, seed, maxcall):
    m_C, m_D = _make_bands(beta_true, tau, t_max_C, seed)
    res = fit_joint_scattering(
        model_C=m_C,
        init_C=_init_for(m_C),
        model_D=m_D,
        init_D=_init_for(m_D),
        beta_bounds=BETA_BOUNDS,
        nlive=nlive,
        nproc=nproc,
        shared_zeta=True,
        verbose=False,
        rstate=np.random.default_rng(seed + 1),
        # POC-proven budget (run_beta_poc.py defaults): bounds the gate's
        # wall-time; recovery at this scale was 0.3% on beta
        maxcall=maxcall,
    )
    b = res["percentiles"]["beta"]
    med, em, ep = b["median"], b["err_minus"], b["err_plus"]
    railed = _railed(med, em, ep)
    if expect_rail:
        ok = railed and (BETA_BOUNDS[1] - med) < 0.15
        detail = f"railed={railed}, gap-to-4={BETA_BOUNDS[1] - med:.3f}"
    else:
        # error side toward the truth, floored so a razor-thin posterior
        # cannot fail on numerical jitter
        sig = max(em if med > beta_true else ep, 0.01)
        ok = (abs(med - beta_true) <= 3.0 * sig) and not railed
        detail = f"|med-true|={abs(med - beta_true):.4f}, 3sig={3 * sig:.4f}, railed={railed}"
    return {
        "case": name,
        "beta_true": beta_true,
        "beta_median": med,
        "err_minus": em,
        "err_plus": ep,
        "alpha_derived": res["percentiles"]["alpha"]["median"],
        "log_evidence": res["log_evidence"],
        "ncall": res["ncall"],
        "expect_rail": expect_rail,
        "railed": railed,
        "ok": bool(ok),
        "detail": detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlive", type=int, default=150)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--maxcall", type=int, default=400_000)
    args = ap.parse_args()

    results = []
    for i, (name, beta_true, tau, t_max_C, expect_rail) in enumerate(CASES):
        print(f"[sim-gate] {name}: injecting beta={beta_true}, fitting ...", flush=True)
        r = run_case(
            name,
            beta_true,
            tau,
            t_max_C,
            expect_rail,
            args.nlive,
            args.nproc,
            args.seed + 100 * i,
            args.maxcall,
        )
        results.append(r)
        print(
            f"[sim-gate] {name}: beta={r['beta_median']:.3f} "
            f"(+{r['err_plus']:.3f}/-{r['err_minus']:.3f}) "
            f"{'OK' if r['ok'] else 'FAIL'} ({r['detail']})",
            flush=True,
        )

    out = Path(__file__).parent / "sim_gate_results.json"
    out.write_text(
        json.dumps(
            {
                "beta_bounds": list(BETA_BOUNDS),
                "nlive": args.nlive,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        )
    )
    n_ok = sum(r["ok"] for r in results)
    print(f"[sim-gate] {n_ok}/{len(results)} correct -> {out}", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
