#!/usr/bin/env python
"""Likelihood equivalence check: Route A vs Route B at identical theta (issue #103).

Route A is the POC's BetaCoupledLogL (analysis/beta_poc/run_beta_poc.py); Route B
is the production driver's _JointLogLikelihoodGainSharedZeta
(scattering/scat_analysis/burstfit_joint.py). Both take the identical 8-vector

    theta = [tau_1ghz, beta, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D]

and wrap the SAME log_likelihood_gain_marginal kernel, so this check verifies
the independent theta-packing / shared-beta application / per-band parameter
wiring -- NOT the kernel itself (owned by the Step 1 analytic audit and the
Step 2 injection-recovery). Totals are compared through each route's own
__call__ (whose theta-unpacking IS the wiring under test); per-band values via
each route's band function are recorded as the diagnostic breakdown. Agreement
is asserted within RTOL = 1e-12 -- identical float ops in identical order, so
the measured difference is expected (and unit-asserted) to be exactly 0.0;
disagreement is a hard failure (nonzero exit + per-band, per-theta report).
The decisive, cheap, deterministic gate before the expensive fit (#104).

The theta grid: all 2^8 prior-bound corners + the center of the POC's sampled
prior box (through its own _ptform_factory), the exp-era-suggested point
(tau_1ghz = 0.119 ms, beta = 3.70 -- a deprecated-fit *suggestion*, not truth),
and beta in {3.99, 4.0}, where both routes must dispatch the BETA_EXP_EPS
exponential-PBF branch identically (ADR-0006; branch point 4.0 - 0.02).

CLI: run on freya's real prepared CHIME+DSA band models (#99 run-configs via
the joint driver's own prepare(), reused through the POC's _prepare_real_bands)
and record the verdict artifact:

    conda run -n flits python analysis/scattering-refit-2026-06/likelihood_equivalence.py
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scattering.scat_analysis.burstfit_joint import (  # noqa: E402
    _JointLogLikelihoodGainSharedZeta,
)

RTOL = 1e-12
THETA_NAMES = (
    "tau_1ghz",
    "beta",
    "zeta_1ghz",
    "x_zeta",
    "t0_C",
    "delta_dm_C",
    "t0_D",
    "delta_dm_D",
)


@functools.lru_cache(maxsize=1)
def _poc():
    """Route A lives in a script, not a package -- load it the way the repo's
    other analysis modules load the local-runs driver."""
    path = REPO / "analysis" / "beta_poc" / "run_beta_poc.py"
    spec = importlib.util.spec_from_file_location("run_beta_poc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel_diff(a: float, b: float) -> float:
    """|a-b| / max(|a|,|b|, 1): symmetric, denominator floored at 1 so values
    near zero compare absolutely; exactly 0.0 on bitwise agreement; NaN
    propagates (adjudicate fails closed on it)."""
    d = abs(a - b)
    return 0.0 if d == 0.0 else d / max(abs(a), abs(b), 1.0)


def theta_grid(t0_C: float, t0_D: float) -> np.ndarray:
    """(260, 8) grid: 256 prior-box corners + center via the POC's own
    _ptform_factory, the exp-era-suggested point, and the two exp-branch betas.
    zeta/x_zeta at the special points reuse the POC's synthetic-truth values
    (near the published free-alpha shared-zeta posterior)."""
    poc = _poc()
    ptform = poc._ptform_factory(t0_C, t0_D)
    pts = [ptform(np.array(bits, dtype=float)) for bits in itertools.product((0.0, 1.0), repeat=8)]
    pts.append(ptform(np.full(8, 0.5)))
    for beta in (3.70, 3.99, 4.0):
        pts.append(np.array([0.119, beta, poc.ZETA1_TRUE, poc.X_ZETA_TRUE, t0_C, 0.0, t0_D, 0.0]))
    return np.array(pts)


def evaluate(m_C, m_D, thetas) -> list[dict]:
    """Per theta: both routes' totals through their own __call__, plus the
    per-band breakdown via each route's band function."""
    poc = _poc()
    route_a = poc.BetaCoupledLogL(m_C, m_D)
    route_b = _JointLogLikelihoodGainSharedZeta(m_C, m_D)
    records = []
    for th in np.asarray(thetas, dtype=float):
        tau, beta, z1, x = (float(th[i]) for i in range(4))
        bands = {}
        for band, m, t0, ddm in (
            ("C", m_C, float(th[4]), float(th[5])),
            ("D", m_D, float(th[6]), float(th[7])),
        ):
            a = float(poc._band_ll(m, tau, beta, z1, x, t0, ddm))
            b = float(route_b._band_ll(m, tau, beta, z1, x, t0, ddm))
            bands[band] = {"route_A": a, "route_B": b, "rel_diff": rel_diff(a, b)}
        a_tot, b_tot = float(route_a(th)), float(route_b(th))
        records.append(
            {
                "theta": {n: float(v) for n, v in zip(THETA_NAMES, th, strict=True)},
                "route_A": a_tot,
                "route_B": b_tot,
                "rel_diff": rel_diff(a_tot, b_tot),
                "bands": bands,
            }
        )
    return records


def adjudicate(records: list[dict], rtol: float = RTOL) -> dict:
    """Hard verdict: every total AND every per-band pair within rtol; any NaN
    fails closed. Diffs are RECOMPUTED here from the stored route_A/route_B
    values -- the verdict never trusts a record's precomputed rel_diff (which
    is informational, for the artifact). Failures carry the full per-band,
    per-theta breakdown."""

    def _diffs(r: dict) -> list[float]:
        return [rel_diff(r["route_A"], r["route_B"])] + [
            rel_diff(b["route_A"], b["route_B"]) for b in r["bands"].values()
        ]

    def _ok(r: dict) -> bool:
        return all(np.isfinite(d) and d <= rtol for d in _diffs(r))

    failures = [{"index": i, **r} for i, r in enumerate(records) if not _ok(r)]
    diffs = [d for r in records for d in _diffs(r)]
    return {
        "passes": not failures,
        "rtol": float(rtol),
        "n_theta": len(records),
        "max_rel_diff": float(np.max(diffs)) if diffs else 0.0,
        "failures": failures,
    }


def run_check(rtol: float = RTOL) -> dict:
    """The freya real-model equivalence artifact: prepared bands via the POC's
    _prepare_real_bands (the joint driver's own prepare(); t0 prior centers
    from the RAW data-driven guess, never the post-#98 MLE refine -- it rails
    beta and drags t0, #101)."""
    poc = _poc()
    m_C, m_D, t0_C, t0_D = poc._prepare_real_bands()
    records = evaluate(m_C, m_D, theta_grid(t0_C, t0_D))
    return {
        "issue": "#103",
        "burst": "freya",
        "routes": {
            "A": "analysis/beta_poc/run_beta_poc.py::BetaCoupledLogL",
            "B": "scattering/scat_analysis/burstfit_joint.py::_JointLogLikelihoodGainSharedZeta",
        },
        "scope_note": (
            "Both routes wrap the same log_likelihood_gain_marginal kernel: this "
            "verifies theta-packing / shared-beta application / per-band wiring, "
            "not the kernel itself (Step 1 analytic audit + Step 2 "
            "injection-recovery own that)."
        ),
        "epistemic_note": (
            "The tau_1ghz = 0.119 ms / beta = 3.70 grid point is the deprecated "
            "exp-era fit's *suggestion* for freya -- a stress-test scale, not "
            "ground truth; whether it survives is what #104-#106 test."
        ),
        "bands": {
            "chime": {"shape": list(m_C.data.shape), "t0_raw_init_ms": float(t0_C)},
            "dsa": {"shape": list(m_D.data.shape), "t0_raw_init_ms": float(t0_D)},
        },
        "theta_names": list(THETA_NAMES),
        "verdict": adjudicate(records, rtol),
        "records": records,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rtol", type=float, default=RTOL)
    ap.add_argument(
        "--out",
        default=None,
        help="artifact path (default local_runs/freya_likelihood_equivalence.json)",
    )
    args = ap.parse_args(argv)

    artifact = run_check(args.rtol)
    out = (
        Path(args.out)
        if args.out
        else (
            REPO
            / "analysis"
            / "scattering-refit-2026-06"
            / "local_runs"
            / "freya_likelihood_equivalence.json"
        )
    )
    out.write_text(json.dumps(artifact, indent=2))
    v = artifact["verdict"]
    print(
        f"[freya] {v['n_theta']} theta points  max_rel_diff={v['max_rel_diff']:.3g}  "
        f"rtol={v['rtol']:.1g}  {'PASS' if v['passes'] else 'FAIL'}",
        flush=True,
    )
    for f in v["failures"]:
        th = "  ".join(f"{k}={x:.6g}" for k, x in f["theta"].items())
        print(f"[freya] DISAGREE theta[{f['index']}]: {th}")
        print(f"        total  A={f['route_A']!r}  B={f['route_B']!r}  rel={f['rel_diff']:.3g}")
        for band, b in f["bands"].items():
            print(
                f"        band {band}  A={b['route_A']!r}  B={b['route_B']!r}  "
                f"rel={b['rel_diff']:.3g}"
            )
    print(f"[freya] wrote {out}", flush=True)
    return 0 if v["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
