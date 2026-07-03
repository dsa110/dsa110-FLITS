#!/usr/bin/env python
"""Tail-coverage preflight for the power-law PBF (issue #101).

Pure, deterministic check that a prepared fit window captures enough of the
pulse-broadening function's heavy power-law tail for the tail SHAPE, not its
truncation, to drive the sampled beta. Run before any sampling (#104 gates on
it at the candidate (tau_1ghz, beta) it adopts).

PBF (gaussian_powerlaw_convolution, burstfit.py; Cordes review sec 11.2):

    p(s) ~ exp(-s)                 for s <= s_c
    p(s) ~ exp(-s_c) (s/s_c)^(-beta/2)   for s > s_c   [continuous at s_c]

with s = lag / tau(nu), crossover s_c = 2 ln(2/(4-beta)), and
tau(nu) = tau_1ghz * nu^(-alpha(beta)), alpha = 2*beta/(beta-2) (turbulence.py).
The tail is worst at a band's LOWEST frequency (largest tau), so the verdict is
evaluated there. Both integrals are closed-form (m = beta/2 - 1 in (0, 1)):

    I(S)     = 1 - e^(-min(S, s_c))
               + [S > s_c] e^(-s_c) s_c/m (1 - (S/s_c)^(-m))
    I(inf)   = 1 - e^(-s_c) + e^(-s_c) s_c/m

Captured fraction F = I(S)/I(inf); "captured e-folds" is the equivalent
exponential depth E = -ln(1 - F), so for the pure-exponential PBF (beta -> 4,
where the model dispatches to the closed exp form within BETA_EXP_EPS of 4,
ADR-0006) E equals the window span in units of tau exactly.

The default threshold (DEFAULT_MIN_EFOLDS = 3, ~95% of the PBF area) is a
caller-tunable engineering default, not scientific adjudication (out of scope
per the issue); the POC's synthetic setup captured ~10 e-folds by design.
Epistemic status of the tau scale this is stress-tested around: the deprecated
exp-era fit *suggested* tau_1ghz ~ 0.119 ms for freya — a candidate to evaluate
at, not ground truth.

CLI: run the preflight on freya's real prepared CHIME+DSA windows (the #99
run-configs through the joint driver's own prepare()) across a (tau, beta)
candidate grid spanning the exp-era suggestion, and record the verdicts:

    conda run -n flits python analysis/scattering-refit-2026-06/tail_coverage.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scattering.scat_analysis.turbulence import (  # noqa: E402
    BETA_EXP_EPS,
    BETA_THIN_SCREEN_MAX,
    BETA_THIN_SCREEN_MIN,
    alpha_from_beta,
)

DEFAULT_MIN_EFOLDS = 3.0

# Candidate grid spanning the deprecated exp-era suggestion (tau ~ 0.119 ms,
# alpha ~ 4.36 -> beta ~ 3.70): half/double in tau; beta from steep (3.1) through
# Kolmogorov (11/3) and the exp-era-implied 3.70 to the near-exponential edge.
TAU_GRID_MS = (0.06, 0.119, 0.24)
BETA_GRID = (3.1, 11.0 / 3.0, 3.7, 3.95)


def pbf_crossover(beta: float) -> float:
    """Exp-to-power-law crossover s_c = 2 ln(2/(4-beta)), floored like the kernel."""
    return max(2.0 * np.log(2.0 / (4.0 - float(beta))), 1e-3)


def pbf_captured_fraction(s_window, beta: float):
    """Analytic fraction of the PBF's total area within lag/tau in [0, s_window].

    Mirrors the model's PBF dispatch (burstfit.FRBModel.__call__): beta clipped
    to the thin-screen interval, pure exponential within BETA_EXP_EPS of 4.
    """
    s = np.maximum(np.asarray(s_window, float), 0.0)
    beta = float(np.clip(beta, BETA_THIN_SCREEN_MIN, BETA_THIN_SCREEN_MAX))
    if beta >= BETA_THIN_SCREEN_MAX - BETA_EXP_EPS:
        return -np.expm1(-s)
    s_c = pbf_crossover(beta)
    m = 0.5 * beta - 1.0
    head = -np.expm1(-np.minimum(s, s_c))
    tail = np.where(
        s > s_c,
        np.exp(-s_c) * s_c / m * (1.0 - (np.maximum(s, s_c) / s_c) ** (-m)),
        0.0,
    )
    total = -np.expm1(-s_c) + np.exp(-s_c) * s_c / m
    return (head + tail) / total


def tail_coverage(
    freq_ghz,
    tau_1ghz_ms: float,
    beta: float,
    time_ms,
    t0_ms: float | None = None,
    min_efolds: float = DEFAULT_MIN_EFOLDS,
) -> dict:
    """Captured e-folds of the PBF tail in a prepared window + hard-threshold verdict.

    freq_ghz: band frequencies (array or scalar, GHz) — evaluated at the minimum
    (largest tau, worst tail). time_ms: the prepared window axis (the driver
    prepare()'s model.time) or (t_start, t_end). t0_ms: burst position in the
    window (defaults to the window start — conservative only when the burst sits
    early; pass the init's t0 for real windows).
    """
    freq = np.atleast_1d(np.asarray(freq_ghz, float))
    t = np.atleast_1d(np.asarray(time_ms, float))
    t_end = float(t[-1])
    t0 = float(t[0]) if t0_ms is None else float(t0_ms)
    beta_eff = float(np.clip(beta, BETA_THIN_SCREEN_MIN, BETA_THIN_SCREEN_MAX))
    exp_branch = beta_eff >= BETA_THIN_SCREEN_MAX - BETA_EXP_EPS
    alpha = alpha_from_beta(beta_eff)
    f_min = float(freq.min())
    tau_fmin = float(tau_1ghz_ms) * f_min ** (-alpha)
    s_window = max((t_end - t0) / tau_fmin, 0.0)
    # fraction capped just under 1 so E = -ln(1-F) stays finite/JSON-serializable
    fraction = float(np.clip(pbf_captured_fraction(s_window, beta_eff), 0.0, 1.0 - 1e-15))
    efolds = float(-np.log1p(-fraction))
    return {
        "efolds": efolds,
        "captured_fraction": fraction,
        "s_window": s_window,
        # None on the exp branch (s_c -> inf as beta -> 4; no crossover to report)
        "s_crossover": None if exp_branch else pbf_crossover(beta_eff),
        "tau_at_fmin_ms": tau_fmin,
        "f_min_ghz": f_min,
        "alpha": alpha,
        "window_span_ms": t_end - t0,
        "min_efolds": float(min_efolds),
        "passes": bool(efolds >= float(min_efolds)),
    }


def _load_driver():
    # Same loading pattern as tests/test_freya_local_runs_smoke.py: the driver
    # resolves its repo from FLITS_REPO at import time (HPCC default otherwise).
    os.environ["FLITS_REPO"] = str(REPO)
    path = REPO / "analysis" / "scattering-refit-2026-06" / "local_runs" / "run_joint_fit.py"
    spec = importlib.util.spec_from_file_location("run_joint_fit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_preflight(burst: str = "freya", min_efolds: float = DEFAULT_MIN_EFOLDS) -> dict:
    """Preflight the (tau, beta) candidate grid on a burst's real prepared windows.

    t0 is the RAW data-driven initial guess, not the driver's MLE-refined init:
    the post-#98 refine rails beta to the thin-screen floor on freya and drags
    t0 with it (observed CHIME: 17.6 -> 31.4 ms, nearly the window end), which
    would corrupt the coverage verdict. Both t0s are recorded per band.
    """
    local_runs = REPO / "analysis" / "scattering-refit-2026-06" / "local_runs"
    driver = _load_driver()
    from scattering.scat_analysis.burstfit_init import data_driven_initial_guess

    bands = {}
    with tempfile.TemporaryDirectory() as tmp:
        for band in ("chime", "dsa"):
            cfg_path = local_runs / "configs" / f"{burst}_{band}_run.yaml"
            model, init_mle = driver.prepare(str(cfg_path), f"{burst}_{band}", tmp)
            t0 = float(
                data_driven_initial_guess(
                    data=model.data,
                    freq=model.freq,
                    time=model.time,
                    dm=float(model.dm_init),
                    verbose=False,
                ).params.t0
            )
            grid = [
                {"tau_1ghz_ms": tau, "beta": beta}
                | tail_coverage(model.freq, tau, beta, model.time, t0_ms=t0, min_efolds=min_efolds)
                for tau in TAU_GRID_MS
                for beta in BETA_GRID
            ]
            bands[band] = {
                "config": str(cfg_path.relative_to(REPO)),
                "window_ms": [float(model.time[0]), float(model.time[-1])],
                "t0_raw_init_ms": t0,
                "t0_mle_init_ms": float(init_mle.t0),
                "n_time": int(model.time.size),
                "freq_ghz": [float(model.freq.min()), float(model.freq.max())],
                "grid": grid,
            }
    all_points = [g for b in bands.values() for g in b["grid"]]
    return {
        "issue": "#101",
        "burst": burst,
        "min_efolds": float(min_efolds),
        "epistemic_note": (
            "tau grid spans the deprecated exp-era fit's *suggestion* for freya "
            "(tau_1ghz ~ 0.119 ms, alpha ~ 4.36 -> beta ~ 3.70) — a stress-test "
            "scale, not ground truth; whether it survives is what #104-#106 test."
        ),
        "bands": bands,
        "overall": {
            "n_grid_points": len(all_points),
            "n_fail": sum(not g["passes"] for g in all_points),
            "worst_efolds": min(g["efolds"] for g in all_points),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--burst", default="freya")
    ap.add_argument("--min-efolds", type=float, default=DEFAULT_MIN_EFOLDS)
    ap.add_argument(
        "--out",
        default=None,
        help="artifact path (default local_runs/<burst>_tail_coverage.json)",
    )
    args = ap.parse_args(argv)

    artifact = run_preflight(args.burst, args.min_efolds)
    out = (
        Path(args.out)
        if args.out
        else (
            REPO
            / "analysis"
            / "scattering-refit-2026-06"
            / "local_runs"
            / f"{args.burst}_tail_coverage.json"
        )
    )
    out.write_text(json.dumps(artifact, indent=2))
    for band, b in artifact["bands"].items():
        for g in b["grid"]:
            flag = "PASS" if g["passes"] else "FAIL"
            print(
                f"[{args.burst}] {band}  tau={g['tau_1ghz_ms']:.3g} ms  "
                f"beta={g['beta']:.3g}  efolds={g['efolds']:.2f}  "
                f"(captured {g['captured_fraction']:.3%})  {flag}",
                flush=True,
            )
    print(f"[{args.burst}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
