#!/usr/bin/env python
"""A1 trigger-calibration campaign driver (Faber2026 plan, Phase 4).

Null grid: dnu_hwhm/dchan x S/N x num_subbands, n_real single-screen
injections per cell -> dlnZ null distributions -> threshold table at
0.5/1/5% false-escalation (conservative max-over-cells envelope).
Power grid: f x m2_ratio at the central noise cell, two-screen truth.
Prior-sensitivity rerun of the central cell with the wide f prior.

Per-cell checkpointing: each finished cell is written to
<out>.cells/<cell>.json immediately; rerunning skips finished cells.

Full grid is an h17 batch job (~tens of core-hours):
    python simulation/scripts/run_a1_trigger_calibration.py \
        --n-real 200 --out reports/a1_trigger_calibration.json
Local smoke:
    ... --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_FLITS_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_FLITS_ROOT / "simulation"), str(_FLITS_ROOT / "scintillation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trigger_calibration import (  # noqa: E402
    null_dlnz_cell,
    power_dlnz_cell,
    threshold_table,
)

CHANNEL_WIDTH_MHZ = 0.05
# Band width scales with subband count so every subband keeps a fixed
# channel budget: calculate_acf requires >= 20 unmasked channels, and a
# fixed total band starved the 8-subband arm (6 MHz / 8 = 15 channels ->
# every injection failed instantly on first launch).
SUBBAND_CHANNELS = 120


def _band_width_mhz(num_subbands):
    return num_subbands * SUBBAND_CHANNELS * CHANNEL_WIDTH_MHZ

NULL_GRID = {
    "dnu_over_dchan": [2, 5, 10, 30, 100],
    "snr": [10.0, 25.0, 50.0, 100.0],
    "num_subbands": [1, 4, 8],
}
POWER_GRID = {"f": [3.0, 10.0, 30.0, 100.0], "m2_ratio": [0.25, 1.0]}
CENTRAL = {"dnu_over_dchan": 10, "snr": 50.0, "num_subbands": 4}
RATES = (0.005, 0.01, 0.05)
SEED0 = 20260713


def _cell_key(kind, **kw):
    return kind + "__" + "_".join(f"{k}-{kw[k]}" for k in sorted(kw))


def _run_cell(kind, params, n_real, nlive, dlogz, n_real_cov, cell_index):
    seed = SEED0 + cell_index * 100_000
    if kind == "null":
        sample = null_dlnz_cell(
            dnu_hwhm_mhz=params["dnu_over_dchan"] * CHANNEL_WIDTH_MHZ,
            snr=params["snr"],
            band_width_mhz=_band_width_mhz(params["num_subbands"]),
            channel_width_mhz=CHANNEL_WIDTH_MHZ,
            num_subbands=params["num_subbands"],
            n_real=n_real, seed=seed, nlive=nlive, dlogz=dlogz,
            n_real_cov=n_real_cov,
        )
    else:
        sample = power_dlnz_cell(
            f=params["f"], m2_ratio=params["m2_ratio"],
            dnu1_hwhm_mhz=CENTRAL["dnu_over_dchan"] * CHANNEL_WIDTH_MHZ,
            snr=CENTRAL["snr"],
            band_width_mhz=_band_width_mhz(CENTRAL["num_subbands"]),
            channel_width_mhz=CHANNEL_WIDTH_MHZ,
            num_subbands=CENTRAL["num_subbands"],
            n_real=n_real, seed=seed, nlive=nlive, dlogz=dlogz,
            n_real_cov=n_real_cov,
        )
    return sample, seed


def _run_and_checkpoint(kind, params, n, idx, cells_dir, nlive, dlogz,
                        n_real_cov):
    """Module-level (picklable for ProcessPoolExecutor) cell runner with
    per-cell checkpointing."""
    cells_dir = Path(cells_dir)
    key = _cell_key(kind, **params)
    ck = cells_dir / f"{key}.json"
    if ck.exists():
        d = json.loads(ck.read_text())
        return kind, key, d["sample"], d["seed"]
    sample, seed = _run_cell(kind, params, n, nlive, dlogz, n_real_cov, idx)
    ck.write_text(json.dumps({"sample": sample, "seed": seed,
                              "params": params}))
    print(f"[cell {idx}] {key}: n={len(sample)} "
          f"finite={int(np.sum(np.isfinite(sample)))}", flush=True)
    return kind, key, sample, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--n-real-power", type=int, default=100)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--n-real-cov", type=int, default=150)
    ap.add_argument("--out", default="reports/a1_trigger_calibration.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid + tiny n for a local end-to-end check")
    ap.add_argument("--workers", type=int, default=1,
                    help="process-parallel cells (h17: ~ncores-4)")
    args = ap.parse_args()

    null_grid, power_grid = NULL_GRID, POWER_GRID
    n_real, n_real_power = args.n_real, args.n_real_power
    nlive, dlogz, n_real_cov = args.nlive, args.dlogz, args.n_real_cov
    if args.smoke:
        null_grid = {"dnu_over_dchan": [10], "snr": [50.0],
                     "num_subbands": [2]}
        power_grid = {"f": [10.0], "m2_ratio": [1.0]}
        n_real = n_real_power = 4
        nlive, dlogz, n_real_cov = 200, 1.0, 60

    out_path = Path(args.out)
    cells_dir = out_path.parent / (out_path.stem + ".cells")
    cells_dir.mkdir(parents=True, exist_ok=True)

    work = []
    idx = 0
    for d in null_grid["dnu_over_dchan"]:
        for s in null_grid["snr"]:
            for nb in null_grid["num_subbands"]:
                work.append(("null", {"dnu_over_dchan": d, "snr": s,
                                      "num_subbands": nb}, n_real, idx))
                idx += 1
    for f in power_grid["f"]:
        for m in power_grid["m2_ratio"]:
            work.append(("power", {"f": f, "m2_ratio": m}, n_real_power,
                         idx))
            idx += 1

    common = (str(cells_dir), nlive, dlogz, n_real_cov)
    nulls, powers = {}, {}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_and_checkpoint, *w, *common) for w in work]
            for fut in as_completed(futs):
                kind, key, sample, _ = fut.result()
                (nulls if kind == "null" else powers)[key] = sample
    else:
        for w in work:
            kind, key, sample, _ = _run_and_checkpoint(*w, *common)
            (nulls if kind == "null" else powers)[key] = sample

    thresholds = threshold_table(nulls, rates=RATES)
    op = thresholds[0.01]
    power_curves = {
        k: float(np.mean(np.asarray(v)[np.isfinite(v)] >= op))
        for k, v in powers.items()
    }

    git_sha = subprocess.run(
        ["git", "-C", str(_FLITS_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    report = {
        "thresholds": {str(r): thresholds[r] for r in RATES},
        "recommended_operating_point": {"rate": 0.01, "dlnz": thresholds[0.01]},
        "null_quantiles_by_cell": {
            k: {"q50": float(np.nanmedian(v)), "q95": float(np.nanquantile(v, 0.95)),
                "q99": float(np.nanquantile(v, 0.99)),
                "n_failed": int(np.sum(~np.isfinite(v)))}
            for k, v in nulls.items()
        },
        "power_curves_at_1pct": power_curves,
        "grid": {"null": null_grid, "power": power_grid, "central": CENTRAL,
                 "channel_width_mhz": CHANNEL_WIDTH_MHZ,
                 "subband_channels": SUBBAND_CHANNELS},
        "settings": {"n_real": n_real, "n_real_power": n_real_power,
                     "nlive": nlive, "dlogz": dlogz,
                     "n_real_cov": n_real_cov, "seed0": SEED0,
                     "smoke": bool(args.smoke)},
        "convention": "HWHM",
        "git_sha": git_sha,
        "command": " ".join(sys.argv),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path} — operating point (1%): dlnz >= "
          f"{thresholds[0.01]:.2f}")


if __name__ == "__main__":
    main()
