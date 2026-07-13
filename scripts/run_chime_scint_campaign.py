#!/usr/bin/env python3
"""P4d campaign runner: all-12 CHIME scintillation analysis, uniform methodology.

Per burst: patch the shipped <nick>_chime_hi.yaml with the local npz path,
run scint_analysis.run_analysis in an isolated output dir (results JSON +
figures + figures.manifest.json land there), bounded local parallelism.
Freya factor-isolation sweep (canfar_reference on/off x first_fit_lag 1-3)
runs as extra jobs tagged freya_sweep_*.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

BURSTS = [
    "casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
    "mahi", "oran", "phineas", "whitney", "wilhelm", "zach",
]


def patch_config(cfg_path: Path, data_path: Path, overrides: dict) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["input_data_path"] = str(data_path)
    for dotted, val in overrides.items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg


def run_job(name: str, cfg: dict, outdir: Path, python: str, pkg_root: Path,
            mwprop_path: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    tel_link = outdir / "telescopes"
    if not tel_link.exists():
        tel_link.symlink_to(pkg_root / "configs" / "telescopes")
    cfg_file = outdir / f"{name}_config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{pkg_root}:{mwprop_path}"
    env["FLITS_ROOT"] = str(pkg_root.parent)
    env.setdefault("MPLBACKEND", "Agg")
    t0 = time.time()
    proc = subprocess.run(
        [python, "-m", "scint_analysis.run_analysis", str(cfg_file)],
        cwd=outdir, env=env, capture_output=True, text=True, timeout=3600,
    )
    (outdir / "run.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    results = sorted(outdir.glob("*_analysis_results.json"))
    return {
        "job": name,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 1),
        "results_json": str(results[0]) if results else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--data-dir", default=str(Path.home() / "Data/Faber2026/dsa110/scintillation/data"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--python", default=str(Path.home() / ".conda/envs/flits/bin/python"))
    ap.add_argument("--mwprop", default="/tmp/flits-mwprop")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bursts", nargs="*", default=BURSTS)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    wt = Path(args.worktree)
    pkg_root = wt / "scintillation"
    cfg_dir = pkg_root / "configs" / "bursts"
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)

    jobs = []
    for n in args.bursts:
        cfg_path = cfg_dir / f"{n}_chime_hi.yaml"
        data = data_dir / f"{n}_chime_hi.npz"
        if not cfg_path.exists() or not data.exists():
            print(f"SKIP {n}: missing {'config' if not cfg_path.exists() else 'data'}",
                  file=sys.stderr)
            continue
        # uniform modern-default methodology for the science pass:
        # full CHIME mitigation stack + intra-pulse m(t), same for all bursts
        science_overrides = {
            "analysis.grid_regularization.enable": True,
            "analysis.bandpass_normalization.enable": True,
            "analysis.acf.enable_intra_pulse_analysis": True,
        }
        jobs.append((n, patch_config(cfg_path, data, dict(science_overrides)), out_root / n))

    if not args.skip_sweep and "freya" in args.bursts:
        base = cfg_dir / "freya_chime_hi.yaml"
        data = data_dir / "freya_chime_hi.npz"
        for mode in ("default", "canfar_reference"):
            for lag in (1, 2, 3):
                ov = {"analysis.acf.first_fit_lag": lag}
                if mode == "canfar_reference":
                    ov["analysis.preprocessing.mode"] = "canfar_reference"
                name = f"freya_sweep_{mode}_lag{lag}"
                jobs.append((name, patch_config(base, data, ov), out_root / "freya_sweep" / name))

    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, n, c, d, args.python, pkg_root, args.mwprop): n
                for n, c, d in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            summary.append(r)
            print(f"[{len(summary)}/{len(jobs)}] {r['job']}: "
                  f"rc={r['returncode']} {r['seconds']}s", flush=True)

    (out_root / "campaign_summary.json").write_text(json.dumps(
        {"jobs": sorted(summary, key=lambda r: r["job"]),
         "n_ok": sum(1 for r in summary if r["returncode"] == 0),
         "n_total": len(summary)}, indent=2))
    print(json.dumps({"n_ok": sum(1 for r in summary if r["returncode"] == 0),
                      "n_total": len(summary)}))


if __name__ == "__main__":
    main()
