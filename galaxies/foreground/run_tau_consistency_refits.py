"""Run α=4 fixed all-exp joint refits for the tau_consistency track."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from galaxies.foreground.tau_consistency import TAU_CONSISTENCY_DIR, co_detected_nicknames

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_JOINT = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "run_joint_fit.py"


def run_burst(burst: str, nlive: int = 600, nproc: int = 8) -> Path:
    TAU_CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)
    out = TAU_CONSISTENCY_DIR / f"{burst}_joint_alpha4_pbf-exp-exp.json"
    if not RUN_JOINT.is_file():
        raise FileNotFoundError(f"missing driver: {RUN_JOINT}")
    cmd = [
        sys.executable,
        str(RUN_JOINT),
        burst,
        str(nlive),
        str(nproc),
        "--alpha-lo",
        "4",
        "--alpha-hi",
        "4",
        "--pbf-C",
        "exp",
        "--pbf-D",
        "exp",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    runs = Path(
        __import__("os").environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
    )
    produced = runs / "data" / "joint" / f"{burst}_joint_fit.json"
    if not produced.is_file():
        raise FileNotFoundError(
            f"joint fit subprocess finished but expected output missing: {produced}"
        )
    with open(produced) as fh:
        payload = json.load(fh)
    payload["alpha_fixed"] = 4.0
    payload["pbf_C"] = "exp"
    payload["pbf_D"] = "exp"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    if not out.is_file():
        raise RuntimeError(f"failed to write tau consistency refit: {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="α=4 fixed joint refits → data/tau_consistency/")
    ap.add_argument("bursts", nargs="*", help="burst nicknames (default: all 12)")
    ap.add_argument("--nlive", type=int, default=600)
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args(argv)
    targets = [b.lower() for b in args.bursts] if args.bursts else co_detected_nicknames()
    for burst in targets:
        path = run_burst(burst, nlive=args.nlive, nproc=args.nproc)
        print(f"[{burst}] wrote {path}")


if __name__ == "__main__":
    main()
