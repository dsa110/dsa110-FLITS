"""Run α=4 joint refits for the accepted July morphology roster."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from galaxies.foreground.tau_consistency import (
    TAU_CONSISTENCY_DIR,
    JulyMorphology,
    alpha4_refit_path,
    load_july_accepted_morphologies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_JOINT = REPO_ROOT / "analysis" / "scattering-refit-2026-06" / "run_joint_fit.py"


def build_alpha4_joint_cmd(burst: str, morph: JulyMorphology, nlive: int, nproc: int) -> list[str]:
    return [
        sys.executable,
        str(RUN_JOINT),
        burst,
        str(nlive),
        str(nproc),
        "--alpha-lo",
        "4",
        "--alpha-hi",
        "4",
        "--components-C",
        str(morph.components_C),
        "--components-D",
        str(morph.components_D),
        "--fixed-delta-dm-C",
        str(morph.fixed_delta_dm_C),
        "--fixed-delta-dm-D",
        str(morph.fixed_delta_dm_D),
    ]


def run_burst(burst: str, nlive: int = 600, nproc: int = 8) -> Path:
    burst = burst.lower()
    morph = load_july_accepted_morphologies().get(burst)
    if morph is None:
        raise ValueError(f"burst is not eligible: no accepted July morphology for {burst!r}")
    TAU_CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)
    out = alpha4_refit_path(burst, morph)
    if not RUN_JOINT.is_file():
        raise FileNotFoundError(f"missing driver: {RUN_JOINT}")
    cmd = build_alpha4_joint_cmd(burst, morph, nlive, nproc)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    runs = Path(__import__("os").environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs"))
    suffix = "" if morph.variant == "C1D1" else f"_C{morph.components_C}D{morph.components_D}"
    produced = runs / "data" / "joint" / f"{burst}_joint_fit{suffix}.json"
    if not produced.is_file():
        raise FileNotFoundError(
            f"joint fit subprocess finished but expected output missing: {produced}"
        )
    with open(produced) as fh:
        payload = json.load(fh)
    payload["components_C"] = morph.components_C
    payload["components_D"] = morph.components_D
    payload["variant"] = morph.variant
    payload["alpha_fixed"] = 4.0
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    if not out.is_file():
        raise RuntimeError(f"failed to write tau consistency refit: {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="α=4 fixed joint refits → data/tau_consistency/")
    ap.add_argument("bursts", nargs="*", help="accepted July burst nicknames")
    ap.add_argument("--nlive", type=int, default=600)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument(
        "--dry-run", action="store_true", help="print refit commands without running them"
    )
    args = ap.parse_args(argv)
    morphologies = load_july_accepted_morphologies()
    targets = [b.lower() for b in args.bursts] if args.bursts else sorted(morphologies)
    for burst in targets:
        morph = morphologies.get(burst)
        if morph is None:
            raise ValueError(f"burst is not eligible: no accepted July morphology for {burst!r}")
        path = alpha4_refit_path(burst, morph)
        if args.dry_run:
            print(
                f"{burst} variant={morph.variant} components_C={morph.components_C} "
                f"components_D={morph.components_D} "
                f"fixed_delta_dm_C={morph.fixed_delta_dm_C} "
                f"fixed_delta_dm_D={morph.fixed_delta_dm_D} output={path} "
                f"nlive={args.nlive} nproc={args.nproc}"
            )
        else:
            path = run_burst(burst, nlive=args.nlive, nproc=args.nproc)
            print(f"[{burst}] wrote {path}")


if __name__ == "__main__":
    main()
