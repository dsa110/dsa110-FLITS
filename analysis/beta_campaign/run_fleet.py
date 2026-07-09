#!/usr/bin/env python
"""Fleet driver: beta-coherent thin-screen joint fits, full 12-burst sample.

Phase 5 of docs/rse/specs/plan-beta-coherent-thin-screen-campaign.md. Each
burst runs at its previously adjudicated multiplicity (ADR-0005 / grade_allexp
CANON) through the beta-native local runner + the beta-native PPC:

  run_joint_fit.py <burst> <nlive> <nproc> --beta-lo 3.0 --beta-hi 4.0 <model>
  joint_ppc_multi.py <burst> <suffix>

freya runs FIRST as a regression gate against the committed verdict
(analysis/beta_poc/freya/freya_beta_verdict.json, beta = 3.6838): if the
re-fit disagrees beyond FREYA_STOP_TOL the fleet stops (plan: Risks).

  conda run -n flits python analysis/beta_campaign/run_fleet.py [--nproc N]
      [--parallel K] [--only burst1,burst2]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("FLITS_RUNS", "/Users/jakobfaber/Developer/scratch/flits-local-runs"))
RUNNER = REPO / "analysis/scattering-refit-2026-06/local_runs/run_joint_fit.py"
PPC = REPO / "analysis/scattering-refit-2026-06/joint_ppc_multi.py"
LOGDIR = RUNS / "logs"

BETA_ARGS = ["--beta-lo", "3.0", "--beta-hi", "4.0"]  # campaign default (alpha in [4, 6])

# burst, nlive, model flags, output suffix (must mirror the runner's tag logic)
FLEET = [
    ("freya", 600, ["--shared-zeta"], "_sharedzeta"),
    ("casey", 600, ["--shared-zeta"], "_sharedzeta"),
    ("chromatica", 600, ["--shared-zeta"], "_sharedzeta"),
    ("wilhelm", 600, ["--shared-zeta"], "_sharedzeta"),
    ("hamilton", 600, ["--shared-zeta"], "_sharedzeta"),
    ("mahi", 600, ["--components-C", "1", "--components-D", "1", "--force-multi"], "_C1D1"),
    # zach's CITABLE product is the bespoke C2D4_cwin refit (refit_runner.py,
    # per-component windows) promoted 2026-07-09 -- NOT reproducible by this
    # fleet recipe. This C1D1 entry is retained only as the historical fleet
    # baseline; do NOT let a fleet re-run overwrite the promoted verdict row
    # (grade SUFFIX["zach"] = "_C2D4_cwin"). Use --only to exclude zach, or
    # restage fits/zach_joint_fit_C2D4_cwin.json after any fleet run.
    ("zach", 600, ["--components-C", "1", "--components-D", "1", "--force-multi"], "_C1D1"),
    ("oran", 600, ["--components-C", "2", "--components-D", "1"], "_C2D1"),
    ("isha", 600, ["--components-C", "2", "--components-D", "1"], "_C2D1"),
    ("johndoeII", 600, ["--components-C", "2", "--components-D", "2"], "_C2D2"),
    ("whitney_fine", 800, ["--components-C", "2", "--components-D", "2"], "_C2D2"),
    ("phineas", 1000, ["--components-C", "3", "--components-D", "3"], "_C3D3"),
]

FREYA_REF = 3.6838  # committed beta (freya_beta_verdict.json)
FREYA_STOP_TOL = 0.05


def run_one(burst: str, nlive: int, flags: list[str], suffix: str, nproc: int) -> dict:
    env = {**os.environ, "FLITS_REPO": str(REPO), "FLITS_RUNS": str(RUNS)}
    log = LOGDIR / f"{burst}_beta.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "flits",
                "python",
                str(RUNNER),
                burst,
                str(nlive),
                str(nproc),
                *BETA_ARGS,
                *flags,
            ],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        ).returncode
    out = {"burst": burst, "suffix": suffix, "rc": rc, "minutes": (time.time() - t0) / 60.0}
    if rc == 0:
        with open(log, "a") as fh:
            ppc_rc = subprocess.run(
                ["conda", "run", "-n", "flits", "python", str(PPC), burst, suffix],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            ).returncode
        out["ppc_rc"] = ppc_rc
        fit = RUNS / "data/joint" / f"{burst}_joint_fit{suffix}.json"
        if fit.exists():
            d = json.loads(fit.read_text())
            out["beta"] = d["beta"]["median"]
            out["alpha"] = d["alpha"]["median"]
            out["tau_1ghz"] = d["tau_1ghz"]["median"]
    status = "ok" if rc == 0 and out.get("ppc_rc", 1) == 0 else "FAILED"
    print(
        f"[fleet] {burst}{suffix}: {status} beta={out.get('beta', float('nan')):.3f} "
        f"({out['minutes']:.0f} min)",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated burst subset (skips the freya gate ordering)",
    )
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)

    fleet = FLEET
    if args.only:
        keep = set(args.only.split(","))
        fleet = [f for f in FLEET if f[0] in keep]

    results = []
    rest = list(fleet)
    if rest and rest[0][0] == "freya" and not args.only:
        r = run_one(*rest.pop(0), args.nproc)
        results.append(r)
        if r["rc"] != 0:
            print("[fleet] freya run FAILED -- stopping fleet", flush=True)
            return 1
        delta = abs(r["beta"] - FREYA_REF)
        print(
            f"[fleet] freya regression: |beta - {FREYA_REF}| = {delta:.4f} "
            f"(stop tol {FREYA_STOP_TOL})",
            flush=True,
        )
        if delta > FREYA_STOP_TOL:
            print("[fleet] freya regression FAILED -- stopping fleet (plan: Risks)", flush=True)
            (Path(__file__).parent / "fleet_status.json").write_text(json.dumps(results, indent=2))
            return 2

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_one, b, n, fl, sfx, args.nproc) for b, n, fl, sfx in rest]
        results.extend(f.result() for f in futs)

    (Path(__file__).parent / "fleet_status.json").write_text(json.dumps(results, indent=2))
    bad = [r["burst"] for r in results if r["rc"] != 0 or r.get("ppc_rc", 1) != 0]
    print(
        f"[fleet] done: {len(results) - len(bad)}/{len(results)} ok"
        + (f"; FAILED: {bad}" if bad else ""),
        flush=True,
    )
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
