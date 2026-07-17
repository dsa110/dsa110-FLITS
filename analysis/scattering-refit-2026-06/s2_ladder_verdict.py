#!/usr/bin/env python
"""Fixed-s2 cross-N Bayes-factor adjudicator for the beta-native joint fits (ADR-0003).

Reads the Phase-B fixed-s2 fits `<burst>_joint_fit_C{C}D{D}_s2-{v}.json` (written by
run_joint_fit.py --gain-s2), and for every configuration pair differing by exactly one
component in one band computes ΔlnZ(N+1 vs N) at each shared s2. A component is REAL only
if ΔlnZ > +5 (Jeffreys "strong") CONSISTENTLY across all s2; a sign flip with the prior
scale s2 means the extra component is prior-driven, not data-driven.

Unlike joint_ladder/_s2verdict.py this does NOT filter by PBF family: the beta-native fits
share ONE sampled beta across both bands, so the PBF is coherent by construction (ADR-0006)
— the pbf_C/pbf_D incomparability of the alpha-era ladder does not apply.

  FLITS_RUNS=~/Developer/scratch/flits-local-runs python s2_ladder_verdict.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

RUNS = Path(os.environ.get("FLITS_RUNS", os.path.expanduser("~/Developer/scratch/flits-local-runs")))
JOINT = RUNS / "data/joint"
THRESH = 5.0

PAT = re.compile(r"(?P<burst>.+?)_joint_fit_C(?P<C>\d+)D(?P<D>\d+)_s2-(?P<s2>\d+)\.json$")


def load_grids():
    """grids[burst][(C,D)][s2] = (lnZ, lnZ_err)."""
    grids = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(JOINT / "*_joint_fit_C*D*_s2-*.json")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        lnz = d.get("log_evidence")
        if lnz is None:
            continue
        grids[m["burst"]][(int(m["C"]), int(m["D"]))][int(m["s2"])] = (
            float(lnz), float(d.get("log_evidence_err") or 0.0))
    return grids


def verdict(deltas):
    if not deltas:
        return "no shared s2"
    signs = {d > 0 for d in deltas}
    if len(signs) > 1:
        return "NOT ROBUST (sign flips across s2) — prior-driven, NOT real"
    if all(d > THRESH for d in deltas):
        return f"REAL (all ΔlnZ > +{THRESH:.0f})"
    if all(d < -THRESH for d in deltas):
        return f"REJECTED (all ΔlnZ < -{THRESH:.0f}) — simpler model favored"
    return "WEAK (consistent sign, |ΔlnZ| < 5) — inconclusive"


def main():
    grids = load_grids()
    if not grids:
        print(f"No fixed-s2 fits under {JOINT} yet (run campaign_B_s2.sh first).")
        return
    for burst in sorted(grids):
        g = grids[burst]
        print(f"\n===== {burst} : fixed-s2 cross-N ladder =====")
        s2set = sorted({s for cfg in g.values() for s in cfg})
        header = "  cfg    " + "".join(f"  s2={s:<4}" for s in s2set)
        print(header)
        for cfg in sorted(g):
            line = f"  C{cfg[0]}D{cfg[1]}  "
            for s in s2set:
                v = g[cfg].get(s)
                line += f"  {v[0]:8.1f}" if v else "  --      "
            print(line)
        for (Ca, Da) in sorted(g):
            for (Cb, Db) in sorted(g):
                if (Cb == Ca + 1 and Db == Da) or (Db == Da + 1 and Cb == Ca):
                    shared = [s for s in s2set if s in g[(Ca, Da)] and s in g[(Cb, Db)]]
                    deltas = [g[(Cb, Db)][s][0] - g[(Ca, Da)][s][0] for s in shared]
                    ds = " / ".join(f"s2={s}:{d:+.1f}" for s, d in zip(shared, deltas))
                    print(f"    Δ(C{Cb}D{Db} vs C{Ca}D{Da}): {ds}   -> {verdict(deltas)}")


if __name__ == "__main__":
    main()
