#!/usr/bin/env python
"""Pre/post-crop param + chi2 comparison for given bursts."""

import json
import os
import sys

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
J = f"{RUNS}/data/joint"
for b in sys.argv[1:]:
    n = json.load(open(f"{J}/{b}_joint_fit.json"))
    o = json.load(open(f"{J}/precrop_backup/{b}_joint_fit.json"))
    pn, po = n["percentiles"], o["percentiles"]
    ppc = {}
    pf = f"{J}/{b}_joint_ppc.json"
    if os.path.exists(pf):
        ppc = json.load(open(pf))

    def row(k, key="median", f="{:.2f}"):
        return f"  {k:9s} PRE {f.format(po[k][key]):>8s} -> POST {f.format(pn[k][key]):>8s}"

    print(f"=== {b} ===")
    print(row("alpha"))
    print(row("tau_1ghz", f="{:.3f}"))
    # zeta params differ by contract (per-band zeta_C/D vs shared zeta_1ghz/x_zeta)
    zkeys = [k for k in ("zeta_C", "zeta_D", "zeta_1ghz", "x_zeta") if k in pn and k in po]
    for k in zkeys:
        print(row(k, f="{:+.3f}" if k == "x_zeta" else "{:.2f}"))
    if not zkeys:
        print("  zeta      (param set changed PRE->POST; not directly comparable)")
    print(
        f"  lnZ       PRE {o.get('log_evidence', 0):8.0f} -> POST {n.get('log_evidence', 0):8.0f}"
    )
    if ppc:
        print(
            f"  POST chi2/dof  CHIME {ppc.get('chi2_chime', float('nan')):.2f}  DSA {ppc.get('chi2_dsa', float('nan')):.2f}"
        )
    print()
