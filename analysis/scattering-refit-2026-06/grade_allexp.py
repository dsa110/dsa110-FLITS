"""Grade the canonical all-exp [1.0,6.0] joint fits through the ADR-0004 gate.

Reads each burst's canonical all-exp fit JSON from a local dir (pulled from HPCC)
and runs gate_one (L1 bounds + rail, L3 alpha-physics; L2 chi2 = unknown -> MARGINAL
until PPC is wired for multi-component fits). Prints the citable-alpha roster.

Usage: python grade_allexp.py <dir-of-pulled-fits>
"""

import json
import sys
from pathlib import Path

from gate_joint_committed import gate_one  # ADR-0004 gate logic (floor 1.0, 3-sigma rail)

# adjudicated canonical model per burst (filename tag). zach=C1D1 (C2D3 3rd comp
# prior-driven), whitney=C2D2 local, johndoeII=C2D1, hamilton excluded (single-band limit).
CANON = {
    "casey": "sharedzeta",
    "chromatica": "sharedzeta",
    "freya": "sharedzeta",
    "wilhelm": "sharedzeta",
    "mahi": "C1D1",
    "phineas": "C3D3",
    "oran": "C2D1",
    "isha": "C2D1",
    "johndoeII": "C2D1",
    "zach": "",  # zach C1D1 -> empty tag
}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for b, m in CANON.items():
        tag = f"_{m}" if m else ""
        fp = d / f"{b}_joint_fit{tag}_pbf-exp-exp.json"
        if not fp.exists():
            print(f"!! {b}: MISSING {fp.name}")
            continue
        fit = json.loads(fp.read_text())
        # paired matplotlib-free PPC (joint_ppc_multi.py) -> per-band reduced chi2 for L2
        ppc_fp = d / f"{b}_joint_ppc_multi{tag}_pbf-exp-exp.json"
        ppc = json.loads(ppc_fp.read_text()) if ppc_fp.exists() else None
        v = gate_one(b, fit, ppc)  # ppc=None still -> L2 MARGINAL (chi2 unknown)
        rows.append(v)

    rows.sort(key=lambda r: -(r["alpha"] or 0))
    print(
        f"{'burst':11s} {'alpha':>18s} {'bounds':>11s} rail {'chi2C/D':>10} "
        f"{'L1':>4} {'L2':>4} {'L3':>4} {'FINAL':>9}  reason"
    )
    for r in rows:
        b = r["burst"]
        fit = json.loads(
            (
                d / f"{b}_joint_fit{('_' + CANON[b]) if CANON[b] else ''}_pbf-exp-exp.json"
            ).read_text()
        )
        a = fit["alpha"]
        astr = f"{a['median']:.3f}+{a.get('err_plus', 0):.3f}/-{a.get('err_minus', 0):.3f}"
        lo, hi = fit["alpha_bounds"]
        cc, cd = r.get("chi2_chime"), r.get("chi2_dsa")
        chi2s = f"{cc:.2f}/{cd:.2f}" if cc is not None and cd is not None else "  -/-"
        print(
            f"{b:11s} {astr:>18s} [{lo},{hi}] {str(r['rail'])[0]:>4} {chi2s:>10} "
            f"{r['l1']:>4} {r['l2']:>4} {r['l3']:>4} {r['final']:>9}  {r['reason']}"
        )


if __name__ == "__main__":
    main()
