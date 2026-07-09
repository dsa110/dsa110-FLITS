#!/usr/bin/env python
"""Grade the beta-coherent thin-screen campaign fits (Phase 6).

For each fleet output {burst}_joint_fit{suffix}.json (+ paired
{burst}_joint_ppc_multi{suffix}.json) under $FLITS_RUNS/data/joint:

- gate_one (gate_joint_committed.py) -- the runtime 3-level verdict,
  beta-native (rails on beta_bounds, alpha derived);
- rail classification on the SAMPLED beta posterior:
    interior    -- measures beta; alpha = 2b/(b-2) quotable as a value
    railed-hi   -- posterior pinned at beta=4 (exponential member):
                   alpha = 4 quotable only as a geometry-conditioned limit;
                   ADR-0007 re-open candidate
    railed-lo   -- pinned at beta=3 (alpha=6 L1 ceiling): unconstrained-steep
  using BOTH the ADR-0004 3-sigma rule and the weighted posterior mass
  within EDGE_MASS_WIDTH of the bound (from the samples npz).

Writes beta_campaign_verdicts.{json,md} beside this script (committed
artifacts; the npz posteriors stay in scratch).

  FLITS_RUNS=... conda run -n flits python analysis/beta_campaign/grade_beta_campaign.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("FLITS_RUNS", "/Users/jakobfaber/Developer/scratch/flits-local-runs"))
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "gate_joint_committed",
    REPO / "analysis" / "scattering-refit-2026-06" / "gate_joint_committed.py",
)
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

# Fleet output suffixes (run_fleet.py FLEET); freya's committed verdict is the
# regression reference.
SUFFIX = {
    "freya": "_sharedzeta",
    "casey": "_sharedzeta",
    "chromatica": "_sharedzeta",
    "wilhelm": "_sharedzeta",
    "hamilton": "_sharedzeta",
    "mahi": "_C1D1",
    # zach: bespoke morphology-audit refit (refit_runner.py, per-component
    # windows) promoted 2026-07-09, NOT a run_fleet product; stage
    # fits/zach_joint_fit_C2D4_cwin.json into RUNS/data/joint to re-grade.
    "zach": "_C2D4_cwin",
    "oran": "_C2D1",
    "isha": "_C2D1",
    "johndoeII": "_C2D2",
    "whitney_fine": "_C2D2",
    "phineas": "_C3D3",
}
EDGE_MASS_WIDTH = 0.05  # beta window at each bound for the posterior-mass rail test
EDGE_MASS_FRAC = 0.30  # >= this much mass in the window => railed


def classify_rail(fit: dict, burst: str, suffix: str) -> dict:
    b = fit["beta"]
    lo, hi = fit["beta_bounds"]
    med, em, ep = b["median"], b["err_minus"], b["err_plus"]
    three_sigma_lo = med - 3.0 * em <= lo
    three_sigma_hi = med + 3.0 * ep >= hi
    mass_lo = mass_hi = None
    npz = RUNS / "data/joint" / f"{burst}_joint_samples{suffix}.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        names = list(d["param_names"])
        if "beta" in names:
            bs = d["samples"][:, names.index("beta")]
            w = d["weights"] / d["weights"].sum()
            mass_lo = float(w[bs <= lo + EDGE_MASS_WIDTH].sum())
            mass_hi = float(w[bs >= hi - EDGE_MASS_WIDTH].sum())
    railed_hi = three_sigma_hi or (mass_hi is not None and mass_hi >= EDGE_MASS_FRAC)
    railed_lo = three_sigma_lo or (mass_lo is not None and mass_lo >= EDGE_MASS_FRAC)
    if railed_hi and not railed_lo:
        cls = "railed-hi (exponential-consistent; alpha=4 as limit; ADR-0007 candidate)"
    elif railed_lo and not railed_hi:
        cls = "railed-lo (alpha=6 ceiling; unconstrained-steep)"
    elif railed_hi and railed_lo:
        cls = "unconstrained (mass at both bounds)"
    else:
        cls = "interior"
    return {
        "class": cls.split(" ")[0],
        "detail": cls,
        "three_sigma_lo": bool(three_sigma_lo),
        "three_sigma_hi": bool(three_sigma_hi),
        "edge_mass_lo": mass_lo,
        "edge_mass_hi": mass_hi,
    }


def main() -> int:
    jdir = RUNS / "data/joint"
    rows = []
    missing = []
    for burst, suffix in SUFFIX.items():
        fp = jdir / f"{burst}_joint_fit{suffix}.json"
        if not fp.exists():
            missing.append(burst)
            continue
        fit = json.loads(fp.read_text())
        if "beta" not in fit:  # legacy alpha-era file sharing the campaign name
            missing.append(burst)
            continue
        ppc_fp = jdir / f"{burst}_joint_ppc_multi{suffix}.json"
        ppc = json.loads(ppc_fp.read_text()) if ppc_fp.exists() else None
        verdict = _gate.gate_one(burst, fit, ppc)
        rail = classify_rail(fit, burst, suffix)
        rows.append(
            {
                **verdict,
                "suffix": suffix,
                "rail_class": rail["class"],
                "rail_detail": rail,
                "beta_err": [fit["beta"]["err_minus"], fit["beta"]["err_plus"]],
                "log_evidence": fit["log_evidence"],
                "ncall": fit.get("ncall"),
            }
        )
        print(
            f"[grade] {burst}{suffix}: {verdict['final']} beta={verdict['beta']:.3f} "
            f"alpha={verdict['alpha']:.2f} rail={rail['class']}",
            flush=True,
        )
    if missing:
        print(f"[grade] MISSING fits: {missing}", flush=True)

    out = Path(__file__).parent / "beta_campaign_verdicts.json"
    out.write_text(json.dumps({"runs_root": str(RUNS), "missing": missing, "rows": rows}, indent=2))

    md = [
        "# Beta-campaign verdicts (thin-screen pass 1)",
        "",
        "| burst | model | beta | alpha | tau_1ghz | chi2 C/D | rail | final | reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cc = f"{r['chi2_chime']:.2f}" if r["chi2_chime"] is not None else "-"
        cd = f"{r['chi2_dsa']:.2f}" if r["chi2_dsa"] is not None else "-"
        md.append(
            f"| {r['burst']} | {r['suffix'].lstrip('_')} | "
            f"{r['beta']:.3f} (+{r['beta_err'][1]:.3f}/-{r['beta_err'][0]:.3f}) | "
            f"{r['alpha']:.2f} | {r['tau']:.4g} | {cc}/{cd} | {r['rail_class']} | "
            f"{r['final']} | {r['reason']} |"
        )
    if missing:
        md += ["", f"Missing fits: {', '.join(missing)}"]
    (Path(__file__).parent / "beta_campaign_verdicts.md").write_text("\n".join(md) + "\n")
    print(f"[grade] wrote {out} (+.md); {len(rows)} graded, {len(missing)} missing", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
