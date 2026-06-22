#!/usr/bin/env python
"""List every co-detected burst's joint-fit quality metrics -> why 3 of 12 pass."""
import glob, json, os
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
rows = []
for fp in sorted(glob.glob(f"{RUNS}/data/joint/*_joint_fit.json")):
    b = os.path.basename(fp).replace("_joint_fit.json", "")
    d = json.load(open(fp))
    pc = d.get("percentiles", {})
    al = pc.get("alpha", {}).get("median")
    # hunt for goodness / whiteness fields wherever they live
    flat = json.dumps(d)
    gof = d.get("goodness_of_fit") or d.get("gof") or {}
    # common metric names
    def grab(*names):
        for n in names:
            for src in (d, gof, d.get("gate", {}) or {}):
                if isinstance(src, dict) and n in src:
                    return src[n]
        return None
    rows.append({
        "burst": b,
        "alpha": round(al, 2) if al else None,
        "verdict": grab("verdict", "classification", "flag", "gate_verdict"),
        "lag1": grab("lag1", "lag1_autocorr", "temporal_lag1", "dw"),
        "chi2_C": grab("chi2_red_C", "chi2_C", "redchi_C"),
        "chi2_D": grab("chi2_red_D", "chi2_D", "redchi_D"),
        "topkeys": [k for k in d.keys() if k not in ("percentiles", "samples")],
    })
print(f"{'burst':12s} {'alpha':>6s} {'verdict':>12s} {'lag1':>10s} {'chi2_C':>8s} {'chi2_D':>8s}")
for r in rows:
    print(f"{r['burst']:12s} {str(r['alpha']):>6s} {str(r['verdict']):>12s} "
          f"{str(r['lag1']):>10s} {str(r['chi2_C']):>8s} {str(r['chi2_D']):>8s}")
print(f"\nN bursts = {len(rows)}")
print("sample topkeys:", rows[0]["topkeys"] if rows else None)
