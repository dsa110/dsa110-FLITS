"""Two-band (CHIME 400-800 MHz + DSA ~1300-1470 MHz) scintillation scaling analysis.

Dual-track per burst (owner directive 2026-07-17):
  (a) PER-BAND alpha + two-screen decomposition  -- physically defensible primary
  (b) FORCED single-screen joint 400-1530 MHz alpha + its consistency statistic
      -- the tested-and-rejected single-screen hypothesis.

One-screen gate: a single power law across the 3.8x lever assumes ONE dominant screen.
Test = does the joint fit reproduce BOTH (i) the DSA/CHIME dnu ratio at band centers and
(ii) the single-screen product tau*dnu_d = C1/2pi, within propagated errors. The C-constants
and verdict come from the committed flits.batch.analysis_logic.check_tau_deltanu_consistency
(single source of truth); we do not reimplement them.

CHIME points: campaign_final (config-path rerun, grid-reg ON, core-boxcar primary,
pre-burst off rule). DSA points: July-7 dsa_lorentzian_components.csv (provisional except
oran). Weighted log-space power law reused from run_dsa_lorentzian_fits._fit_gamma_power_law.
"""
import json, math, csv
import numpy as np

NU_REF = 1400.0

def fit_power_law(pts, nu_ref=NU_REF):
    """pts: list of (nu_mhz, dnu_mhz, dnu_err_mhz). Weighted LS in log space.
    Mirrors run_dsa_lorentzian_fits._fit_gamma_power_law exactly."""
    u = [(nu,g,ge) for (nu,g,ge) in pts if np.isfinite(ge) and ge>0 and g>0]
    if len(u) < 2:
        return None
    nu = np.array([p[0] for p in u]); g = np.array([p[1] for p in u]); ge = np.array([p[2] for p in u])
    x = np.log(nu/nu_ref); y = np.log(g); sy = ge/g
    design = np.column_stack((np.ones_like(x), x))
    prec = 1.0/np.square(sy)
    normal = design.T @ (prec[:,None]*design)
    try: cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError: return None
    coef = cov @ (design.T @ (prec*y))
    # goodness of fit
    yhat = design @ coef
    chi2 = float(np.sum(prec*(y-yhat)**2)); dof = len(u)-2
    return dict(alpha=float(coef[1]), alpha_err=float(math.sqrt(cov[1,1])),
                gamma_ref=float(math.exp(coef[0])), nu_ref=nu_ref,
                n=len(u), chi2=chi2, dof=dof,
                redchi=(chi2/dof if dof>0 else float('nan')))

def load_chime(campaign_jsonl):
    """CHIME ladder points per burst from the campaign: resolved, physical (m<=1.2) subbands."""
    out={}
    for line in open(campaign_jsonl):
        d=json.loads(line); nm=d["name"]; pts=[]
        for s in d.get("subbands",[]):
            if s.get("ok") and s.get("resolved") and s.get("m",9)<=1.2:
                ge=s.get("gamma_err") or s.get("gamma_scintle_err")
                if ge and ge>0:
                    pts.append((float(s["center_mhz"]), float(s["gamma"]), float(ge)))
        out[nm]=dict(points=pts, alpha=(d.get("alpha") or {}))
    return out

def load_dsa(csv_path):
    """DSA gamma-track-1 usable components per burst (dnu, freq, err)."""
    out={}
    for row in csv.DictReader(open(csv_path)):
        nm=row["burst"]
        # gamma track 1 = the narrow scintillation component; usable = no quality flags
        flags=row.get("quality_flags","").strip()
        comp=int(row["component"])
        dnu=float(row["dnu_mhz"]); dnu_err=float(row["dnu_err_mhz"])
        # narrow component only (component 1 is the primary narrow scintle per the catalog)
        if comp==1 and not flags and dnu>0 and dnu_err>0:
            out.setdefault(nm,[]).append((float(row["center_freq_mhz"]), dnu, dnu_err))
    return out

if __name__=="__main__":
    import sys
    chime=load_chime("/Users/jakobfaber/Developer/scratch/campaign_final/campaign_results.jsonl")
    dsa=load_dsa("/Users/jakobfaber/Developer/repos/github.com/jakobtfaber/dsa110-FLITS/analysis/scintillation-dsa-lorentzian-2026-07-07/results/dsa_lorentzian_components.csv")
    triad=["zach","chromatica","freya","hamilton"]
    report={}
    for nm in triad:
        cp=chime.get(nm,{}).get("points",[]); dp=dsa.get(nm,[])
        f_chime=fit_power_law(cp); f_dsa=fit_power_law(dp)
        f_joint=fit_power_law(cp+dp)
        report[nm]=dict(n_chime=len(cp), n_dsa=len(dp),
                        chime=f_chime, dsa=f_dsa, joint=f_joint,
                        chime_pts=cp, dsa_pts=dp)
        def af(f): return f"{f['alpha']:+.2f}+-{f['alpha_err']:.2f} (n{f['n']},rc{f['redchi']:.1f})" if f else "n/a"
        print(f"{nm:11} CHIME[{len(cp)}] {af(f_chime):26} DSA[{len(dp)}] {af(f_dsa):26} JOINT {af(f_joint)}")
    json.dump(report, open("/Users/jakobfaber/Developer/scratch/two_band_tracks.json","w"), indent=2, default=float)
    print("WROTE two_band_tracks.json")
