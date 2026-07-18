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
                # Combined error: curve_fit statistical + finite-scintle (1/sqrt(N_ISS)) in
                # quadrature, IDENTICAL to window_refit's alpha weighting (R5-1). Using only
                # one term would make the two-band CHIME slope disagree with the campaign JSON.
                import math
                gstat=s.get("gamma_err") or 0.0; gscint=s.get("gamma_scintle_err") or 0.0
                ge=math.hypot(float(gstat), float(gscint))
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

DNU_REF_MHZ = 1400.0  # match FREQ_DSA / DNU_REF in analysis_logic (tau scaled to same nu)

def dsa_dnu_at_ref(dsa_pts, alpha, nu_ref=DNU_REF_MHZ):
    """Inverse-variance weighted-mean DSA Delta-nu_d scaled to nu_ref, matching the
    committed beta_campaign/two_screen.dsa_delta_nu convention (each subband scaled by
    (nu_ref/nu)^alpha, then weighted by 1/scaled_err^2). Uses the SAME plotted July-7
    components so the annotated tau*dnu_d and the figure points share provenance (B2)."""
    if not dsa_pts:
        return None
    num=den=0.0
    for nu,g,ge in dsa_pts:
        scale=(nu_ref/nu)**alpha
        sg=g*scale; se=ge*scale
        if se<=0: continue
        w=1.0/se**2; num+=w*sg; den+=w
    if den<=0: return None
    return dict(delta_nu_dc=num/den, delta_nu_dc_err=1.0/math.sqrt(den))

def tau_dnu_consistency(taus_json, tracks, nu_ref=DNU_REF_MHZ):
    """Programmatic tau*dnu_d = C1/2pi test at the PINNED dnu_d values, via the committed
    flits.batch.analysis_logic.check_tau_deltanu_consistency (single source of truth for
    the C-constants and the same_screen/different_screens verdict). tau comes from the
    committed two_screen_consistency.json (beta-campaign joint scattering fits); dnu_d is
    recomputed from this campaign's plotted DSA components. Returns per-burst dict."""
    import sys
    from pathlib import Path
    repo=str(Path(__file__).resolve().parents[2])   # repo root, so `import flits` resolves
    if repo not in sys.path: sys.path.insert(0, repo)
    import pandas as pd
    from flits.batch.analysis_logic import check_tau_deltanu_consistency
    taus={r["burst_name"]: r for r in json.load(open(taus_json)).get("rows",[]) if r.get("telescope")=="dsa"}
    rows=[]; keys=[]
    for nm,t in tracks.items():
        tj=taus.get(nm)
        if tj is None: continue
        alpha=tj.get("alpha") or 4.0
        dnu=dsa_dnu_at_ref(t.get("dsa_pts") or [], alpha, nu_ref)
        if dnu is None: continue
        rows.append(dict(burst_name=nm, telescope="dsa",
                         tau_1ghz=tj["tau_1ghz"], tau_1ghz_err=tj.get("tau_1ghz_err"),
                         delta_nu_dc=dnu["delta_nu_dc"], delta_nu_dc_err=dnu["delta_nu_dc_err"],
                         alpha=alpha)); keys.append(nm)
    if not rows: return {}
    res=check_tau_deltanu_consistency(pd.DataFrame(rows))
    out={}
    for nm,r in zip(keys,res):
        out[nm]=dict(tau_1ghz_ms=float(r.tau_1ghz_ms),
                     dsa_dnu_ref_mhz=float(r.delta_nu_mhz),
                     tau_delta_nu_product=(None if r.tau_delta_nu_product is None else float(r.tau_delta_nu_product)),
                     tau_delta_nu_product_err=(None if r.tau_delta_nu_product_err is None else float(r.tau_delta_nu_product_err)),
                     screen_verdict=r.screen_verdict, interpretation=r.interpretation,
                     nu_ref_mhz=nu_ref)
    return out

if __name__=="__main__":
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent                 # analysis/window-tuning-campaign-2026-07-17/
    REPO = HERE.parents[1]                                  # repo root (analysis/ -> root)
    chime=load_chime(str(HERE/"results"/"campaign_results.jsonl"))
    dsa=load_dsa(str(REPO/"analysis"/"scintillation-dsa-lorentzian-2026-07-07"/"results"/"dsa_lorentzian_components.csv"))
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
    # B2: recompute tau*dnu_d at the pinned dnu_d values via the committed consistency test.
    taus_json = REPO/"analysis"/"beta_campaign"/"two_screen_consistency.json"
    try:
        cons = tau_dnu_consistency(str(taus_json), report)
        for nm,c in cons.items():
            report[nm]["tau_delta_nu"] = c
            p=c["tau_delta_nu_product"]
            print(f"  {nm:11} tau*dnu_d={p:.1f} -> {c['screen_verdict']}" if p is not None
                  else f"  {nm:11} tau*dnu_d=n/a")
    except Exception as e:
        print(f"  [tau*dnu_d recompute skipped: {e}]")
    json.dump(report, open(str(HERE/"results"/"two_band_tracks.json"),"w"), indent=2, default=float)
    print("WROTE results/two_band_tracks.json")
