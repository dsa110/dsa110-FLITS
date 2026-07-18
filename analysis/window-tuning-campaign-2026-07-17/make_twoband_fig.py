"""Headline two-band scintillation figure: per-band alpha + two-screen decomposition.
One panel per triad burst; CHIME (400-800) + DSA (~1320-1470) Delta-nu_d points; per-band
power-law overlays; forced single-screen joint fit drawn as the REJECTED hypothesis
(dashed grey) annotated with tau*dnu_d and the different_screens verdict."""
import json, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from kernel import apply_figure_style; apply_figure_style(sizes=(10,9,8))
except Exception: pass

HERE=Path(__file__).resolve().parent
TR=json.load(open(str(HERE/"results"/"two_band_tracks.json")))
# tau*dnu_d verdicts come from the tracks JSON, recomputed at the PINNED dnu_d values via
# the committed check_tau_deltanu_consistency (two_band_joint.tau_dnu_consistency) - same
# provenance as the plotted DSA points (B2).
def taudnu(nm):
    c=(TR.get(nm) or {}).get("tau_delta_nu") or {}
    return c.get("tau_delta_nu_product"), c.get("screen_verdict")
CH="#c1272d"; DS="#2166ac"; JT="#888888"
bursts=["zach","chromatica","freya"]  # hamilton diagnostic-only, excluded from headline
fig,axes=plt.subplots(1,len(bursts),figsize=(3.1*len(bursts),3.4),sharey=True)
for ax,nm in zip(axes,bursts):
    d=TR[nm]
    cp=d["chime_pts"]; dp=d["dsa_pts"]
    for pts,c,lab in [(cp,CH,"CHIME 0.4-0.8 GHz"),(dp,DS,"DSA ~1.4 GHz")]:
        if pts:
            nu=np.array([p[0] for p in pts]); g=np.array([p[1] for p in pts]); ge=np.array([p[2] for p in pts])
            ax.errorbar(nu,g,yerr=ge,fmt='o',ms=5,color=c,capsize=2,label=lab,zorder=3)
    nus=np.logspace(np.log10(380),np.log10(1550),100)
    # per-band overlays
    for f,c in [(d["chime"],CH),(d["dsa"],DS)]:
        if f: ax.plot(nus,f["gamma_ref"]*(nus/f["nu_ref"])**f["alpha"],'-',color=c,lw=1.5,zorder=2)
    # forced joint = rejected hypothesis
    fj=d["joint"]
    if fj: ax.plot(nus,fj["gamma_ref"]*(nus/fj["nu_ref"])**fj["alpha"],'--',color=JT,lw=1.3,zorder=1,
                   label=f"forced joint (rejected)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("frequency (MHz)")
    # per-band alpha errors are inflated by sqrt(max(1,redchi)) so a poor intra-band fit
    # (e.g. chromatica DSA redchi~34) is not quoted with a deceptively tight error (M3).
    def aerr(f): return f["alpha_err"]*np.sqrt(max(1.0, f.get("redchi",1.0)))
    # annotations
    fc=d["chime"]; fd=d["dsa"]
    marginal = (d.get("n_chime",0) < 2) or (d.get("n_dsa",0) < 2)
    lines=[f"{nm}" + ("  (marginal: 1 pt/band)" if marginal else "")]
    if fc: lines.append(rf"$\alpha_{{\rm CHIME}}$={fc['alpha']:+.2f}$\pm${aerr(fc):.2f}")
    else: lines.append(rf"$\alpha_{{\rm CHIME}}$: {d.get('n_chime',0)} pt")
    if fd: lines.append(rf"$\alpha_{{\rm DSA}}$={fd['alpha']:+.2f}$\pm${aerr(fd):.2f}")
    else: lines.append(rf"$\alpha_{{\rm DSA}}$: {d.get('n_dsa',0)} pt")
    td,verdict=taudnu(nm)
    if td is not None:
        lines.append(rf"$\tau\!\cdot\!\Delta\nu_d$={td:.0f}")
        lines.append((verdict or "").replace("_"," "))
    ax.text(0.04,0.04,"\n".join(lines),transform=ax.transAxes,ha="left",va="bottom",fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.8",alpha=0.9))
    ax.set_title(nm,fontsize=9)
axes[0].set_ylabel(r"$\Delta\nu_d$ (MHz)")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,1.06))
fig.suptitle("Two-band scintillation scaling: single screen rejected across the triad",y=1.13,fontsize=10)
fig.tight_layout()
OUT=HERE/"figures"/"twoband_scint_summary.png"; OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(str(OUT),dpi=200,bbox_inches="tight")
print(f"saved {OUT.relative_to(HERE)}")
for nm in bursts:
    d=TR[nm]; td,verdict=taudnu(nm)
    print(nm,"CHIME",d["chime"]["alpha"] if d["chime"] else None,"DSA",d["dsa"]["alpha"] if d["dsa"] else None,
          "joint",d["joint"]["alpha"] if d["joint"] else None,"taudnu",td,verdict)
