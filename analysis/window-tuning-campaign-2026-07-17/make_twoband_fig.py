"""Headline two-band scintillation figure: per-band alpha + two-screen decomposition.
One panel per triad burst; CHIME (400-800) + DSA (~1320-1470) Delta-nu_d points; per-band
power-law overlays; forced single-screen joint fit drawn as the REJECTED hypothesis
(dashed grey) annotated with tau*dnu_d and the different_screens verdict."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from kernel import apply_figure_style; apply_figure_style(sizes=(10,9,8))
except Exception: pass

TR=json.load(open("/Users/jakobfaber/Developer/scratch/two_band_tracks.json"))
# tau*dnu_d verdicts from committed two_screen_consistency.md (authoritative)
TAUDNU={"chromatica":93.9,"zach":23.9,"freya":2.6,"hamilton":1.64}
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
    # annotations
    fc=d["chime"]; fd=d["dsa"]
    lines=[f"{nm}"]
    if fc: lines.append(rf"$\alpha_{{\rm CHIME}}$={fc['alpha']:+.2f}$\pm${fc['alpha_err']:.2f}")
    else: lines.append(r"$\alpha_{\rm CHIME}$: 1 pt")
    if fd: lines.append(rf"$\alpha_{{\rm DSA}}$={fd['alpha']:+.2f}$\pm${fd['alpha_err']:.2f}")
    else: lines.append(r"$\alpha_{\rm DSA}$: 1 pt")
    td=TAUDNU.get(nm)
    lines.append(rf"$\tau\!\cdot\!\Delta\nu_d$={td:g}")
    lines.append("different screens" if td>2 else "same screen")
    ax.text(0.04,0.04,"\n".join(lines),transform=ax.transAxes,ha="left",va="bottom",fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.8",alpha=0.9))
    ax.set_title(nm,fontsize=9)
axes[0].set_ylabel(r"$\Delta\nu_d$ (MHz)")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="upper center",ncol=4,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,1.06))
fig.suptitle("Two-band scintillation scaling: single screen rejected across the triad",y=1.13,fontsize=10)
fig.tight_layout()
fig.savefig("/Users/jakobfaber/Developer/scratch/twoband_scint_summary.png",dpi=200,bbox_inches="tight")
print("saved twoband_scint_summary.png")
for nm in bursts:
    d=TR[nm]; print(nm,"CHIME",d["chime"]["alpha"] if d["chime"] else None,"DSA",d["dsa"]["alpha"] if d["dsa"] else None,"joint",d["joint"]["alpha"] if d["joint"] else None,"taudnu",TAUDNU[nm])
