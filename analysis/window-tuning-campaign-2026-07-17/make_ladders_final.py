"""Per-burst ACF ladder figures on the PINNED campaign windows (config path, core-boxcar,
pre-burst off rule). Rejected subbands (m>1.2 or gamma-railed) are drawn and ANNOTATED as
rejected, not silently dropped (owner ruling #5)."""
import sys, os, json, numpy as np, matplotlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
# window_refit lives in the scintillation submodule; add it to the path repo-relatively
sys.path.insert(0, str(REPO/"scintillation"/"scint_analysis"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
import window_refit as wr
try:
    from kernel import apply_figure_style; apply_figure_style(sizes=(9,8,7))
except Exception: pass
CAMP=str(HERE/"results")
FIGDIR=HERE/"figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
FOCAL="#c1272d"; DATA="#2b2b2b"; ENV="#4877b0"; REJ="#999999"
M_PHYS=1.2
def ladder(nm, out):
    d=json.load(open(f"{CAMP}/{nm}_campaign.json"))
    w=d["windows"]; burst=w["burst_lims"]; off=w["off_lims"]
    r=wr.refit(nm, burst, off)
    cf=r["center_freqs"]; order=list(r["order"])[::-1]  # hi->lo
    a=d.get("alpha") or {}
    n=len(order)
    fig,axes=plt.subplots(n,1,figsize=(3.6,1.2*n+0.6),sharex=True)
    if n==1: axes=[axes]
    for ax,i in zip(axes,order):
        f=r["fits"][int(i)]
        if not f["ok"]:
            ax.text(0.5,0.5,f"{cf[i]:.0f} MHz — no fit",transform=ax.transAxes,ha="center",va="center",color="0.5"); ax.set_yticks([]); continue
        lp=np.asarray(f["lp"]); ap=np.asarray(f["ap"]); model=np.asarray(f["model"]); oo=np.argsort(lp)
        railed = bool(f.get("gamma_b") and f["gamma_b"]>=19.9)
        m_bad = bool(f.get("resolved") and f.get("m",0)>M_PHYS)
        # HARD reject = excluded from the alpha fit (unresolved, or m>1.2 = not point-source
        # scintillation). These are the SAME criteria the alpha fit applies, so the ladder
        # and the reported n never disagree. A railed BROAD envelope is only a caveat: the
        # NARROW gamma_n still enters the fit, so it is flagged (amber), not rejected.
        rejected = (not f["resolved"]) or m_bad
        fitcol = REJ if rejected else FOCAL
        ax.plot(lp,ap,'.',ms=2.5,color=DATA,alpha=0.5,zorder=1)
        ax.plot(lp[oo],model[oo],'-',lw=1.6,color=fitcol,zorder=3)
        if f.get("gamma_b"):
            broad=f["A_b"]/(1+(lp[oo]/f["gamma_b"])**2)+f.get("c0",0)
            ax.plot(lp[oo],broad,'--',lw=1.0,color=ENV,zorder=2)
        # status label
        if rejected:
            why=[]
            if not f["resolved"]: why.append("unresolved")
            if m_bad: why.append(f"m={f['m']:.1f}>1.2")
            status="REJECTED (excl. from $\\alpha$): "+", ".join(why)
            txtcol="0.45"; boxec=REJ
        elif railed:
            status=r"$\checkmark$ used; broad env. railed"
            txtcol="#9a6a00"; boxec="#d0a000"
        else:
            status=r"$\checkmark$ resolved, physical"
            txtcol="black"; boxec="0.85"
        gtxt=rf"$\nu_d$={f['gamma']:.3f}$\pm${f['gamma_err']:.3f} MHz"
        mtxt=rf"$m$={f['m']:.2f}   {f['model_sel']}"
        y0,y1=ax.get_ylim(); ax.set_ylim(y0,y1+0.6*(y1-y0))
        blk=f"{cf[i]:.0f} MHz\n{gtxt}\n{mtxt}\n{status}"
        ax.text(0.975,0.94,blk,transform=ax.transAxes,ha="right",va="top",linespacing=1.5,fontsize=6.8,
                color=txtcol,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=boxec,alpha=0.9))
        ax.axhline(0,color="0.7",lw=0.6,zorder=0); ax.set_xlim(0,wr.LAG_MAX)
    axes[-1].set_xlabel("frequency lag (MHz)"); axes[n//2].set_ylabel("ACF")
    al=a.get("alpha")
    if al is not None:
        atxt=rf"$\alpha$={al:+.2f}$\pm${a.get('alpha_err',0):.2f} (n={a.get('n')})"
    else:
        atxt=r"$\alpha$: n/a (insufficient resolved physical subbands)"
    fig.suptitle(f"{nm}: CHIME scintillation ACF ladder\n{atxt}",fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out,dpi=200); plt.close(fig)
    return al,a.get("n")
for nm in ["chromatica","zach","freya","hamilton"]:
    out=str(FIGDIR/f"{nm}_acf_ladder.png")
    al,nn=ladder(nm,out); print(f"{nm}: alpha={al} n={nn} -> figures/{os.path.basename(out)}")
