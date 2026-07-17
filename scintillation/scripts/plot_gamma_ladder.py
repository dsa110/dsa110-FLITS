"""Sample-wide gamma(nu) ladder: every subband fit from the standard + _hi campaign
runs on one panel per burst, so the whole excavation is vettable at a glance.

Markers: filled = resolved (all gates incl. shape); open = converged but gated.
Red edge = m > 1.2 (envelope-contaminated amplitude). Dotted guide: nu^4 through
the highest-frequency resolved point. Error bars: sigma_fit (+ sigma_win shaded
band where available).

Usage: python plot_gamma_ladder.py <std_dir> <hi_dir> <out.png>
"""
from __future__ import annotations
import json, sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BURSTS = ["casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
          "mahi", "oran", "phineas", "whitney", "wilhelm", "zach"]


def _load(d, name):
    p = os.path.join(d, f"{name}_campaign.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _points(rec):
    out = []
    if not rec:
        return out
    for b in rec.get("subbands", []):
        if not b.get("ok") or b.get("gamma") is None:
            continue
        out.append(dict(f=b["center_mhz"], g=b["gamma"], ge=b["gamma_err"],
                        m=b.get("m", 0), res=bool(b.get("resolved")),
                        sw=b.get("gamma_win_sys")))
    return out


def main(std_dir, hi_dir, outpath):
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True)
    for ax, name in zip(axes.ravel(), BURSTS):
        for rec, color, lab in ((_load(std_dir, name), "#1f77b4", "std 24.4 kHz"),
                                (_load(hi_dir, name + "_hi"), "#d62728", "_hi")):
            pts = _points(rec)
            for p in pts:
                filled = p["res"]
                mbad = p["m"] > 1.2
                ax.errorbar(p["f"], p["g"], yerr=p["ge"], fmt="o", ms=7 if filled else 5,
                            mfc=color if filled else "none", mec="#c0392b" if mbad else color,
                            ecolor=color, elinewidth=0.8, capsize=2, mew=1.6 if mbad else 1.0,
                            alpha=0.95 if filled else 0.45)
                if filled and p["sw"]:
                    ax.plot([p["f"], p["f"]], [max(p["g"] - p["sw"], 1e-4), p["g"] + p["sw"]],
                            color=color, lw=4, alpha=0.18)
            resolved = [p for p in pts if p["res"] and p["m"] <= 1.2]
            if resolved and color == "#d62728":
                top = max(resolved, key=lambda p: p["f"])
                fr = np.linspace(600, 800, 50)
                ax.plot(fr, top["g"] * (fr / top["f"]) ** 4, ls=":", color="k", lw=0.9, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 25)
        ax.set_title(name, fontsize=11)
        ax.axhline(2 * 0.0244, color="#7f8c8d", lw=0.6, ls="--", alpha=0.7)  # std resolve floor
        ax.grid(alpha=0.2, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\gamma$ (MHz)")
    fig.suptitle("CHIME per-subband decorrelation bandwidths — matched estimator, all gates\n"
                 "filled=resolved, open=gated; red edge: m>1.2; blue=standard product, red=_hi; "
                 "dotted: $\\nu^4$ through top resolved _hi point; dashed grey: 24.4 kHz resolve floor",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print("wrote", outpath)


if __name__ == "__main__":
    main(*sys.argv[1:4])
