#!/usr/bin/env python
"""Dipole-mask discriminant post-process (task #13).

For casey (mask C-band peak) and wilhelm (mask D-band peak), builds the discriminant
team-lead specified: per burst, the free-alpha wedge alpha (median + 90% CI) AND the
masked-data wedge Bayes factor delta-lnZ = lnZ(free-alpha) - lnZ(tied-alpha) computed
ON THE SAME (masked) data vector, for baseline / hard-mask / down-weight.

Why the delta-lnZ column matters (team-lead): alpha could sit sub-4 with an inflated CI
while the wedge's EVIDENCE support collapses -- the delta-lnZ catches that, the alpha
median alone does not. Valid because free-alpha and tied-alpha share the data vector at
each mask level (tied is a submanifold of free). masked-vs-UNMASKED lnZ is NOT comparable
(different data vector) and is never differenced here.

  alpha -> 4 AND wedge delta-lnZ -> ~0 under mask  => PEAK-SHAPE-DRIVEN (dipole systematic).
  alpha stays < 4 with a surviving wedge delta-lnZ  => DISTRIBUTED chromatic scaling.

Baseline pair: free = ab_<burst>_relaxalpha_joint_fit ; tied = <burst>_joint_fit (production).
Masked pair : free = ab_<burst>_dipolemask_<band>_<mode> ; tied = ..._<mode>_tied.

Handles missing inputs gracefully (reports which are present) so it runs incrementally
as the 8 fits (139-146) land.

  python dipolemask_postprocess.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("FLITS_RUNS", "/home/ubuntu/flits-runs")
JOINT = f"{RUNS}/data/joint"
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dipolemask_wedge.png")

BURSTS = ["casey", "wilhelm"]
BAND = {"casey": "C", "wilhelm": "D"}
NU4 = 4.0
VARIANTS = [("baseline", "baseline", "#444444"),
            ("hard", "hard", "#c0392b"),
            ("soft", "soft", "#2c6fbb")]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _free_json(burst, mode):
    if mode == "baseline":
        return f"{JOINT}/ab_{burst}_relaxalpha_joint_fit.json"
    return f"{JOINT}/ab_{burst}_dipolemask_{BAND[burst]}_{mode}_joint_fit.json"


def _tied_json(burst, mode):
    if mode == "baseline":
        return f"{JOINT}/{burst}_joint_fit.json"
    return f"{JOINT}/ab_{burst}_dipolemask_{BAND[burst]}_{mode}_tied_joint_fit.json"


def _free_npz(burst, mode):
    if mode == "baseline":
        return f"{JOINT}/ab_{burst}_relaxalpha_joint_samples.npz"
    return f"{JOINT}/ab_{burst}_dipolemask_{BAND[burst]}_{mode}_joint_samples.npz"


def _samples_alpha(npz_path):
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=True)
    names = list(z["param_names"])
    if "alpha" not in names:
        return None
    return z["samples"][:, names.index("alpha")], z["weights"]


def _tied_npz(burst, mode):
    if mode == "baseline":
        return f"{JOINT}/{burst}_joint_samples.npz"
    return f"{JOINT}/ab_{burst}_dipolemask_{BAND[burst]}_{mode}_tied_joint_samples.npz"


def _beta_ci(npz_path):
    """(median, q05, q95) of the TIED-model beta posterior, or None."""
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=True)
    names = list(z["param_names"])
    if "beta" not in names:
        return None
    s = z["samples"][:, names.index("beta")]
    w = z["weights"]
    o = np.argsort(s)
    c = np.cumsum(w[o])
    c /= c[-1]
    return (float(np.interp(0.5, c, s[o])), float(np.interp(0.05, c, s[o])),
            float(np.interp(0.95, c, s[o])))


def main():
    rows = []
    fig, axes = plt.subplots(1, len(BURSTS), figsize=(11.5, 4.4), squeeze=False)
    for j, burst in enumerate(BURSTS):
        ax = axes[0, j]
        for label, mode, color in VARIANTS:
            df = _load(_free_json(burst, mode))
            dt = _load(_tied_json(burst, mode))
            if df is None:
                rows.append((burst, label, None))
                continue
            a = df["alpha"]
            am, alo, ahi = a["median"], a["err_minus"], a["err_plus"]
            q05, q95 = a.get("q05"), a.get("q95")
            dlnz = (df["log_evidence"] - dt["log_evidence"]) if dt is not None else None
            frac = df.get("mask", {}).get("signal_frac_masked")
            tbeta = _beta_ci(_tied_npz(burst, mode))  # masked-tied beta posterior (median,q05,q95)
            rows.append((burst, label, dict(am=am, alo=alo, ahi=ahi, q05=q05, q95=q95,
                                            beta=df["beta"]["median"], tau=df["tau_1ghz"]["median"],
                                            frac=frac, dlnz=dlnz, tied_pending=(dt is None),
                                            tbeta=tbeta)))
            sa = _samples_alpha(_free_npz(burst, mode))
            if sa is not None:
                x, w = sa
                h, edges = np.histogram(x, bins=90, range=(1.8, 6.2), weights=w, density=True)
                ctr = 0.5 * (edges[:-1] + edges[1:])
                lbl = f"{label}: α={am:.2f}₋{alo:.2f}₊{ahi:.2f}"
                if dlnz is not None:
                    lbl += f"  ΔlnZ={dlnz:+.0f}"
                ax.plot(ctr, h, color=color, lw=1.6, label=lbl)
                ax.fill_between(ctr, h, color=color, alpha=0.12)
        ax.axvline(NU4, color="k", ls="--", lw=1.0)
        ax.set_title(f"{burst}  (mask {BAND[burst]}-band peak)", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"scattering index $\alpha$")
        ax.set_ylabel("posterior density")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.margins(x=0)
        yl = ax.get_ylim()[1]
        ax.text(NU4 + 0.05, yl * 0.55, r"$\nu^{-4}$", fontsize=9, color="0.3")
    fig.suptitle("Dipole-mask wedge discriminant: does the sub-4 α survive excising the peak dipole? "
                 "(α→4 & ΔlnZ→0 ⇒ peak-shape-driven; α<4 with surviving ΔlnZ ⇒ distributed)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=140)

    print(f"\n{'burst':8} {'variant':9} {'alpha (med -/+)':21} {'a 90%CI':14} "
          f"{'wedge dlnZ':11} {'tied-beta [90%CI]':22} {'fmask':6} verdict")
    print("-" * 116)
    for burst, label, r in rows:
        if r is None:
            print(f"{burst:8} {label:9} {'(free pending)':21}")
            continue
        ci = f"[{r['q05']:.2f},{r['q95']:.2f}]" if r["q05"] is not None else "-"
        frac = f"{r['frac']:.2f}" if r["frac"] is not None else "-"
        dz = "(tied pend)" if r["tied_pending"] else f"{r['dlnz']:+.1f}"
        if r.get("tbeta") is not None:
            bm, blo, bhi = r["tbeta"]
            ceil = "*RAIL* " if bhi >= 3.985 else "off-ceil "  # includes 3.99?
            tb = f"{ceil}{bm:.3f}[{blo:.2f},{bhi:.2f}]"
        else:
            tb = "-"
        if label == "baseline":
            verdict = "sub-4 wedge (contaminated)"
        elif r["tied_pending"] or r["q05"] is None:
            verdict = "..."
        elif r["q05"] >= 3.90 and r["dlnz"] is not None and r["dlnz"] < 10:
            verdict = "PEAK-SHAPE-DRIVEN (a->4, dlnZ->0)"
        elif r["am"] < 3.90 and (r["dlnz"] is None or r["dlnz"] > 10):
            verdict = "DISTRIBUTED chromatic (survives)"
        else:
            verdict = "mixed/straddles"
        print(f"{burst:8} {label:9} {r['am']:.3f} -{r['alo']:.3f}/+{r['ahi']:.3f}    "
              f"{ci:14} {dz:11} {tb:22} {frac:6} {verdict}")
    print("\n  tied-beta column: masked-data TIED-model beta posterior. '*RAIL*' = 90% CI still "
          "reaches the 3.99 ceiling (rail robust); 'off-ceil' = CI excludes it (rail peak-associated).")
    print(f"  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
