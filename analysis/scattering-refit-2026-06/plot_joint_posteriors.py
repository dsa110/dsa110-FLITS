#!/usr/bin/env python
"""Corner plots + tau(nu) ladder from joint-fit posterior samples.

Reads <RUNS>/data/joint/<b>_joint_samples.npz (samples, weights, param_names,
alpha_bounds) written by run_joint_fit.py. Produces:
  <OUT>/<b>_corner.png      focused corner: tau_1ghz, alpha, the fit's zeta params
                            (shared zeta_1ghz/x_zeta or per-band zeta_C/D), dDM_C/D
  <OUT>/tau_nu_ladder.png   tau(nu)=tau_1ghz*nu^-alpha, all sightlines, 68% band

  python plot_joint_posteriors.py [b1 b2 ...]
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
from dynesty.utils import resample_equal

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
OUT = f"{RUNS}/data/joint"
BURSTS = sys.argv[1:] or ["johndoeII", "wilhelm", "phineas", "oran"]

LABELS = {
    "tau_1ghz": r"$\tau_{1\,\rm GHz}$ (ms)",
    "alpha": r"$\alpha$",
    "zeta_C": r"$\zeta_{\rm C}$",
    "zeta_D": r"$\zeta_{\rm D}$",
    "zeta_1ghz": r"$\zeta_{1\,\rm GHz}$",
    "x_zeta": r"$x_\zeta$",
    "delta_dm_C": r"$\delta{\rm DM}_{\rm C}$",
    "delta_dm_D": r"$\delta{\rm DM}_{\rm D}$",
}


def focus_for(names):
    # shared zeta(nu) fit carries zeta_1ghz/x_zeta; per-band fit carries zeta_C/D
    z = ["zeta_1ghz", "x_zeta"] if "zeta_1ghz" in names else ["zeta_C", "zeta_D"]
    return ["tau_1ghz", "alpha", *z, "delta_dm_C", "delta_dm_D"]


COLORS = {"johndoeII": "C0", "wilhelm": "C1", "phineas": "C2", "oran": "C3"}


def load(b):
    d = np.load(f"{OUT}/{b}_joint_samples.npz", allow_pickle=True)
    names = list(d["param_names"])
    eq = resample_equal(d["samples"], d["weights"])  # equal-weight posterior
    return names, eq, tuple(float(x) for x in d["alpha_bounds"])


def corner_one(b):
    names, eq, abnd = load(b)
    focus = focus_for(names)
    idx = [names.index(n) for n in focus]
    sub = eq[:, idx]
    K = len(focus)
    fig = corner.corner(
        sub,
        labels=[LABELS[n] for n in focus],
        show_titles=True,
        title_fmt=".2f",
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
    )
    axd = fig.axes[1 * K + 1]  # alpha diagonal panel (FOCUS idx 1)
    for e in abnd:
        axd.axvline(e, color="r", ls=":", lw=1)  # alpha prior edges
    a_med = float(np.median(sub[:, 1]))
    fig.suptitle(f"{b}: joint corner  (alpha~U{abnd}, median={a_med:.2f})", fontsize=12)
    fp = f"{OUT}/{b}_corner.png"
    fig.savefig(fp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fp}  alpha_med={a_med:.2f}")
    return names, eq


def tau_ladder(bursts_data):
    nu = np.linspace(0.40, 1.55, 200)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for b, (names, eq) in bursts_data.items():
        it, ia = names.index("tau_1ghz"), names.index("alpha")
        n = min(4000, eq.shape[0])
        sel = rng.choice(eq.shape[0], n, replace=False)
        tau1, al = eq[sel, it], eq[sel, ia]
        T = tau1[:, None] * nu[None, :] ** (-al[:, None])  # (n, nu), ms
        lo, med, hi = np.percentile(T, [16, 50, 84], axis=0)
        c = COLORS.get(b)
        ax.plot(nu, med, color=c, lw=2, label=f"{b}  (alpha={np.median(al):.2f})")
        ax.fill_between(nu, lo, hi, color=c, alpha=0.18)
    ax.axvspan(0.40, 0.80, color="gray", alpha=0.08)
    ax.axvspan(1.28, 1.53, color="gray", alpha=0.12)
    xt = ax.get_xaxis_transform()
    ax.text(0.60, 0.95, "CHIME", ha="center", fontsize=9, color="0.4", transform=xt)
    ax.text(1.40, 0.95, "DSA", ha="center", fontsize=9, color="0.4", transform=xt)
    ax.set_yscale("log")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(r"$\tau(\nu)$ (ms)")
    ax.set_title(r"Joint-fit $\tau(\nu)=\tau_{1\,\rm GHz}\,\nu^{-\alpha}$ ladder (68% band)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fp = f"{OUT}/tau_nu_ladder.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fp}")


def main():
    data = {}
    for b in BURSTS:
        f = f"{OUT}/{b}_joint_samples.npz"
        if not os.path.exists(f):
            print(f"  MISSING {f} — skip")
            continue
        names, eq = corner_one(b)
        data[b] = (names, eq)
    if data:
        tau_ladder(data)


if __name__ == "__main__":
    main()
