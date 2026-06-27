#!/usr/bin/env python
"""Canonical all-exp tau(nu) ladder (ADR-0003).

tau(nu) = tau_1ghz * nu^-alpha for the seven well-constrained bursts whose
all-exp single-exp-PBF fits the ALLEXP_PBF_RUN.md verdict marks publishable
(|Delta-alpha_exp-mixed| <= 0.1). tau_1ghz/alpha are shared scalars, present in
every joint posterior regardless of component count, so the ladder needs no
model rebuild -- just the canonical all-exp .npz per burst.

Heterogeneous per-burst model: sharedzeta (freya/casey/chromatica/wilhelm) or
component (oran C2D1, phineas C3D3). whitney (C2D2, fit locally) is absent from
HPCC -> rendered when its local .npz is located.

  python _tau_ladder_allexp.py [b1 b2 ...]
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dynesty.utils import resample_equal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # _figsave
from _figsave import save_fig

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
OUT = f"{RUNS}/data/joint"
FIG_OUT = os.environ.get("DSA_FIGS", OUT)

# canonical all-exp posterior .npz per burst (ALLEXP_PBF_RUN.md per-burst model)
NPZ = {
    "freya": "freya_joint_samples_sharedzeta_pbf-exp-exp.npz",
    "casey": "casey_joint_samples_sharedzeta_pbf-exp-exp.npz",
    "chromatica": "chromatica_joint_samples_sharedzeta_pbf-exp-exp.npz",
    "wilhelm": "wilhelm_joint_samples_sharedzeta_pbf-exp-exp.npz",
    "oran": "oran_joint_samples_C2D1_pbf-exp-exp.npz",
    "phineas": "phineas_joint_samples_C3D3_pbf-exp-exp.npz",
    "whitney_fine": "whitney_fine_joint_samples_C2D2_pbf-exp-exp.npz",  # fit locally, pushed to HPCC
}
BURSTS = sys.argv[1:] or list(NPZ)
COLORS = dict(zip(NPZ, plt.cm.turbo(np.linspace(0.05, 0.95, len(NPZ)))))


def load(b):
    d = np.load(f"{OUT}/{NPZ[b]}", allow_pickle=True)
    names = list(d["param_names"])
    eq = resample_equal(d["samples"], d["weights"])  # equal-weight posterior
    return names.index("tau_1ghz"), names.index("alpha"), eq


def main():
    nu = np.linspace(0.40, 1.55, 200)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for b in BURSTS:
        f = f"{OUT}/{NPZ[b]}"
        if not os.path.exists(f):
            print(f"  MISSING {f} -- skip")
            continue
        it, ia, eq = load(b)
        n = min(4000, eq.shape[0])
        sel = rng.choice(eq.shape[0], n, replace=False)
        tau1, al = eq[sel, it], eq[sel, ia]
        T = tau1[:, None] * nu[None, :] ** (-al[:, None])  # (n, nu), ms
        lo, med, hi = np.percentile(T, [16, 50, 84], axis=0)
        c = COLORS[b]
        ax.plot(nu, med, color=c, lw=2, label=f"{b}  ($\\alpha$={np.median(al):.2f})")
        ax.fill_between(nu, lo, hi, color=c, alpha=0.15)
    ax.axvspan(0.40, 0.80, color="gray", alpha=0.08)
    ax.axvspan(1.28, 1.53, color="gray", alpha=0.12)
    xt = ax.get_xaxis_transform()
    ax.text(0.60, 0.96, "CHIME", ha="center", fontsize=9, color="0.4", transform=xt)
    ax.text(1.40, 0.96, "DSA", ha="center", fontsize=9, color="0.4", transform=xt)
    ax.set_yscale("log")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(r"$\tau(\nu)$ (ms)")
    ax.set_title(
        r"All-exp joint-fit $\tau(\nu)=\tau_{1\,\rm GHz}\,\nu^{-\alpha}$ ladder (68% band)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fp = save_fig(fig, f"{FIG_OUT}/tau_nu_ladder_allexp", dpi=120)
    plt.close(fig)
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
