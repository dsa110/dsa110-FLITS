#!/usr/bin/env python
"""Multi-component joint-fit model vs data overlay (gain-marginal).

Reconstructs the fitted N-component model per band from the saved MEDIAN params
using the SAME kernel construction (model(.,"M3")) and the SAME ridge gain solve
as the canonical _gain_marginal_multi_band, at that routine's profiled s2.
NOTE: the per-channel per-component gains absorb the burst spectrum + scint, so
the freq-collapsed profile matches data by construction -- the figure shows
component placement and the shared scattering tail, not an absolute misfit test.

  python model_overlay.py <burst> <tag>     # tag like C2D3, C4D1
"""

import json
import os
import re
import sys
from dataclasses import replace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, RUNS)
sys.path.insert(0, f"{REPO}/scattering")
from run_joint_fit import prepare  # identical band prep as the fit
from scat_analysis.burstfit import FRBParams
from scat_analysis.burstfit_joint import _gain_marginal_multi_band

burst, tag = sys.argv[1], sys.argv[2]
mt = re.match(r"C(\d+)D(\d+)", tag)
nC, nD = int(mt.group(1)), int(mt.group(2))
J = json.load(open(f"{RUNS}/data/joint/{burst}_joint_fit_{tag}.json"))
P = {k: v["median"] for k, v in J["percentiles"].items()}
tau, alpha = P["tau_1ghz"], P["alpha"]


def band_params(prefix, n, ddm):
    return [
        FRBParams(
            c0=1.0,
            t0=P[f"t0_{prefix}{i}"],
            gamma=0.0,
            zeta=P[f"zeta_{prefix}{i}"],
            tau_1ghz=tau,
            alpha=alpha,
            delta_dm=ddm,
        )
        for i in range(1, n + 1)
    ]


def solve_band(model, ps):
    valid = model.valid
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), "M3", freq_subset=valid) for p in ps])
    d = model.data[valid]
    sig = np.clip(model.noise_std[valid], 1e-9, None)
    var = sig**2
    b = np.einsum("nft,ft->fn", Ks, d)
    M = np.einsum("nft,mft->fnm", Ks, Ks)
    _, diag = _gain_marginal_multi_band(model, ps, ["M3"] * len(ps), s2=None)  # profiled s2
    s2 = diag["s2"]
    N = Ks.shape[0]
    A = M + (var / s2)[:, None, None] * np.eye(N)[None]
    g = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (F, N) MAP gains
    pred = np.einsum("fn,nft->ft", g, Ks)
    comps = g.T[:, :, None] * Ks  # (N, F, T)
    resid = d - pred
    chi2 = float(np.sum((resid / sig[:, None]) ** 2) / resid.size)
    return dict(d=d, pred=pred, comps=comps, t=model.time, sig=sig, chi2=chi2, N=N, s2=s2)


RC = solve_band(
    *[
        prepare(f"{RUNS}/configs/{burst}_chime_run.yaml", f"{burst}_chime", f"{RUNS}/data/joint")[
            0
        ],
        band_params("C", nC, P.get("delta_dm_C", 0.0)),
    ]
)
RD = solve_band(
    *[
        prepare(f"{RUNS}/configs/{burst}_dsa_run.yaml", f"{burst}_dsa", f"{RUNS}/data/joint")[0],
        band_params("D", nD, P.get("delta_dm_D", 0.0)),
    ]
)

fig, ax = plt.subplots(3, 2, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1.4, 1]})
for col, (R, name) in enumerate([(RC, "CHIME 400-800 MHz"), (RD, "DSA 1.31-1.50 GHz")]):
    t = R["t"]
    disp = R["d"]
    step = max(1, disp.shape[0] // 400)
    ax[0, col].imshow(
        disp[::step],
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, disp.shape[0]],
        cmap="viridis",
        vmin=np.nanpercentile(disp, 5),
        vmax=np.nanpercentile(disp, 99.5),
    )
    ax[0, col].set_title(f"{name}   chi2/pt={R['chi2']:.2f}  (N={R['N']})")
    ax[0, col].set_ylabel("freq chan")
    dp, mp = np.nansum(R["d"], 0), np.nansum(R["pred"], 0)
    ax[1, col].plot(t, dp, color="0.2", lw=1.0, label="data")
    ax[1, col].plot(t, mp, color="crimson", lw=1.4, label="model")
    for n in range(R["N"]):
        ax[1, col].plot(
            t, np.nansum(R["comps"][n], 0), lw=0.8, ls="--", alpha=0.8, label=f"comp {n + 1}"
        )
    ax[1, col].legend(fontsize=7, ncol=2)
    ax[1, col].set_ylabel("flux (a.u.)")
    ax[1, col].margins(x=0)
    ax[2, col].plot(t, dp - mp, color="0.4", lw=0.8)
    ax[2, col].axhline(0, color="r", lw=0.6)
    ax[2, col].set_ylabel("resid")
    ax[2, col].set_xlabel("time (ms)")
    ax[2, col].margins(x=0)
fig.suptitle(
    f"{burst}  joint model {tag} vs data  --  alpha={alpha:.2f}  tau_1GHz={tau:.3f} ms"
    f"  (gain-marginal; per-channel amp freedom)",
    y=1.0,
)
fig.tight_layout()
fp = f"{RUNS}/data/joint/{burst}_overlay_{tag}.png"
fig.savefig(fp, dpi=120, bbox_inches="tight")
print(
    f"wrote {fp}  chime_chi2/pt={RC['chi2']:.3f} dsa_chi2/pt={RD['chi2']:.3f} s2C={RC['s2']:.3g} s2D={RD['s2']:.3g}"
)
