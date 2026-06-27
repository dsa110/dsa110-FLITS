#!/usr/bin/env python
"""2D (freq x time) joint-fit model vs data: data | model | residual per band.

Same gain-marginal reconstruction as model_overlay.py, but plots the full 2D
predicted waterfall (sum_n g[f,n] K_n[f,t]) next to the data and the residual,
so the per-channel spectrum/scint the gains absorb is visible.

  python model_2d.py <burst> <tag>     # tag like C2D3, C4D1
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
from run_joint_fit import prepare
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
    _, diag = _gain_marginal_multi_band(model, ps, ["M3"] * len(ps), s2=None)
    A = M + (var / diag["s2"])[:, None, None] * np.eye(Ks.shape[0])[None]
    g = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    pred = np.einsum("fn,nft->ft", g, Ks)
    resid = (d - pred) / sig[:, None]  # whitened residual
    chi2 = float(np.sum(resid**2) / resid.size)
    return dict(d=d, pred=pred, resid=resid, t=model.time, f=model.freq[valid], chi2=chi2)


RC = solve_band(
    prepare(f"{RUNS}/configs/{burst}_chime_run.yaml", f"{burst}_chime", f"{RUNS}/data/joint")[0],
    band_params("C", nC, P.get("delta_dm_C", 0.0)),
)
RD = solve_band(
    prepare(f"{RUNS}/configs/{burst}_dsa_run.yaml", f"{burst}_dsa", f"{RUNS}/data/joint")[0],
    band_params("D", nD, P.get("delta_dm_D", 0.0)),
)

fig, ax = plt.subplots(3, 2, figsize=(13, 9))
for col, (R, name) in enumerate([(RC, "CHIME 400-800 MHz"), (RD, "DSA 1.31-1.50 GHz")]):
    t, f = R["t"], R["f"]
    ext = [t[0], t[-1], f[0], f[-1]]
    vmax = np.nanpercentile(R["d"], 99.5)
    vmin = np.nanpercentile(R["d"], 2)
    for row, (img, title, cmap, kw) in enumerate(
        [
            (R["d"], f"{name}  DATA", "viridis", dict(vmin=vmin, vmax=vmax)),
            (R["pred"], f"MODEL ({tag})", "viridis", dict(vmin=vmin, vmax=vmax)),
            (
                R["resid"],
                f"RESIDUAL (whitened)  chi2/pt={R['chi2']:.2f}",
                "RdBu_r",
                dict(vmin=-4, vmax=4),
            ),
        ]
    ):
        im = ax[row, col].imshow(img, aspect="auto", origin="lower", extent=ext, cmap=cmap, **kw)
        ax[row, col].set_title(title, fontsize=9)
        ax[row, col].set_ylabel("freq (GHz)")
        fig.colorbar(im, ax=ax[row, col], pad=0.01, fraction=0.04)
    ax[2, col].set_xlabel("time (ms)")
fig.suptitle(
    f"{burst}  2D joint model {tag} vs data  --  alpha={alpha:.2f}  tau_1GHz={tau:.3f} ms", y=1.0
)
fig.tight_layout()
fp = f"{RUNS}/data/joint/{burst}_2d_{tag}.png"
fig.savefig(fp, dpi=115, bbox_inches="tight")
print(f"wrote {fp}  chime_chi2/pt={RC['chi2']:.3f} dsa_chi2/pt={RD['chi2']:.3f}")
