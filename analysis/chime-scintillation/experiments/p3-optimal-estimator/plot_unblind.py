#!/usr/bin/env python3
"""Figure for the one-shot unblinded P3′ on-pulse scan (owner-authorized
2026-07-15): z(Δν_d) and â(Δν_d) against the calibrated expectations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "unblind_onpulse.json").read_text())
z = d["onpulse"]["z_by_dnu"]
a = d["onpulse"]["a_hat_by_dnu"]
dnu = np.array(sorted(float(k) for k in z))
zv = np.array([z[str(k)] for k in dnu])
av = np.array([a[str(k)] for k in dnu])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(dnu, zv, "o-")
axes[0].axhline(d["z_trials_threshold"], color="r", ls=":", label=f"z_trials = {d['z_trials_threshold']:.2f}")
axes[0].axhline(5, color="gray", ls="--", lw=0.8, label="G3 bar (5σ)")
axes[0].set(xscale="log", xlabel=r"$\Delta\nu_d$ template [kHz]", ylabel="z (null-calibrated)",
            title="On-pulse matched scan: z rises monotonically to the grid edge")
axes[0].legend(fontsize=8)
axes[1].plot(dnu, av, "o-", label=r"on-pulse $\hat{a}$")
axes[1].axhline((0.05 * 0.17) ** 2, color="g", ls="--",
                label=r"calibrated ceiling $(f_b m)^2$, $m=0.17$")
axes[1].set(xscale="log", yscale="log", xlabel=r"$\Delta\nu_d$ template [kHz]",
            ylabel=r"$\hat{a}$",
            title=r"$\hat{a} \approx 10^{-3}$: 11× the scintillation ceiling → intrinsic envelope")
axes[1].legend(fontsize=8)
fig.suptitle("P3′ one-shot unblinding — broad spectral structure, inconsistent with the calibrated scintle model")
fig.tight_layout()
fig.savefig(HERE / "figures" / "unblind_onpulse_scan.png", dpi=150)
print("wrote figures/unblind_onpulse_scan.png")
