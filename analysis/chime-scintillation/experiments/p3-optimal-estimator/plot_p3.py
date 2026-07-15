#!/usr/bin/env python3
"""Diagnostic figures for the P3′ matched-estimator calibration (visual
vetting per the standing owner preference). Reads the gate JSONs written by
``p3_calibration.py``; writes PNGs into ``figures/``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def injection_recovery(g1: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, m in zip(axes, (0.15, 0.17)):
        cells = [c for c in g1["cells"] if c["modulation"] == m]
        for c in cells:
            x = np.full(len(c["recovered_dnu_khz"]), c["dnu_khz"])
            jitter = x * np.random.default_rng(1).uniform(-0.05, 0.05, x.size)
            ax.scatter(x + jitter, c["recovered_dnu_khz"], s=8, alpha=0.35)
            ax.scatter([c["dnu_khz"]], [c["median_recovered_dnu_khz"]], marker="_", s=400, color="k")
        grid = np.geomspace(20, 400, 50)
        ax.plot(grid, grid, "k--", lw=0.8)
        ax.fill_between(grid, grid * 0.7, grid * 1.3, color="g", alpha=0.12, label="±30% certify band")
        ax.set(xscale="log", yscale="log", xlabel="injected Δν_d [kHz]", title=f"m = {m}")
        ax.axvline(213, color="r", lw=0.6, ls=":")
    axes[0].set_ylabel("recovered Δν_d (argmax z) [kHz]")
    axes[0].legend(fontsize=8)
    fig.suptitle("G1″ injection recovery — matched scan (P3′)")
    fig.tight_layout()
    fig.savefig(FIG / "g1_injection_recovery.png", dpi=150)


def pulls(g1: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in g1["cells"]:
        label = f"m={c['modulation']} {c['dnu_khz']:.0f} kHz"
        ax.hist(c["pulls"], bins=20, histtype="step", label=label)
    ax.axvline(0, color="k", lw=0.7)
    for edge in (-2, 2):
        ax.axvline(edge, color="r", lw=0.7, ls=":")
    ax.set(xlabel="amplitude pull (â − a_true)/σ_null", ylabel="count", title="G1″ amplitude pulls")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "g1_pulls.png", dpi=150)


def null_campaign(g2: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(max(g2["cal_max_z"]), max(g2["eval_max_z"])) + 0.5, 30)
    ax.hist(g2["cal_max_z"], bins=bins, alpha=0.5, label=f"calibration nulls (n={g2['n_cal']})")
    ax.hist(g2["eval_max_z"], bins=bins, alpha=0.5, label=f"evaluation nulls (n={g2['n_eval']})")
    ax.axvline(g2["z_trials_threshold"], color="r", label=f"z_trials (p95 eval) = {g2['z_trials_threshold']:.2f}")
    ax.axvline(g2["p95_cal"], color="r", ls=":", label=f"p95 cal = {g2['p95_cal']:.2f}")
    ax.set(xlabel="max z over the 25-point Δν_d scan", ylabel="count", title="G2″ trials-corrected null maxima")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "g2_null_maxima.png", dpi=150)


def scan_assets() -> None:
    assets = np.load(HERE / "scan_assets.npz")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    k = np.arange(1, assets["variance_smoothed"].size + 1)
    for row, dnu in zip(assets["templates"][::6], assets["dnu_khz"][::6]):
        axes[0].plot(k, row, lw=0.8, label=f"{dnu:.0f} kHz")
    axes[0].axvspan(1, 10, color="r", alpha=0.1, label="k < 11 excluded")
    axes[0].set(xscale="log", yscale="log", xlabel="delay bin k", ylabel="T(k)", title="MC template bank (mask + window transfer baked in)")
    axes[0].legend(fontsize=7)
    axes[1].plot(k, assets["variance_smoothed"], lw=0.8)
    axes[1].axvspan(1, 10, color="r", alpha=0.1)
    axes[1].set(xscale="log", yscale="log", xlabel="delay bin k", ylabel="Var_null(k) (smoothed)", title="Measured null variance (calibration half)")
    fig.tight_layout()
    fig.savefig(FIG / "scan_assets.png", dpi=150)


def main() -> None:
    g1 = json.loads((HERE / "g1_matched.json").read_text())
    g2 = json.loads((HERE / "g2_matched.json").read_text())
    injection_recovery(g1)
    pulls(g1)
    null_campaign(g2)
    scan_assets()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
