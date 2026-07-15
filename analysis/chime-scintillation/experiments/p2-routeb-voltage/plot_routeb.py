#!/usr/bin/env python3
"""Diagnostic figures for the P2 Route-B G1/G2 calibration (visual vetting).

Reads the g1_*/g2_* result JSONs and reproduces one example off-pulse
realization to render, for the owner's visual review:

  1. injection-recovery scatter  (median recovered vs injected Δν_d, per cell)
  2. amplitude-pull distributions (per statistic, detectable cells)
  3. null-campaign amplitude/z histogram (G2, with the Šidák threshold)
  4. an example off-pulse ratio spectrum + cross-ACF (null realization)
  5. estimator-sanity contrast: recovery vs injected effective modulation

Run after freeze + g1 + g2.  Writes PNGs into figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
sys.path.insert(0, str(HERE))
import routeb_calibration as rc  # noqa: E402

from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402

COLORS = {0.15: "#1f77b4", 0.17: "#d62728"}
STAT_MARKER = {"S1": "o", "S2": "s"}


def _load(statistic: str, gate: str) -> dict | None:
    path = HERE / f"{gate}_{statistic}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_injection_recovery(statistics):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    lim = 420
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1 (perfect recovery)")
    ax.fill_between([0, lim], [0, 0.7 * lim], [0, 1.3 * lim], color="green", alpha=0.07,
                    label="±30% certify band")
    for statistic in statistics:
        data = _load(statistic, "g1")
        if data is None:
            continue
        for cell in data["cells"]:
            inj = cell["dnu_khz"]
            med = cell["median_recovered_dnu_mhz"]
            med = np.nan if med is None else med * 1e3
            edge = "k" if cell["certify"] else "none"
            ax.scatter(
                inj, med, marker=STAT_MARKER[statistic], s=90,
                facecolor=COLORS[cell["modulation"]], edgecolor=edge, linewidth=1.4, zorder=3,
            )
    ax.axvline(rc.DETECTABLE_DNU_KHZ, color="grey", ls=":", lw=1)
    ax.text(rc.DETECTABLE_DNU_KHZ + 4, 60, "detectable-gate\n(≥127 kHz must certify)",
            fontsize=8, color="grey")
    ax.axvline(rc.CONTROL_DNU_KHZ, color="purple", ls=":", lw=1)
    ax.text(rc.CONTROL_DNU_KHZ + 5, lim * 0.55, "35 kHz control\n(must NOT certify)",
            fontsize=8, color="purple")
    handles = [
        plt.Line2D([], [], marker=STAT_MARKER[s], ls="none", color="grey", label=s)
        for s in statistics if _load(s, "g1") is not None
    ]
    handles += [
        plt.Line2D([], [], marker="o", ls="none", color=COLORS[m], label=f"m={m}")
        for m in (0.15, 0.17)
    ]
    handles += [plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="k", label="certified")]
    ax.legend(handles=handles, frameon=True, framealpha=0.9, fontsize=8,
              loc="center right", ncol=1)
    ax.set(
        xlabel="injected Δν_d (kHz)", ylabel="median recovered Δν_d (kHz)",
        xlim=(0, lim), ylim=(0, lim),
        title="Route-B injection recovery at the real burst contrast (f_b=0.05)",
    )
    ax.grid(alpha=0.2)
    out = FIGURES / "injection_recovery_scatter.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_pull_distributions(statistics):
    fig, axes = plt.subplots(1, len(statistics), figsize=(4.6 * len(statistics), 4.2), squeeze=False)
    grid = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    for ax, statistic in zip(axes[0], statistics, strict=False):
        data = _load(statistic, "g1")
        if data is None:
            ax.set_visible(False)
            continue
        pulls = np.concatenate(
            [np.asarray(c["pulls"]) for c in data["cells"] if c["is_detectable_gate"] and c["pulls"]]
        ) if any(c["is_detectable_gate"] and c["pulls"] for c in data["cells"]) else np.array([])
        if pulls.size:
            ax.hist(pulls, bins=25, density=True, color="#4c72b0", alpha=0.8,
                    label=f"detectable cells\nmedian={np.median(pulls):+.2f}")
        ax.plot(grid, gauss, "k--", lw=1.2, label="N(0,1)")
        ax.axvline(0, color="grey", lw=0.8)
        for z in (-2, 2):
            ax.axvline(z, color="red", ls=":", lw=1)
        ax.set(xlabel="amplitude pull z", title=f"{statistic} amplitude pulls", xlim=(-4.5, 4.5))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIGURES / "pull_distributions.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_null_histogram(statistics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for statistic in statistics:
        data = _load(statistic, "g2")
        if data is None:
            continue
        amps = [r["amplitude"] for r in data["records"] if r["amplitude"] is not None]
        zs = [r["amplitude_z"] for r in data["records"] if r["amplitude_z"] is not None]
        axes[0].hist(amps, bins=14, alpha=0.6, label=f"{statistic} (n={len(amps)})")
        axes[1].hist(zs, bins=14, alpha=0.6, label=f"{statistic}")
        axes[1].axvline(data["sidak_z_threshold"], color="red", ls="--", lw=1.2)
        axes[1].axvline(-data["sidak_z_threshold"], color="red", ls="--", lw=1.2)
    axes[0].set(xlabel="null fitted amplitude", ylabel="count",
                title="G2 off-pulse null amplitudes")
    axes[1].set(xlabel="null amplitude z = A/σ_A", ylabel="count",
                title="G2 null significance (red = Šidák 5% threshold)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIGURES / "null_campaign_histogram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _example_realization(products, statistic, on_gain=None, seed=900000):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(products.off_pool)
    on = np.sort(perm[: rc.N_ON])
    off = np.sort(perm[rc.N_ON:])
    if on_gain is not None:
        delta = rb.lorentzian_gain_field(
            rng, n_channels=products.n_band_channels, width_channels=213.0 / rc.CHANNEL_WIDTH_KHZ
        )
        on_gain = 1.0 + on_gain * delta
    result = products.run_statistic(statistic, on, off, on_gain=on_gain)
    ratios = [
        rb._ratio(rb.row_nanmean(products.dynamic[p][:, on] * (1.0 if on_gain is None else on_gain[:, None])),
                  rb.row_nanmean(products.dynamic[p][:, off]))
        for p in range(2)
    ]
    return result, ratios


def fig_example_offpulse(products):
    result, ratios = _example_realization(products, "S1", on_gain=None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    freq = products.frequencies
    sl = slice(6000, 6400)  # a representative 400-channel slice of the high band
    axes[0].plot(freq[sl], ratios[0][sl], lw=0.7, color="#1f77b4", label="R_pol0")
    axes[0].plot(freq[sl], ratios[1][sl], lw=0.7, color="#d62728", alpha=0.8, label="R_pol1")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set(xlabel="frequency (MHz)", ylabel="ratio R_p(ν) = on/off − 1",
                title="Example off-pulse ratio spectra (common mode cancelled)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.2)

    lags = result.cross.lag_bins * products.channel_width_mhz * 1e3
    axes[1].errorbar(lags, result.cross.acf, yerr=result.cross.error, fmt=".", ms=5,
                     color="#333333", label="off-pulse cross-ACF")
    if result.fit is not None:
        axes[1].plot(np.asarray(result.fit["fit_lags_mhz"]) * 1e3, result.fit["model_acf"],
                     lw=2, color="orange",
                     label=f"null fit Δν_d={result.fit['dnu_mhz']*1e3:.0f} kHz")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(xlabel="frequency lag (kHz)", ylabel="cross-covariance",
                title="Example off-pulse cross-ACF (consistent with null)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    out = FIGURES / "example_offpulse_ratio_crossacf.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_estimator_sanity(products):
    """Recovery vs injected effective modulation s_eff=f_b*m at Δν_d=213 kHz.

    Shows the estimator recovers the width once s_eff clears its threshold, so
    the G1 failure is an S/N ceiling (real s_eff≈0.0085), not a broken fit."""
    dnu = 213.0
    wch = dnu / rc.CHANNEL_WIDTH_KHZ
    s_grid = [0.0085, 0.02, 0.05, 0.10, 0.20]
    medians = []
    spreads = []
    for s_eff in s_grid:
        rec = []
        for r in range(20):
            rng = np.random.default_rng(770000 + r)
            perm = rng.permutation(products.off_pool)
            on = np.sort(perm[: rc.N_ON])
            off = np.sort(perm[rc.N_ON:])
            delta = rb.lorentzian_gain_field(rng, n_channels=products.n_band_channels, width_channels=wch)
            res = products.run_statistic("S1", on, off, on_gain=1.0 + s_eff * delta)
            if res.fit is not None and np.isfinite(res.fit.get("dnu_err_mhz", np.nan)):
                rec.append(res.fit["dnu_mhz"] * 1e3)
        rec = np.asarray(rec)
        medians.append(np.median(rec) if rec.size else np.nan)
        spreads.append((np.percentile(rec, 16), np.percentile(rec, 84)) if rec.size else (np.nan, np.nan))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    lo = [s[0] for s in spreads]
    hi = [s[1] for s in spreads]
    ax.fill_between(s_grid, lo, hi, alpha=0.2, color="#4c72b0", label="16–84%")
    ax.plot(s_grid, medians, "o-", color="#1f77b4", label="median recovered Δν_d")
    ax.axhline(dnu, color="green", ls="--", label=f"injected {dnu:.0f} kHz")
    ax.axvline(rc.BURST_FLUX_FRACTION * 0.17, color="red", ls=":",
               label=f"real s_eff=f_b·m≈{rc.BURST_FLUX_FRACTION*0.17:.4f}")
    ax.set(xscale="log", xlabel="injected effective modulation s_eff = f_b·m",
           ylabel="recovered Δν_d (kHz)",
           title="Estimator sanity: recovery vs signal strength (S1, 213 kHz)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    out = FIGURES / "estimator_sanity_snr.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, default=rc.DEFAULT_POL0)
    parser.add_argument("--pol1", type=Path, default=rc.DEFAULT_POL1)
    parser.add_argument("--frequencies", type=Path, default=rc.DEFAULT_FREQUENCIES)
    parser.add_argument("--time0-metadata", type=Path, default=rc.DEFAULT_METADATA)
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    statistics = [s for s in ("S1", "S2") if _load(s, "g1") is not None]
    written = []
    written.append(fig_injection_recovery(statistics or ["S1", "S2"]))
    written.append(fig_pull_distributions(statistics or ["S1", "S2"]))
    written.append(fig_null_histogram([s for s in ("S1", "S2") if _load(s, "g2") is not None]))
    products = rc.Products(args)
    written.append(fig_example_offpulse(products))
    written.append(fig_estimator_sanity(products))
    print(json.dumps({"figures": [str(p) for p in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
