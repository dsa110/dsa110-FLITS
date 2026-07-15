#!/usr/bin/env python3
"""Render the C1 blinded calibration-matrix verdict as figures.

Reads calibration/cell_*.json, calibration/nulls.json, and
calibration_verdict.json; writes figures/*.png plus figures.manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODS = (0.10, 0.15, 0.17, 0.20, 0.30, 1.00)
WIDTHS = (3.0, 6.0, 10.0, 16.0)
GATED = {(0.15, w) for w in WIDTHS} | {(0.17, w) for w in WIDTHS}


def load_cell(m: float, w: float) -> dict:
    return json.loads((HERE / "calibration" / f"cell_m{m:.2f}_w{w:g}.json").read_text())


def matrix_figure() -> Path:
    # per-cell ratio of observed statistic to its gate limit (>1 = fail)
    width_ratio = np.zeros((len(MODS), len(WIDTHS)))
    m_ratio = np.zeros_like(width_ratio)
    coverage = np.zeros_like(width_ratio)
    passed = np.zeros_like(width_ratio, dtype=bool)
    for i, m in enumerate(MODS):
        for j, w in enumerate(WIDTHS):
            cell = load_cell(m, w)
            width_ratio[i, j] = (
                cell["median_absolute_width_bias_mhz"] / cell["width_bias_limit_mhz"]
            )
            m_ratio[i, j] = (
                cell["median_absolute_modulation_bias"] / cell["modulation_bias_limit"]
            )
            coverage[i, j] = cell["coverage_68"]
            passed[i, j] = cell["pass"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = [
        (width_ratio, "median |width bias| / limit", "log"),
        (m_ratio, "median |m bias| / limit", "log"),
        (coverage, "68% interval coverage", None),
    ]
    lo, hi = 0.53, 0.83
    for ax, (grid, title, scale) in zip(axes, panels):
        if scale == "log":
            im = ax.imshow(
                np.log10(grid), cmap="RdYlGn_r", aspect="auto", vmin=-1.2, vmax=1.2
            )
            fig.colorbar(im, ax=ax, label="log10(observed / limit)")
        else:
            im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0.4, vmax=1.0)
            fig.colorbar(im, ax=ax, label="coverage")
        for i in range(len(MODS)):
            for j in range(len(WIDTHS)):
                if scale == "log":
                    bad = grid[i, j] > 1.0
                    txt = f"{grid[i, j]:.2f}"
                else:
                    bad = not (lo <= grid[i, j] <= hi)
                    txt = f"{grid[i, j]:.2f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold" if bad else "normal",
                )
                if (MODS[i], WIDTHS[j]) in GATED:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, ec="blue", lw=2
                        )
                    )
        ax.set_xticks(range(len(WIDTHS)), [f"{w:g}" for w in WIDTHS])
        ax.set_yticks(range(len(MODS)), [f"{m:g}" for m in MODS])
        ax.set_xlabel("truth width [native channels]")
        ax.set_ylabel("injected modulation index m")
        ax.set_title(title)
    fig.suptitle(
        "C1 all-pairs cross-ACF blinded calibration matrix (128 trials/cell; "
        "blue boxes = qualification-gated cells; bold = gate violated)"
    )
    out = FIG_DIR / "c1_calibration_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def recovery_figure() -> Path:
    fig, axes = plt.subplots(
        len(MODS), len(WIDTHS), figsize=(13, 15), sharex=False, sharey=False
    )
    for i, m in enumerate(MODS):
        for j, w in enumerate(WIDTHS):
            cell = load_cell(m, w)
            truth = cell["truth_width_mhz"]
            rec = np.array(
                [
                    r["fit"]["dnu_mhz"]
                    for r in cell["records"]
                    if np.isfinite(r["fit"]["dnu_mhz"])
                ]
            )
            ax = axes[i, j]
            ax.hist(rec, bins=30, color="steelblue", alpha=0.8)
            ax.axvline(truth, color="green", lw=1.5, label="truth")
            ax.axvline(np.median(rec), color="crimson", lw=1.2, ls="--", label="median")
            ax.set_title(
                f"m={m:g}, w={w:g}ch {'PASS' if cell['pass'] else 'FAIL'}", fontsize=8
            )
            ax.tick_params(labelsize=6)
            if (m, w) in GATED:
                for s in ax.spines.values():
                    s.set_edgecolor("blue")
                    s.set_linewidth(2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(
        "C1 recovered dnu_d per cell (green = truth, red dashed = median; "
        "blue frames = gated cells). Low-m fits pile up at the fit-window "
        "boundaries instead of the truth.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "c1_recovery_histograms.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def nulls_figure() -> Path:
    nulls = json.loads((HERE / "calibration" / "nulls.json").read_text())
    zmax = [r["max_abs_z"] for r in nulls["records"]]
    kinds = [r["kind"] for r in nulls["records"]]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = {"held_out_offpulse": "steelblue", "pairing_scramble": "darkorange"}
    x = np.arange(len(zmax))
    ax.bar(x, zmax, color=[colors.get(k, "gray") for k in kinds])
    ax.axhline(
        nulls["family_wise_threshold"],
        color="crimson",
        lw=1.5,
        label=f"family-wise threshold {nulls['family_wise_threshold']:.3f} "
        f"(alpha={nulls['family_wise_alpha']}, N={nulls['n_comparisons']})",
    )
    ax.set_xlabel("null realization")
    ax.set_ylabel("max |z| over lag bins")
    ax.set_title(
        f"C1 null campaign: max_abs_z={nulls['max_abs_z']:.3f} "
        f"(FAIL), fit-level detections={nulls['n_detections']}"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    labels += list(colors)
    ax.legend(handles, labels, fontsize=8)
    out = FIG_DIR / "c1_nulls.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    verdict = json.loads((HERE / "calibration_verdict.json").read_text())
    figures = {
        matrix_figure().name: (
            "Three heatmaps over the 6x4 (m, width) grid: width-bias/limit and "
            "m-bias/limit on log10 scale (positive = violated) and 68% coverage. "
            "Expected: every m<=0.30 cell violates at least one gate (bold), all "
            "eight blue-framed gated cells (m=0.15/0.17) fail, all m=1.00 cells pass."
        ),
        recovery_figure().name: (
            "Histogram of recovered dnu_d per cell vs truth. Expected: m=1.00 rows "
            "concentrate on the green truth line; low-m rows (esp. width 3) pile up "
            "at the fit-window bounds away from truth."
        ),
        nulls_figure().name: (
            "Per-null-realization max |z| bars vs the family-wise threshold 4.408. "
            "Expected: at least one bar crosses the red line (max 4.810, FAIL); "
            "one fail-closed fit-level detection (bound-clear control fit with "
            "invalid uncertainty counts as a detection)."
        ),
    }
    manifest = {
        "experiment": "c1-allpairs-crossgp",
        "verdict_go": verdict["go"],
        "figures": [
            {"path": f"figures/{name}", "expectation": expectation}
            for name, expectation in figures.items()
        ],
    }
    (HERE / "figures.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"written": list(figures)}, indent=2))


if __name__ == "__main__":
    main()
