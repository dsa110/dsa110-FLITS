#!/usr/bin/env python3
"""Single-system sufficiency check: can any *one* committed foreground galaxy
account for the scattering excess on the four EXCESS sightlines?

This reuses the existing CGM/scattering engine (build_unified.build_unified_records,
sightline_budget._scattering_verdict). It adds no new physics: for each foreground
galaxy (z < z_frb) it compares the per-galaxy predicted intervening tau
(pred_tau_scat_ms_1GHz, and its prior-upper _hi) against the sightline's measured
excess tau, using the same pred/obs >= 0.5 threshold the budget tool uses.

Honest-null caveat: with the *committed* catalogs every one of the four excess
sightlines has zero committed foreground galaxies (results/search_summary.csv:
Zach/Wilhelm/Hamilton/Chromatica num_galaxies=0; no {name}_galaxies.csv on disk).
A positive single-system attribution would require re-running galaxies/foreground/search.py
with ENABLE_EXTRA_ENGINES / ENABLE_ENRICHERS = True (a network-bound step not run
here). This wrapper is the reusable harness + the null it currently returns.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import config
    from . import sightline_budget as sb
    from .build_unified import build_unified_records
except ImportError:  # pragma: no cover - direct script execution.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from galaxies.foreground import config
    from galaxies.foreground import sightline_budget as sb
    from galaxies.foreground.build_unified import build_unified_records

# The four sightlines flagged EXCESS in results/excess_survival_models.csv.
EXCESS_SIGHTLINES = ("Wilhelm", "Zach", "Hamilton", "Chromatica")

_TARGET_BY_NAME = {name: (name, ra, dec, z) for (name, ra, dec, z) in config.TARGETS}

# Schema for the output CSV, so the null case still writes a valid, self-documenting
# header-only file rather than a zero-byte blob that cannot be re-read.
_ATTR_COLUMNS = (
    "sightline",
    "z_frb",
    "z_gal",
    "impact_kpc",
    "b_over_rvir",
    "logM_best",
    "M_halo",
    "R_vir_kpc",
    "dm_halo",
    "dm_cool",
    "g_scatt",
    "cool_fc",
    "pred_tau_scat_ms_1GHz",
    "pred_tau_scat_ms_1GHz_lo",
    "pred_tau_scat_ms_1GHz_hi",
    "tau_excess_ms",
    "verdict",
)


def _excess_tau_ms(name: str, budget_df: pd.DataFrame | None) -> float:
    """Best available measured/excess scattering tau (ms, 1 GHz) for a sightline.

    Uses the PASS-gated tau_obs_ms from the committed budget table when present;
    none of the four currently have one, so this is NaN (an honest "no measured
    tau to attribute" rather than a fabricated target).
    """
    if budget_df is None:
        return math.nan
    # The budget CSV emits TNS designations (#26); match on nickname OR its TNS name.
    from scattering.scat_analysis.burst_metadata import load_tns_name

    keys = {name.lower(), load_tns_name(name).lower()}
    row = budget_df[budget_df["name"].astype(str).str.lower().isin(keys)]
    if row.empty:
        return math.nan
    return sb._f(row.iloc[0].get("tau_obs_ms"))


def attribute(results_dir: str, budget_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per committed foreground system across the four excess sightlines."""
    rows: list[dict] = []
    for name in EXCESS_SIGHTLINES:
        _, ra, dec, z_frb = _TARGET_BY_NAME[name]
        sight = sb.SkyCoord(ra, dec, unit=(sb.u.hourangle, sb.u.deg))
        tau_obs = _excess_tau_ms(name, budget_df)

        csv_path = os.path.join(results_dir, f"{name.lower()}_galaxies.csv")
        if not os.path.exists(csv_path):
            continue
        matches = pd.read_csv(csv_path)
        unified = build_unified_records(
            matches, z_frb=z_frb, sight_ra=sight.ra.deg, sight_dec=sight.dec.deg, enrich=False
        )
        for _, g in unified.iterrows():
            z = sb._f(g.get("z"))
            if not (math.isfinite(z) and z < float(z_frb)):  # foreground only
                continue
            pred = sb._f(g.get("pred_tau_scat_ms_1GHz"))
            pred_hi = sb._f(g.get("pred_tau_scat_ms_1GHz_hi"))
            verdict = sb._scattering_verdict(
                tau_obs,
                pred if math.isfinite(pred) else 0.0,
                pred_hi if math.isfinite(pred_hi) else 0.0,
                n_fg=1,
            )
            rows.append(
                {
                    "sightline": name,
                    "z_frb": float(z_frb),
                    "z_gal": z,
                    "impact_kpc": sb._f(g.get("impact_kpc")),
                    "b_over_rvir": sb._f(g.get("b_over_rvir")),
                    "logM_best": sb._f(g.get("logM_best")),
                    "M_halo": sb._f(g.get("M_halo")),
                    "R_vir_kpc": sb._f(g.get("R_vir_kpc")),
                    "dm_halo": sb._f(g.get("dm_halo")),
                    "dm_cool": sb._f(g.get("dm_cool")),
                    "g_scatt": sb._f(g.get("g_scatt")),
                    "cool_fc": sb._f(g.get("cool_fc")),
                    "pred_tau_scat_ms_1GHz": pred,
                    "pred_tau_scat_ms_1GHz_lo": sb._f(g.get("pred_tau_scat_ms_1GHz_lo")),
                    "pred_tau_scat_ms_1GHz_hi": pred_hi,
                    "tau_excess_ms": tau_obs,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows, columns=list(_ATTR_COLUMNS))


def make_figure(attr_df: pd.DataFrame, budget_df: pd.DataFrame | None = None):
    """Four panels (one per excess sightline) of predicted-vs-measured tau.

    Panels with no committed foreground system render an explicit "no foreground
    systems in committed catalog" message instead of an empty axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150, facecolor=sb.BG_LIGHT)
    for ax, name in zip(axes.flat, EXCESS_SIGHTLINES, strict=True):  # 2x2 axes == 4 sightlines
        ax.set_facecolor(sb.BG_LIGHT)
        sub = attr_df[attr_df["sightline"] == name] if len(attr_df) else attr_df
        tau_obs = _excess_tau_ms(name, budget_df)
        ax.set_title(name, fontsize=12, fontweight="bold", color=sb.DARK_BLUE)
        ax.set_xlabel(r"foreground system", fontsize=9, color=sb.TEXT_DARK)
        ax.set_ylabel(r"$\tau_{\rm scat}$ at 1 GHz (ms)", fontsize=9, color=sb.TEXT_DARK)
        ax.grid(True, axis="y", linestyle=":", color=sb.GRID_COLOR, alpha=0.8)

        if sub is None or len(sub) == 0:
            ax.text(
                0.5,
                0.5,
                "no foreground systems\nin committed catalog",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=sb.HOST_COLOR,
                fontweight="bold",
            )
            ax.set_xticks([])
            continue

        x = np.arange(len(sub))
        pred = pd.to_numeric(sub["pred_tau_scat_ms_1GHz"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(sub["pred_tau_scat_ms_1GHz_lo"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(sub["pred_tau_scat_ms_1GHz_hi"], errors="coerce").to_numpy(float)
        # Prior bracket as an asymmetric error bar around the predicted point.
        err_lo = np.where(np.isfinite(lo), np.maximum(pred - lo, 0.0), 0.0)
        err_hi = np.where(np.isfinite(hi), np.maximum(hi - pred, 0.0), 0.0)
        ax.errorbar(
            x,
            pred,
            yerr=[err_lo, err_hi],
            fmt="o",
            color=sb.INTERV_COLOR,
            ecolor=sb.INTERV_COLOR,
            elinewidth=1.4,
            capsize=4,
            label="predicted intervening (prior bracket)",
        )
        if math.isfinite(tau_obs) and tau_obs > 0:
            ax.axhline(tau_obs, color=sb.TEXT_DARK, lw=2.0, label="measured excess")
        else:
            ax.text(
                0.02,
                0.96,
                "no PASS-gated measured tau",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=sb.HOST_COLOR,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"g{int(i)}" for i in x], fontsize=8)
        if np.any(np.isfinite(pred) & (pred > 0)):
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=7, frameon=True, facecolor="white", edgecolor=sb.GRID_COLOR)

    fig.suptitle(
        "Single-system sufficiency on EXCESS sightlines: predicted intervening vs measured tau",
        fontsize=13,
        fontweight="bold",
        color=sb.DARK_BLUE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _panel_expectation(name: str, attr_df: pd.DataFrame) -> str:
    n = len(attr_df[attr_df["sightline"] == name]) if len(attr_df) else 0
    if n == 0:
        return (
            f"Panel '{name}': explicit 'no foreground systems in committed catalog' "
            f"message (zero committed foreground galaxies for this sightline)."
        )
    return (
        f"Panel '{name}': {n} predicted-intervening tau point(s) with prior bracket; "
        f"horizontal line = measured excess tau if a PASS fit exists."
    )


def _write_manifest(manifest_path: str, png_name: str, attr_df: pd.DataFrame) -> None:
    import json

    entry = {
        png_name: {
            "title": "Single-system sufficiency on EXCESS sightlines",
            "panels": {n: _panel_expectation(n, attr_df) for n in EXCESS_SIGHTLINES},
            "expectation": (
                "Four panels (Wilhelm/Zach/Hamilton/Chromatica). With committed catalogs "
                "all four are empty -> all four show the 'no foreground systems' message. "
                "Any populated panel compares per-galaxy predicted intervening tau (with prior "
                "bracket) to the measured excess tau."
            ),
        }
    }
    existing = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(entry)
    with open(manifest_path, "w") as fh:
        json.dump(existing, fh, indent=2)


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base, "results")

    budget_csv = os.path.join(results_dir, "sightline_dm_scattering_budget.csv")
    budget_df = pd.read_csv(budget_csv) if os.path.exists(budget_csv) else None

    attr = attribute(results_dir, budget_df=budget_df)
    csv_path = os.path.join(results_dir, "excess_sightline_attribution.csv")
    attr.to_csv(csv_path, index=False)
    print(
        f"Wrote {csv_path}  ({len(attr)} foreground system rows across {len(EXCESS_SIGHTLINES)} excess sightlines)"
    )

    fig = make_figure(attr, budget_df=budget_df)
    png_name = "excess_sightline_attribution.png"
    png_path = os.path.join(results_dir, png_name)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")

    _write_manifest(os.path.join(results_dir, "figures.manifest.json"), png_name, attr)
    print(f"Wrote {os.path.join(results_dir, 'figures.manifest.json')}")

    if len(attr) == 0:
        print(
            "NULL RESULT: zero committed foreground systems on any excess sightline; "
            "no single intervening system can be tested. A positive attribution "
            "requires a network re-run of galaxies/foreground/search.py with "
            "ENABLE_EXTRA_ENGINES / ENABLE_ENRICHERS = True (not performed)."
        )


if __name__ == "__main__":
    main()
