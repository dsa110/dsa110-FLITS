#!/usr/bin/env python3
"""Characterizing figures for the dominant intervening systems of the sample:
the three foreground galaxies whose halos source the largest intervening
sightline DM, and the four innermost foreground clusters along the
FRB 20230307A field.

Two composites for the manuscript:
  galaxies_cgm   - 1x3, per galaxy: mNFW hot-halo DM(b) column, sightline impact,
                   0.1 R_vir interior cap, R_vir; annotated DM/tau.
  clusters_icm   - 2x2, per cluster: FRB/ModifiedNFW baryon DM(b) column,
                   R500 and the sightline impact; annotated DM/b/R500.

The foreground set comes from sightline_budget.foreground_unified -- the SAME
acquisition (census registry first, curated results/*_galaxies.csv fallback)
and z < z_frb filter the tabulated budget uses -- and the three panel
sightlines are picked live as the top-3 by capped intervening DM in
results/sightline_dm_scattering_budget.csv. A self-check asserts each panel
sightline's foreground DM sum reproduces that budget row before the figures
are trusted (the pre-2026-07-06 version read a scratch/photoz-fix snapshot
with its own filter and drifted from the budget).

Cluster inputs are the DESI-spec foreground clusters of
Table~\\ref{tab:foreground}, whose R500 follows from the tabulated b and
b/R500 and whose M500 follows from R500 at the cluster z.
"""

from __future__ import annotations

import math
import os
import sys

import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from galaxies.foreground import scattering_predict as scat
    from galaxies.foreground.config import COSMO, TARGETS
    from galaxies.foreground.sightline_budget import INTERIOR_B_OVER_RVIR, foreground_unified
    from scattering.scat_analysis.burst_metadata import load_tns_name
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from galaxies.foreground import scattering_predict as scat
    from galaxies.foreground.config import COSMO
    from galaxies.foreground.sightline_budget import INTERIOR_B_OVER_RVIR

# Palette consistent with sightline_budget.make_budget_figure.
DARK_BLUE = "#1B365D"
HALO_COLOR = "#4A90E2"
COOL_COLOR = "#7FB3E8"
INTERV_COLOR = "#F5A623"
HOST_COLOR = "#D0021B"
TEXT_DARK = "#333333"
GRID_COLOR = "#E5E5E5"
BG_LIGHT = "#FAFBFC"

N_GAL_PANELS = 3  # top-N sightlines by capped intervening DM in the live budget

# Four innermost foreground clusters (by b/R500) in the FRB 20230307A field
# FRB 20230307A field from Table~\ref{tab:foreground} (objid, b_kpc, b/R500, z).
CLUSTER_TARGETS = [
    ("J115120.4+714435", 604.0, 0.83, 0.200),
    ("J115128.2+713637", 1055.0, 1.25, 0.192),
    ("J114944.0+714348", 1569.0, 2.96, 0.244),
    ("J115140.5+712732", 2105.0, 3.32, 0.176),
]

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(_REPO, "results")
# Derived from this file's location, not a machine path: when FLITS is checked
# out as the `pipeline` submodule of Faber2026, _REPO's parent is the manuscript
# root and this resolves to Faber2026/figures. Override with --out-dir for a
# standalone FLITS checkout. (Same treatment ae67f4f gave plot_association_cards.py.)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(_REPO), "figures")


def dominant_foreground_halo(name: str, ra: str, dec: str, z_frb: float, results_dir: str) -> dict:
    """Return the dominant foreground halo record + sightline DM sum for a galaxy."""
    sc = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    uni = foreground_unified(
        name, z_frb, sc.ra.deg, sc.dec.deg, results_dir=results_dir, enrich=False
    )
    # Budget-wide DM sum (all foreground rows, clusters included) for the
    # reproduction self-check against results/sightline_dm_scattering_budget.csv.
    sum_all = (
        pd.to_numeric(uni["dm_halo"], errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(uni["dm_cool"], errors="coerce").fillna(0.0).sum()
    )
    # Dominant GALAXY halo for the panel (clusters are characterized separately).
    fg = uni[uni["mass_source"] != "cluster_catalog"].copy()
    fg["_pt"] = pd.to_numeric(fg["pred_tau_scat_ms_1GHz"], errors="coerce").fillna(-1.0)
    fg["_dh"] = pd.to_numeric(fg["dm_halo"], errors="coerce").fillna(0.0)
    dom = fg.sort_values(["_pt", "_dh"], ascending=False).iloc[0]
    sum_hot = pd.to_numeric(fg["dm_halo"], errors="coerce").fillna(0.0).sum()
    sum_cool = pd.to_numeric(fg["dm_cool"], errors="coerce").fillna(0.0).sum()
    return {
        "sum_dm_int_all": float(sum_all),
        "z_gal": float(dom["z"]),
        "impact_kpc": float(dom["impact_kpc"]),
        "m_halo": float(dom["M_halo"]),
        "logM_halo": float(dom["logM_halo"]),
        "r_vir": float(dom["R_vir_kpc"]),
        "b_over_rvir": float(dom["b_over_rvir"]),
        "mass_source": str(dom["mass_source"]),
        "dm_halo": float(dom["dm_halo"]),
        "dm_cool": float(dom["dm_cool"]),
        "tau": float(dom["pred_tau_scat_ms_1GHz"]),
        "tau_lo": float(dom["pred_tau_scat_ms_1GHz_lo"]),
        "tau_hi": float(dom["pred_tau_scat_ms_1GHz_hi"]),
        "n_foreground": int(len(fg)),
        "sum_dm_int": float(sum_hot + sum_cool),
    }


def cluster_params(b_kpc: float, b_over_r500: float, z: float) -> dict:
    """Recover R500, M500 and the mNFW baryon-column DM for a foreground cluster."""
    r500 = b_kpc / b_over_r500
    rho_crit = COSMO.critical_density(z).to(u.Msun / u.kpc**3).value
    m500 = (4.0 / 3.0) * math.pi * 500.0 * rho_crit * r500**3
    m200 = scat.CLUSTER_M500_TO_M200 * m500
    return {
        "r500_kpc": r500,
        "m500_msun": m500,
        "m200_msun": m200,
        "rvir_mnfw_kpc": scat._frb_mnfw_rvir_kpc(m200, z),
        "logM500": math.log10(m500),
        "dm_at_b": scat.dm_cluster_mnfw_model(m500, z, b_kpc),
    }


def _mass_label(source: str) -> str:
    return {
        "glade_catalog": "GLADE+ $M_\\star$",
        "xsc_kband": "2MASS $K$ $M_\\star$",
        "desi_ls_sed": "DESI SED $M_\\star$",
        "wise_w1": "WISE $W1$ $M_\\star$",
        "ps1_taylor": "PS1 $M_\\star$",
        "assumed": "assumed $L_\\star$",
    }.get(source, source)


def select_gal_targets(results_dir: str, n: int = N_GAL_PANELS) -> list[tuple]:
    """Top-n sightlines by capped intervening DM in the live budget CSV.

    Returns (nickname, tns, ra, dec, z_frb, budget_row) tuples; the budget row
    rides along for the reproduction self-check.
    """
    budget = pd.read_csv(os.path.join(results_dir, "sightline_dm_scattering_budget.csv"))
    by_tns = {r["name"]: r for _, r in budget.iterrows()}
    ranked = []
    for name, ra, dec, z_frb in TARGETS:
        row = by_tns.get(load_tns_name(name))
        if row is None or not row["n_foreground"]:
            continue
        ranked.append((float(row["dm_intervening_capped"]), name, ra, dec, z_frb, row))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [(name, load_tns_name(name), ra, dec, z, row) for _, name, ra, dec, z, row in ranked[:n]]


def make_galaxy_figure(targets: list[tuple], results_dir: str):
    """1xN mNFW hot-halo DM(b) panels for the dominant foreground galaxies."""
    fig, axes = plt.subplots(1, len(targets), figsize=(13.4, 4.5), dpi=150, facecolor=BG_LIGHT)
    for ax, (name, tns, ra, dec, z_frb, _row) in zip(np.atleast_1d(axes).ravel(), targets):
        ax.set_facecolor(BG_LIGHT)
        d = dominant_foreground_halo(name, ra, dec, z_frb, results_dir)
        rvir, b, mh, zg = d["r_vir"], d["impact_kpc"], d["m_halo"], d["z_gal"]
        b_cap = INTERIOR_B_OVER_RVIR * rvir
        interior = d["b_over_rvir"] < INTERIOR_B_OVER_RVIR

        # mNFW projected hot-halo DM as a function of impact parameter.
        bb = np.linspace(0.5, rvir, 240)
        dm_b = np.array([scat.dm_halo_mnfw(mh, zg, float(x)) or 0.0 for x in bb])
        ax.plot(bb, dm_b, color=HALO_COLOR, lw=2.2, zorder=4, label="hot mNFW column")

        # Interior region (b < 0.1 R_vir): the smooth column is extrapolated and
        # capped at the 0.1 R_vir floor.
        ax.axvspan(0, b_cap, color=HOST_COLOR, alpha=0.07, zorder=0)
        ax.axvline(
            b_cap, color=HOST_COLOR, ls=":", lw=1.4, zorder=3, label="$0.1\\,R_{\\rm vir}$ cap"
        )
        ax.axvline(rvir, color=TEXT_DARK, ls="--", lw=1.0, alpha=0.6, zorder=3)
        ax.text(
            rvir,
            ax.get_ylim()[1] * 0.02,
            "$R_{\\rm vir}$",
            fontsize=7,
            color=TEXT_DARK,
            ha="right",
            va="bottom",
            rotation=90,
        )

        # The sightline's actual impact and the DM the dominant halo samples there.
        dm_at_b = scat.dm_halo_mnfw(mh, zg, b) or 0.0
        dm_cap = scat.dm_halo_mnfw(mh, zg, max(b, b_cap)) or 0.0
        ax.axvline(b, color=INTERV_COLOR, lw=2.0, zorder=5)
        ax.scatter([b], [dm_at_b], color=INTERV_COLOR, s=42, zorder=6, edgecolor="white")
        ax.annotate(
            f"DM$={dm_at_b:.0f}$",
            xy=(b, dm_at_b),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8,
            color=INTERV_COLOR,
            fontweight="bold",
        )

        regime = "interior" if interior else "CGM grazer"
        info = (
            f"$z={zg:.3f}$,  $\\log M_{{\\rm halo}}={d['logM_halo']:.1f}$\n"
            f"$b={b:.1f}$ kpc,  $b/R_{{\\rm vir}}={d['b_over_rvir']:.3f}$ ({regime})\n"
            f"halo DM$={d['dm_halo']:.0f}\\!\\to\\!{dm_cap:.0f}$ (capped)\n"
            f"$\\tau_{{1\\rm GHz}}={d['tau']:.2g}$ ms,  {_mass_label(d['mass_source'])}"
        )
        ax.text(
            0.97,
            0.96,
            info,
            transform=ax.transAxes,
            fontsize=7.6,
            va="top",
            ha="right",
            color=TEXT_DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID_COLOR, alpha=0.9),
        )

        ax.set_title(f"{tns}  ({name})", fontsize=10, fontweight="bold", color=DARK_BLUE)
        ax.set_xlabel("impact parameter $b$ (kpc)", fontsize=9, color=TEXT_DARK)
        ax.set_ylabel("hot-halo DM (pc cm$^{-3}$)", fontsize=9, color=TEXT_DARK)
        ax.set_xlim(0, rvir)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", color=GRID_COLOR, alpha=0.8, zorder=0)
        if ax is axes.ravel()[0]:
            ax.legend(
                loc="center right",
                fontsize=7.5,
                frameon=True,
                facecolor="white",
                edgecolor=GRID_COLOR,
            )

    fig.suptitle(
        "Dominant intervening galaxies: mNFW circumgalactic dispersion vs. sightline impact",
        fontsize=12,
        fontweight="bold",
        color=DARK_BLUE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def make_cluster_figure():
    """2x2 mNFW baryon-column DM(b) panels for the four innermost foreground clusters.

    Palette matches the foreground-halo census figure (Figure 2,
    sightline_halo_grid): magma ramp for the mass/DM story, ink text, faint gray
    spines/grid, and the same soft corridor tint for the inside-R500 region.
    """
    # Colors consistent with sightline_halo_grid (Figure 2).
    INK = "#22252b"        # text/spines
    FAINT = "#9aa0a8"      # secondary gray (grid, R500 line)
    CORRIDOR = "#eef0f3"   # soft inside-R500 band (Fig. 2 impact corridor)
    PANEL_BG = "white"
    _magma = plt.get_cmap("magma")
    CURVE_C = _magma(0.45)     # mNFW column curve: magma mid-tone
    CROSS_C = _magma(0.72)     # inside-R500 crossing (bright, = Fig 2 cluster hue)
    OUTSIDE_C = _magma(0.30)   # outside-R500 markers: darker magma

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=150, facecolor=PANEL_BG)
    for ax, (objid, b_kpc, b_over_r500, z) in zip(axes.ravel(), CLUSTER_TARGETS):
        ax.set_facecolor(PANEL_BG)
        c = cluster_params(b_kpc, b_over_r500, z)
        r500, m500 = c["r500_kpc"], c["m500_msun"]
        r_trunc = c["rvir_mnfw_kpc"]

        bb = np.linspace(1.0, r_trunc, 240)
        dm_b = np.array([scat.dm_cluster_mnfw_model(m500, z, float(x)) for x in bb])
        ax.plot(bb, dm_b, color=CURVE_C, lw=2.2, zorder=4, label="hot mNFW column")

        ax.axvspan(0, r500, color=CORRIDOR, zorder=0)
        ax.axvline(r500, color=FAINT, ls="--", lw=1.1, zorder=3)
        ax.text(
            r500,
            ax.get_ylim()[1] * 0.96 if dm_b.max() > 0 else 1.0,
            "$R_{500}$",
            fontsize=7,
            color=INK,
            ha="right",
            va="top",
            rotation=90,
        )

        inside = b_over_r500 < 1.0
        impact_color = CROSS_C if inside else OUTSIDE_C
        if b_kpc <= r_trunc:
            ax.axvline(b_kpc, color=impact_color, lw=2.0, zorder=5)
            ax.scatter(
                [b_kpc], [c["dm_at_b"]], color=impact_color, s=42, zorder=6, edgecolor="white"
            )
        else:
            # Sightline passes beyond the model truncation: mark at the right edge.
            ax.text(
                0.96,
                0.12,
                f"$b={b_kpc:.0f}$ kpc\n(beyond $R_{{\\rm vir,mNFW}}$)",
                transform=ax.transAxes,
                fontsize=7.5,
                color=OUTSIDE_C,
                ha="right",
                va="bottom",
                fontweight="bold",
            )

        verdict = "inside $R_{500}$ (pierces ICM)" if inside else "outside $R_{500}$"
        info = (
            f"$z={z:.3f}$,  $\\log M_{{500}}={c['logM500']:.2f}$\n"
            f"$b={b_kpc:.0f}$ kpc,  $b/R_{{500}}={b_over_r500:.2f}$\n"
            f"{verdict}\n"
            f"$\\mathrm{{DM_{{cl}}}}(b)={c['dm_at_b']:.0f}$ pc cm$^{{-3}}$"
        )
        ax.text(
            0.97,
            0.96,
            info,
            transform=ax.transAxes,
            fontsize=7.6,
            va="top",
            ha="right",
            color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=FAINT, alpha=0.9),
        )

        ax.set_title(objid, fontsize=10, fontweight="bold", color=INK)
        ax.set_xlabel("impact parameter $b$ (kpc)", fontsize=9, color=INK)
        ax.set_ylabel("hot-baryon DM (pc cm$^{-3}$)", fontsize=9, color=INK)
        ax.set_xlim(0, r_trunc)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(FAINT)
        ax.tick_params(colors=FAINT, labelcolor=INK)
        ax.grid(True, ls=":", color=FAINT, alpha=0.5, zorder=0)
        if ax is axes.ravel()[0]:
            ax.legend(
                loc="upper left",
                fontsize=7.5,
                frameon=True,
                facecolor="white",
                edgecolor=FAINT,
            )

    fig.tight_layout()
    return fig


def _selfcheck(targets: list[tuple], results_dir: str) -> None:
    """Assert the figure inputs reproduce the live sightline budget CSV."""
    for name, _tns, ra, dec, z_frb, row in targets:
        d = dominant_foreground_halo(name, ra, dec, z_frb, results_dir)
        ref, tol = float(row["dm_intervening"]), 1.0
        assert abs(d["sum_dm_int_all"] - ref) < tol, (
            f"{name}: DM_int {d['sum_dm_int_all']:.1f} != budget {ref:.1f}"
        )
        # The dominant column sampled at its own impact must match its tabulated dm_halo.
        dm_at_b = scat.dm_halo_mnfw(d["m_halo"], d["z_gal"], d["impact_kpc"])
        assert abs(dm_at_b - d["dm_halo"]) < 0.5, f"{name}: dm_halo(b) mismatch"
    # The single R500-piercing cluster must source a non-trivial mNFW column; the
    # two beyond truncation must return zero.
    inner = cluster_params(*CLUSTER_TARGETS[0][1:])
    assert inner["dm_at_b"] > 50.0, "inner cluster ICM ceiling unexpectedly small"
    for objid, b_kpc, b_over_r500, z in CLUSTER_TARGETS[2:]:
        assert cluster_params(b_kpc, b_over_r500, z)["dm_at_b"] == 0.0, (
            f"{objid} should be truncated"
        )


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="dir of *_galaxies.csv + sightline_dm_scattering_budget.csv",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="figure output directory")
    args = p.parse_args()

    targets = select_gal_targets(args.results_dir)
    _selfcheck(targets, args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for fig, stem in (
        (make_galaxy_figure(targets, args.results_dir), "galaxies_cgm"),
        (make_cluster_figure(), "clusters_icm"),
    ):
        for ext in ("pdf", "svg", "png"):
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
