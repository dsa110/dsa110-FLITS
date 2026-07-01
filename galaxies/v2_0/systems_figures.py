#!/usr/bin/env python3
"""Characterizing figures for the dominant intervening systems of the sample:
the three foreground galaxies whose interiors/CGM source a non-negligible sightline
DM, and the four innermost foreground clusters along the FRB 20230307A field.

Two composites for the manuscript:
  galaxies_cgm   - 1x3, per galaxy: mNFW hot-halo DM(b) column, sightline impact,
                   0.1 R_vir interior cap, R_vir; annotated DM/tau.
  clusters_icm   - 2x2, per cluster: FRB/ModifiedNFW baryon DM(b) column,
                   R500 and the sightline impact; annotated DM/b/R500.

Everything is computed from the repo kernels (scattering_predict + build_unified)
so the figure reproduces the photo-z-corrected sightline budget; a __main__
self-check asserts that reproduction before the figures are trusted.

Galaxy inputs are the photo-z-corrected foreground catalogs (the canonical state
adopted 2026-06-24; pending promotion to results/). Cluster inputs are the
DESI-spec foreground clusters of Table~\\ref{tab:foreground}, whose R500 follows
from the tabulated b and b/R500 and whose M500 follows from R500 at the cluster z.
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
    from galaxies.foreground.build_unified import build_unified_records
    from galaxies.foreground.config import COSMO
    from galaxies.foreground.sightline_budget import INTERIOR_B_OVER_RVIR
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from galaxies.foreground import scattering_predict as scat
    from galaxies.foreground.build_unified import build_unified_records
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

# Three dominant foreground galaxies (nickname, TNS, RA, Dec, z_frb)
# halos under the photo-z-corrected budget (2 galaxy-interior + 1 CGM grazer).
# (Casey/FRB 20240229A is excluded: its only true foreground galaxy, UGC 06371,
# grazes at b/R_vir~=1 for a negligible ~0.3 pc/cm^3; the closer "interior" object
# in its field is a misclassified PSF star, e_zphot~=z, fqual=0.)
GAL_TARGETS = [
    ("phineas", "FRB 20230307A", "11h51m07.52s", "+71d41m44.3s", 0.2710),
    ("whitney", "FRB 20220310F", "08h58m52.92s", "+73d29m27.0s", 0.4790),
    ("isha", "FRB 20221113A", "04h45m38.64s", "+70d18m26.6s", 0.2505),
]

# Four innermost foreground clusters (by b/R500) in the FRB 20230307A field
# FRB 20230307A field from Table~\ref{tab:foreground} (objid, b_kpc, b/R500, z).
CLUSTER_TARGETS = [
    ("J115120.4+714435", 604.0, 0.83, 0.200),
    ("J115128.2+713637", 1055.0, 1.25, 0.192),
    ("J114944.0+714348", 1569.0, 2.96, 0.244),
    ("J115140.5+712732", 2105.0, 3.32, 0.176),
]

# Reference sightline totals from the photo-z-corrected budget (DM_int raw, ms tau)
# used only as a reproduction self-check, not plotted.
_BUDGET_CHECK = {  # name: sum_dm_int_raw (FRB ModifiedNFW hot + cool columns)
    "phineas": 297.2,
    "whitney": 62.0,
    "isha": 40.6,
}

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GAL_DIR = os.path.join(_REPO, "scratch", "photoz-fix")
DEFAULT_OUT_DIR = "/Users/jakobfaber/Developer/overleaf/Faber2026/figures"


def dominant_foreground_halo(name: str, ra: str, dec: str, z_frb: float, gal_dir: str) -> dict:
    """Return the dominant foreground halo record + sightline DM sum for a galaxy."""
    df = pd.read_csv(os.path.join(gal_dir, f"{name}_galaxies.csv"))
    sc = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    uni = build_unified_records(
        df, z_frb=z_frb, sight_ra=sc.ra.deg, sight_dec=sc.dec.deg, enrich=False
    )
    z = pd.to_numeric(uni["z"], errors="coerce")
    fg = uni[(z < z_frb) & (uni["mass_source"] != "cluster_catalog")].copy()
    fg["_pt"] = pd.to_numeric(fg["pred_tau_scat_ms_1GHz"], errors="coerce").fillna(-1.0)
    fg["_dh"] = pd.to_numeric(fg["dm_halo"], errors="coerce").fillna(0.0)
    dom = fg.sort_values(["_pt", "_dh"], ascending=False).iloc[0]
    sum_hot = pd.to_numeric(fg["dm_halo"], errors="coerce").fillna(0.0).sum()
    sum_cool = pd.to_numeric(fg["dm_cool"], errors="coerce").fillna(0.0).sum()
    return {
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


def make_galaxy_figure(gal_dir: str):
    """1x3 mNFW hot-halo DM(b) panels for the three dominant foreground galaxies."""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), dpi=150, facecolor=BG_LIGHT)
    for ax, (name, tns, ra, dec, z_frb) in zip(axes.ravel(), GAL_TARGETS):
        ax.set_facecolor(BG_LIGHT)
        d = dominant_foreground_halo(name, ra, dec, z_frb, gal_dir)
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
    """2x2 mNFW baryon-column DM(b) panels for the four innermost foreground clusters."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=150, facecolor=BG_LIGHT)
    for ax, (objid, b_kpc, b_over_r500, z) in zip(axes.ravel(), CLUSTER_TARGETS):
        ax.set_facecolor(BG_LIGHT)
        c = cluster_params(b_kpc, b_over_r500, z)
        r500, m500 = c["r500_kpc"], c["m500_msun"]
        r_trunc = c["rvir_mnfw_kpc"]

        bb = np.linspace(1.0, r_trunc, 240)
        dm_b = np.array([scat.dm_cluster_mnfw_model(m500, z, float(x)) for x in bb])
        ax.plot(bb, dm_b, color=HALO_COLOR, lw=2.2, zorder=4, label="hot mNFW column")

        ax.axvspan(0, r500, color=INTERV_COLOR, alpha=0.07, zorder=0)
        ax.axvline(r500, color=TEXT_DARK, ls="--", lw=1.1, zorder=3)
        ax.text(
            r500,
            ax.get_ylim()[1] * 0.96 if dm_b.max() > 0 else 1.0,
            "$R_{500}$",
            fontsize=7,
            color=TEXT_DARK,
            ha="right",
            va="top",
            rotation=90,
        )

        inside = b_over_r500 < 1.0
        impact_color = HOST_COLOR if inside else INTERV_COLOR
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
                color=INTERV_COLOR,
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
            color=TEXT_DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID_COLOR, alpha=0.9),
        )

        ax.set_title(objid, fontsize=10, fontweight="bold", color=DARK_BLUE)
        ax.set_xlabel("impact parameter $b$ (kpc)", fontsize=9, color=TEXT_DARK)
        ax.set_ylabel("hot-baryon DM (pc cm$^{-3}$)", fontsize=9, color=TEXT_DARK)
        ax.set_xlim(0, r_trunc)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", color=GRID_COLOR, alpha=0.8, zorder=0)
        if ax is axes.ravel()[0]:
            ax.legend(
                loc="upper left",
                fontsize=7.5,
                frameon=True,
                facecolor="white",
                edgecolor=GRID_COLOR,
            )

    fig.suptitle(
        "Foreground clusters of FRB 20230307A: mNFW baryon column vs. impact",
        fontsize=12,
        fontweight="bold",
        color=DARK_BLUE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _selfcheck(gal_dir: str) -> None:
    """Assert the figure inputs reproduce the photo-z-corrected sightline budget."""
    for name, _tns, ra, dec, z_frb in GAL_TARGETS:
        d = dominant_foreground_halo(name, ra, dec, z_frb, gal_dir)
        ref, tol = _BUDGET_CHECK[name], 1.0
        assert abs(d["sum_dm_int"] - ref) < tol, f"{name}: DM_int {d['sum_dm_int']:.1f} != {ref}"
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
    p.add_argument("--gal-dir", default=DEFAULT_GAL_DIR, help="dir of *_galaxies.csv inputs")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="figure output directory")
    args = p.parse_args()

    _selfcheck(args.gal_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for fig, stem in (
        (make_galaxy_figure(args.gal_dir), "galaxies_cgm"),
        (make_cluster_figure(), "clusters_icm"),
    ):
        for ext in ("pdf", "svg", "png"):
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
