#!/usr/bin/env python3
"""Per-sightline foreground-halo grid, in the style of Khrykin et al. 2024
(FLIMFLAM DR1, arXiv:2402.00505) Figure 4.

One panel per co-detection FRB sightline. Each foreground halo (z < z_frb) is
drawn at (redshift, signed projected impact parameter), as an open gray circle
whose radius is the halo's projected virial radius (R200, delta_def=200 in the
catalog) converted to kpc, with a filled marker colored by log10(M_halo/Msun).
A black spiral-galaxy glyph marks the FRB host at (z_frb, 0); the horizontal
line is the sightline.

The y coordinate is the true projected offset of the halo from the sightline,
signed by the declination difference (halo north of the FRB is +), so halos
spread above and below the sightline as in the reference figure -- the catalog
stores only the unsigned impact parameter b_kpc, which we use for the magnitude
and R_vir geometry and sign by sky position.

Data: results/chime_dsa_gladep_candidates_ranked.csv (per-halo z, b_kpc,
m_delta, r_delta_computed, sky position, frb_z), the same ranked candidate set
the foreground census is built from. Only is_foreground rows enter.
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
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.path import Path as MplPath

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Deterministic style regardless of run location. matplotlib auto-loads the
# repo matplotlibrc (serif/CM fonts, savefig.bbox=tight, ...) only when the
# process starts inside pipeline/, which made both the fonts and the output
# size cwd-dependent. Load that rc explicitly by path so every run gets the
# same fonts, then re-pin the save geometry: 'standard' (not 'tight') is
# required here because tight re-lays-out the axes after the display-space
# circle radii were computed against the drawn geometry, distorting the R200
# circles. (Passing bbox_inches=None to savefig does NOT override the rc.)
_RC = os.path.join(_REPO, "matplotlibrc")
if os.path.exists(_RC):
    matplotlib.rc_file(_RC)
plt.rcParams["savefig.bbox"] = "standard"
plt.rcParams["savefig.pad_inches"] = 0.05
DEFAULT_RESULTS_DIR = os.path.join(_REPO, "results")
DEFAULT_HALO_CSV = os.path.expanduser(
    "~/Data/frb-foreground-halos/results/chime_dsa_gladep_candidates_ranked.csv"
)
# Derived from this file's location, not a machine path: when FLITS is checked
# out as the `pipeline` submodule of Faber2026, _REPO's parent is the manuscript
# root and this resolves to Faber2026/figures. Override with --out-dir for a
# standalone FLITS checkout. (Same treatment ae67f4f gave plot_association_cards.py.)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(_REPO), "figures")

# Colorbar range for log10(M_halo/Msun). Tuned to the recovered halo population
# (bulk 11-13.2, tail to 14.4) so magma's full sweep is used instead of pinning
# most halos into one hue; the massive cluster still saturates the bright end.
MASS_MIN, MASS_MAX = 11.0, 14.0
# magma matches the burst dynamic-spectrum waterfalls (codetection_plots
# water_cmap default), keeping the manuscript's figure palette consistent.
CMAP = "magma"
Y_LIM = 600.0  # impact parameter axis half-range, kpc


def _load(halo_csv: str):
    """Return (foreground-halo frame, {frb_name: frb_z} for all z-known sightlines).

    The panel roster is every sightline with a known FRB redshift -- including
    those with zero foreground halos (a clean sightline is an informative null),
    matching the reference figure which shows empty panels. Sightlines without a
    spectroscopic FRB redshift cannot be placed (no host marker, no z<z_frb cut) and are
    dropped.
    """
    df = pd.read_csv(halo_csv)
    z_known = df[df["frb_z"].notna()].drop_duplicates("frb_name")
    roster = dict(zip(z_known["frb_name"], z_known["frb_z"].astype(float)))

    fg = df[df["is_foreground"].astype(str).str.lower().isin(["true", "1"])].copy()
    fg = fg[fg["z"].notna() & fg["b_kpc"].notna() & fg["m_delta"].notna()]
    fg = fg[fg["frb_name"].isin(roster)]
    # Signed projected offset: magnitude = b_kpc, sign from declination diff.
    ddec = fg["dec"].to_numpy(float) - fg["frb_dec"].to_numpy(float)
    fg["b_signed"] = np.where(ddec >= 0, 1.0, -1.0) * fg["b_kpc"].to_numpy(float)
    fg["logM"] = np.log10(fg["m_delta"].astype(float))
    return fg, roster


CLUSTER_MASS = 1.0e14  # M200 threshold for "cluster" (vs galaxy-scale halo)


def _galaxy_path(incl=0.30, bulge_r=0.42, n=64):
    """Inclined-disk 'mini galaxy' glyph as a single filled Path.

    An ellipse of aspect `incl` (inclined disk) tilted 25 deg, with a round
    central bulge that PROTRUDES beyond the disk minor axis (bulge_r > incl):
    the classic edge-on-galaxy silhouette. The protrusion is load-bearing --
    a bulge hidden inside the same-color disk renders as a plain tilted
    ellipse, which in this figure shares a silhouette class with the R200
    halo Ellipse patches it must be distinguished from. This reads instantly
    as a galaxy at ~12 px raster, where a filled two-arm spiral collapses
    into an ambiguous S/yin-yang shape. Normalized to unit max extent so
    scatter's `s` scales it like a builtin marker.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    tilt = np.deg2rad(25.0)
    ct, st = np.cos(tilt), np.sin(tilt)
    # Disk ellipse (semi-axes 1.0 x incl), rotated by tilt.
    dx, dy = np.cos(t), incl * np.sin(t)
    disk = np.column_stack([dx * ct - dy * st, dx * st + dy * ct])
    disk = np.vstack([disk, disk[0]])  # closing vertex for CLOSEPOLY
    # Central bulge: protrudes above/below the disk plane.
    bulge = MplPath.circle((0.0, 0.0), bulge_r)
    verts = [disk, bulge.vertices]
    codes = [
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(disk) - 2) + [MplPath.CLOSEPOLY],
        list(bulge.codes),
    ]
    p = MplPath(np.vstack(verts), np.concatenate(codes).astype(np.uint8))
    m = np.abs(p.vertices).max()
    return MplPath(p.vertices / m, p.codes)


GALAXY_MARKER = _galaxy_path()


def _crossing_mask(sub: pd.DataFrame) -> np.ndarray:
    """The cluster-scale halo the sightline pierces within R200.

    The manuscript's budget cut retains a single such system across the whole
    sample -- the massive cluster in the FRB 20230307A field. We emphasize
    exactly that: strict R200 intersection AND cluster mass (M200 > 1e14). Lower-
    mass strict intersections are galaxy halos already carried in the galaxy
    budget, so they are not singled out here.
    """
    if "intersects_strict" not in sub:
        return np.zeros(len(sub), dtype=bool)
    strict = sub["intersects_strict"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    massive = sub["m_delta"].astype(float).to_numpy() > CLUSTER_MASS
    return strict & massive


def make_grid(halo_csv: str = DEFAULT_HALO_CSV):
    fg, roster = _load(halo_csv)
    # One panel per z-known sightline, ordered by FRB redshift.
    order = sorted(roster, key=lambda k: roster[k])
    n = len(order)
    ncols = 2
    nrows = math.ceil(n / ncols)

    ink = "#22252b"       # near-black ink for text/spines (softer than pure black)
    faint = "#9aa0a8"     # secondary gray
    sightline_c = "#3a3f47"
    corridor_c = "#eef0f3"  # faint shaded impact corridor

    # Wider-than-tall panels keep the whole 5x2 grid short enough that the
    # figure* + its long caption fit one manuscript page (a taller grid overflows
    # and sends latexmk into an output dead-cycle loop).
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12.0, 1.95 * nrows), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(top=0.885, bottom=0.085, left=0.115, right=0.975,
                        hspace=0.42, wspace=0.06)

    norm = plt.Normalize(vmin=MASS_MIN, vmax=MASS_MAX)
    # magma matches the burst waterfalls; drop the near-black bottom so low-mass
    # points read on white, and reuse the same ramp to tint the R200 disks.
    base = plt.get_cmap(CMAP)
    cmap = ListedColormap(base(np.linspace(0.18, 1.0, 256)))

    z_max = max(roster.values())
    x_hi = 0.05 * math.ceil((z_max + 0.02) / 0.05)

    for i, (ax, name) in enumerate(zip(axes, order)):
        z_frb = roster[name]
        # Faint impact corridor (|b| < 300 kpc), a soft depth band around the LOS.
        ax.axhspan(-300, 300, color=corridor_c, zorder=0)
        ax.axhline(0.0, color=sightline_c, lw=0.9, zorder=1)

        # FRB host: a haloed spiral-galaxy glyph so it reads over any
        # overlapping disk.
        ax.scatter([z_frb], [0.0], marker=GALAXY_MARKER, s=900, color="white",
                   edgecolors="none", zorder=5)
        ax.scatter([z_frb], [0.0], marker=GALAXY_MARKER, s=680, color=ink,
                   edgecolors="none", zorder=6)

        # Title in a translucent chip so it stays legible over any halo disk.
        ax.text(0.025, 0.93,
                f"{name}\n" + r"$z=$" + f"{z_frb:.4f}",
                transform=ax.transAxes, fontsize=9.8, color=ink,
                va="top", ha="left", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="none", alpha=0.78), zorder=7)

        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(-Y_LIM, Y_LIM)
        ax.set_yticks([-500, -250, 0, 250, 500])
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(125))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        # Lighter frame: keep left/bottom, drop top/right. Ticks stay dark and
        # inward on every panel so the redshift / impact-parameter scale is
        # readable even on interior panels (labels remain edge-only via sharex/y).
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(which="major", direction="in", length=4.5, width=0.9,
                       color=ink, labelcolor=ink, labelsize=9.5,
                       top=False, right=False)
        ax.tick_params(which="minor", direction="in", length=2.5, width=0.7,
                       color=ink, top=False, right=False)

    # Halo markers drawn after limits are fixed so R200 disks are round in
    # display space (600 kpc on y vs ~0.5 in z on x is a ~10^5 scale mismatch;
    # a data-unit circle would collapse to a vertical sliver -- we size the disk
    # by R200 on y and map that same pixel radius back through the x scale).
    fig.canvas.draw()
    for ax, name in zip(axes, order):
        bbox = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_per_px = (x1 - x0) / bbox.width
        y_per_px = (y1 - y0) / bbox.height
        sub = fg[fg["frb_name"] == name].reset_index(drop=True)
        cross = _crossing_mask(sub)
        # Draw largest disks first so small halos sit on top (readability).
        draw_order = sub["r_delta_computed"].astype(float).fillna(0).argsort()[::-1]
        for idx in draw_order:
            row = sub.iloc[idx]
            r_kpc = float(row.get("r_delta_computed", float("nan")))
            y = float(row["b_signed"])
            x = float(row["z"])
            col = cmap(norm(row["logM"]))
            is_cross = bool(cross[idx])
            if math.isfinite(r_kpc) and r_kpc > 0:
                r_px = r_kpc / y_per_px
                w, h = 2.0 * r_px * x_per_px, 2.0 * r_px * y_per_px
                # Soft mass-tinted fill = the halo's extent; thin edge for shape.
                ax.add_patch(Ellipse((x, y), w, h, facecolor=col, alpha=0.14,
                                     edgecolor="none", zorder=2))
                ax.add_patch(Ellipse((x, y), w, h, fill=False,
                                     edgecolor=col if not is_cross else ink,
                                     lw=1.4 if is_cross else 0.7,
                                     alpha=0.9 if is_cross else 0.55, zorder=3))
            # Center marker; crossing cluster gets a ring (keyed in the legend).
            if is_cross:
                ax.scatter([x], [y], s=90, facecolors="none", edgecolors=ink,
                           linewidths=1.1, zorder=6)
            ax.scatter([x], [y], c=[col], s=26, edgecolors=ink,
                       linewidths=0.4, zorder=4)

    # Legend key in the leftover slot (odd panel count). It must NOT be one of
    # the shared axes: clearing ticks on a sharex/sharey member propagates to
    # every panel and wipes the redshift / impact-parameter tick labels. So hide
    # the shared slot and drop an independent axis in its place.
    legend_ax = None
    if n < len(axes):
        slot = axes[n]
        slot.set_visible(False)
        legend_ax = fig.add_axes(slot.get_position())
        legend_ax.set_xlim(0.0, x_hi)
        legend_ax.set_ylim(-Y_LIM, Y_LIM)
        for side in ("top", "right", "left", "bottom"):
            legend_ax.spines[side].set_visible(False)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        fig.canvas.draw()
        lb = legend_ax.get_window_extent()
        xpp = x_hi / lb.width
        ypp = (2 * Y_LIM) / lb.height
        # No panel title: the swatch/marker labels are self-explanatory, and a
        # centered "key" text collided with the 500-kpc swatch circle anyway.
        # Two R200 swatches side by side (small + large), each with its label
        # below, so nothing overlaps. Circle radius = R200 on this panel's scale.
        for cxf, r_kpc in [(0.20, 200), (0.52, 500)]:
            cx = cxf * x_hi
            r_px = r_kpc / ypp
            legend_ax.add_patch(Ellipse((cx, 210), 2 * r_px * xpp, 2 * r_px * ypp,
                                        facecolor=faint, alpha=0.16,
                                        edgecolor=faint, lw=0.8, zorder=2))
            legend_ax.text(cx, -230, rf"$R_{{200}}$" + "\n" + rf"${r_kpc}$ kpc",
                           fontsize=8.5, color=ink, va="top", ha="center",
                           linespacing=1.3)
        # FRB host + cluster-crossing entries, right column.
        legend_ax.scatter([0.82 * x_hi], [300], s=520, marker=GALAXY_MARKER,
                          color=ink, zorder=4)
        legend_ax.text(0.88 * x_hi, 300, "FRB\nhost", fontsize=8.5, color=ink,
                       va="center", ha="left", linespacing=1.3)
        legend_ax.scatter([0.82 * x_hi], [-120], s=90, facecolors="none",
                          edgecolors=ink, linewidths=1.1, zorder=4)
        legend_ax.text(0.88 * x_hi, -120, "cluster\ncrossing", fontsize=8.5,
                       color=ink, va="center", ha="left", linespacing=1.3)

    for ax in axes[n + 1 :]:
        ax.set_visible(False)

    # Single shared axis labels (supxlabel/supylabel) instead of per-panel labels:
    # the panels are short, so a per-panel rotated y-label would span the panel
    # height and collide with the neighbor's ticks.
    fig.supylabel("impact parameter  [kpc]", fontsize=12.5, color=ink, x=0.035)
    fig.supxlabel("redshift", fontsize=12.5, color=ink, y=0.045)

    # Bottom-most VISIBLE panel in each column still needs its x-tick LABELS on
    # (sharex hides them; the odd panel count means the legend replaced a slot).
    for col in range(ncols):
        col_idx = [i for i in range(n) if i % ncols == col]
        if col_idx:
            axes[max(col_idx)].tick_params(axis="x", labelbottom=True)

    # Horizontal colorbar, spanning the plot width, ticks on top.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.30, 0.935, 0.45, 0.016])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.outline.set_edgecolor(faint)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(colors=faint, labelcolor=ink, labelsize=9.5, width=0.6)
    cbar.set_label(r"$\log_{10}\,(M_{\mathrm{halo}}/M_\odot)$", fontsize=11.5,
                   color=ink, labelpad=6)

    return fig


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--halo-csv", default=DEFAULT_HALO_CSV)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig = make_grid(args.halo_csv)
    for ext in ("pdf", "svg", "png"):
        path = os.path.join(args.out_dir, f"sightline_halo_grid.{ext}")
        # Save geometry is pinned at import (savefig.bbox='standard') so the
        # output is identical from any working directory.
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
