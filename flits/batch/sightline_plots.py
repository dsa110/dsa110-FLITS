"""
sightline_plots.py
==================

Foreground-structure / sightline figures for FRB DM work: a sky view of the
intervening galaxies and clusters, tomography in virial and angular units, and
a cumulative DM(z) ledger along the line of sight.

Real-data interface
-------------------
Build a :class:`Sightline` from your ``sightline_budget`` outputs: one
:class:`ForegroundHalo` per intervening structure (redshift, mass, impact
parameter, and -- optionally -- precomputed R_200, halo DM, and sky offsets).
Anything left as ``None`` is filled in from :mod:`dm_models`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D

try:
    from .dm_models import Cosmology, ModifiedNFW
except ImportError:  # allow running as a plain script
    from dm_models import Cosmology, ModifiedNFW

__all__ = ["ForegroundHalo", "Sightline", "plot_sightline"]


@dataclass
class ForegroundHalo:
    z: float
    mass_msun: float
    impact_kpc: float
    r200_kpc: Optional[float] = None      # filled from Cosmology if None
    dm_pc_cm3: Optional[float] = None      # filled from ModifiedNFW if None
    dra_arcmin: Optional[float] = None     # sky offset; random PA if None
    ddec_arcmin: Optional[float] = None
    label: Optional[str] = None
    is_cluster: bool = False


@dataclass
class Sightline:
    z_frb: float
    halos: Sequence[ForegroundHalo]
    dm_mw: float = 55.0
    dm_host_obs: float = 50.0
    measured_dm: Optional[float] = None    # default: MW + cosmic + foreground + host


def plot_sightline(sl: Sightline, cosmo: Optional[Cosmology] = None,
                   mnfw: Optional[ModifiedNFW] = None, cmap: str = "viridis",
                   figsize=(8.8, 8.6)):
    """Render the 4-panel sightline figure; returns the Figure."""
    cosmo = cosmo or Cosmology()
    mnfw = mnfw or ModifiedNFW(cosmo=cosmo)
    H = list(sl.halos)
    z = np.array([h.z for h in H], float)
    M = np.array([h.mass_msun for h in H], float)
    b = np.array([h.impact_kpc for h in H], float)
    R200 = np.array([h.r200_kpc if h.r200_kpc else float(cosmo.r200_kpc(h.mass_msun, h.z)) for h in H])
    DM = np.array([h.dm_pc_cm3 if h.dm_pc_cm3 is not None
                   else float(mnfw.dm_of_b(h.mass_msun, h.z, h.impact_kpc)) for h in H])
    is_cl = np.array([h.is_cluster for h in H])
    pierces = b < R200
    Da = np.array([cosmo.angular_diameter_kpc(zz) for zz in z])
    theta = b / Da * (180 / np.pi) * 60
    theta_r200 = R200 / Da * (180 / np.pi) * 60
    cm = mpl.colormaps[cmap]; znorm = mpl.colors.Normalize(0, sl.z_frb)
    rng = np.random.default_rng(0); phi = rng.uniform(0, 2 * np.pi, len(H))
    xs = np.array([h.dra_arcmin if h.dra_arcmin is not None else theta[i] * np.cos(phi[i])
                   for i, h in enumerate(H)])
    ys = np.array([h.ddec_arcmin if h.ddec_arcmin is not None else theta[i] * np.sin(phi[i])
                   for i, h in enumerate(H)])
    msz = 16 + 20 * (np.log10(M) - 11)

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07, hspace=0.26, wspace=0.26)

    a = ax[0, 0]
    for i in range(len(H)):
        a.add_patch(Circle((xs[i], ys[i]), theta_r200[i], fill=False, ec=cm(znorm(z[i])),
                    lw=1.6 if is_cl[i] else 1.0, ls="-" if pierces[i] else "--", alpha=0.85))
        a.scatter(xs[i], ys[i], s=msz[i], color=cm(znorm(z[i])), edgecolor="k", lw=0.4, zorder=5)
        a.annotate(H[i].label or ("C" if is_cl[i] else str(i + 1)), (xs[i], ys[i]),
                   fontsize=6.0, ha="center", va="center", color="w", zorder=6)
    a.scatter(0, 0, marker="*", s=240, color="crimson", edgecolor="k", lw=0.6, zorder=7)
    if len(H):                       # frame so the full R200 circles fit (no clipping)
        lim = float(1.12 * max(np.max(np.abs(xs) + theta_r200), np.max(np.abs(ys) + theta_r200)))
    else:
        lim = 1.0
    a.add_patch(Ellipse((0, 0), 0.10 * lim, 0.07 * lim, angle=25, fill=False, ec="crimson", lw=1.0))
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel(r"$\Delta$RA (arcmin)"); a.set_ylabel(r"$\Delta$Dec (arcmin)")
    a.set_title("(a) Sky view (C = cluster)")
    a.legend(handles=[Line2D([], [], ls="-", c="0.3", label="within $R_{200}$"),
                      Line2D([], [], ls="--", c="0.3", label="outside $R_{200}$")],
             fontsize=6, loc="upper left")

    bx = ax[0, 1]
    bx.axhline(1.0, color="crimson", lw=1.0, alpha=0.7)
    bx.text(0.002, 1.04, r"$R_{200}$ (intersection below)", fontsize=6.5, color="crimson")
    bx.scatter(z, b / R200, c=z, cmap=cmap, norm=znorm, s=msz + 4, edgecolor="k", lw=0.5, zorder=4)
    bx.scatter(z[pierces], (b / R200)[pierces], s=130, facecolors="none", edgecolors="crimson", lw=1.3, zorder=5)
    bx.axvline(sl.z_frb, color="0.4", ls=":", lw=1.0)
    bx.set_xlabel("Redshift $z$"); bx.set_ylabel(r"$b/R_{200}$"); bx.set_title("(b) Tomography (virial units)")
    bx.set_xlim(0, sl.z_frb * 1.05); bx.set_ylim(0, max(3.2, (b / R200).max() * 1.1))

    cx = ax[1, 0]
    for i in range(len(H)):
        cx.plot([z[i], z[i]], [0, theta_r200[i]], color="0.82", lw=2.2, zorder=1, solid_capstyle="round")
    cx.scatter(z, theta, c=z, cmap=cmap, norm=znorm, s=msz + 4, edgecolor="k", lw=0.5, zorder=4)
    cx.scatter(z[pierces], theta[pierces], s=130, facecolors="none", edgecolors="crimson", lw=1.3, zorder=5)
    cx.set_xlabel("Redshift $z$"); cx.set_ylabel(r"Angular impact $\theta$ (arcmin)")
    cx.set_title(r"(c) Angular view (grey bar = $\theta_{200}$)"); cx.set_xlim(0, sl.z_frb * 1.05); cx.set_ylim(0, None)

    dx = ax[1, 1]
    zg = np.linspace(0, sl.z_frb, 300)
    dmcos = np.array([cosmo.dm_cosmic_mean(zz) for zz in zg])
    pierced_dm = np.where(pierces, DM, 0.0)
    cum_fg = np.array([pierced_dm[z <= zz].sum() for zz in zg])
    meas = sl.measured_dm if sl.measured_dm is not None else (sl.dm_mw + dmcos[-1] + pierced_dm.sum() + sl.dm_host_obs)
    dx.fill_between(zg, 0, sl.dm_mw, color="0.7", alpha=0.5, label="MW (ISM+halo)")
    dx.fill_between(zg, sl.dm_mw, sl.dm_mw + dmcos, color="#1f77b4", alpha=0.35, label="+ cosmic/IGM")
    dx.plot(zg, sl.dm_mw + dmcos + cum_fg, color="k", lw=1.8, label="+ foreground halos")
    dx.axhline(meas, color="crimson", ls="--", lw=1.2); dx.text(0.004, meas + 6, "measured DM", color="crimson", fontsize=7)
    dx.set_xlabel("Redshift $z$"); dx.set_ylabel(r"Cumulative DM (pc cm$^{-3}$)")
    dx.set_title("(d) DM ledger along the sightline"); dx.set_xlim(0, sl.z_frb); dx.legend(fontsize=6.3, loc="upper left")

    for _ax in ax.ravel():           # square boxes so all four panels match (incl. the equal-limit sky view)
        _ax.set_box_aspect(1)
    return fig


def _demo():
    cosmo = Cosmology(); mnfw = ModifiedNFW(cosmo=cosmo)
    rng = np.random.default_rng(8); zf = 0.19
    halos = []
    for i in range(14):
        zz = rng.uniform(0.012, zf - 0.005); M = 10 ** rng.uniform(11.4, 13.3)
        R = float(cosmo.r200_kpc(M, zz)); halos.append(ForegroundHalo(zz, M, R * rng.uniform(0.85, 3.2)))
    Mc = 10 ** 14.5; Rc = float(cosmo.r200_kpc(Mc, 0.083))
    halos.append(ForegroundHalo(0.083, Mc, Rc * 0.45, is_cluster=True, label="C"))
    return Sightline(zf, halos)


if __name__ == "__main__":
    fig = plot_sightline(_demo())
    fig.savefig("sightline_demo.png", bbox_inches="tight", dpi=150)
    print("wrote sightline_demo.png")
