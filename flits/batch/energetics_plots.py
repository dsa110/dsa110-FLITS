"""
energetics_plots.py
==================

FRB energetics figures for multi-band (e.g. CHIME + DSA) samples:

- :func:`plot_spectral_sed`       -- broadband SEDs across the band gap + alpha distribution.
- :func:`plot_population`         -- E_iso vs redshift with completeness horizons, and N(>E).
- :func:`plot_energetics_correlations` -- E_iso vs DM_excess / tau / dnu / DM_host (Spearman).
- :func:`plot_occupancy_kcorr`    -- band-energy ratio, and the gap k-correction systematic.

Real-data interface
-------------------
Fill a :class:`BurstSample` with arrays from your fluence/energy/redshift tables
and the propagation measurements; missing fields only disable the figures that
need them. Detection flags ``det_chime`` / ``det_dsa`` come from your search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from .dm_models import Cosmology
except ImportError:
    from dm_models import Cosmology

__all__ = ["BurstSample", "plot_spectral_sed", "plot_population",
           "plot_energetics_correlations", "plot_occupancy_kcorr"]

_ACMAP, _ANORM = "coolwarm", mpl.colors.Normalize(-3.2, 1.2)


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass
class BurstSample:
    z: np.ndarray
    e_iso: np.ndarray                 # erg
    alpha: np.ndarray                 # spectral index F_nu ~ nu^alpha
    f_chime: Optional[np.ndarray] = None    # Jy ms
    f_dsa: Optional[np.ndarray] = None
    tau_ms: Optional[np.ndarray] = None
    dnu_mhz: Optional[np.ndarray] = None
    dm_excess: Optional[np.ndarray] = None
    dm_host: Optional[np.ndarray] = None
    det_chime: Optional[np.ndarray] = None
    det_dsa: Optional[np.ndarray] = None
    nu_chime_mhz: float = 600.0
    nu_dsa_mhz: float = 1400.0


def plot_spectral_sed(bs: BurstSample, bands=((400, 800), (1280, 1530)), gap=(800, 1280),
                      figsize=(9.8, 3.8)):
    fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.7, 1]))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.16, wspace=0.28)
    nlo = np.linspace(*bands[0], 50); nhi = np.linspace(*bands[1], 50)
    ref = bs.f_chime if bs.f_chime is not None else np.ones_like(bs.z)
    for i in range(len(bs.z)):
        col = mpl.colormaps[_ACMAP](_ANORM(bs.alpha[i]))
        ax[0].plot(nlo, ref[i] * (nlo / bs.nu_chime_mhz) ** bs.alpha[i], color=col, lw=1.0, alpha=0.85)
        ax[0].plot(nhi, ref[i] * (nhi / bs.nu_chime_mhz) ** bs.alpha[i], color=col, lw=1.0, alpha=0.85)
    ax[0].axvspan(*gap, color="0.93", hatch="////", ec="0.6", lw=0, zorder=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Frequency (MHz)"); ax[0].set_ylabel(r"Spectral fluence $F_\nu$ (Jy ms)")
    ax[0].set_title("(a) Broadband SEDs across the lever arm")
    sm = plt.cm.ScalarMappable(cmap=_ACMAP, norm=_ANORM); sm.set_array([])
    fig.colorbar(sm, ax=ax[0], pad=0.01).set_label(r"spectral index $\alpha$", fontsize=8)
    ax[1].hist(bs.alpha, bins=8, color="#4c72b0", edgecolor="k", alpha=0.85)
    ax[1].axvline(np.median(bs.alpha), color="crimson", lw=1.4, ls="--",
                  label=r"median $\alpha=%.1f$" % np.median(bs.alpha))
    ax[1].set_xlabel(r"spectral index $\alpha$"); ax[1].set_ylabel("number of bursts")
    ax[1].set_title("(b) Spectral-index distribution"); ax[1].legend(fontsize=7)
    return fig


def plot_population(bs: BurstSample, fth_chime=0.9, fth_dsa=0.7, band_hz=600e6,
                    cosmo: Optional[Cosmology] = None, figsize=(10.2, 3.9)):
    cosmo = cosmo or Cosmology()
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.88, bottom=0.16, wspace=0.27)
    zz = np.linspace(0.02, max(0.6, bs.z.max() * 1.1), 200)
    dLz = np.array([cosmo.luminosity_distance_cm(z_) for z_ in zz])
    Eh_ch = 1e-23 * (fth_chime * 1e-3) * 4 * np.pi * dLz ** 2 * band_hz / (1 + zz)
    Eh_dsa = (fth_dsa / fth_chime) * 1.8 * Eh_ch
    Eh_ch_b = np.interp(bs.z, zz, Eh_ch); Eh_dsa_b = np.interp(bs.z, zz, Eh_dsa)
    dC = bs.det_chime if bs.det_chime is not None else bs.e_iso > Eh_ch_b
    dD = bs.det_dsa if bs.det_dsa is not None else bs.e_iso > Eh_dsa_b
    sel = dC | dD
    for c, col, m in [("both-band", "k", sel & dC & dD), ("CHIME-only", "#1f77b4", sel & dC & ~dD),
                      ("DSA-only", "#d62728", sel & ~dC & dD)]:
        if np.any(m):
            ax[0].scatter(bs.z[m], bs.e_iso[m], s=42, color=col, edgecolor="k", lw=0.4, label=c, zorder=4)
    ax[0].plot(zz, Eh_ch, color="#1f77b4", lw=1.5, ls="--", label="CHIME completeness")
    ax[0].plot(zz, Eh_dsa, color="#d62728", lw=1.5, ls=":", label="DSA completeness")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Redshift $z$"); ax[0].set_ylabel(r"$E_{\rm iso}$ (erg)")
    ax[0].set_title("(a) Energy–distance plane + horizons"); ax[0].legend(fontsize=6.5, loc="lower right")
    Es = np.sort(bs.e_iso[sel])[::-1]; Ng = np.arange(1, int(sel.sum()) + 1)
    ax[1].step(Es, Ng, where="post", color="k", lw=1.6, zorder=4)
    ax[1].errorbar(Es, Ng, yerr=np.sqrt(Ng), fmt="o", ms=3, color="k", elinewidth=0.7, capsize=1.5, zorder=5)
    if len(Es) > 4:
        m = (Es > np.percentile(Es, 12)) & (Es < np.percentile(Es, 92))
        g = np.polyfit(np.log10(Es[m]), np.log10(Ng[m]), 1)
        xf = np.array([Es.min(), Es.max()])
        ax[1].plot(xf, 10 ** g[1] * xf ** g[0], color="crimson", lw=1.3, ls="--",
                   label=r"$N(>E)\propto E^{%.2f}$" % g[0]); ax[1].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$E_{\rm iso}$ (erg)"); ax[1].set_ylabel(r"$N(>E)$"); ax[1].set_title("(b) Cumulative energy distribution")
    return fig


def plot_energetics_correlations(bs: BurstSample, figsize=(8.2, 6.6)):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.09, hspace=0.32, wspace=0.30)
    panels = [(bs.dm_excess, r"DM$_{\rm excess}$ (pc cm$^{-3}$)"), (bs.tau_ms, r"$\tau_{1\,\rm GHz}$ (ms)"),
              (bs.dnu_mhz, r"$\Delta\nu_{\rm DC}$ (MHz)"), (bs.dm_host, r"DM$_{\rm host}$ (pc cm$^{-3}$)")]
    for a, (xq, lab) in zip(ax.ravel(), panels):
        if xq is None:
            a.text(0.5, 0.5, "no data", transform=a.transAxes, ha="center"); a.set_xlabel(lab); continue
        a.scatter(xq, bs.e_iso, s=36, c=bs.alpha, cmap=_ACMAP, norm=_ANORM, edgecolor="k", lw=0.4)
        a.set_yscale("log"); a.set_xlabel(lab); a.set_ylabel(r"$E_{\rm iso}$ (erg)")
        rho = _spearman(xq, np.log10(bs.e_iso))
        a.text(0.04, 0.93, r"$\rho_s=%.2f$" % rho, transform=a.transAxes, fontsize=8, va="top",
               bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.85))
    fig.suptitle(r"Energetics vs propagation  (colour = $\alpha$)", fontsize=9.5)
    return fig


def plot_occupancy_kcorr(bs: BurstSample, full_band=(400.0, 1530.0), alpha_meas=-1.0, alpha_err=0.15,
                         figsize=(10.0, 3.9)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.88, bottom=0.16, wspace=0.27)
    if bs.f_chime is not None and bs.f_dsa is not None:
        ratio = np.log10(bs.f_chime / bs.f_dsa)
        both = (bs.det_chime if bs.det_chime is not None else np.ones(len(bs.z), bool)) & \
               (bs.det_dsa if bs.det_dsa is not None else np.ones(len(bs.z), bool))
        ax[0].axhspan(-0.3, 0.3, color="0.9", zorder=0); ax[0].axhline(0, color="0.5", lw=0.8)
        ax[0].scatter(bs.e_iso[both], ratio[both], s=42, c=bs.alpha[both], cmap=_ACMAP, norm=_ANORM,
                      edgecolor="k", lw=0.4, zorder=4)
        for i in np.where(~both)[0]:
            yl = 1.4 if (bs.det_chime is None or bs.det_chime[i]) else -1.4
            ax[0].scatter(bs.e_iso[i], yl, marker="v" if yl > 0 else "^", s=40, color="0.5", zorder=4)
        ax[0].set_xscale("log"); ax[0].set_ylim(-1.8, 1.8)
        ax[0].set_xlabel(r"$E_{\rm iso}$ (erg)"); ax[0].set_ylabel(r"$\log_{10}(F_{\rm CHIME}/F_{\rm DSA})$")
        ax[0].set_title("(a) Spectral occupancy (grey = broadband)")
    else:
        ax[0].text(0.5, 0.5, "needs f_chime & f_dsa", transform=ax[0].transAxes, ha="center")

    def Eband(a_, lo=full_band[0], hi=full_band[1], nu0=bs.nu_chime_mhz):
        nu = np.linspace(lo, hi, 400); return np.trapezoid((nu / nu0) ** a_, nu)
    ag = np.linspace(-3.2, 1.2, 200); Erel = np.array([Eband(a) for a in ag]) / Eband(alpha_meas)
    ax[1].plot(ag, Erel, color="k", lw=1.7)
    ax[1].axvspan(-3.0, 1.0, color="#d62728", alpha=0.12)
    ax[1].text(0.95, 0.52, "single-band\nprior range", transform=ax[1].transAxes, color="#d62728",
               fontsize=6.8, ha="right", va="center")
    ax[1].axvspan(alpha_meas - alpha_err, alpha_meas + alpha_err, color="#2ca02c", alpha=0.5)
    ax[1].axvline(alpha_meas, color="#2ca02c", lw=1.4)
    ax[1].annotate("two-band\nmeasured $\\alpha$", xy=(alpha_meas, 1.0), xycoords="data",
                   xytext=(0.07, 0.28), textcoords="axes fraction", color="#2ca02c", fontsize=7,
                   arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"assumed spectral index $\alpha$")
    ax[1].set_ylabel(r"inferred $E_{\rm band}/E(\alpha_{\rm meas})$"); ax[1].set_title("(b) Energy k-correction across the gap")
    return fig


def _demo():
    rng = np.random.default_rng(12); N = 15; cosmo = Cosmology()
    z = np.sort(rng.uniform(0.035, 0.55, N))
    dLcm = np.array([cosmo.luminosity_distance_cm(zz) for zz in z])
    alpha = rng.normal(-1.0, 1.3, N)
    f_ch = (0.6 ** (1 - 1.7) + rng.uniform(0, 1, N) * (80.0 ** (1 - 1.7) - 0.6 ** (1 - 1.7))) ** (1 / (1 - 1.7))
    f_dsa = f_ch * (1400 / 600.0) ** alpha
    e_iso = 1e-23 * (f_ch * 1e-3) * 4 * np.pi * dLcm ** 2 * 600e6 / (1 + z)
    tau = 10 ** rng.uniform(-1, 0.7, N); dnu = (1 / (2 * np.pi * tau)); dnu = dnu / dnu.max() * 40 + 0.5
    dm_host = 10 ** rng.uniform(1.3, 2.45, N); dm_exc = dm_host + rng.normal(0, 18, N)
    return BurstSample(z=z, e_iso=e_iso, alpha=alpha, f_chime=f_ch, f_dsa=f_dsa, tau_ms=tau,
                       dnu_mhz=dnu, dm_excess=dm_exc, dm_host=dm_host,
                       det_chime=f_ch > 0.9, det_dsa=f_dsa > 0.7)


if __name__ == "__main__":
    bs = _demo()
    for fn, name in [(plot_spectral_sed, "sed"), (plot_population, "pop"),
                     (plot_energetics_correlations, "corr"), (plot_occupancy_kcorr, "occ")]:
        fn(bs).savefig(f"energetics_{name}_demo.png", bbox_inches="tight", dpi=150)
    print("wrote energetics_*_demo.png")
