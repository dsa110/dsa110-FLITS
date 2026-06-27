"""
dm_budget_plots.py
=================

DM-budget figures:

- :func:`plot_halo_profiles` -- modified-NFW gas density n_e(r), the sightline
  DM(b), and the cumulative line-of-sight DM, for a family of halo masses.
- :func:`plot_dm_budget` -- the DM budget as component PDFs, a "violin"
  decomposition, and the derived host-DM posterior.

Real-data interface
-------------------
:func:`plot_halo_profiles` uses the shared :class:`dm_models.ModifiedNFW`
(swap in your calibrated parameters). :func:`plot_dm_budget` consumes
**Monte-Carlo samples** per component -- e.g. ``{"MW": mw_draws, "Cosmic/IGM":
igm_draws, "Foreground": fg_draws, "Host": host_draws}`` -- which is what a
posterior DM-budget analysis naturally produces. The total and the derived
host posterior are then computed by sample arithmetic (no Gaussian assumption).
"""
from __future__ import annotations

from typing import Optional, Sequence, Mapping

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from .dm_models import Cosmology, ModifiedNFW
except ImportError:
    from dm_models import Cosmology, ModifiedNFW

__all__ = ["plot_halo_profiles", "plot_dm_budget"]

_MASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def plot_halo_profiles(masses: Sequence[float], z: float = 0.1,
                       mnfw: Optional[ModifiedNFW] = None,
                       overlay: Optional[Sequence[tuple]] = None,
                       figsize=(11.0, 3.7)):
    """n_e(r), DM(b), cumulative DM for a family of halo masses.

    overlay : optional list of (b_over_r200, dm_pc_cm3, z_color) points for panel (b).
    """
    mnfw = mnfw or ModifiedNFW()
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.16, wspace=0.30)
    rr = np.linspace(0.01, 1.0, 250)
    for M, col in zip(masses, _MASS_COLORS):
        R200 = float(mnfw.cosmo.r200_kpc(M, z))
        rphys = np.linspace(5.0, R200, 250)
        ax[0].plot(rphys, mnfw.ne(M, z, rphys), color=col, lw=1.7, label=r"$10^{%.1f}$" % np.log10(M))
        ax[1].plot(rr, mnfw.dm_of_b(M, z, rr * R200), color=col, lw=1.7)
        rfrac, cum = mnfw.dm_cumulative_los(M, z)
        ax[2].plot(rfrac, cum / cum[-1], color=col, lw=1.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$r$ (kpc)"); ax[0].set_ylabel(r"$n_e(r)$ (cm$^{-3}$)")
    ax[0].set_title("(a) mNFW gas density")
    ax[0].legend(fontsize=6.5, title=r"$M_{200}/M_\odot$", title_fontsize=6.5); ax[0].grid(alpha=0.25, which="both")
    if overlay is not None and len(overlay):
        ov = np.array(overlay, float)
        ax[1].scatter(ov[:, 0], ov[:, 1], c=ov[:, 2], cmap="viridis", s=40, edgecolor="k", lw=0.5, zorder=5)
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$b/R_{200}$"); ax[1].set_ylabel(r"DM$(b)$ (pc cm$^{-3}$)")
    ax[1].set_title("(b) LOS DM vs impact parameter"); ax[1].grid(alpha=0.25, which="both"); ax[1].set_ylim(0.5, None)
    ax[2].axhline(0.5, color="0.6", ls=":", lw=0.8); ax[2].axhline(0.9, color="0.6", ls=":", lw=0.8)
    ax[2].set_xlabel(r"LOS radius $r/R_{200}$"); ax[2].set_ylabel("cumulative DM fraction")
    ax[2].set_title("(c) Where the DM accumulates"); ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1.02)
    return fig


def plot_dm_budget(components: Mapping[str, np.ndarray], dm_obs: Optional[float] = None,
                   host_key: str = "Host", figsize=(11.2, 3.8), nbins=60):
    """Component PDFs, violin budget, and derived host-DM posterior from MC samples."""
    names = list(components)
    samples = {k: np.asarray(v, float) for k, v in components.items()}
    n = min(len(v) for v in samples.values())
    total = np.sum([samples[k][:n] for k in names], axis=0)
    if dm_obs is None:
        dm_obs = float(np.median(total))
    hi_x = max(np.percentile(total, 99.5), dm_obs * 1.1)
    grid = np.linspace(0, hi_x, 400)
    palette = ["#7f7f7f", "#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#17becf"]

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.16, wspace=0.3)

    # (a) component PDFs + total
    for (k, col) in zip(names, palette):
        h, edges = np.histogram(samples[k], bins=nbins, range=(0, hi_x), density=True)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ax[0].fill_between(ctr, h, color=col, alpha=0.3); ax[0].plot(ctr, h, color=col, lw=1.2, label=k)
    ht, et = np.histogram(total, bins=nbins, range=(0, hi_x), density=True)
    ax[0].plot(0.5 * (et[1:] + et[:-1]), ht, color="k", lw=2.0, label="Total")
    ax[0].axvline(dm_obs, color="crimson", ls="--", lw=1.5)
    ax[0].set_xlabel(r"DM (pc cm$^{-3}$)"); ax[0].set_ylabel("prob. density")
    ax[0].set_title("(a) Component PDFs"); ax[0].legend(fontsize=6.2, loc="upper right")

    # (b) violin budget
    data = [samples[k] for k in names] + [total]
    labels = names + ["Total"]
    parts = ax[1].violinplot(data, positions=range(1, len(data) + 1), showmedians=True, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor((palette + ["k"])[i]); pc.set_alpha(0.45)
    ax[1].axhline(dm_obs, color="crimson", ls="--", lw=1.3, zorder=0)
    ax[1].set_xticks(range(1, len(data) + 1)); ax[1].set_xticklabels(labels, fontsize=6.8, rotation=20)
    ax[1].set_ylabel(r"DM (pc cm$^{-3}$)"); ax[1].set_title("(b) Budget 'violins'")

    # (c) derived host-DM posterior
    if host_key in samples:
        others = np.sum([samples[k][:n] for k in names if k != host_key], axis=0)
        host_post = dm_obs - others
        host_post = host_post[host_post > -50]
        lo, med, hi = np.percentile(host_post, [16, 50, 84])
        ax[2].hist(host_post, bins=50, density=True, color="#2ca02c", alpha=0.35)
        ax[2].axvline(med, color="#2ca02c", lw=1.4)
        ax[2].axvspan(lo, hi, color="#2ca02c", alpha=0.18, label="68% credible")
        ax[2].set_title("(c) Derived host-DM posterior")
        ax[2].text(0.96, 0.85, r"$%.0f^{+%.0f}_{-%.0f}$" % (med, hi - med, med - lo),
                   transform=ax[2].transAxes, ha="right", fontsize=11, color="#2ca02c")
        ax[2].legend(fontsize=7, loc="upper left")
    else:
        ax[2].text(0.5, 0.5, "no '%s' component" % host_key, transform=ax[2].transAxes, ha="center")
    ax[2].set_xlabel(r"DM$_{\rm host}$ (obs frame, pc cm$^{-3}$)"); ax[2].set_ylabel("prob. density")
    return fig


def _demo():
    rng = np.random.default_rng(3); N = 60000
    comps = {
        "MW": np.abs(rng.normal(55, 14, N)),
        "Cosmic/IGM": rng.lognormal(np.log(165), 0.5, N),
        "Foreground": rng.lognormal(np.log(110), 0.55, N),
        "Host": rng.lognormal(np.log(48, ), 0.6, N),
    }
    fig1 = plot_halo_profiles([1e12, 1e13, 1e14, 10 ** 14.5], z=0.1)
    fig2 = plot_dm_budget(comps, dm_obs=380.0, host_key="Host")
    return fig1, fig2


if __name__ == "__main__":
    f1, f2 = _demo()
    f1.savefig("dm_profiles_demo.png", bbox_inches="tight", dpi=150)
    f2.savefig("dm_budget_demo.png", bbox_inches="tight", dpi=150)
    print("wrote dm_profiles_demo.png, dm_budget_demo.png")
