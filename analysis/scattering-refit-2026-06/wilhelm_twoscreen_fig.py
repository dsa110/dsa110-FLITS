#!/usr/bin/env python
"""Wilhelm two-screen science figure (paper draft).

Panel (a): the scattering<->scintillation reciprocity plane. For a SINGLE thin
screen 2*pi*tau*dnu_d = C1 ~ 1 (Cordes & Rickett 1998; Lambert & Rickett 1999,
C1 in [0.5,2]). Wilhelm's measured pulse-broadening tau (CHIME-weighted joint
fit) and its DSA-resolved scintillation bandwidth dnu_d land FAR off that locus
-> the tau-screen and the dnu_d-screen are DIFFERENT screens (>=2 along the LOS).

Panel (b): the two-screen mutual-coherence limit (Nimmo et al. 2025, Eq. 10).
Two resolved, mutually coherent DSA scintillation scales (narrow ~0.12 MHz, broad
~5 MHz) bound the product d_obs,s1 * d_s2,src. For an assumed Milky-Way (near)
screen distance d_MW this caps how far the near-SOURCE screen can sit from the
FRB, d_s2,src. Shown as the excluded region; the bound scales with the (currently
unmeasured) host distance, so the curve is drawn for a fiducial z and annotated.

Numbers + provenance (all in-repo unless flagged):
  tau_1ghz, alpha : wilhelm joint M3 fit (12-vector), data/joint/wilhelm_joint_fit.json
                    alpha=2.69+-0.05, tau_1ghz=0.26 ms  (SCINT_INTEGRATION_PLAN.md)
  dnu_d (DSA)     : scintillation/configs/bursts/wilhelm_dsa.yaml stored_fits
                    (native-res ACF; Lorentzian+Lorentzian; narrow l_1_gamma,
                    broad l_2_gamma; per-subband medians).
  dnu_d (CHIME)   : 0.060 MHz @ 0.684 GHz -- BELOW CHIME native 0.39 MHz channel
                    => UNRESOLVED, shown as an upper limit only.
  l,b             : 107.14, 16.69 (astropy, ICRS->Galactic).
  FIDUCIAL/FLAGGED: host redshift z (no in-repo measurement) -> z~0.47 from a
                    Macquart-relation DM estimate; NE2001 MW dnu_d (~few MHz at
                    1.4 GHz) not yet run (mwprop absent). Both annotated as such.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from ne2025_sightline import at_freq, mw_scattering  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
    }
)

# ---- wilhelm measured quantities -------------------------------------------
L_WILHELM, B_WILHELM = 107.135, 16.691  # Galactic l,b (astropy, ICRS->Galactic)
TAU_1GHZ_MS = 0.26  # joint M3 fit
ALPHA = 2.69  # MEASURED burst scattering index -- NOT the Kolmogorov 4.4 the
# NE2025 MW-floor scaling (ne2025_sightline.at_freq) uses. Do not "reconcile" them.
F_CHIME, F_DSA = 0.684, 1.405  # GHz, band medians
# DSA native-res ACF (wilhelm_dsa.yaml stored_fits), per-subband narrow/broad:
DSA_NARROW = np.array([0.1166, 0.0600, 0.1247, 0.1772])  # MHz (l_1_gamma)
DSA_BROAD = np.array([5.211, 1.527, 8.940, 5.388])  # MHz (l_2_gamma)
DNU_DSA_NARROW = float(np.median(DSA_NARROW))  # ~0.12 MHz
DNU_DSA_BROAD = float(np.median(DSA_BROAD))  # ~5.3 MHz
DNU_CHIME = 0.060  # MHz @0.684 GHz -- UNRESOLVED (< 0.39 MHz native) -> upper limit
CHIME_NATIVE = 0.390625  # MHz
C1_THIN = 0.957  # Cordes&Rickett thin-screen
C1_LO, C1_HI = 0.5, 2.0


def tau_at(fghz):
    return TAU_1GHZ_MS * fghz ** (-ALPHA)  # ms


def dnu_from_tau(tau_ms, C1=C1_THIN):
    # dnu_d = C1/(2 pi tau)  -> MHz
    return C1 / (2 * np.pi * tau_ms * 1e-3) / 1e6


def tau_from_dnu(dnu_mhz, C1=C1_THIN):
    return C1 / (2 * np.pi * dnu_mhz * 1e6) * 1e3  # ms


def fig():
    figw = plt.figure(figsize=(7.2, 3.3))
    gs = figw.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.32)
    ax_a = figw.add_subplot(gs[0, 0])
    ax_b = figw.add_subplot(gs[0, 1])

    # ===== Panel (a): reciprocity plane =====
    tau_grid = np.logspace(-3, 1.0, 200)  # ms  (1 us .. 10 ms)
    line = dnu_from_tau(tau_grid, C1_THIN)
    ax_a.fill_between(
        tau_grid,
        dnu_from_tau(tau_grid, C1_LO),
        dnu_from_tau(tau_grid, C1_HI),
        color="0.85",
        zorder=0,
        label=r"single screen $C_1\in[0.5,2]$",
    )
    ax_a.plot(tau_grid, line, "k-", lw=1.0, zorder=1)

    # measured pulse-broadening tau per band (x) with the DSA-resolved dnu_d (y)
    tC, tD = tau_at(F_CHIME), tau_at(F_DSA)
    # DSA: tau(DSA) vs resolved narrow & broad scintillation
    ax_a.scatter(
        [tD],
        [DNU_DSA_NARROW],
        s=55,
        marker="o",
        fc="#c1272d",
        ec="k",
        lw=0.7,
        zorder=5,
        label="DSA scint (narrow)",
    )
    ax_a.scatter(
        [tD],
        [DNU_DSA_BROAD],
        s=55,
        marker="s",
        fc="#e08214",
        ec="k",
        lw=0.7,
        zorder=5,
        label="DSA scint (broad)",
    )
    # CHIME: dnu_d unresolved -> upper limit (down arrow at native-channel ceiling)
    ax_a.errorbar(
        [tC],
        [CHIME_NATIVE],
        yerr=[[CHIME_NATIVE * 0.55], [0]],
        uplims=True,
        fmt="v",
        ms=6,
        mfc="#3a6",
        mec="k",
        ecolor="#3a6",
        elinewidth=1.0,
        capsize=0,
        zorder=5,
        label="CHIME scint (unresolved)",
    )

    # connect each band's tau to where the single-screen locus predicts its dnu_d
    for tb, fb in [(tD, "DSA"), (tC, "CHIME")]:
        ax_a.plot(
            [tb, tb],
            [dnu_from_tau(tb), DNU_DSA_NARROW if fb == "DSA" else CHIME_NATIVE],
            color="0.5",
            ls=":",
            lw=0.8,
            zorder=2,
        )
        ax_a.scatter([tb], [dnu_from_tau(tb)], s=18, marker="x", color="k", zorder=4)

    ax_a.annotate(
        r"$2\pi\,\tau\,\Delta\nu_d \approx 10^{2}$" "\n(off single-screen)",
        xy=(tD, DNU_DSA_NARROW),
        xytext=(tD * 2.0, DNU_DSA_NARROW * 6),
        fontsize=7.5,
        ha="left",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"),
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"pulse-broadening $\tau$  (ms)")
    ax_a.set_ylabel(r"scintillation bandwidth $\Delta\nu_d$  (MHz)")
    ax_a.set_xlim(8e-3, 3)
    ax_a.set_ylim(3e-4, 30)
    ax_a.legend(fontsize=6.3, loc="lower right", framealpha=0.9, handletextpad=0.4)
    ax_a.text(0.03, 0.95, "(a)", transform=ax_a.transAxes, fontweight="bold", va="top")
    ax_a.set_title("scattering--scintillation reciprocity", fontsize=8.5)

    # ===== Panel (b): inferred line-of-sight screen geometry =====
    # The reciprocity break in (a) demands >=2 screens. The broad RESOLVED DSA
    # scintillation (~5 MHz) matches a Milky-Way screen; the dominant pulse-
    # broadening tau marks a SECOND, strongly-scattering screen near the source.
    # Nimmo+25 Eq.10 turns the two mutually-coherent DSA scales into a bound on the
    # near-source screen distance -- for wilhelm the broad scintles make it WEAK,
    # so we draw it as an *allowed zone*, not a hard wall.
    # NE2025 MW prediction for this sightline anchors the NEAR screen (no longer assumed).
    mw = mw_scattering(L_WILHELM, B_WILHELM)
    sbw_mw_dsa = at_freq(mw["sbw_1ghz_mhz"], F_DSA, "sbw")  # MHz @ DSA
    tau_mw_1ghz_us = mw["tau_1ghz_ms"] * 1e3  # us @ 1 GHz (MW pulse-broadening floor)
    tau_excess = (TAU_1GHZ_MS * 1e3) / tau_mw_1ghz_us  # measured / MW floor @ 1 GHz
    d_mw_kpc = mw["d_eff_kpc"]  # NE2025 effective MW screen distance

    z_fid = 0.47
    D_A_src_mpc = 1330.0  # D_A(z=0.47), Planck18 fiducial (FLAGGED -- no in-repo z)
    nu_hz = F_DSA * 1e9
    dnu1, dnu2 = DNU_DSA_BROAD * 1e6, DNU_DSA_NARROW * 1e6
    d_src_kpc = D_A_src_mpc * 1e3
    d_prod_kpc2 = dnu1 * dnu2 * d_src_kpc**2 / (1.0 * 1.0 * nu_hz**2)
    d_s2_max_mpc = d_prod_kpc2 / d_mw_kpc / 1e3  # upper limit, Mpc (NE2025-anchored d_MW)

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    yc = 0.56
    x_earth, x_mw, x_src, x_far = 0.04, 0.135, 0.86, 0.30
    # line of sight + broken-distance marker (kpc near-zone -> Gpc extragalactic)
    ax_b.plot([x_earth, 0.93], [yc, yc], color="0.55", lw=0.8, zorder=1)
    for xb in (0.225, 0.245):
        ax_b.plot([xb - 0.012, xb + 0.012], [yc - 0.03, yc + 0.03], color="0.55", lw=0.9, zorder=2)
    # multipath cone (source scatters, rays reconverge at Earth)
    for sgn in (+1, -1):
        ax_b.plot(
            [x_src, 0.84, x_mw, x_earth],
            [yc, yc + sgn * 0.05, yc + sgn * 0.115, yc],
            color="#c1272d",
            lw=0.5,
            alpha=0.45,
            zorder=1,
        )
    # Earth / telescopes
    ax_b.scatter([x_earth], [yc], marker="*", s=130, fc="#1fb05a", ec="k", lw=0.6, zorder=4)
    ax_b.text(x_earth, yc - 0.10, "Earth\nCHIME+DSA", ha="center", va="top", fontsize=6.5)
    # MW screen (near) -- broad RESOLVED scintillation
    ax_b.add_patch(
        Rectangle(
            (x_mw - 0.006, yc - 0.16),
            0.012,
            0.32,
            fc="#4575b4",
            ec="k",
            lw=0.5,
            alpha=0.85,
            zorder=3,
        )
    )
    ax_b.text(x_mw, yc + 0.205, "MW screen", ha="center", fontsize=7, color="#2c5aa0")
    ax_b.text(
        x_mw,
        yc - 0.205,
        rf"NE2025: $d_{{\rm eff}}\!=\!{d_mw_kpc:.1f}$ kpc"
        "\n"
        rf"$\Delta\nu_d^{{\rm MW}}\!\approx\!{sbw_mw_dsa:.1f}$ MHz @1.4 GHz"
        "\n"
        r"(cf. broad 5.3 MHz scintle)",
        ha="center",
        va="top",
        fontsize=6.0,
        color="0.25",
    )
    # host galaxy + source
    ax_b.add_patch(
        Ellipse((0.875, yc), 0.135, 0.34, fc="#fee08b", ec="0.6", lw=0.5, alpha=0.55, zorder=1)
    )
    ax_b.scatter([0.905], [yc], marker="*", s=95, fc="w", ec="#c1272d", lw=1.1, zorder=5)
    ax_b.text(
        0.905,
        yc + 0.29,
        "host FRB" "\n" rf"$z\!\approx\!{z_fid}$ (est.)",
        ha="center",
        fontsize=6.5,
    )
    # near-source scattering screen (dominant tau) + allowed-distance zone
    span = x_src - x_far  # screen-x width <-> D_A_src_mpc
    x_lim = x_src - span * min(1.0, d_s2_max_mpc / D_A_src_mpc)
    ax_b.add_patch(
        Rectangle(
            (x_lim, yc - 0.13), x_src - x_lim, 0.26, fc="#c1272d", ec="none", alpha=0.10, zorder=0
        )
    )
    ax_b.add_patch(
        Rectangle(
            (x_src - 0.006, yc - 0.16),
            0.012,
            0.32,
            fc="#c1272d",
            ec="k",
            lw=0.5,
            alpha=0.85,
            zorder=3,
        )
    )
    ax_b.annotate(
        "",
        xy=(x_lim, yc - 0.20),
        xytext=(x_src, yc - 0.20),
        arrowprops=dict(arrowstyle="<->", lw=0.7, color="#7a1519"),
    )
    ax_b.text(
        (x_lim + x_src) / 2,
        yc - 0.225,
        rf"$\tau$-screen $\lesssim\!{d_s2_max_mpc:.0f}$ Mpc from src"
        "\n"
        r"(coherence limit, NE2025 $d_{\rm MW}$)",
        ha="center",
        va="top",
        fontsize=6.0,
        color="#7a1519",
    )
    ax_b.text(0.81, yc + 0.205, r"host $\tau$-screen", ha="center", fontsize=6.5, color="#7a1519")
    # NE2025 excess: the headline result, given its own line along the bottom.
    ax_b.text(
        0.5,
        0.05,
        rf"$\tau_{{1\rm GHz}}\!=\!{TAU_1GHZ_MS * 1e3:.0f}\,\mu$s $\gg$ NE2025 MW floor "
        rf"${tau_mw_1ghz_us:.1f}\,\mu$s ($\times{tau_excess:.0f}$)  $\Rightarrow$  "
        r"$\tau$-screen is extragalactic",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#7a1519",
    )
    ax_b.text(0.02, 0.99, "(b)", transform=ax_b.transAxes, fontweight="bold", va="top")
    ax_b.set_title("inferred line-of-sight screen geometry", fontsize=8.5)

    figw.suptitle(
        rf"FRB 20221203A (wilhelm):  $\ell={L_WILHELM:.1f}^\circ,\ b={B_WILHELM:.1f}^\circ$,  "
        rf"DM$=602.3$,  $\alpha_\tau={ALPHA:.2f}$,  $\tau_{{1\rm GHz}}={TAU_1GHZ_MS:.2f}$ ms"
        "   |   two screens: $\\Delta\\nu_d$-screen $\\neq$ $\\tau$-screen",
        fontsize=8.5,
        y=1.02,
    )
    figw.tight_layout()
    return figw, dict(
        d_prod_kpc2=d_prod_kpc2,
        d_mw_kpc=d_mw_kpc,
        d_s2_max_mpc=d_s2_max_mpc,
        tau_excess=tau_excess,
        sbw_mw_dsa=sbw_mw_dsa,
    )


def main():
    out = os.path.join(os.path.dirname(__file__), "figures", "wilhelm")
    os.makedirs(out, exist_ok=True)
    f, info = fig()
    png = os.path.join(out, "wilhelm_twoscreen.png")
    f.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"d_product_kpc2 = {info['d_prod_kpc2']:.3e}  (d_s2,src <= d_prod/d_MW)")
    print(
        f"  NE2025 d_MW={info['d_mw_kpc']:.2f} kpc -> d_s2,src <= {info['d_s2_max_mpc']:.1f} Mpc "
        f"(z=0.47 fiducial)"
    )
    print(
        f"  MW Dnu_d @DSA = {info['sbw_mw_dsa']:.2f} MHz ; tau excess = {info['tau_excess']:.0f}x"
    )
    print(f"  C_implied (DSA) = {2 * np.pi * tau_at(F_DSA) * 1e-3 * DNU_DSA_NARROW * 1e6:.1f}")
    print(f"  C_implied (CHIME)= {2 * np.pi * tau_at(F_CHIME) * 1e-3 * DNU_CHIME * 1e6:.1f}")


if __name__ == "__main__":
    main()
