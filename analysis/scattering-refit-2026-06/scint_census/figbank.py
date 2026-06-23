"""Paper-ready figure bank for Faber2026 (CHIME-DSA co-detection scattering paper).

Renders publication PDFs (vector, serif, AASTeX column widths) encoding this
campaign's results, with a manifest for the figure-review gate. Run from a dir
WITHOUT the repo's matplotlibrc (its cycler line is malformed); style is set
explicitly here so it does not depend on any rc file.

  python figbank.py            # writes figbank/*.pdf + figures.manifest.json
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

NL = os.path.dirname(os.path.abspath(__file__))
SC = json.load(open(f"{NL}/data/scint/wilhelm_scint_subband.json"))
NE = json.load(open(f"{NL}/data/scint/wilhelm_ne2025_floor.json"))
OUT = f"{NL}/figbank"
os.makedirs(OUT, exist_ok=True)

# AASTeX-ish publication style, self-contained (no rc dependence). DejaVu Serif is
# always present so Greek mathtext (Delta nu_d) renders instead of glyph boxes.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": cycler(
            color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747"]
        ),
    }
)

TAU1, ALPHA = SC["scattering"]["tau_1ghz_ms"], SC["scattering"]["alpha"]
manifest = {"figures": []}


def _save(fig, name, expectation):
    p = f"{OUT}/{name}.pdf"
    fig.savefig(p)
    fig.savefig(f"{OUT}/{name}.png", dpi=150)  # PNG twin for the figure-review gate
    plt.close(fig)
    manifest["figures"].append(
        {"file": f"{name}.png", "pdf": f"{name}.pdf", "expectation": expectation}
    )
    print(f"wrote {p}")


def fig_scint():
    """Delta-nu_d vs nu: measured + scattering-screen + NE2025 MW floor (multi-screen)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rows = SC["subbands"]
    dsa = [r for r in rows if r["telescope"] == "DSA" and r["resolved"]]
    chime = [r for r in rows if r["telescope"] == "CHIME" and r.get("dnud")]
    ax.errorbar(
        [r["freq"] for r in dsa],
        [r["dnud"] for r in dsa],
        yerr=[r["err"] for r in dsa],
        fmt="o",
        ms=5,
        capsize=2,
        color="#0C5DA5",
        label="DSA measured",
        zorder=5,
    )
    ax.scatter(
        [r["freq"] for r in chime],
        [r["chan"] for r in chime],
        marker="v",
        s=30,
        color="#FF2C00",
        label="CHIME upper limit",
        zorder=5,
    )
    nu = np.linspace(400, 1500, 200)
    f = SC["powerlaw"]
    if f:
        ax.plot(
            nu,
            10 ** f["log10_c"] * nu ** f["x_scint"],
            "--",
            color="#0C5DA5",
            lw=1.0,
            label=rf"DSA fit $x={f['x_scint']:.2f}\pm{f['x_scint_err']:.2f}$",
        )
    nu0, d0 = np.median([r["freq"] for r in dsa]), np.median([r["dnud"] for r in dsa])
    ax.plot(nu, d0 * (nu / nu0) ** 4.4, ":", color="0.4", lw=1.0, label=r"Kolmogorov $\nu^{4.4}$")
    tau = TAU1 * 1e-3 * (nu / 1000.0) ** (-ALPHA)
    ax.plot(
        nu,
        1.0 / (2 * np.pi * tau) / 1e6,
        "-.",
        color="#00B945",
        lw=1.0,
        label=r"scattering screen ($C_1{=}1$)",
    )
    # NE2025 MW floor (stars) at the two band centres
    mw_nu = [600.19, 1405.0]
    mw_d = [NE["CHIME"]["bw_kHz"] / 1e3, NE["DSA"]["bw_kHz"] / 1e3]
    ax.plot(mw_nu, mw_d, "*", ms=11, color="#845B97", label="NE2025 MW floor", zorder=6)
    ax.plot(nu, mw_d[1] * (nu / 1405.0) ** 4.4, ":", color="#845B97", lw=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel(r"$\Delta\nu_d$ (MHz)")
    ax.set_xlim(400, 1550)
    ax.set_xticks([400, 600, 800, 1000, 1400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="lower right", frameon=False, ncol=1)
    _save(
        fig,
        "wilhelm_scint_dnud_ne2025",
        "Paper fig: log-log Delta-nu_d vs nu 400-1500 MHz. DSA measured (blue circles) ~0.13 MHz "
        "flat; CHIME red down-triangle upper limits at channel width; blue dashed DSA fit x~-0.23; "
        "grey dotted Kolmogorov nu^4.4; green dash-dot scattering-screen line (~kHz, far below); "
        "purple stars = NE2025 MW floor (CHIME 26 kHz, DSA 1095 kHz) with purple dotted nu^4.4. "
        "Measured DSA sits ~8x BELOW the MW-floor star and ~85x ABOVE the scattering-screen line "
        "=> three distinct screens. Serif fonts, no glyph boxes.",
    )


def fig_pbf_evidence():
    """Per-band PBF: dlnZ vs beta (CHIME prefers Kolmogorov, DSA rejects heavy tails)."""
    # verified per-band screen dlnZ vs exp (sha256 965cf7b5) + joint lnZ (sha256 75ed7bf7)
    beta = [2.5, 3.0, 11 / 3, 3.9]
    chime = [-42.66, -9.05, +6.15, -4.64]
    dsa = [-842.46, -601.46, -90.36, -5.26]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a1.axhline(0, color="0.6", lw=0.8)
    a1.plot(beta, chime, "o-", color="#0C5DA5", label="CHIME")
    a1.plot(beta, dsa, "s-", color="#FF2C00", label="DSA")
    a1.axvline(11 / 3, color="0.5", ls=":", lw=0.8)
    a1.set_xlabel(r"power-law PBF index $\beta$")
    a1.set_ylabel(r"$\Delta\ln Z$ vs exponential")
    a1.set_yscale("symlog", linthresh=10)
    a1.text(11 / 3, a1.get_ylim()[1] * 0.5, "Kolm.\n11/3", fontsize=6, ha="center", color="0.4")
    a1.legend(frameon=False, loc="lower left")
    a1.set_title("per-band PBF preference")
    # joint lnZ ladder
    labels = ["all-exp", "all-power-law\n(11/3)", "per-band\n(C:pl, D:exp)"]
    lnz = [3809.86, 3725.58, 3813.83]
    base = 3809.86
    bars = a2.bar(range(3), [v - base for v in lnz], color=["#474747", "#FF2C00", "#00B945"])
    a2.axhline(0, color="0.6", lw=0.8)
    a2.set_xticks(range(3))
    a2.set_xticklabels(labels, fontsize=6.5)
    a2.set_ylabel(r"$\ln Z - \ln Z_{\rm all-exp}$")
    for i, v in enumerate(lnz):
        a2.text(
            i, (v - base) + (2 if v > base else -6), f"{v - base:+.1f}", ha="center", fontsize=7
        )
    a2.set_title("joint-fit evidence (wilhelm C1D1)")
    fig.tight_layout()
    _save(
        fig,
        "wilhelm_pbf_evidence",
        "Paper fig, 2 panels. LEFT: dlnZ vs beta, CHIME (blue circles) peaks POSITIVE at "
        "beta=11/3 (+6.15), DSA (red squares) all NEGATIVE and steeply worse for smaller beta "
        "(symlog y), vertical dotted at Kolmogorov 11/3. RIGHT: bar chart lnZ-3809.86 for all-exp "
        "(0), all-power-law (-84, red), per-band (+3.97, green) with value labels. Shows bands want "
        "different PBFs; per-band is the only net-positive global config. Serif, no glyph boxes.",
    )


def fig_pbf_shapes():
    """Illustrative exp vs power-law PBF kernel shapes."""
    import sys

    sys.path.insert(0, os.environ["FLITS_REPO"] + "/scattering")
    from scat_analysis.burstfit import analytic_gaussian_exp_convolution as ex
    from scat_analysis.burstfit import gaussian_powerlaw_convolution as pl

    t = (np.arange(800) * 0.01)[None, :]
    mu = np.array([[1.5]])
    sig = np.array([[0.05]])
    tau = np.array([[0.3]])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    e = ex(t, mu, sig, tau)[0]
    ax.plot(t[0], e / e.max(), color="#00B945", label="exponential PBF")
    for b, c in [(3.0, "#FF9500"), (11 / 3, "#FF2C00")]:
        p = pl(t, mu, sig, tau, b)[0]
        ax.plot(t[0], p / p.max(), color=c, label=rf"power-law $\beta={b:.2f}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized intensity")
    ax.legend(frameon=False)
    ax.set_title("PBF shapes (heavier tail at smaller $\\beta$)")
    fig.tight_layout()
    _save(
        fig,
        "pbf_shapes",
        "Paper fig: normalized PBF profiles (log y), Gaussian convolved with exponential (green) vs "
        "power-law beta=3.0 (orange) and beta=11/3 (red). All peak ~1; power-law curves show heavier "
        "late-time tails than the exponential, heavier for smaller beta. Serif, no glyph boxes.",
    )


def fig_census():
    """Cross-codetection scattering excess vs NE2025 floor, with per-sightline significance."""
    fin = json.load(open(f"{NL}/data/scint/scint_mw_final.json"))
    rows = [r for r in fin["bursts"] if r.get("excess")]
    sig_dex = fin["sigma_floor_dex"]
    fig, ax = plt.subplots(figsize=(4.3, 3.2))
    aoff = {
        "wilhelm": (-48, 3),
        "zach": (4, 5),
        "hamilton": (7, -1),
        "chromatica": (-60, -3),
        "casey": (-36, 2),
        "oran": (5, -9),
        "freya": (6, 2),
        "johndoeii": (5, -9),
        "isha": (-20, -10),
        "mahi": (6, -1),
        "whitney": (-34, 3),
        "phineas": (-44, 2),
    }
    LOWCONF = {"casey", "freya"}  # judge flagged these recovered-but-low-confidence
    for r in rows:
        x, exc, z = abs(r["b"]), r["excess"], r["z_sigma"]
        ll = r.get("lower_limit") or (r.get("judge", {}) or {}).get("is_lower_limit")
        lo = exc - exc / 10 ** r["sigma_log10"]
        hi = exc * 10 ** r["sigma_log10"] - exc
        if ll:  # diffractive Dnu_d is an upper limit => excess a LOWER limit
            ax.errorbar(
                x,
                exc,
                yerr=[[0], [exc * (10**sig_dex - 1)]],
                fmt="^",
                ms=5,
                color="#9a6500",
                lolims=True,
                capsize=2,
                alpha=0.85,
                zorder=4,
            )
        else:
            sigjudge = z > 2 and r["burst"] not in LOWCONF
            face = "#0C5DA5" if sigjudge else ("none" if r["burst"] in LOWCONF else "#7Fb3e0")
            edge = "#0C5DA5" if (sigjudge or r["burst"] in LOWCONF) else "#7Fb3e0"
            ax.errorbar(
                x,
                exc,
                yerr=[[lo], [hi]],
                fmt="o",
                ms=6,
                mfc=face,
                mec=edge,
                ecolor=edge,
                capsize=2,
                zorder=6 if sigjudge else 5,
            )
        lab = r["burst"] + (f" {z:.1f}$\\sigma$" if not ll and z > 0 else "")
        ax.annotate(
            lab,
            (x, exc),
            xytext=aoff.get(r["burst"], (4, 3)),
            textcoords="offset points",
            fontsize=5.5,
            color="0.25",
        )
    ax.axhline(1.0, color="#845B97", ls="--", lw=1.0, zorder=2)
    ax.text(48.5, 1.06, "NE2025 MW floor", fontsize=6, color="#845B97", ha="right", va="bottom")
    sigset = [r for r in rows if r["z_sigma"] > 2 and r["burst"] not in LOWCONF]
    if sigset:
        lo_, hi_ = min(r["excess"] for r in sigset), max(r["excess"] for r in sigset)
        ax.axhspan(lo_, hi_, color="#0C5DA5", alpha=0.10, zorder=1, label=r"$z>2$ excess (clean)")
    # low-|b| NE2025 void zone (floor unreliable)
    ax.axvspan(5, 14, color="0.85", alpha=0.5, zorder=0)
    ax.text(9.5, 0.075, "NE2025\nvoids", fontsize=5.5, color="0.45", ha="center", va="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"Galactic latitude $|b|$ (deg)")
    ax.set_ylabel(r"excess  $\Delta\nu_d^{\,\rm MW}\,/\,\Delta\nu_d^{\,\rm meas}$")
    ax.set_ylim(0.06, 30)
    ax.set_xlim(5, 50)
    from matplotlib.lines import Line2D

    leg = [
        Line2D([], [], marker="o", color="#0C5DA5", ls="", label=r"clean, $z>2$"),
        Line2D(
            [],
            [],
            marker="o",
            mfc="none",
            mec="#0C5DA5",
            color="#0C5DA5",
            ls="",
            label="clean, low conf.",
        ),
        Line2D(
            [],
            [],
            marker="o",
            mfc="#7Fb3e0",
            mec="#7Fb3e0",
            color="#7Fb3e0",
            ls="",
            label=r"clean, $z<2$",
        ),
        Line2D([], [], marker="^", color="#9a6500", ls="", label="lower limit"),
    ]
    ax.legend(
        handles=leg,
        loc="lower right",
        frameon=False,
        fontsize=5.5,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout()
    _save(
        fig,
        "codetection_scint_excess",
        "Paper fig: scattering excess (NE2025 MW-floor Dnu_d / measured diffractive Dnu_d) vs Galactic "
        "latitude |b| for all 12 co-detections, with NE2025-floor uncertainty (0.4 dex ~x2.5) + measurement "
        "error in the error bars and per-point z (sigma above the MW floor=1, purple dashed line). FILLED "
        "dark-blue circles = cleanly-resolved diffractive scales significant at z>2: zach 10.7x, casey... "
        "no casey is low-conf OPEN. So filled: zach 10.7x/2.4sig, wilhelm 8.0x/2.2, hamilton 7.2x/2.1 (and "
        "casey 9.6x but OPEN = judge low-confidence single-subband). OPEN circles = low-confidence recovered "
        "(casey, freya). Light-blue circles = clean but z<2 (chromatica 5.9x, oran 2.6x). Orange up-triangles "
        "= lower limits (excess could be higher): phineas, whitney, isha, mahi. Blue shaded band = the z>2 "
        "cluster (~7-11x). Grey vertical band |b|<14 = NE2025 void zone where the floor is unreliable "
        "(mahi/isha/johndoeii fall below 1 there). Message: the excess is ~7-11x and significant (z>2) on "
        "3-4 independent mid-|b| sightlines where the floor is reliable and the diffractive scale resolvable "
        "=> a COMMON (host/intervening) enhancement, not wilhelm-specific. Serif, no glyph boxes.",
    )


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401 (used via matplotlib.ticker above)

    fig_scint()
    fig_pbf_evidence()
    fig_pbf_shapes()
    fig_census()
    json.dump(manifest, open(f"{OUT}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote {OUT}/figures.manifest.json ({len(manifest['figures'])} figures)")
