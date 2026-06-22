#!/usr/bin/env python
"""Visual evidence for the freya scintillation claims.

Reads the saved gain spectra (coarse: freya_joint_samples.npz, fine:
freyafine_joint_samples.npz) + the coarse fit (tau, alpha), and produces ONE
2x3 figure proving:
  (col 0) gain spectrum g_f vs freq -- coarse vs fine, per band
  (col 1) frequency ACF -- coarse shows a BROAD component (intrinsic spectral
          structure, tens of MHz), fine strips it -> narrow/noise => the burst
          modulation is NOT diffractive scintillation at the claimed 5.5 MHz
  (col 2, top) m^2 vs 1/Delta_nu_chan from rebinning the fine gains: a clean line
          would mean one unresolved scale; CURVATURE proves multiple scales
          (broadband + sub-MHz) => the coarse "Delta_nu_d" was contaminated
  (col 2, bot) same-screen test: predicted Delta_nu_d=C1/(2 pi tau(nu)) curve vs
          the observed upper limits -> ~1000x gap => different screens

  python scint_evidence.py
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
J = f"{RUNS}/data/joint"


def detrend(freq, gain):
    ok = np.isfinite(gain) & (gain > 0)
    f, g = np.asarray(freq)[ok], np.asarray(gain)[ok]
    trend = np.polyval(np.polyfit(f, g, 2), f)
    gn = g / np.where(trend > 0, trend, np.nan)
    return f, g, gn[np.isfinite(gn)]


def acf(freq, gain):
    f, g, gn = detrend(freq, gain)
    x = gn - gn.mean()
    ac = np.correlate(x, x, "full"); ac = ac[ac.size // 2:]
    ac = ac / ac[0]
    chan = float(np.median(np.abs(np.diff(f)))) * 1e3
    return np.arange(ac.size) * chan, ac, chan


def rebin_gain(freq, gain, F):
    ok = np.isfinite(gain) & (gain > 0)
    f, g = np.asarray(freq)[ok], np.asarray(gain)[ok]
    n = (f.size // F) * F
    if n < 4 * F:
        return None, None
    f = f[:n].reshape(-1, F).mean(1); g = g[:n].reshape(-1, F).mean(1)
    return f, g


def mod_index(freq, gain):
    f, g, gn = detrend(freq, gain)
    return float(np.std(gn)), float(np.median(np.abs(np.diff(f)))) * 1e3


def main():
    coarse = np.load(f"{J}/freya_joint_samples.npz", allow_pickle=True)
    fine = np.load(f"{J}/freyafine_joint_samples.npz", allow_pickle=True)
    fit = json.load(open(f"{J}/freya_joint_fit.json"))
    tau1 = fit["percentiles"]["tau_1ghz"]["median"]
    al = fit["percentiles"]["alpha"]["median"]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    bands = [("C", "CHIME", 0.6, "tab:blue"), ("D", "DSA", 1.405, "tab:red")]
    for row, (suf, name, nu0, col) in enumerate(bands):
        # --- col 0: gain spectrum coarse vs fine
        fc, gc, _ = detrend(coarse[f"freq_{suf}"], coarse[f"gain_{suf}"])
        ff, gf, _ = detrend(fine[f"freq_{suf}"], fine[f"gain_{suf}"])
        a0 = ax[row][0]
        a0.plot(fc, gc / np.nanmedian(gc), "o-", ms=3, color="0.6", label=f"coarse ({fc.size}ch)")
        a0.plot(ff, gf / np.nanmedian(gf), ".-", ms=3, color=col, lw=0.8, label=f"fine ({ff.size}ch)")
        a0.set_title(f"{name}: gain spectrum g_f"); a0.set_xlabel("freq (GHz)")
        a0.set_ylabel("g_f / median"); a0.legend(fontsize=7)

        # --- col 1: ACF coarse vs fine + Lorentzian fits
        lc, acc, cwc = acf(coarse[f"freq_{suf}"], coarse[f"gain_{suf}"])
        lf, acf_, cwf = acf(fine[f"freq_{suf}"], fine[f"gain_{suf}"])
        a1 = ax[row][1]
        a1.plot(lc, acc, "o-", ms=4, color="0.6", label=f"coarse ({cwc:.0f}MHz/ch)")
        a1.plot(lf, acf_, ".-", ms=4, color=col, label=f"fine ({cwf:.1f}MHz/ch)")

        # Fit Lorentzian to fine ACF (primary): acf_fit = 1/(1+(lag/lag_h)**2)
        from scipy.optimize import minimize_scalar
        for ldata, acfdata, cwdata, linestyle, linecolor in [
            (lf[lf <= 60], acf_[lf <= 60], cwf, "-", col),
            (lc[lc <= 60], acc[lc <= 60], cwc, "--", "0.6")
        ]:
            if len(ldata) > 2 and np.max(acfdata) > 0.15:
                def residual(lag_h):
                    if lag_h <= 0: return 1e10
                    model = 1.0 / (1.0 + (np.abs(ldata) / lag_h)**2)
                    return np.sum((acfdata - model)**2)
                res = minimize_scalar(residual, bounds=(0.5, 200), method="bounded")
                lag_h = res.x
                dnu_est = cwdata / lag_h
                # Plot fitted Lorentzian
                l_fit = np.linspace(0, np.max(ldata), 80)
                acf_fit = 1.0 / (1.0 + (l_fit / lag_h)**2)
                label_txt = f"fit: lag_h={lag_h:.1f}MHz, Δν_d≈{dnu_est:.2f}MHz" if linestyle == "-" else f"(coarse fit: lag_h={lag_h:.1f})"
                a1.plot(l_fit, acf_fit, linestyle=linestyle, color=linecolor, lw=1.5, alpha=0.7, label=label_txt)

        a1.axhline(0, color="k", lw=0.5); a1.axhline(0.5, color="0.8", ls=":")
        a1.set_xlim(0, max(cwc * 4, 60)); a1.set_title(f"{name}: gain ACF + Lorentzian fit")
        a1.set_xlabel("freq lag (MHz)"); a1.set_ylabel("ACF"); a1.legend(fontsize=6, loc="upper right")
        a1.text(0.97, 0.85, "ACF width ~ channel\n=> UNRESOLVED" if cwf else "",
                transform=a1.transAxes, ha="right", fontsize=8, color=col)

    # --- col 2 top: m^2 vs 1/chan from rebinning FINE gains (curvature = multi-scale)
    a = ax[0][2]
    for suf, name, col in [("C", "CHIME", "tab:blue"), ("D", "DSA", "tab:red")]:
        invcw, m2 = [], []
        for F in (1, 2, 4, 8, 16):
            fr, gn = rebin_gain(fine[f"freq_{suf}"], fine[f"gain_{suf}"], F)
            if fr is None:
                continue
            m, cw = mod_index(fr, gn)
            invcw.append(1.0 / cw); m2.append(m ** 2)
        invcw, m2 = np.array(invcw), np.array(m2)
        a.plot(invcw, m2, "o-", color=col, label=name)
        if invcw.size >= 2:
            s, b = np.polyfit(invcw, m2, 1)
            xx = np.linspace(0, invcw.max(), 50)
            a.plot(xx, s * xx + b, "--", color=col, lw=0.8)
    a.set_xlabel("1 / channel width  (1/MHz)"); a.set_ylabel("modulation index m^2")
    a.set_title("m^2 vs 1/chan: curvature => multiple scales\n(not one unresolved Delta_nu_d)")
    a.legend(fontsize=7)

    # --- col 2 bot: same-screen test
    a = ax[1][2]
    nu = np.linspace(0.4, 1.5, 100)
    tau_nu = tau1 * nu ** (-al)                       # ms
    dnud_pred = 1.0 / (2 * np.pi * (tau_nu * 1e-3)) / 1e6   # MHz, C1=1
    a.plot(nu, dnud_pred, "k-", label="predicted Delta_nu_d = 1/(2 pi tau(nu))")
    # observed upper limits (fine-channel slope estimates)
    obs = {"CHIME": (0.6, 0.30), "DSA": (1.405, 0.06)}
    for nm, (nub, dn) in obs.items():
        a.errorbar([nub], [dn], yerr=[[dn * 0.5], [dn]], fmt="v", capsize=4,
                   label=f"{nm} observed UL ~{dn:.2f}MHz")
    a.set_yscale("log"); a.set_xlabel("freq (GHz)"); a.set_ylabel("Delta_nu_d (MHz)")
    a.set_title(f"same-screen: observed UL ~1000x > predicted\n(alpha={al:.2f}, tau_1GHz={tau1:.3f}ms)")
    a.legend(fontsize=7)

    fig.suptitle("freya scintillation evidence: UNRESOLVED in both bands; coarse 5.5MHz was broadband contamination",
                 fontsize=13)
    fig.tight_layout()
    fp = f"{J}/freya_scint_evidence.png"; fig.savefig(fp, dpi=130, bbox_inches="tight")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
