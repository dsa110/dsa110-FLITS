#!/usr/bin/env python
"""Step 1: scintillation from the recovered per-channel gains (no refit).

Loads the gain spectrum g_f saved by the gain-marginalized joint fit, removes the
smooth intrinsic spectrum, and measures the diffractive scintillation bandwidth
Delta-nu_d from the frequency ACF of the residual modulation. Reports, per band:
  - modulation index m = std(g_norm)            (~1 for fully-modulated diffractive)
  - Delta-nu_d (Lorentzian half-width at 1/2, fit to lag>=1 so the zero-lag noise
    spike is excluded), with the channel width and band for context
  - RESOLUTION verdict: need Delta-nu_d > ~3 channels (to resolve a scintle) AND
    > ~5 scintles across the band (for a usable Delta-nu_d error ~1/sqrt(N))
  - SAME-SCREEN test: predicted Delta-nu_d = C1/(2 pi tau(nu_band)) vs measured

  python scint_acf.py <burst>  [C1]
"""

import json
import os
import sys
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
J = f"{RUNS}/data/joint"


def lorentz(dnu, dnud):  # scintillation ACF (normalized, lag>0)
    return 1.0 / (1.0 + (dnu / dnud) ** 2)


def analyse(freq, gain, label):
    ok = np.isfinite(gain) & (gain > 0)
    f, g = freq[ok], gain[ok]
    if f.size < 6:
        print(f"  {label}: too few live channels ({f.size})")
        return None
    dch = np.median(np.diff(f)) * 1e3  # channel width [MHz]
    band = (f.max() - f.min()) * 1e3  # band [MHz]
    # detrend smooth spectrum: low-order poly in freq, divide -> modulation about 1
    coef = P.polyfit(f, g, 2)
    trend = P.polyval(f, coef)
    gnorm = g / np.where(trend > 0, trend, np.nan)
    gnorm = gnorm[np.isfinite(gnorm)]
    gnorm = gnorm - gnorm.mean()
    m_idx = np.std(gnorm) / 1.0  # modulation index (mean~1 after /trend)
    # frequency ACF (biased, normalized to lag0)
    ac = np.correlate(gnorm, gnorm, "full")
    ac = ac[ac.size // 2 :]
    ac = ac / ac[0]
    lags_mhz = np.arange(ac.size) * dch
    # fit Lorentzian to lag>=1 (exclude noisy zero-lag); guard short arrays
    dnud = np.nan
    if ac.size >= 4:
        try:
            popt, _ = curve_fit(
                lorentz, lags_mhz[1:6], ac[1:6], p0=[max(dch, band / 8)], maxfev=4000
            )
            dnud = abs(popt[0])
        except Exception:
            pass
    ch_per_scintle = dnud / dch if dnud == dnud else float("nan")
    n_scintles = band / dnud if dnud == dnud else float("nan")
    resolved = (ch_per_scintle >= 3) and (n_scintles >= 5)
    print(
        f"  {label}: chan={dch:.2f}MHz band={band:.0f}MHz m={m_idx:.2f} | "
        f"Delta-nu_d={dnud:.2f}MHz  ch/scintle={ch_per_scintle:.1f}  N_scintles={n_scintles:.1f}  "
        f"-> {'RESOLVED' if resolved else 'UNRESOLVED'}"
    )
    return dict(dnud=dnud, dch=dch, band=band, m=m_idx, lags=lags_mhz, ac=ac, resolved=resolved)


def main():
    b = sys.argv[1]
    C1 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.16  # Cordes&Rickett 1998 thin-screen
    npz = np.load(f"{J}/{b}_joint_samples.npz", allow_pickle=True)
    if "gain_C" not in npz:
        sys.exit(f"{b}: no saved gains (not a gain-marginalized fit)")
    d = json.load(open(f"{J}/{b}_joint_fit.json"))
    tau1 = d["percentiles"]["tau_1ghz"]["median"]
    al = d["percentiles"]["alpha"]["median"]
    print(f"{b}: tau_1GHz={tau1:.3f}ms alpha={al:.2f}  (C1={C1})")
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (fkey, gkey, name, nu_band) in enumerate(
        [("freq_C", "gain_C", "CHIME", 0.6), ("freq_D", "gain_D", "DSA", 1.405)]
    ):
        r = analyse(npz[fkey], npz[gkey], name)
        res[name] = r
        if r:
            ax[k].plot(r["lags"], r["ac"], "k.-", ms=4, label="ACF")
            if r["dnud"] == r["dnud"]:
                xx = np.linspace(0, r["lags"].max(), 200)
                ax[k].plot(
                    xx, lorentz(xx, r["dnud"]), "r", label=f"Lorentz dnud={r['dnud']:.1f}MHz"
                )
            ax[k].axhline(0.5, color="0.7", ls=":")
            ax[k].set_xlabel("freq lag (MHz)")
            ax[k].set_title(f"{b} {name} gain ACF")
            ax[k].legend(fontsize=8)
    # same-screen test per band
    print("  SAME-SCREEN test (tau and scintillation from one thin screen => equal):")
    for name, nu in [("CHIME", 0.6), ("DSA", 1.405)]:
        r = res.get(name)
        if not r or r["dnud"] != r["dnud"]:
            continue
        tau_nu = tau1 * nu ** (-al)  # ms
        dnud_pred = (
            C1 / (2 * np.pi * tau_nu) * 1e3 / 1e3
        )  # MHz: tau ms -> 1/(2pi tau ms) = kHz? fix below
        # tau in ms -> 1/(2 pi tau[s]) Hz; tau_ms*1e-3 s; dnud = C1/(2 pi tau_s) Hz -> /1e6 MHz
        dnud_pred = C1 / (2 * np.pi * (tau_nu * 1e-3)) / 1e6
        print(
            f"    {name} @ {nu:.2f}GHz: tau={tau_nu * 1e3:.1f}us  "
            f"Delta-nu_d predicted={dnud_pred:.3f}MHz  measured={r['dnud']:.2f}MHz  "
            f"ratio={r['dnud'] / dnud_pred:.2g}"
        )
    fig.tight_layout()
    fp = f"{J}/{b}_scint_acf.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
