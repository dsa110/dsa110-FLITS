#!/usr/bin/env python
"""The fitted scintillation Lorentzian for freya, made explicit.

Delta_nu_d is fit as the half-width of a Lorentzian to the per-channel gain
spectrum's frequency ACF (the GP-marginal does this in the likelihood domain,
noise-weighted; here we show the equivalent, transparent ACF-domain fit).

Two things the dashboard was missing:
  (cols 0-1) freya's measured gain-spectrum ACF with the BEST-FIT LORENTZIAN
             overlaid, per band -- the actual fitted curve.
  (col 2)    why the coarse width is not a detection: the fitted Delta_nu_d
             COLLAPSES as you channelize finer (coarse 16ch -> fine 96ch, Fig 3)
             -> it was broadband leakage, not a scintle. The single-screen
             scattering prediction sits ~1e4x below even the fine upper limit.

  python lorentzian_fit_fig.py     (local; reads coarse freya_joint_samples.npz)
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = np.load("freya_joint_samples.npz", allow_pickle=True)
NAMES = [str(x) for x in Z["param_names"]]


def wmed(col):
    x = Z["samples"][:, col]; w = Z["weights"] / Z["weights"].sum()
    o = np.argsort(x)
    return float(np.interp(0.5, np.cumsum(w[o]), x[o]))


def detrend(freq, gain, deg=2):
    ok = np.isfinite(gain) & (gain > 0)
    f, g = np.asarray(freq)[ok], np.asarray(gain)[ok]
    tr = np.polyval(np.polyfit(f, g, deg), f)
    return f, g / np.where(tr > 0, tr, np.nan)


def acf(gn):
    x = gn - np.nanmean(gn); x = x[np.isfinite(x)]
    ac = np.correlate(x, x, "full"); ac = ac[ac.size // 2:]
    return ac / ac[0]


def lor(lag, A, dnu):
    return A / (1.0 + (lag / dnu) ** 2)


tau, al = wmed(NAMES.index("tau_1ghz")), wmed(NAMES.index("alpha"))
# fine-channel upper limits (96ch analysis, Fig 3 / prior HPCC run) and band centers
FINE = {"C": 0.30, "D": 0.06}
NU0 = {"C": 0.6, "D": 1.405}
BANDS = [("C", "CHIME", "tab:blue"), ("D", "DSA", "tab:red")]

fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
coarse_fit = {}
for col, (suf, name, c) in enumerate(BANDS):
    f, gn = detrend(Z[f"freq_{suf}"], Z[f"gain_{suf}"])
    chan = float(np.median(np.abs(np.diff(f)))) * 1e3            # MHz/channel
    ac = acf(gn); lags = np.arange(ac.size) * chan
    # Lorentzian fit to lags>=1 (lag-0 is inflated by white noise); A<=1 proper ACF.
    p, _ = curve_fit(lor, lags[1:], ac[1:], p0=[min(ac[1], 1.0), chan],
                     bounds=([0, 0.3 * chan], [1.05, lags[-1]]), maxfev=40000)
    Afit, dnu = p; coarse_fit[suf] = dnu
    a = ax[col]
    a.axhline(0, color="k", lw=0.5)
    a.plot(lags, ac, "o", color=c, ms=6, label="measured gain ACF (16 ch)")
    xx = np.linspace(0, lags[-1], 400)
    a.plot(xx, lor(xx, Afit, dnu), "-", color=c, lw=1.8,
           label=f"Lorentzian fit  $\\Delta\\nu_d\\lesssim${dnu:.1f} MHz")
    a.axvline(chan, color="0.5", ls="--", lw=1, label=f"1 channel = {chan:.0f} MHz")
    a.set_xlim(0, min(lags[-1], 6 * chan)); a.set_ylim(-0.4, 1.05)
    a.set_xlabel("frequency lag (MHz)"); a.set_ylabel("ACF")
    a.set_title(f"{name}: fitted Lorentzian on the gain ACF")
    a.legend(fontsize=8, loc="upper right")
    a.text(0.04, 0.06,
           f"width $\\approx$ 1 channel\n$\\Rightarrow$ resolution-limited\n(upper limit, not a scintle)",
           transform=a.transAxes, fontsize=8, color=c, va="bottom")

# col 2: Delta_nu_d collapses with channelization -> unresolved
a = ax[2]
stages = ["coarse\n(16 ch)", "fine\n(96 ch, Fig 3)", "same-screen\nprediction"]
xs = np.arange(3)
for suf, name, c in BANDS:
    pred = 1.0 / (2 * np.pi * (tau * NU0[suf] ** (-al) * 1e-3)) / 1e6   # MHz, C1=1
    ys = [coarse_fit[suf], FINE[suf], pred]
    a.plot(xs, ys, "o-", color=c, ms=9, lw=1.6, label=name)
    for x, y in zip(xs, ys):
        a.annotate(f"{y:.2g}", (x, y), textcoords="offset points",
                   xytext=(6, 6), fontsize=8, color=c)
a.set_yscale("log"); a.set_xticks(xs); a.set_xticklabels(stages, fontsize=9)
a.set_ylabel(r"fitted $\Delta\nu_d$ (MHz)")
a.set_title("the fitted width collapses with resolution\n$\\Rightarrow$ broadband leakage, not a scintle")
a.legend(fontsize=9); a.grid(True, which="both", alpha=0.25)

fig.suptitle(f"freya: the fitted scintillation Lorentzian is an UPPER LIMIT "
             f"(unresolved in both bands)   |   $\\alpha$={al:.2f}, "
             f"$\\tau_{{1GHz}}$={tau:.3f} ms", fontsize=13)
fig.tight_layout()
fig.savefig("freya_lorentzian_fit.png", dpi=130, bbox_inches="tight")
print(f"coarse Lorentzian fits: CHIME={coarse_fit['C']:.2f} MHz  DSA={coarse_fit['D']:.2f} MHz")
print("wrote freya_lorentzian_fit.png")
