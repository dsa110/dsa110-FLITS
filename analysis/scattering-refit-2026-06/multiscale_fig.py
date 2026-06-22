#!/usr/bin/env python
"""Multi-screen test figure for a burst (data-driven; caption set by caller).
 (a) DSA finest ACF + 1/2/3 Lorentzians -> BIC model selection.
 (b) nu-scaling test (annotated unreliable if a band is marginal).
 (c) DSA dominant width vs detrend degree -> resolved scale vs envelope systematic.
 (d) m^2 vs 1/chan -> unresolved (host) component via the affine slope.

  python multiscale_fig.py <burst> "<suptitle caption>"
"""
import json
import re
import sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BURST = sys.argv[1] if len(sys.argv) > 1 else "freya"
CAPTION = sys.argv[2] if len(sys.argv) > 2 else f"{BURST} multi-screen test"
Z = np.load(f"{BURST}_gainladder.npz")
R = json.load(open(f"{BURST}_multiscale_results.json"))
TAU, AL = R["tau_1ghz"], R["alpha"]


def ladder(s):
    return sorted({int(re.match(rf"gain_{s}_ff(\d+)", k).group(1))
                   for k in Z.files if re.match(rf"gain_{s}_ff\d+$", k)}, reverse=True)


def clean(s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    g, v = np.asarray(g), np.asarray(v)
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    if ok.sum() >= 8:                              # edge mask (see multiscale_fit.py)
        ok &= np.abs(g) > 0.05 * np.median(np.abs(g[ok]))
    return np.asarray(fr)[ok], g[ok], v[ok]


def usable(s, ff, snr_min=3.0):
    fr, g, v = clean(s, ff)
    if fr.size < 8:
        return None
    if float(np.nanmedian(g / np.sqrt(v))) < snr_min:
        return None
    return fr, g, v


def finest(s):
    for ff in sorted(ladder(s)):              # asc f_factor -> finest channels first
        if usable(s, ff):
            return ff
    return None


def resid(fr, g, v, deg=1):
    tr = np.polyval(np.polyfit(fr, g, deg), fr)
    return g / tr - 1.0, v / tr ** 2


def acf_nc(r, vr):
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    sig0 = ac[0] - vr.mean()
    return (ac / sig0, float(sig0)) if sig0 > 0 else (None, None)


def lorsum(lags, p):
    out = np.zeros_like(lags)
    for i in range(0, len(p), 2):
        out += p[i] / (1.0 + (lags / p[i + 1]) ** 2)
    return out


def fit_lor(lags, ac, N, chan, span, restarts=16):
    best = None; rng = np.random.default_rng(N)
    lo = [0.0, 0.4 * chan] * N; hi = [1.6, 0.9 * span] * N
    for _ in range(restarts):
        p0 = []
        for _i in range(N):
            p0 += [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        try:
            r = least_squares(lambda p: lorsum(lags, p) - ac, p0, bounds=(lo, hi), max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def width_1L(fr, g, v, deg, lagfrac=0.5):
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    r, vr = resid(fr, g, v, deg); ac, _ = acf_nc(r, vr)
    if ac is None:
        return np.nan
    lags = np.arange(ac.size) * chan; span = lags[-1]
    sel = (np.arange(ac.size) >= 1) & (lags <= lagfrac * span)
    return float(fit_lor(lags[sel], ac[sel], 1, chan, span)[1])


fig, ax = plt.subplots(2, 2, figsize=(14, 9.6))
ffD = finest("D"); ffC = finest("C")

# (a) DSA finest ACF + N=1,2,3
fr, g, v = usable("D", ffD)
chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
r, vr = resid(fr, g, v, 1); ac, m2 = acf_nc(r, vr)
lags = np.arange(ac.size) * chan; span = lags[-1]
sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
a = ax[0][0]
a.plot(lags[sel], ac[sel], "o", color="tab:red", ms=4, label=f"DSA gain ACF ({fr.size}ch @ {chan:.2f}MHz)")
xx = np.linspace(chan, lags[sel].max(), 400)
for N, c, ls in [(1, "k", "-"), (2, "tab:green", "--"), (3, "tab:purple", ":")]:
    p = fit_lor(lags[sel], ac[sel], N, chan, span)
    bicN = R["bands"]["DSA"]["modelsel_finest"].get(str(N), {}).get("bic")
    a.plot(xx, lorsum(xx, p), ls, color=c, lw=1.7, label=f"{N}L  BIC={bicN:.0f}" if bicN else f"{N}L")
a.axhline(0, color="0.7", lw=0.5); a.set_xlim(0, min(35, lags[sel].max())); a.set_ylim(-0.3, 0.7)
a.set_xlabel("frequency lag (MHz)"); a.set_ylabel("noise-corrected ACF")
a.set_title(f"(a) up to 3 Lorentzians on DSA gain ACF  (BIC prefers N={R['bands']['DSA'].get('BIC_preferred_N')})")
a.legend(fontsize=8)

# (b) nu-scaling
a = ax[0][1]
ns = R.get("nu_scaling", {})
tp = R.get("tau_screen_dnu_MHz", {})
if "dnu_CHIME_MHz" in ns:
    dC, dD = ns["dnu_CHIME_MHz"], ns["dnu_DSA_MHz"]
    nu0C = R["bands"]["CHIME"]["nu0_GHz"]; nu0D = R["bands"]["DSA"]["nu0_GHz"]
    nu = np.linspace(min(nu0C, nu0D) * 0.92, max(nu0C, nu0D) * 1.05, 50)
    a.plot(nu, dC * (nu / nu0C) ** 4, "k--", lw=1.3, label=r"common screen $\propto\nu^4$")
    a.plot([nu0C], [dC], "o", color="tab:blue", ms=11, label=f"CHIME {dC:.1f} MHz")
    a.plot([nu0D], [dD], "o", color="tab:red", ms=11, label=f"DSA {dD:.1f} MHz")
    if tp:
        a.plot([nu0C, nu0D], [tp.get("CHIME"), tp.get("DSA")], "v", color="0.5", ms=9,
               label=r"host $\tau$-screen pred.")
    a.set_yscale("log"); a.set_xlabel("frequency (GHz)"); a.set_ylabel(r"$\Delta\nu_d$ (MHz)")
    a.set_title(f"(b) $\\nu$-scaling: ratio {ns['observed_ratio_DSA_over_CHIME']:.2f} vs "
                f"$\\nu^4$={ns['expected_ratio_nu4']:.0f} ($\\alpha$={ns['implied_alpha']:.1f})")
    a.legend(fontsize=8, loc="best")
else:
    a.text(0.5, 0.5, "cross-band $\\nu$-scaling unavailable\n(one band marginal: "
           "insufficient S/N channelizations)", transform=a.transAxes, ha="center", fontsize=11)
    a.set_title("(b) $\\nu$-scaling test skipped")

# (c) DSA width vs detrend degree
a = ax[1][0]
fr, g, v = usable("D", ffD)
degs = [0, 1, 2, 3, 4, 5]
wdeg = [width_1L(fr, g, v, d) for d in degs]
a.plot(degs, wdeg, "s-", color="tab:red", ms=8, lw=1.6)
for d, w in zip(degs, wdeg):
    a.annotate(f"{w:.1f}", (d, w), textcoords="offset points", xytext=(6, 4), fontsize=8)
n = fr.size // 2
wlo = width_1L(fr[:n], g[:n], v[:n], 1); whi = width_1L(fr[n:], g[n:], v[n:], 1)
a.text(0.97, 0.82, f"split-band (linear):\nfull {wdeg[1]:.1f} -> halves {wlo:.1f}/{whi:.1f} MHz",
       transform=a.transAxes, ha="right", fontsize=8, color="tab:purple")
a.set_xlabel("polynomial detrend degree"); a.set_ylabel(r"DSA 1L width $\Delta\nu$ (MHz)")
a.set_title("(c) DSA width vs detrend degree\n(stable = real scale; collapsing = envelope systematic)")

# (d) m^2 vs 1/chan
a = ax[1][1]
inv, m2s = [], []
for ff in ladder("D"):
    u = usable("D", ff)
    if u is None:
        continue
    fr, g, v = u; chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    rr, vr = resid(fr, g, v, 1); _, mm = acf_nc(rr, vr)
    if mm is not None:
        inv.append(1.0 / chan); m2s.append(mm)
inv, m2s = np.array(inv), np.array(m2s)
a.plot(inv, m2s, "o", color="tab:red", ms=9, label="DSA m$^2$")
if inv.size >= 2:
    A = np.vstack([np.ones_like(inv), inv]).T
    aa, bb = np.linalg.lstsq(A, m2s, rcond=None)[0]
    xx = np.linspace(0, inv.max() * 1.05, 50)
    a.plot(xx, aa + bb * xx, "-", color="k", lw=1.6, label=f"affine a+b/chan\na={aa:.3f} b={bb:.3f}")
    trend = "RISES -> unresolved sub-channel component" if bb > 0.5 * abs(aa) + 1e-6 else "flat/weak slope"
    a.set_title(f"(d) m$^2$ vs 1/chan: {trend}")
else:
    a.set_title("(d) m$^2$ vs 1/chan (insufficient points)")
a.set_xlabel("1 / channel width (1/MHz)"); a.set_ylabel("modulation index m$^2$")
a.legend(fontsize=8, loc="best")

fig.suptitle(CAPTION, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{BURST}_multiscale.png", dpi=130, bbox_inches="tight")
print(f"{BURST}: DSA detrend widths {[round(w,2) for w in wdeg]}; m2 {[round(x,4) for x in m2s]}")
print(f"wrote {BURST}_multiscale.png")
