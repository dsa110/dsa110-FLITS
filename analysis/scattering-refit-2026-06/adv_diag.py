#!/usr/bin/env python
"""Diagnose WHY casey DSA width moves under high-degree detrend:
 - print the raw ACF (lag 0..10) for ff24 at deg 0/1/2/3
 - print the polynomial trend amplitude relative to the residual modulation
 - robust width distribution excluding the obvious fit-failures (>3x span)
 - is the ~6 MHz at deg0/1 actually just the broadband envelope leaking in?
"""
import re
import numpy as np
from scipy.optimize import least_squares

Z = np.load("casey_gainladder.npz")
Zf = np.load("freya_gainladder.npz")


def clean(Z, s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def resid(fr, g, v, deg):
    coef = np.polyfit(fr, g, deg)
    tr = np.polyval(coef, fr)
    r = g / tr - 1.0
    vr = v / tr ** 2
    return r, vr, tr


def acf_nc(r, vr):
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    sig0 = ac[0] - vr.mean()
    return ac / sig0, float(sig0)


def lorsum(lags, p):
    out = np.zeros_like(lags)
    for i in range(0, len(p), 2):
        out += p[i] / (1.0 + (lags / p[i + 1]) ** 2)
    return out


def fit1(lags, ac, chan, span, restarts=40):
    best = None
    rng = np.random.default_rng(0)
    for _ in range(restarts):
        p0 = [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        r = least_squares(lambda p: lorsum(lags, p) - ac, p0,
                          bounds=([0.0, 0.4 * chan], [1.6, 0.9 * span]), max_nfev=8000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def diag(Z, s, ff, label):
    fr, g, v = clean(Z, s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    span_full = (fr.max() - fr.min()) * 1e3
    print(f"\n### {label} {s} ff{ff}  nch={fr.size} chan={chan:.3f}MHz spanBW={span_full:.1f}MHz")
    for deg in (0, 1, 2, 3):
        r, vr, tr = resid(fr, g, v, deg)
        ac, sig0 = acf_nc(r, vr)
        lags = np.arange(ac.size) * chan
        span = lags[-1]
        sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
        p = fit1(lags[sel], ac[sel], chan, span)
        # trend curvature: fraction of g variance explained by trend vs residual
        trend_frac = 1.0 - np.var(g - tr) / np.var(g - g.mean()) if deg > 0 else 0.0
        # is the fitted width below resolution or above half the span?
        flag = ""
        if p[1] < 2 * chan:
            flag = " [UNRESOLVED <2chan]"
        if p[1] > 0.45 * span:
            flag = " [RAILED to span]"
        print(f"  deg{deg}: sig0(m2)={sig0:.4f} ac[1]={ac[1]:+.3f} ac[2]={ac[2]:+.3f} "
              f"ac[3]={ac[3]:+.3f}  dnu1L={p1f(p[1])} A={p[0]:.3f} "
              f"trendVarFrac={trend_frac:+.3f}{flag}")


def p1f(x):
    return f"{x:7.3f}"


for s in ("D", "C"):
    for ff in ([24, 48, 96] if s == "D" else [2, 4, 8]):
        diag(Z, s, ff, "casey")
print("\n========== FREYA contrast ==========")
for s in ("D",):
    for ff in [24, 48]:
        diag(Zf, s, ff, "freya")
