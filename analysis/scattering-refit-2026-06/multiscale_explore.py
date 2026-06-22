#!/usr/bin/env python
"""Exploratory: is there a second (Galactic) scintle scale in freya's gains?

Across the channelization ladder, per band: noise-corrected modulation index,
gain-spectrum ACF, and a 1- vs 2-Lorentzian ACF fit (BIC). A real scale should
recur at the SAME Delta_nu across channelizations where resolvable; pure noise
or a degree-of-freedom artifact will not.
"""
import numpy as np
from scipy.optimize import least_squares

Z = np.load("freya_gainladder.npz")
LAD = {"C": [64, 32, 16, 8], "D": [384, 192, 96, 48, 24]}
NAME = {"C": "CHIME", "D": "DSA"}


def clean(fr, g, v):
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def acf_noisecorr(g, v):
    """Mean-removed normalized ACF; lag-0 noise bias subtracted."""
    x = g - g.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size      # autocovariance
    var_noise = v.mean()
    ac0_signal = ac[0] - var_noise                              # remove white-noise power
    if ac0_signal <= 0:
        return None, None
    return ac / ac0_signal, float(ac0_signal / g.mean() ** 2)   # normalized ACF, m^2


def lorsum(lags, params):
    out = np.zeros_like(lags)
    for i in range(0, len(params), 2):
        A, dn = params[i], params[i + 1]
        out += A / (1.0 + (lags / dn) ** 2)
    return out


def fit_lor(lags, ac, N, chan, span, n_restart=12):
    """Fit N Lorentzians to ac(lags>0); return (params, rss)."""
    best = None
    rng = np.random.default_rng(0)
    lo = [0.0, 0.5 * chan] * N
    hi = [1.5, 0.8 * span] * N
    for _ in range(n_restart):
        p0 = []
        for i in range(N):
            p0 += [rng.uniform(0.05, 0.6), rng.uniform(chan, 0.5 * span)]
        try:
            r = least_squares(lambda p: lorsum(lags, p) - ac, p0,
                              bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.inf
    return best.x, float(np.sum((lorsum(lags, best.x) - ac) ** 2))


def bic(rss, n, k):
    return n * np.log(rss / n + 1e-300) + k * np.log(n)


for s, ffs in LAD.items():
    print(f"\n===== {NAME[s]} =====")
    for ff in ffs:
        fr, g, v = clean(Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"])
        if fr.size < 8:
            continue
        chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
        ac, m2 = acf_noisecorr(g, v)
        if ac is None:
            print(f"  ff{ff}: m^2<=0 (noise-dominated)"); continue
        lags = np.arange(ac.size) * chan
        span = lags[-1]
        # fit lags 1..(half span) to avoid edge-starved high lags
        sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
        Lg, A = lags[sel], ac[sel]
        out = {}
        for N in (1, 2):
            if A.size < 2 * N + 1:
                continue
            p, rss = fit_lor(Lg, A, N, chan, span)
            out[N] = (p, bic(rss, A.size, 2 * N))
        s1 = f"m^2={m2:.4f} chan={chan:.2f} ACF[1]={ac[1]:+.2f}"
        if 1 in out:
            p1 = out[1][0]; s1 += f" | 1L: dn={p1[1]:.2f}MHz A={p1[0]:.2f} BIC={out[1][1]:.1f}"
        if 2 in out:
            p2 = out[2][0]
            dns = sorted([(p2[1], p2[0]), (p2[3], p2[2])])
            s1 += f" | 2L: dn=[{dns[0][0]:.2f},{dns[1][0]:.2f}]MHz A=[{dns[0][1]:.2f},{dns[1][1]:.2f}] BIC={out[2][1]:.1f}"
            if 1 in out:
                s1 += f"  dBIC(2-1)={out[2][1]-out[1][1]:+.1f}"
        print(f"  ff{ff:<4d}: {s1}")
