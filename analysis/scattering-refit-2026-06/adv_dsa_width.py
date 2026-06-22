#!/usr/bin/env python
"""Adversarial stress-test: is casey's dominant DSA scintle width (~6 MHz) a
real scale or a detrend/envelope/edge artifact?

Sweep, independently and jointly:
  - detrend polynomial degree 0..3
  - lag-fit fraction of span: 0.3 / 0.5 / 0.7
  - windowing of the gain spectrum (Hann taper before ACF) on/off
  - band-edge trim: drop 0 / 5 / 10 / 15 % of channels from each end
across the full DSA channelization ladder. Report the spread of the recovered
1-Lorentzian half-width and whether it survives as a stable scale.

Same machinery applied to freya (DSA + CHIME) as the contrast case.
"""
import re
import sys
import numpy as np
from scipy.optimize import least_squares


def load(burst):
    return np.load(f"{burst}_gainladder.npz")


def ladder(Z, s):
    return sorted({int(re.match(rf"gain_{s}_ff(\d+)", k).group(1))
                   for k in Z.files if re.match(rf"gain_{s}_ff\d+$", k)}, reverse=True)


def clean(Z, s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def edge_trim(fr, g, v, frac):
    n = fr.size
    k = int(round(frac * n))
    if k == 0:
        return fr, g, v
    sl = slice(k, n - k)
    return fr[sl], g[sl], v[sl]


def resid(fr, g, v, deg, window=False):
    tr = np.polyval(np.polyfit(fr, g, deg), fr)
    tr = np.where(np.abs(tr) > 1e-12, tr, np.nan)
    r = g / tr - 1.0
    vr = v / tr ** 2
    ok = np.isfinite(r) & np.isfinite(vr)
    r, vr = r[ok], vr[ok]
    if window:
        w = np.hanning(r.size)
        # taper residual; renormalize so the variance bookkeeping is consistent
        r = r * w
        vr = vr * w ** 2
    return r, vr


def acf_nc(r, vr):
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    sig0 = ac[0] - vr.mean()
    if sig0 <= 0:
        return None, None
    return ac / sig0, float(sig0)


def lorsum(lags, p):
    out = np.zeros_like(lags)
    for i in range(0, len(p), 2):
        out += p[i] / (1.0 + (lags / p[i + 1]) ** 2)
    return out


def fit_lor1(lags, ac, chan, span, restarts=24):
    best = None
    rng = np.random.default_rng(0)
    lo = [0.0, 0.4 * chan]
    hi = [1.6, 0.9 * span]
    for _ in range(restarts):
        p0 = [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        try:
            r = least_squares(lambda p: lorsum(lags, p) - ac, p0, bounds=(lo, hi), max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return np.nan
    return float(best.x[1])


def width(Z, s, ff, deg, lagfrac, window, edgefrac):
    fr, g, v = clean(Z, s, ff)
    if fr.size < 8:
        return np.nan
    fr, g, v = edge_trim(fr, g, v, edgefrac)
    if fr.size < 8:
        return np.nan
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    r, vr = resid(fr, g, v, deg, window)
    ac, m2 = acf_nc(r, vr)
    if ac is None:
        return np.nan
    lags = np.arange(ac.size) * chan
    span = lags[-1]
    sel = (np.arange(ac.size) >= 1) & (lags <= lagfrac * span)
    if np.count_nonzero(sel) < 4:
        return np.nan
    return fit_lor1(lags[sel], ac[sel], chan, span)


def run(burst, s):
    Z = load(burst)
    ffs = ladder(Z, s)
    print(f"\n===== {burst} {s}  ladder={ffs} =====")
    DEGS = [0, 1, 2, 3]
    FRACS = [0.3, 0.5, 0.7]
    WINS = [False, True]
    EDGES = [0.0, 0.05, 0.10, 0.15]
    # per-finest full sweep
    finest = ffs[-1]
    allw = []
    print(f"--- finest ff{finest} full knob sweep ---")
    for deg in DEGS:
        row = []
        for frac in FRACS:
            for win in WINS:
                for ed in EDGES:
                    w = width(Z, s, finest, deg, frac, win, ed)
                    if np.isfinite(w):
                        allw.append(w)
                    row.append(w)
        finite = [x for x in row if np.isfinite(x)]
        if finite:
            print(f"deg={deg}: n={len(finite):2d}  med={np.median(finite):6.2f}  "
                  f"[{np.min(finite):6.2f},{np.max(finite):6.2f}]")
    allw = np.array([x for x in allw if np.isfinite(x)])
    if allw.size:
        print(f"FINEST all-knobs: n={allw.size} median={np.median(allw):.2f} "
              f"IQR=[{np.percentile(allw,25):.2f},{np.percentile(allw,75):.2f}] "
              f"range=[{allw.min():.2f},{allw.max():.2f}]")
    # ladder stability under default (deg1, frac0.5, no win, no trim) vs detrend deg
    print("--- ladder dnu_1L vs detrend degree (frac0.5,nowin,notrim) ---")
    for ff in ffs:
        ws = [width(Z, s, ff, deg, 0.5, False, 0.0) for deg in DEGS]
        print(f"ff{ff:4d}: " + "  ".join(f"d{d}={w:6.2f}" for d, w in zip(DEGS, ws)))
    return allw


if __name__ == "__main__":
    casey_dsa = run("casey", "D")
    casey_chime = run("casey", "C")
    freya_dsa = run("freya", "D")
    freya_chime = run("freya", "C")
    print("\n===== SUMMARY =====")
    for nm, arr in [("casey DSA", casey_dsa), ("casey CHIME", casey_chime),
                    ("freya DSA", freya_dsa), ("freya CHIME", freya_chime)]:
        if arr.size:
            cv = np.std(arr) / np.median(arr)
            print(f"{nm:12s}: median={np.median(arr):6.2f}  CV={cv:.2f}  "
                  f"range=[{arr.min():6.2f},{arr.max():6.2f}]  n={arr.size}")
