#!/usr/bin/env python
"""Final test: explicit smooth-envelope removal at a scale WELL ABOVE the
claimed 6 MHz scintle. If the 6 MHz is a real diffractive scale it survives
removal of structure on >25 MHz scales (a real 6 MHz scintle is orthogonal to a
25 MHz envelope). If it's envelope leakage it collapses.

Envelope = Savitzky-Golay smooth with window ~ W MHz (W >> 6). Divide it out,
re-fit. Sweep W = 15,25,40,60 MHz.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import least_squares

Z = np.load("casey_gainladder.npz")
Zf = np.load("freya_gainladder.npz")


def clean(Z, s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    fr, g, v = np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]
    o = np.argsort(fr)
    return fr[o], g[o], v[o]


def lorsum(lags, p):
    return p[0] / (1.0 + (lags / p[1]) ** 2)


def fit1(lags, ac, chan, span, restarts=20):
    best = None
    rng = np.random.default_rng(0)
    for _ in range(restarts):
        p0 = [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        r = least_squares(lambda p: lorsum(lags, p) - ac, p0,
                          bounds=([0.0, 0.4 * chan], [1.6, 0.9 * span]), max_nfev=8000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def width(r, chan):
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    ac = ac / ac[0]
    lags = np.arange(ac.size) * chan
    span = lags[-1]
    sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
    p = fit1(lags[sel], ac[sel], chan, span)
    return p[1], float(ac[1])


def envelope_sweep(Z, s, ff, label):
    fr, g, v = clean(Z, s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3  # MHz
    print(f"\n{label} {s}ff{ff} chan={chan:.3f}MHz")
    # baseline: linear detrend
    rlin = g / np.polyval(np.polyfit(fr, g, 1), fr) - 1.0
    w0, a0 = width(rlin, chan)
    print(f"  linear-detrend (current method): dnu={w0:.2f} ac1={a0:+.2f}")
    for W in (15, 25, 40, 60):
        win = int(round(W / chan))
        if win % 2 == 0:
            win += 1
        if win >= fr.size:
            win = fr.size - 1 if (fr.size - 1) % 2 == 1 else fr.size - 2
        if win < 5:
            continue
        env = savgol_filter(g, win, 2)
        r = g / env - 1.0
        w, a1 = width(r, chan)
        print(f"  envelope-removed W={W:2d}MHz (win={win}ch): dnu={w:6.2f} ac1={a1:+.2f}")


for ff in (24,):
    envelope_sweep(Z, "D", ff, "casey")
    envelope_sweep(Zf, "D", ff, "freya")
