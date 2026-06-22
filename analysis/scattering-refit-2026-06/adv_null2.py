#!/usr/bin/env python
"""Correct nulls + sub-band split.

(C2) CHANNEL-SHUFFLE null: randomly permute the detrended residual across
     channels. This DESTROYS any genuine frequency-ordered correlation
     (real scintillation) while preserving the marginal amplitude distribution
     (the modulation / m^2). A real diffractive scale must VANISH (width -> ~chan,
     ac1 -> 0) under shuffling. If the recovered width persists, it is an
     amplitude/envelope artifact independent of channel ordering.
     (Phase-randomization was wrong: Wiener-Khinchin => phase scramble leaves
      the ACF invariant, so it can't discriminate. Channel shuffle is the
      correct ordering-destroying null.)

(D) QUARTER-BAND: split DSA band into 4 sub-bands, fit each.
     Real screen: 4 consistent widths (slow nu^4 drift only).
     Envelope/edge artifact: erratic, with some quarters decorrelated.
"""
import numpy as np
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


def acf_width(r, chan, lagfrac=0.5):
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    ac = ac / ac[0]
    lags = np.arange(ac.size) * chan
    span = lags[-1]
    sel = (np.arange(ac.size) >= 1) & (lags <= lagfrac * span)
    if np.count_nonzero(sel) < 4:
        return np.nan, np.nan
    p = fit1(lags[sel], ac[sel], chan, span)
    return p[1], float(ac[1])


def detrend(fr, g, deg=1):
    return g / np.polyval(np.polyfit(fr, g, deg), fr) - 1.0


def shuffle_null(Z, s, ff, label, ntrial=200):
    fr, g, v = clean(Z, s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    r = detrend(fr, g)
    wreal, ac1real = acf_width(r, chan)
    rng = np.random.default_rng(7)
    ws, a1s = [], []
    for _ in range(ntrial):
        rs = rng.permutation(r)
        w, a1 = acf_width(rs, chan)
        if np.isfinite(w):
            ws.append(w); a1s.append(a1)
    ws, a1s = np.array(ws), np.array(a1s)
    print(f"  SHUFFLE {label} {s}ff{ff}: REAL dnu={wreal:.2f}(ac1={ac1real:+.2f})  "
          f"shuffled dnu med={np.median(ws):.2f}[{np.percentile(ws,5):.2f},{np.percentile(ws,95):.2f}] "
          f"ac1 med={np.median(a1s):+.2f}")


def quarters(Z, s, ff, label):
    fr, g, v = clean(Z, s, ff)
    n = fr.size
    q = n // 4
    out = []
    for i in range(4):
        sl = slice(i * q, (i + 1) * q if i < 3 else n)
        frq, gq = fr[sl], g[sl]
        chan = float(np.median(np.abs(np.diff(frq)))) * 1e3
        r = detrend(frq, gq)
        w, a1 = acf_width(r, chan)
        out.append((frq.mean(), w, a1))
    print(f"  QUARTERS {label} {s}ff{ff}:")
    for fc, w, a1 in out:
        print(f"     nu~{fc:.1f}GHz: dnu={w:6.2f}MHz ac1={a1:+.2f}")
    ws = np.array([w for _, w, _ in out])
    print(f"     -> quarter widths spread: {ws.min():.2f}..{ws.max():.2f} "
          f"(factor {ws.max()/ws.min():.1f}x)")


print("===== (C2) CHANNEL-SHUFFLE NULL =====")
print("(real screen: shuffled dnu collapses to ~chan, ac1~0)")
for ff in (96, 48, 24):
    shuffle_null(Z, "D", ff, "casey")
for ff in (48, 24):
    shuffle_null(Zf, "D", ff, "freya")

print("\n===== (D) QUARTER-BAND consistency =====")
for ff in (24,):
    quarters(Z, "D", ff, "casey")
for ff in (24,):
    quarters(Zf, "D", ff, "freya")
