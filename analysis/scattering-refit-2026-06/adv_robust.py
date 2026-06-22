#!/usr/bin/env python
"""Quantify the robust width range and test the envelope hypothesis directly.

Tests:
 (A) Robust width over the full knob grid, fit-failures removed
     (drop railed-to-span >0.45*span and unresolved <2*chan and m2-blowup>2).
 (B) Envelope test: split the DSA band in HALF (low 94 MHz vs high 94 MHz).
     A real diffractive screen gives the SAME width in both halves.
     A smooth bandpass-envelope artifact gives different widths / vanishes.
 (C) Phase-randomization null: scramble the FFT phases of the gain spectrum
     (destroys any real frequency-correlated structure, preserves the power
     spectrum / smooth envelope). If the recovered "width" survives phase
     scrambling, it is an envelope/windowing artifact, not scintillation.
"""
import numpy as np
from scipy.optimize import least_squares

Z = np.load("casey_gainladder.npz")
Zf = np.load("freya_gainladder.npz")


def clean(Z, s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def lorsum(lags, p):
    out = np.zeros_like(lags)
    for i in range(0, len(p), 2):
        out += p[i] / (1.0 + (lags / p[i + 1]) ** 2)
    return out


def fit1(lags, ac, chan, span, restarts=24):
    best = None
    rng = np.random.default_rng(0)
    for _ in range(restarts):
        p0 = [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        r = least_squares(lambda p: lorsum(lags, p) - ac, p0,
                          bounds=([0.0, 0.4 * chan], [1.6, 0.9 * span]), max_nfev=8000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x[0], best.x[1]


def width_from(fr, g, v, deg, lagfrac):
    if fr.size < 8:
        return None
    coef = np.polyfit(fr, g, deg)
    tr = np.polyval(coef, fr)
    r = g / tr - 1.0
    vr = v / tr ** 2
    ok = np.isfinite(r) & np.isfinite(vr)
    r, vr = r[ok], vr[ok]
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    x = r - r.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
    sig0 = ac[0] - vr.mean()
    if sig0 <= 0:
        return None
    ac = ac / sig0
    lags = np.arange(ac.size) * chan
    span = lags[-1]
    sel = (np.arange(ac.size) >= 1) & (lags <= lagfrac * span)
    if np.count_nonzero(sel) < 4:
        return None
    A, w = fit1(lags[sel], ac[sel], chan, span)
    # quality flags
    railed = w > 0.45 * span
    unres = w < 2 * chan
    blow = sig0 > 2.0
    return dict(w=w, A=A, sig0=sig0, chan=chan, span=span,
                railed=railed, unres=unres, blow=blow, ac1=float(ac[1]), ac3=float(ac[3]))


def robust_range(Z, s, ff, label):
    good = []
    flagged = 0
    for deg in (0, 1, 2, 3):
        for frac in (0.3, 0.5, 0.7):
            fr, g, v = clean(Z, s, ff)
            d = width_from(fr, g, v, deg, frac)
            if d is None:
                continue
            if d["railed"] or d["unres"] or d["blow"]:
                flagged += 1
                continue
            good.append(d["w"])
    good = np.array(good)
    if good.size:
        print(f"{label} {s}ff{ff}: ROBUST(clean) n={good.size} flagged={flagged} "
              f"med={np.median(good):.2f} range=[{good.min():.2f},{good.max():.2f}] "
              f"CV={np.std(good)/np.median(good):.2f}")
    else:
        print(f"{label} {s}ff{ff}: NO clean width survives (all {flagged} flagged)")
    return good


def half_band(Z, s, ff, label):
    fr, g, v = clean(Z, s, ff)
    order = np.argsort(fr)
    fr, g, v = fr[order], g[order], v[order]
    half = fr.size // 2
    out = []
    for tag, sl in [("low", slice(0, half)), ("high", slice(half, None))]:
        d = width_from(fr[sl], g[sl], v[sl], 1, 0.5)
        if d:
            out.append((tag, d["w"], d["ac1"]))
    print(f"  HALF-BAND {label} {s}ff{ff}: " +
          "  ".join(f"{t}: dnu={w:.2f}(ac1={a:+.2f})" for t, w, a in out))
    return out


def phase_null(Z, s, ff, label, ntrial=40):
    fr, g, v = clean(Z, s, ff)
    order = np.argsort(fr)
    fr, g, v = fr[order], g[order], v[order]
    # real width
    d = width_from(fr, g, v, 1, 0.5)
    wreal = d["w"] if d else np.nan
    # detrend, take residual modulation, randomize phases preserving power spectrum
    coef = np.polyfit(fr, g, 1)
    r = g / np.polyval(coef, fr) - 1.0
    R = np.fft.rfft(r)
    mag = np.abs(R)
    rng = np.random.default_rng(1)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    ws = []
    for _ in range(ntrial):
        ph = rng.uniform(0, 2 * np.pi, mag.size)
        ph[0] = 0.0
        Rn = mag * np.exp(1j * ph)
        rn = np.fft.irfft(Rn, n=r.size)
        x = rn - rn.mean()
        ac = np.correlate(x, x, "full")[x.size - 1:] / x.size
        ac = ac / ac[0]
        lags = np.arange(ac.size) * chan
        span = lags[-1]
        sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
        _, w = fit1(lags[sel], ac[sel], chan, span)
        ws.append(w)
    ws = np.array(ws)
    print(f"  PHASE-NULL {label} {s}ff{ff}: real dnu={wreal:.2f}  "
          f"phase-scrambled med={np.median(ws):.2f} [{np.percentile(ws,5):.2f},{np.percentile(ws,95):.2f}]")
    return wreal, ws


print("===== (A) ROBUST clean width range =====")
for ff in (384, 192, 96, 48, 24):
    robust_range(Z, "D", ff, "casey")
print()
for ff in (384, 192, 96, 48, 24):
    robust_range(Zf, "D", ff, "freya")

print("\n===== (B) HALF-BAND consistency (real screen => same width both halves) =====")
for ff in (48, 24):
    half_band(Z, "D", ff, "casey")
for ff in (48, 24):
    half_band(Zf, "D", ff, "freya")

print("\n===== (C) PHASE-RANDOMIZATION NULL (artifact => null reproduces real width) =====")
for ff in (96, 48, 24):
    phase_null(Z, "D", ff, "casey")
for ff in (48, 24):
    phase_null(Zf, "D", ff, "freya")
