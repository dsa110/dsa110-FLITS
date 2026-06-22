#!/usr/bin/env python
"""Master multi-screen test: fit up to 3 Lorentzian scintle scales for a burst.

Per band, across the channelization ladder, on the noise-corrected gain-spectrum
ACF:
  (1) 1/2/3-Lorentzian fits + BIC model selection at the finest usable channels;
  (2) detrend sensitivity (mean/linear/quad) -- a real intermediate scale
      survives modest detrending; a pure broadband envelope does not;
  (3) Delta_nu_d(1L) stability across channelization -- a RESOLVED scale recurs
      at the same width; an unresolved one tracks the channel;
  (4) modulation index m^2 vs 1/chan -- slope = any UNRESOLVED component;
  (5) cross-band nu-scaling -- a common diffractive screen needs
      Delta_nu_d ∝ nu^~4 (DSA/CHIME ≈ (1.4/0.6)^4 ≈ 30).
Channelizations with median gain S/N < SNR_MIN are dropped as noise-dominated.

  python multiscale_fit.py <burst>     e.g.  python multiscale_fit.py casey
Writes <burst>_multiscale_results.json.
"""
import json
import re
import sys
import numpy as np
from scipy.optimize import least_squares

BURST = sys.argv[1] if len(sys.argv) > 1 else "freya"
SNR_MIN = 3.0
Z = np.load(f"{BURST}_gainladder.npz")
TAU, AL = [float(x) for x in Z["tau_alpha"]]
NAME = {"C": "CHIME", "D": "DSA"}


def ladder(s):
    ffs = sorted({int(re.match(rf"gain_{s}_ff(\d+)", k).group(1))
                  for k in Z.files if re.match(rf"gain_{s}_ff\d+$", k)}, reverse=True)
    return ffs


def clean(s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    g, v = np.asarray(g), np.asarray(v)
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    # EDGE MASK: where |g| ~ 0 the g/trend-1 normalization blows up and fabricates
    # a spurious broadband ACF (verified artifact; casey DSA crosses zero). Drop the
    # bottom 5% of |median gain|. ponytail: 5% floor, tighten if a real burst needs it.
    if ok.sum() >= 8:
        ok &= np.abs(g) > 0.05 * np.median(np.abs(g[ok]))
    return np.asarray(fr)[ok], g[ok], v[ok]


def usable(s, ff):
    fr, g, v = clean(s, ff)
    if fr.size < 8:
        return None
    snr = float(np.nanmedian(g / np.sqrt(v)))
    return (fr, g, v, snr) if snr >= SNR_MIN else None


def resid(fr, g, v, deg):
    tr = np.polyval(np.polyfit(fr, g, deg), fr)
    tr = np.where(np.abs(tr) > 1e-12, tr, np.nan)
    r = g / tr - 1.0
    vr = v / tr ** 2
    ok = np.isfinite(r) & np.isfinite(vr)
    return r[ok], vr[ok]


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


def fit_lor(lags, ac, N, chan, span, restarts=16):
    best = None
    rng = np.random.default_rng(N)
    lo = [0.0, 0.4 * chan] * N
    hi = [1.6, 0.9 * span] * N
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
    if best is None:
        return None, np.inf
    return best.x, float(np.sum((lorsum(lags, best.x) - ac) ** 2))


def comps(p):
    cs = sorted([(p[i + 1], p[i]) for i in range(0, len(p), 2)])
    return [(round(dn, 3), round(A, 3)) for dn, A in cs]


def bic(rss, n, k):
    return n * np.log(rss / n + 1e-300) + k * np.log(n)


def band_center(s, ff):
    return float(np.median(Z[f"freq_{s}_ff{ff}"]))


res = {"burst": BURST, "tau_1ghz": TAU, "alpha": AL, "bands": {}}
NU0 = {}
for s in ("C", "D"):
    ffs = ladder(s)
    if not ffs:
        continue
    NU0[s] = band_center(s, ffs[0])
    b = {"ladder": [], "dnu_1L_stability": [], "m2_vs_invchan": [],
         "modelsel_finest": {}, "detrend_sensitivity": {}, "nu0_GHz": round(NU0[s], 4)}
    finest = None
    for ff in ffs:
        u = usable(s, ff)
        if u is None:
            b["ladder"].append({"ff": ff, "dropped": "snr<%.0f or <8ch" % SNR_MIN})
            continue
        fr, g, v, snr = u
        finest = ff
        chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
        r, vr = resid(fr, g, v, 1)
        ac, m2 = acf_nc(r, vr)
        if ac is None:
            b["ladder"].append({"ff": ff, "chan_MHz": round(chan, 3), "m2": None}); continue
        lags = np.arange(ac.size) * chan
        span = lags[-1]
        sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
        p1, _ = fit_lor(lags[sel], ac[sel], 1, chan, span)
        b["ladder"].append({"ff": ff, "chan_MHz": round(chan, 3), "nch": int(fr.size),
                            "snr": round(snr, 1), "m2": round(m2, 5),
                            "acf1": round(float(ac[1]), 3), "dnu_1L": round(float(p1[1]), 3)})
        b["dnu_1L_stability"].append(round(float(p1[1]), 3))
        b["m2_vs_invchan"].append((round(1.0 / chan, 4), round(m2, 5)))
    if finest is not None:
        fr, g, v, _ = usable(s, finest)
        chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
        for deg, tag in [(0, "mean"), (1, "linear"), (2, "quad")]:
            r, vr = resid(fr, g, v, deg)
            ac, m2 = acf_nc(r, vr)
            if ac is None:
                continue
            lags = np.arange(ac.size) * chan; span = lags[-1]
            sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
            p1, _ = fit_lor(lags[sel], ac[sel], 1, chan, span)
            b["detrend_sensitivity"][tag] = {"m2": round(m2, 5), "dnu_1L": round(float(p1[1]), 3)}
        r, vr = resid(fr, g, v, 1); ac, m2 = acf_nc(r, vr)
        lags = np.arange(ac.size) * chan; span = lags[-1]
        sel = (np.arange(ac.size) >= 1) & (lags <= 0.5 * span)
        n = int(np.count_nonzero(sel))
        for N in (1, 2, 3):
            if n < 2 * N + 1:
                continue
            p, rss = fit_lor(lags[sel], ac[sel], N, chan, span)
            b["modelsel_finest"][N] = {"components_dnu_A": comps(p), "rss": round(rss, 5),
                                       "bic": round(bic(rss, n, 2 * N), 2), "nlags": n}
        bics = {N: d["bic"] for N, d in b["modelsel_finest"].items()}
        b["BIC_preferred_N"] = int(min(bics, key=bics.get)) if bics else None
        b["finest_chan_MHz"] = round(chan, 3)
    res["bands"][NAME[s]] = b

# cross-band nu-scaling (only if both bands have usable widths)
if all(res["bands"].get(NAME[s], {}).get("dnu_1L_stability") for s in ("C", "D")):
    dC = float(np.median(res["bands"]["CHIME"]["dnu_1L_stability"]))
    dD = float(np.median(res["bands"]["DSA"]["dnu_1L_stability"]))
    res["nu_scaling"] = {
        "dnu_CHIME_MHz": round(dC, 3), "dnu_DSA_MHz": round(dD, 3),
        "observed_ratio_DSA_over_CHIME": round(dD / dC, 3),
        "expected_ratio_nu4": round((NU0["D"] / NU0["C"]) ** 4, 3),
        "implied_alpha": round(float(np.log(dD / dC) / np.log(NU0["D"] / NU0["C"])), 3),
    }
else:
    res["nu_scaling"] = {"note": "one band lacks usable channelizations (S/N) -- cross-band test skipped"}
res["tau_screen_dnu_MHz"] = {NAME[s]: 1.0 / (2 * np.pi * (TAU * NU0[s] ** (-AL) * 1e-3)) / 1e6
                             for s in NU0}
json.dump(res, open(f"{BURST}_multiscale_results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
