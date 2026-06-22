#!/usr/bin/env python
"""Adversarial referee for WILHELM: TRY to show a 2nd scintle scale is real on DSA.

Bootstrap the noise-corrected gain-spectrum ACF -> refit 1- and 2-Lorentzian
each draw. Record the 2nd-component amplitude/width distributions + per-draw
BIC and the SECOND (narrow) component's behaviour:
  - real:  A2 stays > 0 with a stable, channel-independent width that recurs
  - artifact: A2 scatters through 0, OR the narrow width pins to the fit's
              lower bound (0.4*chan), OR the two Lorentzians collapse to the
              same width (degenerate -> N=2 is just N=1 split in two).

Two bootstrap modes:
  (A) parametric: add per-channel N(0, sqrt(var)) to the gains (the data's OWN
      stated measurement noise) -> this is the signal+noise generator the
      N>=2 claim must survive.
  (B) residual resample: shuffle the detrended fractional residuals with
      replacement -> destroys ACF ordering -> a pure null. Any narrow ACF
      feature that survives here is a noise-floor/fit artifact, not structure.
"""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

BURST = sys.argv[1] if len(sys.argv) > 1 else "wilhelm"
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
Z = np.load(f"{BURST}_gainladder.npz")
NU0 = {"C": 0.6, "D": 1.405}


def clean(s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def resid(fr, g, v, deg=1):
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


def fit_lor(lags, ac, N, chan, span, restarts=24, seed=0):
    best = None
    rng = np.random.default_rng(seed)
    lo = [0.0, 0.4 * chan] * N
    hi = [1.6, 0.9 * span] * N
    for _ in range(restarts):
        p0 = []
        for _i in range(N):
            p0 += [rng.uniform(0.05, 0.7), rng.uniform(chan, 0.6 * span)]
        try:
            r = least_squares(lambda p: lorsum(lags, p) - ac, p0,
                              bounds=(lo, hi), max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None, np.inf, None, None
    rss = float(np.sum((lorsum(lags, best.x) - ac) ** 2))
    return best.x, rss, lo, hi


def sorted_comps(p):
    return sorted([(p[i + 1], p[i]) for i in range(0, len(p), 2)])


def bic(rss, n, k):
    return n * np.log(rss / n + 1e-300) + k * np.log(n)


def run_band(s, ff, nboot, mode):
    fr, g, v = clean(s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    lo_w = 0.4 * chan  # narrow-width lower bound (pin detector)

    r0, vr0 = resid(fr, g, v)
    ac0, m2_0 = acf_nc(r0, vr0)
    lags0 = np.arange(ac0.size) * chan
    span0 = lags0[-1]
    sel0 = (np.arange(ac0.size) >= 1) & (lags0 <= 0.5 * span0)
    p1_0, rss1_0, _, _ = fit_lor(lags0[sel0], ac0[sel0], 1, chan, span0, seed=11)
    p2_0, rss2_0, _, _ = fit_lor(lags0[sel0], ac0[sel0], 2, chan, span0, seed=22)
    n0 = int(np.count_nonzero(sel0))

    rng = np.random.default_rng(abs(hash((BURST, s, ff))) % (2 ** 32))
    A_narrow, W_narrow, A_wide, W_wide = [], [], [], []
    dbic, f_improve = [], []
    pin_lo, degenerate = 0, 0
    nb_ok = 0
    tr_full = np.polyval(np.polyfit(fr, g, 1), fr)
    for b in range(nboot):
        if mode == "param":
            gb = g + rng.normal(0.0, np.sqrt(v))
        else:
            rb = rng.choice(r0, size=r0.size, replace=True)
            gb = tr_full * (1.0 + rb)
        rB, vrB = resid(fr, gb, v)
        acB, m2B = acf_nc(rB, vrB)
        if acB is None:
            continue
        lagsB = np.arange(acB.size) * chan
        spanB = lagsB[-1]
        selB = (np.arange(acB.size) >= 1) & (lagsB <= 0.5 * spanB)
        nB = int(np.count_nonzero(selB))
        if nB < 5:
            continue
        p1, rss1, _, _ = fit_lor(lagsB[selB], acB[selB], 1, chan, spanB, restarts=12, seed=b)
        p2, rss2, _, _ = fit_lor(lagsB[selB], acB[selB], 2, chan, spanB, restarts=12, seed=b + 99991)
        if p1 is None or p2 is None:
            continue
        nb_ok += 1
        c2 = sorted_comps(p2)
        (wn, an), (ww, aw) = c2[0], c2[1]
        A_narrow.append(an); W_narrow.append(wn)
        A_wide.append(aw); W_wide.append(ww)
        if wn <= lo_w * 1.05:
            pin_lo += 1
        if abs(ww - wn) / max(ww, 1e-9) < 0.1:
            degenerate += 1
        dbic.append(bic(rss2, nB, 4) - bic(rss1, nB, 2))
        f_improve.append((rss1 - rss2) / max(rss1, 1e-300))

    out = {"ff": ff, "chan_MHz": round(chan, 4), "nch": int(fr.size),
           "lo_width_bound_MHz": round(lo_w, 4), "n_lags_fit": n0,
           "m2_point": round(m2_0, 5), "nboot_ok": nb_ok,
           "point": {
               "dnu_1L": round(float(p1_0[1]), 4), "rss1": round(rss1_0, 6),
               "N2_comps_wA": [(round(w, 4), round(a, 4)) for w, a in sorted_comps(p2_0)],
               "rss2": round(rss2_0, 6),
               "BIC_N1": round(bic(rss1_0, n0, 2), 3),
               "BIC_N2": round(bic(rss2_0, n0, 4), 3),
               "delta_BIC_N2_minus_N1": round(bic(rss2_0, n0, 4) - bic(rss1_0, n0, 2), 3)}}
    if nb_ok > 0:
        An, Wn = np.array(A_narrow), np.array(W_narrow)
        Aw, Ww = np.array(A_wide), np.array(W_wide)
        db, fi = np.array(dbic), np.array(f_improve)
        out["boot"] = {
            "A_narrow_median": round(float(np.median(An)), 4),
            "A_narrow_q05": round(float(np.percentile(An, 5)), 4),
            "A_narrow_q95": round(float(np.percentile(An, 95)), 4),
            "A_narrow_frac_gt_0p02": round(float(np.mean(An > 0.02)), 3),
            "A_narrow_frac_gt_0p1": round(float(np.mean(An > 0.1)), 3),
            "W_narrow_median": round(float(np.median(Wn)), 4),
            "W_narrow_q05": round(float(np.percentile(Wn, 5)), 4),
            "W_narrow_q95": round(float(np.percentile(Wn, 95)), 4),
            "W_narrow_iqr_over_med": round(float((np.percentile(Wn, 75) - np.percentile(Wn, 25)) / max(np.median(Wn), 1e-9)), 3),
            "A_wide_median": round(float(np.median(Aw)), 4),
            "W_wide_median": round(float(np.median(Ww)), 4),
            "frac_narrow_pinned_to_lo_bound": round(pin_lo / nb_ok, 3),
            "frac_two_comps_degenerate": round(degenerate / nb_ok, 3),
            "delta_BIC_median": round(float(np.median(db)), 3),
            "frac_draws_N2_wins_BIC": round(float(np.mean(db < 0)), 3),
            "frac_RSS_improve_gt_5pct": round(float(np.mean(fi > 0.05)), 3),
            "RSS_improve_median_pct": round(float(np.median(fi) * 100), 3),
        }
    return out


if __name__ == "__main__":
    results = {"burst": BURST, "nboot": NBOOT, "param": {}, "resample": {}}
    for mode in ("param", "resample"):
        for ff in (24, 48):  # finest + one coarser usable DSA channelization
            print(f"=== {BURST} DSA ff{ff} mode={mode} ===", flush=True)
            r = run_band("D", ff, nboot=NBOOT, mode=mode)
            results[mode][f"ff{ff}"] = r
            print(json.dumps(r, indent=2), flush=True)
    json.dump(results, open(f"{BURST}_adv_bootstrap_results.json", "w"), indent=2)
    print(f"\nWROTE {BURST}_adv_bootstrap_results.json", flush=True)
