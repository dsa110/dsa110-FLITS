#!/usr/bin/env python
"""Adversarial referee: TRY to show a 2nd scintle scale is real on DSA.

Bootstrap the per-channel residuals -> rebuild noise-corrected ACF -> refit
1- and 2-Lorentzian each draw. Record the 2nd-component amplitude & width
distributions. If A2 is consistently >0 across draws, N=2 is defensible.
If A2 scatters through 0, the 2nd scale is noise.

Two bootstrap modes:
  (A) parametric: add per-channel Gaussian noise std=sqrt(var) to the gains
  (B) residual resample: resample the detrended fractional residuals with
      replacement (preserves the real ACF structure; kills ordering -> any
      surviving narrow ACF feature is an artifact of the noise floor / fit).

Mode A is the physically correct null+signal generator: it perturbs the data
by its OWN stated measurement noise and asks whether the 2nd component
survives. That is the test the claim must pass.
"""
import json
import numpy as np
from scipy.optimize import least_squares

Z = np.load("freya_gainladder.npz")
NU0 = {"C": 0.6, "D": 1.405}


def clean(s, ff):
    fr, g, v = Z[f"freq_{s}_ff{ff}"], Z[f"gain_{s}_ff{ff}"], Z[f"var_{s}_ff{ff}"]
    ok = np.isfinite(g) & np.isfinite(v) & (v > 0) & (g != 0)
    return np.asarray(fr)[ok], np.asarray(g)[ok], np.asarray(v)[ok]


def resid(fr, g, v, deg=1):
    """Linear-detrend -> fractional residual + propagated noise var."""
    tr = np.polyval(np.polyfit(fr, g, deg), fr)
    tr = np.where(np.abs(tr) > 1e-12, tr, np.nan)
    r = g / tr - 1.0
    vr = v / tr ** 2
    ok = np.isfinite(r) & np.isfinite(vr)
    return r[ok], vr[ok]


def acf_nc(r, vr):
    """Noise-corrected normalized ACF: subtract mean(var) from lag 0."""
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
        return None, np.inf
    return best.x, float(np.sum((lorsum(lags, best.x) - ac) ** 2))


def sorted_comps(p):
    """Return [(width, amp), ...] sorted by ascending width (narrow first)."""
    return sorted([(p[i + 1], p[i]) for i in range(0, len(p), 2)])


def run_band(s, ff, nboot=300, mode="param"):
    fr, g, v = clean(s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3  # MHz/chan
    # point estimate on real data
    r0, vr0 = resid(fr, g, v)
    ac0, m2_0 = acf_nc(r0, vr0)
    lags0 = np.arange(ac0.size) * chan
    span0 = lags0[-1]
    sel0 = (np.arange(ac0.size) >= 1) & (lags0 <= 0.5 * span0)
    p1_0, rss1_0 = fit_lor(lags0[sel0], ac0[sel0], 1, chan, span0, seed=11)
    p2_0, rss2_0 = fit_lor(lags0[sel0], ac0[sel0], 2, chan, span0, seed=22)
    n0 = int(np.count_nonzero(sel0))

    def bic(rss, n, k):
        return n * np.log(rss / n + 1e-300) + k * np.log(n)

    rng = np.random.default_rng(abs(hash((s, ff))) % (2 ** 32))
    A_narrow, W_narrow, A_wide, W_wide = [], [], [], []
    A1, W1 = [], []
    dbic = []  # BIC(N=2) - BIC(N=1); <0 means N=2 wins this draw
    f_improve = []  # fractional RSS improvement N1->N2
    nb_ok = 0
    for b in range(nboot):
        if mode == "param":
            gb = g + rng.normal(0.0, np.sqrt(v))
        else:  # residual resample (destroys ACF structure -> null)
            rb = rng.choice(r0, size=r0.size, replace=True)
            # rebuild a pseudo-gain that detrends to resampled residuals
            tr = np.polyval(np.polyfit(fr, g, 1), fr)
            gb = tr * (1.0 + rb)
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
        p1, rss1 = fit_lor(lagsB[selB], acB[selB], 1, chan, spanB, restarts=12, seed=b)
        p2, rss2 = fit_lor(lagsB[selB], acB[selB], 2, chan, spanB, restarts=12, seed=b + 999)
        if p1 is None or p2 is None:
            continue
        nb_ok += 1
        c1 = sorted_comps(p1)
        c2 = sorted_comps(p2)
        A1.append(c1[0][1]); W1.append(c1[0][0])
        (wn, an), (ww, aw) = c2[0], c2[1]
        A_narrow.append(an); W_narrow.append(wn)
        A_wide.append(aw); W_wide.append(ww)
        dbic.append(bic(rss2, nB, 4) - bic(rss1, nB, 2))
        f_improve.append((rss1 - rss2) / max(rss1, 1e-300))

    out = {
        "ff": ff, "chan_MHz": round(chan, 4), "nch": int(fr.size),
        "n_lags_fit": n0, "m2_point": round(m2_0, 5), "nboot_ok": nb_ok,
        "point": {
            "dnu_1L": round(float(p1_0[1]), 4), "rss1": round(rss1_0, 5),
            "N2_comps_wA": [(round(w, 4), round(a, 4)) for w, a in sorted_comps(p2_0)],
            "rss2": round(rss2_0, 5),
            "BIC_N1": round(bic(rss1_0, n0, 2), 3),
            "BIC_N2": round(bic(rss2_0, n0, 4), 3),
            "delta_BIC_N2_minus_N1": round(bic(rss2_0, n0, 4) - bic(rss1_0, n0, 2), 3),
        },
    }
    if nb_ok > 0:
        An = np.array(A_narrow); Wn = np.array(W_narrow)
        Aw = np.array(A_wide); Ww = np.array(W_wide)
        db = np.array(dbic); fi = np.array(f_improve)
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
            "delta_BIC_median": round(float(np.median(db)), 3),
            "frac_draws_N2_wins_BIC": round(float(np.mean(db < 0)), 3),
            "frac_RSS_improve_gt_5pct": round(float(np.mean(fi > 0.05)), 3),
            "RSS_improve_median_pct": round(float(np.median(fi) * 100), 3),
        }
    return out


if __name__ == "__main__":
    import sys
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    results = {"nboot": nboot, "param": {}, "resample": {}}
    for mode in ("param", "resample"):
        for ff in (24, 48, 96):
            print(f"=== DSA ff{ff} mode={mode} ===", flush=True)
            r = run_band("D", ff, nboot=nboot, mode=mode)
            results[mode][f"ff{ff}"] = r
            print(json.dumps(r, indent=2), flush=True)
    json.dump(results, open("adv_bootstrap_results.json", "w"), indent=2)
    print("\nWROTE adv_bootstrap_results.json", flush=True)
