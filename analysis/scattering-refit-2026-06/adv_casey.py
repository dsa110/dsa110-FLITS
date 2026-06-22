#!/usr/bin/env python
"""Adversarial referee on CASEY DSA: try to make a 2nd Lorentzian scale real.

Finest usable DSA channelization = ff24 (256 ch, 0.735 MHz, SNR 12.4);
one coarser = ff48 (128 ch, 1.476 MHz, SNR 18.1).

For each: point fit of 1L vs 2L on the noise-corrected gain-ACF, then
>=500 bootstraps in two modes:
  (A) param   -- add per-channel N(0, sqrt(var)) to the gains (signal+noise)
  (B) resample-- resample fractional residuals w/ replacement (kills ordering;
                 destroys the real ACF -> a pure null).
N=2 is given a genuinely fair shot: the 2nd component is allowed a width
regime distinct from the broad one (restarts seed narrow+wide), and we report
whether the narrow amplitude is STABLE (>0, tight width) -> real, or PINS to
bound / scatters through 0 -> artifact. BIC counts how often N=2 actually wins.
"""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

Z = np.load("casey_gainladder.npz")


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


def fit_lor(lags, ac, N, chan, span, restarts=24, seed=0, sep_seed=True):
    """N-Lorentzian LS. For N==2, half the restarts seed a narrow+wide split
    so a genuinely distinct 2nd scale can be found (not just a degenerate twin)."""
    best = None
    rng = np.random.default_rng(seed)
    lo = [0.0, 0.4 * chan] * N
    hi = [1.6, 0.9 * span] * N
    for k in range(restarts):
        p0 = []
        if N == 2 and sep_seed and k % 2 == 0:
            # narrow component near a few channels, wide near the envelope
            p0 = [rng.uniform(0.05, 0.8), rng.uniform(chan, 6 * chan),
                  rng.uniform(0.05, 0.8), rng.uniform(0.15 * span, 0.6 * span)]
        else:
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
    return sorted([(p[i + 1], p[i]) for i in range(0, len(p), 2)])  # (width, amp)


def bic(rss, n, k):
    return n * np.log(rss / n + 1e-300) + k * np.log(n)


def run_band(s, ff, nboot=500):
    fr, g, v = clean(s, ff)
    chan = float(np.median(np.abs(np.diff(fr)))) * 1e3
    r0, vr0 = resid(fr, g, v)
    ac0, m2_0 = acf_nc(r0, vr0)
    lags0 = np.arange(ac0.size) * chan
    span0 = lags0[-1]
    sel0 = (np.arange(ac0.size) >= 1) & (lags0 <= 0.5 * span0)
    n0 = int(np.count_nonzero(sel0))
    p1_0, rss1_0 = fit_lor(lags0[sel0], ac0[sel0], 1, chan, span0, restarts=40, seed=11)
    p2_0, rss2_0 = fit_lor(lags0[sel0], ac0[sel0], 2, chan, span0, restarts=40, seed=22)
    span_max = 0.9 * span0  # the width upper bound

    out = {
        "ff": ff, "chan_MHz": round(chan, 4), "nch": int(fr.size),
        "n_lags_fit": n0, "m2_point": round(m2_0, 5),
        "width_upper_bound_MHz": round(span_max, 4),
        "point": {
            "dnu_1L": round(float(p1_0[1]), 4), "rss1": round(rss1_0, 6),
            "N2_comps_wA": [(round(w, 4), round(a, 4)) for w, a in sorted_comps(p2_0)],
            "rss2": round(rss2_0, 6),
            "BIC_N1": round(bic(rss1_0, n0, 2), 3),
            "BIC_N2": round(bic(rss2_0, n0, 4), 3),
            "delta_BIC_N2_minus_N1": round(bic(rss2_0, n0, 4) - bic(rss1_0, n0, 2), 3),
            "RSS_improve_pct": round(100 * (rss1_0 - rss2_0) / max(rss1_0, 1e-300), 4),
        },
    }

    modes = {}
    for mode in ("param", "resample"):
        rng = np.random.default_rng((abs(hash((s, ff, mode))) % (2 ** 31)))
        A_n, W_n, A_w, W_w, dbic, fimp = [], [], [], [], [], []
        nb_ok = 0
        tr1 = np.polyval(np.polyfit(fr, g, 1), fr)
        for b in range(nboot):
            if mode == "param":
                gb = g + rng.normal(0.0, np.sqrt(v))
            else:
                rb = rng.choice(r0, size=r0.size, replace=True)
                gb = tr1 * (1.0 + rb)
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
            p1, rss1 = fit_lor(lagsB[selB], acB[selB], 1, chan, spanB, restarts=16, seed=b)
            p2, rss2 = fit_lor(lagsB[selB], acB[selB], 2, chan, spanB, restarts=16, seed=b + 7919)
            if p1 is None or p2 is None:
                continue
            nb_ok += 1
            (wn, an), (ww, aw) = sorted_comps(p2)
            A_n.append(an); W_n.append(wn); A_w.append(aw); W_w.append(ww)
            dbic.append(bic(rss2, nB, 4) - bic(rss1, nB, 2))
            fimp.append((rss1 - rss2) / max(rss1, 1e-300))
        An = np.array(A_n); Wn = np.array(W_n); Aw = np.array(A_w); Ww = np.array(W_w)
        db = np.array(dbic); fi = np.array(fimp)
        # pin diagnostics: narrow width at its lower bound (0.4*chan) or upper (span_max)
        wlo = 0.4 * chan
        modes[mode] = {
            "nboot_ok": nb_ok,
            "A_narrow_median": round(float(np.median(An)), 4),
            "A_narrow_q05": round(float(np.percentile(An, 5)), 4),
            "A_narrow_q95": round(float(np.percentile(An, 95)), 4),
            "A_narrow_frac_lt_0p02": round(float(np.mean(An < 0.02)), 3),
            "A_narrow_frac_gt_0p1": round(float(np.mean(An > 0.1)), 3),
            "W_narrow_median": round(float(np.median(Wn)), 4),
            "W_narrow_q05": round(float(np.percentile(Wn, 5)), 4),
            "W_narrow_q95": round(float(np.percentile(Wn, 95)), 4),
            "W_narrow_IQR_over_med": round(float(
                (np.percentile(Wn, 75) - np.percentile(Wn, 25)) / max(np.median(Wn), 1e-9)), 3),
            "W_narrow_frac_at_lo_bound": round(float(np.mean(np.abs(Wn - wlo) < 0.02 * wlo)), 3),
            "W_wide_median": round(float(np.median(Ww)), 4),
            "A_wide_median": round(float(np.median(Aw)), 4),
            "delta_BIC_median": round(float(np.median(db)), 3),
            "delta_BIC_q05": round(float(np.percentile(db, 5)), 3),
            "frac_draws_N2_wins_BIC": round(float(np.mean(db < 0)), 4),
            "frac_RSS_improve_gt_2pct": round(float(np.mean(fi > 0.02)), 3),
            "RSS_improve_median_pct": round(float(np.median(fi) * 100), 4),
        }
    out["modes"] = modes
    return out


if __name__ == "__main__":
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    results = {"burst": "casey", "band": "DSA", "nboot": nboot, "bands": {}}
    for ff in (24, 48):
        print(f"=== casey DSA ff{ff} (nboot={nboot}) ===", flush=True)
        r = run_band("D", ff, nboot=nboot)
        results["bands"][f"ff{ff}"] = r
        print(json.dumps(r, indent=2), flush=True)
    json.dump(results, open("adv_casey_results.json", "w"), indent=2)
    print("\nWROTE adv_casey_results.json", flush=True)
