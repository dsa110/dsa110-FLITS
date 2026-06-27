#!/usr/bin/env python
"""Independent sub-banded tau(nu) validation of the joint all-exp scattering fits.

Model-independent cross-check of the joint CHIME-DSA shared (tau_1ghz, alpha):
split each band into sub-bands, collapse each to a 1D profile, fit a scattered-
pulse profile (exponentially-modified Gaussian = Gaussian intrinsic (x) one-sided
exp PBF; tau = scattering timescale at that sub-band) INDEPENDENTLY, then check
whether the per-sub-band tau_i(nu_i) trace the joint power law tau_1ghz*nu^-alpha.

Because tau floats free per sub-band, the sub-band alpha (slope of log tau vs
log nu over detections) is NOT imposed by the joint model -- agreement is a
genuine validation. EMG is the textbook thin-screen scattered pulse and shares
no machinery with the 2D joint fitter (scipy.stats.exponnorm, tau=K*sigma).

Multi-component bursts: a single EMG cannot fit overlapping sub-pulses, so the
component POSITIONS (t0) and INTRINSIC WIDTHS (zeta) are fixed from the joint fit
medians and only the per-component amplitudes + ONE SHARED tau float per sub-band
-- tau stays an independent measurement; only the (well-determined) component
structure is borrowed. Single-component (sharedzeta) bursts fit mu/sigma/tau free.

Detection vs upper limit: tau is a detection if tau > 2*sigma_tau and tau > 0.5*dt
(dt = sub-band time sample); otherwise a 2-sigma upper limit (tau is unresolved at
that frequency -- expected at high DSA frequencies where tau ~ tau_1ghz/5).

  python _subband_tau_validation.py [b1 b2 ...]
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/analysis/scattering-refit-2026-06")  # _figsave, joint_ppc
sys.path.insert(0, f"{REPO}/scattering")
from _figsave import save_fig
from joint_ppc import prepare  # same band prep (BurstDataset) as the fit

OUT = f"{RUNS}/data/joint"

# burst -> (fit tag, n CHIME comps, n DSA comps); n=0 => sharedzeta single-comp free fit
SPEC = {
    "freya": ("sharedzeta", 0, 0),
    "casey": ("sharedzeta", 0, 0),
    "chromatica": ("sharedzeta", 0, 0),
    "wilhelm": ("sharedzeta", 0, 0),
    "oran": ("C2D1", 2, 1),
    "phineas": ("C3D3", 3, 3),
    "whitney_fine": ("C2D2", 2, 2),
}
NSUB = {"chime": 4, "dsa": 3}  # sub-bands per band (S/N-limited; DSA lever is short)
BURSTS = sys.argv[1:] or list(SPEC)


def emg(t, A, mu, sigma, tau, b):
    """Single scattered pulse: A * EMG(mu,sigma,tau) + baseline.  tau = K*sigma."""
    K = max(tau, 1e-6) / max(sigma, 1e-6)
    return A * exponnorm.pdf(t, K, loc=mu, scale=max(sigma, 1e-6)) + b


def emg_multi(t, mus, sigmas, amps, tau, b):
    """Sum of EMGs with shared tau; positions/widths fixed (multi-component)."""
    out = np.full_like(t, b, dtype=float)
    for mu, sig, A in zip(mus, sigmas, amps):
        K = max(tau, 1e-6) / max(sig, 1e-6)
        out = out + A * exponnorm.pdf(t, K, loc=mu, scale=max(sig, 1e-6))
    return out


def subband_profiles(model, nsub):
    """Yield (nu_center_GHz, t_ms, profile, dt) for nsub equal frequency slices."""
    freq = np.asarray(model.freq, float)  # GHz, ascending
    t = np.asarray(model.time, float)  # ms
    dt = float(np.median(np.diff(t)))
    edges = np.linspace(freq.min(), freq.max(), nsub + 1)
    for k in range(nsub):
        lo, hi = edges[k], edges[k + 1]
        sel = (freq >= lo) & (freq < hi if k < nsub - 1 else freq <= hi)
        if sel.sum() < 2:
            continue
        prof = np.nansum(model.data[sel], axis=0)
        yield 0.5 * (lo + hi), t, prof, dt


def _r2(prof, model):
    ss_res = float(np.nansum((prof - model) ** 2))
    ss_tot = float(np.nansum((prof - np.nanmean(prof)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_single(t, prof, dt, mu0, sig0, tau0):
    """Free EMG fit; return (tau, sigma_tau, r2)."""
    base0 = float(np.nanmedian(prof))
    flu = max(float(np.nansum(np.clip(prof - base0, 0, None)) * dt), 1e-6)  # fluence = area scale
    win = float(t.max() - t.min())
    mu0 = min(max(mu0, t.min() + 1e-6), t.max() - 1e-6)
    # A multiplies a unit-area EMG, so it scales with fluence (not peak): a broad
    # large-tau pulse has A >> peak. Bounding by peak pinned tau->0 for oran/whitney.
    p0 = [flu, mu0, min(max(sig0, dt), 0.5 * win), min(max(tau0, dt), 0.9 * win), base0]
    lo = [0, t.min(), 0.3 * dt, 1e-4, base0 - flu]
    hi = [100 * flu, t.max(), win, win, base0 + flu]
    try:
        popt, pcov = curve_fit(emg, t, prof, p0=p0, bounds=(lo, hi), maxfev=20000)
        return float(popt[3]), float(np.sqrt(max(pcov[3, 3], 0))), _r2(prof, emg(t, *popt))
    except Exception as e:
        print(f"      single-fit fail: {e}")
        return np.nan, np.nan, 0.0


def fit_multi(t, prof, dt, mus, sigmas, tau0):
    """Shared-tau sum-of-EMG; positions/widths fixed. Fit amps + tau. Return (tau, sigma_tau, r2)."""
    n = len(mus)
    base0 = float(np.nanmedian(prof))
    flu = max(float(np.nansum(np.clip(prof - base0, 0, None)) * dt), 1e-6)  # fluence = area scale
    win = float(t.max() - t.min())

    def f(t_, *p):
        amps, tau, b = p[:n], p[n], p[n + 1]
        return emg_multi(t_, mus, sigmas, amps, tau, b)

    p0 = [flu / n] * n + [min(max(tau0, dt), 0.9 * win), base0]
    lo = [0] * n + [1e-4, base0 - flu]
    hi = [100 * flu] * n + [win, base0 + flu]
    try:
        popt, pcov = curve_fit(f, t, prof, p0=p0, bounds=(lo, hi), maxfev=30000)
        return float(popt[n]), float(np.sqrt(max(pcov[n, n], 0))), _r2(prof, f(t, *popt))
    except Exception as e:
        print(f"      multi-fit fail: {e}")
        return np.nan, np.nan, 0.0


def joint_curve(b, tag):
    d = json.load(open(f"{OUT}/{b}_joint_fit_{tag}_pbf-exp-exp.json"))
    P = {k: v["median"] for k, v in d["percentiles"].items()}
    return d, P


R2_MIN = 0.5  # below this the EMG does not describe the profile -> tau unreliable, drop


def measure_band(model, nsub, band, tag, P, ncomp):
    """Return arrays nu, tau, tau_err, is_det for one band's sub-bands (bad fits dropped)."""
    nu_c, tau_v, tau_e, det = [], [], [], []
    n_bad = 0
    suf = "C" if band == "chime" else "D"
    for nu, t, prof, dt in subband_profiles(model, nsub):
        tau0 = P["tau_1ghz"] * nu ** (-P["alpha"])  # joint expectation (init only)
        if ncomp == 0:  # sharedzeta single component free fit
            mu0 = P.get(f"t0_{suf}", float(t[np.nanargmax(prof)]))
            sig0 = P.get("zeta_1ghz", dt * 3) * nu ** P.get("x_zeta", 0.0)
            tau, te, r2 = fit_single(t, prof, dt, mu0, max(sig0, dt), tau0)
        else:  # multi-component: fix positions/widths from joint, fit shared tau
            mus = [P[f"t0_{suf}{i}"] for i in range(1, ncomp + 1)]
            sigmas = [max(P[f"zeta_{suf}{i}"], dt) for i in range(1, ncomp + 1)]
            tau, te, r2 = fit_multi(t, prof, dt, mus, sigmas, tau0)
        if not np.isfinite(tau) or not np.isfinite(te):
            continue
        if (
            r2 < R2_MIN
        ):  # poor fit -> tau unreliable (e.g. faint/heavily-scattered CHIME); NOT a limit
            n_bad += 1
            print(
                f"    {band} nu={nu:.3f} GHz: UNCONSTRAINED (R2={r2:.2f}<{R2_MIN}, fit fails) (joint~{tau0:.4f})"
            )
            continue
        is_det = (tau > 2 * te) and (tau > 0.5 * dt) and (te < tau)
        nu_c.append(nu)
        tau_v.append(tau)
        tau_e.append(te)
        det.append(bool(is_det))
        flag = "det" if is_det else "lim"
        print(
            f"    {band} nu={nu:.3f} GHz: tau={tau:.4f}+-{te:.4f} ms R2={r2:.2f} [{flag}] (joint~{tau0:.4f})"
        )
    return np.array(nu_c), np.array(tau_v), np.array(tau_e), np.array(det, bool), n_bad


def powerlaw_alpha(nu, tau, te, det):
    """Weighted slope of log tau vs log nu over detections -> alpha_sb, sigma_alpha."""
    if det.sum() < 2:
        return np.nan, np.nan
    x = np.log(nu[det])
    y = np.log(tau[det])
    sig_logtau = np.clip(
        te[det] / np.clip(tau[det], 1e-9, None), 1e-2, None
    )  # sigma_logtau, floored
    w = 1.0 / sig_logtau**2  # inverse-variance weights in log-tau
    # weighted linear fit y = m x + c ; alpha = -m
    W = np.sum(w)
    mx = np.sum(w * x) / W
    sxx = np.sum(w * (x - mx) ** 2)
    sxy = np.sum(w * (x - mx) * (y - np.sum(w * y) / W))
    m = sxy / sxx
    sm = np.sqrt(1.0 / sxx)  # weighted-LS slope error from the tau measurement errors
    return -m, sm


def panel(ax, b, P, a_j, a_sb, a_se, agree, packed, *, compact=False):
    """Draw one burst's tau(nu) overlay (joint power law + sub-band points/limits)."""
    nug = np.linspace(0.40, 1.55, 200)
    ax.plot(nug, P["tau_1ghz"] * nug ** (-a_j), "k-", lw=1.4, label=f"joint $\\alpha$={a_j:.2f}")
    for nn, tt, ee, dd, c, lab in packed:
        if len(nn) == 0:
            continue
        m = dd
        if m.any():
            ax.errorbar(nn[m], tt[m], yerr=ee[m], fmt="o", color=c, ms=5, capsize=2, label=lab)
        if (~m).any():
            ul = np.clip(tt[~m], 0, None) + 2 * ee[~m]
            ax.errorbar(
                nn[~m],
                ul,
                yerr=0.25 * ul,
                uplims=True,
                fmt="v",
                color=c,
                ms=5,
                alpha=0.6,
                label=f"{lab} (2$\\sigma$ u.l.)",
            )
    ax.axvspan(0.40, 0.80, color="gray", alpha=0.07)
    ax.axvspan(1.28, 1.53, color="gray", alpha=0.10)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    if np.isfinite(a_sb):
        ttl = f"{b}: joint $\\alpha$={a_j:.2f} vs sub-band {a_sb:.2f}$\\pm${a_se:.2f} ({agree})"
    else:
        ttl = f"{b}: joint $\\alpha$={a_j:.2f} (sub-band unconstrained)"
    ax.set_title(ttl, fontsize=8 if compact else 10)


# manuscript grid: 3 confirmations (top) + 2 mild tensions + the lever-limited oran (bottom)
GRID_BURSTS = ["casey", "chromatica", "wilhelm", "freya", "phineas", "oran"]
TNS = {  # publication names (burst_metadata._FALLBACK_TNS) for the manuscript grid
    "casey": "FRB 20240229A",
    "chromatica": "FRB 20240203A",
    "wilhelm": "FRB 20221203A",
    "freya": "FRB 20230325A",
    "phineas": "FRB 20230307A",
    "oran": "FRB 20220506D",
    "whitney_fine": "FRB 20220310F",
}


def main():
    summary = []
    grid_data = {}
    for b in BURSTS:
        tag, nC, nD = SPEC[b]
        print(f"== {b} ({tag}) ==")
        try:
            mC = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", OUT)
            mD = prepare(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa", OUT)
            d, P = joint_curve(b, tag)
        except Exception as e:
            print(f"  prep/load fail: {e}")
            continue
        nuC, tC, teC, dC, badC = measure_band(mC, NSUB["chime"], "chime", tag, P, nC)
        nuD, tD, teD, dD, badD = measure_band(mD, NSUB["dsa"], "dsa", tag, P, nD)
        n_bad = badC + badD
        nu = np.concatenate([nuC, nuD])
        tau = np.concatenate([tC, tD])
        te = np.concatenate([teC, teD])
        det = np.concatenate([dC, dD])
        a_sb, a_se = powerlaw_alpha(nu, tau, te, det)
        a_j = P["alpha"]
        agree = (
            "n/a"
            if not np.isfinite(a_sb) or not np.isfinite(a_se) or a_se == 0
            else f"{abs(a_sb - a_j) / a_se:.1f} sigma"
        )
        summary.append((b, a_j, a_sb, a_se, int(det.sum()), int((~det).sum()), n_bad, agree))
        packed = [
            (nuC, tC, teC, dC, "C0", "CHIME sub-band"),
            (nuD, tD, teD, dD, "C3", "DSA sub-band"),
        ]
        grid_data[b] = (P, a_j, a_sb, a_se, agree, packed)
        # ---- per-burst tau(nu) overlay ----
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        panel(ax, b, P, a_j, a_sb, a_se, agree, packed)
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel(r"$\tau$ (ms)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = save_fig(fig, f"{OUT}/{b}_subband_tau", dpi=120)
        plt.close(fig)
        print(f"  wrote {fp}  joint_a={a_j:.2f} subband_a={a_sb:.2f}+-{a_se:.2f} {agree}")
    # ---- manuscript grid ----
    gb = [b for b in GRID_BURSTS if b in grid_data]
    if gb:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), squeeze=False)
        flat = axes.ravel()
        for ax, b in zip(flat, gb):
            panel(ax, TNS.get(b, b), *grid_data[b], compact=True)
        for ax in flat[len(gb) :]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\tau$ (ms)")
        for ax in axes[-1, :]:
            ax.set_xlabel("frequency (GHz)")
        flat[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(
            r"Independent sub-band $\tau(\nu)$ vs joint-fit power law $\tau_{1\rm GHz}\,\nu^{-\alpha}$",
            fontsize=13,
        )
        fig.tight_layout()
        gp = save_fig(fig, f"{OUT}/subband_tau_validation_grid", dpi=130)
        plt.close(fig)
        print(f"  wrote grid {gp}")
    print("\n=== SUMMARY: joint alpha vs independent sub-band alpha ===")
    print(
        f"{'burst':12} {'a_joint':>8} {'a_subband':>16} {'ndet':>5} {'nlim':>5} {'nbad':>5}  agreement"
    )
    for b, aj, asb, ase, nd, nl, nb, ag in summary:
        sbs = f"{asb:.2f}+-{ase:.2f}" if np.isfinite(asb) else "unconstrained"
        print(f"{b:12} {aj:8.2f} {sbs:>16} {nd:5d} {nl:5d} {nb:5d}  {ag}")
    json.dump(
        [
            {
                "burst": b,
                "alpha_joint": aj,
                "alpha_subband": (None if not np.isfinite(asb) else asb),
                "alpha_subband_err": (None if not np.isfinite(ase) else ase),
                "n_det": nd,
                "n_lim": nl,
                "n_unconstrained": nb,
            }
            for b, aj, asb, ase, nd, nl, nb, _ in summary
        ],
        open(f"{OUT}/subband_tau_validation.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
