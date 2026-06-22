#!/usr/bin/env python
"""Adversarial verification of the zach C2 (CHIME 2-component) joint fit.

The matched grid says adding a 2nd CHIME component buys +3612 nats and drops
alpha 3.32 -> 2.76. That path was never exercised before, so before trusting it:

  1. INTERNAL CONTROL: reconstruct the SINGLE-component (C1) CHIME residual and
     check its band-integrated lag-1 autocorr ~ the +0.82 the gate flagged. If
     my reconstruction reproduces the known flag, the machinery is validated and
     the C2 number below is trustworthy. If not, neither is.
  2. Reconstruct the C2 CHIME residual lag-1 -- does the 2nd component WHITEN it?
  3. Pathology guards: per-channel M conditioning (merge/degenerate?), and the
     fitted t0 separation vs dt_min (is the 2nd comp pinned at the floor?).
  4. Overlay PNG (data vs 1- and 2-comp models + residuals) for the eye.

Reconstruction = the per-channel GLS gain MAP from burstfit_joint's own formula:
  g_f = (M_f + (sigma^2/s2) I)^-1 b_f,  dhat = sum_i g_{f,i} K_{i,f,t}.
s2 is taken from the likelihood's own profiled value (diag["s2"]).
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from dataclasses import replace

import matplotlib.pyplot as plt

sys.path.insert(0, "/central/scratch/jfaber/flits-runs")
REPO = "/home/jfaber/flits/dsa110-FLITS"
sys.path.insert(0, f"{REPO}/scattering")
from run_joint_fit import prepare
from scat_analysis.burstfit import FRBParams
from scat_analysis.burstfit_joint import _gain_marginal_multi_band

RUNS = "/central/scratch/jfaber/flits-runs"
JD = f"{RUNS}/data/joint"
BURST = "zach"


def med(pc, n):
    return pc[n]["median"]


def band_params(pc, tau, alpha, band, ncomp):
    ddm = med(pc, f"delta_dm_{band}")
    out = []
    for i in range(1, ncomp + 1):
        out.append(
            FRBParams(
                c0=1.0,
                t0=med(pc, f"t0_{band}{i}"),
                gamma=0.0,
                zeta=med(pc, f"zeta_{band}{i}"),
                tau_1ghz=tau,
                alpha=alpha,
                delta_dm=ddm,
            )
        )
    return out


def reconstruct(model, params_list):
    """Per-channel GLS gain MAP -> predicted waterfall + band-integrated residual."""
    valid = model.valid
    Ks = np.stack(
        [model(replace(p, c0=1.0, gamma=0.0), "M3", freq_subset=valid) for p in params_list]
    )  # (N,F,T)
    N, F, T = Ks.shape
    d = model.data[valid]
    sig = np.clip(model.noise_std[valid], 1e-9, None)
    var = sig**2
    # profile s2 via the likelihood's own routine (matches the fit)
    _, diag = _gain_marginal_multi_band(model, params_list, ["M3"] * N)
    s2 = float(diag.get("s2") or 1.0)
    b = np.einsum("nft,ft->fn", Ks, d)
    M = np.einsum("nft,mft->fnm", Ks, Ks)
    dhat = np.zeros_like(d)
    nbad = 0
    for f in range(F):
        A = M[f] + (var[f] / s2) * np.eye(N)
        cond = np.linalg.cond(A)
        if cond > 1e10:
            nbad += 1
        g = np.linalg.solve(A, b[f])
        dhat[f] = g @ Ks[:, f, :]
    r = d - dhat
    prof_d = d.sum(0)
    prof_m = dhat.sum(0)
    prof_r = r.sum(0)
    # whiteness on the band-integrated residual (the gate metric)
    rc = prof_r - prof_r.mean()
    lag1 = float(np.sum(rc[1:] * rc[:-1]) / np.sum(rc**2)) if np.sum(rc**2) > 0 else 0.0
    # reduced chi2 of the full residual map
    chi2 = float(np.sum((r / sig[:, None]) ** 2) / (F * T - len(params_list) * F))
    tarr = np.asarray(model.time)
    return dict(
        prof_d=prof_d,
        prof_m=prof_m,
        prof_r=prof_r,
        t=tarr,
        lag1=lag1,
        chi2=chi2,
        frac_culled=diag.get("frac_culled"),
        s2=s2,
        nbad_cond=nbad,
        F=F,
        T=T,
    )


fits = {
    k: json.load(open(f"{JD}/{BURST}_joint_fit_{k}.json")) for k in ("C1D1", "C1D2", "C2D1", "C2D2")
}

print("=== building models ===", flush=True)
mC, _ = prepare(f"{RUNS}/configs/{BURST}_chime_run.yaml", f"{BURST}_chime_v", JD)
mD, _ = prepare(f"{RUNS}/configs/{BURST}_dsa_run.yaml", f"{BURST}_dsa_v", JD)
dt_C = float(np.median(np.abs(np.diff(np.asarray(mC.time)))))
dt_D = float(np.median(np.abs(np.diff(np.asarray(mD.time)))))
dt_min = max(dt_C, dt_D) * 3.0
print(f"dt_min (auto) = {dt_min:.4f} ms  (CHIME dt={dt_C:.4f}, DSA dt={dt_D:.4f})")

# --- CHIME band: C1 (control) vs C2 (test) ---
print("\n=== CHIME residual whiteness (band-integrated lag-1) ===")
recs = {}
for key, ncomp in (("C1D1", 1), ("C2D1", 2)):
    pc = fits[key]["percentiles"]
    tau, alpha = med(pc, "tau_1ghz"), med(pc, "alpha")
    ps = band_params(pc, tau, alpha, "C", ncomp)
    r = reconstruct(mC, ps)
    recs[key] = (r, ps)
    extra = ""
    if ncomp == 2:
        sep = ps[1].t0 - ps[0].t0
        extra = f" | t0 sep={sep:.3f} ms (dt_min={dt_min:.3f}, x{sep / dt_min:.1f})"
    print(
        f"  {key}: lag1={r['lag1']:+.3f}  chi2red={r['chi2']:.2f}  "
        f"frac_culled={r['frac_culled']}  bad_cond_ch={r['nbad_cond']}/{r['F']}{extra}"
    )

print("\n  INTERNAL CONTROL: C1D1 CHIME lag1 should ~match the gate's flagged +0.82.")
print("  If it does, the +3612-nat C2 result is trustworthy; the test is whether")
print("  C2D1 lag1 drops toward 0 (2nd component whitens the flagged residual).")

# --- DSA band: C1 vs C1D2 vs C2D2 (where does the 2nd DSA comp help?) ---
print("\n=== DSA residual whiteness ===")
for key, ncomp in (("C1D1", 1), ("C1D2", 2), ("C2D2", 2)):
    pc = fits[key]["percentiles"]
    tau, alpha = med(pc, "tau_1ghz"), med(pc, "alpha")
    ps = band_params(pc, tau, alpha, "D", ncomp)
    r = reconstruct(mD, ps)
    recs[f"D_{key}"] = (r, ps)
    extra = ""
    if ncomp == 2:
        sep = ps[1].t0 - ps[0].t0
        extra = f" | t0 sep={sep:.3f} ms (x{sep / dt_min:.1f} dt_min)"
    print(
        f"  {key} (DSA): lag1={r['lag1']:+.3f}  chi2red={r['chi2']:.2f}  "
        f"frac_culled={r['frac_culled']}  bad_cond_ch={r['nbad_cond']}/{r['F']}{extra}"
    )

# --- overlay figure ---
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for col, (band, keys, mm) in enumerate(
    [("CHIME", ["C1D1", "C2D1"], mC), ("DSA", ["C1D1", "C1D2"], mD)]
):
    pref = "" if band == "CHIME" else "D_"
    r1 = recs[(pref + keys[0]) if band == "DSA" else keys[0]][0]
    r2 = recs[(pref + keys[1]) if band == "DSA" else keys[1]][0]
    t = r1["t"]
    ax = axes[0, col]
    ax.plot(t, r1["prof_d"], "-o", ms=3, color="k", lw=1.3, label="data")
    ax.plot(
        t,
        r1["prof_m"],
        lw=1.6,
        color="tab:orange",
        label=f"{keys[0]} (1-comp) lag1={r1['lag1']:+.2f}",
    )
    ax.plot(
        t,
        r2["prof_m"],
        lw=1.6,
        color="tab:green",
        label=f"{keys[1]} (2-comp) lag1={r2['lag1']:+.2f}",
    )
    ax.set_title(f"{band}: band-integrated profile + models")
    ax.legend(fontsize=8)
    ax.set_ylabel("flux")
    ax = axes[1, col]
    ax.axhline(0, color="0.7", lw=0.6)
    ax.plot(t, r1["prof_r"], lw=1.2, color="tab:orange", label=f"{keys[0]} resid")
    ax.plot(t, r2["prof_r"], lw=1.2, color="tab:green", label=f"{keys[1]} resid")
    ax.set_title(f"{band}: residuals (flat = white = good)")
    ax.legend(fontsize=8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("data-model")
fig.suptitle(f"{BURST}: 1- vs 2-component fit verification (CHIME flag test + DSA)", fontsize=13)
fig.tight_layout()
fp = f"{JD}/{BURST}_c2_verify.png"
fig.savefig(fp, dpi=140, bbox_inches="tight")
print(f"\nwrote {fp}")
