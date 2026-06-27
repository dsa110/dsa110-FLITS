#!/usr/bin/env python
"""Canonical all-exp joint-PPC montage (ADR-0003).

One row per burst: [CHIME | DSA] freq-collapsed profile, data vs reconstructed
joint model at the posterior medians, with per-band reduced chi2 on the fit's
on-pulse-crop window (crop ON -- run_joint_fit/joint_ppc default; reproduces the
documented chi2). Heterogeneous all-exp inputs, dispatched on fit type:

  sharedzeta -> single component, zeta(nu)=zeta_1ghz*nu^x_zeta, gain via
                gain_spectrum  (joint_ppc.band_chi2, gain=True)
  component  -> N-component scalar zeta_C{i}/zeta_D{j}, gain-marginal ridge solve
                at profiled s2  (model_overlay.solve_band, inlined below)

  python _ppc_montage_allexp.py [b1 b2 ...]
"""

import json
import os
import sys
from dataclasses import replace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/analysis/scattering-refit-2026-06")  # _figsave, joint_ppc
sys.path.insert(0, f"{REPO}/scattering")
from _figsave import save_fig
from joint_ppc import band_chi2, prepare  # fit-consistent prep + sharedzeta chi2
from scat_analysis.burstfit import FRBParams
from scat_analysis.burstfit_joint import _gain_marginal_multi_band

OUT = f"{RUNS}/data/joint"
FIG_OUT = os.environ.get("DSA_FIGS", OUT)
ONPULSE_CROP = os.environ.get("FLITS_ONPULSE_CROP", "1") == "1"
ONPULSE_PAD = float(os.environ.get("FLITS_ONPULSE_PAD", "0.5"))

# canonical all-exp model per burst (ALLEXP_PBF_RUN.md); whitney (C2D2, local) absent on HPCC
SPEC = {
    "freya": ("sharedzeta", "sharedzeta"),
    "casey": ("sharedzeta", "sharedzeta"),
    "chromatica": ("sharedzeta", "sharedzeta"),
    "wilhelm": ("sharedzeta", "sharedzeta"),
    "oran": ("component", "C2D1"),
    "phineas": ("component", "C3D3"),
    "whitney_fine": ("component", "C2D2"),  # fit locally, data+config pushed to HPCC
}
BURSTS = sys.argv[1:] or list(SPEC)


def jpath(b, tag):
    return f"{OUT}/{b}_joint_fit_{tag}_pbf-exp-exp.json"


def medians(b, tag):
    d = json.load(open(jpath(b, tag)))
    return d, {k: v["median"] for k, v in d["percentiles"].items()}


def solve_band(model, ps):
    # ponytail: copied verbatim from model_overlay.solve_band -- that module runs
    # `burst, tag = sys.argv[1:]` at import, so it cannot be imported. Same gain-
    # marginal ridge solve at profiled s2 as the canonical multi-component fit.
    valid = model.valid
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), "M3", freq_subset=valid) for p in ps])
    d = model.data[valid]
    sig = np.clip(model.noise_std[valid], 1e-9, None)
    var = sig**2
    b = np.einsum("nft,ft->fn", Ks, d)
    M = np.einsum("nft,mft->fnm", Ks, Ks)
    _, diag = _gain_marginal_multi_band(model, ps, ["M3"] * len(ps), s2=None)  # profiled s2
    s2 = diag["s2"]
    N = Ks.shape[0]
    A = M + (var / s2)[:, None, None] * np.eye(N)[None]
    g = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (F, N) MAP gains
    pred = np.einsum("fn,nft->ft", g, Ks)
    resid = d - pred
    chi2 = float(np.sum((resid / sig[:, None]) ** 2) / resid.size)
    return model.time, np.nansum(d, 0), np.nansum(pred, 0), chi2


def shared_profiles(b, P, mC, mD):
    tau, al = P["tau_1ghz"], P["alpha"]

    def band(suf, m):
        z = P["zeta_1ghz"] * np.asarray(m.freq, float) ** P["x_zeta"]
        p = FRBParams(
            c0=1.0,
            t0=P[f"t0_{suf}"],
            gamma=0.0,
            zeta=z,
            tau_1ghz=tau,
            alpha=al,
            delta_dm=P[f"delta_dm_{suf}"],
        )
        chi, mod = band_chi2(m, p, gain=True)
        return m.time, np.nansum(m.data, 0), np.nansum(mod, 0), chi

    return band("C", mC), band("D", mD)


def comp_profiles(b, P, tag, mC, mD):
    nC, nD = int(tag[1]), int(tag[3])
    tau, al = P["tau_1ghz"], P["alpha"]

    def ps(prefix, n, ddm):
        return [
            FRBParams(
                c0=1.0,
                t0=P[f"t0_{prefix}{i}"],
                gamma=0.0,
                zeta=P[f"zeta_{prefix}{i}"],
                tau_1ghz=tau,
                alpha=al,
                delta_dm=ddm,
            )
            for i in range(1, n + 1)
        ]

    return (
        solve_band(mC, ps("C", nC, P.get("delta_dm_C", 0.0))),
        solve_band(mD, ps("D", nD, P.get("delta_dm_D", 0.0))),
    )


def main():
    n = len(BURSTS)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n), squeeze=False)
    for i, b in enumerate(BURSTS):
        kind, tag = SPEC[b]
        mC = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", OUT)
        mD = prepare(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa", OUT)
        d, P = medians(b, tag)
        al, tau = P["alpha"], P["tau_1ghz"]
        if kind == "sharedzeta":
            (tC, dC, mCp, chiC), (tD, dD, mDp, chiD) = shared_profiles(b, P, mC, mD)
        else:
            (tC, dC, mCp, chiC), (tD, dD, mDp, chiD) = comp_profiles(b, P, tag, mC, mD)
        print(
            f"{b} ({kind} {tag}): alpha={al:.2f} tau1={tau:.3f} | CHIME {chiC:.2f}  DSA {chiD:.2f}"
        )
        ppc_meta = {
            "burst": b,
            "model_tag": tag,
            "fit_kind": kind,
            "alpha": al,
            "tau_1ghz": tau,
            "chi2_chime": chiC,
            "chi2_dsa": chiD,
            "onpulse_crop": ONPULSE_CROP,
            "onpulse_pad_factor": ONPULSE_PAD,
            "outer_trim": 0.15,
            "pbf": "exp-exp",
        }
        json.dump(
            ppc_meta,
            open(f"{OUT}/{b}_joint_ppc_allexp.json", "w"),
            indent=2,
        )
        fig1, ax1 = plt.subplots(1, 2, figsize=(11, 2.8))
        for ax, t, dp, mp, lab, chi in [
            (ax1[0], tC, dC, mCp, "CHIME", chiC),
            (ax1[1], tD, dD, mDp, "DSA", chiD),
        ]:
            ax.plot(t, dp, "k", lw=0.8, label="data")
            ax.plot(t, mp, "r", lw=1.2, label="joint model")
            ax.set_title(f"{b} {lab}  $\\chi^2$={chi:.2f}", fontsize=9)
            ax.set_xlabel("time (ms)", fontsize=8)
            ax.margins(x=0)
        fig1.suptitle(f"{b}: all-exp joint  $\\alpha$={al:.2f}, $\\tau_{{1\\rm GHz}}$={tau:.3f} ms", fontsize=10)
        fig1.tight_layout()
        out_b = b.replace("_fine", "")
        save_fig(fig1, f"{FIG_OUT}/{out_b}_joint_ppc", dpi=110)
        plt.close(fig1)
        for ax, t, dp, mp, lab, chi in [
            (axes[i][0], tC, dC, mCp, "CHIME", chiC),
            (axes[i][1], tD, dD, mDp, "DSA", chiD),
        ]:
            ax.plot(t, dp, "k", lw=0.8, label="data")
            ax.plot(t, mp, "r", lw=1.2, label="joint model")
            ax.set_title(f"{b} {lab} ({tag})  $\\chi^2$={chi:.2f}", fontsize=9)
            ax.set_xlabel("time (ms)", fontsize=8)
            ax.margins(x=0)
            if i == 0 and lab == "CHIME":
                ax.legend(fontsize=7)
    fig.suptitle(
        r"All-exp joint CHIME$\times$DSA posterior-predictive "
        r"(model at $\tau_{1\rm GHz},\alpha$ medians; $\chi^2$ on fit crop window)",
        fontsize=12,
    )
    fig.tight_layout()
    fp = save_fig(fig, f"{FIG_OUT}/joint_ppc_montage_allexp", dpi=110)
    plt.close(fig)
    print(f"  wrote {fp}")


if __name__ == "__main__":
    main()
