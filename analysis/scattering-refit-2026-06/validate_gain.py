#!/usr/bin/env python
"""Validation gate for the gain-marginalized joint fits.

For every co-detected burst, recompute on the cropped on-pulse window:
  - per-band 2D chi2/dof (now a valid GoF metric -- gain whitens scintillation)
  - per-band temporal whiteness: band-integrated residual lag-1 autocorrelation
    (|lag1| small => the SCATTERING/temporal shape is right; large => the model
    cannot reproduce the pulse shape -> alpha/tau are a forced compromise)
  - alpha prior-rail, zeta sanity
Classify each sightline; only those passing the temporal-whiteness + no-rail +
physical-zeta gate are scattering MEASUREMENTS. Writes a resid map per burst.

  python validate_gain.py [burst ...]   (default: all 12)
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
from scat_analysis.burstfit import FRBParams
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

ALL = (
    "casey chromatica freya hamilton isha johndoeII mahi oran phineas whitney wilhelm zach".split()
)
LAG1_MAX = 0.40  # temporal whiteness ceiling
CHI2_MAX = 2.0  # 2D chi2/dof PASS ceiling (gain-marginalized)
ZETA_MAX = 3.0  # ms; above this zeta is absorbing un-fittable structure
A_LO, A_HI, EDGE = 1.0, 6.0, 0.08


def prep(cfg_path, name):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"],
        RUNS,
        name=name,
        telescope=tel,
        f_factor=int(cfg["f_factor"]),
        t_factor=int(cfg["t_factor"]),
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=True,
        onpulse_pad_factor=0.5,
    )
    m = ds.model
    m.dm_init = float(cfg.get("dm_init", 0.0))
    return m


def band_metrics(m, p, gain):
    K = m(p, "M3")
    mod = (m.gain_spectrum(p, "M3")[:, None] * K) if gain else K
    ns = np.asarray(m.noise_std, float)
    ns = ns[:, None] if ns.ndim == 1 else ns
    resid = (m.data - mod) / np.where(ns > 0, ns, np.inf)
    rv = resid[m.valid] if m.valid is not None else resid
    rv = rv[np.isfinite(rv)]
    chi2 = float(np.sum(rv**2)) / max(rv.size - 7, 1)
    rprof = np.nansum(np.where(np.isfinite(resid), resid, 0.0), axis=0)
    rprof = rprof - rprof.mean()
    denom = np.sum(rprof**2)
    lag1 = float(np.sum(rprof[1:] * rprof[:-1]) / denom) if denom > 0 else 0.0
    return chi2, rv.std(), lag1, mod, resid


def classify(b, P, mC_chi, mD_chi, mC_lag, mD_lag, shared=False):
    al, alo, ahi = P["alpha"]["median"], P["alpha"]["lower"], P["alpha"]["upper"]
    if shared:
        zc = zd = P["zeta_1ghz"]["median"]  # shared zeta(nu): 1-GHz width for the sanity flag
    else:
        zc, zd = P["zeta_C"]["median"], P["zeta_D"]["median"]
    flags = []
    if alo <= A_LO + EDGE:
        flags.append("aRAIL_LO")
    if ahi >= A_HI - EDGE:
        flags.append("aRAIL_HI")
    if max(zc, zd) > ZETA_MAX:
        flags.append(f"zeta_hi({max(zc, zd):.0f})")
    if mC_lag > LAG1_MAX:
        flags.append(f"CHIME_temporal({mC_lag:.2f})")
    if mD_lag > LAG1_MAX:
        flags.append(f"DSA_temporal({mD_lag:.2f})")
    if max(mC_chi, mD_chi) > CHI2_MAX and not any("temporal" in f for f in flags):
        flags.append(f"chi2_hi({max(mC_chi, mD_chi):.1f})")
    verdict = (
        "MEASUREMENT"
        if not flags
        else (
            "EXCLUDE"
            if any(k in " ".join(flags) for k in ("temporal", "RAIL", "zeta_hi"))
            else "MARGINAL"
        )
    )
    return verdict, flags


def main():
    bursts = sys.argv[1:] or ALL
    out = f"{RUNS}/data/joint"
    rows = []
    for b in bursts:
        jf = f"{out}/{b}_joint_fit.json"
        if not os.path.exists(jf):
            rows.append((b, "NO_FIT", [], None))
            continue
        d = json.load(open(jf))
        P = d["percentiles"]
        shared = bool(d.get("shared_zeta", False))  # shared zeta(nu) fit is gain-marginal
        gain = bool(d.get("marginalize_gain", False)) or shared
        p = {k: v["median"] for k, v in P.items()}
        tau, al = p["tau_1ghz"], p["alpha"]
        mC = prep(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime")
        mD = prep(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa")
        # shared zeta -> per-band array zeta_1ghz*nu^x_zeta; else the stored per-band scalar
        zC = p["zeta_1ghz"] * np.asarray(mC.freq, float) ** p["x_zeta"] if shared else p["zeta_C"]
        zD = p["zeta_1ghz"] * np.asarray(mD.freq, float) ** p["x_zeta"] if shared else p["zeta_D"]

        def mkp(s, z):  # c0,gamma absent in gain-marginal/shared fits -> unit kernel
            return FRBParams(
                c0=1.0 if gain else p[f"c0_{s}"],
                t0=p[f"t0_{s}"],
                gamma=0.0 if gain else p[f"gamma_{s}"],
                zeta=z,
                tau_1ghz=tau,
                alpha=al,
                delta_dm=p[f"delta_dm_{s}"],
            )

        cC, sC, lC, modC, rC = band_metrics(mC, mkp("C", zC), gain)
        cD, sD, lD, modD, rD = band_metrics(mD, mkp("D", zD), gain)
        verdict, flags = classify(b, P, cC, cD, lC, lD, shared)
        rows.append(
            (
                b,
                verdict,
                flags,
                dict(
                    al=al,
                    alo=P["alpha"]["lower"],
                    ahi=P["alpha"]["upper"],
                    tau=tau,
                    cC=cC,
                    cD=cD,
                    lC=lC,
                    lD=lD,
                    zC=p["zeta_1ghz"] if shared else p["zeta_C"],
                    zD=p["zeta_1ghz"] if shared else p["zeta_D"],
                    lnZ=d.get("log_evidence", 0),
                ),
            )
        )
        # resid map
        fig, ax = plt.subplots(2, 3, figsize=(13, 7))
        for row, m, mod, resid, nm, ch, lg in [
            (ax[0], mC, modC, rC, "CHIME", cC, lC),
            (ax[1], mD, modD, rD, "DSA", cD, lD),
        ]:
            ext = [m.time.min(), m.time.max(), m.freq.min(), m.freq.max()]
            vmx = np.nanpercentile(np.abs(m.data), 99)
            row[0].imshow(
                m.data, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmx
            )
            row[1].imshow(
                mod, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmx
            )
            row[2].imshow(
                resid, aspect="auto", origin="lower", extent=ext, cmap="coolwarm", vmin=-4, vmax=4
            )
            row[0].set_ylabel(f"{nm}\nf (GHz)")
            row[1].set_title("model")
            row[2].set_title(f"resid chi2={ch:.2f} lag1={lg:.2f}")
        ax[0][0].set_title(f"{b} data")
        fig.suptitle(
            f"{b}: {verdict}  alpha={al:.2f} tau1={tau:.3f}ms  {' '.join(flags)}", fontsize=11
        )
        fig.tight_layout()
        fig.savefig(f"{out}/{b}_resid_map.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(
        f"{'burst':12s} {'verdict':12s} {'alpha[lo,hi]':18s} {'tau1':7s} "
        f"{'chiC':5s} {'chiD':5s} {'lagC':5s} {'lagD':5s} flags"
    )
    print("-" * 105)
    nmeas = 0
    for b, v, flags, m in rows:
        if m is None:
            print(f"{b:12s} {v:12s}")
            continue
        a = f"{m['al']:.2f}[{m['alo']:.2f},{m['ahi']:.2f}]"
        print(
            f"{b:12s} {v:12s} {a:18s} {m['tau']:7.3f} {m['cC']:5.2f} {m['cD']:5.2f} "
            f"{m['lC']:5.2f} {m['lD']:5.2f} {','.join(flags)}"
        )
        if v == "MEASUREMENT":
            nmeas += 1
    print(
        f"\n{nmeas}/{len(bursts)} pass the scattering-measurement gate "
        f"(temporal-white, no alpha rail, physical zeta, chi2<{CHI2_MAX})"
    )


if __name__ == "__main__":
    main()
