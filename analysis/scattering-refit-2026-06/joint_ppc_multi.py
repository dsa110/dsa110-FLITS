#!/usr/bin/env python
"""Matplotlib-free per-band reduced chi2 PPC for the joint fits.

Handles shared_zeta, single-comp (marginalize_gain), and N-component (zeta_C1..)
fits uniformly: per-channel OLS recovery of the component gains (the gain_spectrum
analog generalized to N kernels), summed model, reduced chi2 vs each band noise.
No plotting -> dodges the broken HPCC matplotlibrc. Mirrors joint_ppc.band_chi2
masking + dof (npix-7) so verdicts are comparable to the single-comp PPC.

beta-native (ADR-0006): the fit JSON's sampled beta drives the kernel
(FRBParams is beta-native; the PBF family follows from beta, so the old
pbf/pbf_beta model attributes are gone). alpha-era JSONs without a beta
percentile cannot be re-PPC'd against the beta-native kernel -- use the
alpha-era version of this script from git history for those.

  python joint_ppc_multi.py <burst> <fit_suffix>   e.g. johndoeII _C2D1
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np
import yaml

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.burstfit import FRBParams  # noqa: E402
from scat_analysis.config_utils import load_telescope_block  # noqa: E402
from scat_analysis.pipeline.io import BurstDataset  # noqa: E402


def prepare(cfg_path, name, outdir):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"],
        outdir,
        name=name,
        telescope=tel,
        f_factor=int(cfg["f_factor"]),
        t_factor=int(cfg["t_factor"]),
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=os.environ.get("FLITS_ONPULSE_CROP", "1") == "1",
        onpulse_pad_factor=float(os.environ.get("FLITS_ONPULSE_PAD", "0.5")),
    )
    m = ds.model
    m.dm_init = float(cfg.get("dm_init", 0.0))
    return m


def ols_chi2(model, params_list):
    # per-channel OLS of the data onto the N component kernels -> summed model
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), "M3") for p in params_list])  # (N,F,T)
    d = np.asarray(model.data, float)
    sig = np.clip(np.asarray(model.noise_std, float).reshape(-1), 1e-9, None)  # (F,)
    M = np.einsum("nft,mft->fnm", Ks, Ks)  # (F,N,N)
    b = np.einsum("nft,ft->fn", Ks, d)  # (F,N)
    N = len(params_list)
    jit = 1e-9 * max(float(np.einsum("fnn->f", M).mean()), 1e-30)
    g = np.linalg.solve(M + jit * np.eye(N), b[..., None])[..., 0]  # (F,N)
    mod = np.einsum("fn,nft->ft", g, Ks)  # (F,T)
    r = (d - mod) / sig[:, None]
    valid = model.valid
    if valid is not None:
        v = np.asarray(valid)
        r = r[v] if v.ndim == 1 else r[v]
    r = r[np.isfinite(r)]
    npix = int(r.size)
    return float(np.sum(r**2)) / max(npix - 7, 1), npix


def band_params(p, X, n, tau, beta):
    ddm = float(p.get(f"delta_dm_{X}", 0.0))
    out = []
    for i in range(1, n + 1):
        t0 = p.get(f"t0_{X}{i}", p.get(f"t0_{X}"))
        ze = p.get(f"zeta_{X}{i}", p.get(f"zeta_{X}"))
        out.append(
            FRBParams(
                c0=1.0,
                t0=float(t0),
                gamma=0.0,
                zeta=float(ze),
                tau_1ghz=tau,
                beta=beta,
                delta_dm=ddm,
            )
        )
    return out


def main():
    b = sys.argv[1]
    suf = sys.argv[2] if len(sys.argv) > 2 else ""
    out = f"{RUNS}/data/joint"
    d = json.load(open(f"{out}/{b}_joint_fit{suf}.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    if "beta" not in p:
        sys.exit(
            f"{b}{suf}: no beta percentile -- alpha-era JSON; use the "
            "alpha-era joint_ppc_multi.py from git history"
        )
    tau, beta = p["tau_1ghz"], p["beta"]
    al = d["alpha"]["median"]
    nC, nD = int(d.get("components_C", 1)), int(d.get("components_D", 1))
    mC = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", out)
    mD = prepare(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa", out)
    if d.get("shared_zeta"):
        zC = p["zeta_1ghz"] * np.asarray(mC.freq, float) ** p["x_zeta"]
        zD = p["zeta_1ghz"] * np.asarray(mD.freq, float) ** p["x_zeta"]
        psC = [
            FRBParams(
                c0=1.0,
                t0=p["t0_C"],
                gamma=0.0,
                zeta=zC,
                tau_1ghz=tau,
                beta=beta,
                delta_dm=p["delta_dm_C"],
            )
        ]
        psD = [
            FRBParams(
                c0=1.0,
                t0=p["t0_D"],
                gamma=0.0,
                zeta=zD,
                tau_1ghz=tau,
                beta=beta,
                delta_dm=p["delta_dm_D"],
            )
        ]
    else:
        psC = band_params(p, "C", nC, tau, beta)
        psD = band_params(p, "D", nD, tau, beta)
    chiC, npC = ols_chi2(mC, psC)
    chiD, npD = ols_chi2(mD, psD)
    print(
        f"{b}: beta={beta:.3f} alpha={al:.3f} tau1={tau:.4f} | "
        f"CHIME chi2/dof={chiC:.2f} ({npC}px)  DSA chi2/dof={chiD:.2f} ({npD}px)"
    )
    json.dump(
        {
            "burst": b,
            "beta": beta,
            "alpha": al,
            "tau_1ghz": tau,
            "chi2_chime": chiC,
            "chi2_dsa": chiD,
            "components_C": nC,
            "components_D": nD,
            "suffix": suf,
        },
        open(f"{out}/{b}_joint_ppc_multi{suf}.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
