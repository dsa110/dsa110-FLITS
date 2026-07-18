"""Reconstruct the PL-PBF gain-recovered jointmodel + residual for one burst so the
DSA single-bin dipole can be inspected under the power-law-PBF fit (vs the production
EMG). Reuses dump_jointmodel.recover (OLS per-channel gain) but upgrades the prepared
models to FRBModelPLPBF and builds FRBParamsPLPBF with s_i=10**log10_s_i so
model(p, "M3") dispatches the inner-scale kernel instead of the EMG.

  python dump_jointmodel_plpbf.py <burst>     # reads plpbf_<burst>_joint_fit.json
"""
import json
import os
import sys

import numpy as np
import yaml

REPO = os.environ.get("FLITS_REPO", "/home/ubuntu/worktrees/joint-tf-fits")
RUNS = os.environ.get("FLITS_RUNS", "/home/ubuntu/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joint_tf_prep
from dump_jointmodel import recover
from plpbf_loglike import FRBModelPLPBF, FRBParamsPLPBF


def _band_resid(R):
    v = R["valid"]
    data = R["data"][v]
    model = R["model"][v]
    noise = np.asarray(R["noise"]).reshape(-1)[v]
    pd = np.nansum(data, axis=0)
    pm = np.nansum(model, axis=0)
    sig = float(np.sqrt(np.nansum(noise ** 2)))
    return (pd - pm) / sig


def main():
    b = sys.argv[1]
    out = f"{RUNS}/data/joint"
    d = json.load(open(f"{out}/plpbf_{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, beta = p["tau_1ghz"], p["beta"]
    s_i = 10.0 ** p["log10_s_i"]

    cC = f"{RUNS}/configs/{b}_chime_run.yaml"
    cD = f"{RUNS}/configs/{b}_dsa_run.yaml"
    (mC, mkC), (mD, mkD) = joint_tf_prep.prepare_pair(cC, cD, b, out, auto=True)
    mC.dm_init = float(yaml.safe_load(open(cC)).get("dm_init", 0.0))
    mD.dm_init = float(yaml.safe_load(open(cD)).get("dm_init", 0.0))
    mC.__class__ = FRBModelPLPBF
    mD.__class__ = FRBModelPLPBF
    print(f"{b}: CHIME {mkC.caption()} | DSA {mkD.caption()}")
    print(f"  medians: beta={beta:.4f} tau={tau:.4f} log10_s_i={p['log10_s_i']:.3f} (s_i={s_i:.3g} tau)")

    zC = p["zeta_1ghz"] * np.asarray(mC.freq, float) ** p["x_zeta"]
    zD = p["zeta_1ghz"] * np.asarray(mD.freq, float) ** p["x_zeta"]
    psC = [FRBParamsPLPBF(c0=1.0, t0=p["t0_C"], gamma=0.0, zeta=zC, tau_1ghz=tau,
                          beta=beta, delta_dm=p["delta_dm_C"], s_i=s_i)]
    psD = [FRBParamsPLPBF(c0=1.0, t0=p["t0_D"], gamma=0.0, zeta=zD, tau_1ghz=tau,
                          beta=beta, delta_dm=p["delta_dm_D"], s_i=s_i)]

    C, chiC = recover(mC, psC)
    D, chiD = recover(mD, psD)

    fp = f"{out}/plpbf_{b}_jointmodel.npz"
    np.savez_compressed(
        fp,
        dataC=C["data"], modelC=C["model"], freqC=C["freq"], timeC=C["time"],
        noiseC=C["noise"], validC=C["valid"],
        dataD=D["data"], modelD=D["model"], freqD=D["freq"], timeD=D["time"],
        noiseD=D["noise"], validD=D["valid"],
    )

    for X, (R, chi) in (("C", (C, chiC)), ("D", (D, chiD))):
        r = _band_resid(R)
        imax = int(np.nanargmax(np.abs(r)))
        jd = int(np.nanargmax(np.abs(np.diff(r))))
        print(f"  {X}-band chi2/dof={chi:.3f}  max|resid|={np.nanmax(np.abs(r)):.1f}sigma "
              f"at bin {imax} (r={r[imax]:+.1f})")
        print(f"    largest adjacent-bin swing |dresid|={abs(r[jd+1]-r[jd]):.1f} "
              f"at bins {jd}->{jd+1} (r={r[jd]:+.1f} -> {r[jd+1]:+.1f})  [single-bin dipole probe]")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
