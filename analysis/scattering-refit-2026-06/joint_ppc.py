#!/usr/bin/env python
"""Posterior-predictive per-band check for the joint fits.

Rebuilds each band's FRBModel (same prep as run_joint_fit), evaluates the M3
model at the JOINT best-fit medians, and reports per-band reduced chi2 +
profile overlays. If the joint solution fits BOTH bands well (chi2~1), the
shared tau*nu^-alpha (and its alpha) is a real joint measurement. If one band
is poorly fit, the shallow alpha is a forced compromise between irreconcilable
bands, not a measurement.

  python joint_ppc.py <burst>
"""

import json
import os
import sys

import numpy as np

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scat_analysis.burstfit import FRBParams
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset


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


def band_chi2(model, p, gain=False):
    mod = model(p, "M3")
    if gain:  # apply profiled per-channel gain: model = g_f * K_f
        g = model.gain_spectrum(p, "M3")
        mod = g[:, None] * mod
    noise = np.asarray(model.noise_std, dtype=float)
    if noise.ndim == 1:
        noise = noise[:, None]
    r = (model.data - mod) / np.where(noise > 0, noise, np.inf)
    valid = model.valid
    if valid is not None:
        valid = np.asarray(valid)
        r = r[valid] if valid.ndim == 1 else r[valid]
    r = r[np.isfinite(r)]
    chi2 = float(np.sum(r**2))
    npix = int(r.size)
    dof = max(npix - 7, 1)
    return chi2 / dof, mod


def main():
    b = sys.argv[1]
    out = f"{RUNS}/data/joint"
    cC = f"{RUNS}/configs/{b}_chime_run.yaml"
    cD = f"{RUNS}/configs/{b}_dsa_run.yaml"
    mC = prepare(cC, f"{b}_chime", out)
    mD = prepare(cD, f"{b}_dsa", out)

    d = json.load(open(f"{out}/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    shared = bool(d.get("shared_zeta", False))  # shared zeta(nu) fit is gain-marginal
    gain = bool(d.get("marginalize_gain", False)) or shared
    # shared zeta -> per-band array zeta_1ghz*nu^x_zeta; else the stored per-band scalar
    zC = p["zeta_1ghz"] * np.asarray(mC.freq, float) ** p["x_zeta"] if shared else p["zeta_C"]
    zD = p["zeta_1ghz"] * np.asarray(mD.freq, float) ** p["x_zeta"] if shared else p["zeta_D"]
    if gain:
        pC = FRBParams(
            c0=1.0,
            t0=p["t0_C"],
            gamma=0.0,
            zeta=zC,
            tau_1ghz=tau,
            alpha=al,
            delta_dm=p["delta_dm_C"],
        )
        pD = FRBParams(
            c0=1.0,
            t0=p["t0_D"],
            gamma=0.0,
            zeta=zD,
            tau_1ghz=tau,
            alpha=al,
            delta_dm=p["delta_dm_D"],
        )
    else:
        pC = FRBParams(
            c0=p["c0_C"],
            t0=p["t0_C"],
            gamma=p["gamma_C"],
            zeta=zC,
            tau_1ghz=tau,
            alpha=al,
            delta_dm=p["delta_dm_C"],
        )
        pD = FRBParams(
            c0=p["c0_D"],
            t0=p["t0_D"],
            gamma=p["gamma_D"],
            zeta=zD,
            tau_1ghz=tau,
            alpha=al,
            delta_dm=p["delta_dm_D"],
        )

    chiC, modC = band_chi2(mC, pC, gain=gain)
    chiD, modD = band_chi2(mD, pD, gain=gain)
    print(
        f"{b}: alpha={al:.2f} tau1={tau:.3f} | CHIME chi2/dof={chiC:.2f}  DSA chi2/dof={chiD:.2f}"
    )

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m, mod, nm, ch in [
        (ax[0], mC, modC, f"CHIME chi2={chiC:.2f}", chiC),
        (ax[1], mD, modD, f"DSA chi2={chiD:.2f}", chiD),
    ]:
        prof_d = np.nansum(m.data, axis=0)
        prof_m = np.nansum(mod, axis=0)
        a.plot(m.time, prof_d, "k", lw=0.8, label="data")
        a.plot(m.time, prof_m, "r", lw=1.2, label="joint model")
        a.set_title(f"{b} {nm}")
        a.set_xlabel("time (ms)")
        a.legend(fontsize=8)
    fig.suptitle(f"{b}: joint alpha={al:.2f}, tau_1GHz={tau:.3f} ms")
    fig.tight_layout()
    fp = f"{out}/{b}_joint_ppc.png"
    fig.savefig(fp, dpi=110)
    print(f"  wrote {fp}")
    json.dump(
        {"burst": b, "alpha": al, "tau_1ghz": tau, "chi2_chime": chiC, "chi2_dsa": chiD},
        open(f"{out}/{b}_joint_ppc.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
