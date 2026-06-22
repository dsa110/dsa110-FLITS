#!/usr/bin/env python
"""Recompute the fit-quality gate for ALL 12 co-detected bursts -> why 3 pass.

For each burst reconstruct the gain-marginal joint model g_f*K_f(t) per band at
the fitted (tau,alpha,t0,zeta,delta_dm), form the whitened residual, and measure
the two gate metrics that decide MEASUREMENT/MARGINAL/EXCLUDE:
  - 2D chi^2/dof per band (good 0.8-1.5, fail >3 or <0.3)
  - band-integrated residual lag-1 autocorrelation (temporal whiteness;
    >0.4 = multi-component shape misfit -> the scattering model itself is wrong,
    so a scintillation test on top is meaningless)
plus alpha-rail (<1.3 or >5.7 = pinned to the [1,6] prior bound).
"""
import glob, json, os, sys
from dataclasses import replace
import numpy as np
import yaml
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scat_analysis.burstfit import FRBParams


def build(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    m = ds.model
    m.dm_init = float(cfg.get("dm_init", 0.0))
    return m


def band_gate(burst, tel, p, tau, al, s):
    m = build(burst, tel)
    pp = FRBParams(c0=1.0, t0=p[f"t0_{s}"], gamma=0.0, zeta=p[f"zeta_{s}"],
                   tau_1ghz=tau, alpha=al, delta_dm=p[f"delta_dm_{s}"])
    K = m(replace(pp, c0=1.0, gamma=0.0), "M3")
    d = m.data
    S_dk = np.einsum("ij,ij->i", d, K); S_kk = np.einsum("ij,ij->i", K, K)
    g = np.where(S_kk > 1e-30, S_dk / np.where(S_kk > 1e-30, S_kk, 1.0), 0.0)
    sig = m.noise_std if m.noise_std is not None else m._estimate_noise(None)
    sig = np.asarray(sig, float)
    if sig.ndim > 1:
        sig = sig[:, 0]
    resid = (d - g[:, None] * K) / np.clip(sig[:, None], 1e-9, None)
    chi2 = float(np.sum(resid ** 2) / resid.size)
    prof = resid.sum(0); prof = prof - prof.mean()
    lag1 = float(np.sum(prof[:-1] * prof[1:]) / np.sum(prof * prof))
    return chi2, lag1


bursts = [os.path.basename(f).replace("_joint_fit.json", "")
          for f in sorted(glob.glob(f"{RUNS}/data/joint/*_joint_fit.json"))]
print(f"{'burst':12s}{'alpha':>6s}{'tau':>7s}  {'chi2_C/lag1_C':>16s}  {'chi2_D/lag1_D':>16s}  verdict")
for b in bursts:
    d = json.load(open(f"{RUNS}/data/joint/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    met = {}
    for tel, s in [("chime", "C"), ("dsa", "D")]:
        try:
            met[s] = band_gate(b, tel, p, tau, al, s)
        except Exception as e:
            met[s] = (None, None)
    lags = [met[s][1] for s in "CD" if met[s][1] is not None]
    chis = [met[s][0] for s in "CD" if met[s][0] is not None]
    maxlag = max((abs(x) for x in lags), default=9.0)
    badchi = any((c > 3 or c < 0.3) for c in chis)
    if al < 1.3 or al > 5.7:
        v = "EXCLUDE alpha-rail"
    elif maxlag > 0.4 or badchi:
        v = "EXCLUDE temporal"
    elif maxlag > 0.2:
        v = "MARGINAL"
    else:
        v = "MEASUREMENT"
    def fmt(t):
        return f"{t[0]:.2f}/{t[1]:+.2f}" if t[0] is not None else "  --/--"
    print(f"{b:12s}{al:6.2f}{tau:7.3f}  {fmt(met['C']):>16s}  {fmt(met['D']):>16s}  {v}")
