#!/usr/bin/env python
"""Honest per-channel goodness-of-fit: 2D data/model/residual + whiteness.

For each band of one burst (using the cropped on-pulse window the fit saw):
 - data, model, (data-model)/noise as 2D waterfalls
 - residual pixel histogram vs N(0,1) (white -> fit is good even if chi2>1)
 - chi2/dof, and the off-diagonal time-lag autocorr of the residual profile
   (structure in residuals => coherent misfit, not just bright-burst inflation)

  python resid_map.py <burst>
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scat_analysis.burstfit import FRBParams


def prep(cfg_path, name):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=name, telescope=tel,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    m = ds.model; m.dm_init = float(cfg.get("dm_init", 0.0)); return m


def panel(axrow, m, p, label, gain=True):
    K = m(p, "M3")                      # p has c0=1,gamma=0 -> unit kernel K_f(t)
    if gain:
        g = m.gain_spectrum(p, "M3")
        mod = g[:, None] * K
    else:
        mod = K
    ns = np.asarray(m.noise_std, float)
    ns = ns[:, None] if ns.ndim == 1 else ns
    resid = (m.data - mod) / np.where(ns > 0, ns, np.inf)
    valid = m.valid
    rv = resid[valid] if valid is not None else resid
    rv = rv[np.isfinite(rv)]
    chi2 = np.sum(rv**2) / max(rv.size - 7, 1)
    ext = [m.time.min(), m.time.max(), m.freq.min(), m.freq.max()]
    vmx = np.nanpercentile(np.abs(m.data), 99)
    axrow[0].imshow(m.data, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmx)
    axrow[1].imshow(mod, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmx)
    axrow[2].imshow(resid, aspect="auto", origin="lower", extent=ext, cmap="coolwarm", vmin=-4, vmax=4)
    axrow[0].set_title(f"{label} data"); axrow[1].set_title("model")
    axrow[2].set_title(f"resid/noise  chi2/dof={chi2:.2f}")
    # whiteness: residual band-integrated profile autocorr beyond lag 0
    rprof = np.nansum(np.where(np.isfinite(resid), resid, 0.0), axis=0)
    rprof = rprof / (np.std(rprof) + 1e-9)
    ac = np.correlate(rprof, rprof, "full"); ac = ac[ac.size // 2:] / ac[ac.size // 2]
    lag1 = ac[1] if ac.size > 1 else 0.0
    axrow[3].hist(rv, bins=60, density=True, color="0.6")
    xx = np.linspace(-5, 5, 200)
    axrow[3].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "r")
    axrow[3].set_title(f"resid hist (std={rv.std():.2f}, lag1ac={lag1:.2f})")
    axrow[3].set_xlim(-5, 5)
    return chi2, rv.std(), lag1


def main():
    b = sys.argv[1]
    out = f"{RUNS}/data/joint"
    d = json.load(open(f"{out}/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    gain = bool(d.get("marginalize_gain", False))
    mC = prep(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime")
    mD = prep(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa")
    if gain:  # 8-param fit: no c0/gamma, model = g_f * K_f(t)
        pC = FRBParams(c0=1.0, t0=p["t0_C"], gamma=0.0, zeta=p["zeta_C"], tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_C"])
        pD = FRBParams(c0=1.0, t0=p["t0_D"], gamma=0.0, zeta=p["zeta_D"], tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_D"])
    else:
        pC = FRBParams(c0=p["c0_C"], t0=p["t0_C"], gamma=p["gamma_C"], zeta=p["zeta_C"], tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_C"])
        pD = FRBParams(c0=p["c0_D"], t0=p["t0_D"], gamma=p["gamma_D"], zeta=p["zeta_D"], tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_D"])
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    cC = panel(ax[0], mC, pC, f"{b} CHIME", gain=gain)
    cD = panel(ax[1], mD, pD, f"{b} DSA", gain=gain)
    for a in ax[:, :3].ravel():
        a.set_xlabel("t (ms)"); a.set_ylabel("f (GHz)")
    fig.suptitle(f"{b}: joint alpha={al:.2f} tau1GHz={tau:.3f}ms  | "
                 f"CHIME chi2={cC[0]:.2f} lag1={cC[2]:.2f} | DSA chi2={cD[0]:.2f} lag1={cD[2]:.2f}", fontsize=11)
    fig.tight_layout()
    fp = f"{out}/{b}_resid_map.png"; fig.savefig(fp, dpi=120, bbox_inches="tight")
    print(f"{b}: CHIME chi2={cC[0]:.2f} std={cC[1]:.2f} lag1ac={cC[2]:.2f} | "
          f"DSA chi2={cD[0]:.2f} std={cD[1]:.2f} lag1ac={cD[2]:.2f}  -> {fp}")


if __name__ == "__main__":
    main()
