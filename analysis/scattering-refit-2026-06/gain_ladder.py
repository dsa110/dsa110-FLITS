#!/usr/bin/env python
"""Recompute a burst's per-channel gain spectrum across a channelization LADDER.

For a multi-screen test we need the gain spectrum at several frequency
resolutions: a broad Galactic scintle (few MHz) only shows at coarse channels,
a finer scale needs finer channels -- but finer channels cost per-channel S/N.
The scattering params are channelization-independent, so we take them from the
coarse joint fit and re-evaluate the matched-filter gain g_f=S_dk/S_kk AND its
variance v_f=sig_f^2/S_kk (-> per-channel S/N) at each f_factor. Deterministic,
no sampler. One npz feeds the local 1/2/3-Lorentzian model-selection.

The ladder is ADAPTIVE per band: [cf, cf/2, cf/4, cf/8, cf/16] where cf is the
band's coarse f_factor from its run yaml, so it scales to each burst's native
resolution.

  python gain_ladder.py <burst>     e.g.  python gain_ladder.py casey
"""
import json, os, sys
from dataclasses import replace
import numpy as np
import yaml
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scat_analysis.burstfit import FRBParams

BURST = sys.argv[1] if len(sys.argv) > 1 else "freya"
SUF = {"chime": "C", "dsa": "D"}


def ladder(tel):
    cf = int(yaml.safe_load(open(f"{RUNS}/configs/{BURST}_{tel}_run.yaml"))["f_factor"])
    ffs = [cf // k for k in (1, 2, 4, 8, 16) if cf // k >= 1]
    return sorted(set(ffs), reverse=True)


def build(tel, ff):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{BURST}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{BURST}_{tel}_ff{ff}", telescope=telb,
                      f_factor=int(ff), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    m = ds.model
    m.dm_init = float(cfg.get("dm_init", 0.0))
    return m


def gain_var(m, p, model="M3"):
    K = m(replace(p, c0=1.0, gamma=0.0), model)
    d = m.data
    S_dk = np.einsum("ij,ij->i", d, K)
    S_kk = np.einsum("ij,ij->i", K, K)
    ok = S_kk > 1e-30
    g = np.where(ok, S_dk / np.where(ok, S_kk, 1.0), np.nan)
    sig = m.noise_std
    if sig is None:
        sig = m._estimate_noise(None)
    sig = np.asarray(sig, dtype=float)
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig = np.clip(sig, 1e-12, None)
    v = np.where(ok, sig ** 2 / np.where(ok, S_kk, 1.0), np.nan)
    return m.freq.copy(), g, v


def main():
    d = json.load(open(f"{RUNS}/data/joint/{BURST}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    out = {"tau_alpha": np.array([tau, al])}
    for tel in ("chime", "dsa"):
        s = SUF[tel]
        for ff in ladder(tel):
            try:
                m = build(tel, ff)
            except Exception as e:
                print(f"{tel} ff{ff}: BUILD FAIL {e}")
                continue
            pp = FRBParams(c0=1.0, t0=p[f"t0_{s}"], gamma=0.0, zeta=p[f"zeta_{s}"],
                           tau_1ghz=tau, alpha=al, delta_dm=p[f"delta_dm_{s}"])
            fr, g, v = gain_var(m, pp)
            cw = float(np.nanmedian(np.abs(np.diff(fr)))) * 1e3
            snr = float(np.nanmedian(g / np.sqrt(v)))
            out[f"freq_{s}_ff{ff}"] = fr
            out[f"gain_{s}_ff{ff}"] = g
            out[f"var_{s}_ff{ff}"] = v
            print(f"{tel:5s} ff{ff:<4d}: {fr.size:4d} ch  {cw:7.3f} MHz/ch  med gain S/N={snr:6.1f}")
    fp = f"{RUNS}/data/joint/{BURST}_gainladder.npz"
    np.savez_compressed(fp, **out)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
