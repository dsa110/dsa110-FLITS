#!/usr/bin/env python
"""Recompute per-channel gains at FINER channelization without a refit.

The scattering params (tau, alpha, zeta, t0, delta_dm) are channelization-
independent -- they're physical -- so we take them from the COARSE joint fit and
just re-evaluate the matched-filter gain spectrum g_f = S_dk/S_kk on finer-channel
models. That's a deterministic O(data) calc, no sampler. Saves an npz the
resolution gate reads, so the gate's RESOLVED/UNRESOLVED verdict can be checked at
any channelization in seconds.

  python finer_gains.py <coarse_burst> <fine_cfg_token> [out_token]
  e.g.  python finer_gains.py freya freyafine
"""
import json, os, sys
import numpy as np
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


def main():
    coarse, fine = sys.argv[1], sys.argv[2]
    outtok = sys.argv[3] if len(sys.argv) > 3 else fine
    d = json.load(open(f"{RUNS}/data/joint/{coarse}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    out = {}
    for tel, suf in [("chime", "C"), ("dsa", "D")]:
        m = prep(f"{RUNS}/configs/{fine}_{tel}_run.yaml", f"{fine}_{tel}")
        pp = FRBParams(c0=1.0, t0=p[f"t0_{suf}"], gamma=0.0, zeta=p[f"zeta_{suf}"],
                       tau_1ghz=tau, alpha=al, delta_dm=p[f"delta_dm_{suf}"])
        out[f"gain_{suf}"] = m.gain_spectrum(pp, "M3")
        out[f"freq_{suf}"] = m.freq
        cw = float(np.median(np.abs(np.diff(m.freq)))) * 1e3
        print(f"{tel}: {m.freq.size} ch @ {cw:.2f} MHz/ch  (scattering from {coarse}: "
              f"alpha={al:.2f} tau={tau:.3f})")
    fp = f"{RUNS}/data/joint/{outtok}_joint_samples.npz"
    np.savez_compressed(fp, **out)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
