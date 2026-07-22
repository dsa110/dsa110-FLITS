#!/usr/bin/env python
"""Dump per-band data + joint-fit recovered model (+axes) to .npz for LOCAL
plotting. Same OLS gain recovery as joint_ppc_multi.py; no matplotlib here so it
runs on HPCC despite the broken matplotlibrc. Plot locally with plot_jointmodel.py.

beta-native (ADR-0006): the fit JSON's sampled beta drives the kernel; the PBF
family follows from beta, so the old pbf/pbf_beta model attributes are gone.
alpha-era JSONs without a beta percentile need the alpha-era version of this
script from git history.

  python dump_jointmodel.py <burst> <fit_suffix>
"""

import json
import os
import sys

import numpy as np
import yaml

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so joint_tf_prep imports
import joint_tf_prep
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.joint_model_grid import build_model_grid_arrays
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


def main():
    b = sys.argv[1]
    suf = sys.argv[2] if len(sys.argv) > 2 else ""
    out = f"{RUNS}/data/joint"
    d = json.load(open(f"{out}/{b}_joint_fit{suf}.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    if "beta" not in p:
        sys.exit(
            f"{b}{suf}: no beta percentile -- alpha-era JSON; use the "
            "alpha-era dump_jointmodel.py from git history"
        )
    tau = p["tau_1ghz"]
    cC = f"{RUNS}/configs/{b}_chime_run.yaml"
    cD = f"{RUNS}/configs/{b}_dsa_run.yaml"
    if joint_tf_prep._env_auto():
        # Same S/N-driven resolution + robust common window the fit used, so the
        # dumped data/model grid matches the fit's grid (no per-band hatch mismatch).
        (mC, mkC), (mD, mkD) = joint_tf_prep.prepare_pair(cC, cD, b, out)
        mC.dm_init = float(yaml.safe_load(open(cC)).get("dm_init", 0.0))
        mD.dm_init = float(yaml.safe_load(open(cD)).get("dm_init", 0.0))
        print(f"{b}: AUTO-TF CHIME {mkC.caption()} | DSA {mkD.caption()}")
    else:
        mC = prepare(cC, f"{b}_chime", out)
        mD = prepare(cD, f"{b}_dsa", out)
    arrays = build_model_grid_arrays(mC, mD, d)
    fp = f"{out}/{b}_jointmodel{suf}.npz"
    np.savez_compressed(fp, **arrays)
    print(
        f"{b}: wrote {fp}  alpha={arrays['alpha']:.3f} tau={tau:.4f} "
        f"residual mean-square C={arrays['residual_mean_squareC']:.2f} "
        f"D={arrays['residual_mean_squareD']:.2f}"
    )


if __name__ == "__main__":
    main()
