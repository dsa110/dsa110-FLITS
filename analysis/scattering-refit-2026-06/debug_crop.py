#!/usr/bin/env python
"""Diagnose whether on-pulse crop applied + t0 frame consistency."""
import json, os, sys
import numpy as np
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

def build(cfg_path, name, crop):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=name, telescope=tel,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=crop, onpulse_pad_factor=0.5)
    return ds

b = sys.argv[1]
d = json.load(open(f"{RUNS}/data/joint/{b}_joint_fit.json"))["percentiles"]
for tel, suf in [("chime","C"), ("dsa","D")]:
    cfg = f"{RUNS}/configs/{b}_{tel}_run.yaml"
    full = build(cfg, f"{b}_{tel}", False)
    crop = build(cfg, f"{b}_{tel}", True)
    t0 = d[f"t0_{suf}"]["median"]
    print(f"{b} {tel}: FULL t=[{full.time.min():.2f},{full.time.max():.2f}]ms n={full.data.shape[1]}"
          f" | CROP t=[{crop.time.min():.2f},{crop.time.max():.2f}]ms n={crop.data.shape[1]}"
          f" | fitted t0_{suf}={t0:.2f}  in_crop={crop.time.min()<=t0<=crop.time.max()}"
          f" in_full={full.time.min()<=t0<=full.time.max()}")
