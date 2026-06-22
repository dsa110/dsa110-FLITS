#!/usr/bin/env python
"""Generate HPCC run-configs for all 12 CHIME bursts.

Reads each repo config (preserving per-burst f_factor/t_factor/dm_init/etc.),
repoints `path` to the local scratch copy, and applies the corrected sampler
knobs (nested + evidence, outer_trim 0.15, nlive 400 + pool, alpha_fixed 4.0,
nproc 8). Writes <burst>_chime_run.yaml into the configs dir.
"""
import glob, os, yaml

REPO = "/home/jfaber/flits/dsa110-FLITS"
RUNS = "/central/scratch/jfaber/flits-runs"
DATA = f"{RUNS}/data"
CFG = f"{RUNS}/configs"
TEL = f"{REPO}/scattering/configs/telescopes.yaml"
SAMP = f"{REPO}/scattering/configs/sampler.yaml"

KNOBS = dict(
    telcfg_path=TEL, sampcfg_path=SAMP,
    fitting_method="nested", outer_trim=0.15,
    nlive=400, dlogz=0.5, nlive_walks=15,
    alpha_fixed=4.0, nproc=8,
)

os.makedirs(CFG, exist_ok=True)
made = []
for src in sorted(glob.glob(f"{REPO}/scattering/configs/bursts/chime/*_chime.yaml")):
    burst = os.path.basename(src)[:-len("_chime.yaml")]
    cfg = yaml.safe_load(open(src)) or {}
    fname = os.path.basename(cfg["path"])           # the *_32000b_cntr_bpc.npy name
    local = f"{DATA}/{fname}"
    if not os.path.exists(local):
        print(f"SKIP {burst}: missing data {fname}")
        continue
    cfg["path"] = local
    cfg.update(KNOBS)
    out = f"{CFG}/{burst}_chime_run.yaml"
    with open(out, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
    made.append(burst)

print(f"generated {len(made)} configs: {', '.join(sorted(made))}")
