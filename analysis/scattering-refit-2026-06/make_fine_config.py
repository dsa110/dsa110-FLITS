#!/usr/bin/env python
"""Write finer-channel freya configs for the scintillation GP fit.

f_factor only DOWNsamples the fixed-resolution .npy, so a SMALLER f_factor =
finer channels = better scintillation resolution (at the cost of per-channel S/N).
Gate said freya Delta_nu_d ~5.5 MHz (CHIME) / 0.14 MHz (DSA): push CHIME to ~3
MHz/ch (may resolve), DSA to ~2 MHz/ch (will stay unresolved -- 0.14 MHz is
sub-channel no matter what -- but kept for the comparison).

  python make_fine_config.py
"""
import os
import yaml

CFG = os.environ.get("FLITS_CFG", "/central/scratch/jfaber/flits-runs/configs")
# new f_factor per band (was CHIME 64, DSA 384)
NEW = {"chime": 8, "dsa": 64}

for tel, ff in NEW.items():
    src = f"{CFG}/freya_{tel}_run.yaml"
    cfg = yaml.safe_load(open(src))
    old = int(cfg["f_factor"])
    cfg["f_factor"] = ff
    dst = f"{CFG}/freyafine_{tel}_run.yaml"
    yaml.safe_dump(cfg, open(dst, "w"))
    # report resulting channel width
    import_path = cfg["telcfg_path"]
    tb = yaml.safe_load(open(import_path))[tel]
    df_raw = float(tb["df_MHz_raw"]); fmin = tb["f_min_GHz"]; fmax = tb["f_max_GHz"]
    band = (fmax - fmin) * 1e3
    chan = df_raw * ff
    print(f"{tel}: f_factor {old} -> {ff}  chan={chan:.2f} MHz  (~{band/chan:.0f} ch over {band:.0f} MHz)  -> {dst}")
