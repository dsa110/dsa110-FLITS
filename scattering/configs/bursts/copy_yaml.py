#!/usr/bin/env python3
import os
import glob
import yaml

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Directory where your *_dsa.yaml files live:
input_dir  = "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scattering/yaml/bursts/dsa/"
# Where to write the *_chime.yaml files (can be same as input_dir):
output_dir = "/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scattering/yaml/bursts/chime/"

# Mapping from the prefix (before "_dsa") to the new .npy filename
new_npy_map = {
    "casey"     : "casey_chime_I_491_2085_32000b_cntr_bpc.npy",
    "chromatica": "chromatica_chime_I_272_6382_32000b_cntr_bpc.npy",
    "freya"     : "freya_chime_I_912_4067_32000b_cntr_bpc.npy",
    "hamilton"  : "hamilton_chime_I_518_8007_32000b_cntr_bpc.npy",
    "isha"      : "isha_chime_I_411_4359_32000b_cntr_bpc.npy",
    "johndoeII" : "johndoeII_dsa.yaml_chime_I_696_5184_32000b_cntr_bpc.npy",
    "mahi"      : "mahi_chime_I_960_1316_32000b_cntr_bpc.npy",
    "oran"      : "oran_chime_I_397_0153_32000b_cntr_bpc.npy",
    "phineas"   : "phineas_chime_I_610_2894_32000b_cntr_bpc.npy",
    "whitney"   : "whitney_chime_I_462_1891_32000b_cntr_bpc.npy",
    "wilhelm"   : "wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy",
    "zach"      : "zach_chime_I_262_3621_32000b_cntr_bpc.npy",
}
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(output_dir, exist_ok=True)

for src_path in glob.glob(os.path.join(input_dir, "*_dsa.yaml")):
    fname = os.path.basename(src_path)
    prefix = fname.split("_dsa.yaml")[0]
    if prefix not in new_npy_map:
        print(f"⚠️  Skipping {fname}: no entry in new_npy_map")
        continue
    
    print(src_path)
    # load YAML
    with open(src_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # apply edits
    cfg["telescope"] = "chime"
    cfg["f_factor"]   = 64
    cfg["t_factor"]   = 24

    # rebuild path with same parent directory
    parent_dir = os.path.dirname(cfg["path"])
    cfg["path"] = os.path.join(parent_dir, new_npy_map[prefix])

    # write new file
    out_name = f"{prefix}_chime.yaml"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    print(f"✅  Written {out_name}")
   
