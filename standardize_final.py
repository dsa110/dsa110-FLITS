import numpy as np
import os

files_to_flip = [
    "data/chime/hamilton_chime_I_518_8007_32000b_cntr_bpc.npy",
    "data/chime/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy",
    "data/chime/whitney_chime_I_462_1891_32000b_cntr_bpc.npy",
    "data/dsa/johndoeII_dsa_I_696_506_2500b_cntr_bpc.npy",
    "data/dsa/hamilton_dsa_I_518_799_2500b_cntr_bpc.npy",
    "data/dsa/chromatica_dsa_I_272_368_2500b_cntr_bpc.npy",
    "data/dsa/wilhelm_dsa_I_602_346_2500b_cntr_bpc.npy",
    "data/dsa/mahi_dsa_I_960_128_2500b_cntr_bpc.npy"
]

for f in files_to_flip:
    if os.path.exists(f):
        print(f"Flipping {f}...")
        data = np.load(f)
        data_flipped = np.flip(data, axis=0)
        np.save(f, data_flipped)
        print("Done.")
    else:
        print(f"File not found: {f}")
