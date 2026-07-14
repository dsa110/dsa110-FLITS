"""Dump per-coarse-channel time0 metadata (fpga_count) from a singlebeam h5.
Validated against freya_time0_metadata.json (2026-07-04 extraction) 2026-07-06."""
import json, sys
import h5py
import numpy as np

h5path = sys.argv[1]
out = sys.argv[2]
with h5py.File(h5path, "r") as f:
    fpga = np.asarray(f["time0"]["fpga_count"])
    freq = np.asarray(f["index_map"]["freq"]["centre"])
    fid = np.asarray(f["index_map"]["freq"]["id"])
meta = {
    "h5": h5path,
    "delta_time": 2.56e-6,
    "fpga_count": [int(x) for x in fpga],
    "freq_mhz": [float(x) for x in freq],
    "freq_id": [int(x) for x in fid],
}
json.dump(meta, open(out, "w"))
print(f"{h5path}: {len(fpga)} channels, span {(max(fpga)-min(fpga))*2.56e-6:.3f} s")
