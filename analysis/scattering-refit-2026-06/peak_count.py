#!/usr/bin/env python
"""Deterministic sub-burst count: scipy.signal.find_peaks on the band-integrated
on-pulse profile, prominence threshold in units of robust noise (MAD). Removes
the subjective/rendering element that made visual counts disagree.

A "sub-burst" = a local max with prominence >= PROM_SIGMA * noise AND separated
from neighbours by >= MIN_SEP samples. Prominence (in sigma) is reported per peak
so marginal 2nd components are explicit, not hidden behind a yes/no.
"""
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import yaml

REPO = "/home/jfaber/flits/dsa110-FLITS"
RUNS = "/central/scratch/jfaber/flits-runs"
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

BURSTS = ["oran", "hamilton", "chromatica", "isha", "mahi", "phineas",
          "whitney", "zach", "freya", "johndoeII"]
PROM_SIGMA = 4.0     # peak prominence threshold in noise sigma
MIN_SEP = 2          # min samples between peaks
SMOOTH = 1.0         # light Gaussian smoothing (samples)


def model_for(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}_pk", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    return ds.model


def count(burst, tel):
    m = model_for(burst, tel)
    d = np.asarray(m.data); t = np.asarray(m.time)
    prof = d.sum(0).astype(float)
    prof -= np.median(prof)
    noise = 1.4826 * np.median(np.abs(prof - np.median(prof)))  # robust MAD
    if noise <= 0:
        noise = np.std(prof) or 1.0
    sm = gaussian_filter1d(prof, SMOOTH) if SMOOTH > 0 else prof
    pk, props = find_peaks(sm, prominence=PROM_SIGMA * noise, distance=MIN_SEP)
    proms = props["prominences"] / noise
    # order by prominence desc
    order = np.argsort(proms)[::-1]
    peaks = [(round(float(t[pk[i]]), 3), round(float(proms[i]), 1)) for i in order]
    nsamp = prof.size
    return {"n": int(len(pk)), "peaks_ms_sigma": peaks, "nsamp": int(nsamp),
            "noise": round(float(noise), 4)}


print(f"prominence>= {PROM_SIGMA}sigma, min_sep={MIN_SEP}, smooth={SMOOTH}")
print(f"{'burst':12}{'CHIME N':>9}{'DSA N':>7}   peaks (t_ms, prominence_sigma)")
out = {}
for b in BURSTS:
    rc = count(b, "chime"); rd = count(b, "dsa")
    out[b] = {"chime": rc, "dsa": rd}
    print(f"{b:12}{rc['n']:>9}{rd['n']:>7}   "
          f"C{rc['nsamp']}t:{rc['peaks_ms_sigma']}  D{rd['nsamp']}t:{rd['peaks_ms_sigma']}")
import json
json.dump(out, open(f"{RUNS}/figs/peak_counts.json", "w"), indent=2)
print("wrote peak_counts.json")
