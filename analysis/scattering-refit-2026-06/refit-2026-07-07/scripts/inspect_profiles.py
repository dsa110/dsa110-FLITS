#!/usr/bin/env python3
"""Peak inspection for multi-component window placement (hamilton, casey, zach, wilhelm).

Builds each band with the SAME preprocessing as the fits (BurstDataset, freya
local-runs pattern) and reports find_peaks candidates on the valid-channel
band-integrated profile, in ms on the fit time axis.
"""
import sys, os
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import find_peaks

RUN_DIR = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("FABER2026_PIPELINE", Path(__file__).resolve().parents[4]))
DATA = Path(
    os.environ.get("FABER2026_BURST_DATA", "/Users/jakobfaber/Data/Faber2026/dsa110/DSA_bursts")
)
OUT = Path(os.environ.get("FABER2026_REFIT_RUNS", RUN_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))

from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

BURSTS = {
    "hamilton": dict(dm=518.799),
    "casey": dict(dm=491.207),
    "zach": dict(dm=262.368),
    "wilhelm": dict(dm=602.346),
}
BAND = {
    "chime": dict(f_factor=16, t_factor=24, dm_init=0.0),
    "dsa": dict(f_factor=384, t_factor=2, dm_init=None),  # catalog
}


def data_path(burst, band):
    hits = sorted(DATA.glob(f"{burst}_{band}_I_*.npy"))
    assert len(hits) == 1, (burst, band, hits)
    return hits[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    telcfg = REPO / "scattering/configs/telescopes.yaml"
    for burst, meta in BURSTS.items():
        for band, bc in BAND.items():
            tel = load_telescope_block(str(telcfg), band)
            ds = BurstDataset(
                str(data_path(burst, band)),
                str(OUT / "prep"),
                name=f"{burst}_{band}",
                telescope=tel,
                f_factor=bc["f_factor"],
                t_factor=bc["t_factor"],
                outer_trim=0.15,
                onpulse_crop=True,
                onpulse_pad_factor=0.5,
            )
            m = ds.model
            m.dm_init = meta["dm"] if bc["dm_init"] is None else bc["dm_init"]
            d = np.asarray(m.data, float)
            t = np.asarray(m.time, float)
            v = np.ones(d.shape[0], bool) if m.valid is None else np.asarray(m.valid, bool).reshape(-1)
            prof = d[v].sum(axis=0)
            # off-pulse noise: outer quartiles of the window
            n = len(prof)
            off = np.r_[prof[: n // 8], prof[-n // 8 :]]
            sig = 1.4826 * np.median(np.abs(off - np.median(off)))
            pk, props = find_peaks(prof, prominence=5 * sig, distance=3)
            order = np.argsort(props["prominences"])[::-1][:6]
            print(f"{burst} {band}: dt={np.median(np.diff(t))*1e3:.1f} us  "
                  f"window=[{t[0]:.2f},{t[-1]:.2f}] ms  npeaks={len(pk)}")
            for i in order:
                print(f"    t0={t[pk[i]]:8.3f} ms  prom={props['prominences'][i]/sig:6.1f} sig  "
                      f"snr_peak={(prof[pk[i]]-np.median(off))/sig:6.1f}")


if __name__ == "__main__":
    main()
