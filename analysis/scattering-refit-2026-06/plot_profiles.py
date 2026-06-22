#!/usr/bin/env python
"""On-pulse-cropped profiles + waterfalls, exactly as the joint fit sees them.

Any non-white temporal residual the single-component fit leaves MUST originate
within this on-pulse window (structure outside it is not fit). So this view is
the fair test of "is there a visible 2nd component, or is the residual a
tail-shape / DM / drift effect within one pulse?"

Controls: freya (clean PASS) and johndoeII (alpha-railed but residuals CLEAN)
should look single-peaked if the chi2/lag-1 gate truly tracks pulse count.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO = "/home/jfaber/flits/dsa110-FLITS"
RUNS = "/central/scratch/jfaber/flits-runs"
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

BURSTS = ["oran", "hamilton", "chromatica", "isha", "mahi", "phineas",
          "whitney", "zach", "freya", "johndoeII"]
GATE = {  # (band that fails, the lag1/chi2 note) for the suptitle
    "oran": "DSA chi2=5.2 lag1=+0.80 (alpha-rail HIGH)",
    "hamilton": "CHIME chi2=3.6 lag1=+0.61 (alpha-rail LOW)",
    "chromatica": "DSA chi2=9.1 lag1=+0.88 (worst)",
    "isha": "C lag1=+0.77 & D chi2=3.4 (tau tiny)",
    "mahi": "DSA chi2=3.9 lag1=+0.67 (long tail)",
    "phineas": "DSA lag1=+0.81 chi2=2.1",
    "whitney": "CHIME chi2=2.9 lag1=+0.89",
    "zach": "CHIME chi2=2.3 lag1=+0.82",
    "freya": "CONTROL clean PASS (alpha=4.48)",
    "johndoeII": "CONTROL clean residual, alpha-rail LOW",
}


def model_for(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}_prof", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    return ds.model


outdir = f"{RUNS}/figs/profiles"
os.makedirs(outdir, exist_ok=True)
for b in BURSTS:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, tel in enumerate(("chime", "dsa")):
        try:
            m = model_for(b, tel)
        except Exception as e:
            ax[0][j].text(0.5, 0.5, f"{tel}: {type(e).__name__}\n{str(e)[:60]}",
                          ha="center", transform=ax[0][j].transAxes)
            ax[1][j].axis("off")
            continue
        d = np.asarray(m.data)
        t = np.asarray(m.time)
        prof = d.sum(0)
        # robust intensity scaling for the waterfall
        vmax = np.nanpercentile(d, 99.0)
        ax[0][j].imshow(d, aspect="auto", origin="lower", vmin=np.nanpercentile(d, 5),
                        vmax=vmax, extent=[t[0], t[-1], m.freq[0], m.freq[-1]], cmap="viridis")
        ax[0][j].set_title(f"{tel.upper()} waterfall  {d.shape[0]}ch x {d.shape[1]}t")
        ax[0][j].set_ylabel("freq (GHz)")
        # SNR-ish profile: subtract median, scale by off-region std if any
        p = prof - np.median(prof)
        ax[1][j].plot(t, p, lw=1.1, color="tab:blue")
        ax[1][j].axhline(0, color="0.7", lw=0.5)
        # mark the peak + half-max crossings to help the eye spot shoulders
        pk = int(np.argmax(p))
        ax[1][j].axvline(t[pk], color="tab:red", lw=0.6, ls=":")
        ax[1][j].set_title(f"{tel.upper()} band-integrated profile")
        ax[1][j].set_xlabel("time (ms)")
        ax[1][j].set_ylabel("flux - median")
    fig.suptitle(f"{b}: on-pulse cropped (fit view)   [{GATE.get(b,'')}]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = f"{outdir}/{b}_profile.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    print("wrote", fp, flush=True)
print("DONE")
