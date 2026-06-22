#!/usr/bin/env python
"""High-res zoomed hamilton profiles to settle single- vs multi-component by eye.

Zooms the time axis to the on-pulse window, larger canvas, and overlays 3
frequency sub-band profiles so a frequency-dependent (drifting) 2nd component
shows up. Single big band-integrated profile can blur two close peaks; the
sub-band overlay + zoom resolves them.
"""
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


def model_for(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}_zoom", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    return ds.model


fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for j, tel in enumerate(("chime", "dsa")):
    m = model_for("hamilton", tel)
    d = np.asarray(m.data)
    t = np.asarray(m.time)
    prof = d.sum(0) - np.median(d.sum(0))
    pk = int(np.argmax(prof))
    # zoom window: +/- 35% of the span around the peak
    span = t[-1] - t[0]
    lo, hi = t[pk] - 0.35 * span, t[pk] + 0.45 * span
    sel = (t >= lo) & (t <= hi)
    # waterfall zoomed
    ti = np.where(sel)[0]
    ax[0][j].imshow(d[:, ti], aspect="auto", origin="lower",
                    vmin=np.nanpercentile(d, 5), vmax=np.nanpercentile(d, 99),
                    extent=[t[ti[0]], t[ti[-1]], m.freq[0], m.freq[-1]], cmap="viridis")
    ax[0][j].set_title(f"hamilton {tel.upper()} waterfall (zoom)  {d.shape[0]}ch x {d.shape[1]}t")
    ax[0][j].set_ylabel("freq (GHz)")
    # band-integrated + 3 sub-band profiles (markers to spot a 2nd peak / drift)
    ax[1][j].plot(t[sel], prof[sel], "-o", ms=3, lw=1.4, color="k", label="band-integrated")
    nf = d.shape[0]
    for k, (a, b, c) in enumerate([(0, nf // 3, "tab:blue"), (nf // 3, 2 * nf // 3, "tab:green"),
                                   (2 * nf // 3, nf, "tab:red")]):
        sp = d[a:b].sum(0)
        sp = sp - np.median(sp)
        sp = sp / (np.max(sp) + 1e-9) * np.max(prof[sel])  # scale to overlay
        ax[1][j].plot(t[sel], sp[sel], lw=0.9, color=c, alpha=0.7,
                      label=f"sub-band {k+1} ({m.freq[a]:.2f}-{m.freq[min(b,nf-1)]:.2f}GHz)")
    ax[1][j].axhline(0, color="0.7", lw=0.5)
    ax[1][j].set_title(f"{tel.upper()} profile (zoom): one peak or two?")
    ax[1][j].set_xlabel("time (ms)")
    ax[1][j].legend(fontsize=7)
fig.suptitle("hamilton on-pulse zoom (CHIME chi2=3.6 lag1=+0.61, alpha railed LOW) — single or multi-component?",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{RUNS}/figs/profiles/hamilton_zoom.png", dpi=140)
print("wrote hamilton_zoom.png")
