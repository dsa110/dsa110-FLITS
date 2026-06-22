#!/usr/bin/env python
"""Tightly-windowed per-burst profiles. Previous version zoomed to a fixed
fraction of the FULL on-pulse-cropped span -> for wide DSA windows (e.g. whitney
875t) the burst is a tiny spike and a close 2nd sub-pulse gets compressed into
one. Here the window brackets only the significant emission (first->last sample
above a fraction of the peak), so close doublets are resolved in BOTH bands.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

REPO = "/home/jfaber/flits/dsa110-FLITS"
RUNS = "/central/scratch/jfaber/flits-runs"
sys.path.insert(0, f"{REPO}/scattering")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

BURSTS = ["oran", "hamilton", "chromatica", "isha", "mahi", "phineas",
          "whitney", "zach", "freya", "johndoeII"]


def model_for(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}_tt", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    return ds.model


def onpulse_bounds(p, thresh=0.12, pad_l=0.4, pad_r=0.8):
    """First/last sample above thresh*peak (brackets ALL emission incl. a 2nd
    pulse separated by a sub-threshold dip), then pad (more on the tail side)."""
    pk = np.argmax(p)
    above = np.where(p > thresh * p[pk])[0]
    if above.size == 0:
        return 0, len(p) - 1
    lo, hi = above[0], above[-1]
    w = max(hi - lo, 3)
    a = max(0, int(lo - pad_l * w))
    b = min(len(p) - 1, int(hi + pad_r * w))
    return a, b


outdir = f"{RUNS}/figs/profiles_tight"
os.makedirs(outdir, exist_ok=True)
for b in BURSTS:
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.55, 1.0, 0.55, 1.0], hspace=0.35)
    for i, tel in enumerate(("chime", "dsa")):
        try:
            m = model_for(b, tel)
        except Exception as e:
            fig.add_subplot(gs[2 * i]).text(0.5, 0.5, f"{tel}: {e}", ha="center")
            continue
        d = np.asarray(m.data); t = np.asarray(m.time)
        prof = d.sum(0) - np.median(d.sum(0))
        a, bb = onpulse_bounds(prof)
        ti = np.arange(a, bb + 1)
        nsamp = bb - a + 1
        axw = fig.add_subplot(gs[2 * i])
        axw.imshow(d[:, ti], aspect="auto", origin="lower",
                   vmin=np.nanpercentile(d, 5), vmax=np.nanpercentile(d, 99),
                   extent=[t[a], t[bb], m.freq[0], m.freq[-1]], cmap="viridis")
        axw.set_title(f"{tel.upper()} waterfall  {d.shape[0]}ch, on-pulse window {nsamp}t "
                      f"(of {d.shape[1]})", fontsize=10)
        axw.set_ylabel("GHz", fontsize=8)
        axp = fig.add_subplot(gs[2 * i + 1])
        axp.plot(t[ti], prof[ti], "-o", ms=5, lw=1.8, color="k", label="band-integrated")
        axp.axhline(0, color="0.7", lw=0.5)
        nf = d.shape[0]
        for k, (lo3, hi3, c) in enumerate([(0, nf // 3, "tab:blue"), (nf // 3, 2 * nf // 3, "tab:green"),
                                           (2 * nf // 3, nf, "tab:red")]):
            sp = d[lo3:hi3].sum(0); sp = sp - np.median(sp)
            axp.plot(t[ti], sp[ti] / (np.max(sp) + 1e-9) * np.max(prof[ti]),
                     lw=1.0, color=c, alpha=0.6, label=f"sb{k+1} {m.freq[lo3]:.2f}-{m.freq[min(hi3,nf-1)]:.2f}")
        axp.set_title(f"{tel.upper()} profile (TIGHT on-pulse zoom) — count the peaks", fontsize=10)
        axp.set_xlabel("time (ms)"); axp.set_ylabel("flux-med"); axp.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{b}: tight on-pulse window (off-pulse cropped away)", fontsize=13, y=0.995)
    fp = f"{outdir}/{b}_tight.png"; fig.savefig(fp, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", fp, flush=True)
print("DONE")
