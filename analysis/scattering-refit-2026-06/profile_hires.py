#!/usr/bin/env python
"""High-fidelity per-burst profiles: full-width profile panels (big enough to
survive image downscaling) + sub-band overlays, so close doublets (e.g. the
~0.3 ms hamilton CHIME pair) are visible. One tall PNG per burst.
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
GATE = {
    "oran": "DSA chi2=5.2 lag1=+0.80, alpha-rail HIGH", "hamilton": "CHIME chi2=3.6 lag1=+0.61, alpha-rail LOW",
    "chromatica": "DSA chi2=9.1 lag1=+0.88", "isha": "C lag1=+0.77 & D chi2=3.4, tau tiny",
    "mahi": "DSA chi2=3.9 lag1=+0.67", "phineas": "DSA lag1=+0.81 chi2=2.1",
    "whitney": "CHIME chi2=2.9 lag1=+0.89", "zach": "CHIME chi2=2.3 lag1=+0.82",
    "freya": "CONTROL clean PASS", "johndoeII": "CONTROL clean residual, rail LOW",
}


def model_for(burst, tel):
    cfg = yaml.safe_load(open(f"{RUNS}/configs/{burst}_{tel}_run.yaml"))
    telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(cfg["path"], RUNS, name=f"{burst}_{tel}_hi", telescope=telb,
                      f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                      outer_trim=float(cfg.get("outer_trim", 0.15)),
                      onpulse_crop=True, onpulse_pad_factor=0.5)
    return ds.model


outdir = f"{RUNS}/figs/profiles_hi"
os.makedirs(outdir, exist_ok=True)
for b in BURSTS:
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.55, 1.0, 0.55, 1.0], hspace=0.35)
    for i, tel in enumerate(("chime", "dsa")):
        try:
            m = model_for(b, tel)
        except Exception as e:
            axw = fig.add_subplot(gs[2 * i]); axw.text(0.5, 0.5, f"{tel}: {e}", ha="center")
            continue
        d = np.asarray(m.data); t = np.asarray(m.time)
        prof = d.sum(0) - np.median(d.sum(0))
        pk = int(np.argmax(prof)); span = t[-1] - t[0]
        lo, hi = t[pk] - 0.30 * span, t[pk] + 0.55 * span
        sel = (t >= lo) & (t <= hi); ti = np.where(sel)[0]
        axw = fig.add_subplot(gs[2 * i])
        axw.imshow(d[:, ti], aspect="auto", origin="lower",
                   vmin=np.nanpercentile(d, 5), vmax=np.nanpercentile(d, 99),
                   extent=[t[ti[0]], t[ti[-1]], m.freq[0], m.freq[-1]], cmap="viridis")
        axw.set_title(f"{tel.upper()} waterfall  {d.shape[0]}ch x {d.shape[1]}t", fontsize=10)
        axw.set_ylabel("GHz", fontsize=8)
        axp = fig.add_subplot(gs[2 * i + 1])
        axp.plot(t[sel], prof[sel], "-o", ms=4, lw=1.6, color="k", label="band-integrated")
        axp.axhline(0, color="0.7", lw=0.5)
        axp.axvline(t[pk], color="tab:red", lw=0.7, ls=":")
        nf = d.shape[0]
        for k, (a, bb, c) in enumerate([(0, nf // 3, "tab:blue"), (nf // 3, 2 * nf // 3, "tab:green"),
                                        (2 * nf // 3, nf, "tab:red")]):
            sp = d[a:bb].sum(0); sp = sp - np.median(sp)
            axp.plot(t[sel], sp[sel] / (np.max(sp) + 1e-9) * np.max(prof[sel]),
                     lw=1.0, color=c, alpha=0.65, label=f"sb{k+1} {m.freq[a]:.2f}-{m.freq[min(bb,nf-1)]:.2f}")
        axp.set_title(f"{tel.upper()} profile (zoom) — single peak or multiple?", fontsize=10)
        axp.set_xlabel("time (ms)"); axp.set_ylabel("flux-med"); axp.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{b}: on-pulse cropped (fit view)   [{GATE.get(b,'')}]", fontsize=13, y=0.995)
    fp = f"{outdir}/{b}_hi.png"; fig.savefig(fp, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", fp, flush=True)
print("DONE")
