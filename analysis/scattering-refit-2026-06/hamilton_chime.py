#!/usr/bin/env python
"""hamilton CHIME only, maximum clarity — settle single vs double peak."""
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

cfg = yaml.safe_load(open(f"{RUNS}/configs/hamilton_chime_run.yaml"))
telb = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
ds = BurstDataset(cfg["path"], RUNS, name="hamilton_chime_only", telescope=telb,
                  f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
                  outer_trim=float(cfg.get("outer_trim", 0.15)),
                  onpulse_crop=True, onpulse_pad_factor=0.5)
m = ds.model
d = np.asarray(m.data)
t = np.asarray(m.time)
prof = d.sum(0) - np.median(d.sum(0))
pk = int(np.argmax(prof))
span = t[-1] - t[0]
lo, hi = t[pk] - 0.30 * span, t[pk] + 0.50 * span
sel = (t >= lo) & (t <= hi)
ti = np.where(sel)[0]

fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                       gridspec_kw={"height_ratios": [1.1, 1.3, 1.3]})
ax[0].imshow(d[:, ti], aspect="auto", origin="lower",
             vmin=np.nanpercentile(d, 5), vmax=np.nanpercentile(d, 99),
             extent=[t[ti[0]], t[ti[-1]], m.freq[0], m.freq[-1]], cmap="viridis")
ax[0].set_title(f"hamilton CHIME waterfall  {d.shape[0]}ch x {d.shape[1]}t  (chi2=3.6, lag1=+0.61, alpha railed LOW)")
ax[0].set_ylabel("freq (GHz)")

ax[1].plot(t[sel], prof[sel], "-o", ms=4, lw=1.6, color="k")
ax[1].axhline(0, color="0.7", lw=0.5)
ax[1].axvline(t[pk], color="tab:red", lw=0.8, ls=":", label=f"main peak {t[pk]:.2f} ms")
ax[1].set_ylabel("flux - median")
ax[1].set_title("band-integrated profile (markers = actual time samples)")
ax[1].legend(fontsize=9)

nf = d.shape[0]
for k, (a, b, c) in enumerate([(0, nf // 3, "tab:blue"), (nf // 3, 2 * nf // 3, "tab:green"),
                               (2 * nf // 3, nf, "tab:red")]):
    sp = d[a:b].sum(0)
    sp = sp - np.median(sp)
    ax[2].plot(t[sel], sp[sel] / (np.max(sp) + 1e-9), lw=1.3, color=c, alpha=0.8,
               label=f"sub-band {k+1}  {m.freq[a]:.2f}-{m.freq[min(b,nf-1)]:.2f} GHz")
ax[2].axhline(0, color="0.7", lw=0.5)
ax[2].set_ylabel("normalized flux")
ax[2].set_xlabel("time (ms)")
ax[2].set_title("per-sub-band profiles (a frequency-drifting 2nd component shows here)")
ax[2].legend(fontsize=9)

fig.tight_layout()
fig.savefig(f"{RUNS}/figs/profiles/hamilton_CHIME.png", dpi=150)
print("wrote hamilton_CHIME.png")
