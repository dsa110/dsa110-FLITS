"""P2 prototype: corrected CHIME structure-DM extraction on zach.

Rebuild on library dedispersion (no hand-rolled inter-channel roll, no 1e-3 bug):
  coherent_dedisp(bb, dm_c)            # proper, time_shift=True -> clean vertical burst (zach SNR~15)
  -> band-collapse, find burst, tight window
  -> DMPhaseEstimator on a PHYSICAL residual-DM grid around 0 (data already at dm_c)
  -> REQUIRE a sharp interior peak (flat_ratio >> noise floor), else 'no structure-DM'.

Validation target: a clean vertical burst in panel 1 AND a sharply peaked DM-phase curve in panel 3.
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
PATH = "/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5"
DM_C = 262.368

bb = BBData.from_file(PATH)
dt = float(bb.attrs["delta_time"])
freq = np.asarray(bb.index_map["freq"]["centre"], float)  # 871 descending
ref = freq.max()

# (1) proper coherent dedispersion -> intensity
bbdd = coherent_dedisp(bb, DM_C)
I = np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2  # (871, ntime)

# (2) RFI/dead-channel mask from per-channel variance; band-collapse -> find burst
csd = np.nanstd(I, axis=1)
med = np.nanmedian(csd[csd > 0])
chan_ok = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
from scipy.signal import savgol_filter

collapse = np.nansum(np.nan_to_num(I[chan_ok]), 0)
pk = int(np.argmax(savgol_filter(collapse, 51, 3)))
noise = np.nanstd(collapse[: len(collapse) // 4]) + 1e-9
snr = (collapse[pk] - np.nanmedian(collapse)) / noise
print(
    f"dt={dt:.3e}s ntime={I.shape[1]} chan_ok={chan_ok.sum()} burst@{pk} (t={pk * dt * 1e3:.1f}ms) SNR~{snr:.1f}"
)

# (3) tight window around the burst, downsample channels for speed
W = 200  # +/- samples (2.56us -> ~0.5ms window; burst is intra-channel narrow after dedisp)
lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
good = np.where(chan_ok)[0][::3]
frg = freq[good]
win = np.nan_to_num(I[good][:, lo:hi])  # (nch, 2W) RAW

# (4) DM-phase on a PHYSICAL residual grid around 0 (data already dedispersed at DM_C)
grid = np.arange(-3.0, 3.0, 0.05)
est = DMPhaseEstimator(win.T, frg, dt, grid, ref="top", n_boot=60, random_state=0)
curve = est.result()["dm_curve"]
i = int(np.argmax(curve))
bs_peaks = grid[np.argmax(est._bs_curves, axis=1)]
dm_resid = float(grid[i])
dm_chime = DM_C + dm_resid
dm_err = float(np.std(bs_peaks, ddof=1))
flat = float(curve.max() / curve.min())
interior = bool(0 < i < len(grid) - 1)
print(
    f"PHYSICAL grid: dm_resid={dm_resid:+.3f} dm_chime={dm_chime:.3f}±{dm_err:.3f} "
    f"flat_ratio={flat:.2f} interior={interior}"
)
print(f"  -> {'SHARP PEAK (real structure-DM)' if flat > 3.0 else 'FLAT (no trustworthy DM)'}")

# (5) figure: waterfall + profile + DM-phase curve
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
nf = (chan_ok.sum() // 4) * 4
wf_disp = np.nan_to_num(I[chan_ok][:nf, lo:hi])
mu = wf_disp.mean(1, keepdims=True)
sd = wf_disp.std(1, keepdims=True) + 1e-9
ax[0].imshow(
    ((wf_disp - mu) / sd).reshape(nf // 4, 4, hi - lo).mean(1),
    aspect="auto",
    origin="lower",
    extent=[(lo - pk) * dt * 1e3, (hi - pk) * dt * 1e3, freq.min(), freq.max()],
    vmin=-0.5,
    vmax=5,
    cmap="magma",
)
ax[0].set(title=f"zach proper-dedisp ({chan_ok.sum()} ch)", xlabel="t (ms)", ylabel="freq MHz")
ax[1].plot((np.arange(hi - lo) - (pk - lo)) * dt * 1e3, collapse[lo:hi], lw=0.8, color="0.2")
ax[1].set(title=f"profile SNR~{snr:.1f}", xlabel="t (ms)")
ax[2].plot(DM_C + grid, curve, ".-", ms=3)
ax[2].axvline(DM_C, color="k", ls=":", label=f"DM_c={DM_C:g}")
ax[2].axvspan(dm_chime - dm_err, dm_chime + dm_err, color="r", alpha=0.2)
ax[2].axvline(dm_chime, color="r", label=f"DM={dm_chime:.2f}±{dm_err:.2f}")
ax[2].set(title=f"DM-phase flat={flat:.2f} interior={interior}", xlabel="trial DM")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT + "/diagnostics/proto_zach.png", dpi=110)
print("wrote diagnostics/proto_zach.png")
