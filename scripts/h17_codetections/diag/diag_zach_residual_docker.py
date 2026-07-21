"""Docker: validate the tractable DM-phase recipe on zach.
coherent_dedisp(time_shift=False) [intra removed, 871 ch] -> numpy roll-align at DM_c
(for windowing only) -> tight DM-independent window -> RESIDUAL grid around 0 -> time-flip
-> robust peak (argmax of mean curve; sigma = std of bootstrap-curve argmaxes)."""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import K_DM, DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
F = "/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5"
DM_C = 262.368

bb = BBData.from_file(F)
dt = float(bb.attrs["delta_time"])
freq = np.asarray(bb.index_map["freq"]["centre"], float)  # 871, descending
ref = freq.max()
bb_coh = coherent_dedisp(bb, DM_C, time_shift=False)
I = (
    np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2
)  # (871, ntime), intra-removed, swept
assert I.shape[0] == freq.size, (I.shape, freq.size)

# coarse roll-align at DM_c (integer samples) just to find/center the burst for windowing
shift = np.round((1e-3 * K_DM * (1.0 / freq**2 - 1.0 / ref**2) * DM_C) / dt).astype(int)
Idd = np.stack([np.roll(I[j], -shift[j]) for j in range(I.shape[0])])
pk = int(np.nanargmax(np.nansum(np.nan_to_num(Idd), 0)))
W = 512
lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
fr2 = freq[::2]
win = np.nan_to_num(Idd[::2, lo:hi])  # aligned, compact, 436 ch
print(f"peak@{pk} window={win.shape} freqs={fr2.size}", flush=True)


def robust(wf2d, freqs, tag):
    grid = np.arange(-4.0, 4.0, 0.1)  # residual around 0 (data aligned at DM_c)
    est = DMPhaseEstimator(wf2d.T, freqs, dt, grid, ref="top", n_boot=60, random_state=0)
    curve = est.result()["dm_curve"]
    i = int(np.argmax(curve))
    bs_peaks = grid[np.argmax(est._bs_curves, axis=1)]
    print(
        f"  [{tag}] CHIME_DM={DM_C + grid[i]:.3f} resid={grid[i]:+.2f} "
        f"sigma={np.std(bs_peaks, ddof=1):.3f} interior={0 < i < len(grid) - 1} "
        f"flat={curve.max() / curve.min():.4f}",
        flush=True,
    )


print("forward:", flush=True)
robust(win, fr2, "fwd")
print("time-flip:", flush=True)
robust(win[:, ::-1], fr2, "flip")
