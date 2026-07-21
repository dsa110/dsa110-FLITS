"""Docker: pin the DM-phase sign/centering recipe on zach using the clean
coherent-dedispersed (intra-removed, swept) waterfall. Tries both time
orientations + an under-dedispersed control."""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
F = "/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5"
DM_C = 262.368

bb = BBData.from_file(F)
dt = float(bb.attrs["delta_time"])
freq = np.asarray(bb.index_map["freq"]["centre"], float)  # descending
bb_coh = coherent_dedisp(bb, DM_C, time_shift=False)
I = np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2  # (nfreq, ntime), swept
coarse = np.nansum(np.nan_to_num(I), axis=0)
pk = int(np.nanargmax(coarse))
W = 1280
lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
fr2 = freq[::2]
win = np.nan_to_num(I[::2, lo:hi])
print(f"peak@{pk} window={win.shape}", flush=True)


def run(wf2d, freqs, grid, tag):
    est = DMPhaseEstimator(wf2d.T, freqs, dt, grid, ref="top", n_boot=20, random_state=0)
    c = est.result()["dm_curve"]
    b, s = est.get_dm()
    i = int(np.argmax(c))
    print(
        f"  [{tag}] dm_best={b:.3f} sigma={s:.3f} argmax_DM={grid[i]:.3f} "
        f"interior={0 < i < len(grid) - 1} flat={c.max() / c.min():.4f}",
        flush=True,
    )


grid = np.arange(DM_C - 3, DM_C + 3, 0.1)
print("absolute grid @ DM_C, forward:", flush=True)
run(win, fr2, grid, "fwd")
print("absolute grid @ DM_C, time-flip:", flush=True)
run(win[:, ::-1], fr2, grid, "flip")
