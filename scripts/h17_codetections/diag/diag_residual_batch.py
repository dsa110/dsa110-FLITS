"""Docker batch residual-DM check on the raw baseband (canonical path).

Generalizes diag_zach_residual_docker.py over all 12 sightlines:
 coherent_dedisp(time_shift=False) -> intra-channel de-chirp only
 -> numpy roll-align at DM_c to center/window the burst
 -> DMPhaseEstimator over a WIDE residual grid (+/-50 pc/cc) around 0
 -> report DM_c + residual and whether the peak is interior (a real detection).
A residual pinned at a grid edge or a flat curve => no clean CHIME DM (expected
for the audit's 4 unconstrained bursts). Validates against casey/freya (known good).
"""
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import K_DM, DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
H5 = "/data/Faber2026/data/chime-frb/{name}/singlebeam_{id}.h5"

TARGETS = {
    "zach": ("210456524", 262.368),
    "chromatica": ("356959136", 272.664),
    "wilhelm": ("253635173", 602.346),
    "oran": ("224263996", 396.882),
    "hamilton": ("318353610", 518.799),
    "johndoeII": ("311723353", 696.506),
    # batch-1 controls (known-good CHIME detections per the audit):
    "casey": ("362593221", 491.207),
    "freya": ("278720455", 912.4),
}


def run(name, hid, dm_c, grid_hw=50.0, grid_step=0.25):
    bb = BBData.from_file(H5.format(name=nm.lower(), id=hid))
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)  # descending
    ref = freq.max()
    bb_coh = coherent_dedisp(bb, dm_c, time_shift=False)
    I = np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2  # (nch, ntime), swept
    shift = np.round((1e-3 * K_DM * (1.0 / freq**2 - 1.0 / ref**2) * dm_c) / dt).astype(int)
    Idd = np.stack([np.roll(I[j], -shift[j]) for j in range(I.shape[0])])
    prof = np.nansum(np.nan_to_num(Idd), 0)
    pk = int(np.nanargmax(prof))
    med = np.median(prof); mad = np.median(np.abs(prof - med)) + 1e-12
    pksnr = (prof[pk] - med) / (1.4826 * mad)
    W = 512
    lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
    fr2 = freq[::2]
    win = np.nan_to_num(Idd[::2, lo:hi])
    grid = np.arange(-grid_hw, grid_hw, grid_step)
    est = DMPhaseEstimator(win.T, fr2, dt, grid, ref="top", n_boot=40, random_state=0)
    curve = est.result()["dm_curve"]
    i = int(np.argmax(curve))
    bs = grid[np.argmax(est._bs_curves, axis=1)]
    interior = 0 < i < len(grid) - 1
    print(
        f"{name:11s} DM_c={dm_c:8.3f} resid={grid[i]:+7.2f} -> DM_meas={dm_c + grid[i]:8.3f} "
        f"sigma={np.std(bs, ddof=1):6.3f} interior={str(interior):5s} "
        f"flat={curve.max() / curve.min():.3f} pksnr={pksnr:6.1f}",
        flush=True,
    )


for nm, (hid, dm) in TARGETS.items():
    try:
        run(nm, hid, dm)
    except Exception as e:
        print(f"{nm:11s} ERROR {type(e).__name__}: {e}", flush=True)
