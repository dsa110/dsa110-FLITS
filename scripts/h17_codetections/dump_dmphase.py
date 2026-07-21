"""Docker: per-burst DM-phase residual + aligned dynamic spectrum, dumped as small npz.

Canonical path (diag_zach_residual_docker.py): coherent_dedisp(time_shift=False) on the
per-channel-windowed singlebeam baseband, roll to center the burst, DMPhaseEstimator over a
residual grid around DM_c. Dumps a downsampled dynamic spectrum + profile + DM-phase curve
(with bootstrap envelope) for local plotting. No heavy arrays leave h17.
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
OUT = "/tmp/dmphase_dump"
os.makedirs(OUT, exist_ok=True)

TARGETS = {
    "zach": ("210456524", 262.368),
    "chromatica": ("356959136", 272.664),
    "wilhelm": ("253635173", 602.346),
    "oran": ("224263996", 396.882),
    "hamilton": ("318353610", 518.799),
    "johndoeII": ("311723353", 696.506),
    "casey": ("362593221", 491.207),   # batch-1 control (clean)
    "freya": ("278720455", 912.4),      # batch-1 control (clean)
}


def run(name, hid, dm_c):
    bb = BBData.from_file(H5.format(name=nm.lower(), id=hid))
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)  # descending
    ref = freq.max()
    bb_coh = coherent_dedisp(bb, dm_c, time_shift=False)
    I = np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2  # (nch, ntime)
    # residual roll to center burst (small; per-channel windowing already removed the bulk sweep)
    shift = np.round((1e-3 * K_DM * (1.0 / freq**2 - 1.0 / ref**2) * dm_c) / dt).astype(int)
    Idd = np.stack([np.roll(I[j], -shift[j]) for j in range(I.shape[0])])
    prof = np.nansum(np.nan_to_num(Idd), 0)
    pk = int(np.nanargmax(prof))
    W = 400
    lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
    fr = freq[::2]
    win = np.nan_to_num(Idd[::2, lo:hi])  # (nch/2, 2W)

    grid = np.arange(-25.0, 25.0, 0.25)
    est = DMPhaseEstimator(win.T, fr, dt, grid, ref="top", n_boot=20, random_state=0)
    curve = est.dm_curve
    bs = est._bs_curves  # (n_boot, ngrid)
    i = int(np.argmax(curve))
    interior = 0 < i < len(grid) - 1
    bs_peaks = grid[np.argmax(bs, axis=1)]

    # downsample dynamic spectrum for display: ~256 freq x ~400 time
    ff = max(1, win.shape[0] // 256)
    tf = max(1, win.shape[1] // 400)
    nf = (win.shape[0] // ff) * ff
    nt = (win.shape[1] // tf) * tf
    ds = win[:nf, :nt].reshape(nf // ff, ff, nt // tf, tf).mean(axis=(1, 3))
    prof_win = win.sum(0)
    t_ms = (np.arange(win.shape[1]) - (pk - lo)) * dt * 1e3

    np.savez_compressed(
        f"{OUT}/{name}.npz",
        ds=ds.astype(np.float32),
        f_lo=fr.min(), f_hi=fr.max(),
        t0_ms=t_ms[0], t1_ms=t_ms[-1],
        prof=prof_win.astype(np.float32),
        grid=grid, curve=curve, bs_curves=bs.astype(np.float32),
        dm_c=dm_c, resid_best=grid[i], interior=interior,
        sigma=float(np.std(bs_peaks, ddof=1)),
        flat=float(curve.max() / curve.min()),
    )
    print(
        f"{name:11s} DM_c={dm_c:8.3f} resid={grid[i]:+7.2f} DM_meas={dm_c + grid[i]:8.3f} "
        f"interior={str(interior):5s} flat={curve.max() / curve.min():.3f} "
        f"sigma={np.std(bs_peaks, ddof=1):.3f}",
        flush=True,
    )


for nm, (hid, dm) in TARGETS.items():
    try:
        run(nm, hid, dm)
    except Exception as e:
        print(f"{nm:11s} ERROR {type(e).__name__}: {e}", flush=True)
