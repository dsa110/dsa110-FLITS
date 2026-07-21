"""P2 sweep: does window width control the DM-phase peak sharpness?

For zach (DM262, band-limited, low S/N) and freya (DM912, best-case 'real'):
  proper coherent_dedisp -> find burst -> for several half-windows W, run DM-phase on a
  PHYSICAL residual grid and report flat_ratio (peak sharpness) + recovered residual DM.
Window must be wide enough to contain the residual-DM sweep across the burst's band:
  delay(ms) = K_DM*1e3 * dDM * (1/f_lo^2 - 1/f_hi^2). Report that scale per burst.
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import K_DM, DMPhaseEstimator
from scipy.signal import savgol_filter

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
CASES = [("zach", "210456524", 262.368), ("freya", "278720455", 912.4)]


def run(name, cid, dm_c):
    bb = BBData.from_file(
        f"/data/Faber2026/data/chime-frb/{name.lower()}/singlebeam_{cid}.h5"
    )
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)
    bbdd = coherent_dedisp(bb, dm_c)
    I = np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2
    csd = np.nanstd(I, axis=1)
    med = np.nanmedian(csd[csd > 0])
    chan_ok = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    collapse = np.nansum(np.nan_to_num(I[chan_ok]), 0)
    pk = int(np.argmax(savgol_filter(collapse, 51, 3)))
    # band where signal actually lives (channels whose peak-region power is high)
    chs = np.where(chan_ok)[0]
    flo, fhi = freq[chs].min(), freq[chs].max()
    delay_per_dm_ms = K_DM * 1e3 * (1.0 / flo**2 - 1.0 / fhi**2)  # ms per 1 DM unit across band
    print(
        f"\n[{name}] dt={dt * 1e6:.2f}us burst@{pk} band[{flo:.0f},{fhi:.0f}]MHz  "
        f"delay/DM={delay_per_dm_ms:.2f}ms  (window must exceed ~grid_max*this)"
    )
    good = chs[::3]
    frg = freq[good]
    # time-downsample to ~ms resolution: DM-phase wants ms structure, 2.56us is ~1000x oversampled.
    # Window expressed in MILLISECONDS (physical), converted to samples per burst.
    grid = np.arange(-1.5, 1.5, 0.03)  # narrow: data already dedispersed at DM_c, residual is small
    for win_ms, tds in ((4.0, 8), (12.0, 16), (30.0, 32)):
        Wd = int(win_ms / (dt * 1e3))  # half-window in raw samples
        lo, hi = max(pk - Wd, 0), min(pk + Wd, I.shape[1])
        seg = np.nan_to_num(I[good][:, lo:hi])
        nt = (seg.shape[1] // tds) * tds
        segd = seg[:, :nt].reshape(seg.shape[0], nt // tds, tds).mean(2)  # downsample in time
        dt_ds = dt * tds
        est = DMPhaseEstimator(segd.T, frg, dt_ds, grid, ref="top", n_boot=40, random_state=0)
        c = est.result()["dm_curve"]
        i = int(np.argmax(c))
        flat = c.max() / c.min()
        print(
            f"   win=±{win_ms:5.1f}ms tds={tds:3d} (dt_ds={dt_ds * 1e3:.2f}ms, nt={nt // tds:4d})  "
            f"flat_ratio={flat:5.2f}  dm_resid={grid[i]:+.2f}  dm={dm_c + grid[i]:.2f}  "
            f"interior={0 < i < len(grid) - 1}"
        )


for c in CASES:
    run(*c)
