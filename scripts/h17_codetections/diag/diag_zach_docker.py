"""Docker diagnostic (zach): is the burst clean once coherent dedispersion removes
intra-channel smear? Mirrors the validated TOA-extraction dedispersion path.
Run: bin/baseband_analysis_python.sh scripts/diag_zach_docker.py
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
F = "/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5"
OUT = ROOT + "/diagnostics/diag_zach_docker.png"
DM_C = 262.368

bb = BBData.from_file(F)
dt = float(bb.attrs["delta_time"])
freq = np.asarray(bb.index_map["freq"]["centre"], float)
print("freq", round(freq[0], 1), "->", round(freq[-1], 1), "dt", dt, flush=True)

# (a) coherent dedisp only (intra-channel removed, inter-channel sweep kept)
bb_coh = coherent_dedisp(bb, DM_C, time_shift=False)
bb["tiedbeam_baseband"][:] = bb_coh
I_coh = np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2  # (nfreq, ntime)

# (b) + incoherent dedisp (fully aligned, as TOA script)
bb_inc, _f, _fid = incoherent_dedisp(bb, DM_C, fill_wfall=True)
I_inc = np.abs(bb_inc[:, 0, :]) ** 2 + np.abs(bb_inc[:, 1, :]) ** 2


def norm_and_window(I, half=1536):
    coarse = np.nansum(np.nan_to_num(I), axis=0)
    pk = int(np.nanargmax(coarse))
    off = np.ones(I.shape[1], bool)
    off[max(pk - 3000, 0) : pk + 3000] = False
    mu = np.nanmean(I[:, off], axis=1, keepdims=True)
    sd = np.nanstd(I[:, off], axis=1, keepdims=True) + 1e-9
    In = np.nan_to_num((I - mu) / sd)
    lo, hi = max(pk - half, 0), min(pk + half, I.shape[1])
    return In[:, lo:hi], pk, (np.arange(hi - lo) - (pk - lo)) * dt * 1e3


fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for col, (I, tag) in enumerate(
    [(I_coh, "coherent only (intra removed)"), (I_inc, "coherent+incoherent (aligned)")]
):
    win, pk, t_ms = norm_and_window(I)
    prof = np.nansum(win, axis=0)
    nf = win.shape[0] // 4 * 4
    disp = win[:nf].reshape(nf // 4, 4, win.shape[1]).mean(1)
    ax[0, col].imshow(
        disp,
        aspect="auto",
        origin="lower",
        extent=[t_ms[0], t_ms[-1], freq[-1], freq[0]],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0, col].set(title=f"zach {tag}", ylabel="freq (MHz)")
    ax[0, col].set_xlim(-6, 14)
    snr = prof.max() / prof[: len(prof) // 6].std()
    ax[1, col].plot(t_ms, prof, lw=0.8, color="0.2")
    ax[1, col].set(title=f"profile  snr~{snr:.1f}", xlabel="t (ms)")
    ax[1, col].set_xlim(-6, 14)
    print(f"{tag}: peak@{pk} snr~{snr:.1f}", flush=True)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("wrote", OUT, flush=True)
