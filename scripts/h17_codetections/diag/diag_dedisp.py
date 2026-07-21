"""Root-cause diagnostic: is the CHIME-side dedispersion/alignment broken?

Compares, for one burst:
  (A) baseband_analysis coherent_dedisp default (proper time_shift) -> collapse
  (B) the extraction recipe: coherent_dedisp(time_shift=False) + hand-rolled shift
and prints the physical full-band delay vs the shift the recipe actually applies.
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
from dmphase_standalone import K_DM

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
PATH = "/data/Faber2026/data/chime-frb/zach/singlebeam_210456524.h5"
DM_C = 262.368

bb = BBData.from_file(PATH)
dt = float(bb.attrs["delta_time"])
freq = np.asarray(bb.index_map["freq"]["centre"], float)  # 871, descending
flo, fhi = freq.min(), freq.max()
ntime = bb["tiedbeam_baseband"].shape[-1]

K_PHYS = 4.148808e3  # s * MHz^2 / (pc cm^-3); t[s] = K_PHYS * DM * (f^-2 - fref^-2), f in MHz
full_delay_s = K_PHYS * DM_C * (1.0 / flo**2 - 1.0 / fhi**2)
recipe_delay_s = 1e-3 * K_DM * (1.0 / flo**2 - 1.0 / fhi**2) * DM_C  # what the extraction rolls by

print(f"dt={dt:.3e}s  ntime={ntime}  freq[{flo:.1f},{fhi:.1f}]MHz  K_DM(standalone)={K_DM:g}")
print(
    f"PHYSICAL full-band inter-channel delay @DM={DM_C}: {full_delay_s:.4f}s = {full_delay_s / dt:.1f} samples"
)
print(
    f"RECIPE roll delay (1e-3*K_DM):                    {recipe_delay_s:.6f}s = {recipe_delay_s / dt:.1f} samples"
)
print(f"ratio physical/recipe = {full_delay_s / recipe_delay_s:.1f}")

# (A) proper dedispersion (default time_shift) -- burst should be a clean vertical pulse
bbA = coherent_dedisp(bb, DM_C)
IA = np.abs(bbA[:, 0, :]) ** 2 + np.abs(bbA[:, 1, :]) ** 2
colA = np.nansum(IA, 0)
pkA = int(np.nanargmax(colA))
snrA = (colA[pkA] - np.nanmedian(colA)) / (np.nanstd(colA[: len(colA) // 4]) + 1e-9)
print(
    f"(A) proper dedisp: collapse peak @sample {pkA} (t={pkA * dt * 1e3:.2f}ms)  crude SNR~{snrA:.1f}"
)

# (B) recipe: time_shift=False + hand-rolled shift
bbB = coherent_dedisp(bb, DM_C, time_shift=False)
IB = np.abs(bbB[:, 0, :]) ** 2 + np.abs(bbB[:, 1, :]) ** 2
shift = np.round((1e-3 * K_DM * (1.0 / freq**2 - 1.0 / fhi**2) * DM_C) / dt).astype(int)
IBdd = np.stack([np.roll(IB[j], -s) for j, s in enumerate(shift)])
colB = np.nansum(IBdd, 0)
pkB = int(np.nanargmax(colB))
snrB = (colB[pkB] - np.nanmedian(colB)) / (np.nanstd(colB[: len(colB) // 4]) + 1e-9)
print(
    f"(B) recipe path:   collapse peak @sample {pkB} (t={pkB * dt * 1e3:.2f}ms)  crude SNR~{snrB:.1f}"
)
print(
    f"    recipe shift range: [{shift.min()},{shift.max()}] samples (max should ~= physical {full_delay_s / dt:.0f})"
)

# plot both windows around their peaks
W = 400
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, (I, pk, lab) in zip(ax, [(IA, pkA, "A proper dedisp"), (IBdd, pkB, "B recipe")]):
    lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
    nf = I.shape[0] // 4 * 4
    a.imshow(
        np.nan_to_num(I[:nf, lo:hi]).reshape(nf // 4, 4, hi - lo).mean(1),
        aspect="auto",
        origin="lower",
        extent=[(lo - pk) * dt * 1e3, (hi - pk) * dt * 1e3, flo, fhi],
        cmap="magma",
    )
    a.set(title=lab, xlabel="t (ms)", ylabel="freq MHz")
fig.tight_layout()
fig.savefig(ROOT + "/diagnostics/diag_dedisp_zach.png", dpi=110)
print("wrote diagnostics/diag_dedisp_zach.png")
