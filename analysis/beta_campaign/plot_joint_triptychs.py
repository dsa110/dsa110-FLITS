#!/usr/bin/env python
"""Per-burst joint-fit triptychs: data | model | whitened residual | profile,
one row per band (CHIME, DSA), from the campaign {burst}_jointmodel_*.npz
dumps. Whitening matches dump_jointmodel.py: r = (data - model)/noise over
valid channels; chi2 = sum(r^2)/(N-7).
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

RUNS = os.environ.get("FLITS_RUNS", "/Users/jakobfaber/Developer/scratch/flits-local-runs")
JDIR = os.path.join(RUNS, "data/joint")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triptychs")
os.makedirs(OUT, exist_ok=True)

CMAP = plt.get_cmap("magma").copy()
CMAP.set_bad("0.35")
RMAP = plt.get_cmap("RdBu_r").copy()
RMAP.set_bad("0.35")


def band_panels(axes, prof_ax, data, model, freq, time, noise, valid, band, chi2):
    v = valid.astype(bool)
    d = np.where(v[:, None], data, np.nan)
    m = np.where(v[:, None], model, np.nan)
    r = np.where(v[:, None], (data - model) / noise[:, None], np.nan)
    lo, hi = np.nanpercentile(d, [1, 99.5])
    ext = [time[0], time[-1], freq[0], freq[-1]]  # freq already GHz
    kw = dict(aspect="auto", origin="lower", extent=ext, interpolation="nearest")
    axes[0].imshow(d, cmap=CMAP, vmin=lo, vmax=hi, **kw)
    axes[1].imshow(m, cmap=CMAP, vmin=lo, vmax=hi, **kw)
    axes[2].imshow(r, cmap=RMAP, vmin=-4, vmax=4, **kw)
    axes[0].set_ylabel(f"{band}\nGHz", fontsize=8)
    for ax, t in zip(axes, ("data", "model", r"whitened residual ($\pm4\sigma$)"), strict=True):
        ax.set_title(f"{band} {t}", fontsize=8)
        ax.tick_params(labelsize=7)

    # Noise-weighted frequency-summed profile (valid channels only).
    w = np.zeros_like(noise)
    w[v] = 1.0 / noise[v] ** 2
    pw = w.sum()
    pd = np.nansum(np.where(v[:, None], data, 0.0) * w[:, None], axis=0) / pw
    pm = np.nansum(np.where(v[:, None], model, 0.0) * w[:, None], axis=0) / pw
    prof_ax.step(time, pd, where="mid", color="0.2", lw=0.7, label="data")
    prof_ax.plot(time, pm, color="crimson", lw=1.2, label="model")
    prof_ax.set_title(rf"{band} profile   $\chi^2_\nu$={chi2:.2f}", fontsize=8)
    prof_ax.tick_params(labelsize=7)
    prof_ax.legend(fontsize=6, frameon=False)
    prof_ax.margins(x=0)


def main():
    files = sorted(glob.glob(os.path.join(JDIR, "*_jointmodel_*.npz")))
    pdf_path = os.path.join(OUT, "joint_triptychs_all12.pdf")
    pngs = []
    with PdfPages(pdf_path) as pdf:
        for fp in files:
            z = np.load(fp)
            burst = str(z["burst"])
            disp = burst.removesuffix("_fine")
            fig, ax = plt.subplots(2, 4, figsize=(13, 5.6), dpi=140)
            band_panels(
                ax[0, :3],
                ax[0, 3],
                z["dataC"],
                z["modelC"],
                z["freqC"],
                z["timeC"],
                z["noiseC"],
                z["validC"],
                "CHIME",
                float(z["chi2C"]),
            )
            band_panels(
                ax[1, :3],
                ax[1, 3],
                z["dataD"],
                z["modelD"],
                z["freqD"],
                z["timeD"],
                z["noiseD"],
                z["validD"],
                "DSA",
                float(z["chi2D"]),
            )
            for a in ax[1, :]:
                a.set_xlabel("ms", fontsize=8)
            fig.suptitle(
                rf"{disp} -- joint $\beta$-fit: $\beta$={float(z['beta']):.3f}, "
                rf"$\alpha$={float(z['alpha']):.2f}, "
                rf"$\tau_{{1\rm GHz}}$={float(z['tau_1ghz']):.3g} ms",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            png = os.path.join(OUT, f"{burst}_triptych.png")
            fig.savefig(png, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pngs.append(png)
            print("wrote", png, flush=True)
    print("wrote", pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
