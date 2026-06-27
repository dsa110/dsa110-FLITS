"""Assemble a manuscript montage: per-burst DSA data | model mini-panels.

  python plot_jointmodel_montage.py <fig-dir-with-npz> <out-base>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# manuscript display order (TNS-ish by nickname)
ORDER = [
    "casey",
    "chromatica",
    "freya",
    "hamilton",
    "isha",
    "johndoeII",
    "mahi",
    "oran",
    "phineas",
    "whitney",
    "wilhelm",
    "zach",
]


def _mini(ax_d, ax_m, z):
    d = z["dataD"]
    m = z["modelD"]
    f = z["freqD"]
    t = z["timeD"]
    valid = z["validD"].astype(bool)
    finite = d[np.isfinite(d)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    kw = dict(
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        interpolation="nearest",
    )
    ax_d.imshow(d, **kw)
    ax_m.imshow(m, **kw)
    ax_d.set_ylabel("GHz", fontsize=6)
    for a in (ax_d, ax_m):
        a.tick_params(labelsize=5)
        a.set_xlabel("ms", fontsize=6)


def main():
    fig_dir = Path(sys.argv[1])
    out_base = Path(sys.argv[2])
    npz_dir = fig_dir if list(fig_dir.glob("*_jointmodel*.npz")) else fig_dir.parent / "data" / "joint"
    if not list(npz_dir.glob("*_jointmodel*.npz")):
        npz_dir = Path(
            sys.argv[1].replace("jointmodel_figs", "data/joint")
            if "jointmodel_figs" in sys.argv[1]
            else "/central/scratch/jfaber/flits-runs/data/joint"
        )
    # prefer npz co-located with png dir's sibling joint path
    alt = fig_dir.resolve().parent / "data" / "joint"
    if alt.exists() and list(alt.glob("*_jointmodel*.npz")):
        npz_dir = alt
    elif (Path(fig_dir) / ".." / "data" / "joint").resolve().exists():
        npz_dir = (Path(fig_dir).parent / "data" / "joint").resolve()

    bursts = [b for b in ORDER if list(npz_dir.glob(f"{b}_jointmodel*.npz"))]
    n = len(bursts)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 3.2, nrows * 2.4))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, b in enumerate(bursts):
        row, col = divmod(i, ncols)
        fp = sorted(npz_dir.glob(f"{b}_jointmodel*.npz"))[0]
        z = np.load(fp, allow_pickle=True)
        al = float(z["alpha"])
        cD = float(z["chi2D"])
        ax_d = axes[row, col * 2]
        ax_m = axes[row, col * 2 + 1]
        _mini(ax_d, ax_m, z)
        ax_d.set_title(f"{b}\nDSA data", fontsize=7)
        ax_m.set_title(f"model  α={al:.2f} χ²={cD:.1f}", fontsize=7)
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")
    fig.suptitle("Joint-fit DSA dynamic spectra — data vs recovered model (all co-detections)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "svg", "png"):
        fp = out_base.with_suffix(f".{ext}")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"wrote {fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
