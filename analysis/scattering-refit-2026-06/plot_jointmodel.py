"""Plot joint-fit recovered model vs original data, DSA (top) + CHIME (bottom).

Reads the .npz dumped by dump_jointmodel.py (per-band data, recovered model,
axes, noise, valid mask) and makes one PNG per burst: each band gets a row of
[data waterfall | model waterfall | residual (data-model)/noise | freq-summed
profile data-vs-model]. Models come straight from the joint fit; this is the
visual fit-quality check.

  python plot_jointmodel.py <npz-dir> <out-dir> [burst ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _wf(ax, t, f, img, title, vmin, vmax, cmap="magma"):
    ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8)


def _band_row(axrow, z, band, color):
    d = z[f"data{band}"]
    m = z[f"model{band}"]
    f = z[f"freq{band}"]
    t = z[f"time{band}"]
    sig = z[f"noise{band}"]
    valid = z[f"valid{band}"].astype(bool)
    # shared robust color scale for data & model
    finite = d[np.isfinite(d)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    _wf(axrow[0], t, f, d, f"{band} data", vmin, vmax)
    _wf(axrow[1], t, f, m, f"{band} model", vmin, vmax)
    resid = (d - m) / sig[:, None]
    rr = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 99) if np.isfinite(resid).any() else 1
    _wf(axrow[2], t, f, resid, f"{band} resid/σ", -rr, rr, cmap="coolwarm")
    # freq-summed profile over valid channels
    pd = np.nansum(d[valid], axis=0)
    pm = np.nansum(m[valid], axis=0)
    axrow[3].plot(t, pd, "k", lw=0.8, label="data")
    axrow[3].plot(t, pm, color=color, lw=1.3, label="model")
    axrow[3].set_title(f"{band} profile", fontsize=8)
    axrow[3].legend(fontsize=7, loc="upper right")
    axrow[3].set_xlim(t[0], t[-1])
    for a in axrow[:3]:
        a.set_ylabel("freq (GHz)", fontsize=7)
    axrow[3].set_xlabel("time (ms)", fontsize=7)


def plot_one(npz_fp, out_dir):
    z = np.load(npz_fp, allow_pickle=True)
    b = str(z["burst"])
    al, tau = float(z["alpha"]), float(z["tau_1ghz"])
    cC, cD = float(z["chi2C"]), float(z["chi2D"])
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.2))
    _band_row(ax[0], z, "D", "tab:cyan")  # DSA top
    _band_row(ax[1], z, "C", "tab:red")  # CHIME bottom
    fig.suptitle(
        f"{b}  —  joint α={al:.3f}, τ₁GHz={tau:.4f} ms   |   "
        f"DSA χ²/dof={cD:.2f}   CHIME χ²/dof={cC:.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = Path(out_dir) / f"{b}_jointmodel.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    print(f"wrote {fp}  (DSA χ²={cD:.2f}, CHIME χ²={cC:.2f})")


def main():
    npz_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(sys.argv[3:])
    for fp in sorted(npz_dir.glob("*_jointmodel*.npz")):
        b = fp.name.split("_jointmodel")[0]
        if want and b not in want:
            continue
        plot_one(fp, out_dir)


if __name__ == "__main__":
    main()
