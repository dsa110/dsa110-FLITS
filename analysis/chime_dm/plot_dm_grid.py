#!/usr/bin/env python
"""12-panel manuscript grid for the independent CHIME-side DM (association Pillar 2).

One cell per co-detected burst: (top) coherent-dedispersed-at-DSA-DM waterfall, (bottom) the
scatter-deconvolved sub-band arrival times t0 vs K_DM(nu^-2 - nu_ref^-2) with the weighted-linear
fit whose slope is the residual DM offset from DSA. 8/12 constrain CHIME DM (slope consistent with
0 within the 1 pc/cm^3 agreement floor -> de-biased near DSA); 4/12 are non-detections (<3 sub-bands
above S/N 4). The heavy dedispersion is done once off-repo (scripts/dump_grid_data.py); this script
only renders, so it needs no baseband/docker. See .agents/audit-chime-side-dm.md P5.

Run from the repo root:  python analysis/chime_dm/plot_dm_grid.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

K_DM = 4.148808e3  # s MHz^2 pc^-1 cm^3 (== flits.common.constants.K_DM)
REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("CHIME_DM_OUT", str(REPO / "analysis" / "chime_dm")))
SUFFIX = os.environ.get("CHIME_DM_SUFFIX", "")  # output filename suffix (for comparison renders)
EXTS = os.environ.get("CHIME_DM_EXTS", "svg,pdf,png").split(",")
TSMOOTH = int(
    os.environ.get("CHIME_DM_TSMOOTH", "12")
)  # display-only boxcar [time bins ~82 us]; ~1 ms. Does not touch the DM regression.
# Heavy per-burst plotting data lives with the (off-repo) baseband products; override with CHIME_DM_DATA.
DATA = Path(
    os.environ.get(
        "CHIME_DM_DATA", "/data/research/astrophysics/frbs/chime-dsa-codetections/results"
    )
)
NCOL, MIN_GOOD = 4, 3
OK, BAD = "#00B945", "#9e9e9e"  # repo prop_cycle green / grey


def _load():
    fits = json.loads((DATA / "chime_dm_grid_fits.json").read_text())
    wfs = np.load(DATA / "chime_dm_grid_waterfalls.npz")
    return fits, wfs


def _waterfall(ax, wf, extent, left):
    if TSMOOTH > 1:  # boxcar-smooth in time before z-scoring -> noise floor drops, pulse stands out
        wf = uniform_filter1d(wf, size=TSMOOTH, axis=1, mode="nearest")
    mu, sd = wf.mean(1, keepdims=True), wf.std(1, keepdims=True) + 1e-9
    ax.imshow(
        (wf - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=-0.5,
        vmax=5,
        cmap="magma",
        rasterized=True,
    )
    ax.set_xticklabels([])
    ax.tick_params(labelsize=9)
    if left:
        ax.set_ylabel(r"$\nu$ [MHz]", fontsize=10)
    else:
        ax.set_yticklabels([])


def _regression(ax, f, left, bottom):
    nu = np.asarray(f["nu"], float)
    if f["beta"] is not None and nu.size >= MIN_GOOD:
        x = K_DM * (1.0 / nu**2 - 1.0 / f["nu_ref"] ** 2)
        t0, err = np.asarray(f["t0"]) * 1e3, np.asarray(f["err"]) * 1e3
        ax.errorbar(x, t0, yerr=err, fmt="o", ms=4, color=OK, ecolor=OK, capsize=2, zorder=3)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(
            xs, (f["beta"][0] * xs + f["beta"][1]) * 1e3, "-", color="#FF2C00", lw=1.6, zorder=2
        )
        ax.text(
            0.04,
            0.92,
            rf"$\Delta$DM$={f['dm_offset']:+.2f}$"
            + "\n"
            + rf"$\sigma={f['sigma']:.2f}$ ($n={f['n_good']}$)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"non-detection\n($n={f['n_good']}$ sub-bands)",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=9,
            color=BAD,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(labelsize=9)
    if left:
        ax.set_ylabel(r"$t_0$ [ms]", fontsize=10)
    if bottom:
        ax.set_xlabel(r"$K_{\rm DM}(\nu^{-2}-\nu_{\rm ref}^{-2})$ [s pc$^{-1}$cm$^3$]", fontsize=9)


def main():
    fits, wfs = _load()
    nrow = int(np.ceil(len(fits) / NCOL))
    fig = plt.figure(figsize=(4.3 * NCOL, 3.6 * nrow))
    outer = fig.add_gridspec(nrow, NCOL, hspace=0.32, wspace=0.28)
    for i, f in enumerate(fits):
        r, c = divmod(i, NCOL)
        cell = outer[r, c].subgridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.06)
        a_wf, a_rg = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
        _waterfall(a_wf, wfs[f["name"]], f["extent"], left=(c == 0))
        col = OK if f["constrains"] else BAD
        dm = "--" if f["dm"] is None else f"{f['dm']:.2f}"
        a_wf.set_title(
            f"{f['name'].capitalize()}   DM$={dm}$  (DSA $={f['dm_dsa']:.2f}$)",
            fontsize=10.5,
            color=col,
            pad=3,
        )
        _regression(a_rg, f, left=(c == 0), bottom=(r == nrow - 1))
    fig.suptitle(
        "Independent CHIME-side DM vs DSA: arrival regression on coherent-dedispersed data "
        "(8/12 constrain, slope $\\approx0$ within the 1 pc cm$^{-3}$ floor; 4/12 non-detections)",
        fontsize=12,
        y=0.997,
    )
    for ext in EXTS:
        fig.savefig(OUT / f"chime_dm_grid{SUFFIX}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    nC = sum(f["constrains"] for f in fits)
    print(
        f"wrote chime_dm_grid{SUFFIX}.{{{','.join(EXTS)}}} (tsmooth={TSMOOTH}, {nC}/{len(fits)} constrain)"
    )


if __name__ == "__main__":
    main()
