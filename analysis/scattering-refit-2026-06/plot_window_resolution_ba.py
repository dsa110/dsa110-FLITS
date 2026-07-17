#!/usr/bin/env python
"""Before/after figure for the joint-fit window + resolution fix.

Left column = OLD preprocessing (fixed config f_factor/t_factor + the unstable
``_crop_on_pulse`` window). Right column = NEW preprocessing (S/N-driven
resolution + robust common window from ``joint_tf_prep.prepare_pair``). Each panel
stacks DSA (top) over CHIME (bottom) on a shared peak-relative time axis; times
where a band has no data are hatched -- exactly the cross-band hatching the owner
flagged. The fix should (a) leave both bands covering the same time span (no
spurious hatch), and (b) show honest per-cell S/N (neither noise-mush nor
over-averaged).

  FLITS_RUNS=~/Developer/scratch/flits-local-runs conda run -n flits \
    python plot_window_resolution_ba.py --burst whitney_fine --burst oran ...
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scattering"))
sys.path.insert(0, str(HERE))

from scat_analysis.config_utils import load_telescope_block  # noqa: E402
from scat_analysis.pipeline.io import BurstDataset  # noqa: E402

import joint_tf_prep as J  # noqa: E402

RUNS = Path(os.environ.get("FLITS_RUNS", os.path.expanduser("~/Developer/scratch/flits-local-runs")))

# CHIME 400-800, gap, DSA ~1311-1499 (MHz). Fixed frequency frame for both cols.
F_LO, F_HI = 380.0, 1520.0


def _peak_rel_time(data, dt_ms):
    prof = np.nansum(data, axis=0)
    pk = int(np.argmax(prof))
    return (np.arange(data.shape[1]) - pk) * dt_ms


def _old_band(cfg_path, name):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"], "/tmp/ba_old", name=name, telescope=tel,
        f_factor=int(cfg["f_factor"]), t_factor=int(cfg["t_factor"]),
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=True, onpulse_pad_factor=0.5,
    )
    data = np.asarray(ds.data, float)
    return dict(data=data, t=_peak_rel_time(data, ds.dt_ms),
                f=np.asarray(ds.freq, float) * 1e3, dt=ds.dt_ms,
                noise=np.asarray(ds.model.noise_std, float).reshape(-1))


def _draw_band(ax, b, xlim):
    """pcolormesh one band in its freq range; return (f0,f1)."""
    t, f, d = b["t"], b["f"], b["data"]
    noise = b.get("noise")
    if noise is not None:
        d = d / np.clip(np.median(noise), 1e-9, None)
    df = np.median(np.diff(f))
    tf = np.median(np.diff(t))
    tedges = np.r_[t - tf / 2, t[-1] + tf / 2]
    fedges = np.r_[f - df / 2, f[-1] + df / 2]
    vmax = np.nanpercentile(d, 99.5)
    ax.pcolormesh(tedges, fedges, d, cmap="magma", vmin=np.nanpercentile(d, 20), vmax=vmax, rasterized=True)
    return f[0] - df / 2, f[-1] + df / 2


def _hatch_outside(ax, bands_franges, xlim, flo, fhi):
    """Hatch the whole frame, then the bands overwrite their coverage; here we
    instead hatch the complement: inter-band gap + any time a band lacks data."""
    # inter-band gap
    (c0, c1), (d0, d1) = bands_franges
    for g0, g1 in [(c1, d0)]:
        ax.add_patch(plt.Rectangle((xlim[0], g0), xlim[1] - xlim[0], g1 - g0,
                     facecolor="0.85", edgecolor="0.6", hatch="////", lw=0.0, zorder=1))


def _band_time_hatch(ax, b, xlim, frange):
    """Hatch the times within a band's freq range where it has no data."""
    t = b["t"]
    f0, f1 = frange
    tf = np.median(np.diff(t))
    lo, hi = t[0] - tf / 2, t[-1] + tf / 2
    if lo > xlim[0]:
        ax.add_patch(plt.Rectangle((xlim[0], f0), lo - xlim[0], f1 - f0,
                     facecolor="0.85", edgecolor="0.6", hatch="////", lw=0.0, zorder=1))
    if hi < xlim[1]:
        ax.add_patch(plt.Rectangle((hi, f0), xlim[1] - hi, f1 - f0,
                     facecolor="0.85", edgecolor="0.6", hatch="////", lw=0.0, zorder=1))


def _panel(ax, chime, dsa, title):
    tlo = min(chime["t"][0], dsa["t"][0])
    thi = max(chime["t"][-1], dsa["t"][-1])
    xlim = (tlo, thi)
    cfr = _draw_band(ax, chime, xlim)
    dfr = _draw_band(ax, dsa, xlim)
    _hatch_outside(ax, (cfr, dfr), xlim, F_LO, F_HI)
    _band_time_hatch(ax, chime, xlim, cfr)
    _band_time_hatch(ax, dsa, xlim, dfr)
    ax.set_xlim(*xlim)
    ax.set_ylim(F_LO, F_HI)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time from peak (ms)")


def make(burst, out_dir):
    cC = RUNS / "configs" / f"{burst}_chime_run.yaml"
    cD = RUNS / "configs" / f"{burst}_dsa_run.yaml"
    old_C = _old_band(cC, f"{burst}_chime")
    old_D = _old_band(cD, f"{burst}_dsa")
    (mC, MC), (mD, MD) = J.prepare_pair(str(cC), str(cD), burst, "/tmp/ba_new")
    new_C = dict(data=np.asarray(mC.data), t=_peak_rel_time(np.asarray(mC.data), MC.dt_ms),
                 f=np.asarray(mC.freq) * 1e3, dt=MC.dt_ms, noise=np.asarray(mC.noise_std).reshape(-1))
    new_D = dict(data=np.asarray(mD.data), t=_peak_rel_time(np.asarray(mD.data), MD.dt_ms),
                 f=np.asarray(mD.freq) * 1e3, dt=MD.dt_ms, noise=np.asarray(mD.noise_std).reshape(-1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    oc = yaml.safe_load(open(cC))
    od = yaml.safe_load(open(cD))
    _panel(axes[0], old_C, old_D,
           f"OLD  {burst}\nCHIME f{oc['f_factor']}/t{oc['t_factor']}  DSA f{od['f_factor']}/t{od['t_factor']}")
    _panel(axes[1], new_C, new_D,
           f"NEW  {burst}\nCHIME {MC.n_chan}ch/{MC.dt_ms*1e3:.0f}us S/N{MC.peak_pixel_snr:.0f}  "
           f"DSA {MD.n_chan}ch/{MD.dt_ms*1e3:.0f}us S/N{MD.peak_pixel_snr:.0f}")
    axes[0].set_ylabel("Frequency (MHz)")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{burst}_window_resolution_ba.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp, MC, MD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burst", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "joint_tf_figs")
    a = ap.parse_args()
    for b in a.burst:
        fp, MC, MD = make(b, a.out_dir)
        print(f"{b}: {fp}")
        print(f"   CHIME {MC.caption()}")
        print(f"   DSA   {MD.caption()}")


if __name__ == "__main__":
    main()
