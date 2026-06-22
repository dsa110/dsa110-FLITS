#!/usr/bin/env python
"""Wilhelm (or any co-detection) full-band absolute-TOA joint waterfall.

Layout: DSA (~1.31-1.50 GHz) on TOP, CHIME (~0.40-0.80 GHz) on BOTTOM, the
0.80-1.31 GHz inter-band GAP blank (no telescope observes there), on a shared
frequency axis.

Time axis is ABSOLUTE and inter-observatory-delay corrected: each band's
joint-fit peak t0 is placed at tau=0, so the two dedispersed bursts are aligned
to a common arrival (rather than each band's own arbitrary window origin). The
crossmatch geometric delay (CHIME@DRAO vs DSA@OVRO) and the measured TOA offset
are annotated; a dashed marker shows where DSA would sit WITHOUT the correction.

Three panels share the axis:
  DATA      - real CHIME+DSA, per-block peak-normalized (honest: separate obs)
  JOINT MODEL - shared-(tau_1ghz, alpha) M3 fit, continuous across the band,
                gap filled by frequency-interpolating the per-band intrinsics
  RESIDUAL  - (data-model)/noise, per band (blank in the gap)

  FLITS_RUNS=... FLITS_REPO=... python fullband_aligned.py <burst>
"""

import json
import os
import sys

import matplotlib
import numpy as np
import numpy.ma as ma

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

GAPNCH = 64


def prep(cfg_path, name):
    c = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(c["telcfg_path"], c["telescope"])
    ds = BurstDataset(
        c["path"],
        RUNS,
        name=name,
        telescope=tel,
        f_factor=int(c["f_factor"]),
        t_factor=int(c["t_factor"]),
        outer_trim=float(c.get("outer_trim", 0.15)),
        onpulse_crop=True,
        onpulse_pad_factor=0.5,
    )
    m = ds.model
    m.dm_init = float(c.get("dm_init", 0.0))
    return m


def band_norm(w):
    """per-channel baseline subtract, scale to [0,1] by block 99.5 pct (data)."""
    x = np.asarray(w, float)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    s = np.nanpercentile(x, 99.5)
    return np.clip(x / (s if s > 0 else 1.0), 0, 1)


def row_norm(w):
    """per-channel peak-normalize -> continuous burst SHAPE for the model panel."""
    x = np.asarray(w, float)
    x = x - np.nanmedian(x, axis=1, keepdims=True)
    pk = np.nanmax(x, axis=1, keepdims=True)
    return np.clip(x / np.where(pk > 0, pk, 1.0), 0, 1)


def main():
    b = sys.argv[1] if len(sys.argv) > 1 else "wilhelm"
    out = f"{RUNS}/data/joint"
    mC = prep(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime")
    mD = prep(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa")
    d = json.load(open(f"{out}/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    cm = json.load(open(f"{REPO}/crossmatching/toa_crossmatch_results.json"))[b]
    moff, geo = cm["measured_offset_ms"], cm["geometric_delay_ms"]
    clock = moff - geo
    dm = cm.get("dm", float("nan"))

    pC = FRBParams(
        c0=p["c0_C"],
        t0=p["t0_C"],
        gamma=p["gamma_C"],
        zeta=p["zeta_C"],
        tau_1ghz=tau,
        alpha=al,
        delta_dm=p["delta_dm_C"],
    )
    pD = FRBParams(
        c0=p["c0_D"],
        t0=p["t0_D"],
        gamma=p["gamma_D"],
        zeta=p["zeta_D"],
        tau_1ghz=tau,
        alpha=al,
        delta_dm=p["delta_dm_D"],
    )
    modC, modD = mC(pC, "M3"), mD(pD, "M3")

    oC, oD = np.argsort(mC.freq), np.argsort(mD.freq)
    fCv, fDv = mC.freq[oC], mD.freq[oD]

    # absolute, delay-corrected axis: each band's fit peak t0 -> tau=0 (aligned).
    tauC = mC.time - p["t0_C"]
    tauD = mD.time - p["t0_D"]

    # residual on native grid (per band), then everything regridded to common tau
    nsC = np.asarray(mC.noise_std, float)[:, None]
    nsD = np.asarray(mD.noise_std, float)[:, None]
    resC = (mC.data - modC) / np.where(nsC > 0, nsC, np.inf)
    resD = (mD.data - modD) / np.where(nsD > 0, nsD, np.inf)

    tlo = max(tauC.min(), tauD.min() if tauD.min() > -3 else -1.5)
    tlo = min(tauC.min(), tauD.min())
    thi = max(tauC.max(), tauD.max())
    tg = np.linspace(tlo, thi, 500)

    def regrid(tt, arr, order):
        out = np.full((arr.shape[0], tg.size), np.nan)
        for i, row in enumerate(arr[order]):
            out[i] = np.interp(tg, tt, row, left=np.nan, right=np.nan)
        return out

    dC, dD = regrid(tauC, mC.data, oC), regrid(tauD, mD.data, oD)
    yC, yD = regrid(tauC, modC, oC), regrid(tauD, modD, oD)
    rC, rD = regrid(tauC, resC, oC), regrid(tauD, resD, oD)

    # gap model: interpolate per-band intrinsics in frequency, eval channelwise on
    # the aligned axis (both peaks at tau=0 -> gap t0 interpolates to ~0).
    fclo, fchi = float(fCv.max()), float(fDv.min())
    gfreq = np.linspace(fclo + 0.005, fchi - 0.005, GAPNCH)

    def ip(nc, nd):
        return np.interp(gfreq, [fclo, fchi], [p[nc], p[nd]])

    c0g, gmg = ip("c0_C", "c0_D"), ip("gamma_C", "gamma_D")
    ztg, ddg = ip("zeta_C", "zeta_D"), ip("delta_dm_C", "delta_dm_D")
    yG = np.zeros((GAPNCH, tg.size))
    for i, fg in enumerate(gfreq):
        m1 = FRBModel(time=tg, freq=np.array([fg]), data=np.zeros((1, tg.size)), dm_init=mD.dm_init)
        pg = FRBParams(
            c0=c0g[i], t0=0.0, gamma=gmg[i], zeta=ztg[i], tau_1ghz=tau, alpha=al, delta_dm=ddg[i]
        )
        yG[i] = m1(pg, "M3")[0]

    freq_all = np.concatenate([fCv, gfreq, fDv])
    blank = np.full((GAPNCH, tg.size), np.nan)
    data_all = np.vstack([band_norm(dC), blank, band_norm(dD)])
    mdl_all = row_norm(np.vstack([yC, yG, yD]))
    res_all = np.vstack([rC, blank, rD])

    fig, (axd, axm, axr) = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True, sharex=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.85")
    rcmap = plt.get_cmap("coolwarm").copy()
    rcmap.set_bad("0.85")

    for ax, W, cm_, vlo, vhi, title in [
        (axd, data_all, cmap, 0, 1, "REAL DATA"),
        (axm, mdl_all, cmap, 0, 1, "JOINT MODEL (gap = extrapolation)"),
        (axr, res_all, rcmap, -4, 4, "RESIDUAL (data-model)/noise"),
    ]:
        ax.pcolormesh(
            tg, freq_all, ma.masked_invalid(W), cmap=cm_, shading="nearest", vmin=vlo, vmax=vhi
        )
        ax.axhline(fclo, color="cyan", lw=0.8, ls="--")
        ax.axhline(fchi, color="cyan", lw=0.8, ls="--")
        ax.axvline(0.0, color="lime", lw=0.8, ls=":")  # aligned arrival
        ax.axvline(-moff, color="white", lw=0.8, ls="--", alpha=0.6)  # uncorrected DSA
        ax.set_xlabel("time since aligned arrival (ms)")
        ax.set_title(title, fontsize=10)
    axd.set_ylabel("frequency (GHz)")
    axd.text(
        tg.mean(),
        0.5 * (fclo + fchi),
        "inter-band gap\n(no data)",
        ha="center",
        va="center",
        color="0.45",
        fontsize=9,
    )

    note = (
        f"{b}:  joint alpha={al:.2f}  tau(1GHz)={tau:.3f} ms  DM={dm:.1f} pc/cm3\n"
        f"inter-observatory TOA correction (CHIME@DRAO vs DSA@OVRO):\n"
        f"  measured offset = {moff:+.2f} ms  =  geometric {geo:+.2f} ms  +  clock {clock:+.2f} ms\n"
        f"  green dotted = aligned absolute arrival;  white dashed = DSA pre-correction ({-moff:+.2f} ms)"
    )
    fig.suptitle(note, fontsize=10, ha="left", x=0.01, y=1.02)
    fig.tight_layout()
    fp = f"{out}/{b}_fullband_aligned.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    print(f"wrote {fp}")
    print(f"alpha={al:.3f} tau1={tau:.4f} moff={moff:.3f} geo={geo:.3f} clock={clock:.3f}")
    print(
        f"CHIME tau=[{tauC.min():.2f},{tauC.max():.2f}] DSA tau=[{tauD.min():.2f},{tauD.max():.2f}]"
    )


if __name__ == "__main__":
    main()
