#!/usr/bin/env python
"""Full-band joint waterfall for one co-detected burst.

LEFT  = REAL data on a shared frequency axis: CHIME (~0.4-0.8 GHz, bottom) and
        DSA (~1.3-1.5 GHz, top); the ~0.8-1.3 GHz inter-band GAP is blank — no
        telescope observes there.
RIGHT = JOINT MODEL: the shared-(tau_1ghz, alpha) M3 fit rendered CONTINUOUSLY
        across the full band, the gap FILLED by frequency-interpolating the
        per-band intrinsic params (c0,t0,gamma,zeta,delta_dm). Scattering
        tau(nu)=tau_1ghz*nu^-alpha and the dispersion sweep are continuous, so
        this shows what the model predicts the burst looks like everywhere,
        including where neither telescope sees it.

Display: each band block is peak-normalized separately (CHIME and DSA differ
greatly in absolute brightness), so the burst is visible in both.

  python fullband_waterfall.py <burst>
"""

import json
import os
import sys

import matplotlib
import numpy as np
import numpy.ma as ma

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("FLITS_REPO", "/home/jfaber/flits/dsa110-FLITS")
RUNS = os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs")
sys.path.insert(0, f"{REPO}/scattering")
import yaml
from scat_analysis.burstfit import FRBModel, FRBParams
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset

GAPNCH = 64


def prepare(cfg_path, name, outdir):
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"],
        outdir,
        name=name,
        telescope=tel,
        f_factor=int(cfg["f_factor"]),
        t_factor=int(cfg["t_factor"]),
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=os.environ.get("FLITS_ONPULSE_CROP", "1") == "1",
        onpulse_pad_factor=float(os.environ.get("FLITS_ONPULSE_PAD", "0.5")),
    )
    m = ds.model
    m.dm_init = float(cfg.get("dm_init", 0.0))
    return m


def band_norm(w):
    """Per-channel baseline subtract + scale to [0,1] by the block's 99.5 pct.
    Used for the DATA panel only (CHIME/DSA are genuinely separate observations
    with a real blank gap, so per-block scaling is honest there)."""
    w = np.asarray(w, float)
    x = w - np.nanmedian(w, axis=1, keepdims=True)
    s = np.nanpercentile(x, 99.5)
    return np.clip(x / (s if s > 0 else 1.0), 0, 1)


def row_norm(w):
    """Per-CHANNEL peak-normalize (each frequency row -> [0,1]). Continuous across
    the full band (no block boundaries), so the MODEL panel shows the burst SHAPE
    -- the scattering broadening + dispersion sweep -- smoothly through the gap.
    Trades away the amplitude spectrum (every channel peaks at 1) for continuity."""
    w = np.asarray(w, float)
    x = w - np.nanmedian(w, axis=1, keepdims=True)
    pk = np.nanmax(x, axis=1, keepdims=True)
    return np.clip(x / np.where(pk > 0, pk, 1.0), 0, 1)


def main():
    b = sys.argv[1]
    out = f"{RUNS}/data/joint"
    mC = prepare(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime", out)
    mD = prepare(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa", out)
    d = json.load(open(f"{out}/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al = p["tau_1ghz"], p["alpha"]
    shared = bool(d.get("shared_zeta", False))  # shared zeta(nu) fit is gain-marginal
    gain = bool(d.get("marginalize_gain", False)) or shared

    def zlaw(freqs):  # per-channel intrinsic width law
        return p["zeta_1ghz"] * np.asarray(freqs, float) ** p["x_zeta"]

    oC, oD = np.argsort(mC.freq), np.argsort(mD.freq)  # ascending freq for display
    fCv, fDv = mC.freq[oC], mD.freq[oD]

    # gain-marginal / shared-zeta fits carry no per-band c0,gamma -> unit kernel
    c0C, gmC = (1.0, 0.0) if gain else (p["c0_C"], p["gamma_C"])
    c0D, gmD = (1.0, 0.0) if gain else (p["c0_D"], p["gamma_D"])
    zC = zlaw(mC.freq) if shared else p["zeta_C"]
    zD = zlaw(mD.freq) if shared else p["zeta_D"]
    pC = FRBParams(
        c0=c0C, t0=p["t0_C"], gamma=gmC, zeta=zC, tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_C"]
    )
    pD = FRBParams(
        c0=c0D, t0=p["t0_D"], gamma=gmD, zeta=zD, tau_1ghz=tau, alpha=al, delta_dm=p["delta_dm_D"]
    )
    modC, modD = mC(pC, "M3"), mD(pD, "M3")

    tlo = max(mC.time.min(), mD.time.min())
    thi = min(mC.time.max(), mD.time.max())
    tg = np.linspace(tlo, thi, 400)

    def rt(time, arr, order):
        return np.vstack([np.interp(tg, time, row) for row in arr[order]])

    dC, dD = rt(mC.time, mC.data, oC), rt(mD.time, mD.data, oD)
    yC, yD = rt(mC.time, modC, oC), rt(mD.time, modD, oD)

    # --- gap model: interpolate per-band intrinsics in frequency, eval channelwise
    fclo, fchi = float(fCv.max()), float(fDv.min())
    gfreq = np.linspace(fclo + 0.005, fchi - 0.005, GAPNCH)

    def ip(nc, nd):
        # anchor at the band EDGES (not centers) so gap params equal each band's
        # params at the boundary -> the model connects continuously there.
        return np.interp(gfreq, [fclo, fchi], [p[nc], p[nd]])

    t0g, ddg = ip("t0_C", "t0_D"), ip("delta_dm_C", "delta_dm_D")
    if gain:  # no per-band c0,gamma -> unit kernel across the gap too
        c0g, gmg = np.full(GAPNCH, 1.0), np.zeros(GAPNCH)
    else:
        c0g, gmg = ip("c0_C", "c0_D"), ip("gamma_C", "gamma_D")
    # shared law eval'd on the gap freqs is continuous with each band edge by construction
    ztg = zlaw(gfreq) if shared else ip("zeta_C", "zeta_D")
    yG = np.zeros((GAPNCH, tg.size))
    for i, fg in enumerate(gfreq):
        m1 = FRBModel(time=tg, freq=np.array([fg]), data=np.zeros((1, tg.size)), dm_init=mD.dm_init)
        pg = FRBParams(
            c0=c0g[i], t0=t0g[i], gamma=gmg[i], zeta=ztg[i], tau_1ghz=tau, alpha=al, delta_dm=ddg[i]
        )
        yG[i] = m1(pg, "M3")[0]

    freq_all = np.concatenate([fCv, gfreq, fDv])
    gap_blank = np.full((GAPNCH, tg.size), np.nan)
    data_all = np.vstack([band_norm(dC), gap_blank, band_norm(dD)])
    mdl_all = row_norm(np.vstack([yC, yG, yD]))  # continuous per-channel norm

    fig, (axd, axm) = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.85")  # blank gap -> light gray in the data panel
    for ax, W, title in [
        (axd, data_all, f"{b}  REAL DATA  (CHIME + DSA; inter-band gap unobserved)"),
        (
            axm,
            mdl_all,
            f"{b}  JOINT MODEL  (row-normalized; gap = extrapolation; alpha={al:.2f}, tau1GHz={tau:.2f} ms)",
        ),
    ]:
        ax.pcolormesh(
            tg, freq_all, ma.masked_invalid(W), cmap=cmap, shading="nearest", vmin=0, vmax=1
        )
        ax.axhline(fclo, color="cyan", lw=0.8, ls="--")
        ax.axhline(fchi, color="cyan", lw=0.8, ls="--")
        ax.set_xlabel("time (ms)")
        ax.set_title(title, fontsize=10)
    axd.set_ylabel("frequency (GHz)")
    axd.text(
        tg.mean(),
        0.5 * (fclo + fchi),
        "inter-band gap\n(no data)",
        ha="center",
        va="center",
        color="0.45",
        fontsize=10,
    )
    axm.text(
        tg.min() + 0.04 * (tg.max() - tg.min()),
        0.5 * (fclo + fchi),
        "gap: tau(nu)+dispersion\nextrapolation only\n(intrinsic envelope interp.)",
        ha="left",
        va="center",
        color="cyan",
        fontsize=8,
        alpha=0.85,
    )
    fig.suptitle(f"{b}: full-band joint waterfall  (data vs continuous model)", fontsize=12)
    fig.tight_layout()
    fp = f"{out}/{b}_fullband_waterfall.png"
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    print(f"wrote {fp}  alpha={al:.2f} tau1={tau:.3f} gap=[{fclo:.2f},{fchi:.2f}]GHz")


if __name__ == "__main__":
    main()
