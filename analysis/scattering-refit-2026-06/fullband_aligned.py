#!/usr/bin/env python
"""Full-band absolute-TOA UNIFIED-model waterfall for a co-detected burst.

Reads the SHARED-zeta(nu) joint fit (<burst>_joint_fit.json, the canonical
default since shared zeta became the run_joint_fit default): one source
intrinsic-width law zeta(nu) = zeta_1ghz * nu^x_zeta and one scattering
law tau(nu) = tau_1ghz * nu^-alpha spanning BOTH telescopes. The model is a
single coherent burst from the DSA top (~1.50 GHz) through the unobserved
0.80-1.31 GHz gap to the CHIME bottom (~0.40 GHz) -- not two per-band fits
stitched together.

Layout: DSA TOP, CHIME BOTTOM, gap blank, shared frequency axis. Time axis is
ABSOLUTE and inter-observatory-delay corrected: each band's fit peak t0 -> tau=0,
so the two dedispersed bursts share a common arrival. The crossmatch geometric
delay (CHIME@DRAO vs DSA@OVRO) + measured TOA offset are annotated.

Three panels share the axis:
  REAL DATA      - windowed CHIME+DSA, per-block normalized
  UNIFIED MODEL  - ONE zeta(nu)/tau(nu) law, dm_init=0, continuous across the gap
  RESIDUAL       - (data - g_f*K_f)/noise, gain-marginal: the profiled per-channel
                   gain g_f absorbs the burst spectrum + scintillation, so the
                   residual shows pure temporal SHAPE misfit (white = good)

Writes <burst>_fullband_unified.png AND figures.manifest.json into the gated dir
(data/joint/gated_figures/<burst>/) so the repo figure-review Stop gate enforces
a visual review.

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


def gain_resid(m, p):
    """Gain-marginal residual (data - g_f*K_f)/noise on the band's native grid.

    K_f is the unit-amplitude scattering kernel; g_f = gain_spectrum is the
    profiled per-channel amplitude (burst spectrum * scintillation). Subtracting
    g_f*K_f whitens amplitude, so what remains is the temporal shape misfit only --
    the honest goodness-of-fit for a gain-marginal fit.
    """
    K = m(p, "M3")
    g = m.gain_spectrum(p, "M3")[:, None]
    ns = np.asarray(m.noise_std, float)[:, None]
    return (m.data - g * K) / np.where(ns > 0, ns, np.inf)


def write_manifest(gated_dir, png_name, b, al, tau, z1, xz, dm, moff, geo, clock):
    """Emit figures.manifest.json so the repo figure-review Stop gate covers this."""
    expectation = (
        f"Full-band UNIFIED-model joint waterfall for {b}. Layout: DSA "
        f"(1.31-1.50 GHz) TOP, CHIME (0.40-0.80 GHz) BOTTOM, 0.80-1.31 GHz gap "
        f"blank/gray. Three panels L->R: REAL DATA, UNIFIED MODEL, RESIDUAL "
        f"(gain-marginal, +-4 sigma). Time axis = inter-observatory-delay-corrected "
        f"absolute arrival: both band peaks at tau=0 (green dotted). MODEL is built "
        f"from ONE source law -- intrinsic width zeta(nu)=zeta_1ghz*nu^x_zeta "
        f"(zeta_1ghz={z1:.3f} ms, x_zeta={xz:+.2f}; x_zeta<0 = narrows with freq, "
        f"RFM) and scattering tau(nu)=tau_1ghz*nu^-alpha (tau_1ghz={tau:.3f} ms, "
        f"alpha={al:.2f}) -- evaluated per channel with dm_init=0 across CHIME+gap+DSA. "
        f"EXPECT in the MODEL panel: a single coherent burst whose width varies "
        f"MONOTONICALLY and SMOOTHLY from narrow at DSA (top) through the gap to the "
        f"broad scattered tail at CHIME (bottom), with NO width discontinuity at "
        f"either band/gap edge (the gap is the seamless bridge), spanning the full "
        f"time axis in every channel. RESIDUAL: gain-marginal so amplitude/scint is "
        f"absorbed -- expect ~white (|resid|<~3) if the scattering+width law fits; a "
        f"coherent red/blue block = remaining temporal shape misfit (e.g. unmodelled "
        f"sub-structure). Paper-styled: serif/stix fonts, panels labelled (a) data / "
        f"(b) unified model / (c) residual, each with its own colorbar (norm. flux for "
        f"a+b, residual sigma for c); 'DSA'/'CHIME' band labels at the top/bottom-right "
        f"of panel (a); cyan dashed = band edges, green dotted = aligned arrival. "
        f"Annotation: measured offset {moff:+.2f} ms = geometric {geo:+.2f} + clock "
        f"{clock:+.2f}, DM={dm:.1f}. The white-dashed DSA pre-correction marker at "
        f"{-moff:+.2f} ms is drawn ONLY if it lands within the time axis (else omitted)."
    )
    manifest = {
        "generated_by": "fullband_aligned.py",
        "burst": b,
        "figures": [{"path": png_name, "expectation": expectation}],
    }
    with open(f"{gated_dir}/figures.manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)


def main():
    b = sys.argv[1] if len(sys.argv) > 1 else "wilhelm"
    out = f"{RUNS}/data/joint"
    mC = prep(f"{RUNS}/configs/{b}_chime_run.yaml", f"{b}_chime")
    mD = prep(f"{RUNS}/configs/{b}_dsa_run.yaml", f"{b}_dsa")
    d = json.load(open(f"{out}/{b}_joint_fit.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, al, z1, xz = p["tau_1ghz"], p["alpha"], p["zeta_1ghz"], p["x_zeta"]
    cm = json.load(open(f"{REPO}/crossmatching/toa_crossmatch_results.json"))[b]
    moff, geo = cm["measured_offset_ms"], cm["geometric_delay_ms"]
    clock = moff - geo
    dm = cm.get("dm", float("nan"))

    # Per-band ACTUAL model for the residual: real dm_init + per-band (t0, delta_dm),
    # zeta = the shared law evaluated on each band's full channel axis (an array).
    def band_params(m, t0, ddm):
        zeta_nu = z1 * np.asarray(m.freq, float) ** xz
        return FRBParams(
            c0=1.0, t0=t0, gamma=0.0, zeta=zeta_nu, tau_1ghz=tau, alpha=al, delta_dm=ddm
        )

    pC = band_params(mC, p["t0_C"], p["delta_dm_C"])
    pD = band_params(mD, p["t0_D"], p["delta_dm_D"])
    resC, resD = gain_resid(mC, pC), gain_resid(mD, pD)

    oC, oD = np.argsort(mC.freq), np.argsort(mD.freq)
    fCv, fDv = mC.freq[oC], mD.freq[oD]

    # absolute, delay-corrected axis: each band's fit peak t0 -> tau=0 (aligned).
    tauC = mC.time - p["t0_C"]
    tauD = mD.time - p["t0_D"]

    tlo = min(tauC.min(), tauD.min())
    thi = max(tauC.max(), tauD.max())
    tg = np.linspace(tlo, thi, 500)

    def regrid(tt, arr, order):
        out = np.full((arr.shape[0], tg.size), np.nan)
        for i, row in enumerate(arr[order]):
            out[i] = np.interp(tg, tt, row, left=np.nan, right=np.nan)
        return out

    dC, dD = regrid(tauC, mC.data, oC), regrid(tauD, mD.data, oD)
    rC, rD = regrid(tauC, resC, oC), regrid(tauD, resD, oD)

    fclo, fchi = float(fCv.max()), float(fDv.min())
    gfreq = np.linspace(fclo + 0.005, fchi - 0.005, GAPNCH)
    freq_all = np.concatenate([fCv, gfreq, fDv])

    # UNIFIED MODEL panel = ONE coherent burst across the full band. For EVERY
    # frequency (CHIME + gap + DSA) the width is the single source law
    # zeta(nu)=zeta_1ghz*nu^x_zeta and the tail is tau(nu)=tau_1ghz*nu^-alpha,
    # evaluated with dm_init=0, delta_dm=0, t0=0 (peak at tau=0). No per-band
    # interpolation and no instrumental DM smearing -> width is set purely by
    # physics and varies smoothly, so the burst is continuous through both
    # band/gap edges (the gap is the model's seamless bridge).
    ymodel = np.zeros((freq_all.size, tg.size))
    for i, ff in enumerate(freq_all):
        mm = FRBModel(time=tg, freq=np.array([ff]), data=np.zeros((1, tg.size)), dm_init=0.0)
        pp = FRBParams(
            c0=1.0,
            t0=0.0,
            gamma=0.0,
            zeta=float(z1 * ff**xz),
            tau_1ghz=tau,
            alpha=al,
            delta_dm=0.0,
        )
        ymodel[i] = mm(pp, "M3")[0]

    blank = np.full((GAPNCH, tg.size), np.nan)
    data_all = np.vstack([band_norm(dC), blank, band_norm(dD)])
    mdl_all = row_norm(ymodel)
    res_all = np.vstack([rC, blank, rD])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, (axd, axm, axr) = plt.subplots(
        1, 3, figsize=(15, 6.0), sharey=True, sharex=True, constrained_layout=True
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.85")
    rcmap = plt.get_cmap("coolwarm").copy()
    rcmap.set_bad("0.85")

    draw_precorr = tlo <= -moff <= thi  # only mark DSA pre-correction if it lands on-axis
    panels = [
        (axd, data_all, cmap, 0, 1, "(a) data", "norm. flux"),
        (
            axm,
            mdl_all,
            cmap,
            0,
            1,
            r"(b) unified model: one $\zeta(\nu)$, $\tau(\nu)$",
            "norm. flux",
        ),
        (axr, res_all, rcmap, -4, 4, "(c) residual (gain-marginal)", r"residual ($\sigma$)"),
    ]
    for ax, W, cm_, vlo, vhi, title, clab in panels:
        im = ax.pcolormesh(
            tg, freq_all, ma.masked_invalid(W), cmap=cm_, shading="nearest", vmin=vlo, vmax=vhi
        )
        ax.axhline(fclo, color="cyan", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(fchi, color="cyan", lw=0.7, ls="--", alpha=0.7)
        ax.axvline(0.0, color="lime", lw=0.9, ls=":")  # aligned absolute arrival
        if draw_precorr:
            ax.axvline(-moff, color="white", lw=0.8, ls="--", alpha=0.55)
        ax.set_xlabel("time since aligned arrival (ms)")
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(clab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axd.set_ylabel("frequency (GHz)")
    axd.text(
        0.96,
        0.975,
        "DSA",
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="w",
        fontweight="bold",
    )
    axd.text(
        0.96,
        0.025,
        "CHIME",
        transform=axd.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="w",
        fontweight="bold",
    )
    axd.text(
        tg.mean(),
        0.5 * (fclo + fchi),
        "inter-band gap\n(no data)",
        ha="center",
        va="center",
        color="0.45",
        fontsize=8,
    )

    foot = (
        rf"$\alpha={al:.2f}$,  $\tau_{{1\rm GHz}}={tau:.3f}$ ms,  $\zeta_{{1\rm GHz}}={z1:.3f}$ ms,  "
        rf"$x_\zeta={xz:+.2f}$,  DM$={dm:.1f}$ pc cm$^{{-3}}$   |   inter-observatory TOA: "
        rf"offset ${moff:+.2f}$ ms $=$ geom ${geo:+.2f}$ $+$ clock ${clock:+.2f}$ ms "
        rf"(green dotted $=$ aligned arrival)"
    )
    fig.suptitle(
        rf"co-detection {b}: unified shared-$\zeta(\nu)$ joint scattering fit" "\n" + foot,
        fontsize=10.5,
    )

    gated = f"{out}/gated_figures/{b}"
    os.makedirs(gated, exist_ok=True)
    png_name = f"{b}_fullband_unified.png"
    fig.savefig(f"{gated}/{png_name}", dpi=200, bbox_inches="tight")
    write_manifest(gated, png_name, b, al, tau, z1, xz, dm, moff, geo, clock)
    print(f"wrote {gated}/{png_name} + figures.manifest.json")
    print(f"alpha={al:.3f} tau1={tau:.4f} zeta1={z1:.4f} x_zeta={xz:.3f}")
    print(
        f"CHIME tau=[{tauC.min():.2f},{tauC.max():.2f}] DSA tau=[{tauD.min():.2f},{tauD.max():.2f}]"
    )


if __name__ == "__main__":
    main()
