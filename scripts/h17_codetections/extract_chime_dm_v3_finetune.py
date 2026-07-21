"""P4 re-run for the scatter-limited bursts (user recipe 2026-06-23).

The v2 arrival-regression read these 6 bursts +2..+3.5 pc/cm^3 high: their band-collapsed S/N(DM)
coarse stage was biased by off-pulse RFI + zero-fill triangles (huge in the high-DM, far-dedispersed
waterfalls). Recipe: coherent_dedisp@DSA -> mask -> CROP tight around the burst (kills the off-pulse
noise/zero-fill that biased S/N-max) -> S/N-max coarse on the crop -> preliminary correction ->
(a) arrival-regression cross-check, (b) DM-phase structure-max fine-tune (flat_ratio gates whether
the structure peak is real; ~2 = non-detection, the smooth-burst floor; >>1 = real).

chime_dm.py and dmphase_standalone.py are vendored in scripts/ (docker has no flits).
Run: bin/baseband_analysis_python.sh scripts/extract_chime_dm_v3_finetune.py
"""

import json
import os
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from chime_dm import _coarse_dm, _dedisperse, measure_dm
from dmphase_standalone import DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
DIAG = ROOT + "/diagnostics/chime_dm_v3"
os.makedirs(DIAG, exist_ok=True)
TDS = 16  # 2.56us -> 41us; bursts are ms-scale
TARGETS = set(
    os.environ.get("DM_TARGETS", "whitney,oran,isha,wilhelm,phineas,johndoeII,mahi").split(",")
)


def _mask_intensity(bbdd):
    intensity = np.nan_to_num(np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2)
    csd = intensity.std(axis=1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    return intensity[good], good


def _crop(wf, dt, pre_ms=10.0, post_ms=35.0):
    """Tight FIXED window around the burst peak (per-channel baseline-subtracted band collapse).

    A robust fixed window (not a fragile threshold) holds burst+scattering-tail at the DSA DM while
    being SHORTER than the dispersive sweep of a wrong DM: a +4 pc/cm^3 trial smears the 400-800 MHz
    band by ~78 ms, so a ~45 ms window zero-fills most of that trial -> if the +offset S/N-max is
    dispersive it collapses back toward DSA; if it survives, the offset is intra-burst (scattering).
    """
    base = np.median(wf, axis=1, keepdims=True)
    prof = (wf - base).sum(0)
    box = max(int(1.5e-3 / dt), 1)
    sm = np.convolve(prof, np.ones(box) / box, mode="same")
    pk = int(np.argmax(sm))
    lo = max(pk - int(pre_ms * 1e-3 / dt), 0)
    hi = min(pk + int(post_ms * 1e-3 / dt), wf.shape[1])
    return lo, hi


def _downsample_t(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


def _bin_freq(wf, freqs, nbin):
    """Block-average channels down to ~nbin (DM-phase structure-max needs coherence, not 800 ch)."""
    nf = wf.shape[0]
    if nf <= nbin:
        return wf, freqs
    k = nf // nbin
    m = (nf // k) * k
    return wf[:m].reshape(m // k, k, wf.shape[1]).mean(1), freqs[:m].reshape(m // k, k).mean(1)


def extract_one(path, dm_dsa):
    bb = BBData.from_file(path)
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)
    bbdd = coherent_dedisp(bb, dm_dsa)
    iw, good = _mask_intensity(bbdd)
    fw = freq[good]

    wf = _downsample_t(iw, TDS)
    dt_ds = dt * TDS
    lo, hi = _crop(wf, dt_ds)
    wfc = wf[:, lo:hi]
    crop_ms = (lo * dt_ds * 1e3, hi * dt_ds * 1e3)

    # ascending freq for the dispersion helpers
    order = np.argsort(fw)
    wfo, fro = wfc[order], fw[order]
    nu_ref = float(fro.max())

    # (1) S/N-max coarse on the clean crop
    ddm_c, grid, snr_curve, ic = _coarse_dm(wfo, fro, dt_ds, nu_ref, 5.0, 0.1)
    coarse_dm = dm_dsa + ddm_c

    # (a) arrival-regression cross-check on the crop (re-run the v2 estimator, now un-biased window)
    arr = measure_dm(
        wfc, fw, dt_ds, dm_dsa, n_subband=8, dm_window=5.0, dm_step=0.1, dm_err_max=20.0
    )

    # (2) preliminary correction, then DM-phase structure-max fine-tune
    wf_pre = _dedisperse(wfo, fro, dt_ds, ddm_c, nu_ref)
    wf_ph, fro_ph = _bin_freq(wf_pre, fro, 256)
    dm_grid = np.arange(-2.0, 2.0 + 1e-9, 0.1)
    est = DMPhaseEstimator(wf_ph.T, fro_ph, dt_ds, dm_grid, ref="top", n_boot=30, random_state=0)
    r = est.result()
    curve = np.asarray(r["dm_curve"], float)
    flat_ratio = float(curve.max() / (curve.min() + 1e-30))
    dmphase_dm = coarse_dm + float(r["dm_best"])

    res = {
        "dm_dsa": dm_dsa,
        "crop_ms": [round(crop_ms[0], 1), round(crop_ms[1], 1)],
        "coarse_dm": float(coarse_dm),
        "coarse_peak_snr": float(snr_curve[ic]),
        "arr_dm": arr["dm"],
        "arr_err": arr["dm_err"],
        "arr_nsub": arr["n_good_subbands"],
        "arr_constrains": arr["constrains_dm"],
        "dmphase_dm": dmphase_dm,
        "dmphase_sigma": float(r["dm_sigma"]),
        "flat_ratio": flat_ratio,
        "n_chan_ok": int(good.sum()),
    }
    fig = _figure(
        wf_pre,
        fro,
        dt_ds,
        dm_dsa,
        coarse_dm,
        grid + dm_dsa,
        snr_curve,
        r["dm_grid"] + coarse_dm,
        curve,
        res,
    )
    return res, fig


def _figure(wf_pre, freq, dt, dm_dsa, coarse_dm, snr_dm, snr_curve, ph_dm, ph_curve, res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    nf = (wf_pre.shape[0] // 4) * 4
    m = wf_pre[:nf].reshape(nf // 4, 4, wf_pre.shape[1]).mean(1)
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    ax[0].imshow(
        (m - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=[0, wf_pre.shape[1] * dt * 1e3, freq.min(), freq.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(title=f"cropped+precorr @ {coarse_dm:.2f}", xlabel="t (ms)", ylabel="freq MHz")
    ax[1].plot(snr_dm, snr_curve, ".-", ms=3)
    ax[1].axvline(dm_dsa, color="k", ls=":", label=f"DSA={dm_dsa:g}")
    ax[1].axvline(coarse_dm, color="r", label=f"coarse={coarse_dm:.2f}")
    ax[1].set(title=f"S/N-max (cropped) peak~{res['coarse_peak_snr']:.0f}", xlabel="trial DM")
    ax[1].legend(fontsize=8)
    ax[2].plot(ph_dm, ph_curve, ".-", ms=3)
    ax[2].axvline(dm_dsa, color="k", ls=":")
    ax[2].axvline(res["dmphase_dm"], color="r", label=f"DM-phase={res['dmphase_dm']:.2f}")
    ax[2].set(
        title=f"DM-phase fine: flat_ratio={res['flat_ratio']:.2f}",
        xlabel="trial DM",
        ylabel="coherent power",
    )
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    bursts = json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]
    out, manifest = [], []
    for meta in bursts:
        name = meta["name"]
        if name not in TARGETS:
            continue
        cid, dm = meta["chime_id"], float(meta["dm"])
        path = f"{SB}/{name.lower()}/singlebeam_{cid}.h5"
        if not os.path.exists(path):
            continue
        try:
            res, fig = extract_one(path, dm)
            png = f"{DIAG}/{name}_dm.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            res.update(name=name, chime_id=cid)
            out.append(res)
            arr_s = "None" if res["arr_dm"] is None else f"{res['arr_dm']:.2f}"
            manifest.append(
                {
                    "path": os.path.basename(png),
                    "expectation": f"{name}: cropped {res['crop_ms']} ms, S/N-max coarse={res['coarse_dm']:.2f} "
                    f"(DSA={dm:g}); arrival DM={arr_s} (nsub={res['arr_nsub']}); DM-phase fine={res['dmphase_dm']:.2f} "
                    f"flat_ratio={res['flat_ratio']:.2f} (flat~2 => structure non-detection)",
                }
            )
            print(
                f"[OK] {name:11s} DSA={dm:8.2f} coarse={res['coarse_dm']:8.2f} "
                f"arr={arr_s:>8}(n{res['arr_nsub']}) dmphase={res['dmphase_dm']:8.2f} "
                f"flat={res['flat_ratio']:5.2f}",
                flush=True,
            )
        except Exception as exc:
            out.append({"name": name, "chime_id": cid, "status": f"error: {exc}"})
            print(f"[ERR] {name}: {exc}\n{traceback.format_exc()}", flush=True)
        json.dump(out, open(ROOT + "/results/chime_dm_v3.json", "w"), indent=2)
        json.dump({"figures": manifest}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote results/chime_dm_v3.json ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
