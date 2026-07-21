"""Extract CHIME-side inputs for association pillars 2 & 4 from singlebeam voltages.

Pillar 2 — independent CHIME DM via structure-maximizing DM-phase:
  coherent_dedisp(time_shift=False)  [removes ~13 ms intra-channel smear; keeps 871 ch]
  -> numpy roll-align at DM_c (windowing only)
  -> tight DM-INDEPENDENT window -> DM-phase RESIDUAL grid around 0, TIME-FLIP orientation
     (real physical dispersion is recovered in the flipped orientation; verified on zach)
  -> robust peak: dm_best = DM_c + grid[argmax(mean curve)];
     sigma = std of bootstrap-curve argmaxes (the quadratic vertex fit is unreliable on the
     shallow curves of these scattered bursts).
Pillar 4 — CHIME tied-beam position: tiedbeam_locations ra/dec (the position the beam was
  formed at; an independent CHIME point, no error ellipse in singlebeam).

Run: bin/baseband_analysis_python.sh scripts/extract_chime_side_inputs.py
Writes chime_side_inputs.json + per-burst diagnostics + figures.manifest.json.
"""

import json
import os
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from dmphase_standalone import K_DM, DMPhaseEstimator

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
DIAG = ROOT + "/diagnostics/chime_side_dm"
os.makedirs(DIAG, exist_ok=True)

W = 512  # half-window (samples) around aligned burst — DM-independent
FDS = 3  # frequency downsample factor (speed)
GRID = np.arange(-6.0, 6.0, 0.15)  # residual DM grid around DM_c
N_BOOT = 40


def extract_one(path, dm_c):
    bb = BBData.from_file(path)
    dt = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)  # 871, descending
    ref = freq.max()
    ra = float(np.asarray(bb["tiedbeam_locations"]["ra"]).ravel()[0])
    dec = float(np.asarray(bb["tiedbeam_locations"]["dec"]).ravel()[0])

    from scipy.signal import savgol_filter

    bb_coh = coherent_dedisp(bb, dm_c, time_shift=False)
    I = np.abs(bb_coh[:, 0, :]) ** 2 + np.abs(bb_coh[:, 1, :]) ** 2  # (871, ntime) intra-removed
    shift = np.round((1e-3 * K_DM * (1.0 / freq**2 - 1.0 / ref**2) * dm_c) / dt).astype(int)
    Idd = np.stack([np.roll(I[j], -s) for j, s in enumerate(shift)])  # roll-align @ DM_c

    # RFI / dead-channel mask from per-channel variance (drop dead AND RFI-loud channels);
    # NO per-channel normalization (it amplified dead channels -> mis-placed windows before).
    csd = np.nanstd(Idd, axis=1)
    med = np.nanmedian(csd[csd > 0])
    chan_ok = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    good = np.where(chan_ok)[0][::FDS]  # subsample surviving channels for speed
    frg = freq[good]

    # robust coarse peak: RFI-masked band-collapse, savgol-smoothed
    collapse = np.nansum(np.nan_to_num(Idd[good]), 0)
    pk = int(np.argmax(savgol_filter(collapse, 51, 3)))
    lo, hi = max(pk - W, 0), min(pk + W, I.shape[1])
    win = np.nan_to_num(Idd[good][:, lo:hi])  # RAW: DMPhaseEstimator applies its own MAD weights

    est = DMPhaseEstimator(win[:, ::-1].T, frg, dt, GRID, ref="top", n_boot=N_BOOT, random_state=0)
    curve = est.result()["dm_curve"]
    i = int(np.argmax(curve))
    bs_peaks = GRID[np.argmax(est._bs_curves, axis=1)]
    dm_chime = float(dm_c + GRID[i])
    dm_err = float(np.std(bs_peaks, ddof=1))
    interior = bool(0 < i < len(GRID) - 1)
    flat = float(curve.max() / curve.min())

    # display normalization (figure only)
    mu = win.mean(1, keepdims=True)
    sd = win.std(1, keepdims=True) + 1e-9
    wn = (win - mu) / sd
    prof = wn.sum(0)
    noise = prof[: len(prof) // 6].std() + 1e-9
    snr = float((prof.max() - np.median(prof)) / noise)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    nf = wn.shape[0] // 4 * 4
    ax[0].imshow(
        wn[:nf].reshape(nf // 4, 4, wn.shape[1]).mean(1),
        aspect="auto",
        origin="lower",
        extent=[-W * dt * 1e3, W * dt * 1e3, frg.min(), frg.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(
        title=f"aligned waterfall ({chan_ok.sum()} ch ok)", xlabel="t (ms)", ylabel="freq (MHz)"
    )
    ax[1].plot((np.arange(wn.shape[1]) - wn.shape[1] // 2) * dt * 1e3, prof, lw=0.7, color="0.2")
    ax[1].set(title=f"profile  snr~{snr:.1f}", xlabel="t (ms)")
    ax[2].plot(dm_c + GRID, curve, ".-", ms=3)
    ax[2].axvline(dm_c, color="k", ls=":", label=f"DM_c={dm_c:g}")
    ax[2].axvspan(dm_chime - dm_err, dm_chime + dm_err, color="r", alpha=0.2)
    ax[2].axvline(dm_chime, color="r", label=f"DM_chime={dm_chime:.2f}±{dm_err:.2f}")
    ax[2].set(title=f"DM-phase (flat={flat:.2f}, interior={interior})", xlabel="trial DM")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    return {
        "dm_chime": dm_chime,
        "dm_chime_err": dm_err,
        "interior": interior,
        "flat_ratio": flat,
        "snr": snr,
        "n_chan_ok": int(chan_ok.sum()),
        "chime_ra_deg": ra,
        "chime_dec_deg": dec,
    }, fig


def main():
    by_id = {
        r["chime_id"]: r for r in json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]
    }
    out, manifest = [], []
    for path in sorted(__import__("glob").glob(SB + "/*/singlebeam_*.h5")):
        cid = os.path.basename(path).split("_")[-1].split(".")[0]
        meta = by_id.get(cid)
        if meta is None:
            continue
        try:
            rec, fig = extract_one(path, float(meta["dm"]))
            png = f"{DIAG}/{meta['name']}_dmphase.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            rec.update(
                name=meta["name"],
                chime_id=cid,
                dm_dsa=float(meta["dm"]),
                method="DM-phase structure-max (dmphasev2), residual grid, flip orient",
            )
            out.append(rec)
            manifest.append(
                {
                    "path": os.path.basename(png),
                    "expectation": f"{meta['name']}: aligned vertical burst; DM-phase curve "
                    f"peaks interior near DM_c={meta['dm']:g}; DM_chime={rec['dm_chime']:.2f}"
                    f"±{rec['dm_chime_err']:.2f} (flat={rec['flat_ratio']:.2f})",
                }
            )
            print(
                f"[OK] {meta['name']:11s} DM_dsa={meta['dm']:.3f} DM_chime={rec['dm_chime']:.3f}"
                f"±{rec['dm_chime_err']:.2f} interior={rec['interior']} flat={rec['flat_ratio']:.2f} "
                f"snr~{rec['snr']:.0f}",
                flush=True,
            )
        except Exception as exc:
            out.append({"name": meta["name"], "chime_id": cid, "status": f"error: {exc}"})
            print(f"[ERR] {meta['name']}: {exc}\n{traceback.format_exc()}", flush=True)
        # incremental write so a timeout preserves completed bursts
        json.dump(out, open(ROOT + "/results/chime_side_inputs.json", "w"), indent=2)
        json.dump({"figures": manifest}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote {ROOT}/results/chime_side_inputs.json ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
