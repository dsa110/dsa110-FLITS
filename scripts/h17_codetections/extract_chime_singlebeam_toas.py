#!/usr/bin/env python
"""Re-extract CHIME 400 MHz TOAs from singlebeam HDF5 files.

Faithful reproduction of crossmatching/toa_crossmatch.ipynb cell 11:
  BBData.from_file -> coherent_dedisp(time_shift=False) -> incoherent_dedisp(fill_wfall=True)
  -> Stokes I = |X|^2+|Y|^2 -> per-channel noise-window normalisation -> freq-collapse
  -> peak = argmax(savgol_filter(I_t, 9, 3))
  -> t0 = time0[ctime+ctime_offset][-1]; offset = peak*delta_time; f_center = freq centre[-1]
  -> toa_unix_400 = t0 + offset + K_DM*DM*(1/400^2 - 1/f_center^2)

Reproduction gap vs notebook: bursts.fits (per-burst `time_limits` noise window and
`rfi_limits` channel mask) is absent on h17. The noise window is derived automatically
here (off-pulse region around a first-pass coarse peak) and a mild auto RFI mask is
applied. Both deviations are recorded per burst in the output.

Run inside the CANFAR baseband-analysis image:
  bin/baseband_analysis_python.sh scripts/extract_chime_singlebeam_toas.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp

ROOT = Path("/data/research/astrophysics/frbs/chime-dsa-codetections")
SB_DIR = Path("/data/Faber2026/data/chime-frb")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
DIAG = ROOT / "diagnostics" / "chime_singlebeam_toa"
RESULTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s  (notebook value)
F_REF_MHZ = 400.0
SAVGOL_WIN, SAVGOL_POLY = 9, 3


def auto_noise_mask(ntime: int, coarse_peak: int, fwhm_guess: int = 200) -> np.ndarray:
    """Off-pulse boolean mask: everything outside +/- ~8*fwhm of the coarse peak."""
    guard = max(fwhm_guess * 8, 500)
    idx = np.arange(ntime)
    mask = (idx < coarse_peak - guard) | (idx > coarse_peak + guard)
    if mask.sum() < ntime * 0.1:  # fall back to outer 40% if window too wide
        mask = (idx < int(0.2 * ntime)) | (idx > int(0.8 * ntime))
    return mask


def extract_one(path: Path, dm: float, ref_mhz: float) -> dict:
    bb = BBData.from_file(str(path))
    rec = {
        "file": path.name,
        "event_id": str(bb.attrs.get("event_id")),
        "event_date": str(bb.attrs.get("event_date")),
        "delta_time_s": float(bb.attrs["delta_time"]),
        "baseband_shape": list(bb["tiedbeam_baseband"].shape),
        "baseband_dtype": str(bb["tiedbeam_baseband"].dtype),
        "power_shape": list(bb["tiedbeam_power"].shape),
        "power_dtype": str(bb["tiedbeam_power"].dtype),
        "has_tiedbeam_baseband": "tiedbeam_baseband" in bb,
        "has_tiedbeam_power": "tiedbeam_power" in bb,
        "has_time0": "time0" in bb,
        "has_first_packet_recv_time": "first_packet_recv_time" in bb,
    }
    freq = np.asarray(bb.index_map["freq"]["centre"], dtype=float)
    rec["freq_top_mhz"] = float(freq[0])
    rec["freq_bottom_mhz"] = float(freq[-1])
    rec["nfreq"] = int(freq.size)

    delta_time = float(bb.attrs["delta_time"])

    # --- notebook dedispersion path ---
    bb_coh = coherent_dedisp(bb, float(dm), time_shift=False)
    bb["tiedbeam_baseband"][:] = bb_coh
    bb_inc, _f, _fid = incoherent_dedisp(bb, float(dm), fill_wfall=True)

    X, Y = bb_inc[:, 0, :], bb_inc[:, 1, :]
    I = np.abs(X) ** 2 + np.abs(Y) ** 2  # (nfreq, ntime)
    ntime = I.shape[-1]

    # coarse peak (un-normalised) to seed the noise window
    coarse = np.nansum(np.nan_to_num(I), axis=0)
    coarse_peak = int(np.nanargmax(coarse))

    noise = auto_noise_mask(ntime, coarse_peak)
    n_nan = int(np.isnan(I).sum())
    mu = np.nanmean(I[:, noise], axis=-1)[:, None]
    sd = np.nanstd(I[:, noise], axis=-1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        In = (I - mu) / sd
    In = np.nan_to_num(In, nan=0.0, posinf=0.0, neginf=0.0)

    # mild auto RFI mask: drop channels with zero/non-finite noise std (dead/saturated)
    chan_ok = np.isfinite(sd[:, 0]) & (sd[:, 0] > 0)
    n_rfi = int((~chan_ok).sum())
    In = In * chan_ok[:, None]

    I_t = np.nansum(In, axis=0)
    I_t_s = savgol_filter(I_t, SAVGOL_WIN, SAVGOL_POLY)
    peak_idx = int(np.nanargmax(I_t_s))

    # simple FWHM (samples) on smoothed collapsed profile, for the plot window
    half = I_t_s[peak_idx] / 2.0
    above = np.where(I_t_s > half)[0]
    fwhm_samp = int(above[-1] - above[0] + 1) if above.size else SAVGOL_WIN
    fwhm_ms = fwhm_samp * delta_time * 1e3

    # timing (notebook convention: bottom channel == index -1)
    t0 = bb["time0"]
    t0_unix = float(t0["ctime"][-1]) + float(t0["ctime_offset"][-1])
    offset_s = peak_idx * delta_time
    f_center = float(freq[-1])
    shift_s = K_DM * dm * (1.0 / ref_mhz**2 - 1.0 / f_center**2)
    toa_unix_400 = t0_unix + offset_s + shift_s
    toa_utc_400 = Time(toa_unix_400, format="unix", scale="utc")
    toa_utc_400.precision = 6

    base = np.nanmedian(I_t)
    mad = np.nanmedian(np.abs(I_t - base))
    snr_like = float((I_t[peak_idx] - base) / (1.4826 * mad)) if mad > 0 else float("inf")

    rec.update(
        {
            "method": "notebook_repro: coh+incoh dedisp, StokesI, auto-noise-norm+auto-RFI, "
            "freq-collapse, savgol(9,3) argmax peak",
            "profile_source": "tiedbeam_baseband (coherent+incoherent dedispersed)",
            "dedispersion": "coherent_dedisp(time_shift=False) + incoherent_dedisp(fill_wfall=True) at DM",
            "dm": float(dm),
            "reference_frequency_mhz": float(ref_mhz),
            "f_center_mhz": f_center,
            "peak_idx": peak_idx,
            "peak_offset_s": offset_s,
            "t0_unix_bottom_chan": t0_unix,
            "shift_to_400_s": shift_s,
            "toa_unix_400": toa_unix_400,
            "toa_utc_400": toa_utc_400.iso,
            "fwhm_ms": fwhm_ms,
            "snr_like": snr_like,
            "n_nan_samples_in_I": n_nan,
            "n_rfi_channels_masked": n_rfi,
            "noise_window_offpulse_frac": float(noise.mean()),
            "first_packet_recv_time": float(np.asarray(bb["first_packet_recv_time"]).ravel()[0]),
            "status": "ok",
        }
    )

    # --- diagnostic plot ---
    t_ms = np.arange(ntime) * delta_time * 1e3
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_ms, I_t, lw=0.6, color="0.3")
    lo = max(peak_idx - fwhm_samp, 0)
    hi = min(peak_idx + fwhm_samp, ntime - 1)
    ax.axvspan(t_ms[lo], t_ms[hi], color="r", alpha=0.25, label="FWHM window")
    ax.axvline(t_ms[peak_idx], color="r", lw=1.2, label="peak / TOA")
    ax.set_xlim(
        t_ms[max(peak_idx - 25 * fwhm_samp, 0)], t_ms[min(peak_idx + 25 * fwhm_samp, ntime - 1)]
    )
    ax.set_xlabel("Time since bottom-chan t0 (ms)")
    ax.set_ylabel("Stokes I (norm., a.u.)")
    ax.set_title(
        f"{path.stem}  event_id={rec['event_id']}  DM={dm:g}  "
        f"peak={peak_idx}  TOA400={toa_unix_400:.6f}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(DIAG / f"{path.stem}_toa.png", dpi=110)
    plt.close(fig)
    return rec


def main() -> int:
    inputs = json.load(open(SCRIPTS / "burst_inputs.json"))["bursts"]
    by_id = {r["chime_id"]: r for r in inputs}
    out = []
    for path in sorted(SB_DIR.glob("*/singlebeam_*.h5")):
        cid = path.stem.split("_")[-1]
        meta = by_id.get(cid)
        if meta is None:
            out.append({"file": path.name, "event_id": cid, "status": "no_fixture_row"})
            print(f"[SKIP] {path.name}: no fixture row")
            continue
        try:
            rec = extract_one(path, dm=meta["dm"], ref_mhz=meta["reference_frequency_mhz"])
            rec["name"] = meta["name"]
            rec["fixture_toa_unix_400"] = meta["fixture_toa_unix_400"]
            rec["residual_ms"] = (rec["toa_unix_400"] - meta["fixture_toa_unix_400"]) * 1e3
            out.append(rec)
            print(
                f"[OK]   {meta['name']:11s} {cid}  peak={rec['peak_idx']:6d}  "
                f"resid={rec['residual_ms']:+.4f} ms  snr~{rec['snr_like']:.1f}"
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "file": path.name,
                    "event_id": cid,
                    "name": meta["name"],
                    "status": f"error: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERR]  {path.name}: {exc}")

    json.dump(out, open(RESULTS / "chime_singlebeam_toa_reextract.json", "w"), indent=2)

    cols = [
        "name",
        "event_id",
        "file",
        "dm",
        "f_center_mhz",
        "peak_idx",
        "t0_unix_bottom_chan",
        "shift_to_400_s",
        "toa_unix_400",
        "toa_utc_400",
        "fixture_toa_unix_400",
        "residual_ms",
        "snr_like",
        "n_rfi_channels_masked",
        "fwhm_ms",
        "status",
    ]
    import csv

    with open(RESULTS / "chime_singlebeam_toa_reextract.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"\nWrote {RESULTS / 'chime_singlebeam_toa_reextract.json'}")
    print(f"Wrote {RESULTS / 'chime_singlebeam_toa_reextract.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
