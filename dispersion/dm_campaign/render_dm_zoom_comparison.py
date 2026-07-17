#!/usr/bin/env python3
"""Zoomed dynamic-spectrum comparison of the adopted CHIME vs cross-check DSA DM.

For each co-detection burst this renders the on-pulse dynamic spectrum
dedispersed to (a) the CHIME/FRB phase-coherence DM (the manuscript-adopted
value) and (b) the DSA-110 DM, using the same dedispersion machinery that the
DM-phase v2 campaign used to fit them.  The visual test is direct: a burst that
is vertical (aligned across frequency) is correctly dedispersed; residual
sweep/tilt indicates the wrong DM.  Where the two DMs differ, the difference is
visible as a change in the frequency-dependent arrival time.

Inputs come from the campaign ``fits.json`` (per-band raw-product path,
product DM, and fitted DM).  This script does not refit; it only displays the
already-fitted DMs against the raw voltage-derived Stokes-I waterfalls.

Usage
-----
    python -m dispersion.dm_campaign.render_dm_zoom_comparison \
        --fits  <path>/results/fits.json \
        --out   <path>/results/diagnostics/dm_zoom_comparison.png \
        [--bursts phineas isha ...] [--window-ms 8]

Run from the pipeline root (so ``dispersion`` is importable), e.g.::

    cd Faber2026/pipeline && \
    /Users/.../py312/bin/python -m dispersion.dm_campaign.render_dm_zoom_comparison ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DSA_DT_S,
    _freq_grid_mhz,
    _orient_waterfall_to_ascending_frequency,
    shift_waterfall_residual_dm,
)


# ---------------------------------------------------------------------------
# Vendored verbatim from the DM-phase v2 campaign module ``dispersion.dm_joint_phase``
# (FLITS branch agent/dm-phase-v2, commit c07f1f166). That module is not in the
# pinned pipeline submodule, so the two display helpers the diagnostic needs are
# reproduced here to keep this script self-contained and byte-consistent with the
# fitted DMs. If dm_joint_phase later lands in the pin, prefer importing from it.
# ---------------------------------------------------------------------------
def normalise_channels(waterfall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robustly standardise channels and return the retained-channel mask."""
    wf = np.asarray(waterfall, dtype=float)
    if wf.ndim != 2:
        raise ValueError("waterfall must have shape (frequency, time)")
    finite = np.isfinite(wf).mean(axis=1)
    median = np.nanmedian(wf, axis=1)
    mad = np.nanmedian(np.abs(wf - median[:, None]), axis=1)
    sigma = 1.4826 * mad
    valid = (finite >= 0.90) & np.isfinite(sigma) & (sigma > 0)
    if valid.sum() < 16:
        raise ValueError("fewer than 16 valid frequency channels")
    out = (wf[valid] - median[valid, None]) / sigma[valid, None]
    return np.nan_to_num(out), valid


def crop_on_pulse(
    waterfall: np.ndarray,
    sample_time_s: float,
    *,
    window_s: float = 0.030,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Extract a fixed-width, burst-centred window with off-pulse margin."""
    z, valid = normalise_channels(waterfall)
    profile = np.mean(np.clip(z, 0.0, None), axis=0)
    smooth = max(1, int(round(1.0e-4 / sample_time_s)))
    if smooth > 1:
        profile = np.convolve(profile, np.ones(smooth) / smooth, mode="same")
    peak = int(np.argmax(profile))
    width = min(waterfall.shape[1], max(128, int(round(window_s / sample_time_s))))
    start = max(0, min(peak - width // 2, waterfall.shape[1] - width))
    stop = start + width
    return np.asarray(waterfall[valid, start:stop], dtype=float), (start, stop), valid

# Manuscript colour convention: CHIME = adopted, DSA = cross-check.
CH_COLOR = "#1f5fbf"
DS_COLOR = "#e08214"

TNS = {
    "zach": "20220207C", "whitney": "20220310F", "oran": "20220506D",
    "isha": "20221113A", "wilhelm": "20221203A", "phineas": "20230307A",
    "freya": "20230325A", "johndoeII": "20230814B", "hamilton": "20230913A",
    "mahi": "20240122A", "chromatica": "20240203A", "casey": "20240229A",
}
# Chronological order for a stable panel layout.
ORDER = ["zach", "whitney", "oran", "isha", "wilhelm", "phineas",
         "freya", "johndoeII", "hamilton", "mahi", "chromatica", "casey"]


def _dedispersed_display(band: dict, target_dm: float, window_s: float, max_chan: int, max_time: int):
    """Return (image, time_ms, freq_mhz) for ``band`` dedispersed to ``target_dm``.

    Reproduces the campaign display path: orient to ascending frequency, crop a
    burst-centred on-pulse window, per-channel standardise, shift by the
    residual DM (target minus the product DM the waterfall was written at), then
    block-average to a legible size.
    """
    telescope = band["telescope"]
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    raw = np.load(band["input_path"], mmap_mode="r")
    oriented = _orient_waterfall_to_ascending_frequency(raw, telescope)
    frequency = _freq_grid_mhz(telescope, oriented.shape[0])
    cropped, _crop, valid = crop_on_pulse(oriented, dt_s, window_s=window_s)
    freq = frequency[valid]
    z, valid2 = normalise_channels(cropped)
    freq = freq[valid2]
    residual = float(target_dm - band["product_dm"])
    shifted = shift_waterfall_residual_dm(z, freq, dt_s, residual, mode="zero_fill")
    # block-average for display without inventing samples
    ff = max(1, shifted.shape[0] // max_chan)
    tf = max(1, shifted.shape[1] // max_time)
    nf = (shifted.shape[0] // ff) * ff
    nt = (shifted.shape[1] // tf) * tf
    img = np.nanmean(shifted[:nf].reshape(nf // ff, ff, shifted.shape[1]), axis=1)
    img = np.nanmean(img[:, :nt].reshape(img.shape[0], nt // tf, tf), axis=2)
    dfreq = np.nanmean(freq[:nf].reshape(nf // ff, ff), axis=1)
    peak = int(np.argmax(np.nanmean(np.clip(img, 0, None), axis=0)))
    time_ms = (np.arange(img.shape[1]) - peak) * dt_s * tf * 1e3
    return img, time_ms, dfreq


K_DM_MS = 4.148808  # dispersion constant in ms for GHz^2, pc^-1 cm^3


def _sweep_ms(delta_dm: float, f_lo_mhz: float, f_hi_mhz: float) -> float:
    """Edge-to-edge dispersive delay (ms) produced by a DM error ``delta_dm``."""
    return K_DM_MS * delta_dm * ((f_lo_mhz / 1e3) ** -2 - (f_hi_mhz / 1e3) ** -2)


def _delay_ms(delta_dm: float, f_mhz, f_ref_mhz: float):
    """Cold-plasma residual delay (ms) at ``f_mhz`` relative to ``f_ref_mhz``."""
    f = np.asarray(f_mhz, dtype=float) / 1e3
    return K_DM_MS * delta_dm * (f ** -2 - (f_ref_mhz / 1e3) ** -2)


def _subband_arrival_times(band: dict, target_dm: float, window_s: float, n_sub: int):
    """Measure per-sub-band on-pulse arrival time after dedispersing to ``target_dm``.

    Returns (sub_freq_mhz, arrival_ms, ref_freq_mhz, dt_s). Arrival time is the
    smoothed-profile peak in each frequency sub-band, referenced to the
    band-integrated peak (so a correctly dedispersed burst is flat at ~0 ms and
    a DM error appears as a frequency-dependent tilt). Peaks are refined by a
    3-point parabolic interpolation. Sub-bands whose peak S/N is below 3 are
    returned as NaN so noise-dominated points do not masquerade as a tilt.
    """
    telescope = band["telescope"]
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    raw = np.load(band["input_path"], mmap_mode="r")
    oriented = _orient_waterfall_to_ascending_frequency(raw, telescope)
    frequency = _freq_grid_mhz(telescope, oriented.shape[0])
    cropped, _crop, valid = crop_on_pulse(oriented, dt_s, window_s=window_s)
    freq = frequency[valid]
    z, valid2 = normalise_channels(cropped)
    freq = freq[valid2]
    residual = float(target_dm - band["product_dm"])
    shifted = shift_waterfall_residual_dm(z, freq, dt_s, residual, mode="zero_fill")
    order = np.argsort(freq)
    shifted, freq = shifted[order], freq[order]

    # band-integrated reference peak (light boxcar smoothing)
    smooth = max(1, int(round(1.0e-4 / dt_s)))
    kern = np.ones(smooth) / smooth
    band_prof = np.convolve(np.nanmean(np.clip(shifted, 0, None), axis=0), kern, mode="same")
    ref_peak = int(np.argmax(band_prof))
    ref_freq = float(np.nanmedian(freq))

    # Restrict the per-sub-band peak search to the central 60% of the window so a
    # scattered low-frequency tail cannot place a spurious peak at the window edge.
    ntime = shifted.shape[1]
    guard = int(0.20 * ntime)
    search = slice(guard, ntime - guard)

    nchan = shifted.shape[0]
    edges = np.linspace(0, nchan, n_sub + 1, dtype=int)
    sub_freq = np.full(n_sub, np.nan)
    arrival = np.full(n_sub, np.nan)
    for i in range(n_sub):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 1:
            continue
        sub_freq[i] = float(np.nanmean(freq[lo:hi]))
        prof = np.convolve(np.nanmean(shifted[lo:hi], axis=0), kern, mode="same")
        noise = 1.4826 * np.nanmedian(np.abs(prof - np.nanmedian(prof)))
        pk = guard + int(np.argmax(prof[search]))
        if noise <= 0 or (prof[pk] - np.nanmedian(prof)) / noise < 3.0:
            continue
        # 3-point parabolic refinement of the peak sample
        if 0 < pk < len(prof) - 1:
            y0, y1, y2 = prof[pk - 1], prof[pk], prof[pk + 1]
            denom = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            shift = 0.0
        arrival[i] = (pk + shift - ref_peak) * dt_s * 1e3
    return sub_freq, arrival, ref_freq, dt_s


def _panel(ax, img, time_ms, freq, title, color):
    vmax = np.nanpercentile(img, 99.0)
    vmin = np.nanpercentile(img, 5.0)
    ax.imshow(
        img, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
        extent=[time_ms[0], time_ms[-1], freq[0], freq[-1]],
    )
    ax.axvline(0.0, color="white", lw=0.6, ls=":", alpha=0.6)
    ax.set_title(title, color=color, fontsize=7.5)


def render(fits_path: Path, out_path: Path, bursts, window_ms: float) -> Path:
    data = json.loads(Path(fits_path).read_text())
    byb = {e["burst"]: e for e in data}
    names = [b for b in ORDER if b in byb] if bursts is None else list(bursts)
    window_s = window_ms * 1e-3

    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 2.15 * n), squeeze=False)
    for row, b in enumerate(names):
        e = byb[b]
        cdm, ddm = e["chime"]["dm"], e["dsa"]["dm"]
        ddm_diff = e["joint"]["chime_minus_dsa"]
        for col, (band_key, dm, color, label) in enumerate([
            ("chime", cdm, CH_COLOR, "CHIME DM (adopted)"),
            ("dsa", ddm, DS_COLOR, "DSA DM (cross-check)"),
        ]):
            band = e[band_key]
            img, t_ms, freq = _dedispersed_display(
                band, dm, window_s, max_chan=220, max_time=260
            )
            ttl = f"{b}: {label} = {dm:.4f}"
            _panel(axes[row, col], img, t_ms, freq, ttl, color)
            if col == 0:
                axes[row, col].set_ylabel(f"FRB {TNS[b]}\nfreq (MHz)", fontsize=6.5)
        axes[row, 1].text(
            0.97, 0.04, f"$\\Delta$DM = {ddm_diff:+.4f}", transform=axes[row, 1].transAxes,
            ha="right", va="bottom", color="white", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5),
        )
    for col in range(2):
        axes[-1, col].set_xlabel("time from peak (ms)", fontsize=7)
    fig.suptitle(
        "Dynamic-spectrum test of the adopted DM: burst is vertical when correctly dedispersed\n"
        "left = CHIME DM (adopted), right = DSA DM (cross-check)",
        fontsize=9, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_discriminate(fits_path: Path, out_path: Path, bursts, window_ms: float) -> Path:
    """Dedisperse each band at BOTH candidate DMs to test which one is correct.

    A wrong DM leaves a residual frequency-dependent tilt.  The CHIME band
    (0.4-0.8 GHz) resolves a small DM error as a large sweep; the DSA band
    (1.3-1.5 GHz) sees the same error compressed to ~one sample.  This is why
    CHIME is the higher-leverage DM: the four columns are, per burst,
    CHIME@DM_CHIME, CHIME@DM_DSA, DSA@DM_CHIME, DSA@DM_DSA.  Panels dedispersed
    at the band's own best DM should be vertical; the cross-DM panels show the
    expected residual tilt (annotated as the edge-to-edge sweep in ms).
    """
    data = json.loads(Path(fits_path).read_text())
    byb = {e["burst"]: e for e in data}
    names = [b for b in ORDER if b in byb] if bursts is None else list(bursts)
    window_s = window_ms * 1e-3
    n = len(names)
    fig, axes = plt.subplots(n, 4, figsize=(13.0, 2.35 * n), squeeze=False)
    col_spec = [
        ("chime", "chime", CH_COLOR, "CHIME @ CHIME-DM"),
        ("chime", "dsa", DS_COLOR, "CHIME @ DSA-DM"),
        ("dsa", "chime", CH_COLOR, "DSA @ CHIME-DM"),
        ("dsa", "dsa", DS_COLOR, "DSA @ DSA-DM"),
    ]
    for row, b in enumerate(names):
        e = byb[b]
        dm_of = {"chime": e["chime"]["dm"], "dsa": e["dsa"]["dm"]}
        for col, (band_key, dm_key, color, label) in enumerate(col_spec):
            band = e[band_key]
            target = dm_of[dm_key]
            img, t_ms, freq = _dedispersed_display(
                band, target, window_s, max_chan=200, max_time=240
            )
            own = band_key == dm_key
            ttl = f"{label}\n{'aligned' if own else 'cross'}"
            _panel(axes[row, col], img, t_ms, freq, ttl, color if own else "0.4")
            axes[row, col].set_xlim(-window_ms / 2, window_ms / 2)
            if not own:
                sweep = _sweep_ms(target - dm_of[band_key], float(freq.min()), float(freq.max()))
                axes[row, col].text(
                    0.97, 0.04, f"tilt {sweep:+.2f} ms", transform=axes[row, col].transAxes,
                    ha="right", va="bottom", color="white", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
                )
            if col == 0:
                axes[row, col].set_ylabel(f"FRB {TNS[b]}\nfreq (MHz)", fontsize=6.5)
    for col in range(4):
        axes[-1, col].set_xlabel("time from peak (ms)", fontsize=7)
    fig.suptitle(
        "Which DM is correct? Each band dedispersed at both candidate DMs.\n"
        "Aligned (own-DM) panels are vertical; cross-DM panels tilt by the annotated sweep. "
        "A DM error is resolvable in CHIME (0.4-0.8 GHz) but sub-sample in DSA (1.3-1.5 GHz).",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_tilt(fits_path: Path, out_path: Path, bursts, window_ms: float, n_sub: int = 6) -> Path:
    """Sub-band arrival-time (tilt) test: turn 'is the burst vertical?' into a slope.

    For each band, the per-sub-band on-pulse arrival time is measured at BOTH
    candidate DMs and plotted against frequency, with the cold-plasma
    expectation for each DM overplotted. A correctly dedispersed burst is a
    vertical line (arrival independent of frequency); a DM error tilts it. The
    two candidate-DM series separate cleanly in CHIME (0.4-0.8 GHz) but lie on
    top of each other in DSA (1.3-1.5 GHz), because the same DM difference
    projects to hundreds of samples at CHIME and ~1 sample at DSA. This is the
    quantitative form of the discriminate figure: only CHIME can vote on the DM.
    """
    data = json.loads(Path(fits_path).read_text())
    byb = {e["burst"]: e for e in data}
    names = [b for b in ORDER if b in byb] if bursts is None else list(bursts)
    window_s = window_ms * 1e-3
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(8.0, 2.6 * n), squeeze=False)
    for row, b in enumerate(names):
        e = byb[b]
        dm_of = {"chime": e["chime"]["dm"], "dsa": e["dsa"]["dm"]}
        # Common time ruler for the whole burst row, set by the CHIME-band split
        # of the two candidate DMs. On this shared scale the DSA panel shows the
        # DM difference at its true (sub-sample) size instead of a misleading zoom.
        chime_split = abs(_sweep_ms(dm_of["chime"] - dm_of["dsa"], 400.0, 800.0))
        xlim = max(0.25, 1.4 * chime_split)
        for col, band_key in enumerate(["chime", "dsa"]):
            ax = axes[row, col]
            band = e[band_key]
            for dm_key, color in [("chime", CH_COLOR), ("dsa", DS_COLOR)]:
                sub_f, arr, ref_f, _dt = _subband_arrival_times(
                    band, dm_of[dm_key], window_s, n_sub
                )
                good = np.isfinite(arr)
                lbl = f"@ {dm_key.upper()}-DM"
                ax.plot(arr[good], sub_f[good], "o", color=color, ms=4.5, label=f"data {lbl}")
                sub_lo, sub_hi = float(np.nanmin(sub_f)), float(np.nanmax(sub_f))
                # Cold-plasma residual after dedispersing the band's own-DM burst
                # to the target DM: the residual delay is proportional to
                # (dm_true - dm_target) = dm_of[band_key] - dm_of[dm_key]. Over-
                # dedispersing (target > true) pulls low-frequency channels early;
                # under-dedispersing leaves them late. This must match the sign of
                # the shift applied to the data in _subband_arrival_times.
                fgrid = np.linspace(sub_lo, sub_hi, 100)
                theory = _delay_ms(dm_of[band_key] - dm_of[dm_key], fgrid, ref_f)
                ax.plot(theory, fgrid, "-", color=color, lw=1.2, alpha=0.7, label=f"cold-plasma {lbl}")
            ax.axvline(0.0, color="0.6", lw=0.7, ls=":")
            ax.set_title(f"{b}  {band_key.upper()} band", fontsize=8,
                         color=(CH_COLOR if band_key == "chime" else DS_COLOR))
            if col == 0:
                ax.set_ylabel(f"FRB {TNS[b]}\nfrequency (MHz)", fontsize=7)
            ax.set_xlim(-xlim, xlim)
            # Annotate the CHIME-vs-DSA DM split projected into THIS band, on the
            # shared row ruler. The DM is separable only where the split is large
            # relative to the panel scale; in DSA it is sub-sample (see caption).
            # Use the frequency limits measured from the loaded waterfall rather
            # than a catalog field, so the mode works on any product with the
            # documented input_path/product_dm/dm schema.
            sweep = _sweep_ms(dm_of["chime"] - dm_of["dsa"], sub_lo, sub_hi)
            ax.text(0.03, 0.03, f"DM split in band\n{abs(sweep):.3f} ms",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    for col in range(2):
        axes[-1, col].set_xlabel("arrival time from band peak (ms)", fontsize=7.5)
    axes[0, 0].legend(fontsize=5.6, frameon=False, loc="upper right", ncol=1)
    fig.suptitle(
        "Sub-band arrival-time test: a correct DM is a vertical line; a wrong DM tilts.\n"
        "CHIME (left) resolves the two candidate DMs as distinct slopes; DSA (right) cannot separate them.",
        fontsize=9.5, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bursts", nargs="*", default=None,
                    help="subset of burst nicknames; default = all, chronological")
    ap.add_argument("--window-ms", type=float, default=8.0)
    ap.add_argument("--mode", choices=["adopted", "discriminate", "tilt"], default="adopted",
                    help="'adopted': each band at its own DM (2 cols); "
                         "'discriminate': each band at both DMs (4 cols); "
                         "'tilt': sub-band arrival-time vs frequency at both DMs (2 cols)")
    ap.add_argument("--n-sub", type=int, default=6,
                    help="tilt mode: number of frequency sub-bands for arrival-time measurement")
    args = ap.parse_args()
    if args.mode == "discriminate":
        out = render_discriminate(args.fits, args.out, args.bursts, args.window_ms)
    elif args.mode == "tilt":
        out = render_tilt(args.fits, args.out, args.bursts, args.window_ms, args.n_sub)
    else:
        out = render(args.fits, args.out, args.bursts, args.window_ms)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
