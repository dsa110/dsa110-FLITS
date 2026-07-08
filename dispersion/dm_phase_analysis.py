from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DEFAULT_DATA_DIR,
    DEFAULT_DM_STEP,
    DSA_DT_S,
    _dm_ref,
    _dm_ref_source,
    _downsample_for_fit,
    _freq_grid_mhz,
    _freq_grid_source,
    _orient_waterfall_to_ascending_frequency,
    load_manifest_rows,
    shift_waterfall_residual_dm,
)
from dispersion.dmphasev2 import DMPhaseEstimator, dmphase_trial_to_physical_residual_dm

DEFAULT_F_CUT_HZ = (50.0, 1500.0)
DEFAULT_DM_WINDOW = 2.0
DEFAULT_N_BOOT = 24
DEFAULT_MAX_TIME = 2048
DEFAULT_MAX_CHANNELS_CHIME = 256
DEFAULT_MAX_CHANNELS_DSA = 128


def measure_dm_phase(
    waterfall: np.ndarray,
    freq_mhz: np.ndarray,
    dt_s: float,
    dm_ref: float,
    residual_dm_grid: np.ndarray,
    *,
    f_cut_hz: tuple[float, float] = DEFAULT_F_CUT_HZ,
    n_boot: int = DEFAULT_N_BOOT,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Measure a DM-phase residual around an explicit reference DM."""
    wf = _normalise_channels(waterfall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = DMPhaseEstimator(
            wf.T,
            freq_mhz,
            dt_s,
            residual_dm_grid,
            ref="top",
            f_cut=f_cut_hz,
            n_boot=n_boot,
            random_state=random_state,
        )
    raw_trial = float(estimator.dm_best)
    physical_residual = dmphase_trial_to_physical_residual_dm(raw_trial)
    step = float(np.nanmin(np.diff(residual_dm_grid)))
    sigma = max(float(estimator.dm_sigma), 0.5 * step)
    curve = np.asarray(estimator.dm_curve, dtype=float)
    shifted = shift_waterfall_residual_dm(
        waterfall,
        freq_mhz,
        dt_s,
        physical_residual,
        mode="zero_fill",
    )
    slope_before = _alignment_slope_abs(waterfall, freq_mhz)
    slope_after = _alignment_slope_abs(shifted, freq_mhz)
    slope_ratio = (
        float(slope_after / slope_before)
        if np.isfinite(slope_before) and slope_before > 0
        else float("nan")
    )
    edge_peak = bool(
        abs(raw_trial - float(residual_dm_grid.min())) < 1e-9
        or abs(raw_trial - float(residual_dm_grid.max())) < 1e-9
    )
    peak_snr = _curve_peak_snr(curve)
    accepted = bool(
        (not edge_peak)
        and np.isfinite(slope_ratio)
        and slope_ratio <= 1.10
        and (peak_snr >= 3.0 or abs(physical_residual) <= 0.05)
    )
    return {
        "dm_ref": float(dm_ref),
        "dmphase_raw_trial_dm": raw_trial,
        "physical_residual_dm": float(physical_residual),
        "candidate_dm": float(dm_ref + physical_residual),
        "candidate_sigma": sigma,
        "f_cut_hz": [float(f_cut_hz[0]), float(f_cut_hz[1])],
        "dm_grid": residual_dm_grid.tolist(),
        "dm_phase_curve": curve.tolist(),
        "peak_snr": peak_snr,
        "edge_peak": edge_peak,
        "slope_before": float(slope_before),
        "slope_after": float(slope_after),
        "slope_ratio": slope_ratio,
        "accepted": accepted,
        "reason": _acceptance_reason(accepted, edge_peak, slope_ratio, peak_snr, physical_residual),
        "dt_s": float(dt_s),
        "nu_ref_mhz": float(np.nanmax(freq_mhz)),
        "method": "DM-phase phase-coherence residual search around explicit dm_ref",
    }


def run_all(
    *,
    root: Path,
    data_dir: Path,
    output_json: Path,
    figure_dir: Path,
    sheet_path: Path,
    memo_path: Path,
    dm_window: float,
    dm_step: float,
    f_cut_hz: tuple[float, float],
    n_boot: int,
    max_time: int,
    telescopes: set[str],
    bursts: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_manifest_rows(root)
        if row["telescope"] in telescopes and (bursts is None or row["burst"] in bursts)
    ]
    residual_grid = np.arange(-dm_window, dm_window + 0.5 * dm_step, dm_step)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        result = _measure_manifest_row(
            row,
            data_dir=data_dir,
            residual_grid=residual_grid,
            f_cut_hz=f_cut_hz,
            n_boot=n_boot,
            max_time=max_time,
        )
        results.append(result)
        if result.get("dm_grid") and result.get("dm_phase_curve"):
            _plot_result(result, figure_dir)
        output_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        _write_memo(results, memo_path, figure_dir, output_json, sheet_path)
    _write_contact_sheet(results, sheet_path)
    output_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _write_memo(results, memo_path, figure_dir, output_json, sheet_path)
    return results


def _measure_manifest_row(
    row: dict[str, Any],
    *,
    data_dir: Path,
    residual_grid: np.ndarray,
    f_cut_hz: tuple[float, float],
    n_boot: int,
    max_time: int,
) -> dict[str, Any]:
    path = data_dir / row["filename"]
    telescope = row["telescope"]
    burst = row["burst"]
    dm_ref = _dm_ref(row)
    if not path.exists():
        return _row_failure(row, dm_ref, str(path), "input file missing")
    raw = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
    wf = _orient_waterfall_to_ascending_frequency(raw, telescope)
    freq = _freq_grid_mhz(telescope, wf.shape[0])
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    cropped, crop = _crop_on_peak(wf, dt_s)
    selected = cropped
    selected_freq = freq
    variant = "baseline128"
    max_channels = DEFAULT_MAX_CHANNELS_DSA
    if telescope == "chime":
        channel_mask = _valid_channel_mask(cropped)
        selected = cropped[channel_mask]
        selected_freq = freq[channel_mask]
        variant = "dropbad256"
        max_channels = DEFAULT_MAX_CHANNELS_CHIME
    if selected.shape[0] < 16:
        return _row_failure(row, dm_ref, str(path), "fewer than 16 valid channels after masking")
    fit_wf, fit_freq, fit_dt_s, factors = _downsample_for_fit(
        selected,
        selected_freq,
        dt_s,
        max_channels,
        max_time,
    )
    try:
        result = measure_dm_phase(
            fit_wf,
            fit_freq,
            fit_dt_s,
            dm_ref,
            residual_grid,
            f_cut_hz=f_cut_hz,
            n_boot=n_boot,
            random_state=_stable_seed(burst, telescope),
        )
    except Exception as exc:
        return _row_failure(row, dm_ref, str(path), f"fit failed: {exc}")
    return {
        **result,
        "burst": burst,
        "telescope": telescope,
        "input_path": str(path),
        "input_kind": "npy_manifest_cube_intensity",
        "source_manifest_filename": row["filename"],
        "dm_ref_provenance": _dm_ref_source(row),
        "freq_mhz_provenance": _freq_grid_source(telescope, raw.shape[0]),
        "dt_s_provenance": f"dispersion.dm_power_analysis.{telescope.upper()}_DT_S",
        "freq_mhz_min": float(np.nanmin(fit_freq)),
        "freq_mhz_max": float(np.nanmax(fit_freq)),
        "nchan": int(fit_wf.shape[0]),
        "ntime": int(fit_wf.shape[1]),
        "downsample_freq_factor": factors["freq"],
        "downsample_time_factor": factors["time"],
        "preprocess_variant": variant,
        "raw_nchan": int(raw.shape[0]),
        "raw_ntime": int(raw.shape[1]),
        "selected_nchan": int(selected.shape[0]),
        "crop_time_start": int(crop["start"]),
        "crop_time_stop": int(crop["stop"]),
        "crop_peak_index": int(crop["peak"]),
    }


def _normalise_channels(wf: np.ndarray) -> np.ndarray:
    x = np.asarray(wf, dtype=float)
    med = np.nanmedian(x, axis=1)[:, None]
    mad = np.nanmedian(np.abs(x - med), axis=1)[:, None]
    sig = np.where(mad > 0, 1.4826 * mad, np.nanstd(x, axis=1)[:, None])
    sig = np.where(sig > 0, sig, 1.0)
    return np.nan_to_num((x - med) / sig, nan=0.0, posinf=0.0, neginf=0.0)


def _crop_on_peak(wf: np.ndarray, dt_s: float, window_s: float = 0.030) -> tuple[np.ndarray, dict[str, int]]:
    z = _normalise_channels(wf)
    profile = np.nanmean(np.clip(z, 0.0, None), axis=0)
    if profile.size >= 9:
        profile = np.convolve(profile, np.ones(9) / 9, mode="same")
    peak = int(np.nanargmax(profile))
    half = max(24, int(0.5 * window_s / dt_s))
    lo = max(0, peak - half)
    hi = min(wf.shape[1], peak + half)
    lo = max(0, min(lo, wf.shape[1] - 2 * half))
    hi = min(wf.shape[1], lo + 2 * half)
    return wf[:, lo:hi], {"start": int(lo), "stop": int(hi), "peak": int(peak)}


def _valid_channel_mask(wf: np.ndarray) -> np.ndarray:
    x = np.asarray(wf, dtype=float)
    med = np.nanmedian(x, axis=1)
    mad = np.nanmedian(np.abs(x - med[:, None]), axis=1)
    return (np.isfinite(x).mean(axis=1) > 0.95) & (mad > 1e-8)


def _alignment_slope_abs(wf: np.ndarray, freq_mhz: np.ndarray) -> float:
    z = _normalise_channels(wf)
    profile = np.clip(z, 0.0, None)
    if profile.shape[1] >= 7:
        kernel = np.ones(7) / 7
        profile = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, profile)
    peak = np.argmax(profile, axis=1).astype(float)
    amp = np.max(profile, axis=1)
    valid = np.isfinite(peak) & np.isfinite(freq_mhz) & (amp > np.nanpercentile(amp, 60))
    if valid.sum() < 8:
        return float("nan")
    x = (freq_mhz[valid] - np.nanmean(freq_mhz[valid])) / 100.0
    try:
        return float(abs(np.polyfit(x, peak[valid], 1, w=amp[valid])[0]))
    except Exception:
        return float("nan")


def _curve_peak_snr(curve: np.ndarray) -> float:
    c = np.asarray(curve, dtype=float)
    med = float(np.nanmedian(c))
    mad = float(np.nanmedian(np.abs(c - med)))
    return float((float(np.nanmax(c)) - med) / (1.4826 * mad + 1e-30))


def _acceptance_reason(
    accepted: bool,
    edge_peak: bool,
    slope_ratio: float,
    peak_snr: float,
    residual_dm: float,
) -> str:
    if accepted:
        return "accepted by DM-phase smoke gate"
    if edge_peak:
        return "DM-phase peak is on search-grid edge"
    if np.isfinite(slope_ratio) and slope_ratio > 1.10:
        return f"residual shift worsens dynamic-spectrum slope (ratio={slope_ratio:.2f})"
    if peak_snr < 3.0 and abs(residual_dm) > 0.05:
        return f"weak DM-phase peak (peak_snr={peak_snr:.2f})"
    return "rejected by DM-phase smoke gate"


def _display_reduce(
    wf: np.ndarray,
    freq: np.ndarray,
    max_chan: int = 256,
    max_time: int = 700,
) -> tuple[np.ndarray, np.ndarray, int]:
    out = np.asarray(wf, dtype=float)
    f = np.asarray(freq, dtype=float)
    f_factor = max(1, out.shape[0] // max_chan)
    t_factor = max(1, out.shape[1] // max_time)
    if f_factor > 1:
        n = (out.shape[0] // f_factor) * f_factor
        out = np.nanmean(out[:n].reshape(n // f_factor, f_factor, out.shape[1]), axis=1)
        f = np.nanmean(f[:n].reshape(n // f_factor, f_factor), axis=1)
    if t_factor > 1:
        n = (out.shape[1] // t_factor) * t_factor
        out = np.nanmean(out[:, :n].reshape(out.shape[0], n // t_factor, t_factor), axis=2)
    return out, f, int(t_factor)


def _plot_result(result: dict[str, Any], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    wf = np.asarray(np.load(result["input_path"], mmap_mode="r"), dtype=float)
    wf = _orient_waterfall_to_ascending_frequency(wf, result["telescope"])
    freq = _freq_grid_mhz(result["telescope"], wf.shape[0])
    dt_s = CHIME_DT_S if result["telescope"] == "chime" else DSA_DT_S
    cropped = wf[:, int(result["crop_time_start"]) : int(result["crop_time_stop"])]
    selected = cropped
    selected_freq = freq
    if result["telescope"] == "chime":
        mask = _valid_channel_mask(cropped)
        selected = cropped[mask]
        selected_freq = freq[mask]
    shifted = shift_waterfall_residual_dm(
        selected,
        selected_freq,
        dt_s,
        float(result["physical_residual_dm"]),
        mode="zero_fill",
    )
    disp_in, dfreq, t_factor = _display_reduce(_normalise_channels(selected), selected_freq)
    disp_shift, _dfreq2, _t_factor2 = _display_reduce(_normalise_channels(shifted), selected_freq)
    effective_dt = dt_s * t_factor
    t0 = (int(result["crop_time_start"]) - int(result["crop_peak_index"])) * dt_s * 1e3
    times = t0 + np.arange(disp_in.shape[1]) * effective_dt * 1e3
    extent = [times[0], times[-1], float(dfreq.min()), float(dfreq.max())]
    vals = np.concatenate([disp_in.ravel(), disp_shift.ravel()])
    vmin, vmax = np.nanpercentile(vals, [2, 99.5])

    dm_ref = float(result["dm_ref"])
    grid = np.asarray(result["dm_grid"], dtype=float)
    curve = np.asarray(result["dm_phase_curve"], dtype=float)
    x = dm_ref - grid
    order = np.argsort(x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, image, title in [
        (axes[0], disp_in, f"input DMref={dm_ref:.3f}"),
        (axes[1], disp_shift, f"DM-phase cand={result['candidate_dm']:.3f}"),
    ]:
        im = ax.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(0.0, color="w", lw=0.8, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ms from input peak")
    axes[0].set_ylabel("MHz")
    axes[2].plot(x[order], curve[order], color="black", lw=1.2)
    axes[2].axvline(
        result["candidate_dm"],
        color="tab:red",
        ls="--",
        label=f"{result['candidate_dm']:.3f}+/-{result['candidate_sigma']:.3f}",
    )
    axes[2].axvline(dm_ref, color="0.5", ls=":", label="DMref")
    status = "ACCEPT" if result.get("accepted") else "REJECT"
    axes[2].set_title(f"{result['burst']} {result['telescope']} ({status})")
    axes[2].set_xlabel("absolute DM, physical sign")
    axes[2].set_ylabel("phase coherence")
    axes[2].legend(fontsize=7)
    fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.02)
    fig.suptitle(
        (
            f"{result['burst']} {result['telescope']}: f_cut={tuple(result['f_cut_hz'])}, "
            f"resid={result['physical_residual_dm']:+.3f}, snr={result['peak_snr']:.1f}, "
            f"slope {result['slope_before']:.2f}->{result['slope_after']:.2f}, "
            f"{result['preprocess_variant']}"
        ),
        fontsize=10,
    )
    fig.tight_layout()
    path = figure_dir / f"{result['burst']}_{result['telescope']}_dm_phase.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    result["figure_path"] = str(path)


def _write_contact_sheet(results: list[dict[str, Any]], sheet_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    items = [(r, Path(r["figure_path"])) for r in results if r.get("figure_path") and Path(r["figure_path"]).exists()]
    if not items:
        return
    cols = 2
    thumb_w = 760
    thumb_h = 300
    label_h = 34
    pad = 18
    nrows = (len(items) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, nrows * (thumb_h + label_h) + (nrows + 1) * pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    for i, (result, path) in enumerate(items):
        x = pad + (i % cols) * (thumb_w + pad)
        y = pad + (i // cols) * (thumb_h + label_h + pad)
        status = "ACCEPT" if result.get("accepted") else "REJECT"
        label = (
            f"{status} {result['burst']} {result['telescope']} "
            f"cand={result['candidate_dm']:.4f} "
            f"resid={result['physical_residual_dm']:+.4f} "
            f"snr={result['peak_snr']:.1f} ratio={result['slope_ratio']:.2f}"
        )
        draw.text((x, y), label, fill="black", font=font)
        img = Image.open(path).convert("RGB")
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
        img.thumbnail((thumb_w, thumb_h), resample)
        sheet.paste(img, (x + (thumb_w - img.width) // 2, y + label_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def _write_memo(
    results: list[dict[str, Any]],
    memo_path: Path,
    figure_dir: Path,
    output_json: Path,
    sheet_path: Path,
) -> None:
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = [r for r in results if r.get("accepted")]
    output_link = Path(os.path.relpath(output_json, memo_path.parent))
    figure_link = Path(os.path.relpath(figure_dir, memo_path.parent))
    sheet_link = Path(os.path.relpath(sheet_path, memo_path.parent))
    lines = [
        "# DM-phase fit memo",
        "",
        "Status: experimental cross-check, not promoted to citable DM_obs.",
        "",
        f"Results JSON: `{output_link}`",
        f"Figure directory: `{figure_link}`",
        f"Contact sheet: `{sheet_link}`",
        "",
        f"Accepted by diagnostic gate: {len(accepted)}/{len(results)}",
        "",
        "| Burst | Telescope | DM ref | Candidate DM | sigma | residual | peak S/N | slope ratio | Gate | Reason | Figure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in results:
        fig = ""
        if r.get("figure_path"):
            rel_fig = Path(os.path.relpath(r["figure_path"], memo_path.parent))
            fig = f"[curve]({rel_fig})"
        gate = "accept" if r.get("accepted") else "reject"
        if r.get("candidate_dm") is None:
            lines.append(
                f"| {r['burst']} | {r['telescope']} | {r['dm_ref']:.4f} |  |  |  |  |  | {gate} | {r['reason']} | {fig} |"
            )
        else:
            lines.append(
                
                    f"| {r['burst']} | {r['telescope']} | {r['dm_ref']:.4f} | "
                    f"{r['candidate_dm']:.4f} | {r['candidate_sigma']:.4f} | "
                    f"{r['physical_residual_dm']:+.4f} | {r['peak_snr']:.2f} | "
                    f"{r['slope_ratio']:.2f} | {gate} | {r['reason']} | {fig} |"
                
            )
    memo_path.write_text("\n".join(lines) + "\n")


def _row_failure(row: dict[str, Any], dm_ref: float, input_path: str, reason: str) -> dict[str, Any]:
    return {
        "burst": row["burst"],
        "telescope": row["telescope"],
        "input_path": input_path,
        "dm_ref": float(dm_ref),
        "candidate_dm": None,
        "candidate_sigma": None,
        "physical_residual_dm": None,
        "accepted": False,
        "reason": reason,
        "method": "DM-phase phase-coherence residual search around explicit dm_ref",
    }


def _stable_seed(burst: str, telescope: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(f"{burst}:{telescope}:dmphase")) % 2**32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=Path("results/dm_phase_results.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/dm_phase_figures"))
    parser.add_argument("--sheet", type=Path, default=Path("results/dm_phase_contact_sheet.png"))
    parser.add_argument("--memo", type=Path, default=Path("results/dm_phase_fit_memo.md"))
    parser.add_argument("--dm-window", type=float, default=DEFAULT_DM_WINDOW)
    parser.add_argument("--dm-step", type=float, default=DEFAULT_DM_STEP)
    parser.add_argument("--f-cut-low", type=float, default=DEFAULT_F_CUT_HZ[0])
    parser.add_argument("--f-cut-high", type=float, default=DEFAULT_F_CUT_HZ[1])
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--telescope", choices=["both", "chime", "dsa"], default="both")
    parser.add_argument("--burst", action="append", help="Restrict to one burst nickname; repeatable.")
    args = parser.parse_args(argv)

    telescopes = {"chime", "dsa"} if args.telescope == "both" else {args.telescope}
    bursts = set(args.burst) if args.burst else None
    run_all(
        root=args.root,
        data_dir=args.data_dir,
        output_json=args.output_json,
        figure_dir=args.figure_dir,
        sheet_path=args.sheet,
        memo_path=args.memo,
        dm_window=args.dm_window,
        dm_step=args.dm_step,
        f_cut_hz=(args.f_cut_low, args.f_cut_high),
        n_boot=args.n_boot,
        max_time=args.max_time,
        telescopes=telescopes,
        bursts=bursts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
