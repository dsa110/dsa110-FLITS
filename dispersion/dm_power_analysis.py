from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from dispersion.chime_dm import K_DM

CHIME_DT_S = 2.56e-6
CHIME_DF_MHZ = 0.390625
DSA_DT_S = 3.2768e-5
DSA_FCH1_MHZ = 1498.75
DSA_FOFF_MHZ = -0.03051757812
DEFAULT_DM_STEP = 0.05
CHIME_FULL_ROOT_DEFAULT = Path("~/Data/Faber2026/chimefrb/CHIME_bursts").expanduser()
DSA_FULL_ROOT_DEFAULT = Path("~/Data/Faber2026/dsa110/DSA_bursts").expanduser()
FREQ_DESCENDING_TELESCOPES = frozenset({"chime", "dsa"})


def root_for_telescope(
    telescope: str,
    chime_full_root: Path,
    dsa_full_root: Path,
) -> Path:
    roots = {"chime": Path(chime_full_root), "dsa": Path(dsa_full_root)}
    try:
        return roots[telescope]
    except KeyError as exc:
        raise ValueError(f"unknown telescope {telescope!r}") from exc


def residual_delay_s(
    freq_mhz: np.ndarray,
    residual_dm: float,
    nu_ref_mhz: float,
) -> np.ndarray:
    """Dispersive residual delay in seconds using the local `chime_dm.K_DM` scale."""
    freq = np.asarray(freq_mhz, dtype=float)
    return K_DM * float(residual_dm) * (1.0 / freq**2 - 1.0 / float(nu_ref_mhz) ** 2)


def shift_waterfall_residual_dm(
    waterfall: np.ndarray,
    freq_mhz: np.ndarray,
    dt_s: float,
    residual_dm: float,
    *,
    nu_ref_mhz: float | None = None,
    mode: str = "zero_fill",
) -> np.ndarray:
    """Shift a `(nchan, ntime)` waterfall by a residual DM.

    `zero_fill` is the real-data default. `fourier_circular` exists only for
    upstream-compatible comparisons and tests that prove why wrapping is unsafe.
    """
    wf, freq, _order = _validate_and_sort(waterfall, freq_mhz)
    nu_ref = float(freq.max() if nu_ref_mhz is None else nu_ref_mhz)
    shifts = residual_delay_s(freq, residual_dm, nu_ref) / float(dt_s)
    if mode == "zero_fill":
        return _shift_zero_fill(wf, shifts)
    if mode == "fourier_circular":
        return _shift_fourier_circular(wf, shifts)
    raise ValueError("mode must be 'zero_fill' or 'fourier_circular'")


def dm_power_curve(
    waterfall: np.ndarray,
    freq_mhz: np.ndarray,
    dt_s: float,
    residual_dm_grid: np.ndarray,
    *,
    n_boot: int = 100,
    random_state: int | None = None,
    shift_mode: str = "zero_fill",
    n_log_bins: int = 12,
) -> dict[str, Any]:
    """Compute bootstrap DM-power curves over a residual-DM grid."""
    wf, freq, order = _validate_and_sort(waterfall, freq_mhz)
    wf = _normalise_channels(wf)
    grid = np.asarray(residual_dm_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("residual_dm_grid must be a 1-D array with at least three points")

    rng = np.random.default_rng(random_state)
    nchan = wf.shape[0]
    n_boot = max(1, int(n_boot))
    log_power = np.full((grid.size, n_boot, n_log_bins), np.nan, dtype=float)
    score = np.full((grid.size, n_boot), np.nan, dtype=float)
    nu_ref = float(freq.max())

    for i, ddm in enumerate(grid):
        shifted = shift_waterfall_residual_dm(
            wf,
            freq,
            dt_s,
            float(ddm),
            nu_ref_mhz=nu_ref,
            mode=shift_mode,
        )
        on, off = _on_off_windows(shifted)
        weights = _svd_channel_weights(on)
        weighted_on = on * weights[:, None]
        weighted_off = off * weights[:, None]
        for b in range(n_boot):
            idx = rng.choice(nchan, size=nchan, replace=True)
            pulse = np.abs(np.sum(weighted_on[idx], axis=0))
            noise = np.abs(np.sum(weighted_off[idx], axis=0))
            power = _profile_power_subtract(pulse, noise)
            rebinned = _log_rebin_positive_power(power, n_log_bins)
            log_power[i, b] = rebinned
            positive = np.clip(rebinned, 0.0, None)
            score[i, b] = float(np.nanmean(positive))

    return {
        "residual_dm_grid": grid,
        "freq_mhz": freq,
        "dt_s": float(dt_s),
        "nu_ref_mhz": nu_ref,
        "shift_mode": shift_mode,
        "channel_order": order,
        "log_power": log_power,
        "score": score,
    }


def fit_dm_power_result(
    curve: dict[str, Any],
    *,
    dm_ref: float,
    min_successful_bins: int = 3,
    min_peak_snr: float = 2.5,
) -> dict[str, Any]:
    """Fit residual-DM peaks and return an absolute-DM result."""
    grid = np.asarray(curve["residual_dm_grid"], dtype=float)
    log_power = np.asarray(curve["log_power"], dtype=float)
    score = np.asarray(curve["score"], dtype=float)
    mean_score = np.nanmean(score, axis=1)
    score_peak_i = int(np.nanargmax(mean_score))
    peak_snr = _curve_peak_snr(mean_score)
    curve_max_err = _bootstrap_grid_peak_err(grid, score)
    base = {
        "dm_ref": float(dm_ref),
        "residual_dm_grid_min": float(grid.min()),
        "residual_dm_grid_max": float(grid.max()),
        "nu_ref_mhz": float(curve["nu_ref_mhz"]),
        "dt_s": float(curve["dt_s"]),
        "shift_mode": str(curve["shift_mode"]),
        "score_curve": mean_score.tolist(),
        "residual_dm_grid": grid.tolist(),
        "peak_structure_metric": float(np.nanmax(mean_score)),
        "peak_snr": float(peak_snr),
        "global_score_grid_max_residual_dm": float(grid[score_peak_i]),
        "global_score_grid_max_err": curve_max_err,
    }
    if not np.isfinite(peak_snr):
        return _unconstrained(base, "DM-power score curve is not finite")

    bin_peaks: list[float] = []
    bin_sigmas: list[float] = []
    mean_log = np.nanmean(log_power, axis=1)
    for b in range(mean_log.shape[1]):
        peak = _fit_gaussian_peak(grid, mean_log[:, b])
        if peak is None:
            continue
        x0, sigma = peak
        bin_peaks.append(x0)
        bin_sigmas.append(max(sigma, np.diff(grid).min()))

    if len(bin_peaks) < min_successful_bins:
        return _unconstrained(base, f"only {len(bin_peaks)} delay-frequency bins fit")

    peaks = np.asarray(bin_peaks)
    sigmas = np.asarray(bin_sigmas)
    global_peak = _fit_quadratic_peak(grid, mean_score)
    weights = 1.0 / np.square(sigmas)
    weighted_bin_peak = float(np.sum(peaks * weights) / np.sum(weights))
    bin_scatter = 1.4826 * float(np.nanmedian(np.abs(peaks - np.nanmedian(peaks))))
    bin_consistent = bin_scatter <= max(float(np.nanmedian(sigmas)), 2.0 * float(np.diff(grid).min()))
    if bin_consistent:
        residual_best = weighted_bin_peak
        fit_strategy = "per_delay_bin_inverse_variance"
    else:
        if global_peak is None:
            return _unconstrained(base, "delay-bin peaks disagree and global score has no interior peak")
        if peak_snr < min_peak_snr:
            return _unconstrained(
                base,
                f"delay-bin peaks disagree and global score is weak (peak_snr={peak_snr:.2f})",
            )
        residual_best = float(global_peak)
        fit_strategy = "global_score_fallback"

    boot_peaks = []
    if fit_strategy == "per_delay_bin_inverse_variance":
        for b in range(log_power.shape[1]):
            trial_peaks: list[float] = []
            trial_sigmas: list[float] = []
            for k in range(log_power.shape[2]):
                peak = _fit_gaussian_peak(grid, log_power[:, b, k])
                if peak is None:
                    continue
                x0, sigma = peak
                trial_peaks.append(x0)
                trial_sigmas.append(max(sigma, np.diff(grid).min()))
            if len(trial_peaks) >= min_successful_bins:
                trial_peaks_arr = np.asarray(trial_peaks)
                trial_sigmas_arr = np.asarray(trial_sigmas)
                trial_weights = 1.0 / np.square(trial_sigmas_arr)
                boot_peaks.append(float(np.sum(trial_peaks_arr * trial_weights) / np.sum(trial_weights)))
    else:
        for b in range(score.shape[1]):
            peak = _fit_quadratic_peak(grid, score[:, b])
            if peak is None:
                continue
            boot_peaks.append(peak)
    dm_err = float(np.nanstd(boot_peaks, ddof=1)) if len(boot_peaks) > 2 else float("nan")
    if not np.isfinite(dm_err) or dm_err <= 0:
        dm_err = float(max(np.diff(grid).min(), np.nanmedian(sigmas) / np.sqrt(len(sigmas))))
    if dm_err < 0.25 * np.diff(grid).min():
        dm_err = float(0.25 * np.diff(grid).min())

    if residual_best <= grid.min() or residual_best >= grid.max():
        return _unconstrained(base, "fitted residual DM outside search-grid interior")

    return {
        **base,
        "residual_dm_best": residual_best,
        "dm": float(dm_ref + residual_best),
        "dm_err": dm_err,
        "constrains_dm": True,
        "reason": "ok",
        "n_successful_bins": len(bin_peaks),
        "fit_strategy": fit_strategy,
        "bin_peak_scatter": bin_scatter,
        "global_score_residual_dm": None if global_peak is None else float(global_peak),
        "weighted_bin_residual_dm": weighted_bin_peak,
        "bin_peak_residual_dms": peaks.tolist(),
        "bin_peak_sigmas": sigmas.tolist(),
    }


def measure_dm_power(
    waterfall: np.ndarray,
    freq_mhz: np.ndarray,
    dt_s: float,
    dm_ref: float,
    residual_dm_grid: np.ndarray,
    *,
    n_boot: int = 100,
    random_state: int | None = None,
    shift_mode: str = "zero_fill",
    input_path: str | None = None,
    input_kind: str | None = None,
) -> dict[str, Any]:
    """Measure absolute DM by DM-power residual search around an explicit reference DM."""
    curve = dm_power_curve(
        waterfall,
        freq_mhz,
        dt_s,
        residual_dm_grid,
        n_boot=n_boot,
        random_state=random_state,
        shift_mode=shift_mode,
    )
    result = fit_dm_power_result(curve, dm_ref=dm_ref)
    wf = np.asarray(waterfall)
    freq = np.asarray(freq_mhz, dtype=float)
    return {
        **result,
        "input_path": input_path,
        "input_kind": input_kind,
        "freq_mhz_min": float(np.nanmin(freq)),
        "freq_mhz_max": float(np.nanmax(freq)),
        "nchan": int(wf.shape[0]),
        "ntime": int(wf.shape[1]),
        "method": "DM-power residual intensity search around explicit dm_ref",
    }


def mark_diagnostic_candidate_only(
    result: dict[str, Any],
    reason: str = "DM-power diagnostic candidate only; acceptance gate is not validated for this sample",
) -> dict[str, Any]:
    """Preserve a DM-power peak as a candidate without promoting it to a measurement."""
    marked = dict(result)
    curve_max_dm = None
    if result.get("global_score_grid_max_residual_dm") is not None and result.get("dm_ref") is not None:
        curve_max_dm = float(result["dm_ref"] + result["global_score_grid_max_residual_dm"])
    if result.get("dm") is not None:
        marked["dm_power_per_bin_candidate_dm"] = result["dm"]
        marked["dm_power_per_bin_candidate_err"] = result.get("dm_err")
        marked["dm_power_candidate_residual_dm"] = result.get("residual_dm_best")
        marked["dm_power_candidate_reason"] = result.get("reason")
    if curve_max_dm is not None:
        marked["dm_power_curve_max_dm"] = curve_max_dm
        marked["dm_power_curve_max_err"] = result.get("global_score_grid_max_err")
        marked["dm_power_candidate_dm"] = curve_max_dm
        marked["dm_power_candidate_err"] = result.get("global_score_grid_max_err")
    elif result.get("dm") is not None:
        marked["dm_power_candidate_dm"] = result["dm"]
        marked["dm_power_candidate_err"] = result.get("dm_err")
    marked["dm"] = None
    marked["dm_err"] = None
    marked["residual_dm_best"] = None
    marked["constrains_dm"] = False
    marked["reason"] = reason
    return marked


def load_manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest = list(csv.DictReader((root / "data-manifest.csv").open()))
    side_inputs = {
        str(row["name"]).lower(): row
        for row in json.loads((root / "crossmatching/chime_side_inputs.json").read_text())
    }
    fixture = json.loads((root / "crossmatching/notebook_reproduction_fixture.json").read_text())
    by_name = {str(row["name"]).lower(): row for row in fixture["bursts"]}
    rows = []
    for row in manifest:
        name_key = row["burst"].lower()
        side = side_inputs.get(name_key)
        fixed = by_name.get(name_key)
        rows.append({**row, "side_input": side, "fixture": fixed})
    return rows


def run_all(
    *,
    root: Path,
    chime_full_root: Path,
    dsa_full_root: Path,
    output_json: Path,
    figure_dir: Path,
    memo_path: Path,
    dm_window: float,
    dm_step: float,
    n_boot: int,
    max_channels: int,
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
            chime_full_root=chime_full_root,
            dsa_full_root=dsa_full_root,
            residual_grid=residual_grid,
            n_boot=n_boot,
            max_channels=max_channels,
            max_time=max_time,
        )
        results.append(result)
        _plot_result(result, figure_dir)
        output_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        _write_memo(results, memo_path, figure_dir, output_json)
    output_json.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _write_memo(results, memo_path, figure_dir, output_json)
    return results


def _measure_manifest_row(
    row: dict[str, Any],
    *,
    chime_full_root: Path,
    dsa_full_root: Path,
    residual_grid: np.ndarray,
    n_boot: int,
    max_channels: int,
    max_time: int,
) -> dict[str, Any]:
    telescope = row["telescope"]
    path = root_for_telescope(telescope, chime_full_root, dsa_full_root) / row["filename"]
    burst = row["burst"]
    dm_ref = _dm_ref(row)
    dm_ref_source = _dm_ref_source(row)
    if not path.exists():
        return _row_failure(row, dm_ref, str(path), "input file missing")
    data = np.load(path, mmap_mode="r")
    wf = _orient_waterfall_to_ascending_frequency(np.asarray(data, dtype=float), telescope)
    freq = _freq_grid_mhz(telescope, wf.shape[0])
    freq_source = _freq_grid_source(telescope, wf.shape[0])
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    dt_s_source = f"dispersion.dm_power_analysis.{telescope.upper()}_DT_S"
    wf, freq, dt_s, factors = _downsample_for_fit(wf, freq, dt_s, max_channels, max_time)
    try:
        result = measure_dm_power(
            wf,
            freq,
            dt_s,
            dm_ref,
            residual_grid,
            n_boot=n_boot,
            random_state=_stable_seed(burst, telescope),
            shift_mode="zero_fill",
            input_path=str(path),
            input_kind=(
                "npy_coherently_dedispersed_intensity"
                if telescope == "chime"
                else "npy_filterbank_intensity"
            ),
        )
    except Exception as exc:  # keep batch runs inspectable
        return _row_failure(row, dm_ref, str(path), f"fit failed: {exc}")
    return {
        **mark_diagnostic_candidate_only(result),
        "burst": burst,
        "telescope": telescope,
        "chime_id": (row.get("side_input") or {}).get("chime_id"),
        "downsample_freq_factor": factors["freq"],
        "downsample_time_factor": factors["time"],
        "source_manifest_filename": row["filename"],
        "dm_ref_provenance": dm_ref_source,
        "freq_mhz_provenance": freq_source,
        "dt_s_provenance": dt_s_source,
    }


def _validate_and_sort(
    waterfall: np.ndarray,
    freq_mhz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wf = np.asarray(waterfall, dtype=float)
    freq = np.asarray(freq_mhz, dtype=float)
    if wf.ndim != 2:
        raise ValueError("waterfall must have shape (nchan, ntime)")
    if freq.ndim != 1 or freq.shape[0] != wf.shape[0]:
        raise ValueError("freq_mhz must be a 1-D array matching waterfall.shape[0]")
    order = np.argsort(freq)
    return wf[order], freq[order], order


def _normalise_channels(wf: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=True)
    med = np.nanmedian(x, axis=1)[:, None]
    mad = np.nanmedian(np.abs(x - med), axis=1)[:, None]
    sig = np.where(mad > 0, 1.4826 * mad, np.nanstd(x, axis=1)[:, None])
    sig = np.where(sig > 0, sig, 1.0)
    return (x - med) / sig


def _shift_zero_fill(wf: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(wf)
    n = wf.shape[1]
    for j, shift in enumerate(np.rint(shifts).astype(int)):
        if abs(shift) >= n:
            continue
        if shift > 0:
            out[j, : n - shift] = wf[j, shift:]
        elif shift < 0:
            out[j, -shift:] = wf[j, : n + shift]
        else:
            out[j] = wf[j]
    return out


def _shift_fourier_circular(wf: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.empty_like(wf, dtype=float)
    freq = np.fft.fftfreq(wf.shape[1])
    for j, shift in enumerate(shifts):
        phase = np.exp(2j * np.pi * float(shift) * freq)
        out[j] = np.fft.ifft(np.fft.fft(wf[j]) * phase).real
    return out


def _on_off_windows(wf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ntime = wf.shape[1]
    q1 = int(0.25 * ntime)
    q3 = int(0.75 * ntime)
    on = wf[:, q1:q3]
    off = np.concatenate([wf[:, :q1], wf[:, q3:]], axis=1)
    n = min(on.shape[1], off.shape[1])
    return on[:, :n], off[:, :n]


def _svd_channel_weights(on: np.ndarray) -> np.ndarray:
    try:
        u, _s, _vh = np.linalg.svd(on, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.ones(on.shape[0], dtype=float)
    weights = u[:, 0]
    if np.nansum(weights) < 0:
        weights = -weights
    norm = np.sqrt(np.nansum(weights**2))
    return weights / norm if norm > 0 else np.ones(on.shape[0], dtype=float)


def _profile_power_subtract(pulse: np.ndarray, noise: np.ndarray) -> np.ndarray:
    pulse = pulse - np.nanmean(pulse)
    noise = noise - np.nanmean(noise)
    return np.abs(np.fft.rfft(pulse)) ** 2 - np.abs(np.fft.rfft(noise)) ** 2


def _log_rebin_positive_power(power: np.ndarray, n_bins: int) -> np.ndarray:
    p = np.asarray(power[1:], dtype=float)
    if p.size == 0:
        return np.full(n_bins, np.nan)
    idx = np.arange(1, p.size + 1)
    edges = np.unique(np.rint(np.geomspace(1, p.size + 1, n_bins + 1)).astype(int))
    if edges.size < 2:
        return np.full(n_bins, np.nan)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):  # noqa: B905 - h17 Python lacks zip(strict=...)
        mask = (idx >= lo) & (idx < hi)
        bins.append(float(np.nanmean(p[mask])) if np.any(mask) else np.nan)
    if len(bins) < n_bins:
        bins.extend([np.nan] * (n_bins - len(bins)))
    return np.asarray(bins[:n_bins], dtype=float)


def _fit_gaussian_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    i = int(np.argmax(y))
    if i == 0 or i == x.size - 1:
        return None
    amp = float(y[i] - np.nanmin(y))
    if amp <= 0:
        return None
    step = float(np.nanmin(np.diff(x)))
    p0 = [amp, float(x[i]), max(2.0 * step, 0.2), float(np.nanmin(y))]
    bounds = ([0.0, float(x.min()), step / 2.0, -np.inf], [np.inf, float(x.max()), np.inf, np.inf])
    try:
        popt, _pcov = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds, maxfev=2000)
    except (RuntimeError, ValueError):
        peak = _fit_quadratic_peak(x, y)
        return None if peak is None else (peak, max(2.0 * step, 0.2))
    return float(popt[1]), float(abs(popt[2]))


def _fit_quadratic_peak(x: np.ndarray, y: np.ndarray) -> float | None:
    if not np.all(np.isfinite(y)):
        return None
    i = int(np.argmax(y))
    if i == 0 or i == x.size - 1:
        return None
    lo = max(i - 2, 0)
    hi = min(i + 3, x.size)
    try:
        a, b, _c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    except np.linalg.LinAlgError:
        return float(x[i])
    if a >= 0:
        return float(x[i])
    return float(np.clip(-b / (2.0 * a), x[lo], x[hi - 1]))


def _gaussian(x: np.ndarray, amp: float, x0: float, sigma: float, offset: float) -> np.ndarray:
    return offset + amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _curve_peak_snr(curve: np.ndarray) -> float:
    c = np.asarray(curve, dtype=float)
    if c.size < 3 or not np.all(np.isfinite(c)):
        return float("nan")
    peak = float(np.nanmax(c))
    med = float(np.nanmedian(c))
    mad = float(np.nanmedian(np.abs(c - med)))
    return (peak - med) / (1.4826 * mad + 1e-12)


def _bootstrap_grid_peak_err(grid: np.ndarray, score: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    score = np.asarray(score, dtype=float)
    step = float(np.nanmin(np.diff(grid)))
    floor = 0.5 * step
    if score.ndim != 2 or score.shape[1] < 2:
        return floor
    peaks = []
    for b in range(score.shape[1]):
        col = score[:, b]
        if np.all(np.isfinite(col)):
            peaks.append(float(grid[int(np.nanargmax(col))]))
    if len(peaks) < 2:
        return floor
    err = float(np.nanstd(peaks, ddof=1))
    return max(err, floor)


def _unconstrained(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "residual_dm_best": None,
        "dm": None,
        "dm_err": None,
        "constrains_dm": False,
        "reason": reason,
        "n_successful_bins": 0,
    }


def _freq_grid_mhz(telescope: str, nchan: int) -> np.ndarray:
    if telescope == "chime":
        start = 400.0 + CHIME_DF_MHZ / 2.0
        return start + CHIME_DF_MHZ * np.arange(nchan)
    if telescope == "dsa":
        descending = DSA_FCH1_MHZ + DSA_FOFF_MHZ * np.arange(nchan)
        return np.sort(descending)
    raise ValueError(f"unknown telescope {telescope!r}")


def _dm_ref(row: dict[str, Any]) -> float:
    if row["telescope"] == "chime" and row.get("side_input"):
        return float(row["side_input"]["dm_dsa"])
    if row.get("fixture"):
        return float(row["fixture"]["dm"])
    return float(row["dm_pc_cm3"])


def _dm_ref_source(row: dict[str, Any]) -> str:
    if row["telescope"] == "chime" and row.get("side_input"):
        return "crossmatching/chime_side_inputs.json:dm_dsa"
    if row.get("fixture"):
        return "crossmatching/notebook_reproduction_fixture.json:dm"
    return "data-manifest.csv:dm_pc_cm3"


def _freq_grid_source(telescope: str, nchan: int) -> str:
    if telescope == "chime":
        return (
            "synthetic CHIME intensity grid from CHIME_DF_MHZ="
            f"{CHIME_DF_MHZ} MHz, start=400+CHIME_DF_MHZ/2, nchan={nchan}; "
            "raw rows flipped from freq_descending telescope config"
        )
    if telescope == "dsa":
        return (
            "synthetic DSA filterbank grid from DSA_FCH1_MHZ="
            f"{DSA_FCH1_MHZ}, DSA_FOFF_MHZ={DSA_FOFF_MHZ}, nchan={nchan}; "
            "raw rows flipped from freq_descending telescope config"
        )
    raise ValueError(f"unknown telescope {telescope!r}")


def _orient_waterfall_to_ascending_frequency(wf: np.ndarray, telescope: str) -> np.ndarray:
    """Return waterfall rows in ascending frequency order for DM-delay math."""
    if telescope in FREQ_DESCENDING_TELESCOPES:
        return np.flipud(wf)
    return wf


def _downsample_for_fit(
    wf: np.ndarray,
    freq: np.ndarray,
    dt_s: float,
    max_channels: int,
    max_time: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, int]]:
    out = np.asarray(wf, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.asarray(freq, dtype=float)
    f_factor = max(1, out.shape[0] // max_channels) if max_channels > 0 else 1
    t_factor = max(1, out.shape[1] // max_time) if max_time > 0 else 1
    if f_factor > 1:
        n = (out.shape[0] // f_factor) * f_factor
        out = np.nanmean(out[:n].reshape(n // f_factor, f_factor, out.shape[1]), axis=1)
        f = np.nanmean(f[:n].reshape(n // f_factor, f_factor), axis=1)
    if t_factor > 1:
        n = (out.shape[1] // t_factor) * t_factor
        out = np.nanmean(out[:, :n].reshape(out.shape[0], n // t_factor, t_factor), axis=2)
        dt_s *= t_factor
    return out, f, float(dt_s), {"freq": int(f_factor), "time": int(t_factor)}


def _row_failure(row: dict[str, Any], dm_ref: float, input_path: str, reason: str) -> dict[str, Any]:
    return {
        "burst": row["burst"],
        "telescope": row["telescope"],
        "input_path": input_path,
        "input_kind": None,
        "dm_ref": dm_ref,
        "residual_dm_best": None,
        "dm": None,
        "dm_err": None,
        "constrains_dm": False,
        "reason": reason,
        "method": "DM-power residual intensity search around explicit dm_ref",
    }


def _stable_seed(burst: str, telescope: str) -> int:
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(f"{burst}:{telescope}")) % 2**32)


def _plot_result(result: dict[str, Any], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grid = result.get("residual_dm_grid")
    score = result.get("score_curve")
    if not grid or not score:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    dm_ref = result["dm_ref"]
    x = np.asarray(grid, dtype=float) + dm_ref
    ax.plot(x, score, color="black", lw=1.5)
    if result.get("constrains_dm"):
        ax.axvline(result["dm"], color="tab:red", ls="--", label=f"DM={result['dm']:.3f}")
        ax.legend(loc="best", fontsize=8)
    elif result.get("dm_power_curve_max_dm") is not None:
        ax.axvline(
            result["dm_power_curve_max_dm"],
            color="0.45",
            ls=":",
            label=f"curve max={result['dm_power_curve_max_dm']:.3f}",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{result['burst']} {result['telescope']} DM-power")
    ax.set_xlabel("absolute DM [pc cm^-3]")
    ax.set_ylabel("DM-power structure metric")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / f"{result['burst']}_{result['telescope']}_dm_power.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    result["figure_path"] = str(path)


def _write_memo(
    results: list[dict[str, Any]],
    memo_path: Path,
    figure_dir: Path,
    output_json: Path,
) -> None:
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.get("constrains_dm")]
    bad = [r for r in results if not r.get("constrains_dm")]
    output_link = Path(os.path.relpath(output_json, memo_path.parent))
    figure_link = Path(os.path.relpath(figure_dir, memo_path.parent))
    lines = [
        "# DM-power fit memo",
        "",
        "Status: experimental cross-check, not promoted to citable DM_obs.",
        "",
        f"Results JSON: `{output_link}`",
        f"Figure directory: `{figure_link}`",
        "",
        f"Constrained fits: {len(ok)}/{len(results)}",
        "",
        "| Burst | Telescope | DM ref | Candidate DM | Candidate sigma | Accepted DM | sigma | Reason | Figure |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        if r.get("figure_path"):
            rel_fig = Path(os.path.relpath(r["figure_path"], memo_path.parent))
            fig = f"[curve]({rel_fig})"
        else:
            fig = ""
        candidate = (
            ""
            if r.get("dm_power_candidate_dm") is None
            else f"{r['dm_power_candidate_dm']:.4f}"
        )
        candidate_err = (
            ""
            if r.get("dm_power_candidate_err") is None
            else f"{r['dm_power_candidate_err']:.4f}"
        )
        dm = "" if r.get("dm") is None else f"{r['dm']:.4f}"
        err = "" if r.get("dm_err") is None else f"{r['dm_err']:.4f}"
        lines.append(
            f"| {r['burst']} | {r['telescope']} | {r['dm_ref']:.4f} | {candidate} | "
            f"{candidate_err} | {dm} | {err} | "
            f"{r['reason']} | {fig} |"
        )
    if bad:
        lines.extend(["", "## Non-constraining Fits", ""])
        for r in bad:
            lines.append(f"- {r['burst']} {r['telescope']}: {r['reason']}")
    lines.extend(["", "## Fit Figures", ""])
    for r in results:
        if not r.get("figure_path"):
            continue
        rel_fig = Path(os.path.relpath(r["figure_path"], memo_path.parent))
        lines.extend(
            [
                f"### {r['burst']} {r['telescope']}",
                "",
                f"![{r['burst']} {r['telescope']} DM-power curve]({rel_fig})",
                "",
            ]
        )
    lines.append("")
    memo_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DM-power residual fits for co-detections.")
    parser.add_argument("--chime-full-root", type=Path, default=CHIME_FULL_ROOT_DEFAULT)
    parser.add_argument("--dsa-full-root", type=Path, default=DSA_FULL_ROOT_DEFAULT)
    parser.add_argument("--output-json", type=Path, default=Path("results/dm_power_results.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/dm_power_figures"))
    parser.add_argument("--memo", type=Path, default=None)
    parser.add_argument("--dm-window", type=float, default=3.0)
    parser.add_argument("--dm-step", type=float, default=DEFAULT_DM_STEP)
    parser.add_argument("--n-boot", type=int, default=32)
    parser.add_argument("--max-channels", type=int, default=128)
    parser.add_argument("--max-time", type=int, default=2048)
    parser.add_argument("--telescope", choices=["chime", "dsa", "both"], default="both")
    parser.add_argument("--burst", action="append", help="Burst name to run; repeatable")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    memo_path = args.memo if args.memo is not None else root.parent / "docs/rse/specs/dm-power-fit-memo.md"
    telescopes = {"chime", "dsa"} if args.telescope == "both" else {args.telescope}
    run_all(
        root=root,
        chime_full_root=args.chime_full_root.expanduser(),
        dsa_full_root=args.dsa_full_root.expanduser(),
        output_json=args.output_json,
        figure_dir=args.figure_dir,
        memo_path=memo_path,
        dm_window=args.dm_window,
        dm_step=args.dm_step,
        n_boot=args.n_boot,
        max_channels=args.max_channels,
        max_time=args.max_time,
        telescopes=telescopes,
        bursts=set(args.burst) if args.burst else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
