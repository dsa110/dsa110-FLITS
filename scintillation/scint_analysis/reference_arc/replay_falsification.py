"""Matched-window falsification battery for the recovered Freya notebook.

This module deliberately preserves the numerical choices in
``reference_arc/notebooks/scint_freya.ipynb`` while separating the notebook's
hard-coded ``725:875`` window from the actual burst location in the surviving
pickle.  It is a diagnostic replay, not a production scintillation estimator.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from scipy.signal import correlate
from scipy.stats import chi2

LEGACY_WINDOW = (725, 875)
LEGACY_F_RES_KHZ = 30.51757812


def lorentz(x, gamma1, m1, c):
    """Notebook Lorentzian: ``gamma1`` is the reported HWHM."""
    return m1**2 / (1.0 + (x / gamma1) ** 2) + c


@dataclass
class WindowFit:
    label: str
    start: int
    stop: int
    contains_peak: bool
    gamma_khz: float | None
    gamma_err_khz: float | None
    amplitude_m: float | None
    baseline: float | None
    r_squared: float | None
    redchi: float | None
    white: bool
    whiteness_p: float
    max_abs_acf_z: float


def prepare_intensity(data: dict) -> tuple[np.ndarray, dict]:
    """Apply cells 2's crop and integer roll without changing its arithmetic."""
    intensity = np.asarray(data["I"], dtype=float)
    outerbound = int(intensity.shape[1] * 2 / 12)
    cropped = intensity[:, outerbound:-outerbound]
    tres = float(data["delta_t (ms)"])
    timeseries = np.nansum(cropped, axis=0)
    timesamples = np.linspace(0, cropped.shape[1] * tres, cropped.shape[1])
    timerange = abs(timesamples[-1] - timesamples[0])
    centered_time = np.linspace(-timerange / 2, timerange / 2, len(timesamples))
    pre_roll_peak = int(np.nanargmax(timeseries))
    shift = -int(centered_time[pre_roll_peak] / tres)
    rolled = np.roll(cropped, shift, axis=1)
    post_roll_timeseries = np.nansum(rolled, axis=0)
    post_roll_peak = int(np.nanargmax(post_roll_timeseries))
    return rolled, {
        "raw_shape": list(intensity.shape),
        "cropped_shape": list(cropped.shape),
        "delta_t_ms": tres,
        "pickle_delta_f_mhz": float(data["delta_f (MHz)"]),
        "legacy_fit_delta_f_mhz": LEGACY_F_RES_KHZ / 1e3,
        "pre_roll_peak": pre_roll_peak,
        "notebook_shift": shift,
        "post_roll_peak": post_roll_peak,
    }


def legacy_positive_acf(
    spectrum: np.ndarray, max_lag_channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the old helper's positive-lag ACF from lag 2 onward.

    The archived helper centers the full spectrum once, then normalizes every
    overlapping lag by the two overlap energies.  FFT correlation plus prefix
    sums is algebraically equivalent but avoids its repeated 3N allocations.
    """
    x = np.asarray(spectrum, dtype=float).copy()
    finite = np.isfinite(x)
    if not finite.all():
        replacement = float(np.nanmedian(x)) if finite.any() else 0.0
        x[~finite] = replacement
    x -= x.mean()
    n = x.size
    max_k = min(int(max_lag_channels), n - 1)
    corr = correlate(x, x, mode="full", method="fft")
    prefix = np.concatenate(([0.0], np.cumsum(x * x)))
    ks = np.arange(2, max_k + 1, dtype=int)
    numerator = corr[n - 1 + ks]
    left_energy = prefix[n - ks]
    right_energy = prefix[n] - prefix[ks]
    denominator = np.sqrt(left_energy * right_energy)
    rho = np.divide(
        numerator,
        denominator,
        out=np.full(ks.shape, np.nan, dtype=float),
        where=denominator > 0,
    )
    return ks, rho


def whiteness_test(rho: np.ndarray, n_channels: int, *, n_lags: int = 64) -> dict:
    """Portmanteau plus maximum-z whiteness gate for an off-pulse spectrum."""
    values = np.asarray(rho[:n_lags], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return {"pass": False, "p_value": 0.0, "max_abs_z": float("inf")}
    lag_numbers = np.arange(2, 2 + values.size, dtype=float)
    q = n_channels * (n_channels + 2.0) * np.sum(values**2 / (n_channels - lag_numbers))
    p_value = float(chi2.sf(q, values.size))
    max_abs_z = float(np.max(np.abs(values)) * np.sqrt(n_channels))
    return {
        "pass": bool(p_value >= 0.01 and max_abs_z <= 4.5),
        "p_value": p_value,
        "max_abs_z": max_abs_z,
    }


def fit_spectrum(
    spectrum: np.ndarray,
    *,
    fit_lag_khz: float = 2000.0,
    first_lag_channels: int = 2,
    max_lag_khz: float = 10000.0,
) -> dict:
    """Run the notebook's normalized-spectrum ACF and unconstrained lmfit."""
    spectrum = np.asarray(spectrum, dtype=float)
    scale = float(np.nanmax(spectrum))
    spec_norm = spectrum / scale if np.isfinite(scale) and scale != 0 else spectrum
    max_channels = int(np.ceil(max_lag_khz / LEGACY_F_RES_KHZ))
    ks, rho = legacy_positive_acf(spec_norm, max_channels)
    lags = np.concatenate((-ks[::-1], ks)) * LEGACY_F_RES_KHZ
    acf = np.concatenate((rho[::-1], rho))
    # The notebook selects ``int(fit_lag / f_res)`` samples on each side of
    # the concatenated ACF's midpoint.  Because its first retained lag is
    # channel 2, a nominal 2000-kHz window includes channels 2..66 (130 data
    # points), not only the channels whose coordinates are <= 2000 kHz.
    notebook_last_lag_channel = int(fit_lag_khz / LEGACY_F_RES_KHZ) + 1
    keep = (
        (np.abs(lags) >= first_lag_channels * LEGACY_F_RES_KHZ)
        & (np.abs(lags) <= notebook_last_lag_channel * LEGACY_F_RES_KHZ)
        & np.isfinite(acf)
    )
    result = Model(lorentz).fit(acf[keep], x=lags[keep], gamma1=1, m1=1, c=0)
    gamma = result.params["gamma1"]
    white = whiteness_test(rho, spectrum.size)
    return {
        "gamma_khz": float(gamma.value),
        "gamma_err_khz": None if gamma.stderr is None else float(gamma.stderr),
        "amplitude_m": float(result.params["m1"].value),
        "baseline": float(result.params["c"].value),
        "r_squared": float(result.rsquared),
        "redchi": float(result.redchi),
        "lags_khz": lags,
        "acf": acf,
        "model": lorentz(
            lags,
            result.params["gamma1"],
            result.params["m1"],
            result.params["c"],
        ),
        "white": white,
    }


def fit_window(
    intensity: np.ndarray,
    peak: int,
    label: str,
    start: int,
    stop: int,
    **fit_kwargs,
) -> tuple[WindowFit, dict]:
    spectrum = np.nansum(intensity[:, start:stop], axis=1)
    fit = fit_spectrum(spectrum, **fit_kwargs)
    record = WindowFit(
        label=label,
        start=int(start),
        stop=int(stop),
        contains_peak=bool(start <= peak < stop),
        gamma_khz=fit["gamma_khz"],
        gamma_err_khz=fit["gamma_err_khz"],
        amplitude_m=fit["amplitude_m"],
        baseline=fit["baseline"],
        r_squared=fit["r_squared"],
        redchi=fit["redchi"],
        white=bool(fit["white"]["pass"]),
        whiteness_p=float(fit["white"]["p_value"]),
        max_abs_acf_z=float(fit["white"]["max_abs_z"]),
    )
    return record, fit


def matched_off_windows(
    n_time: int, on_window: tuple[int, int], *, max_windows: int = 24
) -> list[tuple[int, int]]:
    width = on_window[1] - on_window[0]
    exclusion = (on_window[0] - 2 * width, on_window[1] + 2 * width)
    candidates = []
    for start in range(0, n_time - width + 1, width + 25):
        stop = start + width
        if stop <= exclusion[0] or start >= exclusion[1]:
            candidates.append((start, stop))
    if len(candidates) <= max_windows:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, max_windows).round().astype(int)
    return [candidates[i] for i in np.unique(indices)]


def _ratio(values: list[float]) -> float | None:
    finite = [abs(float(v)) for v in values if v is not None and np.isfinite(v) and v != 0]
    return max(finite) / min(finite) if len(finite) >= 2 else None


def run_battery(pickle_path: Path, output_dir: Path) -> dict:
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    intensity, provenance = prepare_intensity(data)
    peak = int(provenance["post_roll_peak"])
    width = LEGACY_WINDOW[1] - LEGACY_WINDOW[0]
    on_window = (peak - width // 2, peak + width // 2)

    legacy_record, legacy_fit = fit_window(intensity, peak, "legacy_725_875", *LEGACY_WINDOW)
    on_record, on_fit = fit_window(intensity, peak, "actual_centered_burst", *on_window)

    off_records: list[WindowFit] = []
    off_fits: list[dict] = []
    for index, (start, stop) in enumerate(matched_off_windows(intensity.shape[1], on_window)):
        record, fit = fit_window(intensity, peak, f"off_{index:02d}", start, stop)
        off_records.append(record)
        off_fits.append(fit)

    shifts = [-300, -150, -75, 75, 150, 300]
    shifted_records = []
    for offset in shifts:
        start, stop = on_window[0] + offset, on_window[1] + offset
        record, _ = fit_window(intensity, peak, f"shift_{offset:+d}", start, stop)
        shifted_records.append(record)

    fit_windows_khz = [1000.0, 1500.0, 2000.0, 3000.0, 4000.0]
    fit_window_records = []
    on_spectrum = np.nansum(intensity[:, on_window[0] : on_window[1]], axis=1)
    for fit_lag in fit_windows_khz:
        fit = fit_spectrum(on_spectrum, fit_lag_khz=fit_lag)
        fit_window_records.append(
            {
                "fit_lag_khz": fit_lag,
                "gamma_khz": fit["gamma_khz"],
                "gamma_err_khz": fit["gamma_err_khz"],
                "r_squared": fit["r_squared"],
            }
        )

    first_lags = [2, 3, 4, 6, 8]
    low_lag_records = []
    for first_lag in first_lags:
        fit = fit_spectrum(on_spectrum, first_lag_channels=first_lag)
        low_lag_records.append(
            {
                "first_lag_channels": first_lag,
                "gamma_khz": fit["gamma_khz"],
                "gamma_err_khz": fit["gamma_err_khz"],
                "r_squared": fit["r_squared"],
            }
        )

    split_records = []
    midpoint = intensity.shape[0] // 2
    for label, channel_slice in (
        ("lower_channels", (0, midpoint)),
        ("upper_channels", (midpoint, intensity.shape[0])),
    ):
        start, stop = channel_slice
        spectrum = np.nansum(intensity[start:stop, on_window[0] : on_window[1]], axis=1)
        fit = fit_spectrum(spectrum)
        split_records.append(
            {
                "label": label,
                "channel_slice": [start, stop],
                "gamma_khz": fit["gamma_khz"],
                "gamma_err_khz": fit["gamma_err_khz"],
                "r_squared": fit["r_squared"],
                "white": fit["white"],
            }
        )

    off_widths = [r.gamma_khz for r in off_records if r.gamma_khz is not None and r.gamma_khz > 0]
    off_median = float(np.median(off_widths)) if off_widths else None
    on_off_ratio = (
        max(abs(on_record.gamma_khz) / off_median, off_median / abs(on_record.gamma_khz))
        if off_median and on_record.gamma_khz and on_record.gamma_khz != 0
        else None
    )
    off_white_fraction = float(np.mean([r.white for r in off_records])) if off_records else 0.0
    fit_window_ratio = _ratio([r["gamma_khz"] for r in fit_window_records])
    low_lag_ratio = _ratio([r["gamma_khz"] for r in low_lag_records])
    split_ratio = _ratio([r["gamma_khz"] for r in split_records])

    checks = {
        "legacy_window_contains_burst": {
            "pass": legacy_record.contains_peak,
            "reason": (
                "hard-coded notebook window contains the centered burst"
                if legacy_record.contains_peak
                else "hard-coded notebook window is off-pulse in the surviving capture"
            ),
        },
        "off_pulse_whiteness": {
            "pass": off_white_fraction >= 0.90,
            "white_fraction": off_white_fraction,
            "n_windows": len(off_records),
        },
        "on_off_width_separation": {
            "pass": on_off_ratio is not None and on_off_ratio > 2.0,
            "on_gamma_khz": on_record.gamma_khz,
            "off_median_gamma_khz": off_median,
            "ratio": on_off_ratio,
        },
        "fit_window_stability": {
            "pass": fit_window_ratio is not None and fit_window_ratio <= 2.0,
            "max_to_min_ratio": fit_window_ratio,
        },
        "low_lag_stability": {
            "pass": low_lag_ratio is not None and low_lag_ratio <= 2.0,
            "max_to_min_ratio": low_lag_ratio,
        },
        "split_band_stability": {
            "pass": split_ratio is not None and split_ratio <= 2.0,
            "max_to_min_ratio": split_ratio,
        },
    }
    failed = [name for name, value in checks.items() if value["pass"] is not True]
    result = {
        "status": "candidate_pass" if not failed else "falsified",
        "failed_checks": failed,
        "provenance": provenance,
        "legacy_window": asdict(legacy_record),
        "actual_on_window": asdict(on_record),
        "off_windows": [asdict(r) for r in off_records],
        "shifted_windows": [asdict(r) for r in shifted_records],
        "fit_window_scan": fit_window_records,
        "low_lag_scan": low_lag_records,
        "split_band_scan": split_records,
        "checks": checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "matched_window_falsification.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    timeseries = np.nansum(intensity, axis=0)
    axes[0, 0].plot(timeseries, color="black", lw=0.8)
    axes[0, 0].axvspan(*LEGACY_WINDOW, color="tab:orange", alpha=0.35, label="legacy 725:875")
    axes[0, 0].axvspan(*on_window, color="tab:blue", alpha=0.25, label="actual centered burst")
    axes[0, 0].axvline(peak, color="tab:blue", ls="--", lw=1)
    axes[0, 0].set(
        title="Surviving capture after notebook roll", xlabel="Time bin", ylabel="Summed intensity"
    )
    axes[0, 0].legend()

    axes[0, 1].plot(
        legacy_fit["lags_khz"],
        legacy_fit["acf"],
        color="tab:orange",
        lw=1,
        label="legacy window ACF",
    )
    axes[0, 1].plot(
        legacy_fit["lags_khz"],
        legacy_fit["model"],
        color="tab:red",
        lw=2,
        label=f"legacy fit {legacy_record.gamma_khz:.1f} kHz",
    )
    axes[0, 1].plot(
        on_fit["lags_khz"], on_fit["acf"], color="black", lw=1, label="actual burst ACF"
    )
    axes[0, 1].plot(
        on_fit["lags_khz"],
        on_fit["model"],
        color="tab:blue",
        lw=2,
        label=f"actual fit {on_record.gamma_khz:.1f} kHz",
    )
    axes[0, 1].set(
        xlim=(-2000, 2000),
        title="Notebook-parity ACF fits",
        xlabel="Frequency lag (kHz)",
        ylabel="ACF",
    )
    axes[0, 1].legend(fontsize=8)

    starts = [r.start for r in off_records]
    widths = [r.gamma_khz for r in off_records]
    colors = ["tab:green" if r.white else "tab:red" for r in off_records]
    axes[1, 0].scatter(starts, widths, c=colors, label="matched off-pulse")
    axes[1, 0].scatter(
        [LEGACY_WINDOW[0]],
        [legacy_record.gamma_khz],
        marker="*",
        s=180,
        color="tab:orange",
        label="legacy",
    )
    axes[1, 0].scatter(
        [on_window[0]],
        [on_record.gamma_khz],
        marker="*",
        s=180,
        color="tab:blue",
        label="actual burst",
    )
    axes[1, 0].set(
        title="Matched-window fitted widths", xlabel="Window start bin", ylabel="gamma (kHz)"
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        [r["fit_lag_khz"] for r in fit_window_records],
        [r["gamma_khz"] for r in fit_window_records],
        "o-",
        label="fit-window scan",
    )
    axes[1, 1].plot(
        [r["first_lag_channels"] * LEGACY_F_RES_KHZ for r in low_lag_records],
        [r["gamma_khz"] for r in low_lag_records],
        "s-",
        label="low-lag scan",
    )
    axes[1, 1].axhline(abs(on_record.gamma_khz), color="black", ls="--", lw=1)
    axes[1, 1].set(
        title="Actual-burst stability",
        xlabel="Fit limit or first included lag (kHz)",
        ylabel="gamma (kHz)",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"Freya recovered-notebook falsification: {result['status']}")
    fig.tight_layout()
    fig.savefig(output_dir / "matched_window_falsification.png", dpi=160)
    plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = run_battery(args.pickle_path, args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "failed_checks": result["failed_checks"],
                "legacy_window": result["legacy_window"],
                "actual_on_window": result["actual_on_window"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
