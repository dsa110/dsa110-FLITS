#!/usr/bin/env python3
"""Injection/null qualification of the clean low-frequency Oran DSA subband."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from scintillation.scint_analysis import analysis  # noqa: E402
from scintillation.scint_analysis import config as config_mod  # noqa: E402
from scintillation.scint_analysis.pipeline import ScintillationAnalysis  # noqa: E402

DRIVER_PATH = Path(__file__).with_name("run_dsa_lorentzian_fits.py")
SPEC = importlib.util.spec_from_file_location("dsa_lorentzian_driver", DRIVER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {DRIVER_PATH}")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

BURST = "oran"
SUBBAND_INDEX = 0
NUM_SUBBANDS = 4
WIDTHS_MHZ = (0.05, 0.10, 0.15, 0.16, 0.18, 0.20, 0.40, 0.60, 0.65, 0.70, 0.75, 0.80, 1.20)
MODULATION = 0.83
N_TRIALS = 64
CENTRAL_TRIALS = 256
FIT_RANGE_MHZ = 12.0


def _trial_count(width_mhz: float) -> int:
    return CENTRAL_TRIALS if np.isclose(width_mhz, 0.40) else N_TRIALS


def _monotonic_nonincreasing(values: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def _invert_decreasing_grid(truths: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    monotonic = _monotonic_nonincreasing(probabilities)
    return float(np.interp(target, monotonic[::-1], truths[::-1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stationary_lorentzian(
    rng: np.random.Generator, n_channels: int, width_channels: float
) -> np.ndarray:
    distance = np.minimum(np.arange(n_channels), n_channels - np.arange(n_channels))
    covariance = 1.0 / (1.0 + (distance / width_channels) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    sample = np.real(np.fft.ifft(np.fft.fft(rng.normal(size=n_channels)) * np.sqrt(power)))
    return (sample - sample.mean()) / sample.std()


def _fit_spectrum(
    spectrum: np.ma.MaskedArray,
    channel_width_mhz: float,
    off_mean: float | None,
    fit_range_mhz: float,
    first_lag_bin: int = 1,
) -> dict | None:
    acf = analysis.calculate_acf(
        spectrum,
        channel_width_mhz,
        off_burst_spectrum_mean=off_mean,
        max_lag_bins=int(fit_range_mhz / channel_width_mhz) + 1,
    )
    if acf is None:
        return None
    lags = np.asarray(acf.lags, dtype=float)
    values = np.asarray(acf.acf, dtype=float)
    errors = None if acf.err is None else np.asarray(acf.err, dtype=float)
    keep = (
        np.isfinite(lags)
        & np.isfinite(values)
        & (lags >= first_lag_bin * channel_width_mhz - 1e-12)
        & (lags <= fit_range_mhz)
    )
    if errors is not None:
        keep &= np.isfinite(errors) & (errors > 0)
    x = lags[keep]
    y = values[keep]
    sigma = np.ones_like(y) if errors is None else errors[keep]

    def model(parameters: np.ndarray) -> np.ndarray:
        width, modulation, constant = parameters
        return modulation**2 / (1.0 + (x / width) ** 2) + constant

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (y - model(parameters)) / sigma

    candidates = []
    for width_start in (0.08, 0.15, 0.30, 0.60, 1.20, 1.80):
        fit = least_squares(
            residual,
            x0=(width_start, 0.8, 0.0),
            bounds=((0.5 * channel_width_mhz, 0.0, -5.0), (2.0, 100.0, 5.0)),
            max_nfev=5000,
        )
        if fit.success and np.all(np.isfinite(fit.x)):
            candidates.append(fit)
    if not candidates:
        return None
    fit = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    dof = max(y.size - fit.x.size, 1)
    redchi = float(np.sum(fit.fun**2) / dof)
    try:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * max(1.0, redchi)
        parameter_error = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        parameter_error = np.full(3, np.nan)
    return {
        "dnu_mhz": float(fit.x[0]),
        "dnu_err": float(parameter_error[0]),
        "m": float(fit.x[1]),
        "m_err": float(parameter_error[1]),
        "constant": float(fit.x[2]),
        "redchi": redchi,
        "n_selected": 1,
        "width_upper_bound_mhz": 2.0,
    }


def _fixed_width_amplitude(
    spectrum: np.ma.MaskedArray,
    channel_width_mhz: float,
    off_mean: float,
    fit_range_mhz: float,
    width_mhz: float,
) -> dict:
    """Weighted matched-template amplitude at a fixed Lorentzian width."""
    acf = analysis.calculate_acf(
        spectrum,
        channel_width_mhz,
        off_burst_spectrum_mean=off_mean,
        max_lag_bins=int(fit_range_mhz / channel_width_mhz) + 1,
    )
    if acf is None or acf.err is None:
        raise ValueError("matched-template ACF or errors missing")
    lags = np.asarray(acf.lags, dtype=float)
    values = np.asarray(acf.acf, dtype=float)
    errors = np.asarray(acf.err, dtype=float)
    keep = (
        np.isfinite(lags)
        & np.isfinite(values)
        & np.isfinite(errors)
        & (errors > 0)
        & (lags > 0)
        & (lags <= fit_range_mhz)
    )
    template = 1.0 / (1.0 + (lags[keep] / width_mhz) ** 2)
    design = np.column_stack([template, np.ones(np.count_nonzero(keep))])
    weights = 1.0 / errors[keep] ** 2
    covariance = np.linalg.inv((design.T * weights) @ design)
    parameters = covariance @ ((design.T * weights) @ values[keep])
    return {
        "amplitude": float(parameters[0]),
        "amplitude_err": float(np.sqrt(covariance[0, 0])),
        "z": float(parameters[0] / np.sqrt(covariance[0, 0])),
        "constant": float(parameters[1]),
    }


def _prepare(flits_root: Path, output: Path):
    os.environ["FLITS_ROOT"] = str(flits_root.resolve())
    config_path = Path(f"scintillation/configs/bursts/{BURST}_dsa.yaml")
    cfg = driver._config_for_fresh_acf(config_mod.load_config(config_path), output_dir=output)
    cfg = driver._config_with_subband_count(cfg, NUM_SUBBANDS)
    cfg = copy.deepcopy(cfg)
    cfg["analysis"]["noise"]["disable_template"] = True
    pipe = ScintillationAnalysis(cfg)
    pipe.run()
    return config_path, cfg, pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flits-root",
        type=Path,
        default=Path.home() / "Data/Faber2026/dsa110",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/oran-dsa-validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path, cfg, pipe = _prepare(args.flits_root, args.output_dir)
    acfs = pipe.acf_results
    if not acfs or pipe.masked_spectrum is None:
        raise RuntimeError("prepared spectrum or ACFs missing")
    channel_slice = tuple(acfs["subband_channel_slices"][SUBBAND_INDEX])
    c0, c1 = channel_slice
    channel_width = float(acfs["subband_channel_widths_mhz"][SUBBAND_INDEX])
    center_frequency = float(acfs["subband_center_freqs_mhz"][SUBBAND_INDEX])
    fit_range = min(FIT_RANGE_MHZ, (c1 - c0) * channel_width / 2.0)
    burst_lims = tuple(pipe.burst_lims)
    off_lims = tuple(pipe.off_pulse_lims)
    duration = int(burst_lims[1] - burst_lims[0])
    starts = list(range(int(off_lims[0]) + 2, int(off_lims[1]) - duration, duration + 4))
    starts = starts[:12]
    if len(starts) < 8:
        raise RuntimeError("insufficient independent off-pulse windows")

    dynamic = pipe.masked_spectrum
    burst_spectrum = dynamic.get_spectrum(burst_lims)[c0:c1]
    if pipe.noise_descriptor is not None:
        off_mean = float(
            pipe.noise_descriptor.mu if pipe.noise_descriptor.kind == "intensity" else 0.0
        )
    else:
        normalization_end = max(int(burst_lims[0]) - 1, 0)
        normalization_start = max(normalization_end - duration, 0)
        normalization_spectrum = dynamic.get_spectrum((normalization_start, normalization_end))[
            c0:c1
        ]
        off_mean = float(np.ma.mean(normalization_spectrum))
    signal_mean = float(np.ma.mean(burst_spectrum) - off_mean)
    if not np.isfinite(signal_mean) or signal_mean <= 0:
        raise RuntimeError("non-positive burst signal normalization")

    real_fit = _fit_spectrum(burst_spectrum, channel_width, off_mean, fit_range)
    fit_window_records = {
        f"{window:g}": _fit_spectrum(
            burst_spectrum,
            channel_width,
            off_mean,
            min(window, (c1 - c0) * channel_width / 2.0),
        )
        for window in (8.0, 12.0, 18.0, 25.0)
    }
    records = []
    cells = []
    for width in WIDTHS_MHZ:
        cell_records = []
        n_trials = _trial_count(width)
        for trial in range(n_trials):
            seed = 20260714 + int(round(width * 1000)) * 1000 + trial
            rng = np.random.default_rng(seed)
            start = starts[trial % len(starts)]
            background = dynamic.get_spectrum((start, start + duration))[c0:c1]
            scintle = _stationary_lorentzian(rng, len(background), width / channel_width)
            signal = signal_mean * (1.0 + MODULATION * scintle)
            truth_spectrum = np.ma.array(
                off_mean + signal,
                mask=np.ma.getmaskarray(background),
            )
            injected = np.ma.array(
                background.data + signal,
                mask=np.ma.getmaskarray(background),
            )
            truth_fit = _fit_spectrum(truth_spectrum, channel_width, off_mean, fit_range)
            fit = _fit_spectrum(injected, channel_width, off_mean, fit_range)
            record = {
                "nominal_truth_mhz": width,
                "trial": trial,
                "seed": seed,
                "truth_fit": truth_fit,
                "fit": fit,
            }
            records.append(record)
            cell_records.append(record)
        finite = [
            item
            for item in cell_records
            if item["fit"] is not None and item["truth_fit"] is not None
        ]
        realized_truth = np.asarray([item["truth_fit"]["dnu_mhz"] for item in finite])
        recovered = np.asarray([item["fit"]["dnu_mhz"] for item in finite])
        errors = np.asarray([item["fit"]["dnu_err"] for item in finite])
        recovered_m = np.asarray([item["fit"]["m"] for item in finite])
        generator_bias = (
            abs(float(np.median(realized_truth)) - width) if realized_truth.size else np.inf
        )
        bias = abs(float(np.median(recovered)) - width) if recovered.size else np.inf
        coverage = (
            float(np.mean(np.abs(recovered - realized_truth) <= errors)) if recovered.size else 0.0
        )
        m_bias = float(np.median(np.abs(recovered_m - MODULATION))) if recovered_m.size else np.inf
        bias_limit = max(0.10 * width, 0.25 * channel_width)
        m_limit = max(0.10 * MODULATION, 0.05)
        cell_pass = bool(
            len(finite) >= 0.95 * n_trials
            and generator_bias < 0.10 * width
            and bias < bias_limit
            and np.percentile(recovered, 16) <= width <= np.percentile(recovered, 84)
            and m_bias < m_limit
        )
        cells.append(
            {
                "truth_mhz": width,
                "modulation": MODULATION,
                "n_finite": len(finite),
                "n_trials": n_trials,
                "absolute_ensemble_median_generator_bias_mhz": generator_bias,
                "generator_bias_limit_mhz": 0.10 * width,
                "absolute_ensemble_median_recovery_bias_mhz": bias,
                "bias_limit_mhz": bias_limit,
                "coverage_68": coverage,
                "empirical_recovery_interval_68_mhz": np.percentile(recovered, [16, 84]).tolist(),
                "recovered_median_mhz": float(np.median(recovered)),
                "observed_fit_cdf": float(np.mean(recovered <= real_fit["dnu_mhz"])),
                "median_absolute_modulation_bias": m_bias,
                "modulation_bias_limit": m_limit,
                "pass": cell_pass,
            }
        )

    onpulse_template = _fixed_width_amplitude(
        burst_spectrum,
        channel_width,
        off_mean,
        fit_range,
        real_fit["dnu_mhz"],
    )
    offpulse_templates = []
    for start in starts:
        off_spectrum = dynamic.get_spectrum((start, start + duration))[c0:c1]
        reference = float(np.ma.mean(off_spectrum) - signal_mean)
        offpulse_templates.append(
            _fixed_width_amplitude(
                off_spectrum,
                channel_width,
                reference,
                fit_range,
                real_fit["dnu_mhz"],
            )
        )
    offpulse_amplitudes = np.asarray([item["amplitude"] for item in offpulse_templates])
    offpulse_mean = float(np.mean(offpulse_amplitudes))
    offpulse_sem = float(np.std(offpulse_amplitudes, ddof=1) / np.sqrt(len(offpulse_amplitudes)))
    amplitude_ratio = float(onpulse_template["amplitude"] / np.max(np.abs(offpulse_amplitudes)))

    lag_excision_records = {
        str(first - 1): _fit_spectrum(
            burst_spectrum,
            channel_width,
            off_mean,
            fit_range,
            first_lag_bin=first,
        )
        for first in (2, 3, 4)
    }
    window_widths = np.asarray(
        [item["dnu_mhz"] for item in fit_window_records.values() if item is not None]
    )
    window_movement = float((window_widths.max() - window_widths.min()) / np.median(window_widths))
    lag_widths = np.asarray(
        [item["dnu_mhz"] for item in lag_excision_records.values() if item is not None]
    )
    lag_ratios = lag_widths / real_fit["dnu_mhz"]

    truths = np.asarray([cell["truth_mhz"] for cell in cells])
    response = np.asarray([cell["recovered_median_mhz"] for cell in cells])
    response_monotonic = np.maximum.accumulate(response)
    response_unique, unique_indices = np.unique(response_monotonic, return_index=True)
    calibrated_point = float(
        np.interp(real_fit["dnu_mhz"], response_unique, truths[unique_indices])
    )
    observed_cdf = np.asarray([cell["observed_fit_cdf"] for cell in cells])
    interval_68 = [
        _invert_decreasing_grid(truths, observed_cdf, 0.84),
        _invert_decreasing_grid(truths, observed_cdf, 0.16),
    ]
    response_spearman = float(spearmanr(truths, response).statistic)
    central_cell = next(cell for cell in cells if np.isclose(cell["truth_mhz"], 0.40))

    gates = {
        "independent_offpulse_null": {
            "pass": bool(
                len(offpulse_templates) == len(starts)
                and onpulse_template["z"] >= 5.0
                and np.max(np.abs([item["z"] for item in offpulse_templates])) <= 3.0
                and abs(offpulse_mean / offpulse_sem) <= 3.0
                and amplitude_ratio >= 5.0
            ),
            "onpulse": onpulse_template,
            "offpulse": offpulse_templates,
            "offpulse_mean_amplitude": offpulse_mean,
            "offpulse_sem_amplitude": offpulse_sem,
            "on_to_max_abs_off_amplitude_ratio": amplitude_ratio,
            "thresholds": {
                "minimum_onpulse_z": 5.0,
                "maximum_abs_offpulse_z": 3.0,
                "maximum_abs_offpulse_mean_z": 3.0,
                "minimum_on_to_max_abs_off_amplitude_ratio": 5.0,
            },
        },
        "fit_window_stability": {
            "pass": bool(len(window_widths) == 4 and window_movement < 0.20),
            "fractional_movement": window_movement,
            "maximum_fractional_movement": 0.20,
            "fits": fit_window_records,
        },
        "low_lag_stability": {
            "pass": bool(
                len(lag_widths) == 3 and np.all((lag_ratios >= 0.5) & (lag_ratios <= 2.0))
            ),
            "ratios_to_full": lag_ratios.tolist(),
            "fits": lag_excision_records,
        },
        "simulation_calibration": {
            "pass": bool(
                response_spearman >= 0.97
                and central_cell["n_finite"] == central_cell["n_trials"]
                and central_cell["median_absolute_modulation_bias"]
                < central_cell["modulation_bias_limit"]
            ),
            "raw_response_spearman": response_spearman,
            "truth_grid_mhz": truths.tolist(),
            "raw_median_response_mhz": response.tolist(),
            "monotonic_median_response_mhz": response_monotonic.tolist(),
            "central_cell": central_cell,
        },
        "calibrated_interval": {
            "pass": bool(
                interval_68[0] < calibrated_point < interval_68[1]
                and interval_68[0] > 4.0 * channel_width
                and interval_68[1] < 2.0
            ),
            "confidence": 0.68,
            "interval_mhz": interval_68,
            "minimum_resolved_channels": 4.0,
        },
    }

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    recovery_low = np.asarray([cell["empirical_recovery_interval_68_mhz"][0] for cell in cells])
    recovery_high = np.asarray([cell["empirical_recovery_interval_68_mhz"][1] for cell in cells])
    axes[0].errorbar(
        truths,
        response,
        yerr=[response - recovery_low, recovery_high - response],
        fmt="o",
        capsize=2,
        label="real-background recovery",
    )
    axes[0].plot(truths, truths, "k--", lw=1, label="identity")
    axes[0].axhline(real_fit["dnu_mhz"], color="tab:red", label="observed raw fit")
    axes[0].axvspan(interval_68[0], interval_68[1], color="tab:red", alpha=0.12)
    axes[0].set(xlabel="Injected bandwidth (MHz)", ylabel="Recovered bandwidth (MHz)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)
    window_x = np.asarray([float(key) for key in fit_window_records])
    window_y = np.asarray([fit_window_records[key]["dnu_mhz"] for key in fit_window_records])
    axes[1].plot(window_x, window_y, "o-", label="on-pulse fit")
    axes[1].axhspan(
        interval_68[0], interval_68[1], color="tab:red", alpha=0.12, label="68% calibrated CI"
    )
    axes[1].set(xlabel="Fit-window maximum lag (MHz)", ylabel="Bandwidth (MHz)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Oran DSA 1328 MHz scintillation qualification")
    figure_path = figure_dir / "oran_dsa_calibrated_measurement.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    source_path = Path(cfg["input_data_path"])
    result = {
        "experiment": "Oran DSA low-subband real-background injection qualification",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "subband_index": SUBBAND_INDEX,
        "num_subbands": NUM_SUBBANDS,
        "channel_slice": list(channel_slice),
        "center_frequency_mhz": center_frequency,
        "channel_width_mhz": channel_width,
        "burst_window": list(burst_lims),
        "offpulse_windows": [[start, start + duration] for start in starts],
        "signal_mean": signal_mean,
        "real_fit": real_fit,
        "calibrated_measurement": {
            "dnu_mhz": calibrated_point,
            "confidence_interval_68_mhz": interval_68,
            "raw_fit_mhz": real_fit["dnu_mhz"],
            "definition": "Lorentzian HWHM decorrelation bandwidth",
        },
        "fit_window_records": fit_window_records,
        "onpulse_matched_template": onpulse_template,
        "offpulse_matched_templates": offpulse_templates,
        "lag_excision_records": lag_excision_records,
        "gates": gates,
        "injection_cells": cells,
        "figure": str(figure_path),
        "records": records,
    }
    result["machine_status"] = (
        "pass" if all(gate["pass"] for gate in gates.values()) else "documented_fail"
    )
    (args.output_dir / "validation.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "machine_status": result["machine_status"],
                "measurement": result["calibrated_measurement"],
                "gates": {name: gate["pass"] for name, gate in gates.items()},
            }
        )
    )
    return 0 if result["machine_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
