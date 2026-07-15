#!/usr/bin/env python3
"""Run and render the remaining fail-closed Freya H2 validation battery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scintillation.scint_analysis.chime_product import (  # noqa: E402
    ChimeProductConfig,
    build_chime_products,
)

DRIVER = ROOT / "analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py"
BASE_CONFIG = ROOT / "scintillation/configs/bursts/freya_chime.yaml"
DEFAULT_DATA = Path.home() / "Data/Faber2026/dsa110/scintillation-data"
CHANNEL_WIDTH_MHZ = 0.006103608758678547


def _figure_record(output_dir: Path, path: Path, expectation: str) -> dict[str, str]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expectation": expectation,
    }


def _driver_module():
    spec = importlib.util.spec_from_file_location("freya_h2_driver", DRIVER)
    driver = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(driver)
    return driver


def _prepared_config(driver, product: Path, manifest: Path, output_dir: Path):
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["input_data_path"] = str(product.resolve())
    analysis = config.setdefault("analysis", {})
    analysis["bandpass_normalization"] = {"enable": True}
    analysis["instrumental_background_correction"] = {
        "enable": True,
        "manifest_path": str(manifest.resolve()),
        "validation": {},
    }
    config = driver._config_for_fresh_acf(config, output_dir=output_dir)
    return driver._config_with_subband_count(config, 2)


def _fit_component(driver, lags, acf, err, *, fit_range, channel_width, harmonic_cfg):
    lags, acf, err = driver._slice_fit_window(lags, acf, err, fit_range)
    lags, acf, err = driver._select_physical_fit_lags(
        lags,
        acf,
        err,
        channel_width_mhz=channel_width,
        telescope="chime",
    )
    lags, acf, err, _ = driver.guards.apply_harmonic_mask_to_fit(lags, acf, err, harmonic_cfg)
    verdict = driver.compare_lorentzian_components(lags, acf, max_components=3, acf_err=err)
    fit = driver._selected_fit(verdict)
    components = sorted(fit.get("components", []), key=lambda item: item.get("dnu_mhz", np.inf))
    if not fit.get("success") or not components:
        return None
    component = components[0]
    return {
        "dnu_mhz": float(component["dnu_mhz"]),
        "dnu_err_mhz": float(component.get("dnu_err", np.nan)),
        "m": float(component.get("m", np.nan)),
        "redchi": float(fit.get("redchi", np.nan)),
    }


def _fit_window_check(driver, plot_subbands, harmonic_cfg):
    records = []
    passed = True
    for payload in plot_subbands:
        summary = payload["summary"]
        reference = float(summary["selected_components"][0]["dnu_mhz"])
        variants = {}
        for fit_range in (0.5, 0.75, 1.0):
            fit = _fit_component(
                driver,
                np.asarray(payload["lags"]),
                np.asarray(payload["acf"]),
                None if payload["err"] is None else np.asarray(payload["err"]),
                fit_range=fit_range,
                channel_width=float(summary["channel_width_mhz"]),
                harmonic_cfg=harmonic_cfg,
            )
            variants[str(fit_range)] = fit
        shifts = [
            abs(item["dnu_mhz"] - reference) / reference
            for item in variants.values()
            if item is not None
        ]
        item_pass = len(shifts) == 3 and max(shifts) <= 0.30
        passed &= item_pass
        records.append(
            {
                "subband": int(summary["index"]),
                "reference_dnu_mhz": reference,
                "variants": variants,
                "max_fractional_shift": max(shifts) if shifts else None,
                "pass": item_pass,
            }
        )
    return {"pass": bool(passed), "threshold_fractional_shift": 0.30, "subbands": records}


def _comb_residual_check(driver, plot_subbands, harmonic_cfg):
    records = []
    passed = True
    spacing = float(harmonic_cfg["spacing_mhz"])
    halfwidth = float(harmonic_cfg["halfwidth_mhz"])
    for payload in plot_subbands:
        summary = payload["summary"]
        fit = payload["fit"]
        lags = np.asarray(payload["lags"], dtype=float)
        acf = np.asarray(payload["acf"], dtype=float)
        err = np.asarray(payload["err"], dtype=float)
        model = np.full_like(lags, float(fit.get("constant", 0.0)))
        for component in fit.get("components", []):
            model += driver._lorentzian_curve(
                lags, float(component["dnu_mhz"]), float(component["m"])
            )
        positive = (lags > 0) & (lags <= float(summary["fit_range_mhz"]))
        harmonic = positive & (
            np.abs(lags / spacing - np.round(lags / spacing)) * spacing <= halfwidth
        )
        valid = positive & np.isfinite(acf) & np.isfinite(err) & (err > 0)
        z = np.abs(acf - model) / err
        harmonic_z = z[valid & harmonic]
        background_z = z[valid & ~harmonic]
        median_harmonic = float(np.median(harmonic_z)) if harmonic_z.size else math.inf
        median_background = float(np.median(background_z)) if background_z.size else math.nan
        ratio = median_harmonic / median_background if median_background > 0 else math.inf
        systematic = float(summary["harmonic_mask_systematic"]["systematic_frac"])
        item_pass = (
            harmonic_z.size > 0 and median_harmonic <= 3.0 and ratio <= 2.0 and systematic <= 0.25
        )
        passed &= item_pass
        records.append(
            {
                "subband": int(summary["index"]),
                "median_abs_z_harmonic": median_harmonic,
                "median_abs_z_background": median_background,
                "harmonic_to_background_ratio": ratio,
                "mask_systematic_fraction": systematic,
                "pass": item_pass,
            }
        )
    return {
        "pass": bool(passed),
        "thresholds": {"median_abs_z": 3.0, "ratio": 2.0, "systematic_fraction": 0.25},
        "subbands": records,
    }


def _split_time_check(driver, config, harmonic_cfg):
    pipe = driver.ScintillationAnalysis(config)
    pipe.run()
    start, stop = map(int, pipe.burst_lims)
    midpoint = (start + stop) // 2
    windows = ((start, midpoint), (midpoint, stop))
    off_spectrum = pipe.masked_spectrum.get_spectrum(pipe.off_pulse_lims)
    records = []
    passed = True
    for index, channel_slice in enumerate(pipe.acf_results["subband_channel_slices"]):
        c0, c1 = map(int, channel_slice)
        channel_width = float(pipe.acf_results["subband_channel_widths_mhz"][index])
        fits = []
        for window in windows:
            spectrum = pipe.masked_spectrum.get_spectrum(window)[c0:c1]
            off_mean = float(np.ma.mean(off_spectrum[c0:c1]))
            acf = driver.analysis.calculate_acf(
                spectrum,
                channel_width,
                off_burst_spectrum_mean=off_mean,
                max_lag_bins=int(1.0 / channel_width),
            )
            fits.append(
                None
                if acf is None
                else _fit_component(
                    driver,
                    np.asarray(acf.lags),
                    np.asarray(acf.acf),
                    None if acf.err is None else np.asarray(acf.err),
                    fit_range=1.0,
                    channel_width=channel_width,
                    harmonic_cfg=harmonic_cfg,
                )
            )
        if any(item is None for item in fits):
            item_pass = False
            difference_sigma = None
            fractional_difference = None
        else:
            difference = abs(fits[0]["dnu_mhz"] - fits[1]["dnu_mhz"])
            combined_error = math.hypot(fits[0]["dnu_err_mhz"], fits[1]["dnu_err_mhz"])
            difference_sigma = difference / combined_error if combined_error > 0 else math.inf
            fractional_difference = difference / np.mean([fits[0]["dnu_mhz"], fits[1]["dnu_mhz"]])
            item_pass = difference_sigma <= 2.0 or fractional_difference <= 0.50
        passed &= item_pass
        records.append(
            {
                "subband": index,
                "windows": [list(window) for window in windows],
                "fits": fits,
                "difference_sigma": difference_sigma,
                "fractional_difference": fractional_difference,
                "pass": item_pass,
            }
        )
    return {
        "pass": bool(passed),
        "thresholds": {"difference_sigma": 2.0, "fractional_difference": 0.50},
        "subbands": records,
    }


def _split_band_check(result):
    subbands = result["subbands"]
    widths = [float(item["selected_components"][0]["dnu_mhz"]) for item in subbands]
    frequencies = [float(item["center_freq_mhz"]) for item in subbands]
    alpha = math.log(widths[1] / widths[0]) / math.log(frequencies[1] / frequencies[0])
    artifact_pass = all(item["off_pulse_null"]["null_pass"] for item in subbands)
    stability_pass = all(item["low_lag_stability"]["stable"] for item in subbands)
    passed = artifact_pass and stability_pass and widths[1] > widths[0] and 0.0 < alpha < 8.0
    return {
        "pass": passed,
        "frequencies_mhz": frequencies,
        "dnu_mhz": widths,
        "scintillation_bandwidth_scaling_index": alpha,
        "requirement": "both bands pass artifact gates and bandwidth increases with frequency",
    }


def _positive_acf(spectrum, max_lag=24):
    values = np.asarray(spectrum, dtype=float)
    values = values - np.nanmean(values)
    variance = np.nanmean(values * values)
    return np.asarray(
        [np.nanmean(values[:-lag] * values[lag:]) / variance for lag in range(1, max_lag + 1)]
    )


def _kernel_check(driver, uncorrected_path: Path, corrected_path: Path):
    raw = np.load(uncorrected_path)["power_2d"].astype(float)
    corrected = np.load(corrected_path)["power_2d"].astype(float)
    frequencies = np.load(corrected_path)["frequencies_mhz"]
    records = []
    passed = True
    for lo, hi in ((400.0, 600.0), (600.0, 800.0)):
        select = (frequencies >= lo) & (frequencies < hi)
        gain = np.nanmedian(raw[select, 10:200], axis=1)

        def spectrum(array, window, selected=select, channel_gain=gain):
            normalized = array[selected, window] / channel_gain[:, None]
            return np.nanmean(normalized, axis=1)

        train_raw = _positive_acf(spectrum(raw, slice(10, 105)))
        train_corrected = _positive_acf(spectrum(corrected, slice(10, 105)))
        test_raw = _positive_acf(spectrum(raw, slice(105, 200)))
        test_corrected = _positive_acf(spectrum(corrected, slice(105, 200)))
        kernel = train_raw - train_corrected
        chunk_predictions = []
        for start in range(10, 180, 17):
            first = _positive_acf(spectrum(raw, slice(start, start + 17)))
            second = _positive_acf(spectrum(corrected, slice(start, start + 17)))
            chunk_predictions.append(first - second)
        uncertainty = np.std(chunk_predictions, axis=0, ddof=1) / math.sqrt(len(chunk_predictions))
        uncertainty = np.maximum(uncertainty, 0.01)
        verdict = driver.correction.kernel_crosscheck(
            test_raw,
            test_corrected,
            kernel,
            uncertainty=uncertainty,
        )
        passed &= verdict["pass"]
        records.append(
            {
                "band_mhz": [lo, hi],
                "train_kernel": kernel.tolist(),
                "test_raw_acf": test_raw.tolist(),
                "test_corrected_acf": test_corrected.tolist(),
                "uncertainty": uncertainty.tolist(),
                **verdict,
            }
        )
    return {"pass": bool(passed), "method": "held-out off-pulse half", "bands": records}


def _injection_check(driver):
    from scipy.linalg import toeplitz

    rng = np.random.default_rng(20260712)
    injected = []
    recovered = []
    lower = []
    upper = []
    nchan, ntime = 256, 80
    frequencies = 600.0 + np.arange(nchan) * CHANNEL_WIDTH_MHZ
    coarse = np.asarray([600.0, 600.390625, 600.78125, 601.171875])
    burst_window = slice(50, 58)
    burst_mask = np.zeros((nchan, ntime), dtype=bool)
    burst_mask[:, burst_window] = True
    widths = np.repeat([0.02, 0.04, 0.08, 0.12], 6)
    for width in widths:
        lag_axis = np.arange(nchan) * CHANNEL_WIDTH_MHZ
        covariance = toeplitz(1.0 / (1.0 + (lag_axis / width) ** 2))
        scintillation = np.linalg.cholesky(covariance + np.eye(nchan) * 1e-8) @ rng.normal(
            size=nchan
        )
        scintillation = (scintillation - scintillation.mean()) / scintillation.std()
        t = np.linspace(0, 4 * np.pi, ntime)
        mode1 = np.linspace(0.5, 1.4, nchan)[:, None] * np.sin(t)[None, :]
        mode2 = np.cos(np.linspace(0, 2 * np.pi, nchan))[:, None] * np.cos(2.3 * t)[None, :]
        power = 10.0 * (1.0 + 0.12 * mode1 + 0.09 * mode2)
        power += rng.normal(0.0, 0.02, power.shape) * 10.0
        power[:, burst_window] += 10.0 * (5.0 + 1.8 * scintillation[:, None])
        result = build_chime_products(
            power,
            frequencies,
            coarse,
            coarse_offsets=np.zeros(coarse.size, dtype=int),
            burst_mask=burst_mask,
            config=ChimeProductConfig(
                target="injection",
                dm=100.0,
                upchannel_factor=64,
                dt_s=1.0,
                off_pulse=(0, 40),
                correction_rank=2,
            ),
        )
        on_spectrum = np.ma.masked_invalid(np.nanmean(result.corrected[:, burst_window], axis=1))
        off_mean = float(np.nanmean(result.corrected[:, :40]))
        acf = driver.analysis.calculate_acf(
            on_spectrum,
            CHANNEL_WIDTH_MHZ,
            off_burst_spectrum_mean=off_mean,
            max_lag_bins=80,
        )
        fit = _fit_component(
            driver,
            acf.lags,
            acf.acf,
            acf.err,
            fit_range=0.4,
            channel_width=CHANNEL_WIDTH_MHZ,
            harmonic_cfg={"enable": False},
        )
        if fit is None or not np.isfinite(fit["dnu_err_mhz"]):
            continue
        injected.append(width)
        recovered.append(fit["dnu_mhz"])
        lower.append(fit["dnu_mhz"] - fit["dnu_err_mhz"])
        upper.append(fit["dnu_mhz"] + fit["dnu_err_mhz"])
    summary = driver.correction.injection_recovery_summary(
        np.asarray(injected),
        np.asarray(recovered),
        np.asarray(lower),
        np.asarray(upper),
        channel_width_mhz=CHANNEL_WIDTH_MHZ,
    )
    summary.update(
        {
            "injected_mhz": injected,
            "recovered_mhz": recovered,
            "lower_mhz": lower,
            "upper_mhz": upper,
        }
    )
    return summary


def _render_figures(output_dir, result, plot_subbands, checks, uncorrected_path, corrected_path):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Canonical ACF diagnostic from the tracked producer.
    driver = _driver_module()
    canonical = driver._plot_burst_acfs(
        "freya", plot_subbands, figure_dir=figure_dir, band="chime_h2"
    )

    injection = checks["injection_recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    x = np.asarray(injection["injected_mhz"]) * 1e3
    y = np.asarray(injection["recovered_mhz"]) * 1e3
    lo = (np.asarray(injection["recovered_mhz"]) - np.asarray(injection["lower_mhz"])) * 1e3
    hi = (np.asarray(injection["upper_mhz"]) - np.asarray(injection["recovered_mhz"])) * 1e3
    for ax in axes:
        ax.errorbar(x, y, yerr=np.vstack((lo, hi)), fmt="o", color="#5b3c88", alpha=0.75)
        ax.plot((0, 130), (0, 130), color="black", linestyle="--", label="identity")
        ax.set(
            xlabel=r"Injected HWHM $\gamma$ (kHz)",
            ylabel=r"Recovered HWHM $\gamma$ (kHz)",
            xlim=(0, 130),
        )
        ax.grid(alpha=0.2)
    axes[0].set_yscale("symlog", linthresh=20)
    axes[0].set_title("Full outcome range (symmetric log)")
    axes[1].set_ylim(-20, 300)
    axes[1].set_title("Zoomed measurement range")
    axes[1].legend(frameon=False)
    fig.suptitle("Freya H2 known-truth injection recovery")
    fig.tight_layout()
    injection_path = figure_dir / "freya_h2_injection_recovery.png"
    fig.savefig(injection_path, dpi=180)
    plt.close(fig)

    raw = np.load(uncorrected_path)["power_2d"].astype(float)
    corrected = np.load(corrected_path)["power_2d"].astype(float)
    frequencies = np.load(corrected_path)["frequencies_mhz"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, (lo_band, hi_band) in zip(axes, ((400, 600), (600, 800)), strict=True):
        select = (frequencies >= lo_band) & (frequencies < hi_band)
        gain = np.nanmedian(raw[select, 10:200], axis=1)
        for array, label, color in (
            (raw, "uncorrected", "#b04a5a"),
            (corrected, "rank-2 corrected", "#345995"),
        ):
            spectrum = np.nanmean(array[select, 10:200] / gain[:, None], axis=1)
            ax.plot(
                np.arange(1, 25) * CHANNEL_WIDTH_MHZ * 1e3,
                _positive_acf(spectrum),
                marker="o",
                ms=3,
                label=label,
                color=color,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{lo_band}-{hi_band} MHz off-pulse")
        ax.set_xlabel("Fine-channel lag (kHz)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized frequency ACF")
    axes[1].legend(frameon=False)
    fig.suptitle("Freya H2 correction diagnostic")
    fig.tight_layout()
    correction_path = figure_dir / "freya_h2_correction_diagnostic.png"
    fig.savefig(correction_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    window = checks["fit_window_stability"]
    for subband in window["subbands"]:
        x_values = [float(key) for key in subband["variants"]]
        y_values = [subband["variants"][str(value)]["dnu_mhz"] * 1e3 for value in x_values]
        axes[0, 0].plot(x_values, y_values, marker="o", label=f"subband {subband['subband']}")
    axes[0, 0].set(
        xlabel="Fit window (MHz)", ylabel=r"$\gamma$ (kHz)", title="Fit-window stability"
    )
    axes[0, 0].legend(frameon=False)

    split = checks["split_time_stability"]
    for subband in split["subbands"]:
        values = [item["dnu_mhz"] * 1e3 if item else np.nan for item in subband["fits"]]
        errors = [item["dnu_err_mhz"] * 1e3 if item else np.nan for item in subband["fits"]]
        axes[0, 1].errorbar(
            (0, 1), values, yerr=errors, marker="o", label=f"subband {subband['subband']}"
        )
    axes[0, 1].set(
        xticks=(0, 1),
        xticklabels=("early", "late"),
        ylabel=r"$\gamma$ (kHz)",
        title="Split-time stability",
    )
    axes[0, 1].legend(frameon=False)

    comb = checks["comb_residual"]
    indices = np.arange(len(comb["subbands"]))
    axes[1, 0].bar(
        indices - 0.17,
        [item["median_abs_z_harmonic"] for item in comb["subbands"]],
        width=0.34,
        label="harmonic bins",
    )
    axes[1, 0].bar(
        indices + 0.17,
        [item["median_abs_z_background"] for item in comb["subbands"]],
        width=0.34,
        label="other bins",
    )
    axes[1, 0].axhline(3, color="black", linestyle="--", lw=1)
    axes[1, 0].set(
        xticks=indices,
        xlabel="Subband",
        ylabel=r"Median |residual| / $\sigma$",
        title="Comb residual",
    )
    axes[1, 0].legend(frameon=False)

    kernel = checks["kernel_crosscheck"]
    for band in kernel["bands"]:
        axes[1, 1].plot(
            np.arange(1, len(band["test_corrected_acf"]) + 1) * CHANNEL_WIDTH_MHZ * 1e3,
            band["test_corrected_acf"],
            marker="o",
            ms=3,
            label=f"{int(band['band_mhz'][0])}-{int(band['band_mhz'][1])} measured",
        )
        predicted = np.asarray(band["test_raw_acf"]) - np.asarray(band["train_kernel"])
        axes[1, 1].plot(
            np.arange(1, len(predicted) + 1) * CHANNEL_WIDTH_MHZ * 1e3,
            predicted,
            linestyle="--",
            label=f"{int(band['band_mhz'][0])}-{int(band['band_mhz'][1])} predicted",
        )
    axes[1, 1].set(
        xlabel="Fine-channel lag (kHz)",
        ylabel="Normalized ACF",
        title="Held-out kernel cross-check",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Freya H2 validation battery | science status: {result['science_status']}")
    fig.tight_layout()
    battery_path = figure_dir / "freya_h2_validation_battery.png"
    fig.savefig(battery_path, dpi=180)
    plt.close(fig)

    figures = [
        _figure_record(
            output_dir,
            Path(canonical["figure_png"]),
            "Two CHIME subband ACF panels show resolved positive-lag Lorentzian wings and no claim of final measurement status.",
        ),
        _figure_record(
            output_dir,
            injection_path,
            "Recovered widths track the identity line without a width-dependent trend; uncertainty bars are visible.",
        ),
        _figure_record(
            output_dir,
            correction_path,
            "Rank-2 correction reduces low-band short-lag off-pulse correlation without producing a new narrow spike; high-band behavior remains near zero.",
        ),
        _figure_record(
            output_dir,
            battery_path,
            "Fit-window, split-time, comb, and held-out-kernel panels are legible and agree with the machine verdicts in validation.json.",
        ),
    ]
    (output_dir / "figures.manifest.json").write_text(
        json.dumps({"figures": figures}, indent=2, sort_keys=True) + "\n"
    )
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrected",
        type=Path,
        default=DEFAULT_DATA / "freya_chime_coarse_rank2_v1_corrected.npz",
    )
    parser.add_argument(
        "--uncorrected",
        type=Path,
        default=DEFAULT_DATA / "freya_chime_coarse_rank2_v1_uncorrected.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATA / "freya_chime_coarse_rank2_v1_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/chime-recovery-2026-07-12/results/h2",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    driver = _driver_module()

    with tempfile.TemporaryDirectory(prefix="freya-h2-validation-") as temp_dir:
        config = _prepared_config(driver, args.corrected, args.manifest, Path(temp_dir))
        result, plot_subbands = driver._fit_prepared_config(
            config,
            BASE_CONFIG,
            output_dir=Path(temp_dir),
            max_components=3,
        )
        harmonic_cfg = config["analysis"]["fitting"]["harmonic_mask"]
        checks = {
            "manifest_verification": result["correction_validation"]["checks"][
                "manifest_verification"
            ],
            "off_pulse_null": {"pass": result["artifact_control"]["off_pulse_null"]["null_pass"]},
            "low_lag_stability": {
                "pass": result["artifact_control"]["low_lag_stability"]["stable"]
            },
            "injection_recovery": _injection_check(driver),
            "fit_window_stability": _fit_window_check(driver, plot_subbands, harmonic_cfg),
            "split_time_stability": _split_time_check(driver, config, harmonic_cfg),
            "split_band_stability": _split_band_check(result),
            "comb_residual": _comb_residual_check(driver, plot_subbands, harmonic_cfg),
            "kernel_crosscheck": _kernel_check(driver, args.uncorrected, args.corrected),
            "manual_review": {"pass": None, "reason": "pending visual inspection"},
        }

    fitted_widths = [
        item["selected_components"][0]["dnu_mhz"]
        for item in result["subbands"]
        if item["selected_components"] and not item["selected_components"][0]["quality_flags"]
    ]
    correction = driver.correction.adjudicate_chime_result(
        checks,
        fitted_dnu_mhz=float(np.median(fitted_widths)) if fitted_widths else None,
    )
    correction["science_status"] = driver.correction.combine_science_status(
        artifact_status=result["artifact_control"]["measurement_status"],
        correction_status=correction,
    )
    result["correction_validation"] = {"checks": checks, **correction}
    result["product_correction_status"] = correction["product_correction_status"]
    result["science_status"] = correction["science_status"]
    result["measurement_status"] = correction["science_status"]
    result["validation_contract"] = {
        "fit_window_max_fractional_shift": 0.30,
        "split_time_max_sigma": 2.0,
        "split_time_max_fractional_difference": 0.50,
        "comb_max_median_abs_z": 3.0,
        "comb_max_background_ratio": 2.0,
        "kernel_max_abs_z": 3.0,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(driver._jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    figures = _render_figures(
        args.output_dir,
        result,
        plot_subbands,
        checks,
        args.uncorrected,
        args.corrected,
    )
    print(
        json.dumps(
            {
                "product_correction_status": correction["product_correction_status"],
                "science_status": correction["science_status"],
                "failed_checks": correction["failed_checks"],
                "pending_checks": correction["pending_checks"],
                "figures": figures,
            },
            indent=2,
        )
    )
    return 0 if not correction["failed_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
