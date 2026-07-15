#!/usr/bin/env python3
"""Qualify an additive off-pulse covariance likelihood for Freya CHIME."""

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
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

H2_RUNNER = Path(__file__).with_name("validate_freya_h2.py")
DATA = Path.home() / "Data/Faber2026/dsa110/scintillation-data"
DEFAULT_PRODUCT = DATA / "freya_chime_coarse_rank1_v1_corrected.npz"
DEFAULT_MANIFEST = DATA / "freya_chime_coarse_rank1_v1_manifest.json"
FIT_RANGE_MHZ = 0.40
HARMONIC_SPACING_MHZ = 0.390625
HARMONIC_HALFWIDTH_MHZ = 0.05
N_TRAIN = 6
N_TEST = 6
MAX_HELDOUT_Z = 3.0
HELDOUT_REDCHI_RANGE = (0.5, 2.0)


def _figure_record(output_dir: Path, path: Path, expectation: str) -> dict[str, str]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expectation": expectation,
    }


WIDTH_CHANNELS = (2.0, 4.0, 8.0, 16.0)
MODULATION_INDICES = (0.3, 1.0)
N_SEEDS = 3


def _h2_module():
    spec = importlib.util.spec_from_file_location("freya_h2_validation", H2_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _raw_acf(
    spectrum: np.ma.MaskedArray,
    *,
    normalization: float,
    max_lag_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-lag covariance and product-mean standard errors."""
    values = np.ma.asarray(spectrum, dtype=float)
    data = values.filled(np.nan)
    data -= float(np.ma.mean(values))
    denom = float(normalization) ** 2
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("positive finite signal normalization is required")
    acf = []
    error = []
    for lag in range(1, max_lag_bins + 1):
        products = data[:-lag] * data[lag:]
        products = products[np.isfinite(products)]
        if products.size < 20:
            acf.append(np.nan)
            error.append(np.nan)
            continue
        acf.append(float(np.mean(products) / denom))
        error.append(float(np.std(products, ddof=1) / math.sqrt(products.size) / denom))
    return np.asarray(acf), np.asarray(error)


def _regularized_covariance(covariance: np.ndarray, diagonal_error: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    diagonal_error = np.asarray(diagonal_error, dtype=float)
    combined = covariance + np.diag(diagonal_error**2)
    scale = float(np.nanmedian(np.diag(combined)))
    floor = max(scale * 1e-6, 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(combined)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _fit_additive_likelihood(
    lags_mhz: np.ndarray,
    observed_acf: np.ndarray,
    observed_error: np.ndarray,
    kernel_mean: np.ndarray,
    kernel_mean_covariance: np.ndarray,
    *,
    channel_width_mhz: float,
) -> dict | None:
    lags = np.asarray(lags_mhz, dtype=float)
    observed = np.asarray(observed_acf, dtype=float)
    error = np.asarray(observed_error, dtype=float)
    kernel = np.asarray(kernel_mean, dtype=float)
    covariance = np.asarray(kernel_mean_covariance, dtype=float)
    keep = (
        np.isfinite(lags)
        & np.isfinite(observed)
        & np.isfinite(error)
        & (error > 0)
        & (lags >= 1.5 * channel_width_mhz)
        & (lags <= FIT_RANGE_MHZ)
    )
    harmonic_number = np.round(lags / HARMONIC_SPACING_MHZ)
    harmonic_distance = np.abs(lags / HARMONIC_SPACING_MHZ - harmonic_number) * (
        HARMONIC_SPACING_MHZ
    )
    harmonic = (harmonic_number >= 1) & (harmonic_distance <= HARMONIC_HALFWIDTH_MHZ)
    keep &= ~harmonic
    indices = np.flatnonzero(keep)
    if indices.size < 12:
        return None
    x = lags[indices]
    y = observed[indices]
    k = kernel[indices]
    cov = covariance[np.ix_(indices, indices)]
    cov = _regularized_covariance(cov, error[indices])
    factor, lower = cho_factor(cov, lower=True, check_finite=False)

    def model(parameters):
        gamma, modulation, constant = parameters
        return k + modulation**2 / (1.0 + (x / gamma) ** 2) + constant

    def residual(parameters):
        return solve_triangular(
            factor,
            y - model(parameters),
            lower=lower,
            check_finite=False,
        )

    starts = channel_width_mhz * np.asarray([2.0, 4.0, 8.0, 16.0, 32.0])
    candidates = []
    initial_m = float(np.sqrt(max(np.nanmax(y - k), 0.01)))
    for initial_gamma in starts:
        fit = least_squares(
            residual,
            x0=(min(initial_gamma, FIT_RANGE_MHZ * 0.8), min(initial_m, 2.5), 0.0),
            bounds=(
                (0.25 * channel_width_mhz, 0.0, -5.0),
                (FIT_RANGE_MHZ, 3.0, 5.0),
            ),
            max_nfev=5000,
        )
        if fit.success and np.all(np.isfinite(fit.x)):
            candidates.append(fit)
    if not candidates:
        return None
    fit = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    dof = max(1, indices.size - fit.x.size)
    redchi = float(np.sum(fit.fun**2) / dof)
    try:
        parameter_covariance = np.linalg.inv(fit.jac.T @ fit.jac)
        parameter_error = np.sqrt(np.diag(parameter_covariance))
    except np.linalg.LinAlgError:
        parameter_error = np.full(3, np.nan)
    return {
        "dnu_mhz": float(fit.x[0]),
        "dnu_err_mhz": float(parameter_error[0]),
        "m": float(fit.x[1]),
        "m_err": float(parameter_error[1]),
        "constant": float(fit.x[2]),
        "redchi": redchi,
        "n_fit_points": int(indices.size),
        "fit_lags_mhz": x.tolist(),
        "model_acf": model(fit.x).tolist(),
        "observed_acf": y.tolist(),
        "kernel_acf": k.tolist(),
    }


def _stationary_gaussian(
    rng: np.random.Generator,
    *,
    n_channels: int,
    width_mhz: float,
    channel_width_mhz: float,
) -> np.ndarray:
    distances = np.minimum(np.arange(n_channels), n_channels - np.arange(n_channels))
    covariance = 1.0 / (1.0 + (distances * channel_width_mhz / width_mhz) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    sample = np.real(np.fft.ifft(np.fft.fft(rng.normal(size=n_channels)) * np.sqrt(power)))
    sample -= sample.mean()
    sample /= sample.std()
    return sample


def _kernel_from_spectra(
    spectra: list[np.ma.MaskedArray],
    *,
    signal_mean: float,
    max_lag_bins: int,
) -> dict:
    acfs = []
    errors = []
    for spectrum in spectra:
        acf, error = _raw_acf(
            spectrum,
            normalization=signal_mean,
            max_lag_bins=max_lag_bins,
        )
        acfs.append(acf)
        errors.append(error)
    array = np.asarray(acfs)
    sample_covariance = np.cov(array, rowvar=False, ddof=1)
    return {
        "acfs": array,
        "errors": np.asarray(errors),
        "mean": np.mean(array, axis=0),
        "sample_covariance": sample_covariance,
        "mean_covariance": sample_covariance / len(array),
    }


def _heldout_check(kernel: dict, test_spectra, *, signal_mean, max_lag_bins):
    records = []
    predictive_kernel_covariance = kernel["sample_covariance"] * (
        1.0 + 1.0 / kernel["acfs"].shape[0]
    )
    for spectrum in test_spectra:
        acf, error = _raw_acf(
            spectrum,
            normalization=signal_mean,
            max_lag_bins=max_lag_bins,
        )
        covariance = _regularized_covariance(predictive_kernel_covariance, error)
        residual = acf - kernel["mean"]
        sigma = np.sqrt(np.diag(covariance))
        z = residual / sigma
        chi2 = float(residual @ np.linalg.solve(covariance, residual))
        redchi = chi2 / residual.size
        passed = bool(
            np.max(np.abs(z)) <= MAX_HELDOUT_Z
            and HELDOUT_REDCHI_RANGE[0] <= redchi <= HELDOUT_REDCHI_RANGE[1]
        )
        records.append(
            {
                "pass": passed,
                "max_abs_z": float(np.max(np.abs(z))),
                "reduced_predictive_chi2": redchi,
                "acf": acf.tolist(),
                "error": error.tolist(),
                "z": z.tolist(),
            }
        )
    return {
        "pass": all(item["pass"] for item in records),
        "thresholds": {
            "max_abs_z": MAX_HELDOUT_Z,
            "reduced_predictive_chi2": list(HELDOUT_REDCHI_RANGE),
        },
        "records": records,
    }


def _injection_check(driver, subbands: list[dict]) -> dict:
    records = []
    for band_index, band in enumerate(subbands):
        channel_width = band["channel_width_mhz"]
        n_channels = band["n_channels"]
        lags = np.arange(1, band["max_lag_bins"] + 1) * channel_width
        for width_channels in WIDTH_CHANNELS:
            width = width_channels * channel_width
            for modulation in MODULATION_INDICES:
                for seed_index in range(N_SEEDS):
                    seed = 20260713 + 1000 * band_index + 100 * seed_index
                    seed += int(width_channels) + int(10 * modulation)
                    rng = np.random.default_rng(seed)
                    scintillation = _stationary_gaussian(
                        rng,
                        n_channels=n_channels,
                        width_mhz=width,
                        channel_width_mhz=channel_width,
                    )
                    background = band["test_spectra"][seed_index % len(band["test_spectra"])]
                    injected = np.ma.asarray(background).copy()
                    injected += band["signal_mean"] * (1.0 + modulation * scintillation)
                    acf, error = _raw_acf(
                        injected,
                        normalization=band["signal_mean"],
                        max_lag_bins=band["max_lag_bins"],
                    )
                    fit = _fit_additive_likelihood(
                        lags,
                        acf,
                        error,
                        band["kernel"]["mean"],
                        band["kernel"]["mean_covariance"],
                        channel_width_mhz=channel_width,
                    )
                    records.append(
                        {
                            "band_mhz": band["band_mhz"],
                            "seed": seed,
                            "injected_dnu_mhz": width,
                            "injected_m": modulation,
                            "fit": fit,
                        }
                    )
    finite = [item for item in records if item["fit"] is not None]
    injected_width = np.asarray([item["injected_dnu_mhz"] for item in finite])
    recovered_width = np.asarray([item["fit"]["dnu_mhz"] for item in finite])
    width_error = np.asarray([item["fit"]["dnu_err_mhz"] for item in finite])
    width_summary = driver.correction.injection_recovery_summary(
        injected_width,
        recovered_width,
        recovered_width - width_error,
        recovered_width + width_error,
        channel_width_mhz=float(np.median([item["channel_width_mhz"] for item in subbands])),
    )
    amplitude_bias = np.asarray([abs(item["fit"]["m"] - item["injected_m"]) for item in finite])
    amplitude_limit = np.asarray([max(0.10 * item["injected_m"], 0.05) for item in finite])
    amplitude_pass = bool(len(finite) == len(records) and np.all(amplitude_bias < amplitude_limit))
    return {
        "pass": bool(len(finite) == len(records) and width_summary["pass"] and amplitude_pass),
        "n_trials": len(records),
        "n_finite": len(finite),
        "width": width_summary,
        "amplitude_pass": amplitude_pass,
        "max_absolute_amplitude_bias": (
            float(amplitude_bias.max()) if amplitude_bias.size else None
        ),
        "amplitude_limit": "max(10 percent, 0.05 absolute)",
        "records": records,
    }


def _prepare_subbands(pipe) -> list[dict]:
    start, stop = map(int, pipe.burst_lims)
    width = stop - start
    off_start, off_stop = map(int, pipe.off_pulse_lims)
    starts = list(range(off_start, off_stop - width + 1, width))
    if len(starts) < N_TRAIN + N_TEST:
        raise ValueError("not enough burst-duration off-pulse slices for the predeclared split")
    starts = starts[: N_TRAIN + N_TEST]
    on_full = pipe.masked_spectrum.get_spectrum(pipe.burst_lims)
    off_full = pipe.masked_spectrum.get_spectrum(pipe.off_pulse_lims)
    subbands = []
    for index, channel_slice in enumerate(pipe.acf_results["subband_channel_slices"]):
        c0, c1 = map(int, channel_slice)
        channel_width = float(pipe.acf_results["subband_channel_widths_mhz"][index])
        on_spectrum = on_full[c0:c1]
        off_spectrum = off_full[c0:c1]
        signal_mean = float(np.ma.mean(on_spectrum) - np.ma.mean(off_spectrum))
        spectra = [
            pipe.masked_spectrum.get_spectrum((slice_start, slice_start + width))[c0:c1]
            for slice_start in starts
        ]
        max_lag_bins = min(int(FIT_RANGE_MHZ / channel_width) + 8, c1 - c0 - 1)
        kernel = _kernel_from_spectra(
            spectra[:N_TRAIN],
            signal_mean=signal_mean,
            max_lag_bins=max_lag_bins,
        )
        heldout = _heldout_check(
            kernel,
            spectra[N_TRAIN:],
            signal_mean=signal_mean,
            max_lag_bins=max_lag_bins,
        )
        frequencies = np.asarray(pipe.masked_spectrum.frequencies[c0:c1], dtype=float)
        subbands.append(
            {
                "index": index,
                "band_mhz": [float(frequencies.min()), float(frequencies.max())],
                "channel_slice": [c0, c1],
                "channel_width_mhz": channel_width,
                "n_channels": c1 - c0,
                "signal_mean": signal_mean,
                "off_pulse_starts": starts,
                "max_lag_bins": max_lag_bins,
                "on_spectrum": on_spectrum,
                "test_spectra": spectra[N_TRAIN:],
                "kernel": kernel,
                "heldout": heldout,
            }
        )
    return subbands


def _jsonable_band(band: dict) -> dict:
    return {
        key: value
        for key, value in band.items()
        if key not in {"on_spectrum", "test_spectra", "kernel"}
    } | {"kernel": {key: value.tolist() for key, value in band["kernel"].items()}}


def _render(output_dir: Path, subbands: list[dict], injection: dict):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, band in zip(axes, subbands, strict=True):
        lags = np.arange(1, band["max_lag_bins"] + 1) * band["channel_width_mhz"] * 1e3
        kernel = band["kernel"]
        for curve in kernel["acfs"]:
            ax.plot(lags, curve, color="#777777", alpha=0.35, lw=0.8)
        mean_error = np.sqrt(np.diag(kernel["mean_covariance"]))
        ax.plot(lags, kernel["mean"], color="#1f77b4", lw=2, label="training mean")
        ax.fill_between(
            lags,
            kernel["mean"] - mean_error,
            kernel["mean"] + mean_error,
            color="#1f77b4",
            alpha=0.2,
        )
        for record in band["heldout"]["records"]:
            ax.plot(lags, record["acf"], color="#d62728", alpha=0.45, lw=0.9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            title=f"{band['band_mhz'][0]:.0f}-{band['band_mhz'][1]:.0f} MHz",
            xlabel="Frequency lag (kHz)",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Covariance / mean signal$^2$")
    axes[0].legend(frameon=False)
    fig.suptitle("Freya A1 additive off-pulse covariance: train and held-out")
    fig.tight_layout()
    path = figure_dir / "freya_a1_offpulse_covariance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Held-out off-pulse ACFs are statistically consistent with the six-slice training kernel in both subbands.",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, band in zip(axes, subbands, strict=True):
        records = band["heldout"]["records"]
        x = np.arange(1, len(records) + 1)
        ax.bar(
            x - 0.18,
            [item["max_abs_z"] for item in records],
            width=0.36,
            label="max abs(z)",
        )
        ax.bar(
            x + 0.18,
            [item["reduced_predictive_chi2"] for item in records],
            width=0.36,
            label="reduced chi-square",
        )
        ax.axhline(MAX_HELDOUT_Z, color="black", linestyle="--", lw=1)
        ax.axhspan(*HELDOUT_REDCHI_RANGE, color="#2ca02c", alpha=0.12)
        ax.set(
            title=f"{band['band_mhz'][0]:.0f}-{band['band_mhz'][1]:.0f} MHz",
            xlabel="Held-out slice",
            ylabel="Qualification statistic",
        )
        ax.grid(alpha=0.2)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Freya A1 held-out covariance qualification")
    fig.tight_layout()
    path = figure_dir / "freya_a1_heldout_qualification.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Every held-out slice has max abs(z) at most 3 and reduced predictive chi-square between 0.5 and 2.0.",
        )
    )

    finite = [item for item in injection["records"] if item["fit"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for modulation, marker in zip(MODULATION_INDICES, ("o", "s"), strict=True):
        selected = [item for item in finite if item["injected_m"] == modulation]
        x = np.asarray([item["injected_dnu_mhz"] for item in selected]) * 1e3
        width = np.asarray([item["fit"]["dnu_mhz"] for item in selected]) * 1e3
        recovered_m = np.asarray([item["fit"]["m"] for item in selected])
        axes[0].scatter(x, width, marker=marker, alpha=0.75, label=f"injected m={modulation}")
        axes[1].scatter(
            np.full_like(recovered_m, modulation),
            recovered_m,
            marker=marker,
            alpha=0.75,
            label=f"injected m={modulation}",
        )
    limit = max([item["injected_dnu_mhz"] for item in finite], default=0.1) * 1e3 * 1.15
    axes[0].plot((0, limit), (0, limit), "k--", label="identity")
    axes[1].plot((0, 1.1), (0, 1.1), "k--", label="identity")
    axes[0].set(
        xlabel="Injected HWHM (kHz)",
        ylabel="Recovered HWHM (kHz)",
        title="Width recovery",
        xlim=(0, limit),
    )
    axes[1].set(
        xlabel="Injected modulation index",
        ylabel="Recovered modulation index",
        title="Amplitude recovery",
        xlim=(0, 1.1),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Freya A1 real-background known-truth injections")
    fig.tight_layout()
    path = figure_dir / "freya_a1_injection_recovery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Recovered widths and modulation indices follow identity for all 48 real-background injections.",
        )
    )

    (output_dir / "figures.manifest.json").write_text(
        json.dumps({"figures": figures}, indent=2, sort_keys=True) + "\n"
    )
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/chime-recovery-2026-07-12/results/a1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h2 = _h2_module()
    driver = h2._driver_module()

    with tempfile.TemporaryDirectory(prefix="freya-a1-") as temp_dir:
        config = h2._prepared_config(driver, args.product, args.manifest, Path(temp_dir))
        pipe = driver.ScintillationAnalysis(config)
        pipe.run()
        subbands = _prepare_subbands(pipe)

    heldout_pass = all(band["heldout"]["pass"] for band in subbands)
    injection = _injection_check(driver, subbands)
    qualification_pass = heldout_pass and injection["pass"]
    validation = {
        "experiment": "A1 additive off-pulse covariance likelihood",
        "source_product": str(args.product.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "on_pulse_fit_performed": False,
        "on_pulse_fit_rule": "forbidden unless held-out and injection gates both pass",
        "thresholds": {
            "heldout_max_abs_z": MAX_HELDOUT_Z,
            "heldout_reduced_predictive_chi2": list(HELDOUT_REDCHI_RANGE),
            "width_bias": "max(10 percent, 0.25 channel)",
            "coverage_68_tolerance": 0.15,
            "amplitude_bias": "max(10 percent, 0.05 absolute)",
        },
        "checks": {
            "heldout_covariance_prediction": {"pass": heldout_pass},
            "width_amplitude_injection_recovery": injection,
            "manual_review": {"pass": None, "reason": "pending visual inspection"},
        },
        "subbands": [_jsonable_band(item) for item in subbands],
        "qualification_status": "pass" if qualification_pass else "inconclusive",
        "science_status": "diagnostic_only",
    }
    figures = _render(args.output_dir, subbands, injection)
    (args.output_dir / "validation.json").write_text(
        json.dumps(driver._jsonable(validation), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "heldout_pass": heldout_pass,
                "injection": {key: value for key, value in injection.items() if key != "records"},
                "on_pulse_fit_performed": False,
                "figures": figures,
            },
            indent=2,
        )
    )
    return 0 if qualification_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
