#!/usr/bin/env python3
"""Qualify the bounded Freya H3 stationary-covariance whitening hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.linalg import toeplitz

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

H2_RUNNER = Path(__file__).with_name("validate_freya_h2.py")
DEFAULT_PRODUCT = (
    Path.home()
    / "Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank1_v1_corrected.npz"
)
CHANNEL_WIDTH_MHZ = 0.006103608758678547
BLOCK_SIZE = 64
TRAIN = slice(10, 105)
TEST = slice(105, 200)
EIGEN_FLOOR_FRACTION = 0.10
MAX_KERNEL_Z = 3.0
WIDTHS_MHZ = CHANNEL_WIDTH_MHZ * np.asarray([2.0, 4.0, 8.0, 16.0])
MODULATION_INDICES = (0.3, 1.0)


def _figure_record(output_dir: Path, path: Path, expectation: str) -> dict[str, str]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expectation": expectation,
    }


N_SEEDS = 3


def _h2_module():
    spec = importlib.util.spec_from_file_location("freya_h2_validation", H2_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _blocks_for_band(frequencies: np.ndarray, lo: float, hi: float) -> list[slice]:
    if frequencies.ndim != 1 or frequencies.size % BLOCK_SIZE:
        raise ValueError("frequency grid must contain complete 64-channel blocks")
    blocks = []
    for start in range(0, frequencies.size, BLOCK_SIZE):
        block = slice(start, start + BLOCK_SIZE)
        center = float(np.mean(frequencies[block]))
        if lo <= center < hi:
            blocks.append(block)
    if not blocks:
        raise ValueError(f"no complete blocks in {lo}-{hi} MHz")
    return blocks


def _training_standardization(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmean(power[:, TRAIN], axis=1)
    scale = np.nanstd(power[:, TRAIN], axis=1, ddof=1)
    valid = np.isfinite(location) & np.isfinite(scale) & (scale > 0)
    if not valid.all():
        raise ValueError("non-finite or zero-variance channel in whitening training window")
    return location, scale


def _lag_correlations(samples: np.ndarray, max_lag: int) -> np.ndarray:
    variance = float(np.nanmean(samples * samples))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("cannot estimate covariance from zero-variance samples")
    return np.asarray(
        [
            1.0 if lag == 0 else float(np.nanmean(samples[:-lag] * samples[lag:]) / variance)
            for lag in range(max_lag + 1)
        ]
    )


def _estimate_whitener(
    power: np.ndarray,
    frequencies: np.ndarray,
    band: tuple[float, float],
) -> dict:
    location, scale = _training_standardization(power)
    standardized = (power - location[:, None]) / scale[:, None]
    blocks = _blocks_for_band(frequencies, *band)
    block_rho = []
    for block in blocks:
        samples = standardized[block, TRAIN]
        block_rho.append(_lag_correlations(samples, BLOCK_SIZE - 1))
    rho = np.nanmedian(np.asarray(block_rho), axis=0)
    covariance = toeplitz(rho)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = EIGEN_FLOOR_FRACTION * float(np.median(eigenvalues))
    clipped = np.maximum(eigenvalues, floor)
    whitener = (eigenvectors * clipped**-0.5) @ eigenvectors.T
    dc_gain = float(np.mean(whitener @ np.ones(BLOCK_SIZE)))
    if not np.isfinite(dc_gain) or abs(dc_gain) < 1e-12:
        raise ValueError("whitener has invalid DC gain")
    whitener /= dc_gain
    return {
        "band_mhz": list(band),
        "blocks": blocks,
        "rho": rho,
        "eigenvalues": eigenvalues,
        "clipped_eigenvalues": clipped,
        "eigenvalue_floor": floor,
        "dc_gain_before_normalization": dc_gain,
        "condition_number_after_floor": float(clipped.max() / clipped.min()),
        "whitener": whitener,
        "location": location,
        "scale": scale,
    }


def _apply_to_blocks(samples: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    if samples.shape[0] % BLOCK_SIZE:
        raise ValueError("sample vector must contain complete 64-channel blocks")
    corrected = np.empty_like(samples, dtype=float)
    for start in range(0, samples.shape[0], BLOCK_SIZE):
        block = slice(start, start + BLOCK_SIZE)
        corrected[block] = whitener @ samples[block]
    return corrected


def _pooled_correlation(samples: np.ndarray, blocks: list[slice], max_lag: int) -> np.ndarray:
    values = [[] for _ in range(max_lag)]
    for block in blocks:
        block_samples = samples[block]
        variance = float(np.nanmean(block_samples * block_samples))
        for lag in range(1, max_lag + 1):
            values[lag - 1].append(
                float(np.nanmean(block_samples[:-lag] * block_samples[lag:]) / variance)
            )
    return np.asarray([np.nanmedian(item) for item in values])


def _held_out_check(power: np.ndarray, transfer: dict) -> dict:
    standardized = (power - transfer["location"][:, None]) / transfer["scale"][:, None]
    blocks = transfer["blocks"]
    raw = _pooled_correlation(standardized[:, TEST], blocks, 12)
    whitened = standardized.copy()
    for block in blocks:
        whitened[block, TEST] = transfer["whitener"] @ standardized[block, TEST]
    corrected = _pooled_correlation(whitened[:, TEST], blocks, 12)

    chunk_values = []
    for chunk in np.array_split(np.arange(TEST.start, TEST.stop), 5):
        chunk_values.append(_pooled_correlation(whitened[:, chunk], blocks, 12))
    uncertainty = np.std(np.asarray(chunk_values), axis=0, ddof=1) / math.sqrt(len(chunk_values))
    uncertainty = np.maximum(uncertainty, 1e-4)
    z = corrected / uncertainty
    return {
        "pass": bool(np.max(np.abs(z)) <= MAX_KERNEL_Z),
        "raw_correlation": raw.tolist(),
        "whitened_correlation": corrected.tolist(),
        "uncertainty": uncertainty.tolist(),
        "z": z.tolist(),
        "max_abs_z": float(np.max(np.abs(z))),
        "threshold_max_abs_z": MAX_KERNEL_Z,
        "train_bins": [TRAIN.start, TRAIN.stop],
        "test_bins": [TEST.start, TEST.stop],
    }


def _fit_component(driver, spectrum: np.ndarray) -> dict | None:
    acf = driver.analysis.calculate_acf(
        np.ma.masked_invalid(spectrum),
        CHANNEL_WIDTH_MHZ,
        off_burst_spectrum_mean=1.0,
        max_lag_bins=80,
    )
    if acf is None:
        return None
    lags, values, errors = driver._slice_fit_window(acf.lags, acf.acf, acf.err, 0.4)
    lags, values, errors = driver._select_physical_fit_lags(
        lags,
        values,
        errors,
        channel_width_mhz=CHANNEL_WIDTH_MHZ,
        telescope="chime",
    )
    verdict = driver.compare_lorentzian_components(lags, values, max_components=1, acf_err=errors)
    fit = driver._selected_fit(verdict)
    components = fit.get("components", [])
    if not fit.get("success") or len(components) != 1:
        return None
    component = components[0]
    return {
        "dnu_mhz": float(component["dnu_mhz"]),
        "dnu_err_mhz": float(component.get("dnu_err", np.nan)),
        "m": float(component.get("m", np.nan)),
        "m_err": float(component.get("m_err", np.nan)),
    }


def _injection_check(transfers: list[dict]) -> dict:
    driver = _h2_module()._driver_module()
    records = []
    for band_index, transfer in enumerate(transfers):
        whitener = transfer["whitener"]
        for width in WIDTHS_MHZ:
            lag_axis = np.arange(BLOCK_SIZE * 8) * CHANNEL_WIDTH_MHZ
            covariance = toeplitz(1.0 / (1.0 + (lag_axis / width) ** 2))
            factor = np.linalg.cholesky(covariance + np.eye(covariance.shape[0]) * 1e-10)
            for modulation in MODULATION_INDICES:
                for seed_index in range(N_SEEDS):
                    seed = 20260713 + 1000 * band_index + 100 * seed_index
                    seed += int(round(width / CHANNEL_WIDTH_MHZ)) + int(10 * modulation)
                    rng = np.random.default_rng(seed)
                    scintillation = factor @ rng.normal(size=factor.shape[0])
                    scintillation -= scintillation.mean()
                    scintillation /= scintillation.std()
                    injected = 1.0 + modulation * scintillation
                    whitened = 1.0 + _apply_to_blocks(modulation * scintillation, whitener)
                    before = _fit_component(driver, injected)
                    after = _fit_component(driver, whitened)
                    records.append(
                        {
                            "band_mhz": transfer["band_mhz"],
                            "seed": seed,
                            "injected_dnu_mhz": float(width),
                            "injected_m": modulation,
                            "recovered_m_direct": float(np.std(whitened - 1.0)),
                            "before": before,
                            "after": after,
                        }
                    )

    finite = [item for item in records if item["after"] is not None]
    injected_width = np.asarray([item["injected_dnu_mhz"] for item in finite])
    recovered_width = np.asarray([item["after"]["dnu_mhz"] for item in finite])
    width_error = np.asarray([item["after"]["dnu_err_mhz"] for item in finite])
    width_summary = driver.correction.injection_recovery_summary(
        injected_width,
        recovered_width,
        recovered_width - width_error,
        recovered_width + width_error,
        channel_width_mhz=CHANNEL_WIDTH_MHZ,
    )
    amplitude_bias = np.asarray(
        [abs(item["recovered_m_direct"] - item["injected_m"]) for item in finite]
    )
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


def _jsonable_transfer(transfer: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in transfer.items()
        if key not in {"blocks", "location", "scale", "whitener"}
    } | {
        "n_blocks": len(transfer["blocks"]),
        "whitener": transfer["whitener"].tolist(),
    }


def _render(output_dir: Path, transfers: list[dict], held_out: list[dict], injection: dict):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for transfer in transfers:
        label = f"{int(transfer['band_mhz'][0])}-{int(transfer['band_mhz'][1])} MHz"
        axes[0].plot(
            np.arange(1, BLOCK_SIZE) * CHANNEL_WIDTH_MHZ * 1e3,
            transfer["rho"][1:],
            label=label,
        )
        axes[1].plot(np.sort(transfer["eigenvalues"]), label=f"{label} raw")
        axes[1].plot(
            np.sort(transfer["clipped_eigenvalues"]), linestyle="--", label=f"{label} floored"
        )
    axes[0].set(
        xlabel="Fine-channel lag (kHz)",
        ylabel="Training correlation",
        title="Off-pulse stationary kernel",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xlabel="Eigenvalue index",
        ylabel="Covariance eigenvalue",
        yscale="log",
        title="Fixed 10% eigenvalue floor",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Freya H3 off-pulse whitening transfer")
    fig.tight_layout()
    path = figure_dir / "freya_h3_whitening_transfer.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Training kernels and the fixed eigenvalue regularization are finite, smooth, and distinct between the two half-bands.",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    lags = np.arange(1, 13) * CHANNEL_WIDTH_MHZ * 1e3
    for ax, transfer, check in zip(axes, transfers, held_out, strict=True):
        sigma = np.asarray(check["uncertainty"])
        ax.plot(lags, check["raw_correlation"], marker="o", label="raw held-out")
        ax.errorbar(
            lags, check["whitened_correlation"], yerr=sigma, marker="o", label="whitened held-out"
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            title=f"{int(transfer['band_mhz'][0])}-{int(transfer['band_mhz'][1])} MHz | max abs(z)={check['max_abs_z']:.2f}",
            xlabel="Fine-channel lag (kHz)",
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Held-out frequency correlation")
    fig.suptitle("Freya H3 independent kernel cross-check")
    fig.tight_layout()
    path = figure_dir / "freya_h3_held_out_kernel.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Whitened held-out correlations are consistent with zero within the predeclared 3-standard-error maximum.",
        )
    )

    finite = [item for item in injection["records"] if item["after"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for modulation, marker in zip(MODULATION_INDICES, ("o", "s"), strict=True):
        selected = [item for item in finite if item["injected_m"] == modulation]
        x = np.asarray([item["injected_dnu_mhz"] for item in selected]) * 1e3
        y = np.asarray([item["after"]["dnu_mhz"] for item in selected]) * 1e3
        m = np.asarray([item["recovered_m_direct"] for item in selected])
        axes[0].scatter(x, y, marker=marker, alpha=0.75, label=f"injected m={modulation}")
        axes[1].scatter(
            np.full_like(m, modulation),
            m,
            marker=marker,
            alpha=0.75,
            label=f"injected m={modulation}",
        )
    axes[0].plot((0, 105), (0, 105), "k--", label="identity")
    axes[1].plot((0, 1.1), (0, 1.1), "k--", label="identity")
    axes[0].set(
        xlabel="Injected HWHM (kHz)",
        ylabel="Recovered HWHM (kHz)",
        title="Width transfer",
        xlim=(0, 105),
    )
    axes[1].set(
        xlabel="Injected modulation index",
        ylabel="Recovered modulation index",
        title="Amplitude transfer",
        xlim=(0, 1.1),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Freya H3 known-truth whitening injections")
    fig.tight_layout()
    path = figure_dir / "freya_h3_injection_recovery.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    figures.append(
        _figure_record(
            output_dir,
            path.with_suffix(".svg"),
            "Recovered widths and modulation indices follow the identity lines at both near-resolution and resolved scales.",
        )
    )

    (output_dir / "figures.manifest.json").write_text(
        json.dumps({"figures": figures}, indent=2, sort_keys=True) + "\n"
    )
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/chime-recovery-2026-07-12/results/h3",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    product = np.load(args.product)
    power = product["power_2d"].astype(float)
    frequencies = product["frequencies_mhz"].astype(float)
    if power.shape[0] != frequencies.size or not np.all(np.diff(frequencies) > 0):
        raise ValueError("H3 requires a frequency-ascending product")

    transfers = [
        _estimate_whitener(power, frequencies, band) for band in ((400.0, 600.0), (600.0, 800.0))
    ]
    held_out = [_held_out_check(power, transfer) for transfer in transfers]
    injection = _injection_check(transfers)
    checks = {
        "independent_kernel_crosscheck": {
            "pass": all(item["pass"] for item in held_out),
            "bands": held_out,
        },
        "width_amplitude_injection_recovery": injection,
        "manual_review": {"pass": None, "reason": "pending visual inspection"},
    }
    transfer_pass = all(check["pass"] is True for check in checks.values())
    validation = {
        "hypothesis": "H3 stationary fine-channel covariance whitening",
        "source_product": str(args.product.resolve()),
        "source_role": "rank-1 additive-common-mode baseline; rank-2 is not reused",
        "on_pulse_application_performed": False,
        "on_pulse_application_rule": "forbidden unless kernel, injection, and manual-review gates all pass",
        "thresholds": {
            "eigen_floor_fraction": EIGEN_FLOOR_FRACTION,
            "kernel_max_abs_z": MAX_KERNEL_Z,
            "width_bias": "max(10 percent, 0.25 channel)",
            "coverage_68_tolerance": 0.15,
            "amplitude_bias": "max(10 percent, 0.05 absolute)",
        },
        "transfers": [_jsonable_transfer(item) for item in transfers],
        "checks": checks,
        "transfer_qualification_status": "pass" if transfer_pass else "inconclusive",
        "science_status": "diagnostic_only",
    }
    figures = _render(args.output_dir, transfers, held_out, injection)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"checks": checks, "figures": figures}, indent=2))
    return 0 if transfer_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
