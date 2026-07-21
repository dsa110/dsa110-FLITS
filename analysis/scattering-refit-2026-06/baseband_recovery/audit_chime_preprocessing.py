#!/usr/bin/env python3
"""Audit source masking, RFI excision, and bandpass correction off-pulse only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(text: str) -> tuple[int, int]:
    start, stop = (int(value) for value in text.split(":", 1))
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("interval must be START:STOP with 0 <= START < STOP")
    return start, stop


def _slice(interval: tuple[int, int], ntime: int) -> slice:
    start, stop = interval
    if stop > ntime:
        raise ValueError(f"interval {start}:{stop} exceeds {ntime} time bins")
    return slice(start, stop)


def _apply_mask(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(data, dtype=np.float64).copy()
    result[~valid] = np.nan
    return result


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanstd(values: np.ndarray, axis: int, ddof: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(values, axis=axis, ddof=ddof)


def _package_rfi_mask(
    data: np.ndarray,
    source_valid: np.ndarray,
    interval: tuple[int, int],
    *,
    threshold_mean: float,
    threshold_std: float,
) -> np.ndarray:
    """Learn the baseband-analysis 1.9.0 channel mask on one off-pulse interval."""
    from baseband_analysis.core.flagging import get_RFI_channels  # noqa: PLC0415

    compact_indices = np.flatnonzero(source_valid)
    compact = np.asarray(data)[compact_indices, _slice(interval, data.shape[1])]
    compact_mask = get_RFI_channels(
        compact,
        diagnostic_plots=False,
        thres_mean=threshold_mean,
        thres_std=threshold_std,
    )
    result = np.zeros(source_valid.shape, dtype=bool)
    result[compact_indices] = compact_mask
    return result


def _bandpass_model(
    data: np.ndarray,
    valid: np.ndarray,
    interval: tuple[int, int],
    *,
    minimum_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = _apply_mask(data, valid)[:, _slice(interval, data.shape[1])]
    finite_count = np.isfinite(training).sum(axis=1)
    required = int(np.ceil(training.shape[1] * minimum_fraction))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = _nanmean(training, axis=1)
        scale = _nanstd(training, axis=1, ddof=1)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    scale_floor = np.nanmedian(positive) * 1e-6 if positive.size else np.inf
    model_valid = (
        valid
        & (finite_count >= required)
        & np.isfinite(mean)
        & np.isfinite(scale)
        & (scale > scale_floor)
    )
    return mean, scale, model_valid


def _normalize(data: np.ndarray, mean: np.ndarray, scale: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(np.asarray(data).shape, np.nan, dtype=np.float64)
    result[valid] = (np.asarray(data)[valid] - mean[valid, None]) / scale[valid, None]
    return result


def _robust_spread(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    center = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - center)))


def _frequency_lag_correlation(spectrum: np.ndarray, lag: int) -> float:
    left = spectrum[:-lag]
    right = spectrum[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _temporal_lag_one(data: np.ndarray) -> float:
    correlations = []
    for row in data:
        valid = np.isfinite(row[:-1]) & np.isfinite(row[1:])
        if valid.sum() >= 3 and np.nanstd(row[:-1][valid]) > 0 and np.nanstd(row[1:][valid]) > 0:
            correlations.append(np.corrcoef(row[:-1][valid], row[1:][valid])[0, 1])
    return float(np.nanmedian(correlations)) if correlations else float("nan")


def _coarse_comb_fraction(spectrum: np.ndarray, upchan_factor: int) -> float:
    valid = np.isfinite(spectrum)
    if valid.sum() < upchan_factor * 2:
        return float("nan")
    filled = np.where(valid, spectrum - np.nanmedian(spectrum), 0.0)
    index = np.arange(spectrum.size)
    harmonic = np.abs(
        np.sum(filled * np.exp(-2j * np.pi * index / upchan_factor))
    )
    total = np.sum(np.abs(filled))
    return float(harmonic / total) if total > 0 else float("nan")


def _metrics(
    data: np.ndarray,
    valid: np.ndarray,
    interval: tuple[int, int],
    upchan_factor: int,
) -> dict:
    validation = _apply_mask(data, valid)[:, _slice(interval, data.shape[1])]
    channel_mean = _nanmean(validation, axis=1)
    channel_std = _nanstd(validation, axis=1, ddof=1)
    spectrum = channel_mean.copy()
    return {
        "valid_fine_positions": int(valid.sum()),
        "masked_fraction_of_nominal": float(1.0 - valid.mean()),
        "validation_channel_mean_robust_spread": _robust_spread(channel_mean),
        "validation_channel_std_robust_spread": _robust_spread(channel_std),
        "validation_temporal_lag_one_median": _temporal_lag_one(validation[valid]),
        "validation_frequency_lag_correlations": {
            str(lag): _frequency_lag_correlation(spectrum, lag)
            for lag in range(1, 11)
        },
        "validation_coarse_comb_fraction": _coarse_comb_fraction(
            spectrum, upchan_factor
        ),
    }


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def _write_figure(
    path: Path,
    frequency: np.ndarray,
    validation: tuple[int, int],
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True, constrained_layout=True)
    for name in ("grid_only", "bandpass_only", "combined"):
        data, valid = variants[name]
        spectrum = _nanmean(
            _apply_mask(data, valid)[:, _slice(validation, data.shape[1])], axis=1
        )
        finite = np.isfinite(spectrum)
        scale = np.nanmedian(np.abs(spectrum[finite])) if finite.any() else 1.0
        axes[0].plot(frequency, spectrum / scale, linewidth=0.55, label=name.replace("_", " "))
    axes[0].set_ylabel("Held-out mean / median amplitude")
    axes[0].legend(frameon=False, ncol=3)

    for name in ("rfi_only", "bandpass_only", "combined"):
        valid = variants[name][1]
        coarse_valid = valid.reshape(1024, -1).mean(axis=1)
        coarse_frequency = frequency.reshape(1024, -1).mean(axis=1)
        axes[1].plot(coarse_frequency, coarse_valid, linewidth=0.8, label=name.replace("_", " "))
    axes[1].set_ylabel("Retained fine-channel fraction")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=3)
    fig.savefig(path)
    plt.close(fig)


def audit(
    product_dir: Path,
    output_dir: Path,
    target: str,
    training: tuple[int, int],
    validation: tuple[int, int],
    threshold_mean: float,
    threshold_std: float,
) -> dict:
    prefix = f"{target}_chime"
    paths = {
        "stokes_i": product_dir / f"{prefix}_upchan.npy",
        "frequency": product_dir / f"{prefix}_freq.npy",
        "source_valid": product_dir / f"{prefix}_source_valid.npy",
        "metadata": product_dir / f"{prefix}_preprocessing_metadata.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=False)

    data = np.load(paths["stokes_i"], mmap_mode="r")
    frequency = np.load(paths["frequency"])
    source_valid = np.load(paths["source_valid"]).astype(bool)
    metadata = json.loads(paths["metadata"].read_text())
    upchan_factor = int(metadata["upchannel_factor"])
    if data.shape[0] != 1024 * upchan_factor or source_valid.shape != (data.shape[0],):
        raise ValueError("product is not on the nominal fine grid")
    _slice(training, data.shape[1])
    _slice(validation, data.shape[1])
    if max(training[0], validation[0]) < min(training[1], validation[1]):
        raise ValueError("training and validation intervals overlap")

    initial_rfi = _package_rfi_mask(
        data,
        source_valid,
        training,
        threshold_mean=threshold_mean,
        threshold_std=threshold_std,
    )
    rfi_valid = source_valid & ~initial_rfi
    mean, scale, bandpass_valid = _bandpass_model(data, source_valid, training)
    normalized = _normalize(data, mean, scale, bandpass_valid)
    bandpass_first_rfi = _package_rfi_mask(
        normalized,
        bandpass_valid,
        training,
        threshold_mean=threshold_mean,
        threshold_std=threshold_std,
    )
    bandpass_then_rfi_valid = bandpass_valid & ~bandpass_first_rfi
    bandpass_then_rfi = _apply_mask(normalized, bandpass_then_rfi_valid)
    combined_mean, combined_scale, combined_valid = _bandpass_model(
        data, rfi_valid, training
    )
    combined = _normalize(data, combined_mean, combined_scale, combined_valid)
    post_rfi = _package_rfi_mask(
        combined,
        combined_valid,
        training,
        threshold_mean=threshold_mean,
        threshold_std=threshold_std,
    )
    combined_valid &= ~post_rfi
    combined = _apply_mask(combined, combined_valid)

    midpoint = (training[0] + training[1]) // 2
    first_half = (training[0], midpoint)
    second_half = (midpoint, training[1])
    rfi_first = _package_rfi_mask(
        data, source_valid, first_half,
        threshold_mean=threshold_mean, threshold_std=threshold_std,
    )
    rfi_second = _package_rfi_mask(
        data, source_valid, second_half,
        threshold_mean=threshold_mean, threshold_std=threshold_std,
    )
    _, scale_first, valid_first = _bandpass_model(data, source_valid, first_half)
    _, scale_second, valid_second = _bandpass_model(data, source_valid, second_half)
    gain_valid = valid_first & valid_second
    gain_log_ratio = np.log(scale_first[gain_valid] / scale_second[gain_valid])

    variants = {
        "raw_compact": (np.asarray(data), source_valid),
        "grid_only": (np.asarray(data), source_valid),
        "rfi_only": (_apply_mask(data, rfi_valid), rfi_valid),
        "bandpass_only": (normalized, bandpass_valid),
        "bandpass_then_rfi": (bandpass_then_rfi, bandpass_then_rfi_valid),
        "combined": (combined, combined_valid),
    }
    metrics = {
        name: _metrics(values, valid, validation, upchan_factor)
        for name, (values, valid) in variants.items()
    }

    mask_paths = {}
    for name, mask in (
        ("initial_rfi_mask", initial_rfi),
        ("post_bandpass_rfi_mask", post_rfi),
        ("bandpass_first_rfi_mask", bandpass_first_rfi),
        ("bandpass_then_rfi_valid_mask", bandpass_then_rfi_valid),
        ("combined_valid_mask", combined_valid),
    ):
        path = output_dir / f"{target}_{name}.npy"
        np.save(path, mask)
        mask_paths[name] = {"path": path.name, "sha256": _sha256(path)}
    for name, values in (
        ("bandpass_mean", combined_mean),
        ("bandpass_scale", combined_scale),
    ):
        path = output_dir / f"{target}_{name}.npy"
        np.save(path, values)
        mask_paths[name] = {"path": path.name, "sha256": _sha256(path)}

    figure_path = output_dir / f"{target}_rfi_bandpass_audit.svg"
    _write_figure(figure_path, frequency, validation, variants)
    report = {
        "schema": "faber2026-chime-preprocessing-audit-v1",
        "status": "diagnostic_only_no_science_fit",
        "target": target,
        "training_interval": list(training),
        "validation_interval": list(validation),
        "rfi_algorithm": {
            "implementation": "baseband_analysis.core.flagging.get_RFI_channels",
            "threshold_mean": threshold_mean,
            "threshold_std": threshold_std,
            "learned_from": "off_pulse_training_only",
        },
        "bandpass_algorithm": {
            "implementation": "per-channel off-pulse mean subtraction and sample-standard-deviation division",
            "minimum_finite_fraction": 0.8,
            "learned_from": "off_pulse_training_only",
        },
        "stability": {
            "training_half_rfi_jaccard": _jaccard(rfi_first, rfi_second),
            "training_half_bandpass_scale_log_ratio_robust_spread": _robust_spread(gain_log_ratio),
            "training_half_bandpass_comparable_positions": int(gain_valid.sum()),
        },
        "mask_counts": {
            "source_missing": int((~source_valid).sum()),
            "initial_rfi": int(initial_rfi.sum()),
            "bandpass_first_rfi": int(bandpass_first_rfi.sum()),
            "bandpass_then_rfi_valid": int(bandpass_then_rfi_valid.sum()),
            "bandpass_invalid_after_initial_rfi": int((rfi_valid & ~combined_valid & ~post_rfi).sum()),
            "post_bandpass_rfi": int(post_rfi.sum()),
            "combined_valid": int(combined_valid.sum()),
        },
        "metrics": metrics,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            **mask_paths,
            "figure": {"path": figure_path.name, "sha256": _sha256(figure_path)},
        },
    }
    report_path = output_dir / f"{target}_preprocessing_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="zach")
    parser.add_argument("--training", type=_interval, default=(55, 137))
    parser.add_argument("--validation", type=_interval, default=(138, 220))
    parser.add_argument("--threshold-mean", type=float, default=5.0)
    parser.add_argument("--threshold-std", type=float, default=3.0)
    args = parser.parse_args()
    report = audit(
        args.product_dir,
        args.output_dir,
        args.target,
        args.training,
        args.validation,
        args.threshold_mean,
        args.threshold_std,
    )
    print(json.dumps({"status": report["status"], "mask_counts": report["mask_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
