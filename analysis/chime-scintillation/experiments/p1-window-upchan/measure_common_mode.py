#!/usr/bin/env python3
"""Measure the predeclared off-pulse common-mode response of a CHIME product."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

BASELINE_LAG1 = 0.587
BASELINE_AMPLITUDE = 0.586
SUPPRESSION_FACTOR = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_demean(values: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for block in np.unique(blocks):
        selected = blocks == block
        out[selected] -= np.nanmean(out[selected])
    return out


def _lorentzian(lag_mhz: np.ndarray, amplitude: float, width_mhz: float, offset: float):
    return amplitude / (1.0 + (lag_mhz / width_mhz) ** 2) + offset


def measure_common_mode(
    pol0: np.ndarray,
    pol1: np.ndarray,
    frequencies_mhz: np.ndarray,
    *,
    off_pulse: tuple[int, int] = (0, 200),
    band_mhz: tuple[float, float] = (627.0, 800.0),
    channels_per_coarse: int = 64,
    max_lag_bins: int = 16,
) -> dict:
    """Return deterministic block-demeaned cross-ACF metrics."""
    left = np.asarray(pol0)
    right = np.asarray(pol1)
    frequencies = np.asarray(frequencies_mhz, dtype=float)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("polarization products must be matching channel x time arrays")
    if frequencies.shape != (left.shape[0],):
        raise ValueError("frequency array must match the product channel axis")
    start, stop = off_pulse
    if not 0 <= start < stop <= left.shape[1]:
        raise ValueError("off-pulse window lies outside the time axis")

    selected = (frequencies >= band_mhz[0]) & (frequencies <= band_mhz[1])
    if np.count_nonzero(selected) < 2 * channels_per_coarse:
        raise ValueError("selected band does not contain two complete coarse-channel blocks")
    # Products are written with complete U-channel coarse blocks.  Construct
    # parent IDs before the band cut so a partial edge block is not relabelled.
    parent = np.arange(frequencies.size, dtype=int) // channels_per_coarse
    blocks = parent[selected]
    # Demean complete coarse blocks before selecting the high band.  This is
    # important at the 627 MHz boundary, which cuts through one coarse block,
    # and keeps the coarse-channel operation independent of the band cut.
    spectrum0 = _block_demean(np.nanmean(left[:, start:stop], axis=1), parent)[selected]
    spectrum1 = _block_demean(np.nanmean(right[:, start:stop], axis=1), parent)[selected]
    normalization = float(
        np.sqrt(np.nanmean(spectrum0 * spectrum0) * np.nanmean(spectrum1 * spectrum1))
    )
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("off-pulse spectra have no finite variance")

    correlations = []
    reverse_correlations = []
    auto0 = []
    for lag in range(1, max_lag_bins + 1):
        within_block = blocks[:-lag] == blocks[lag:]
        valid = (
            within_block
            & np.isfinite(spectrum0[:-lag])
            & np.isfinite(spectrum0[lag:])
            & np.isfinite(spectrum1[:-lag])
            & np.isfinite(spectrum1[lag:])
        )
        x0 = spectrum0[:-lag][valid]
        x1 = spectrum0[lag:][valid]
        y0 = spectrum1[:-lag][valid]
        y1 = spectrum1[lag:][valid]
        forward = float(np.mean(x0 * y1) / normalization)
        reverse = float(np.mean(y0 * x1) / normalization)
        correlations.append(0.5 * (forward + reverse))
        reverse_correlations.append(reverse)
        auto0.append(
            float(np.mean(x0 * x1) / np.nanmean(spectrum0 * spectrum0))
        )

    channel_width_mhz = float(abs(np.nanmedian(np.diff(frequencies[selected]))))
    lags_mhz = channel_width_mhz * np.arange(1, max_lag_bins + 1, dtype=float)
    values = np.asarray(correlations)
    initial = (
        max(float(values[0] - values[-1]), 1e-3),
        6.0 * channel_width_mhz,
        float(values[-1]),
    )
    parameters, _ = curve_fit(
        _lorentzian,
        lags_mhz,
        values,
        p0=initial,
        bounds=(
            (0.0, 0.25 * channel_width_mhz, -1.0),
            (2.0, 20.0 * max(lags_mhz), 1.0),
        ),
        maxfev=20000,
    )
    fitted = _lorentzian(lags_mhz, *parameters)
    residual_sum = float(np.sum((values - fitted) ** 2))
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    amplitude, width_mhz, offset = (float(value) for value in parameters)
    lag1 = float(values[0])
    lag1_limit = BASELINE_LAG1 / SUPPRESSION_FACTOR
    amplitude_limit = BASELINE_AMPLITUDE / SUPPRESSION_FACTOR
    return {
        "schema_version": 1,
        "measurement": "offpulse_block_demeaned_polarization_cross_acf_v1",
        "off_pulse_bins": [int(start), int(stop)],
        "band_mhz": [float(band_mhz[0]), float(band_mhz[1])],
        "selected_channels": int(np.count_nonzero(selected)),
        "channels_per_coarse": int(channels_per_coarse),
        "channel_width_mhz": channel_width_mhz,
        "lag_bins": list(range(1, max_lag_bins + 1)),
        "lag_mhz": lags_mhz.tolist(),
        "cross_correlation": correlations,
        "reverse_cross_correlation": reverse_correlations,
        "pol0_auto_correlation": auto0,
        "lorentzian_fit": {
            "amplitude": amplitude,
            "width_mhz": width_mhz,
            "offset": offset,
            "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else None,
        },
        "mechanism_gate": {
            "suppression_factor": SUPPRESSION_FACTOR,
            "baseline_lag1": BASELINE_LAG1,
            "baseline_amplitude": BASELINE_AMPLITUDE,
            "lag1_limit": lag1_limit,
            "amplitude_limit": amplitude_limit,
            "lag1_pass": bool(abs(lag1) <= lag1_limit),
            "amplitude_pass": bool(amplitude <= amplitude_limit),
            "eligible": bool(abs(lag1) <= lag1_limit and amplitude <= amplitude_limit),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, required=True)
    parser.add_argument("--pol1", type=Path, required=True)
    parser.add_argument("--frequencies", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing measurement: {args.output}")

    result = measure_common_mode(
        np.load(args.pol0, mmap_mode="r"),
        np.load(args.pol1, mmap_mode="r"),
        np.load(args.frequencies),
    )
    inputs = (args.pol0, args.pol1, args.frequencies)
    result["inputs"] = [
        {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs
    ]
    if args.metadata is not None:
        metadata_payload = json.loads(args.metadata.read_text())
        result["metadata"] = {
            "path": str(args.metadata.resolve()),
            "sha256": _sha256(args.metadata),
        }
        channelizer = metadata_payload.get("channelizer", {})
        result["variant"] = {
            "window": channelizer.get("window"),
            "oversample": channelizer.get("oversample"),
            "implementation": channelizer.get("implementation"),
        }
    result["producer"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["lorentzian_fit"] | result["mechanism_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
