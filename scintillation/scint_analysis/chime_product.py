"""Build provenance-bearing CHIME products for scintillation analysis.

The correction is deliberately applied before frequency-dependent alignment.
At that point a dispersed burst can be masked while a common instrumental
time mode is estimated from the remaining samples.  The corrected and raw
products are then aligned with padded placement; circular wrapping is never
used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

K_DM_S_MHZ2 = 1.0 / 2.41e-4
SUPPORTED_CORRECTION_ALGORITHMS = {
    1: "robust_coarse_rank1_v1",
    2: "robust_coarse_rank2_v1",
}


@dataclass(frozen=True)
class ChimeProductConfig:
    target: str
    dm: float
    upchannel_factor: int
    dt_s: float
    off_pulse: tuple[int, int]
    guard_bins: int = 0
    reference_frequency_mhz: float = 400.0
    correction_rank: int = 1
    aligned_burst_bin: int | None = None
    burst_half_width_bins: int | None = None


@dataclass
class ChimeProductResult:
    uncorrected: np.ndarray
    corrected: np.ndarray
    frequencies_mhz: np.ndarray
    times_s: np.ndarray
    channel_gain: np.ndarray
    manifest: dict


def load_chime_target(target: str | None = None, path: str | Path | None = None):
    """Load the versioned 12-target CHIME product registry."""
    registry_path = (
        Path(path) if path else Path(__file__).parents[1] / "configs/chime_products.yaml"
    )
    payload = yaml.safe_load(registry_path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("targets"), dict):
        raise ValueError("unsupported or malformed CHIME product registry")
    targets = payload["targets"]
    if target is None:
        return targets
    if target not in targets:
        raise KeyError(f"unknown CHIME target: {target}")
    return targets[target]


def burst_track_mask(
    *,
    n_channels: int,
    n_times: int,
    channel_offsets: np.ndarray,
    aligned_center_bin: int,
    half_width_bins: int,
) -> np.ndarray:
    """Map a protected aligned burst window onto the pre-alignment array."""
    offsets = np.asarray(channel_offsets, dtype=int)
    if offsets.shape != (n_channels,) or half_width_bins < 0:
        raise ValueError(
            "channel_offsets must match channels and half_width_bins must be non-negative"
        )
    mask = np.zeros((n_channels, n_times), dtype=bool)
    for channel, offset in enumerate(offsets):
        center = int(aligned_center_bin) - int(offset)
        start = max(0, center - half_width_bins)
        stop = min(n_times, center + half_width_bins + 1)
        if start < stop:
            mask[channel, start:stop] = True
    return mask


def coarse_alignment_offsets(
    coarse_frequencies_mhz: np.ndarray,
    fpga_count: np.ndarray,
    *,
    delta_time_s: float,
    dm: float,
    dt_s: float,
    reference_frequency_mhz: float = 400.0,
) -> np.ndarray:
    """Return non-negative integer coarse-channel placement offsets."""
    coarse = np.asarray(coarse_frequencies_mhz, dtype=float)
    fpga = np.asarray(fpga_count, dtype=float)
    if coarse.ndim != 1 or coarse.shape != fpga.shape:
        raise ValueError("coarse frequencies and fpga_count must be matching 1-D arrays")
    ctime = (fpga - fpga[-1]) * float(delta_time_s)
    delay = ctime - K_DM_S_MHZ2 * float(dm) * (
        1.0 / coarse**2 - 1.0 / float(reference_frequency_mhz) ** 2
    )
    offsets = np.rint(delay / float(dt_s)).astype(int)
    return offsets - offsets.min()


def _expand_mask(mask: np.ndarray, guard_bins: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if guard_bins <= 0:
        return mask.copy()
    expanded = mask.copy()
    for shift in range(1, guard_bins + 1):
        expanded[:, shift:] |= mask[:, :-shift]
        expanded[:, :-shift] |= mask[:, shift:]
    return expanded


def _rank1_background(
    normalized: np.ndarray,
    valid: np.ndarray,
    *,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted deterministic rank-1 least-squares model of ``normalized-1``."""
    residual = np.asarray(normalized, dtype=float) - 1.0
    weights = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    if not weights.any():
        return (
            np.zeros_like(residual),
            np.zeros(residual.shape[0], dtype=float),
            np.zeros(residual.shape[1], dtype=float),
        )
    filled = np.where(weights, residual, 0.0)
    temporal = np.zeros(residual.shape[1], dtype=float)
    supported_times = np.any(weights, axis=0)
    temporal[supported_times] = np.nanmedian(
        np.where(weights[:, supported_times], residual[:, supported_times], np.nan), axis=0
    )
    if np.linalg.norm(temporal) == 0:
        temporal = np.ones(residual.shape[1], dtype=float)

    spectral = np.ones(residual.shape[0], dtype=float)
    for _ in range(iterations):
        denom_f = np.sum(weights * temporal[None, :] ** 2, axis=1)
        numer_f = np.sum(filled * temporal[None, :], axis=1)
        spectral = np.divide(numer_f, denom_f, out=np.zeros_like(numer_f), where=denom_f > 0)

        denom_t = np.sum(weights * spectral[:, None] ** 2, axis=0)
        numer_t = np.sum(filled * spectral[:, None], axis=0)
        temporal = np.divide(numer_t, denom_t, out=np.zeros_like(numer_t), where=denom_t > 0)

        norm = np.linalg.norm(temporal)
        if norm > 0:
            temporal /= norm
            spectral *= norm

    return spectral[:, None] * temporal[None, :], spectral, temporal


def _coarse_low_rank_background(
    normalized: np.ndarray,
    valid: np.ndarray,
    parent_coarse: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic sequential rank-1 modes within each coarse block."""
    model = np.zeros_like(normalized, dtype=float)
    spectral = np.zeros(normalized.shape[0], dtype=float)
    temporal_modes = []
    for parent in np.unique(parent_coarse):
        channels = parent_coarse == parent
        block_model = np.zeros_like(normalized[channels], dtype=float)
        block_spectral_modes = []
        block_temporal_modes = []
        for _ in range(rank):
            component, component_spectral, component_temporal = _rank1_background(
                normalized[channels] - block_model, valid[channels]
            )
            block_model += component
            block_spectral_modes.append(component_spectral)
            block_temporal_modes.append(component_temporal)
        model[channels] = block_model
        spectral[channels] = np.sqrt(np.sum(np.square(block_spectral_modes), axis=0))
        temporal_modes.append(block_temporal_modes)
    return model, spectral, np.asarray(temporal_modes)


def _align(power: np.ndarray, channel_offsets: np.ndarray) -> np.ndarray:
    nt_in = power.shape[1]
    nt_out = nt_in + int(channel_offsets.max(initial=0))
    aligned = np.full((power.shape[0], nt_out), np.nan, dtype=np.float32)
    for channel, offset in enumerate(channel_offsets):
        aligned[channel, offset : offset + nt_in] = power[channel]
    return aligned


def build_chime_products(
    power: np.ndarray,
    frequencies_mhz: np.ndarray,
    coarse_frequencies_mhz: np.ndarray,
    *,
    coarse_offsets: np.ndarray,
    burst_mask: np.ndarray,
    config: ChimeProductConfig,
    rfi_mask: np.ndarray | None = None,
) -> ChimeProductResult:
    """Correct and align a detected CHIME dynamic spectrum."""
    if not config.target.strip():
        raise ValueError("target is required for product provenance")
    if not np.isfinite(config.dm) or config.dm <= 0:
        raise ValueError("positive finite dm is required")
    if config.upchannel_factor < 1 or config.dt_s <= 0:
        raise ValueError("positive upchannel_factor and dt_s are required")
    if config.correction_rank not in SUPPORTED_CORRECTION_ALGORITHMS:
        raise ValueError("correction_rank must be 1 or 2")

    power = np.asarray(power, dtype=float)
    frequencies = np.asarray(frequencies_mhz, dtype=float)
    coarse = np.asarray(coarse_frequencies_mhz, dtype=float)
    offsets = np.asarray(coarse_offsets, dtype=int)
    burst_mask = np.asarray(burst_mask, dtype=bool)
    if power.ndim != 2 or frequencies.shape != (power.shape[0],):
        raise ValueError("power must be channel x time and match frequencies")
    if burst_mask.shape != power.shape:
        raise ValueError("burst_mask must match power")
    if offsets.shape != coarse.shape or np.any(offsets < 0):
        raise ValueError("coarse_offsets must be non-negative and match coarse frequencies")

    parent = np.argmin(np.abs(frequencies[:, None] - coarse[None, :]), axis=1)
    channel_offsets = offsets[parent]
    start, stop = config.off_pulse
    if not (0 <= start < stop <= power.shape[1]):
        raise ValueError("off_pulse must be inside the input time axis")

    channel_gain = np.nanmedian(power[:, start:stop], axis=1)
    good_gain = np.isfinite(channel_gain) & (channel_gain > 0)
    normalized = np.divide(
        power,
        channel_gain[:, None],
        out=np.full_like(power, np.nan),
        where=good_gain[:, None],
    )
    excluded = _expand_mask(burst_mask, config.guard_bins)
    if rfi_mask is not None:
        if np.asarray(rfi_mask).shape != power.shape:
            raise ValueError("rfi_mask must match power")
        excluded |= np.asarray(rfi_mask, dtype=bool)
    valid = np.isfinite(normalized) & ~excluded
    model, spectral, temporal = _coarse_low_rank_background(
        normalized, valid, parent, config.correction_rank
    )
    corrected = power - channel_gain[:, None] * model
    corrected[~np.isfinite(power)] = np.nan

    uncorrected_aligned = _align(power, channel_offsets)
    corrected_aligned = _align(corrected, channel_offsets)
    times = np.arange(uncorrected_aligned.shape[1], dtype=float) * config.dt_s
    retained = int(valid.sum())
    total_finite = int(np.isfinite(power).sum())
    model_rms = float(np.sqrt(np.mean(model[valid] ** 2))) if retained else 0.0
    pre_off = _off_pulse_diagnostics(uncorrected_aligned, channel_gain, config.off_pulse)
    post_off = _off_pulse_diagnostics(corrected_aligned, channel_gain, config.off_pulse)
    manifest = {
        "schema_version": 1,
        "target": config.target,
        "dm_pc_cm3": float(config.dm),
        "upchannel_factor": int(config.upchannel_factor),
        "channel_width_mhz": float(np.nanmedian(np.diff(np.sort(frequencies)))),
        "dt_s": float(config.dt_s),
        "alignment": {
            "method": "padded_integer_placement_v1",
            "coarse_offsets": offsets.tolist(),
            "min_offset": int(offsets.min(initial=0)),
            "max_offset": int(offsets.max(initial=0)),
            "circular": False,
        },
        "masks": {
            "burst_samples": int(burst_mask.sum()),
            "burst_mask_sha256": _array_sha256(burst_mask),
            "guard_bins": int(config.guard_bins),
            "excluded_samples": int(excluded.sum()),
            "excluded_mask_sha256": _array_sha256(excluded),
        },
        "windows": {
            "off_pulse_input_bins": [int(start), int(stop)],
            "aligned_burst_bin": config.aligned_burst_bin,
            "burst_half_width_bins": config.burst_half_width_bins,
        },
        "correction": {
            "algorithm": SUPPORTED_CORRECTION_ALGORITHMS[config.correction_rank],
            "rank": config.correction_rank,
            "iterations": 12,
            "retained_fraction": retained / total_finite if total_finite else 0.0,
            "model_rms_fractional": model_rms,
            "spectral_mode_rms": float(np.sqrt(np.mean(spectral**2))),
            "temporal_mode_rms": float(np.sqrt(np.mean(temporal**2))),
        },
        "off_pulse_diagnostics": {"pre": pre_off, "post": post_off},
        "product_correction_status": "inconclusive",
        "science_status": "diagnostic_only",
    }
    return ChimeProductResult(
        uncorrected=uncorrected_aligned,
        corrected=corrected_aligned,
        frequencies_mhz=frequencies,
        times_s=times,
        channel_gain=channel_gain,
        manifest=manifest,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def write_chime_products(
    result: ChimeProductResult,
    output_prefix: str | Path,
    *,
    input_paths: list[str | Path],
) -> dict[str, Path]:
    """Write paired NPZs and a hash-addressed JSON manifest."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = prefix.with_name(prefix.name + "_uncorrected.npz")
    corrected_path = prefix.with_name(prefix.name + "_corrected.npz")
    manifest_path = prefix.with_name(prefix.name + "_manifest.json")
    existing = [path for path in (raw_path, corrected_path, manifest_path) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing CHIME products: {names}")
    common = {
        "frequencies_mhz": result.frequencies_mhz.astype(np.float64),
        "times_s": result.times_s.astype(np.float64),
    }
    np.savez(raw_path, power_2d=np.flip(result.uncorrected, axis=0), **common)
    np.savez(corrected_path, power_2d=np.flip(result.corrected, axis=0), **common)

    manifest = dict(result.manifest)
    manifest["inputs"] = [
        {"path": str(Path(path)), "sha256": _sha256(Path(path))} for path in input_paths
    ]
    manifest["products"] = {
        "uncorrected": raw_path.name,
        "uncorrected_sha256": _sha256(raw_path),
        "corrected": corrected_path.name,
        "corrected_sha256": _sha256(corrected_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"uncorrected": raw_path, "corrected": corrected_path, "manifest": manifest_path}


def verify_product_manifest(
    manifest_path: str | Path,
    corrected_path: str | Path,
    *,
    expected_target: str | None = None,
) -> dict:
    """Fail closed unless a corrected NPZ matches its versioned manifest."""
    manifest_file = Path(manifest_path)
    product_file = Path(corrected_path)
    if not manifest_file.is_file() or not product_file.is_file():
        return {"valid": False, "reason": "manifest or corrected product missing"}
    try:
        manifest = json.loads(manifest_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "reason": "manifest is unreadable or invalid JSON"}
    if manifest.get("schema_version") != 1:
        return {"valid": False, "reason": "unsupported manifest schema"}
    algorithm = manifest.get("correction", {}).get("algorithm")
    if algorithm not in SUPPORTED_CORRECTION_ALGORITHMS.values():
        return {"valid": False, "reason": "unsupported correction algorithm"}
    if expected_target is not None and manifest.get("target") != expected_target:
        return {
            "valid": False,
            "reason": "manifest target does not match configured burst",
            "expected_target": expected_target,
            "manifest_target": manifest.get("target"),
        }
    expected = manifest.get("products", {}).get("corrected_sha256")
    if not expected or _sha256(product_file) != expected:
        return {"valid": False, "reason": "corrected product hash mismatch"}
    return {
        "valid": True,
        "reason": "corrected product and manifest verified",
        "target": manifest.get("target"),
        "algorithm": algorithm,
        "manifest": manifest,
    }


def _positive_lag_correlation(spectrum: np.ndarray, max_lag: int = 6) -> float:
    x = np.asarray(spectrum, dtype=float)
    x = x - np.nanmean(x)
    variance = np.nanmean(x * x)
    if not np.isfinite(variance) or variance <= 0:
        return 0.0
    last_lag = min(max_lag, x.size - 1)
    if last_lag < 1:
        return 0.0
    values = [np.nanmean(x[:-lag] * x[lag:]) / variance for lag in range(1, last_lag + 1)]
    return float(np.nanmean(values))


def _off_pulse_diagnostics(
    aligned: np.ndarray, channel_gain: np.ndarray, off_pulse: tuple[int, int]
) -> dict:
    start, stop = off_pulse
    stop = min(stop, aligned.shape[1])
    spectrum = np.nanmean(aligned[:, start:stop], axis=1) / channel_gain
    return {
        "aggregate_lags_1_6": _positive_lag_correlation(spectrum),
        "fitted_null_width_mhz": None,
        "fit_status": "pending_acf_fit",
    }


def simulate_alignment_streak(*, seed: int = 20260712) -> dict:
    """Seeded mechanism harness for common-mode noise sheared by alignment."""
    rng = np.random.default_rng(seed)
    ncoarse, fine_per_coarse, ntime = 64, 16, 96
    coarse = np.linspace(600.0, 800.0, ncoarse, endpoint=False)
    frequencies = np.repeat(coarse, fine_per_coarse) + np.tile(
        np.linspace(-0.18, 0.18, fine_per_coarse), ncoarse
    )
    offsets = np.arange(ncoarse) // 2

    innovations = rng.normal(0.0, 1.0, ntime)
    temporal = np.zeros(ntime)
    for index in range(1, ntime):
        temporal[index] = 0.82 * temporal[index - 1] + innovations[index]
    temporal /= np.std(temporal)
    spectral = np.repeat(np.linspace(0.7, 1.3, ncoarse), fine_per_coarse)
    power = 10.0 * (1.0 + 0.22 * spectral[:, None] * temporal[None, :])
    power += rng.normal(0.0, 0.015, power.shape) * 10.0

    result = build_chime_products(
        power,
        frequencies,
        coarse,
        coarse_offsets=offsets,
        burst_mask=np.zeros_like(power, dtype=bool),
        config=ChimeProductConfig(
            target="forward-model",
            dm=900.0,
            upchannel_factor=fine_per_coarse,
            dt_s=1.0,
            off_pulse=(0, 80),
        ),
    )
    window = slice(20, 36)
    pre_spectrum = np.nanmean(result.uncorrected[:, window], axis=1) / result.channel_gain
    post_spectrum = np.nanmean(result.corrected[:, window], axis=1) / result.channel_gain
    temporal_corr = []
    for lag in range(6):
        if lag == 0:
            temporal_corr.append(1.0)
        else:
            temporal_corr.append(float(np.corrcoef(temporal[:-lag], temporal[lag:])[0, 1]))
    return {
        "seed": int(seed),
        "pre_low_lag_correlation": _positive_lag_correlation(pre_spectrum),
        "post_low_lag_correlation": _positive_lag_correlation(post_spectrum),
        "temporal_correlation": temporal_corr,
        "coarse_offset_span": int(offsets.max() - offsets.min()),
    }
