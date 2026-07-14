#!/usr/bin/env python3
"""Run frozen Freya B2 matched-scallop, phase-cycled voltage injections."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B1_PATH = (
    ROOT
    / "analysis/chime-baseband-calibration-2026-07-13/run_voltage_injections.py"
)
U = 64
ROUTES = (
    "matched_noise_free",
    "matched_phase_cycled",
    "unmatched_noise_free",
)


def _load_b1():
    spec = importlib.util.spec_from_file_location("freya_b1", B1_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load B1 helpers from {B1_PATH}")
    spec.loader.exec_module(module)
    return module


def _folded_gain(power: np.ndarray, upchannel_factor: int = U) -> tuple[np.ndarray, np.ndarray]:
    """Return separable coarse gain and repeating folded scallop template."""
    values = np.asarray(power, dtype=float)
    if values.ndim != 2 or values.shape[0] % upchannel_factor:
        raise ValueError("power must be (coarse*upchannel, time)")
    spectrum = np.nanmedian(values, axis=1)
    blocks = spectrum.reshape(-1, upchannel_factor)
    coarse_seed = np.nanmedian(blocks, axis=1)
    if np.any(~np.isfinite(coarse_seed)) or np.any(coarse_seed <= 0):
        raise ValueError("invalid native-channel gain while folding scallop")
    normalized = blocks / coarse_seed[:, None]
    center = float(np.nanmedian(normalized))
    mad = float(np.nanmedian(np.abs(normalized - center)))
    if not np.isfinite(mad) or mad <= 0:
        raise ValueError("folded scallop has zero robust scale")
    robust_z = 0.6744897501960817 * (normalized - center) / mad
    normalized = np.where(np.abs(robust_z) <= 3.0, normalized, np.nan)
    scallop = np.nanmedian(normalized, axis=0)
    scallop /= float(np.nanmean(scallop))
    if np.any(~np.isfinite(scallop)) or np.any(scallop <= 0):
        raise ValueError("folded scallop is not finite and positive")
    coarse_gain = np.nanmedian(blocks / scallop[None, :], axis=1)
    gain = (coarse_gain[:, None] * scallop[None, :]).reshape(-1)
    if np.any(~np.isfinite(gain)) or np.any(gain <= 0):
        raise ValueError("separable gain is not finite and positive")
    return gain, scallop


def _phase_cycled_signal(
    plus_power: np.ndarray,
    minus_power: np.ndarray,
    baseline_power: np.ndarray,
) -> np.ndarray:
    """Cancel the signal-noise cross term from a matched +/- voltage pair."""
    return 0.5 * (np.asarray(plus_power) + np.asarray(minus_power)) - np.asarray(
        baseline_power
    )


def _width_pass(item: dict, channel_width_mhz: float) -> bool:
    truth_fit = item.get("truth_fit")
    recovered_fit = item.get("recovered_fit")
    if not truth_fit or not recovered_fit:
        return False
    truth = truth_fit.get("dnu_mhz")
    recovered = recovered_fit.get("dnu_mhz")
    if not np.isfinite(truth) or not np.isfinite(recovered) or truth <= 0:
        return False
    return bool(abs(recovered - truth) < max(0.10 * truth, 0.25 * channel_width_mhz))


def _median_abs_fractional_error(records: list[dict]) -> float:
    values = []
    for item in records:
        truth_fit = item.get("truth_fit")
        recovered_fit = item.get("recovered_fit")
        if not truth_fit or not recovered_fit:
            return math.inf
        truth = truth_fit.get("dnu_mhz")
        recovered = recovered_fit.get("dnu_mhz")
        if not np.isfinite(truth) or not np.isfinite(recovered) or truth <= 0:
            return math.inf
        values.append(abs(recovered - truth) / truth)
    return float(np.median(values)) if values else math.inf


def _build_signal(
    b1,
    dedispersed_crop: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    gain: np.ndarray,
    coarse_offsets: np.ndarray,
    aligned_center: int,
    signal_scale: float,
    rng: np.random.Generator,
    *,
    forward_gain: bool,
) -> np.ndarray:
    signal = np.zeros_like(dedispersed_crop)
    target_cursor = 0
    for coarse_index in range(signal.shape[0]):
        fine_slice = slice(coarse_index * U, (coarse_index + 1) * U)
        use = selected[fine_slice]
        if not np.any(use):
            continue
        block_target = np.zeros(U)
        count = int(np.count_nonzero(use))
        block_target[use] = target[target_cursor : target_cursor + count]
        target_cursor += count
        response = gain[fine_slice] if forward_gain else np.ones(U)
        amplitude = signal_scale * np.sqrt(block_target * response)
        for delta in (-1, 0, 1):
            phase = rng.uniform(0.0, 2.0 * np.pi, U)
            fine_voltage = amplitude * np.exp(1j * phase)
            fft_bins = np.repeat(fine_voltage, b1.DOWNFREQ)
            timeseries = np.fft.ifft(np.fft.ifftshift(fft_bins)).astype(signal.dtype)
            block = aligned_center + delta - int(coarse_offsets[coarse_index])
            start = block * b1.FFTSIZE
            signal[coarse_index, 0, start : start + b1.FFTSIZE] += timeseries
    return signal


def _power_from_spec(spec: np.ndarray) -> np.ndarray:
    return np.abs(spec[0]) ** 2 + np.abs(spec[1]) ** 2


def _extract_corrected(
    b1,
    power_time_frequency: np.ndarray,
    gain: np.ndarray,
    fine_offsets: np.ndarray,
    selected: np.ndarray,
    aligned_center: int,
) -> np.ndarray:
    aligned = b1._align(power_time_frequency.T, fine_offsets)
    corrected = aligned / gain[:, None]
    return np.nanmean(
        corrected[selected, aligned_center - 1 : aligned_center + 2], axis=1
    )


def _trial(
    b1,
    dedispersed_crop: np.ndarray,
    baseline_spec: np.ndarray,
    fine_frequencies: np.ndarray,
    fine_channel_ids: np.ndarray,
    coarse_frequency_ids: np.ndarray,
    fine_offsets: np.ndarray,
    coarse_offsets: np.ndarray,
    gain: np.ndarray,
    *,
    band: tuple[float, float],
    width_bins: float,
    power_ratio: float,
    aligned_center: int,
    seed: int,
) -> list[dict]:
    from baseband_analysis.core.sampling import _upchannel  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    selected = (fine_frequencies >= band[0]) & (fine_frequencies < band[1])
    selected_ids = np.asarray(fine_channel_ids[selected], dtype=int)
    target_full = b1._make_target(
        rng, int(selected_ids.max() - selected_ids.min() + 1), width_bins
    )
    target = target_full[selected_ids - selected_ids.min()]
    truth_fit = b1._fit_width(target, selected_ids)
    if truth_fit is None:
        return []

    baseline_power = _power_from_spec(baseline_spec)
    baseline_aligned = b1._align(baseline_power.T, fine_offsets) / gain[:, None]
    noise_level = float(np.nanmedian(baseline_aligned[selected, aligned_center]))
    signal_scale = math.sqrt(power_ratio * noise_level)

    matched_signal = _build_signal(
        b1,
        dedispersed_crop,
        selected,
        target,
        gain,
        coarse_offsets,
        aligned_center,
        signal_scale,
        rng,
        forward_gain=True,
    )
    unmatched_signal = _build_signal(
        b1,
        dedispersed_crop,
        selected,
        target,
        gain,
        coarse_offsets,
        aligned_center,
        signal_scale,
        rng,
        forward_gain=False,
    )

    matched_only_spec, recovered_freq, channel_id = _upchannel(
        matched_signal,
        freq_id=coarse_frequency_ids,
        fftsize=b1.FFTSIZE,
        downfreq=b1.DOWNFREQ,
    )
    unmatched_only_spec, _, _ = _upchannel(
        unmatched_signal,
        freq_id=coarse_frequency_ids,
        fftsize=b1.FFTSIZE,
        downfreq=b1.DOWNFREQ,
    )
    plus_spec, _, _ = _upchannel(
        dedispersed_crop + matched_signal,
        freq_id=coarse_frequency_ids,
        fftsize=b1.FFTSIZE,
        downfreq=b1.DOWNFREQ,
    )
    minus_spec, _, _ = _upchannel(
        dedispersed_crop - matched_signal,
        freq_id=coarse_frequency_ids,
        fftsize=b1.FFTSIZE,
        downfreq=b1.DOWNFREQ,
    )
    route_power = {
        "matched_noise_free": _power_from_spec(matched_only_spec),
        "matched_phase_cycled": _phase_cycled_signal(
            _power_from_spec(plus_spec), _power_from_spec(minus_spec), baseline_power
        ),
        "unmatched_noise_free": _power_from_spec(unmatched_only_spec),
    }
    expected_mean = float(np.nanmean(power_ratio * noise_level * target))
    records = []
    for route, power in route_power.items():
        recovered_signal = _extract_corrected(
            b1, power, gain, fine_offsets, selected, aligned_center
        )
        recovered_fit = b1._fit_width(recovered_signal, selected_ids)
        records.append(
            {
                "route": route,
                "band_mhz": list(band),
                "nominal_width_channels": width_bins,
                "power_ratio": power_ratio,
                "aligned_center": aligned_center,
                "seed": seed,
                "truth_fit": truth_fit,
                "recovered_fit": recovered_fit,
                "power_recovery_ratio": float(
                    np.nanmean(recovered_signal) / expected_mean
                ),
                "recovered_frequency_check": bool(
                    np.allclose(recovered_freq, fine_frequencies)
                ),
                "channel_id_check": bool(channel_id.size == fine_frequencies.size),
            }
        )
    return records


def _plot(records: list[dict], output: Path) -> None:
    colors = {
        "matched_noise_free": "#2ca02c",
        "matched_phase_cycled": "#1f77b4",
        "unmatched_noise_free": "#d62728",
    }
    labels = {
        "matched_noise_free": "matched, noise-free",
        "matched_phase_cycled": "matched, phase-cycled",
        "unmatched_noise_free": "unmatched, noise-free",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for route in ROUTES:
        chosen = [
            item
            for item in records
            if item["route"] == route and item.get("truth_fit") and item.get("recovered_fit")
        ]
        truth = [item["truth_fit"]["dnu_mhz"] * 1e3 for item in chosen]
        recovered = [item["recovered_fit"]["dnu_mhz"] * 1e3 for item in chosen]
        axes[0].scatter(
            truth,
            recovered,
            s=28,
            alpha=0.7,
            color=colors[route],
            label=labels[route],
        )
        axes[1].scatter(
            truth,
            [item["power_recovery_ratio"] for item in chosen],
            s=28,
            alpha=0.7,
            color=colors[route],
        )
    limit = max(120.0, axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, limit], [0, limit], "k--", lw=1.5, label="identity")
    axes[0].set(xlabel="Fitted target HWHM (kHz)", ylabel="Recovered HWHM (kHz)")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.5)
    axes[1].set(xlabel="Fitted target HWHM (kHz)", ylabel="Recovered / injected power")
    fig.suptitle("Freya B2 matched folded-scallop voltage transfer")
    fig.tight_layout()
    fig.savefig(output / "freya_b2_scallop_recovery.svg")
    fig.savefig(output / "freya_b2_scallop_recovery.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-waterfall", type=Path, required=True)
    parser.add_argument("--canonical-frequency", type=Path, required=True)
    parser.add_argument("--replay-waterfall", type=Path, required=True)
    parser.add_argument("--replay-frequency", type=Path, required=True)
    args = parser.parse_args()

    from baseband_analysis.core.bbdata import BBData  # noqa: PLC0415
    from baseband_analysis.core.dedispersion import coherent_dedisp  # noqa: PLC0415
    from baseband_analysis.core.sampling import _upchannel  # noqa: PLC0415

    b1 = _load_b1()
    args.output.mkdir(parents=True, exist_ok=True)
    replay = b1._verify_replay_provenance(
        args.h5,
        args.provenance,
        args.canonical_waterfall,
        args.canonical_frequency,
        args.replay_waterfall,
        args.replay_frequency,
    )
    data = BBData.from_file(str(args.h5))
    coarse_offsets = b1._alignment_offsets(data)
    dedispersed = coherent_dedisp(data, b1.DM, time_shift=False)
    coarse_frequency_ids = np.asarray(data.index_map["freq"]["id"], dtype=int)
    start, stop = (item * b1.FFTSIZE for item in b1.CROP_BLOCKS)
    crop = dedispersed[:, :, start:stop]
    baseline_spec, fine_frequencies, channel_id = _upchannel(
        crop,
        freq_id=coarse_frequency_ids,
        fftsize=b1.FFTSIZE,
        downfreq=b1.DOWNFREQ,
    )
    fine_frequencies = np.asarray(fine_frequencies)
    fine_parent = np.repeat(np.arange(crop.shape[0]), U)
    fine_offsets = coarse_offsets[fine_parent]
    baseline_power = _power_from_spec(baseline_spec).T
    gain, scallop = _folded_gain(baseline_power)
    midpoint = baseline_power.shape[1] // 2
    _, scallop_a = _folded_gain(baseline_power[:, :midpoint])
    _, scallop_b = _folded_gain(baseline_power[:, midpoint:])
    split_rms = float(np.sqrt(np.mean(((scallop_a - scallop_b) / scallop) ** 2)))

    records = []
    for band_index, band in enumerate(b1.BANDS):
        for width in b1.WIDTH_CHANNELS:
            for ratio in b1.POWER_RATIOS:
                for center_index, center in enumerate(b1.CENTERS):
                    seed = (
                        20260713
                        + 10000 * band_index
                        + 1000 * int(width)
                        + 100 * int(ratio)
                        + center_index
                    )
                    print(
                        f"trial band={band} width={width:g} ratio={ratio:g} center={center}",
                        flush=True,
                    )
                    records.extend(
                        _trial(
                            b1,
                            crop,
                            baseline_spec,
                            fine_frequencies,
                            channel_id,
                            coarse_frequency_ids,
                            fine_offsets,
                            coarse_offsets,
                            gain,
                            band=band,
                            width_bins=width,
                            power_ratio=ratio,
                            aligned_center=center,
                            seed=seed,
                        )
                    )

    route_records = {route: [r for r in records if r["route"] == route] for route in ROUTES}
    finite = [r for r in records if r.get("recovered_fit")]
    target_pass = []
    for item in route_records["matched_phase_cycled"]:
        nominal = item["nominal_width_channels"] * b1.CHANNEL_WIDTH_MHZ
        target_pass.append(abs(item["truth_fit"]["dnu_mhz"] - nominal) <= 0.10 * nominal)

    matched_noise_width = [
        _width_pass(item, b1.CHANNEL_WIDTH_MHZ)
        for item in route_records["matched_noise_free"]
    ]
    matched_phase_width = [
        _width_pass(item, b1.CHANNEL_WIDTH_MHZ)
        for item in route_records["matched_phase_cycled"]
    ]
    phase_power = [
        abs(item["power_recovery_ratio"] - 1.0) <= 0.10
        for item in route_records["matched_phase_cycled"]
    ]

    matched_error = _median_abs_fractional_error(route_records["matched_phase_cycled"])
    unmatched_error = _median_abs_fractional_error(route_records["unmatched_noise_free"])
    checks = {
        "baseline_replay": replay,
        "gain_positive_finite": {"pass": bool(np.all(np.isfinite(gain) & (gain > 0)))},
        "scallop_split_half": {
            "pass": split_rms <= 0.05,
            "rms_fractional_difference": split_rms,
            "threshold": 0.05,
        },
        "all_fits_finite": {
            "pass": len(finite) == 144,
            "n_finite": len(finite),
            "n_trials": 144,
        },
        "target_generator": {"pass": len(target_pass) == 48 and all(target_pass)},
        "matched_noise_free_width": {
            "pass": len(matched_noise_width) == 48 and all(matched_noise_width),
            "n_pass": sum(matched_noise_width),
        },
        "matched_phase_cycled_width": {
            "pass": len(matched_phase_width) == 48 and all(matched_phase_width),
            "n_pass": sum(matched_phase_width),
        },
        "matched_phase_cycled_power": {
            "pass": len(phase_power) == 48 and all(phase_power),
            "n_pass": sum(phase_power),
        },
        "matched_improves_over_unmatched": {
            "pass": bool(
                np.isfinite(matched_error)
                and np.isfinite(unmatched_error)
                and matched_error * 2.0 <= unmatched_error
            ),
            "matched_median_abs_fractional_error": matched_error,
            "unmatched_median_abs_fractional_error": unmatched_error,
        },
        "manual_review": {"pass": None, "reason": "pending visual inspection"},
    }
    automated_pass = all(
        check["pass"] is True for name, check in checks.items() if name != "manual_review"
    )
    qualification_pass = all(check["pass"] is True for check in checks.values())
    payload = {
        "experiment": "B2 matched folded-scallop phase-cycled voltage calibration",
        "qualification_status": "inconclusive",
        "science_status": "diagnostic_only",
        "on_pulse_fit_performed": False,
        "container_image_digest": "sha256:f510909d892d0d5224c982c590cbe80967a49a59b79c396ab72bb710105c4c41",
        "injection_boundary": "after coherent_dedisp; before _upchannel; synthetic sky voltage forward-folded by sqrt(separable scalar gain)",
        "scallop": {
            "upchannel_factor": U,
            "shape": scallop.tolist(),
            "split_half_shape_a": scallop_a.tolist(),
            "split_half_shape_b": scallop_b.tolist(),
        },
        "checks": checks,
        "records": records,
    }
    (args.output / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _plot(records, args.output)
    print(
        json.dumps(
            {
                "checks": checks,
                "automated_pass": automated_pass,
                "qualification_pass": qualification_pass,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if qualification_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
