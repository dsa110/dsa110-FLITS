#!/usr/bin/env python3
"""Run frozen Freya B1 complex-voltage injections inside the CHIME container."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DM = 912.4
U = 64
FFTSIZE = 2 * U
DOWNFREQ = 2
DT_NATIVE_S = 2.56e-6
DT_OUT_S = DT_NATIVE_S * FFTSIZE
CHANNEL_WIDTH_MHZ = 0.390625 / U
K_DM_S_MHZ2 = 1.0 / 2.41e-4
BANDS = ((400.0, 627.0), (627.0, 800.0))
WIDTH_CHANNELS = (2.0, 4.0, 8.0, 16.0)
POWER_RATIOS = (1.0, 4.0)
CENTERS = (18, 28, 38)
CROP_BLOCKS = (50, 100)
FIT_MAX_MHZ = 0.35


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_replay_provenance(
    h5: Path,
    provenance_path: Path,
    canonical_waterfall: Path,
    canonical_frequency: Path,
    replay_waterfall: Path,
    replay_frequency: Path,
) -> dict:
    """Bind the baseline replay to the exact voltage input and expected products."""
    provenance = json.loads(provenance_path.read_text())
    expected_input = provenance["input_h5"]["sha256"]
    expected_replay = provenance["baseline_replay"]
    input_h5_sha256 = _sha256(h5)
    canonical_waterfall_sha256 = _sha256(canonical_waterfall)
    replay_waterfall_sha256 = _sha256(replay_waterfall)
    canonical_frequency_sha256 = _sha256(canonical_frequency)
    replay_frequency_sha256 = _sha256(replay_frequency)
    replay = {
        "input_h5_match": input_h5_sha256 == expected_input,
        "input_h5_sha256": input_h5_sha256,
        "expected_input_h5_sha256": expected_input,
        "waterfall_match": canonical_waterfall_sha256 == replay_waterfall_sha256,
        "frequency_match": canonical_frequency_sha256 == replay_frequency_sha256,
        "canonical_waterfall_sha256": canonical_waterfall_sha256,
        "replay_waterfall_sha256": replay_waterfall_sha256,
        "canonical_frequency_sha256": canonical_frequency_sha256,
        "replay_frequency_sha256": replay_frequency_sha256,
    }
    replay["provenance_waterfall_match"] = (
        replay["canonical_waterfall_sha256"] == expected_replay["waterfall_sha256"]
    )
    replay["provenance_frequency_match"] = (
        replay["canonical_frequency_sha256"] == expected_replay["frequency_sha256"]
    )
    required = (
        "input_h5_match",
        "waterfall_match",
        "frequency_match",
        "provenance_waterfall_match",
        "provenance_frequency_match",
    )
    replay["pass"] = all(replay[name] for name in required)
    if not replay["pass"]:
        raise ValueError("input HDF5 or baseline replay does not match frozen provenance")
    return replay


def _qualification_pass(checks: dict) -> bool:
    """Require every gate, including manual review, to pass explicitly."""
    return all(check["pass"] is True for check in checks.values())


def _alignment_offsets(data) -> np.ndarray:
    coarse = np.asarray(data.index_map["freq"]["centre"], dtype=float)
    fpga = np.asarray(data["time0"]["fpga_count"], dtype=float)
    ctime = (fpga - fpga[-1]) * float(data.attrs["delta_time"])
    delay = ctime - K_DM_S_MHZ2 * DM * (1.0 / coarse**2 - 1.0 / 400.0**2)
    offsets = np.rint(delay / DT_OUT_S).astype(int)
    return offsets - offsets.min()


def _align(power: np.ndarray, fine_offsets: np.ndarray) -> np.ndarray:
    output = np.full(
        (power.shape[0], power.shape[1] + int(fine_offsets.max())),
        np.nan,
        dtype=float,
    )
    for channel, offset in enumerate(fine_offsets):
        output[channel, offset : offset + power.shape[1]] = power[channel]
    return output


def _stationary_gaussian(rng: np.random.Generator, n: int, width_bins: float) -> np.ndarray:
    distance = np.minimum(np.arange(n), n - np.arange(n))
    covariance = 1.0 / (1.0 + (distance / width_bins) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, n // 2 - 1)
    coefficients = np.zeros(n, dtype=complex)
    coefficients[1 : n // 2] = np.sqrt(power[1 : n // 2]) * np.exp(1j * phase)
    coefficients[-(n // 2 - 1) :] = np.conj(coefficients[1 : n // 2][::-1])
    if n % 2 == 0:
        coefficients[n // 2] = math.sqrt(power[n // 2])
    sample = np.real(np.fft.ifft(coefficients))
    return (sample - sample.mean()) / sample.std()


def _acf(spectrum: np.ndarray, max_lag: int = 57) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(spectrum, dtype=float)
    mean = float(np.nanmean(values))
    centered = values - mean
    lags = np.arange(1, min(max_lag, values.size - 20) + 1)
    acf = []
    for lag in lags:
        products = centered[:-lag] * centered[lag:]
        products = products[np.isfinite(products)]
        acf.append(float(np.mean(products) / mean**2) if products.size >= 20 else np.nan)
    acf = np.asarray(acf)
    return lags * CHANNEL_WIDTH_MHZ, acf


def _fit_width(spectrum: np.ndarray, channel_ids: np.ndarray | None = None) -> dict | None:
    if channel_ids is not None:
        ids = np.asarray(channel_ids, dtype=int)
        if ids.shape != np.asarray(spectrum).shape or np.unique(ids).size != ids.size:
            raise ValueError("channel_ids must be unique and match the spectrum")
        full = np.full(int(ids.max() - ids.min() + 1), np.nan, dtype=float)
        full[ids - ids.min()] = spectrum
        spectrum = full
    if np.count_nonzero(np.isfinite(spectrum)) < 100 or not np.isfinite(np.nanmean(spectrum)):
        return None
    lags, acf = _acf(spectrum)
    keep = np.isfinite(acf) & (lags >= 1.5 * CHANNEL_WIDTH_MHZ) & (lags <= FIT_MAX_MHZ)
    x = lags[keep]
    y = acf[keep]
    if x.size < 20:
        return None

    def residual(parameters):
        gamma, amplitude, constant = parameters
        return y - (amplitude / (1.0 + (x / gamma) ** 2) + constant)

    candidates = []
    initial_amplitude = min(max(float(np.nanmax(y)), 0.05), 9.0)
    for start in np.asarray(WIDTH_CHANNELS) * CHANNEL_WIDTH_MHZ:
        fit = least_squares(
            residual,
            x0=(start, initial_amplitude, 0.0),
            bounds=((0.25 * CHANNEL_WIDTH_MHZ, 0.0, -5.0), (FIT_MAX_MHZ, 10.0, 5.0)),
            max_nfev=3000,
        )
        if fit.success and np.all(np.isfinite(fit.x)):
            candidates.append(fit)
    if not candidates:
        return None
    fit = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    return {
        "dnu_mhz": float(fit.x[0]),
        "amplitude": float(fit.x[1]),
        "constant": float(fit.x[2]),
        "lags_mhz": x.tolist(),
        "acf": y.tolist(),
    }


def _make_target(rng: np.random.Generator, n: int, width_bins: float) -> np.ndarray:
    gaussian = _stationary_gaussian(rng, n, width_bins)
    modulation = min(0.20, 0.90 / abs(float(np.min(gaussian))))
    return 1.0 + modulation * gaussian


def _inject_trial(
    dedispersed_crop: np.ndarray,
    baseline_spec: np.ndarray,
    fine_frequencies: np.ndarray,
    fine_channel_ids: np.ndarray,
    coarse_frequency_ids: np.ndarray,
    fine_offsets: np.ndarray,
    coarse_offsets: np.ndarray,
    *,
    band: tuple[float, float],
    width_bins: float,
    power_ratio: float,
    aligned_center: int,
    seed: int,
):
    from baseband_analysis.core.sampling import _upchannel  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    selected = (fine_frequencies >= band[0]) & (fine_frequencies < band[1])
    selected_ids = np.asarray(fine_channel_ids[selected], dtype=int)
    target_full = _make_target(rng, int(selected_ids.max() - selected_ids.min() + 1), width_bins)
    target = target_full[selected_ids - selected_ids.min()]
    truth_fit = _fit_width(target, selected_ids)
    if truth_fit is None:
        return None

    baseline_power = np.abs(baseline_spec[0]) ** 2 + np.abs(baseline_spec[1]) ** 2
    baseline_aligned = _align(baseline_power.T, fine_offsets)
    valid_off = np.ones(baseline_aligned.shape[1], dtype=bool)
    valid_off[max(0, aligned_center - 3) : aligned_center + 4] = False
    gain = np.nanmedian(baseline_aligned[:, valid_off], axis=1)
    noise_level = float(np.nanmedian(baseline_aligned[selected, aligned_center]))
    signal_scale = math.sqrt(power_ratio * noise_level)

    injected = dedispersed_crop.copy()
    target_cursor = 0
    n_coarse = injected.shape[0]
    for coarse_index in range(n_coarse):
        fine_slice = slice(coarse_index * U, (coarse_index + 1) * U)
        use = selected[fine_slice]
        if not np.any(use):
            continue
        block_target = np.zeros(U)
        count = int(np.count_nonzero(use))
        block_target[use] = target[target_cursor : target_cursor + count]
        target_cursor += count
        amplitude = signal_scale * np.sqrt(block_target)
        for delta in (-1, 0, 1):
            phase = rng.uniform(0.0, 2.0 * np.pi, U)
            fine_voltage = amplitude * np.exp(1j * phase)
            fft_bins = np.repeat(fine_voltage, DOWNFREQ)
            timeseries = np.fft.ifft(np.fft.ifftshift(fft_bins)).astype(injected.dtype)
            block = aligned_center + delta - int(coarse_offsets[coarse_index])
            start = block * FFTSIZE
            injected[coarse_index, 0, start : start + FFTSIZE] += timeseries

    spec, recovered_freq, channel_id = _upchannel(
        injected,
        freq_id=coarse_frequency_ids,
        fftsize=FFTSIZE,
        downfreq=DOWNFREQ,
    )
    recovered_power = np.abs(spec[0]) ** 2 + np.abs(spec[1]) ** 2
    recovered_aligned = _align(recovered_power.T, fine_offsets)
    baseline_norm = baseline_aligned / gain[:, None]
    recovered_norm = recovered_aligned / gain[:, None]
    paired = np.nanmean(
        recovered_norm[:, aligned_center - 1 : aligned_center + 2]
        - baseline_norm[:, aligned_center - 1 : aligned_center + 2],
        axis=1,
    )
    recovered_signal = paired[selected]
    recovered_fit = _fit_width(recovered_signal, selected_ids)
    expected_mean = float(
        np.nanmean(power_ratio * noise_level * target / gain[selected])
    )
    power_recovery = float(np.nanmean(recovered_signal) / expected_mean)
    return {
        "band_mhz": list(band),
        "nominal_width_channels": width_bins,
        "power_ratio": power_ratio,
        "aligned_center": aligned_center,
        "seed": seed,
        "truth_fit": truth_fit,
        "recovered_fit": recovered_fit,
        "power_recovery_ratio": power_recovery,
        "target_spectrum": target.tolist(),
        "recovered_spectrum": recovered_signal.tolist(),
        "recovered_frequency_check": bool(np.allclose(recovered_freq, fine_frequencies)),
        "channel_id_check": bool(channel_id.size == fine_frequencies.size),
    }


def _plot(records: list[dict], output: Path) -> None:
    finite = [item for item in records if item and item["recovered_fit"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {1.0: "#1f77b4", 4.0: "#2ca02c"}
    markers = {400.0: "o", 627.0: "s"}
    for band in BANDS:
        for ratio in POWER_RATIOS:
            chosen = [
                item
                for item in finite
                if item["power_ratio"] == ratio and item["band_mhz"][0] == band[0]
            ]
            truth = np.asarray([item["truth_fit"]["dnu_mhz"] for item in chosen]) * 1e3
            recovered = np.asarray([item["recovered_fit"]["dnu_mhz"] for item in chosen]) * 1e3
            label = f"{band[0]:.0f}-{band[1]:.0f} MHz; ratio {ratio:g}"
            axes[0].scatter(
                truth,
                recovered,
                s=30,
                alpha=0.75,
                color=colors[ratio],
                marker=markers[band[0]],
                label=label,
            )
            axes[1].scatter(
                truth,
                [item["power_recovery_ratio"] for item in chosen],
                s=30,
                alpha=0.75,
                color=colors[ratio],
                marker=markers[band[0]],
            )
    limit = max(120.0, axes[0].get_xlim()[1], axes[0].get_ylim()[1])
    axes[0].plot([0, limit], [0, limit], "k--", lw=1.5, label="identity")
    axes[0].set(xlabel="Fitted target HWHM (kHz)", ylabel="Recovered HWHM (kHz)")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.5)
    axes[1].set(xlabel="Fitted target HWHM (kHz)", ylabel="Recovered / injected signal power")
    fig.suptitle("Freya B1 pre-upchannelization voltage-injection recovery")
    fig.tight_layout()
    fig.savefig(output / "freya_b1_voltage_recovery.svg")
    fig.savefig(output / "freya_b1_voltage_recovery.png", dpi=180)
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

    args.output.mkdir(parents=True, exist_ok=True)
    replay = _verify_replay_provenance(
        args.h5,
        args.provenance,
        args.canonical_waterfall,
        args.canonical_frequency,
        args.replay_waterfall,
        args.replay_frequency,
    )

    data = BBData.from_file(str(args.h5))
    coarse_offsets = _alignment_offsets(data)
    dedispersed = coherent_dedisp(data, DM, time_shift=False)
    coarse_frequency_ids = np.asarray(data.index_map["freq"]["id"], dtype=int)
    start, stop = (item * FFTSIZE for item in CROP_BLOCKS)
    crop = dedispersed[:, :, start:stop]
    baseline_spec, fine_frequencies, channel_id = _upchannel(
        crop,
        freq_id=coarse_frequency_ids,
        fftsize=FFTSIZE,
        downfreq=DOWNFREQ,
    )
    fine_frequencies = np.asarray(fine_frequencies)
    fine_parent = np.repeat(np.arange(crop.shape[0]), U)
    fine_offsets = coarse_offsets[fine_parent]

    records = []
    for band_index, band in enumerate(BANDS):
        for width in WIDTH_CHANNELS:
            for ratio in POWER_RATIOS:
                for center_index, center in enumerate(CENTERS):
                    seed = 20260713 + 10000 * band_index + 1000 * int(width) + 100 * int(ratio) + center_index
                    print(f"trial band={band} width={width:g} ratio={ratio:g} center={center}", flush=True)
                    records.append(
                        _inject_trial(
                            crop,
                            baseline_spec,
                            fine_frequencies,
                            channel_id,
                            coarse_frequency_ids,
                            fine_offsets,
                            coarse_offsets,
                            band=band,
                            width_bins=width,
                            power_ratio=ratio,
                            aligned_center=center,
                            seed=seed,
                        )
                    )

    finite = [item for item in records if item and item["recovered_fit"]]
    nominal_pass = []
    width_pass = []
    power_pass = []
    for item in finite:
        nominal = item["nominal_width_channels"] * CHANNEL_WIDTH_MHZ
        truth = item["truth_fit"]["dnu_mhz"]
        recovered = item["recovered_fit"]["dnu_mhz"]
        nominal_pass.append(abs(truth - nominal) <= 0.10 * nominal)
        width_pass.append(abs(recovered - truth) < max(0.10 * truth, 0.25 * CHANNEL_WIDTH_MHZ))
        tolerance = 0.20 if item["power_ratio"] == 1.0 else 0.10
        power_pass.append(abs(item["power_recovery_ratio"] - 1.0) <= tolerance)

    checks = {
        "baseline_replay": replay,
        "all_fits_finite": {"pass": len(finite) == 48, "n_finite": len(finite), "n_trials": 48},
        "target_generator": {"pass": len(nominal_pass) == 48 and all(nominal_pass)},
        "width_recovery": {"pass": len(width_pass) == 48 and all(width_pass), "n_pass": sum(width_pass)},
        "power_recovery": {"pass": len(power_pass) == 48 and all(power_pass), "n_pass": sum(power_pass)},
        "manual_review": {"pass": None, "reason": "pending visual inspection"},
    }
    automated_checks_pass = all(
        check["pass"] is True for name, check in checks.items() if name != "manual_review"
    )
    qualification_pass = _qualification_pass(checks)
    payload = {
        "experiment": "B1 pre-upchannelization complex-voltage calibration",
        "qualification_status": (
            "inconclusive"
            if checks["manual_review"]["pass"] is None
            else ("pass" if qualification_pass else "fail")
        ),
        "science_status": "diagnostic_only",
        "on_pulse_fit_performed": False,
        "container_image_digest": "sha256:f510909d892d0d5224c982c590cbe80967a49a59b79c396ab72bb710105c4c41",
        "injection_boundary": "after coherent_dedisp; before _upchannel, Stokes I, normalization, and padded alignment",
        "checks": checks,
        "records": records,
    }
    (args.output / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot(records, args.output)
    print(
        json.dumps(
            {
                "checks": checks,
                "automated_checks_pass": automated_checks_pass,
                "qualification_pass": qualification_pass,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if qualification_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
