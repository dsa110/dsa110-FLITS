#!/usr/bin/env python3
"""Qualify a four-stream time-disjoint high-band ACF for Freya CHIME (B4)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scintillation.scint_analysis.chime_product import (  # noqa: E402
    ChimeProductConfig,
    build_chime_products,
    burst_track_mask,
    coarse_alignment_offsets,
    load_chime_target,
)
from scintillation.scint_analysis.cross_acf import (  # noqa: E402
    CrossACF,
    blockwise_cross_acf_pairs,
    fit_cross_lorentzian,
)

DATA = Path.home() / "Data/Faber2026/dsa110/upchan_codetections/crossacf-2026-07-14"
DEFAULT_POL0 = DATA / "freya_chime_pol0_upchan.npy"
DEFAULT_POL1 = DATA / "freya_chime_pol1_upchan.npy"
DEFAULT_STOKES = DATA / "freya_chime_upchan.npy"
DEFAULT_FREQUENCIES = DATA / "freya_chime_freq.npy"
DEFAULT_METADATA = DATA / "freya_crossacf_metadata.json"
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "b4_fourstream"

BAND_MHZ = (627.0, 800.0)
LTE_EXCLUSION_MHZ = (730.0, 760.0)
OFF_PULSE = (10, 200)
BURST_WINDOW = (253, 268)
ALIGNED_BURST_BIN = 254
BURST_HALF_WIDTH = 6
MAX_LAG_BINS = 40
FIRST_LAG_BIN = 2
# Fine channels per parent coarse channel (the upchannelization factor); the
# fit model must be the block-demeaned Lorentzian expectation at this length.
CHANNELS_PER_COARSE = 64
FIT_MAXIMA_MHZ = (0.15, 0.20, 0.25)
# Grid must bracket the expected recovery: NE2025 MW floor scaled to this band
# is ~66 kHz at 713 MHz (~11 fine channels), so 3--6 channels alone would
# leave the plausible on-pulse width unvalidated.
WIDTH_CHANNELS = (3.0, 6.0, 10.0, 16.0)
MODULATION_INDICES = (0.15, 0.2, 0.3, 1.0)
N_TRIALS = 64
# Fixed scaling index for cross-subband width comparison (Kolmogorov-like
# nu^4.4, the same convention as the DM/scattering budget analyses); genuine
# scintillation widths must not be compared raw across subbands.
ALPHA_SCALING = 4.4
EXPECTED_SOURCE_H5_SHA256 = "676a9033c10926c213603939bee78c44d6d1a011c01e4279b41bccc97127df52"


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
        return _jsonable(value.tolist())
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.nan
    center = float(np.median(finite))
    return 1.4826 * float(np.median(np.abs(finite - center)))


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    """Row means without emitting warnings for fully masked channels."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    return np.divide(
        np.nansum(values, axis=1),
        count,
        out=np.full(values.shape[0], np.nan),
        where=count > 0,
    )


def _row_nanstd(values: np.ndarray) -> np.ndarray:
    """Population standard deviations without all-NaN slice warnings."""
    values = np.asarray(values, dtype=float)
    mean = _row_nanmean(values)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    squared = np.where(finite, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
    return np.sqrt(np.divide(squared, count, out=np.full(values.shape[0], np.nan), where=count > 0))


def _metadata_gate(metadata: dict, paths: dict[str, Path]) -> dict:
    """Bind the campaign to the authoritative raw observation and products."""
    products = metadata.get("products", {})
    polarizations = products.get("polarizations", [])
    declared = {
        "pol0": polarizations[0].get("sha256") if len(polarizations) > 0 else None,
        "pol1": polarizations[1].get("sha256") if len(polarizations) > 1 else None,
        "stokes": products.get("stokes_i", {}).get("sha256"),
        "frequencies": products.get("frequencies", {}).get("sha256"),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    checks = {
        "schema_version": metadata.get("schema_version") == 1,
        "target": metadata.get("target") == "freya",
        "time_shift_disabled": metadata.get("time_shift") is False,
        "source_h5_sha256": metadata.get("source_h5_sha256") == EXPECTED_SOURCE_H5_SHA256,
        "producer_sha256_well_formed": len(str(metadata.get("producer_sha256", ""))) == 64,
        "product_sha256": declared == observed,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "source_h5": metadata.get("source_h5"),
        "source_h5_sha256": metadata.get("source_h5_sha256"),
        "producer": metadata.get("producer"),
        "producer_sha256": metadata.get("producer_sha256"),
        "declared_product_sha256": declared,
        "observed_product_sha256": observed,
    }


def _channel_mask(power_by_pol: list[np.ndarray], frequencies: np.ndarray) -> np.ndarray:
    good = np.ones(frequencies.size, dtype=bool)
    start, stop = OFF_PULSE
    for power in power_by_pol:
        off = np.asarray(power[:, start:stop], dtype=float)
        gain = np.nanmedian(off, axis=1)
        fractional_rms = _row_nanstd(off) / gain
        center = float(np.nanmedian(fractional_rms))
        scale = _mad(fractional_rms)
        good &= np.isfinite(gain) & (gain > 0) & np.isfinite(fractional_rms)
        if np.isfinite(scale) and scale > 0:
            good &= np.abs(fractional_rms - center) <= 5.0 * scale
    good &= ~((frequencies >= LTE_EXCLUSION_MHZ[0]) & (frequencies <= LTE_EXCLUSION_MHZ[1]))
    return good


def _stationary_lorentzian(
    rng: np.random.Generator,
    *,
    n_channels: int,
    width_bins: float,
) -> np.ndarray:
    distances = np.minimum(np.arange(n_channels), n_channels - np.arange(n_channels))
    covariance = 1.0 / (1.0 + (distances / float(width_bins)) ** 2)
    power = np.maximum(np.real(np.fft.fft(covariance)), 0.0)
    sample = np.real(np.fft.ifft(np.fft.fft(rng.normal(size=n_channels)) * np.sqrt(power)))
    return (sample - sample.mean()) / sample.std()


def _build_polarization_product(
    power: np.ndarray,
    frequencies: np.ndarray,
    coarse_frequencies: np.ndarray,
    offsets: np.ndarray,
    good_channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.argmin(np.abs(frequencies[:, None] - coarse_frequencies[None, :]), axis=1)
    burst_mask = burst_track_mask(
        n_channels=power.shape[0],
        n_times=power.shape[1],
        channel_offsets=offsets[parent],
        aligned_center_bin=ALIGNED_BURST_BIN,
        half_width_bins=BURST_HALF_WIDTH,
    )
    rfi_mask = np.broadcast_to(~good_channels[:, None], power.shape)
    target = load_chime_target("freya")
    result = build_chime_products(
        power,
        frequencies,
        coarse_frequencies,
        coarse_offsets=offsets,
        burst_mask=burst_mask,
        rfi_mask=rfi_mask,
        config=ChimeProductConfig(
            target="freya",
            dm=float(target["dm"]),
            upchannel_factor=int(target["upchannel_factor"]),
            dt_s=2.56e-6 * 2 * int(target["upchannel_factor"]),
            off_pulse=OFF_PULSE,
            guard_bins=1,
            correction_rank=1,
            aligned_burst_bin=ALIGNED_BURST_BIN,
            burst_half_width_bins=BURST_HALF_WIDTH,
        ),
    )
    normalized = np.divide(
        result.corrected,
        result.channel_gain[:, None],
        out=np.full_like(result.corrected, np.nan, dtype=float),
        where=np.isfinite(result.channel_gain[:, None]) & (result.channel_gain[:, None] > 0),
    )
    normalized[~good_channels] = np.nan
    return normalized, result.channel_gain


def _spectrum(dynamic: np.ndarray, window: tuple[int, int], baseline: np.ndarray) -> np.ndarray:
    return _row_nanmean(dynamic[:, window[0] : window[1]]) - baseline


def _half_spectra(
    dynamic: np.ndarray, window: tuple[int, int], baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Window-relative even/odd time-sample spectra for the disjoint estimator."""
    block = dynamic[:, window[0] : window[1]]
    return (
        _row_nanmean(block[:, ::2]) - baseline,
        _row_nanmean(block[:, 1::2]) - baseline,
    )


def _window_half_spectra(
    dynamic: list[np.ndarray],
    baselines: list[np.ndarray],
    window: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        _half_spectra(item, window, baseline)
        for item, baseline in zip(dynamic, baselines, strict=True)
    ]


def _half_norms(halves: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[float, float]]:
    return [(float(np.nanmean(even)), float(np.nanmean(odd))) for even, odd in halves]


def _disjoint_cross(
    halves: list[tuple[np.ndarray, np.ndarray]],
    block_ids: np.ndarray,
    norms: list[tuple[float, float]],
) -> CrossACF | None:
    """Average every independent-time ACF across the two polarizations.

    The four pairs are Xe-Xo, Ye-Yo, Xe-Yo, and Xo-Ye.  Equal-time products
    never enter, so polarized source self-noise and common burst-time RFI have
    zero expectation, while using the within-pol pairs recovers sensitivity
    left on the table by the original cross-pol-only B3 estimator.
    Returns None instead of raising when a window is too weak to normalize,
    so callers stay fail-closed.
    """
    (x_even, x_odd), (y_even, y_odd) = halves
    (nxe, nxo), (nye, nyo) = norms
    try:
        return blockwise_cross_acf_pairs(
            [
                (x_even, x_odd, nxe, nxo),
                (y_even, y_odd, nye, nyo),
                (x_even, y_odd, nxe, nyo),
                (x_odd, y_even, nxo, nye),
            ],
            block_ids,
            max_lag_bins=MAX_LAG_BINS,
        )
    except ValueError:
        return None


def _remove_instrument_template(cross: CrossACF, controls: list[CrossACF]) -> CrossACF:
    """Subtract an independently measured lag-domain instrumental template."""
    if not controls:
        raise ValueError("at least one independent control is required")
    control_acfs = np.stack([item.acf for item in controls])
    template = np.mean(control_acfs, axis=0)
    template_covariance = (
        np.cov(control_acfs, rowvar=False, ddof=1) / len(controls)
        if len(controls) > 1
        else controls[0].covariance
    )
    covariance = np.asarray(cross.covariance) + template_covariance
    block_acfs = (
        None if cross.block_acfs is None else np.asarray(cross.block_acfs) - template[None, :]
    )
    return CrossACF(
        lag_bins=cross.lag_bins,
        acf=np.asarray(cross.acf) - template,
        error=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        n_blocks=cross.n_blocks,
        covariance=covariance,
        block_acfs=block_acfs,
    )


def _offpulse_crosses(
    dynamic: list[np.ndarray],
    baselines: list[np.ndarray],
    block_ids: np.ndarray,
    norms: list[tuple[float, float]],
) -> list[CrossACF]:
    width = BURST_WINDOW[1] - BURST_WINDOW[0]
    starts = list(range(OFF_PULSE[0], OFF_PULSE[1] - width + 1, width))[:12]
    crosses = []
    for start in starts:
        cross = _disjoint_cross(
            _window_half_spectra(dynamic, baselines, (start, start + width)),
            block_ids,
            norms,
        )
        if cross is None:
            raise ValueError(f"off-pulse control {start}:{start + width} is not measurable")
        crosses.append(cross)
    return crosses


def _offpulse_gate(
    dynamic: list[np.ndarray],
    baselines: list[np.ndarray],
    block_ids: np.ndarray,
    norms: list[tuple[float, float]],
) -> dict:
    width = BURST_WINDOW[1] - BURST_WINDOW[0]
    starts = list(range(OFF_PULSE[0], OFF_PULSE[1] - width + 1, width))[:12]
    raw_crosses = _offpulse_crosses(dynamic, baselines, block_ids, norms)
    records = []
    # A 3-sigma pointwise cutoff applied to all 480 lag/window cells has a
    # ~73% false-failure probability under white Gaussian noise.  Use the
    # two-sided 1% family-wise threshold for 480 comparisons instead.
    simultaneous_threshold = 4.254
    for index, (start, raw_cross) in enumerate(zip(starts, raw_crosses, strict=True)):
        controls = raw_crosses[:index] + raw_crosses[index + 1 :]
        cross = _remove_instrument_template(raw_cross, controls)
        finite = np.isfinite(cross.acf) & np.isfinite(cross.error) & (cross.error > 0)
        z = cross.acf[finite] / cross.error[finite]
        covariance = cross.covariance[np.ix_(finite, finite)]
        scale = float(np.nanmedian(np.diag(covariance)))
        floor = max(scale * 1e-6, 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
        chi2 = float(cross.acf[finite] @ np.linalg.solve(covariance, cross.acf[finite]))
        reduced_chi2 = chi2 / np.count_nonzero(finite)
        records.append(
            {
                "window": [start, start + width],
                "pass": bool(z.size and np.max(np.abs(z)) <= simultaneous_threshold),
                "max_abs_z": float(np.max(np.abs(z))) if z.size else None,
                "reduced_chi2": reduced_chi2,
                "acf": cross.acf.tolist(),
                "error": cross.error.tolist(),
            }
        )
    redchi = float(np.mean([record["reduced_chi2"] for record in records])) if records else np.nan
    passed = bool(
        len(records) == 12 and all(record["pass"] for record in records) and 0.5 <= redchi <= 2.0
    )
    return {
        "pass": passed,
        "aggregate_reduced_chi2": redchi,
        "thresholds": {
            "max_abs_z": simultaneous_threshold,
            "family_wise_alpha": 0.01,
            "n_comparisons": 480,
            "aggregate_reduced_chi2": [0.5, 2.0],
        },
        "instrument_template_acf": np.mean(
            np.stack([item.acf for item in raw_crosses]), axis=0
        ).tolist(),
        "records": records,
    }


def _injection_gate(
    dynamic: list[np.ndarray],
    baselines: list[np.ndarray],
    block_ids: np.ndarray,
    norms: tuple[float, float],
    channel_width_mhz: float,
    envelope: np.ndarray,
    off_sigmas: list[np.ndarray],
    offpulse_crosses: list[CrossACF],
) -> dict:
    """Dynamic-spectrum injections into real per-pol backgrounds.

    Each trial writes the common scintillation signal into both polarization
    waterfalls with the measured burst envelope, then adds the radiometer
    source-noise excess: per-sample noise scales with total power, so the
    extra standard deviation over the off-pulse level is
    sigma_off*sqrt((1+s)^2-1).  The excess draw is SHARED between the two
    polarizations -- the fully polarized worst case, in which source
    self-noise is completely correlated at equal times -- so the gate
    validates exactly the contamination channel the time-disjoint estimator
    claims to remove.  Spectra, normalizations, and fitting then follow the
    identical path used for the real on-pulse measurement.
    """
    width = BURST_WINDOW[1] - BURST_WINDOW[0]
    starts = list(range(OFF_PULSE[0], OFF_PULSE[1] - width + 1, width))[:12]
    records = []
    cells = []
    for width_channels in WIDTH_CHANNELS:
        for modulation in MODULATION_INDICES:
            cell_records = []
            for trial in range(N_TRIALS):
                seed = 20260714 + 10000 * int(width_channels) + 100 * int(10 * modulation) + trial
                rng = np.random.default_rng(seed)
                common = _stationary_lorentzian(
                    rng,
                    n_channels=block_ids.size,
                    width_bins=width_channels,
                )
                start = starts[trial % len(starts)]
                shared_noise = rng.normal(size=(block_ids.size, width))
                halves = []
                for item, baseline, sigma, norm in zip(
                    dynamic, baselines, off_sigmas, norms, strict=True
                ):
                    signal = norm * envelope[None, :] * (1.0 + modulation * common[:, None])
                    excess = np.sqrt(np.maximum((1.0 + signal) ** 2 - 1.0, 0.0))
                    injected = (
                        item[:, start : start + width]
                        + signal
                        + sigma[:, None] * excess * shared_noise
                    )
                    halves.append(
                        (
                            _row_nanmean(injected[:, ::2]) - baseline,
                            _row_nanmean(injected[:, 1::2]) - baseline,
                        )
                    )
                cross = _disjoint_cross(halves, block_ids, _half_norms(halves))
                if cross is not None:
                    control_index = trial % len(starts)
                    controls = (
                        offpulse_crosses[:control_index] + offpulse_crosses[control_index + 1 :]
                    )
                    cross = _remove_instrument_template(cross, controls)
                fit = (
                    fit_cross_lorentzian(
                        cross,
                        channel_width_mhz=channel_width_mhz,
                        first_lag_bin=FIRST_LAG_BIN,
                        fit_max_mhz=FIT_MAXIMA_MHZ[-1],
                        block_length=CHANNELS_PER_COARSE,
                    )
                    if cross is not None
                    else None
                )
                record = {
                    "width_channels": width_channels,
                    "modulation_index": modulation,
                    "trial": trial,
                    "seed": seed,
                    "fit": fit,
                }
                records.append(record)
                cell_records.append(record)
            finite = [record for record in cell_records if record["fit"] is not None]
            truth = width_channels * channel_width_mhz
            recovered = np.asarray([record["fit"]["dnu_mhz"] for record in finite])
            errors = np.asarray([record["fit"]["dnu_err_mhz"] for record in finite])
            recovered_m = np.asarray([record["fit"]["m"] for record in finite])
            width_bias = float(np.median(np.abs(recovered - truth))) if recovered.size else np.inf
            coverage = (
                float(np.mean(np.abs(recovered - truth) <= errors)) if recovered.size else 0.0
            )
            m_bias = (
                float(np.median(np.abs(recovered_m - modulation))) if recovered_m.size else np.inf
            )
            width_limit = max(0.10 * truth, 0.25 * channel_width_mhz)
            m_limit = max(0.10 * modulation, 0.05)
            passed = bool(
                len(finite) == N_TRIALS
                and width_bias < width_limit
                and 0.53 <= coverage <= 0.83
                and m_bias < m_limit
            )
            cells.append(
                {
                    "width_channels": width_channels,
                    "modulation_index": modulation,
                    "n_finite": len(finite),
                    "median_absolute_width_bias_mhz": width_bias,
                    "width_bias_limit_mhz": width_limit,
                    "coverage_68": coverage,
                    "median_absolute_modulation_bias": m_bias,
                    "modulation_bias_limit": m_limit,
                    "pass": passed,
                }
            )
    return {
        "pass": all(cell["pass"] for cell in cells),
        "n_trials": len(records),
        "cells": cells,
        "records": records,
    }


def _fit_for_selection(
    halves: list[tuple[np.ndarray, np.ndarray]],
    block_ids: np.ndarray,
    norms: list[tuple[float, float]],
    channel_width_mhz: float,
    controls: list[CrossACF] | None = None,
) -> tuple[CrossACF | None, dict[str, dict | None]]:
    cross = _disjoint_cross(halves, block_ids, norms)
    if cross is None:
        return None, {f"{fit_max:.2f}": None for fit_max in FIT_MAXIMA_MHZ}
    if controls is not None:
        cross = _remove_instrument_template(cross, controls)
    fits = {
        f"{fit_max:.2f}": fit_cross_lorentzian(
            cross,
            channel_width_mhz=channel_width_mhz,
            first_lag_bin=FIRST_LAG_BIN,
            fit_max_mhz=fit_max,
            block_length=CHANNELS_PER_COARSE,
        )
        for fit_max in FIT_MAXIMA_MHZ
    }
    return cross, fits


def _render(output: Path, result: dict, cross: CrossACF | None, on_fit: dict | None) -> list[str]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale in figure_dir.glob("freya_b4_*.png"):
        stale.unlink()
    paths = []

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for record in result["gates"]["independent_noise_null"]["records"]:
        ax.plot(np.arange(1, len(record["acf"]) + 1), record["acf"], alpha=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Fine-channel lag",
        ylabel="Cross covariance",
        title="Freya B4 high-band off-pulse ACFs",
    )
    ax.grid(alpha=0.2)
    path = figure_dir / "freya_b4_offpulse_null.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    cells = result["gates"]["injection_recovery"]["cells"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(cells))
    ax.bar(x, [cell["median_absolute_width_bias_mhz"] for cell in cells])
    ax.plot(x, [cell["width_bias_limit_mhz"] for cell in cells], "k_", ms=16, label="bias limit")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{cell['width_channels']:.0f} ch / {cell['modulation_index']:.2g}" for cell in cells],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set(
        xlabel="Injected width / modulation index",
        ylabel="Median absolute width bias (MHz)",
        title="Freya B4 real-background injection recovery",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, axis="y")
    path = figure_dir / "freya_b4_injection_recovery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    if cross is not None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        lags = cross.lag_bins * result["channel_width_mhz"] * 1e3
        ax.errorbar(
            lags,
            cross.acf,
            yerr=cross.error,
            fmt=".",
            ms=4,
            alpha=0.8,
            label="four-stream ACF",
        )
        if on_fit is not None:
            ax.plot(
                np.asarray(on_fit["fit_lags_mhz"]) * 1e3,
                on_fit["model_acf"],
                lw=2,
                label="Lorentzian fit",
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            xlabel="Frequency lag (kHz)",
            ylabel="Cross covariance",
            title="Freya 627-800 MHz four-stream time-disjoint ACF",
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        path = figure_dir / "freya_b4_onpulse_acf.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def _write_figure_manifest(output: Path, paths: list[str]) -> Path:
    expectations = {
        "freya_b4_offpulse_null.png": "template-corrected off-pulse curves scatter around zero",
        "freya_b4_injection_recovery.png": "bias bars expose the validated modulation region",
        "freya_b4_onpulse_acf.png": "on-pulse points expose the weak boundary-dependent feature",
    }
    figures = []
    for raw_path in paths:
        path = Path(raw_path)
        pixels = plt.imread(path)
        figures.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
                "pixel_shape": list(pixels.shape),
                "expectation": expectations[path.name],
            }
        )
    manifest = output / "figures.manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "path_base": "manifest_directory", "figures": figures},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, default=DEFAULT_POL0)
    parser.add_argument("--pol1", type=Path, default=DEFAULT_POL1)
    parser.add_argument("--stokes", type=Path, default=DEFAULT_STOKES)
    parser.add_argument("--frequencies", type=Path, default=DEFAULT_FREQUENCIES)
    parser.add_argument("--time0-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frequencies_full = np.load(args.frequencies)
    select = (frequencies_full >= BAND_MHZ[0]) & (frequencies_full <= BAND_MHZ[1])
    frequencies = np.asarray(frequencies_full[select], dtype=float)
    power = [
        np.asarray(np.load(path, mmap_mode="r")[select], dtype=float)
        for path in (args.pol0, args.pol1)
    ]
    stokes = np.asarray(np.load(args.stokes, mmap_mode="r")[select], dtype=float)
    parity_max = float(np.nanmax(np.abs((power[0] + power[1]) - stokes)))
    parity_scale = float(np.nanmax(np.abs(stokes)))
    producer_parity = bool(parity_max <= max(1e-5 * parity_scale, 1e-6))

    metadata = json.loads(args.time0_metadata.read_text())
    provenance = _metadata_gate(
        metadata,
        {
            "pol0": args.pol0,
            "pol1": args.pol1,
            "stokes": args.stokes,
            "frequencies": args.frequencies,
        },
    )
    coarse = np.asarray(metadata["freq_mhz"], dtype=float)
    target = load_chime_target("freya")
    dt_s = 2.56e-6 * 2 * int(target["upchannel_factor"])
    offsets = coarse_alignment_offsets(
        coarse,
        np.asarray(metadata["fpga_count"]),
        delta_time_s=float(metadata["delta_time"]),
        dm=float(target["dm"]),
        dt_s=dt_s,
    )
    good_channels = _channel_mask(power, frequencies)
    dynamic = [
        _build_polarization_product(item, frequencies, coarse, offsets, good_channels)[0]
        for item in power
    ]
    baselines = [_row_nanmean(item[:, OFF_PULSE[0] : OFF_PULSE[1]]) for item in dynamic]
    aligned_profile = np.nansum(
        np.stack(
            [item - baseline[:, None] for item, baseline in zip(dynamic, baselines, strict=True)]
        ),
        axis=(0, 1),
    )
    peak_bin = int(np.nanargmax(aligned_profile))
    burst_finite_fraction = float(
        np.mean(
            np.isfinite(np.stack([item[:, BURST_WINDOW[0] : BURST_WINDOW[1]] for item in dynamic]))
        )
    )
    offpulse_finite_fraction = float(
        np.mean(np.isfinite(np.stack([item[:, OFF_PULSE[0] : OFF_PULSE[1]] for item in dynamic])))
    )
    alignment = {
        "pass": bool(
            BURST_WINDOW[0] <= peak_bin < BURST_WINDOW[1]
            and burst_finite_fraction >= 0.75
            and offpulse_finite_fraction >= 0.75
        ),
        "peak_bin": peak_bin,
        "required_peak_window": list(BURST_WINDOW),
        "burst_finite_fraction": burst_finite_fraction,
        "offpulse_finite_fraction": offpulse_finite_fraction,
        "minimum_finite_fraction": 0.75,
    }
    on_spectra = [
        _spectrum(item, BURST_WINDOW, baseline)
        for item, baseline in zip(dynamic, baselines, strict=True)
    ]
    norms = tuple(float(np.nanmean(spectrum)) for spectrum in on_spectra)
    on_halves = _window_half_spectra(dynamic, baselines, BURST_WINDOW)
    on_half_norms = _half_norms(on_halves)
    parent = np.argmin(np.abs(frequencies[:, None] - coarse[None, :]), axis=1)
    channel_width = float(np.nanmedian(np.diff(frequencies)))

    # Measured burst envelope (window-mean 1) drives the injections; the
    # per-channel off-pulse scatter sets the radiometer source-noise scale.
    profile = np.nansum(
        [
            np.asarray(item[:, BURST_WINDOW[0] : BURST_WINDOW[1]]) - baseline[:, None]
            for item, baseline in zip(dynamic, baselines, strict=True)
        ],
        axis=0,
    )
    envelope = np.nanmean(profile, axis=0)
    if not np.isfinite(envelope).all() or float(np.mean(envelope)) <= 0:
        raise SystemExit("burst envelope is not positive; check alignment and BURST_WINDOW")
    envelope = envelope / float(np.mean(envelope))
    off_sigmas = [_row_nanstd(item[:, OFF_PULSE[0] : OFF_PULSE[1]]) for item in dynamic]

    control_crosses = _offpulse_crosses(dynamic, baselines, parent, on_half_norms)
    offpulse = _offpulse_gate(dynamic, baselines, parent, on_half_norms)
    injection = _injection_gate(
        dynamic,
        baselines,
        parent,
        norms,
        channel_width,
        envelope,
        off_sigmas,
        control_crosses,
    )
    # Always calculate the blinded diagnostic fit, even when a qualification
    # gate fails.  It remains explicitly non-scientific until every gate has
    # passed, but its recovered modulation index tells the next injection
    # campaign which sensitivity regime must be qualified.
    diagnostic_cross, diagnostic_fits = _fit_for_selection(
        on_halves, parent, on_half_norms, channel_width, control_crosses
    )
    prerequisites = (
        producer_parity
        and provenance["pass"]
        and alignment["pass"]
        and offpulse["pass"]
        and injection["pass"]
    )
    cross = None
    on_fit = None
    fit_window_gate = {"pass": False, "reason": "prerequisite gate failed", "fits": {}}
    compatibility = {"pass": False, "reason": "prerequisite gate failed", "records": []}
    width_envelope_gate = {"pass": False, "reason": "prerequisite gate failed"}
    if prerequisites:
        cross, fits = _fit_for_selection(
            on_halves, parent, on_half_norms, channel_width, control_crosses
        )
        on_fit = fits[f"{FIT_MAXIMA_MHZ[-1]:.2f}"]
        finite_fits = [fit for fit in fits.values() if fit is not None]
        widths = np.asarray([fit["dnu_mhz"] for fit in finite_fits])
        bound_clear = all(
            fit is not None
            and fit["dnu_mhz"] > 0.55 * channel_width
            and fit["dnu_mhz"] < 0.95 * float(key)
            for key, fit in fits.items()
        )
        movement = (
            float((widths.max() - widths.min()) / np.median(widths))
            if widths.size == len(FIT_MAXIMA_MHZ)
            else np.inf
        )
        fit_window_gate = {
            "pass": bool(
                len(finite_fits) == len(FIT_MAXIMA_MHZ) and bound_clear and movement < 0.20
            ),
            "max_fractional_movement": movement,
            "fits": fits,
        }

        selections = [
            ("early", np.ones(parent.size, dtype=bool), (BURST_WINDOW[0], 260)),
            ("late", np.ones(parent.size, dtype=bool), (260, BURST_WINDOW[1])),
            ("low_highband", frequencies < 713.5, BURST_WINDOW),
            ("upper_highband", frequencies >= 713.5, BURST_WINDOW),
        ]
        # Widths from disjoint subbands may not be compared raw: genuine
        # scintillation scales ~nu^4.4, an expected ~1.8x across this split.
        # Scale each subband to the good-channel mean frequency of the full
        # band before the compatibility comparison; time splits scale by 1.
        nu_reference = float(np.nanmean(frequencies[good_channels]))
        compatibility_records = []
        for name, channel_select, window in selections:
            halves = [
                (even[channel_select], odd[channel_select])
                for even, odd in _window_half_spectra(dynamic, baselines, window)
            ]
            local_cross = _disjoint_cross(halves, parent[channel_select], _half_norms(halves))
            fit = (
                fit_cross_lorentzian(
                    local_cross,
                    channel_width_mhz=channel_width,
                    first_lag_bin=FIRST_LAG_BIN,
                    fit_max_mhz=FIT_MAXIMA_MHZ[-1],
                    block_length=CHANNELS_PER_COARSE,
                )
                if local_cross is not None
                else None
            )
            selected = channel_select & good_channels
            nu_selection = float(np.nanmean(frequencies[selected])) if selected.any() else np.nan
            scale = (
                (nu_reference / nu_selection) ** ALPHA_SCALING
                if name in ("low_highband", "upper_highband") and np.isfinite(nu_selection)
                else 1.0
            )
            compatibility_records.append(
                {
                    "name": name,
                    "fit": fit,
                    "mean_frequency_mhz": nu_selection,
                    "width_scale_to_reference": scale,
                }
            )
        comparison_fits = [record["fit"] for record in compatibility_records]
        compatible = on_fit is not None and all(fit is not None for fit in comparison_fits)
        if compatible:
            for record in compatibility_records:
                fit = record["fit"]
                scale = record["width_scale_to_reference"]
                difference = abs(scale * fit["dnu_mhz"] - on_fit["dnu_mhz"])
                sigma = np.hypot(scale * fit["dnu_err_mhz"], on_fit["dnu_err_mhz"])
                compatible &= difference <= max(0.25 * on_fit["dnu_mhz"], 2.0 * sigma)
            by_name = {record["name"]: record["fit"] for record in compatibility_records}
            raw_width_increases = bool(
                by_name["upper_highband"]["dnu_mhz"] > by_name["low_highband"]["dnu_mhz"]
            )
            compatible &= raw_width_increases
        else:
            raw_width_increases = False
        compatibility = {
            "pass": bool(compatible),
            "alpha_scaling": ALPHA_SCALING,
            "reference_frequency_mhz": nu_reference,
            "raw_width_increases_with_frequency": raw_width_increases,
            "records": compatibility_records,
        }

        validated_low = WIDTH_CHANNELS[0] * channel_width
        validated_high = WIDTH_CHANNELS[-1] * channel_width
        width_envelope_gate = {
            "pass": bool(
                on_fit is not None and validated_low <= on_fit["dnu_mhz"] <= validated_high
            ),
            "validated_envelope_mhz": [validated_low, validated_high],
            "onpulse_width_mhz": None if on_fit is None else on_fit["dnu_mhz"],
        }

    machine_pass = bool(
        prerequisites
        and fit_window_gate["pass"]
        and compatibility["pass"]
        and width_envelope_gate["pass"]
    )
    result = {
        "experiment": "B4 four-stream time-disjoint high-band ACF",
        "band_mhz": list(BAND_MHZ),
        "channel_width_mhz": channel_width,
        "n_selected_channels": int(select.sum()),
        "n_good_channels": int(good_channels.sum()),
        "normalizations": list(norms),
        "source": {
            "pol0": str(args.pol0),
            "pol0_sha256": _sha256(args.pol0),
            "pol1": str(args.pol1),
            "pol1_sha256": _sha256(args.pol1),
            "stokes": str(args.stokes),
            "stokes_sha256": _sha256(args.stokes),
            "frequencies": str(args.frequencies),
            "metadata": str(args.time0_metadata),
            "metadata_sha256": _sha256(args.time0_metadata),
            "raw_h5": provenance["source_h5"],
            "raw_h5_sha256": provenance["source_h5_sha256"],
            "producer": provenance["producer"],
            "producer_sha256": provenance["producer_sha256"],
        },
        "gates": {
            "provenance": provenance,
            "producer_parity": {
                "pass": producer_parity,
                "max_absolute_difference": parity_max,
                "relative_to_peak": parity_max / parity_scale,
            },
            "burst_alignment": alignment,
            "independent_noise_null": offpulse,
            "injection_recovery": injection,
            "fit_window_stability": fit_window_gate,
            "polarization_common_signal_compatibility": compatibility,
            "onpulse_width_within_validated_envelope": width_envelope_gate,
            "manual_figure_review": {"pass": False, "status": "pending"},
        },
        "onpulse_fit": on_fit,
        "diagnostic_onpulse_fits": diagnostic_fits,
        "machine_status": "pass_pending_figure_review" if machine_pass else "documented_fail",
        "science_status": "diagnostic_only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = _render(
        args.output_dir,
        result,
        diagnostic_cross if cross is None else cross,
        diagnostic_fits[f"{FIT_MAXIMA_MHZ[-1]:.2f}"] if on_fit is None else on_fit,
    )
    result["figures"] = figures
    result["figure_manifest"] = str(_write_figure_manifest(args.output_dir, figures))
    (args.output_dir / "validation.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "machine_status": result["machine_status"],
                "gates": {name: gate["pass"] for name, gate in result["gates"].items()},
            },
            sort_keys=True,
        )
    )
    return 0 if machine_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
