"""Notebook-derived scintillation bandwidth measurement for Freya.

This module keeps the useful part of ``scint_freya.ipynb``: extract an on-pulse
spectrum, measure its frequency ACF, fit the central Lorentzian width as
``Delta nu_d``, and write the diagnostic figures needed to audit the result.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from . import config as config_module
from .analysis import calculate_acf
from .core import ACF, DynamicSpectrum

log = logging.getLogger(__name__)

# Relative tolerance for declaring a fitted Lorentzian width pinned at a
# curve_fit bound; TRF converges asymptotically close to, not exactly on, a
# bound, so an equality test would miss real boundary hits.
_GAMMA_BOUND_REL_TOL = 0.01

# Minimum off-pulse time bins required to estimate a per-channel bandpass by
# averaging. Fewer bins leave the estimate noise-dominated, and flat-fielding
# would then imprint that noise as gain rather than removing the true bandpass.
# Mirrors the >50-bin off-pulse guard used for polynomial baseline subtraction
# (pipeline.py / issue #118).
_MIN_BANDPASS_OFF_BINS = 50

# A fine channel whose mean off-pulse level sits below this fraction of the
# median off-pulse level is gain-starved: dividing by it amplifies noise
# without bound, so it is masked rather than normalised.
_BANDPASS_FLOOR_FRAC = 1.0e-3

# Relative spread of channel spacings below which a frequency grid counts as
# uniform. Gapped grids are not subtle (a single missing channel doubles one
# spacing), so this only needs to absorb float rounding in the stored axis.
_GRID_UNIFORM_REL_TOL = 1.0e-3

# Snap distance above this fraction of a native channel is worth surfacing in
# the log: nearest-grid placement bounds every snap at 0.5 channel by
# construction, but a large population of near-half-channel snaps (freya
# CHIME hi: 49% of channels >0.25, max 0.4996) documents real inter-block
# registration drift in the upstream upchannelized product.
_GRID_SNAP_REPORT_FRAC = 0.25


@dataclass(frozen=True)
class FigureRecord:
    kind: str
    path: str
    description: str


@dataclass(frozen=True)
class ACFBandwidthResult:
    """Result of the ACF Lorentzian bandwidth fit.

    ``modulation_index`` is std/mean of the on-pulse spectrum WITHOUT
    off-level subtraction: the radiometer noise floor sits in both numerator
    and denominator, so it is a noise-diluted diagnostic, not the physical
    modulation depth (freya CHIME pass-2: 0.187 diluted vs ~0.52 physical).
    ``modulation_index_acf`` is the physical estimate sqrt(zero-lag ACF
    amplitude above baseline) from the Lorentzian fit; ``calculate_acf``
    normalizes by (mean_on - off_mean)^2, so the fitted amplitude ~ m^2.
    That reading is only valid when an off-pulse mean was supplied -- without
    one the ACF denominator falls back to mean_on^2 and the amplitude is
    floor-diluted like the std/mean diagnostic -- so the field is None unless
    the fit succeeded AND ``off_burst_spectrum_mean`` was provided.
    """

    success: bool
    delta_nu_mhz: float | None
    delta_nu_err_mhz: float | None
    modulation_index: float
    modulation_index_acf: float | None
    channel_width_mhz: float
    fit_lag_mhz: float
    max_lag_mhz: float
    message: str
    lags_mhz: list[float | None]
    acf: list[float | None]
    acf_model: list[float | None]


@dataclass(frozen=True)
class StructureBandwidthResult:
    method: str
    delta_nu_mhz: float
    lag_index: int
    channel_width_mhz: float
    frequency_lags_mhz: list[float | None]
    structure_function: list[float | None]


@dataclass(frozen=True)
class NotebookStyleResult:
    """Full analysis output.

    ``fit_window_scan`` (optional, from ``analysis.fitting.fit_lag_scan_mhz``)
    refits the ACF per fit window; ``fit_window_systematic_mhz`` is the spread
    (max - min) of the successful widths. The freya CHIME grid carries
    burst-locked instrumental structure near the coarse-channel spacing inside
    the default fit window, so the window choice is a systematic comparable to
    the statistical error (measured 44.7/57.2/67.7 kHz at 1.0/0.3/0.2 MHz) and
    must be reported with the number, not hidden by one window choice.
    """

    burst_id: str
    burst_lims: tuple[int, int]
    off_pulse_lims: tuple[int, int] | None
    acf: ACFBandwidthResult
    structure: StructureBandwidthResult
    figures: list[FigureRecord]
    result_path: str | None
    fit_window_scan: list[dict] | None = None
    fit_window_systematic_mhz: float | None = None


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _lorentzian_with_baseline(lag_mhz, amplitude, gamma_mhz, baseline):
    return amplitude / (1.0 + (lag_mhz / gamma_mhz) ** 2) + baseline


def _finite_float_list(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def _normalise_spectrum(spectrum: np.ndarray | np.ma.MaskedArray) -> tuple[np.ma.MaskedArray, float]:
    masked = np.ma.masked_invalid(np.ma.asarray(spectrum, dtype=float))
    if masked.count() == 0:
        raise ValueError("spectrum contains no finite samples")
    median = float(np.ma.median(masked))
    if median == 0.0:
        return masked, 1.0
    return masked / median, median


def measure_scintillation_bandwidth(
    spectrum: np.ndarray | np.ma.MaskedArray,
    *,
    channel_width_mhz: float,
    off_burst_spectrum_mean: float | None = None,
    max_lag_mhz: float = 20.0,
    fit_lag_mhz: float = 2.0,
) -> ACFBandwidthResult:
    """Measure ``Delta nu_d`` from the frequency ACF Lorentzian HWHM.

    The zero-lag point is excluded from the fit, matching the notebook and the
    Nimmo-style workflow where the zero-lag noise spike is not part of the
    scintillation component.
    """
    if channel_width_mhz <= 0:
        raise ValueError("channel_width_mhz must be positive")
    if max_lag_mhz <= channel_width_mhz:
        raise ValueError("max_lag_mhz must exceed one channel")
    if fit_lag_mhz <= channel_width_mhz:
        raise ValueError("fit_lag_mhz must exceed one channel")

    spec, scale = _normalise_spectrum(spectrum)
    normalised_off_mean = (
        None if off_burst_spectrum_mean is None else float(off_burst_spectrum_mean) / scale
    )
    valid = spec.compressed()
    modulation_index = float(np.nanstd(valid) / np.nanmean(valid)) if valid.size else float("nan")
    max_lag_bins = max(3, int(max_lag_mhz / channel_width_mhz))
    acf_obj = calculate_acf(
        spec,
        channel_width_mhz,
        off_burst_spectrum_mean=normalised_off_mean,
        max_lag_bins=max_lag_bins,
    )
    if acf_obj is None:
        return ACFBandwidthResult(
            success=False,
            delta_nu_mhz=None,
            delta_nu_err_mhz=None,
            modulation_index=modulation_index,
            modulation_index_acf=None,
            channel_width_mhz=float(channel_width_mhz),
            fit_lag_mhz=float(fit_lag_mhz),
            max_lag_mhz=float(max_lag_mhz),
            message="ACF calculation failed",
            lags_mhz=[],
            acf=[],
            acf_model=[],
        )

    lags = np.asarray(acf_obj.lags, dtype=float)
    acf_vals = np.asarray(acf_obj.acf, dtype=float)
    fit_mask = (np.abs(lags) > 0.5 * channel_width_mhz) & (np.abs(lags) <= fit_lag_mhz)
    fit_mask &= np.isfinite(lags) & np.isfinite(acf_vals)
    if int(fit_mask.sum()) < 5:
        return _failed_fit_result(
            acf_obj,
            modulation_index,
            channel_width_mhz,
            fit_lag_mhz,
            max_lag_mhz,
            "fewer than five finite ACF lag points in fit range",
        )

    x = lags[fit_mask]
    y = acf_vals[fit_mask]
    err = (
        np.asarray(acf_obj.err, dtype=float)[fit_mask]
        if getattr(acf_obj, "err", None) is not None
        else np.full_like(y, 0.05)
    )
    err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err[err > 0]))
    err = np.where(np.isfinite(err) & (err > 0), err, 0.05)

    amplitude0 = max(float(np.nanmax(y) - np.nanmedian(y)), 1.0e-3)
    gamma0 = max(2.0 * channel_width_mhz, min(fit_lag_mhz / 3.0, fit_lag_mhz))
    baseline0 = float(np.nanmedian(y[-min(5, y.size) :]))
    try:
        popt, pcov = curve_fit(
            _lorentzian_with_baseline,
            x,
            y,
            sigma=err,
            p0=[amplitude0, gamma0, baseline0],
            bounds=([0.0, 0.25 * channel_width_mhz, -np.inf], [np.inf, fit_lag_mhz, np.inf]),
            maxfev=20000,
        )
    except Exception as exc:
        return _failed_fit_result(
            acf_obj,
            modulation_index,
            channel_width_mhz,
            fit_lag_mhz,
            max_lag_mhz,
            f"Lorentzian fit failed: {type(exc).__name__}",
        )

    model = _lorentzian_with_baseline(lags, *popt)
    gamma = float(abs(popt[1]))
    if not np.all(np.isfinite(pcov)):
        # A non-finite covariance means curve_fit could not characterise the
        # minimum; the width is uninformative and must not read as a measurement.
        return _failed_fit_result(
            acf_obj,
            modulation_index,
            channel_width_mhz,
            fit_lag_mhz,
            max_lag_mhz,
            f"uninformative fit: non-finite covariance (gamma={gamma:.4g} MHz)",
            acf_model=_finite_float_list(model),
        )
    # gamma is hard-bounded to [0.25*channel_width, fit_lag_mhz] in curve_fit; a
    # solution pinned at either bound is a limit, not a measurement (freya's
    # config has fit_lagrange_mhz=25 while stored fits include a ~259 MHz wide
    # component, so the upper-bound case is realistic).
    gamma_lower_mhz = 0.25 * channel_width_mhz
    if gamma >= (1.0 - _GAMMA_BOUND_REL_TOL) * fit_lag_mhz:
        return _failed_fit_result(
            acf_obj,
            modulation_index,
            channel_width_mhz,
            fit_lag_mhz,
            max_lag_mhz,
            f"gamma at fit-range upper bound (gamma={gamma:.4g} MHz, "
            f"fit_lag_mhz={fit_lag_mhz:.4g}): treat as a lower limit on Delta nu_d, "
            "not a measurement; widen fit_lag_mhz to resolve",
            acf_model=_finite_float_list(model),
        )
    if gamma <= (1.0 + _GAMMA_BOUND_REL_TOL) * gamma_lower_mhz:
        return _failed_fit_result(
            acf_obj,
            modulation_index,
            channel_width_mhz,
            fit_lag_mhz,
            max_lag_mhz,
            f"gamma at lower bound (gamma={gamma:.4g} MHz, "
            f"0.25*channel_width={gamma_lower_mhz:.4g}): bandwidth unresolved "
            "by channelization",
            acf_model=_finite_float_list(model),
        )
    perr = np.sqrt(np.diag(pcov))
    gamma_err = float(perr[1]) if np.isfinite(perr[1]) else None

    return ACFBandwidthResult(
        success=True,
        delta_nu_mhz=gamma,
        delta_nu_err_mhz=gamma_err,
        modulation_index=modulation_index,
        modulation_index_acf=(
            float(np.sqrt(max(float(popt[0]), 0.0)))
            if off_burst_spectrum_mean is not None
            else None
        ),
        channel_width_mhz=float(channel_width_mhz),
        fit_lag_mhz=float(fit_lag_mhz),
        max_lag_mhz=float(max_lag_mhz),
        message="ok",
        lags_mhz=_finite_float_list(lags),
        acf=_finite_float_list(acf_vals),
        acf_model=_finite_float_list(model),
    )


def _failed_fit_result(
    acf_obj: ACF,
    modulation_index: float,
    channel_width_mhz: float,
    fit_lag_mhz: float,
    max_lag_mhz: float,
    message: str,
    acf_model: list[float | None] | None = None,
) -> ACFBandwidthResult:
    return ACFBandwidthResult(
        success=False,
        delta_nu_mhz=None,
        delta_nu_err_mhz=None,
        modulation_index=modulation_index,
        modulation_index_acf=None,
        channel_width_mhz=float(channel_width_mhz),
        fit_lag_mhz=float(fit_lag_mhz),
        max_lag_mhz=float(max_lag_mhz),
        message=message,
        lags_mhz=_finite_float_list(np.asarray(acf_obj.lags, dtype=float)),
        acf=_finite_float_list(np.asarray(acf_obj.acf, dtype=float)),
        acf_model=acf_model if acf_model is not None else [],
    )


def estimate_structure_bandwidth(
    spectrum: np.ndarray | np.ma.MaskedArray,
    *,
    channel_width_mhz: float,
    method: str = "half_power",
) -> StructureBandwidthResult:
    """Estimate bandwidth from a 1-D second-order structure function."""
    if method != "half_power":
        raise ValueError("only method='half_power' is currently supported")
    spec, _scale = _normalise_spectrum(spectrum)
    values = spec.filled(np.nan)
    valid = np.isfinite(values)
    if valid.sum() < 20:
        raise ValueError("at least 20 finite spectrum samples are required")
    values = values - float(np.nanmean(values[valid]))

    n = values.size
    structure = np.full(n, np.nan, dtype=float)
    structure[0] = 0.0
    for lag_index in range(1, n):
        pair_valid = valid[:-lag_index] & valid[lag_index:]
        if np.any(pair_valid):
            delta = values[lag_index:][pair_valid] - values[:-lag_index][pair_valid]
            structure[lag_index] = float(np.nanmean(delta**2))
    tail = structure[int(0.8 * n) :]
    finite_tail = tail[np.isfinite(tail)]
    finite_structure = structure[np.isfinite(structure)]
    if finite_tail.size:
        d_inf = float(np.nanmedian(finite_tail))
    elif finite_structure.size:
        d_inf = float(np.nanmax(finite_structure))
    else:
        d_inf = 0.0
    threshold = max(d_inf / 2.0, 0.0)
    crossing = np.where(np.isfinite(structure) & (structure >= threshold))[0]
    lag_index = int(crossing[0]) if crossing.size else n - 1
    if lag_index == 0 and crossing.size > 1:
        lag_index = int(crossing[1])
    delta_nu_mhz = float(lag_index * channel_width_mhz)
    freq_lags = np.arange(n, dtype=float) * channel_width_mhz
    return StructureBandwidthResult(
        method=method,
        delta_nu_mhz=delta_nu_mhz,
        lag_index=lag_index,
        channel_width_mhz=float(channel_width_mhz),
        frequency_lags_mhz=_finite_float_list(freq_lags),
        structure_function=_finite_float_list(structure),
    )


def determine_windows(
    spectrum: DynamicSpectrum,
    cfg: dict,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    rfi_cfg = cfg.get("analysis", {}).get("rfi_masking", {})
    manual_on = rfi_cfg.get("manual_burst_window")
    if manual_on and len(manual_on) == 2:
        burst_lims = (int(manual_on[0]), int(manual_on[1]))
    else:
        found = spectrum.find_burst_envelope(
            thres=rfi_cfg.get("find_burst_thres", 5.0),
            padding_factor=rfi_cfg.get("padding_factor", 0.2),
        )
        burst_lims = (int(found[0]), int(found[1]))

    manual_off = rfi_cfg.get("manual_noise_window")
    if manual_off and len(manual_off) == 2:
        off_lims = (int(manual_off[0]), int(manual_off[1]))
    else:
        off_end = max(0, burst_lims[0] - 200)
        off_lims = (max(0, off_end - 500), off_end)
    return burst_lims, off_lims


def normalize_bandpass(
    spectrum: DynamicSpectrum,
    off_pulse_lims: tuple[int, int],
    *,
    floor_frac: float = _BANDPASS_FLOOR_FRAC,
) -> DynamicSpectrum:
    """Flat-field: divide every fine channel by its mean off-pulse level.

    A per-fine-channel off-pulse mean *is* the static instrumental bandpass
    (the system gain, including the PFB coarse-channel scallop), because that
    gain multiplies the off-pulse noise and the on-pulse burst identically.
    Dividing the whole dynamic spectrum by it cancels the scallop exactly while
    leaving the (time-variable) scintillation imprinted on the burst untouched --
    unlike a low-order polynomial baseline, which cannot follow a periodic
    scallop at all. Pass-1 freya_chime inspection: the frequency ACF was
    dominated by a ~0.39 MHz coarse-channel ripple ~7x the fit baseline, static
    in time; only a multiplicative flat-field removes it.

    Guard rails mirror the #118 baseline-subtraction philosophy: fail loudly on
    an off-pulse window too short to average down noise, and mask (never divide
    by) channels whose off-pulse mean is non-positive, non-finite, or gain-
    starved relative to the median.
    """
    start, end = int(off_pulse_lims[0]), int(off_pulse_lims[1])
    n_off = end - start
    if n_off < _MIN_BANDPASS_OFF_BINS:
        raise ValueError(
            f"bandpass normalization needs >= {_MIN_BANDPASS_OFF_BINS} off-pulse "
            f"time bins to estimate the per-channel gain, got {n_off} "
            f"(off-pulse window [{start}, {end}))"
        )

    off_mean = np.ma.filled(np.ma.mean(spectrum.power[:, start:end], axis=1), np.nan)
    finite_positive = off_mean[np.isfinite(off_mean) & (off_mean > 0.0)]
    if finite_positive.size == 0:
        raise ValueError(
            "bandpass normalization: off-pulse window has no finite, positive "
            "channel means to define a reference level"
        )
    floor = float(floor_frac) * float(np.median(finite_positive))
    bad_channel = ~(np.isfinite(off_mean) & (off_mean > floor))

    # Bad channels carry a placeholder gain of 1.0 only to keep the division
    # finite; they are masked below and never read as data.
    gain = np.where(bad_channel, 1.0, off_mean)
    normalised = spectrum.power.data / gain[:, np.newaxis]

    if spectrum.power.mask is np.ma.nomask or np.isscalar(spectrum.power.mask):
        base_mask = np.zeros(spectrum.power.shape, dtype=bool)
    else:
        base_mask = spectrum.power.mask.copy()
    final_mask = base_mask | np.broadcast_to(bad_channel[:, None], spectrum.power.shape)

    n_masked = int(bad_channel.sum())
    if n_masked:
        log.info("Bandpass normalization masked %d gain-starved channel(s).", n_masked)
    new_power = np.ma.MaskedArray(normalised, mask=final_mask)
    return DynamicSpectrum(new_power, spectrum.frequencies.copy(), spectrum.times.copy())


def _grid_stretch_ratio(frequencies: np.ndarray) -> tuple[float, float, float]:
    """Return (native step, mean step, mean/native ratio) for a frequency axis.

    The native step is the median spacing: on a gapped-but-commensurate grid
    the overwhelming majority of spacings are the true channel width (freya
    CHIME hi: 26,466 of 26,527), so the median is immune to the gap junctions
    that corrupt the mean.
    """
    diffs = np.diff(np.asarray(frequencies, dtype=float))
    native = float(np.median(diffs))
    mean = float(np.mean(diffs))
    return native, mean, mean / native


def regularize_frequency_grid(spectrum: DynamicSpectrum) -> DynamicSpectrum:
    """Re-embed a gapped frequency grid onto the full uniform native grid.

    ``calculate_acf`` correlates by channel INDEX and labels lags with
    ``channel_width_mhz`` = mean(diff). On a grid with missing channels that
    mislabels every lag by mean/native (freya CHIME hi: 1.2340x) and, worse,
    index pairs straddling a gap mix physically distant channels into low-lag
    bins -- a shape distortion no post-hoc axis rescale can undo (measured:
    naive /1.234 gives 40.2 kHz where the regridded ACF gives 44.7 kHz).
    Re-embedding every channel at its physical grid position with masked
    fillers restores index lag == physical lag; the masked-array ACF already
    handles the missing rows.

    Channels are snapped to the nearest uniform-grid position: upchannelized
    blocks are internally uniform, but inter-block offsets drift by up to half
    a fine step (freya_chime_hi: max 3.05 kHz = 0.4996 channel), so exact grid
    membership cannot be assumed. Nearest-grid placement bounds every snap at
    half a channel by construction; two channels claiming the same grid
    position (a compressed or corrupt axis) is the real failure mode and is
    rejected, so regularization can place channels but never merge them.

    Returns ``spectrum`` unchanged when the grid is already uniform.
    Experiment provenance: Faber2026
    docs/rse/specs/experiment-freya-chime-dnu-science-readiness.md (E1/E2).
    """
    freqs = np.asarray(spectrum.frequencies, dtype=float)
    if freqs.size < 3:
        return spectrum
    native, mean, ratio = _grid_stretch_ratio(freqs)
    if native <= 0:
        raise ValueError("frequency axis is not strictly increasing")
    diffs = np.diff(freqs)
    if float(np.max(np.abs(diffs - native))) <= _GRID_UNIFORM_REL_TOL * native:
        return spectrum

    n_full = int(round((freqs[-1] - freqs[0]) / native)) + 1
    idx = np.round((freqs - freqs[0]) / native).astype(int)
    if np.unique(idx).size != idx.size:
        raise ValueError(
            "frequency grid regularization: two channels snap to the same grid "
            "position; the axis is compressed or corrupt (or the native-step "
            "estimate from the median spacing is wrong)"
        )
    snap_err = np.abs(freqs - (freqs[0] + native * idx))
    frac_large = float(np.mean(snap_err > _GRID_SNAP_REPORT_FRAC * native))
    log.info(
        "Grid snap distances: max %.4g channel, %.0f%% of channels beyond %.2f "
        "channel (inter-block registration drift in the upstream product).",
        float(np.max(snap_err)) / native,
        100.0 * frac_large,
        _GRID_SNAP_REPORT_FRAC,
    )

    power_full = np.full((n_full, spectrum.power.shape[1]), np.nan, dtype=float)
    power_full[idx, :] = spectrum.power.filled(np.nan)
    freqs_full = freqs[0] + native * np.arange(n_full)
    log.info(
        "Regularized frequency grid: %d -> %d channels (%d masked fillers), "
        "removing a %.4fx lag-axis stretch (mean spacing %.6g -> native %.6g MHz).",
        freqs.size,
        n_full,
        n_full - freqs.size,
        ratio,
        mean,
        native,
    )
    return DynamicSpectrum(power_full, freqs_full, spectrum.times.copy())


def apply_grid_regularization(spectrum: DynamicSpectrum, cfg: dict) -> DynamicSpectrum:
    """Apply ``analysis.grid_regularization`` gating to a freshly loaded
    spectrum: regularize when enabled, warn loudly when disabled on a
    non-uniform grid, no-op otherwise.

    Shared by every data-preparation path (this module's
    ``prepare_spectrum_from_config`` and ``pipeline.ScintillationAnalysis``),
    so a config that enables the flag cannot be silently bypassed by one
    entry point. Must run BEFORE ``downsample``: block-averaging across a gap
    junction mixes non-adjacent frequencies exactly like the index-lag ACF
    does.
    """
    grid_cfg = cfg.get("analysis", {}).get("grid_regularization", {})
    if grid_cfg.get("enable", False):
        return regularize_frequency_grid(spectrum)
    native, mean, ratio = _grid_stretch_ratio(spectrum.frequencies)
    if abs(ratio - 1.0) > _GRID_UNIFORM_REL_TOL:
        log.warning(
            "Frequency grid is non-uniform (mean spacing %.6g MHz vs native "
            "%.6g MHz): the ACF lag axis and Delta nu_d will be overstated "
            "by ~%.4fx and gap-straddling lags will mix distant channels. "
            "Enable analysis.grid_regularization to fix (issue #120).",
            mean,
            native,
            ratio,
        )
    return spectrum


def prepare_spectrum_from_config(
    cfg: dict,
) -> tuple[DynamicSpectrum, tuple[int, int], tuple[int, int] | None]:
    ds_cfg = cfg.get("pipeline_options", {}).get("downsample", {})
    f_factor = int(ds_cfg.get("f_factor", 1))
    t_factor = int(ds_cfg.get("t_factor", 1))
    spectrum = DynamicSpectrum.from_numpy_file(cfg["input_data_path"])
    spectrum = apply_grid_regularization(spectrum, cfg)
    spectrum = spectrum.downsample(f_factor, t_factor)
    masked = spectrum.mask_rfi(cfg)
    burst_lims, off_lims = determine_windows(masked, cfg)

    bandpass_cfg = cfg.get("analysis", {}).get("bandpass_normalization", {})
    if bandpass_cfg.get("enable", False):
        # Flat-field before any additive baseline step: the coarse-channel
        # scallop is multiplicative, so it must be divided out on the raw
        # bandpass, not after a polynomial has already been subtracted.
        if off_lims is None or off_lims[1] <= off_lims[0]:
            raise ValueError("bandpass normalization requires a valid off-pulse window")
        masked = normalize_bandpass(
            masked,
            off_lims,
            floor_frac=float(bandpass_cfg.get("floor_frac", _BANDPASS_FLOOR_FRAC)),
        )

    baseline_cfg = cfg.get("analysis", {}).get("baseline_subtraction", {})
    if baseline_cfg.get("enable", False):
        # Same off-pulse validity guard as pipeline.py (>50 bins): a polynomial
        # fit to a handful of bins models noise, not bandpass, and would be
        # subtracted silently.
        if off_lims and off_lims[1] > off_lims[0] + 50:
            off_spec = masked.get_spectrum(off_lims)
            masked, _baseline = masked.subtract_poly_baseline(
                off_spec,
                poly_order=int(baseline_cfg.get("poly_order", 1)),
            )
        else:
            log.warning("Not enough off-pulse data to model baseline. Skipping subtraction.")
    return masked, burst_lims, off_lims


def _scan_fit_windows(
    on_spectrum: np.ndarray | np.ma.MaskedArray,
    *,
    channel_width_mhz: float,
    off_burst_spectrum_mean: float | None,
    max_lag_mhz: float,
    scan_lags_mhz: list[float],
) -> tuple[list[dict], float | None]:
    """Refit the ACF Lorentzian per fit window; return per-window records and
    the spread (max - min) of the successful widths (None if fewer than two
    windows succeed)."""
    records: list[dict] = []
    widths: list[float] = []
    for fit_lag in scan_lags_mhz:
        if not np.isfinite(fit_lag):
            # A NaN/inf from YAML must surface as a failed record, not crash
            # the strict allow_nan=False JSON writer downstream.
            records.append(
                {
                    "fit_lag_mhz": None,
                    "success": False,
                    "delta_nu_mhz": None,
                    "delta_nu_err_mhz": None,
                    "message": f"invalid fit window: non-finite value {fit_lag!r}",
                }
            )
            continue
        try:
            r = measure_scintillation_bandwidth(
                on_spectrum,
                channel_width_mhz=channel_width_mhz,
                off_burst_spectrum_mean=off_burst_spectrum_mean,
                max_lag_mhz=max_lag_mhz,
                fit_lag_mhz=float(fit_lag),
            )
            record = {
                "fit_lag_mhz": float(fit_lag),
                "success": r.success,
                "delta_nu_mhz": r.delta_nu_mhz,
                "delta_nu_err_mhz": r.delta_nu_err_mhz,
                "message": r.message,
            }
            if r.success and r.delta_nu_mhz is not None:
                widths.append(float(r.delta_nu_mhz))
        except ValueError as exc:
            # An invalid window (e.g. below one channel) must not kill the
            # primary result; record it as a visible failure instead.
            record = {
                "fit_lag_mhz": float(fit_lag),
                "success": False,
                "delta_nu_mhz": None,
                "delta_nu_err_mhz": None,
                "message": f"invalid fit window: {exc}",
            }
        records.append(record)
    systematic = float(max(widths) - min(widths)) if len(widths) >= 2 else None
    return records, systematic


def run_notebook_style_analysis(
    spectrum: DynamicSpectrum,
    *,
    burst_id: str,
    burst_lims: tuple[int, int],
    off_pulse_lims: tuple[int, int] | None,
    output_dir: str | Path,
    max_lag_mhz: float = 20.0,
    fit_lag_mhz: float = 2.0,
    fit_lag_scan_mhz: list[float] | None = None,
    write_figures: bool = True,
) -> NotebookStyleResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    on_spectrum = spectrum.get_spectrum(burst_lims)
    off_mean = None
    if off_pulse_lims is not None and off_pulse_lims[1] > off_pulse_lims[0]:
        off_mean = float(np.ma.mean(spectrum.get_spectrum(off_pulse_lims)))

    acf_result = measure_scintillation_bandwidth(
        on_spectrum,
        channel_width_mhz=spectrum.channel_width_mhz,
        off_burst_spectrum_mean=off_mean,
        max_lag_mhz=max_lag_mhz,
        fit_lag_mhz=fit_lag_mhz,
    )
    structure_result = estimate_structure_bandwidth(
        on_spectrum,
        channel_width_mhz=spectrum.channel_width_mhz,
    )
    figures: list[FigureRecord] = []
    if write_figures:
        figures.extend(
            write_diagnostic_figures(
                spectrum,
                on_spectrum,
                burst_id=burst_id,
                burst_lims=burst_lims,
                acf_result=acf_result,
                structure_result=structure_result,
                output_dir=output,
            )
        )

    fit_window_scan: list[dict] | None = None
    fit_window_systematic_mhz: float | None = None
    if fit_lag_scan_mhz:
        fit_window_scan, fit_window_systematic_mhz = _scan_fit_windows(
            on_spectrum,
            channel_width_mhz=spectrum.channel_width_mhz,
            off_burst_spectrum_mean=off_mean,
            max_lag_mhz=max_lag_mhz,
            scan_lags_mhz=list(fit_lag_scan_mhz),
        )

    result_path = output / f"{burst_id}_scintillation.json"
    result = NotebookStyleResult(
        burst_id=burst_id,
        burst_lims=burst_lims,
        off_pulse_lims=off_pulse_lims,
        acf=acf_result,
        structure=structure_result,
        figures=figures,
        result_path=str(result_path),
        fit_window_scan=fit_window_scan,
        fit_window_systematic_mhz=fit_window_systematic_mhz,
    )
    result_path.write_text(
        json.dumps(to_jsonable(result), indent=2, cls=NumpyJSONEncoder, allow_nan=False) + "\n"
    )
    return result


def write_diagnostic_figures(
    spectrum: DynamicSpectrum,
    on_spectrum: np.ma.MaskedArray,
    *,
    burst_id: str,
    burst_lims: tuple[int, int],
    acf_result: ACFBandwidthResult,
    structure_result: StructureBandwidthResult,
    output_dir: Path,
) -> list[FigureRecord]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records: list[FigureRecord] = []
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [1, 3]},
    )
    profile = np.ma.mean(spectrum.power, axis=0)
    axes[0].plot(spectrum.times, profile, color="k", lw=1.2)
    axes[0].axvspan(
        spectrum.times[burst_lims[0]], spectrum.times[burst_lims[1] - 1], color="C3", alpha=0.2
    )
    axes[0].set_ylabel("profile")
    power = spectrum.power.filled(np.nan)
    vmin, vmax = np.nanpercentile(power, [2, 98])
    im = axes[1].imshow(
        power,
        aspect="auto",
        origin="lower",
        extent=[
            float(spectrum.times[0]),
            float(spectrum.times[-1]),
            float(spectrum.frequencies[0]),
            float(spectrum.frequencies[-1]),
        ],
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("frequency (MHz)")
    fig.colorbar(im, ax=axes[1], label="intensity")
    fig.tight_layout()
    path = output_dir / f"{burst_id}_dynamic_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    records.append(
        FigureRecord(
            kind="dynamic_spectrum",
            path=path.name,
            description="Dynamic spectrum with the on-pulse window used for the spectrum ACF.",
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    if acf_result.lags_mhz:
        ax.step(acf_result.lags_mhz, acf_result.acf, where="mid", color="k", label="ACF")
    if acf_result.acf_model:
        ax.plot(acf_result.lags_mhz, acf_result.acf_model, color="C3", lw=2, label="Lorentzian fit")
    if acf_result.delta_nu_mhz is not None:
        ax.axvline(acf_result.delta_nu_mhz, color="C3", ls=":", alpha=0.8)
        ax.axvline(-acf_result.delta_nu_mhz, color="C3", ls=":", alpha=0.8)
    ax.set_xlim(-acf_result.fit_lag_mhz, acf_result.fit_lag_mhz)
    ax.set_xlabel("frequency lag (MHz)")
    ax.set_ylabel("ACF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / f"{burst_id}_acf_lorentzian.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    records.append(
        FigureRecord(
            kind="acf",
            path=path.name,
            description="Frequency ACF and Lorentzian HWHM fit for Delta nu_d.",
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(
        structure_result.frequency_lags_mhz,
        structure_result.structure_function,
        where="mid",
        color="k",
    )
    ax.axvline(structure_result.delta_nu_mhz, color="C4", ls=":", alpha=0.8)
    ax.set_xlim(
        0.0,
        min(max(structure_result.delta_nu_mhz * 5.0, 1.0), structure_result.frequency_lags_mhz[-1]),
    )
    ax.set_xlabel("frequency lag (MHz)")
    ax.set_ylabel("structure function")
    fig.tight_layout()
    path = output_dir / f"{burst_id}_structure_function.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    records.append(
        FigureRecord(
            kind="structure_function",
            path=path.name,
            description="Notebook-style structure-function bandwidth cross-check.",
        )
    )
    return records


def to_jsonable(result: NotebookStyleResult) -> dict:
    payload = asdict(result)
    return payload


def _default_output_dir(cfg: dict) -> Path:
    opts = cfg.get("pipeline_options", {})
    workspace_root = Path(cfg.get("_workspace_root", "."))
    if opts.get("results_directory"):
        return config_module.resolve_path(opts["results_directory"], base_dir=workspace_root)
    if opts.get("output_plot_path"):
        return config_module.resolve_path(opts["output_plot_path"], base_dir=workspace_root).parent
    return Path.cwd()


def run_config_path(
    burst_config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    write_figures: bool = True,
) -> NotebookStyleResult:
    cfg = config_module.load_config(burst_config_path)
    log_level = cfg.get("pipeline_options", {}).get("log_level", "INFO").upper()
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    spectrum, burst_lims, off_lims = prepare_spectrum_from_config(cfg)
    acf_cfg = cfg.get("analysis", {}).get("acf", {})
    fitting_cfg = cfg.get("analysis", {}).get("fitting", {})
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_output_dir(cfg)
    )
    scan_cfg = fitting_cfg.get("fit_lag_scan_mhz")
    return run_notebook_style_analysis(
        spectrum,
        burst_id=cfg.get("burst_id", "freya"),
        burst_lims=burst_lims,
        off_pulse_lims=off_lims,
        output_dir=out,
        max_lag_mhz=float(acf_cfg.get("max_lag_mhz", 20.0)),
        fit_lag_mhz=float(fitting_cfg.get("fit_lagrange_mhz", 2.0)),
        fit_lag_scan_mhz=[float(v) for v in scan_cfg] if scan_cfg else None,
        write_figures=write_figures,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Freya scintillation bandwidth from a pipeline scintillation config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="scintillation/configs/bursts/freya_dsa.yaml",
        help="Burst YAML config path.",
    )
    parser.add_argument("--out", default=None, help="Output directory for JSON and figures.")
    parser.add_argument(
        "--no-figures", action="store_true", help="Do not write diagnostic figures."
    )
    args = parser.parse_args(argv)
    result = run_config_path(args.config, output_dir=args.out, write_figures=not args.no_figures)
    print(json.dumps(to_jsonable(result), indent=2, cls=NumpyJSONEncoder, allow_nan=False))
    return 0 if result.acf.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
