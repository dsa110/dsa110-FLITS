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


@dataclass(frozen=True)
class FigureRecord:
    kind: str
    path: str
    description: str


@dataclass(frozen=True)
class ACFBandwidthResult:
    success: bool
    delta_nu_mhz: float | None
    delta_nu_err_mhz: float | None
    modulation_index: float
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
    burst_id: str
    burst_lims: tuple[int, int]
    off_pulse_lims: tuple[int, int] | None
    acf: ACFBandwidthResult
    structure: StructureBandwidthResult
    figures: list[FigureRecord]
    result_path: str | None


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


def prepare_spectrum_from_config(
    cfg: dict,
) -> tuple[DynamicSpectrum, tuple[int, int], tuple[int, int] | None]:
    ds_cfg = cfg.get("pipeline_options", {}).get("downsample", {})
    f_factor = int(ds_cfg.get("f_factor", 1))
    t_factor = int(ds_cfg.get("t_factor", 1))
    spectrum = DynamicSpectrum.from_numpy_file(cfg["input_data_path"]).downsample(
        f_factor, t_factor
    )
    masked = spectrum.mask_rfi(cfg)
    burst_lims, off_lims = determine_windows(masked, cfg)

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


def run_notebook_style_analysis(
    spectrum: DynamicSpectrum,
    *,
    burst_id: str,
    burst_lims: tuple[int, int],
    off_pulse_lims: tuple[int, int] | None,
    output_dir: str | Path,
    max_lag_mhz: float = 20.0,
    fit_lag_mhz: float = 2.0,
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

    result = NotebookStyleResult(
        burst_id=burst_id,
        burst_lims=burst_lims,
        off_pulse_lims=off_pulse_lims,
        acf=acf_result,
        structure=structure_result,
        figures=figures,
        result_path=None,
    )
    result_path = output / f"{burst_id}_scintillation.json"
    result = NotebookStyleResult(
        burst_id=result.burst_id,
        burst_lims=result.burst_lims,
        off_pulse_lims=result.off_pulse_lims,
        acf=result.acf,
        structure=result.structure,
        figures=result.figures,
        result_path=str(result_path),
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
    return run_notebook_style_analysis(
        spectrum,
        burst_id=cfg.get("burst_id", "freya"),
        burst_lims=burst_lims,
        off_pulse_lims=off_lims,
        output_dir=out,
        max_lag_mhz=float(acf_cfg.get("max_lag_mhz", 20.0)),
        fit_lag_mhz=float(fitting_cfg.get("fit_lagrange_mhz", 2.0)),
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
