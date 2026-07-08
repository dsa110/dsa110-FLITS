#!/usr/bin/env python3
"""Fresh DSA ACF Lorentzian scintillation-bandwidth fits.

This driver intentionally bypasses legacy YAML ``stored_fits`` and any rescued
``acf_results.pkl`` products. It recomputes ACFs from the staged DSA `.npz`
dynamic spectra, then applies the existing 1/2/3-Lorentzian BIC + nested-F
selector to each sub-band ACF.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Use the checked-out pipeline source for this analysis, even if another FLITS
# checkout is installed editable in the active Python environment. Disable numba
# JIT before importing scintillation modules; old cross-checkout numba caches can
# try to resurrect modules by the stale top-level name ``scint_analysis``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from scintillation.scint_analysis import analysis  # noqa: E402
from scintillation.scint_analysis import config as config_mod  # noqa: E402
from scintillation.scint_analysis.pipeline import ScintillationAnalysis  # noqa: E402
from scintillation.scint_analysis.revalidation import (  # noqa: E402
    compare_lorentzian_components,
)

BURSTS = [
    "casey",
    "chromatica",
    "freya",
    "hamilton",
    "isha",
    "johndoeII",
    "mahi",
    "oran",
    "phineas",
    "whitney",
    "wilhelm",
    "zach",
]

SUBBAND_CANDIDATES = (2, 3, 4)
MIN_SUBBAND_CHANNELS = 512
MIN_FIT_RANGE_MHZ = 8.0
MIN_POSITIVE_FIT_POINTS = 30


def _lorentzian_curve(x: np.ndarray, gamma: float, m: float) -> np.ndarray:
    return (m**2) / (1.0 + (x / gamma) ** 2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _config_for_fresh_acf(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(config)

    # Keep the checked-in science choices, but remove fit/result reuse knobs from
    # this generated run configuration.
    cfg.pop("stored_fits", None)

    pipe_opts = cfg.setdefault("pipeline_options", {})
    pipe_opts["force_recalc"] = True
    pipe_opts["save_intermediate_steps"] = False
    pipe_opts["halt_after_acf"] = True
    pipe_opts["cache_directory"] = str(output_dir / "cache" / cfg.get("burst_id", "unknown"))
    pipe_opts.setdefault("log_level", "INFO")
    pipe_opts["diagnostic_plots"] = {"enable": False}

    analysis_cfg = cfg.setdefault("analysis", {})
    noise_cfg = analysis_cfg.setdefault("noise", {})
    noise_cfg.setdefault("disable", False)
    # The Lorentzian-only selector does not consume the MC template. Disabling it
    # keeps this pass deterministic and much faster without changing the ACF.
    noise_cfg["disable_template"] = True

    analysis_cfg.setdefault("fit_2d", {})["enable"] = False
    return cfg


def _format_threshold(value: float | int) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _config_with_subband_count(config: dict[str, Any], num_subbands: int) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    acf_cfg = cfg.setdefault("analysis", {}).setdefault("acf", {})
    acf_cfg["num_subbands"] = int(num_subbands)
    acf_cfg["use_snr_subbanding"] = True
    return cfg


def _candidate_rejection_reasons(candidate: dict[str, Any]) -> list[str]:
    requested = int(candidate.get("requested_num_subbands", candidate.get("num_subbands", 0)))
    actual = int(candidate.get("num_subbands", 0))
    if actual != requested:
        return [f"requested {requested} subbands but produced {actual}"]

    subbands = candidate.get("subbands", [])
    for subband in subbands:
        idx = int(subband.get("index", 0))
        n_chan = int(subband.get("num_channels", 0))
        if n_chan < MIN_SUBBAND_CHANNELS:
            return [
                f"subband {idx} num_channels {n_chan} < "
                f"{_format_threshold(MIN_SUBBAND_CHANNELS)}"
            ]

        fit_range = float(subband.get("fit_range_mhz", np.nan))
        if not np.isfinite(fit_range) or fit_range < MIN_FIT_RANGE_MHZ:
            shown = _format_threshold(fit_range) if np.isfinite(fit_range) else "nonfinite"
            return [
                f"subband {idx} fit_range_mhz {shown} < "
                f"{_format_threshold(MIN_FIT_RANGE_MHZ)}"
            ]

        n_fit_points = int(subband.get("n_fit_points", 0))
        if n_fit_points < MIN_POSITIVE_FIT_POINTS:
            return [
                f"subband {idx} n_fit_points {n_fit_points} < "
                f"{_format_threshold(MIN_POSITIVE_FIT_POINTS)}"
            ]

        components = subband.get("selected_components", [])
        if not components:
            return [f"subband {idx} has no selected component"]
        if all(comp.get("quality_flags") for comp in components):
            return [f"subband {idx} has no unflagged selected component"]
    return []


def _candidate_warning_summary(candidate: dict[str, Any]) -> dict[str, int]:
    flagged_components = 0
    subbands_without_unflagged_components = 0
    for subband in candidate.get("subbands", []):
        components = subband.get("selected_components", [])
        flagged_components += sum(1 for comp in components if comp.get("quality_flags"))
        if components and all(comp.get("quality_flags") for comp in components):
            subbands_without_unflagged_components += 1
    return {
        "flagged_components": flagged_components,
        "subbands_without_unflagged_components": subbands_without_unflagged_components,
    }


def _select_subband_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluations = []
    metrics = []
    viable = []
    for candidate in candidates:
        n_subbands = int(candidate.get("requested_num_subbands", candidate.get("num_subbands", 0)))
        reasons = _candidate_rejection_reasons(candidate)
        evaluation = {
            "num_subbands": n_subbands,
            "viable": not reasons,
            "reasons": reasons,
        }
        evaluations.append(evaluation)
        metrics.append(
            {
                "num_subbands": n_subbands,
                **_candidate_warning_summary(candidate),
            }
        )
        if not reasons:
            viable.append(candidate)

    if viable:
        selected = max(
            viable,
            key=lambda c: int(c.get("requested_num_subbands", c.get("num_subbands", 0))),
        )
        selected_policy = "largest_viable_equal_snr_subband_count"
    elif candidates:
        selected = min(
            candidates,
            key=lambda c: (
                _candidate_warning_summary(c)["subbands_without_unflagged_components"],
                _candidate_warning_summary(c)["flagged_components"],
                int(c.get("requested_num_subbands", c.get("num_subbands", 0))),
            ),
        )
        selected_policy = "least_pathological_equal_snr_subband_count"
    else:
        raise RuntimeError("no subband candidates were evaluated")

    selected_n = int(selected.get("requested_num_subbands", selected.get("num_subbands", 0)))
    report = {
        "policy": "explicit_equal_snr_subband_candidate_selection",
        "selected_policy": selected_policy,
        "candidate_counts": list(SUBBAND_CANDIDATES),
        "gates": {
            "min_subband_channels": MIN_SUBBAND_CHANNELS,
            "min_fit_range_mhz": MIN_FIT_RANGE_MHZ,
            "min_positive_fit_points": MIN_POSITIVE_FIT_POINTS,
        },
        "selected_num_subbands": selected_n,
        "candidates": evaluations,
        "candidate_metrics": metrics,
    }
    return selected, report


def _slice_fit_window(
    lags: np.ndarray, acf: np.ndarray, err: np.ndarray | None, fit_range_mhz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range_mhz)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)
    sliced_err = err[mask] if err is not None else None
    return lags[mask], acf[mask], sliced_err


def _plurality_n(per_subband: list[dict[str, Any]]) -> int:
    counts = Counter(int(v.get("n_preferred", 1)) for v in per_subband)
    if not counts:
        return 1
    top = max(counts.values())
    return min(n for n, count in counts.items() if count == top)


def _selected_fit(verdict: dict[str, Any]) -> dict[str, Any]:
    n_pref = int(verdict.get("n_preferred", 1))
    for fit in verdict.get("fits", []):
        if int(fit.get("n", -1)) == n_pref:
            return fit
    return {"n": n_pref, "success": False, "components": []}


def _model_curve(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    y = np.full_like(x, float(fit.get("constant", 0.0)), dtype=float)
    for component in fit.get("components", []):
        gamma = float(component.get("dnu_mhz", np.nan))
        m = float(component.get("m", np.nan))
        if np.isfinite(gamma) and gamma > 0 and np.isfinite(m):
            y += _lorentzian_curve(x, gamma, m)
    return y


QUALITY_FLAG_LABELS = {
    "invalid_dnu": "invalid dnu",
    "dnu_exceeds_fit_window": "broad",
    "fractional_dnu_err_gt_1": "weak dnu",
    "modulation_gt_3": "high m",
    "fractional_mod_err_gt_1": "weak m",
}


def _format_sigfig(value: float, *, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}g}"


def _flag_note(components: list[dict[str, Any]]) -> str | None:
    notes = []
    for comp_idx, component in enumerate(components, start=1):
        labels = [
            QUALITY_FLAG_LABELS.get(flag, flag.replace("_", " "))
            for flag in component.get("quality_flags", [])
        ]
        if labels:
            shown = "/".join(labels[:2])
            suffix = "+" if len(labels) > 2 else ""
            notes.append(f"c{comp_idx} {shown}{suffix}")
    return "; ".join(notes) if notes else None


def _panel_annotation(subband: dict[str, Any]) -> str:
    components = subband["selected_components"]
    dnu_values = [
        _format_sigfig(float(component.get("dnu_mhz", np.nan)))
        for component in components
    ]
    lines = [
        f"n={subband['n_preferred']}, redchi={_format_sigfig(float(subband.get('selected_redchi', np.nan)))}",
        "dnu: " + ", ".join(dnu_values) + " MHz",
    ]
    flag_note = _flag_note(components)
    if flag_note:
        lines.append(f"flagged: {flag_note}")
    return "\n".join(lines)


def _decimated_indices(mask: np.ndarray, *, max_points: int) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size <= max_points:
        return idx
    positions = np.linspace(0, idx.size - 1, max_points).round().astype(int)
    return np.unique(idx[positions])


def _component_quality_flags(component: dict[str, Any], *, fit_range_mhz: float) -> list[str]:
    flags = []
    dnu = float(component.get("dnu_mhz", np.nan))
    dnu_err = float(component.get("dnu_err", np.nan))
    mod = float(component.get("m", np.nan))
    mod_err = float(component.get("m_err", np.nan))

    if not np.isfinite(dnu) or dnu <= 0:
        flags.append("invalid_dnu")
    elif dnu > fit_range_mhz:
        flags.append("dnu_exceeds_fit_window")

    if np.isfinite(dnu) and dnu > 0 and np.isfinite(dnu_err) and dnu_err / dnu > 1.0:
        flags.append("fractional_dnu_err_gt_1")
    if np.isfinite(mod) and mod > 3.0:
        flags.append("modulation_gt_3")
    if np.isfinite(mod) and mod > 0 and np.isfinite(mod_err) and mod_err / mod > 1.0:
        flags.append("fractional_mod_err_gt_1")

    return flags


def _plot_burst_acfs(
    burst: str,
    plot_subbands: list[dict[str, Any]],
    *,
    figure_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "axes.linewidth": 0.8,
            "font.family": "serif",
            "font.size": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 240,
            "svg.fonttype": "none",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    n_subbands = len(plot_subbands)
    ncols = 2 if n_subbands > 1 else 1
    nrows = int(np.ceil(n_subbands / ncols))
    panel_width = 3.85
    panel_height = 2.55
    title_clearance = 0.85 if nrows == 1 else 0.55
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows + title_clearance),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    acf_color = "#2f5f8f"
    uncertainty_color = "#8ab1d3"
    fit_color = "#111827"
    baseline_color = "#8b949e"
    component_colors = ["#d97706", "#0f766e", "#7c3aed"]

    for ax, payload in zip(axes_flat, plot_subbands, strict=False):
        lags = payload["lags"]
        acf = payload["acf"]
        err = payload["err"]
        subband = payload["summary"]
        fit = payload["fit"]
        fit_range = float(subband["fit_range_mhz"])

        display = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range)
        nonzero = display & (lags != 0)
        err_ok = err is not None and np.any(np.isfinite(err[nonzero]) & (err[nonzero] > 0))

        if err_ok:
            idx = _decimated_indices(nonzero, max_points=26)
            ax.errorbar(
                lags[idx],
                acf[idx],
                yerr=np.asarray(err)[idx],
                fmt="none",
                elinewidth=0.45,
                capsize=0,
                ecolor=uncertainty_color,
                alpha=0.28,
                zorder=1,
            )

        ax.scatter(
            lags[nonzero],
            acf[nonzero],
            s=4,
            color=acf_color,
            alpha=0.46,
            linewidths=0,
            zorder=2,
        )

        zero = display & (lags == 0)
        if np.any(zero):
            ax.scatter(
                lags[zero],
                acf[zero],
                s=13,
                facecolors="white",
                edgecolors="#4b5563",
                linewidths=0.8,
                zorder=4,
            )

        xfit = np.linspace(-fit_range, fit_range, 900)
        yfit = _model_curve(xfit, fit)
        ax.plot(xfit, yfit, color=fit_color, lw=1.6, zorder=5)
        constant = float(fit.get("constant", 0.0))
        ax.axhline(constant, color=baseline_color, lw=0.8, ls=(0, (1.4, 1.8)), zorder=0)
        ax.axvline(0.0, color="#cbd5e1", lw=0.65, zorder=0)

        components = subband["selected_components"]
        for comp_idx, component in enumerate(components, start=1):
            if len(components) < 2:
                continue
            gamma = float(component.get("dnu_mhz", np.nan))
            m = float(component.get("m", np.nan))
            if not (np.isfinite(gamma) and gamma > 0 and np.isfinite(m)):
                continue
            y_component = constant + _lorentzian_curve(xfit, gamma, m)
            ax.plot(
                xfit,
                y_component,
                color=component_colors[(comp_idx - 1) % len(component_colors)],
                lw=0.95,
                ls="--",
                alpha=0.62,
                zorder=3,
            )

        y_candidates = [acf[nonzero], yfit[np.isfinite(yfit)]]
        finite_y = np.concatenate([v[np.isfinite(v)] for v in y_candidates if v.size])
        if finite_y.size:
            lo, hi = np.nanpercentile(finite_y, [0.5, 99.5])
            pad = max(0.04, 0.14 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)

        ax.text(
            0.02,
            0.98,
            _panel_annotation(subband),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color="#111827",
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.4},
        )
        ax.set_title(
            f"subband {subband['index']}  |  {subband['center_freq_mhz']:.1f} MHz",
            fontsize=9,
            pad=4,
        )
        ax.grid(axis="y", color="#e5e7eb", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(
            axis="x",
            which="both",
            direction="out",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            length=3,
            width=0.8,
        )
        ax.tick_params(
            axis="y",
            which="both",
            direction="out",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            length=3,
            width=0.8,
        )

    for ax in axes_flat[n_subbands:]:
        ax.axis("off")

    title_y = 1.08 if nrows == 1 else 1.025
    fig.suptitle(f"{burst}: DSA frequency ACF fits", fontsize=11, y=title_y)
    fig.supxlabel("Frequency lag (MHz)", fontsize=10)
    fig.supylabel("ACF", fontsize=10)
    png = figure_dir / f"{burst}_dsa_acf_lorentzian_fits.png"
    svg = figure_dir / f"{burst}_dsa_acf_lorentzian_fits.svg"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {"figure_png": str(png), "figure_svg": str(svg)}


def _fit_prepared_config(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    output_dir: Path,
    max_components: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    burst = str(cfg.get("burst_id", config_path.stem.removesuffix("_dsa")))

    analysis.clear_noise_acf_cache()
    pipe = ScintillationAnalysis(cfg)
    pipe.run()
    acf_results = pipe.acf_results
    if not acf_results or not acf_results.get("subband_acfs"):
        raise RuntimeError(f"{burst}: no ACF results produced")

    fit_cfg = cfg.get("analysis", {}).get("fitting", {})
    configured_fit_range = float(fit_cfg.get("fit_lagrange_mhz", 45.0))

    subbands = []
    plot_subbands = []
    for i, acf in enumerate(acf_results["subband_acfs"]):
        lags = np.asarray(acf_results["subband_lags_mhz"][i], dtype=float)
        acf_arr = np.asarray(acf, dtype=float)
        err_values = acf_results.get("subband_acfs_err")
        err = np.asarray(err_values[i], dtype=float) if err_values else None

        center_freq = float(acf_results["subband_center_freqs_mhz"][i])
        chan_width = float(acf_results["subband_channel_widths_mhz"][i])
        n_chan = int(acf_results["subband_num_channels"][i])
        subband_bw = n_chan * chan_width
        fit_range = min(configured_fit_range, subband_bw / 2.0)
        fit_lags, fit_acf, fit_err = _slice_fit_window(lags, acf_arr, err, fit_range)
        verdict = compare_lorentzian_components(
            fit_lags,
            fit_acf,
            max_components=max_components,
            acf_err=fit_err,
        )
        fit = _selected_fit(verdict)
        components = sorted(
            fit.get("components", []),
            key=lambda c: float(c.get("dnu_mhz", np.inf)),
        )
        for component in components:
            component["quality_flags"] = _component_quality_flags(
                component,
                fit_range_mhz=fit_range,
            )

        subbands.append(
            {
                "index": i,
                "center_freq_mhz": center_freq,
                "channel_width_mhz": chan_width,
                "num_channels": n_chan,
                "fit_range_mhz": fit_range,
                "n_fit_points": int(np.sum(fit_lags > 0)),
                "n_preferred": int(verdict.get("n_preferred", 1)),
                "criterion": verdict.get("criterion"),
                "delta_bic": verdict.get("delta_bic", {}),
                "f_test_p": verdict.get("f_test", {}),
                "selected_bic": fit.get("bic"),
                "selected_redchi": fit.get("redchi"),
                "selected_components": components,
                "all_fit_summaries": [
                    {
                        "n": int(f.get("n", 0)),
                        "success": bool(f.get("success", False)),
                        "bic": f.get("bic"),
                        "aic": f.get("aic"),
                        "chi2": f.get("chi2"),
                        "redchi": f.get("redchi"),
                        "n_params": f.get("n_params"),
                        "ndata": f.get("ndata"),
                        "constant": f.get("constant"),
                        "constant_err": f.get("constant_err"),
                        "components": sorted(
                            f.get("components", []),
                            key=lambda c: float(c.get("dnu_mhz", np.inf)),
                        ),
                    }
                    for f in verdict.get("fits", [])
                ],
            }
        )
        plot_subbands.append(
            {
                "lags": lags,
                "acf": acf_arr,
                "err": err,
                "summary": subbands[-1],
                "fit": fit,
            }
        )

    component_bands: dict[int, list[float]] = defaultdict(list)
    usable_component_bands: dict[int, list[float]] = defaultdict(list)
    for subband in subbands:
        for comp_idx, comp in enumerate(subband["selected_components"], start=1):
            dnu = comp.get("dnu_mhz")
            if dnu is not None and np.isfinite(float(dnu)):
                component_bands[comp_idx].append(float(dnu))
                if not comp.get("quality_flags"):
                    usable_component_bands[comp_idx].append(float(dnu))

    result = {
        "burst": burst,
        "config_path": str(config_path),
        "input_data_path": cfg.get("input_data_path"),
        "fit_lagrange_mhz": configured_fit_range,
        "max_components": max_components,
        "num_subbands": len(subbands),
        "burst_preferred_n": _plurality_n(subbands),
        "n_per_subband": [s["n_preferred"] for s in subbands],
        "component_median_dnu_mhz": {
            str(k): float(np.nanmedian(v)) for k, v in sorted(component_bands.items())
        },
        "component_usable_median_dnu_mhz": {
            str(k): float(np.nanmedian(v))
            for k, v in sorted(usable_component_bands.items())
            if v
        },
        "subbands": subbands,
    }
    return result, plot_subbands


def _fit_one_burst(
    config_path: Path,
    *,
    output_dir: Path,
    max_components: int,
    make_figures: bool,
) -> dict[str, Any]:
    loaded = config_mod.load_config(config_path)
    base_cfg = _config_for_fresh_acf(loaded, output_dir=output_dir)
    burst = str(base_cfg.get("burst_id", config_path.stem.removesuffix("_dsa")))

    candidates = []
    plot_payloads = {}
    for num_subbands in SUBBAND_CANDIDATES:
        cfg = _config_with_subband_count(base_cfg, num_subbands)
        result, plot_subbands = _fit_prepared_config(
            cfg,
            config_path,
            output_dir=output_dir,
            max_components=max_components,
        )
        result["requested_num_subbands"] = num_subbands
        candidates.append(result)
        plot_payloads[num_subbands] = plot_subbands

    result, selection = _select_subband_candidate(candidates)
    result["subband_selection"] = selection
    if make_figures:
        selected_n = int(result["requested_num_subbands"])
        result.update(
            _plot_burst_acfs(burst, plot_payloads[selected_n], figure_dir=output_dir / "figures")
        )
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "burst",
        "subband",
        "center_freq_mhz",
        "n_preferred",
        "component",
        "dnu_mhz",
        "dnu_err_mhz",
        "modulation_m",
        "modulation_err",
        "fit_range_mhz",
        "selected_bic",
        "selected_redchi",
        "quality_flags",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _selection_summary(result: dict[str, Any]) -> str:
    selection = result.get("subband_selection", {})
    rejected = [
        f"n={candidate['num_subbands']}: {'; '.join(candidate['reasons'])}"
        for candidate in selection.get("candidates", [])
        if not candidate.get("viable", False)
    ]
    if not rejected:
        return "largest viable candidate"
    return "rejected " + "<br>".join(rejected)


def _write_markdown(results: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# DSA Lorentzian ACF Fit Summary",
        "",
        "Fresh DSA ACFs were computed from the staged `.npz` dynamic spectra. Each sub-band",
        "was fit with 1, 2, and 3 Lorentzian components; adding a component required both",
        "strong BIC improvement and the nested-F test threshold in the existing",
        "`compare_lorentzian_components` selector.",
        "",
        "The number of DSA sub-bands is selected within this run, not inherited from",
        "the checked-in burst YAML. For each burst the driver evaluates 2, 3, and 4",
        "equal-S/N frequency splits, then chooses the largest candidate for which",
        "every produced sub-band passes fixed viability gates: at least 512 unmasked",
        "channels, at least an 8 MHz fitted lag window, and at least 30 positive-lag",
        "fit samples, with at least one selected component not carrying a quality",
        "flag. If no candidate satisfies all gates, the least pathological candidate",
        "is retained and the fallback policy is recorded.",
        "",
        "## Burst Overview",
        "",
        "| burst | selected subbands | preferred n by subband | plurality n | median dnu by component (MHz) | selection note |",
        "|---|---:|---|---:|---|---|",
    ]
    for result in results:
        usable = result.get("component_usable_median_dnu_mhz", {})
        if usable:
            med = ", ".join(f"c{k}={v:.4g}" for k, v in usable.items())
        else:
            med = "no unflagged components"
        lines.append(
            "| {burst} | {num_subbands} | {n_per_subband} | {burst_preferred_n} | {med} | {note} |".format(
                med=med or "-",
                note=_selection_summary(result),
                **result,
            )
        )

    lines.extend(
        [
            "",
            "## Component Rows",
            "",
            "| burst | subband | freq MHz | n | component | dnu MHz | dnu err | m | redchi | flags |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {burst} | {subband} | {center_freq_mhz:.3f} | {n_preferred} | {component} | "
            "{dnu_mhz:.6g} | {dnu_err_mhz:.3g} | {modulation_m:.4g} | "
            "{selected_redchi:.4g} | {quality_flags} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## ACF Fit Figures",
            "",
            "Blue points are ACF samples, pale blue whiskers show a decimated uncertainty",
            "sample, the black curve is the selected Lorentzian model, the dotted gray",
            "line is the fitted constant baseline, and dashed colored curves show",
            "individual components for multi-component fits.",
            "",
        ]
    )
    for result in results:
        figure_png = result.get("figure_png")
        if not figure_png:
            continue
        figure_path = Path(figure_png)
        try:
            rel = figure_path.resolve().relative_to(path.parent.resolve())
        except ValueError:
            rel = figure_path
        lines.extend([f"### {result['burst']}", "", f"![{result['burst']} ACF fits]({rel})", ""])

    path.write_text("\n".join(lines).rstrip() + "\n")


def _flatten_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for subband in result["subbands"]:
            for comp_idx, comp in enumerate(subband["selected_components"], start=1):
                rows.append(
                    {
                        "burst": result["burst"],
                        "subband": subband["index"],
                        "center_freq_mhz": float(subband["center_freq_mhz"]),
                        "n_preferred": int(subband["n_preferred"]),
                        "component": comp_idx,
                        "dnu_mhz": float(comp.get("dnu_mhz", np.nan)),
                        "dnu_err_mhz": float(comp.get("dnu_err", np.nan)),
                        "modulation_m": float(comp.get("m", np.nan)),
                        "modulation_err": float(comp.get("m_err", np.nan)),
                        "fit_range_mhz": float(subband["fit_range_mhz"]),
                        "selected_bic": float(subband.get("selected_bic", np.nan)),
                        "selected_redchi": float(subband.get("selected_redchi", np.nan)),
                        "quality_flags": ";".join(comp.get("quality_flags", [])),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--flits-root",
        type=Path,
        default=Path(os.environ.get("FLITS_ROOT", Path.home() / "Data/Faber2026/dsa110")),
        help="Root containing scintillation/data/{burst}.npz.",
    )
    parser.add_argument("--max-components", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--bursts", nargs="*", default=BURSTS, help="Burst nicknames to run.")
    parser.add_argument("--no-figures", action="store_true", help="Skip ACF/fitted-curve plots.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failed bursts and continue instead of raising immediately.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ["FLITS_ROOT"] = str(args.flits_root.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for burst in args.bursts:
        config_path = Path("scintillation/configs/bursts") / f"{burst}_dsa.yaml"
        logging.info("Running %s from %s", burst, config_path)
        try:
            result = _fit_one_burst(
                config_path,
                output_dir=args.output_dir,
                max_components=args.max_components,
                make_figures=not args.no_figures,
            )
        except Exception as exc:
            logging.exception("%s failed", burst)
            failures.append({"burst": burst, "error": str(exc)})
            if not args.keep_going:
                raise
        else:
            results.append(result)
            burst_path = args.output_dir / f"{burst}_lorentzian_fits.json"
            burst_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))

    rows = _flatten_rows(results)
    all_results = {
        "run": {
            "flits_root": os.environ["FLITS_ROOT"],
            "max_components": args.max_components,
            "bursts_requested": args.bursts,
            "n_success": len(results),
            "n_failure": len(failures),
            "failures": failures,
            "figures_enabled": not args.no_figures,
            "figure_directory": str(args.output_dir / "figures") if not args.no_figures else None,
            "notes": (
                "Fresh DSA ACFs from npz; YAML stored_fits and pkl ACF products are not read. "
                "Pipeline caches, diagnostic plots, MC noise templates, and 2D fits are disabled. "
                "When enabled, figures show each sub-band ACF with the selected Lorentzian model."
            ),
        },
        "results": results,
    }
    (args.output_dir / "dsa_lorentzian_fits.json").write_text(
        json.dumps(_jsonable(all_results), indent=2, sort_keys=True)
    )
    _write_csv(rows, args.output_dir / "dsa_lorentzian_components.csv")
    _write_markdown(results, rows, args.output_dir / "DSA_LORENTZIAN_FITS.md")

    if failures:
        logging.error("Completed with %d failures", len(failures))
        return 1
    logging.info("Completed %d bursts; wrote %s", len(results), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
