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
from scintillation.scint_analysis import chime_artifact_guards as guards  # noqa: E402
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


def _reference_power_law(
    rows: list[dict[str, Any]],
    *,
    ref_alpha: float = 4.0,
    nu_ref_mhz: float | None = None,
    min_unique_freqs: int = 2,
) -> dict[str, float] | None:
    usable = [
        row
        for row in rows
        if row.get("usable", True)
        and np.isfinite(float(row.get("center_freq_mhz", np.nan)))
        and np.isfinite(float(row.get("dnu_mhz", np.nan)))
        and float(row.get("center_freq_mhz", np.nan)) > 0
        and float(row.get("dnu_mhz", np.nan)) > 0
    ]
    if not usable:
        return None

    freqs = np.array([float(row["center_freq_mhz"]) for row in usable], dtype=float)
    unique_freqs = np.unique(np.round(freqs, 9))
    if unique_freqs.size < min_unique_freqs:
        return None

    dnu = np.array([float(row["dnu_mhz"]) for row in usable], dtype=float)
    err = np.array([float(row.get("dnu_err_mhz", np.nan)) for row in usable], dtype=float)
    nu_ref = float(nu_ref_mhz) if nu_ref_mhz is not None else float(np.mean(freqs))
    basis = (freqs / nu_ref) ** ref_alpha
    weights = np.where(np.isfinite(err) & (err > 0), 1.0 / err**2, 1.0)
    scale = float(np.sum(weights * dnu * basis) / np.sum(weights * basis**2))
    return {
        "alpha": float(ref_alpha),
        "nu_ref_mhz": float(np.round(nu_ref, 12)),
        "scale_mhz": float(np.round(scale, 12)),
    }


def _bandwidth_axis_limits(rows: list[dict[str, Any]]) -> tuple[float, float]:
    clean = [
        float(row["dnu_mhz"])
        for row in rows
        if row.get("usable", True)
        and np.isfinite(float(row.get("dnu_mhz", np.nan)))
        and float(row["dnu_mhz"]) > 0
    ]
    values = clean or [
        float(row["dnu_mhz"])
        for row in rows
        if np.isfinite(float(row.get("dnu_mhz", np.nan))) and float(row["dnu_mhz"]) > 0
    ]
    if not values:
        return 0.1, 1.0

    lo = min(values)
    hi = max(values)
    if np.isclose(lo, hi):
        return float(lo / 1.8), float(hi * 1.8)
    return float(lo / 1.45), float(hi * 1.45)


MANUSCRIPT_PURPLE = "purple"
MANUSCRIPT_GUIDE = "gray"
MANUSCRIPT_GRID = "#d9d9d9"


def _plot_burst_acfs(
    burst: str,
    plot_subbands: list[dict[str, Any]],
    *,
    figure_dir: Path,
    band: str = "dsa",
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: PLC0415

    from flits.plotting import use_flits_style  # noqa: PLC0415

    use_flits_style()
    plt.rcParams.update(
        {
            "axes.linewidth": 0.9,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "font.size": 7.5,
            "legend.fontsize": 7.0,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            # STIX math keeps a serif look consistent with the AASTeX manuscript
            # while emitting a correct (ToUnicode) PDF text layer; Computer-Modern
            # mathtext maps glyphs like gamma to codepoint 0xb0 and corrupts the
            # embedded text under PDF search/copy/accessibility tools.
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "xtick.direction": "in",
            "xtick.labelsize": 6.8,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": 6.8,
            "ytick.right": True,
        }
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    n_subbands = len(plot_subbands)
    fig = plt.figure(
        figsize=(7.1, max(3.4, 1.75 * max(n_subbands, 1))),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        max(n_subbands, 1),
        2,
        width_ratios=[1.15, 1.0],
        hspace=0.36,
        wspace=0.25,
    )
    ax_bw = fig.add_subplot(gs[:, 0])
    fit_color = "black"
    stack_colors = [plt.get_cmap("plasma")(x) for x in np.linspace(0.15, 0.78, max(n_subbands, 1))]

    component_rows = []
    for payload in plot_subbands:
        subband = payload["summary"]
        for comp_idx, component in enumerate(subband["selected_components"], start=1):
            dnu = float(component.get("dnu_mhz", np.nan))
            if not (np.isfinite(dnu) and dnu > 0):
                continue
            component_rows.append(
                {
                    "subband": int(subband["index"]),
                    "component": comp_idx,
                    "center_freq_mhz": float(subband["center_freq_mhz"]),
                    "dnu_mhz": dnu,
                    "dnu_err_mhz": float(component.get("dnu_err", np.nan)),
                    "usable": not component.get("quality_flags"),
                }
            )

    clean_rows = [row for row in component_rows if row["usable"]]
    flagged_rows = [row for row in component_rows if not row["usable"]]
    bandwidth_limits = _bandwidth_axis_limits(component_rows)
    bandwidth_rows = clean_rows or flagged_rows
    if bandwidth_rows:
        x = [row["center_freq_mhz"] for row in bandwidth_rows]
        y = [row["dnu_mhz"] for row in bandwidth_rows]
        yerr = [
            row["dnu_err_mhz"]
            if row["usable"] and np.isfinite(row["dnu_err_mhz"]) and row["dnu_err_mhz"] > 0
            else 0.0
            for row in bandwidth_rows
        ]
        if any(err > 0 for err in yerr):
            ax_bw.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=MANUSCRIPT_PURPLE if clean_rows else "0.55",
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                alpha=0.9,
                zorder=1,
            )
        marker = "o" if clean_rows else "^"
        color = MANUSCRIPT_PURPLE if clean_rows else "0.55"
        edge = MANUSCRIPT_PURPLE if clean_rows else "0.35"
        label = r"Lorentzian Component ($\gamma$)" if clean_rows else "flagged only"
        ax_bw.scatter(
            x,
            y,
            marker=marker,
            s=24,
            color=color,
            edgecolors=edge,
            linewidths=0.7,
            alpha=0.95 if clean_rows else 0.55,
            label=label,
            zorder=3,
        )

    reference = _reference_power_law(component_rows, ref_alpha=4.0)
    if reference is not None:
        freqs = np.array(
            [row["center_freq_mhz"] for row in clean_rows or component_rows], dtype=float
        )
        nu = np.linspace(float(np.nanmin(freqs)) - 8.0, float(np.nanmax(freqs)) + 8.0, 240)
        dnu = reference["scale_mhz"] * (nu / reference["nu_ref_mhz"]) ** reference["alpha"]
        ax_bw.plot(
            nu,
            dnu,
            "--",
            color=MANUSCRIPT_GUIDE,
            lw=1.1,
            label=rf"$\gamma \propto \nu^{{{reference['alpha']:g}}}$",
            zorder=2,
        )

    if component_rows:
        ax_bw.set_yscale("log")
        ax_bw.set_ylim(*bandwidth_limits)
        ax_bw.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax_bw.yaxis.set_minor_formatter(NullFormatter())
    ax_bw.set_xlabel("Center Frequency (MHz)")
    ax_bw.set_ylabel(r"Decorrelation Bandwidth, $\gamma$ (MHz)")
    ax_bw.grid(axis="y", color=MANUSCRIPT_GRID, alpha=0.55, lw=0.45)
    ax_bw.tick_params(top=True, right=True, which="both", direction="in")
    ax_bw.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.35,
    )

    for row_idx, payload in enumerate(plot_subbands):
        ax_acf = fig.add_subplot(gs[row_idx, 1])
        lags = np.asarray(payload["lags"], dtype=float)
        acf = np.asarray(payload["acf"], dtype=float)
        err = np.asarray(payload["err"], dtype=float) if payload.get("err") is not None else None
        subband = payload["summary"]
        fit = payload["fit"]
        fit_range = float(subband["fit_range_mhz"])
        lag_zoom = min(fit_range, 12.0)
        center_freq = float(subband["center_freq_mhz"])

        display = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= lag_zoom)
        nonzero = display & (lags != 0)
        if not np.any(nonzero):
            ax_acf.axis("off")
            continue

        xfit = np.linspace(-lag_zoom, lag_zoom, 700)
        yfit = _model_curve(xfit, fit)
        if err is not None:
            err_mask = nonzero & np.isfinite(err) & (err > 0)
            if np.any(err_mask):
                idx = _decimated_indices(err_mask, max_points=450)
                ax_acf.errorbar(
                    lags[idx],
                    acf[idx],
                    yerr=err[idx],
                    fmt="none",
                    ecolor="lightgrey",
                    elinewidth=0.6,
                    alpha=0.75,
                    zorder=0,
                )
        ax_acf.plot(
            lags[nonzero],
            acf[nonzero],
            color=stack_colors[row_idx],
            lw=0.8,
            alpha=0.85,
            zorder=2,
        )

        ax_acf.plot(
            xfit,
            yfit,
            color=fit_color,
            lw=1.25,
            zorder=5,
        )

        redchi = subband.get("selected_redchi")
        label = rf"$\nu_c$={center_freq:.0f} MHz"
        if redchi is not None and np.isfinite(float(redchi)):
            label += "\n" + rf"$\chi_r^2$={float(redchi):.2f}"
        ax_acf.text(
            0.03,
            0.92,
            label,
            transform=ax_acf.transAxes,
            va="top",
            fontsize=6.4,
            bbox={"facecolor": "white", "alpha": 0.75, "boxstyle": "round,pad=0.2"},
        )
        ax_acf.axhline(0.0, color="0.7", lw=0.55, zorder=1)
        ax_acf.set_xlim(-lag_zoom, lag_zoom)
        ax_acf.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_acf.grid(axis="y", color=MANUSCRIPT_GRID, alpha=0.5, lw=0.4)
        if row_idx == n_subbands - 1:
            ax_acf.set_xlabel("Frequency Lag (MHz)")
        else:
            ax_acf.tick_params(labelbottom=False)
        if row_idx == max(n_subbands // 2 - 1, 0):
            ax_acf.set_ylabel(r"ACF Power  ($m^2$)")
        ax_acf.tick_params(top=True, right=True, which="both", direction="in")

    png = figure_dir / f"{burst}_{band}_acf_lorentzian_fits.png"
    svg = figure_dir / f"{burst}_{band}_acf_lorentzian_fits.svg"
    pdf = figure_dir / f"{burst}_{band}_acf_lorentzian_fits.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    # Vector PDF for direct manuscript/Overleaf inclusion (graphicspath consumes PDF).
    fig.savefig(pdf, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {"figure_png": str(png), "figure_svg": str(svg), "figure_pdf": str(pdf)}


def _summary_subband_status(subband: dict[str, Any]) -> str:
    components = subband.get("selected_components", [])
    if not components:
        return "flagged_only"
    usable = [not component.get("quality_flags") for component in components]
    if all(usable):
        return "clean"
    if any(usable):
        return "mixed"
    return "flagged_only"


def _summary_component_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        burst = result["burst"]
        selected_num_subbands = int(
            result.get("requested_num_subbands", result.get("num_subbands", 0))
        )
        for subband in result.get("subbands", []):
            status = _summary_subband_status(subband)
            for comp_idx, component in enumerate(subband.get("selected_components", []), start=1):
                dnu = float(component.get("dnu_mhz", np.nan))
                if not (np.isfinite(dnu) and dnu > 0):
                    continue
                dnu_err = float(component.get("dnu_err", np.nan))
                flags = component.get("quality_flags", [])
                rows.append(
                    {
                        "burst": burst,
                        "selected_num_subbands": selected_num_subbands,
                        "subband": int(subband["index"]),
                        "subband_status": status,
                        "center_freq_mhz": float(subband["center_freq_mhz"]),
                        "component": comp_idx,
                        "dnu_mhz": dnu,
                        "dnu_err_mhz": dnu_err,
                        "usable": not flags,
                        "quality_flags": list(flags),
                    }
                )
    return rows


def _plot_sample_summary(results: list[dict[str, Any]], *, figure_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: PLC0415

    from flits.plotting import use_flits_style  # noqa: PLC0415

    rows = _summary_component_rows(results)
    clean_rows = [row for row in rows if row["usable"]]
    plot_rows = clean_rows or rows

    use_flits_style()
    plt.rcParams.update(
        {
            "axes.linewidth": 0.9,
            "axes.labelsize": 8.0,
            "axes.titlesize": 7.2,
            "font.size": 7.2,
            "legend.fontsize": 7.0,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            # STIX math keeps a serif look consistent with the AASTeX manuscript
            # while emitting a correct (ToUnicode) PDF text layer; Computer-Modern
            # mathtext maps glyphs like gamma to codepoint 0xb0 and corrupts the
            # embedded text under PDF search/copy/accessibility tools.
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "xtick.direction": "in",
            "xtick.labelsize": 6.4,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": 6.4,
            "ytick.right": True,
        }
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    ncols = 4
    nrows = max(1, (len(results) + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, 5.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    if plot_rows:
        freqs_all = np.array([row["center_freq_mhz"] for row in rows], dtype=float)
        finite_freqs = freqs_all[np.isfinite(freqs_all)]
        x_span = float(np.nanmax(finite_freqs)) - float(np.nanmin(finite_freqs))
        x_pad = 0.07 * (x_span or 1.0)
        xlim = (float(np.nanmin(finite_freqs)) - x_pad, float(np.nanmax(finite_freqs)) + x_pad)
        ylim = _bandwidth_axis_limits(plot_rows)
        xguide = np.linspace(xlim[0], xlim[1], 160)
    else:
        xlim = (1300.0, 1500.0)
        ylim = (0.1, 10.0)
        xguide = np.linspace(*xlim, 160)

    rows_by_burst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_burst[str(row["burst"])].append(row)

    for panel_idx, ax in enumerate(axes_flat):
        if panel_idx >= len(results):
            ax.set_visible(False)
            continue

        result = results[panel_idx]
        burst = str(result["burst"])
        burst_rows = rows_by_burst.get(burst, [])
        burst_clean = [row for row in burst_rows if row["usable"]]
        burst_flagged = [row for row in burst_rows if not row["usable"]]

        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", color=MANUSCRIPT_GRID, alpha=0.5, lw=0.4)
        ax.tick_params(top=True, right=True, which="both", direction="in", pad=1.5)
        ax.text(
            0.04,
            0.92,
            burst,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
        )

        if burst_clean:
            freqs = np.array([row["center_freq_mhz"] for row in burst_clean], dtype=float)
            dnu = np.array([row["dnu_mhz"] for row in burst_clean], dtype=float)
            yerr = np.array(
                [
                    row["dnu_err_mhz"]
                    if np.isfinite(row["dnu_err_mhz"]) and row["dnu_err_mhz"] > 0
                    else np.nan
                    for row in burst_clean
                ],
                dtype=float,
            )
            finite_err = np.isfinite(yerr) & (yerr > 0)
            if np.any(finite_err):
                ax.errorbar(
                    freqs[finite_err],
                    dnu[finite_err],
                    yerr=yerr[finite_err],
                    fmt="none",
                    ecolor=MANUSCRIPT_PURPLE,
                    elinewidth=0.65,
                    alpha=0.75,
                    zorder=1,
                )
            ax.scatter(
                freqs,
                dnu,
                s=12,
                marker="o",
                color=MANUSCRIPT_PURPLE,
                edgecolors=MANUSCRIPT_PURPLE,
                linewidths=0.35,
                alpha=0.95,
                zorder=3,
            )
            reference = _reference_power_law(burst_clean, ref_alpha=4.0)
            if reference is not None:
                guide = reference["scale_mhz"] * (xguide / reference["nu_ref_mhz"]) ** reference[
                    "alpha"
                ]
                ax.plot(xguide, guide, "--", color=MANUSCRIPT_GUIDE, lw=0.75, zorder=2)
        elif burst_flagged:
            shown = [
                row
                for row in burst_flagged
                if ylim[0] <= float(row["dnu_mhz"]) <= ylim[1]
            ]
            if shown:
                ax.scatter(
                    [row["center_freq_mhz"] for row in shown],
                    [row["dnu_mhz"] for row in shown],
                    s=13,
                    marker="^",
                    color="0.6",
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.65,
                    zorder=3,
                )
            ax.text(
                0.04,
                0.08,
                "flagged",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.2,
                color="0.35",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no fit",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.2,
                color="0.35",
            )

        if panel_idx % ncols != 0:
            ax.tick_params(labelleft=False)
        if panel_idx < (nrows - 1) * ncols:
            ax.tick_params(labelbottom=False)

    fig.supxlabel("Center Frequency (MHz)", fontsize=8.5)
    fig.supylabel(r"Decorrelation Bandwidth, $\gamma$ (MHz)", fontsize=8.5)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MANUSCRIPT_PURPLE,
            markeredgecolor=MANUSCRIPT_PURPLE,
            markersize=4.2,
            label="clean DSA sub-band",
        ),
        Line2D(
            [0],
            [0],
            color=MANUSCRIPT_GUIDE,
            lw=0.85,
            ls="--",
            label=r"$\gamma \propto \nu^4$ per burst",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        columnspacing=1.2,
        handlelength=1.5,
    )

    png = figure_dir / "dsa_lorentzian_summary.png"
    svg = figure_dir / "dsa_lorentzian_summary.svg"
    pdf = figure_dir / "dsa_lorentzian_summary.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    # Vector PDF for direct manuscript/Overleaf inclusion (graphicspath consumes PDF).
    fig.savefig(pdf, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {
        "summary_figure_png": str(png),
        "summary_figure_svg": str(svg),
        "summary_figure_pdf": str(pdf),
    }


def _representative_width_mhz(verdict: dict[str, Any]) -> float | None:
    """Narrowest selected-component Delta_nu (MHz) from a compare verdict.

    The scintillation scale is the narrowest coherent component; the guards
    (off-pulse null, low-lag stability, harmonic systematic) compare a single
    representative width, so collapse the multi-component verdict to it.
    """
    fit = _selected_fit(verdict)
    if not fit.get("success", False):
        return None
    dnus = [
        float(c["dnu_mhz"])
        for c in fit.get("components", [])
        if c.get("dnu_mhz") is not None and np.isfinite(float(c.get("dnu_mhz", np.nan)))
    ]
    return min(dnus) if dnus else None


def _fit_width(
    lags: np.ndarray,
    acf: np.ndarray,
    err: np.ndarray | None,
    *,
    max_components: int,
) -> float | None:
    """Run the standard selector on a prepared ACF slice, return the width."""
    if lags.size < 4:
        return None
    try:
        verdict = compare_lorentzian_components(
            lags, acf, max_components=max_components, acf_err=err
        )
    except Exception as exc:  # a degenerate/failed fit is a null-ish outcome, not a crash
        logging.debug("guard fit failed: %s", exc)
        return None
    return _representative_width_mhz(verdict)


def _low_lag_excision_widths(
    lags: np.ndarray,
    acf: np.ndarray,
    err: np.ndarray | None,
    chan_width_mhz: float,
    *,
    max_components: int,
    ks: tuple[int, ...] = (1, 2, 3),
) -> dict[int, float | None]:
    """Refit after excising the first k positive-lag channel bins (arm B1).

    Drops bins with 0 < |lag| <= k * channel_width and refits; a real wing
    keeps its width, a low-lag artifact collapses.
    """
    out: dict[int, float | None] = {}
    for k in ks:
        keep = ~((np.abs(lags) > 0) & (np.abs(lags) <= (k + 0.5) * chan_width_mhz))
        e = None if err is None else err[keep]
        out[k] = _fit_width(lags[keep], acf[keep], e, max_components=max_components)
    return out


def _off_pulse_null_widths(
    pipe: ScintillationAnalysis,
    channel_slice: tuple[int, int],
    chan_width_mhz: float,
    fit_range_mhz: float,
    *,
    max_components: int,
    max_slices: int = 12,
) -> list[float]:
    """Fit off-pulse (noise) ACFs on the SAME channels as the sub-band (arm A).

    Slices the burst-free off-pulse window into non-overlapping segments the
    width of the burst, forms a channel-sliced time-averaged spectrum for each,
    computes its ACF with the identical estimator, and fits it. Widths that
    bracket the on-pulse scale mean the correlation is instrumental.
    """
    spec = pipe.masked_spectrum
    off_lims = pipe.off_pulse_lims
    burst_lims = pipe.burst_lims
    if spec is None or off_lims is None or burst_lims is None:
        return []
    c0, c1 = channel_slice
    w = max(int(burst_lims[1] - burst_lims[0]), 4)
    lo = int(off_lims[0]) + 2
    hi = int(off_lims[1]) - w
    if hi <= lo:
        return []
    starts = list(range(lo, hi, w + 4))[:max_slices]
    max_lag_bins = int(fit_range_mhz / chan_width_mhz) if chan_width_mhz > 0 else None

    widths: list[float] = []
    for s in starts:
        try:
            full_spec = spec.get_spectrum((s, s + w))  # time-avg, all channels
            sub = full_spec[c0:c1]
            # Self-normalize (off_burst_spectrum_mean=None): the off-burst mean
            # only scales the ACF denominator (amplitude), not the lag at which
            # it decorrelates, so it does not affect the fitted width we compare.
            acf_obj = analysis.calculate_acf(
                sub,
                chan_width_mhz,
                off_burst_spectrum_mean=None,
                max_lag_bins=max_lag_bins,
            )
        except Exception as exc:
            logging.debug("off-pulse slice %d ACF failed: %s", s, exc)
            acf_obj = None
        if acf_obj is None:
            continue
        lags = np.asarray(acf_obj.lags, dtype=float)
        acf = np.asarray(acf_obj.acf, dtype=float)
        err = None if acf_obj.err is None else np.asarray(acf_obj.err, dtype=float)
        keep = np.isfinite(lags) & np.isfinite(acf) & (np.abs(lags) <= fit_range_mhz)
        if err is not None:
            keep &= np.isfinite(err) & (err > 0)
        e = None if err is None else err[keep]
        width = _fit_width(lags[keep], acf[keep], e, max_components=max_components)
        if width is not None and np.isfinite(width) and width > 0:
            widths.append(float(width))
    return widths


def _fit_prepared_config(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    output_dir: Path,
    max_components: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    burst = str(cfg.get("burst_id", config_path.stem.split("_")[0]))

    analysis.clear_noise_acf_cache()
    pipe = ScintillationAnalysis(cfg)
    pipe.run()
    acf_results = pipe.acf_results
    if not acf_results or not acf_results.get("subband_acfs"):
        raise RuntimeError(f"{burst}: no ACF results produced")

    fit_cfg = cfg.get("analysis", {}).get("fitting", {})
    configured_fit_range = float(fit_cfg.get("fit_lagrange_mhz", 45.0))
    harmonic_cfg = fit_cfg.get("harmonic_mask", {})

    # Fail-closed provenance gate (evaluated once per burst; CHIME only).
    provenance = guards.chime_provenance_status(cfg)
    channel_slices = acf_results.get("subband_channel_slices") or []

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

        # --- Harmonic (coarse-channel comb) mask as a first-class fit mask ---
        # Previously the driver ignored analysis.fitting.harmonic_mask entirely
        # (the confirmed --band chime trap). Apply it to the fit-window slice
        # BEFORE the selector, and keep the unmasked slice for the systematic.
        unmasked_fit_lags, unmasked_fit_acf, unmasked_fit_err = fit_lags, fit_acf, fit_err
        fit_lags, fit_acf, fit_err, harmonic_record = guards.apply_harmonic_mask_to_fit(
            fit_lags, fit_acf, fit_err, harmonic_cfg
        )

        verdict = compare_lorentzian_components(
            fit_lags,
            fit_acf,
            max_components=max_components,
            acf_err=fit_err,
        )

        # --- Artifact-control guards (masked fit is the primary measurement) ---
        dnu_masked = _representative_width_mhz(verdict)
        # Harmonic-mask systematic: refit the unmasked slice and compare widths.
        if harmonic_record["enabled"]:
            dnu_unmasked = _fit_width(
                unmasked_fit_lags,
                unmasked_fit_acf,
                unmasked_fit_err,
                max_components=max_components,
            )
        else:
            dnu_unmasked = dnu_masked
        harmonic_systematic = guards.harmonic_mask_systematic(dnu_unmasked, dnu_masked)

        # Off-pulse ACF null (arm A) on identical channels for this sub-band.
        channel_slice = tuple(channel_slices[i]) if i < len(channel_slices) else None
        if channel_slice is not None:
            off_widths = _off_pulse_null_widths(
                pipe,
                channel_slice,
                chan_width,
                fit_range,
                max_components=max_components,
            )
        else:
            off_widths = []
        off_pulse_null = guards.off_pulse_null_verdict(dnu_masked, off_widths)

        # Low-lag excision stability (arm B1): refit dropping first k ch bins.
        excision_widths = _low_lag_excision_widths(
            fit_lags, fit_acf, fit_err, chan_width, max_components=max_components
        )
        low_lag_stability = guards.low_lag_stability_verdict(dnu_masked, excision_widths)

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
                "harmonic_mask": harmonic_record,
                "harmonic_mask_systematic": harmonic_systematic,
                "off_pulse_null": off_pulse_null,
                "low_lag_stability": low_lag_stability,
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

    # --- Burst-level artifact-control verdict ------------------------------
    # A CHIME burst fails the off-pulse null (or the low-lag stability) if ANY
    # sub-band that produced a usable width fails it. Aggregate the per-sub-band
    # verdicts to burst level, then combine with the fail-closed provenance gate.
    null_pass_flags = [
        s["off_pulse_null"]["null_pass"]
        for s in subbands
        if s.get("off_pulse_null", {}).get("null_pass") is not None
    ]
    stable_flags = [
        s["low_lag_stability"]["stable"]
        for s in subbands
        if s.get("low_lag_stability", {}).get("stable") is not None
    ]
    burst_null = {
        "null_pass": (all(null_pass_flags) if null_pass_flags else None),
        "n_subbands_judged": len(null_pass_flags),
        "n_subbands_failed": sum(1 for f in null_pass_flags if f is False),
    }
    burst_stability = {
        "stable": (all(stable_flags) if stable_flags else None),
        "n_subbands_judged": len(stable_flags),
        "n_subbands_failed": sum(1 for f in stable_flags if f is False),
    }
    finalized = guards.finalize_measurement_status(
        provenance,
        off_pulse_null=burst_null if burst_null["null_pass"] is not None else None,
        low_lag_stability=burst_stability if burst_stability["stable"] is not None else None,
    )
    result["artifact_control"] = {
        "provenance": provenance,
        "off_pulse_null": burst_null,
        "low_lag_stability": burst_stability,
        "measurement_status": finalized["status"],
        "downgraded": finalized["downgraded"],
        "failed_checks": finalized["failed_checks"],
    }
    result["measurement_status"] = finalized["status"]
    return result, plot_subbands


def _fit_one_burst(
    config_path: Path,
    *,
    output_dir: Path,
    max_components: int,
    make_figures: bool,
    band: str = "dsa",
) -> dict[str, Any]:
    loaded = config_mod.load_config(config_path)
    base_cfg = _config_for_fresh_acf(loaded, output_dir=output_dir)
    burst = str(base_cfg.get("burst_id", config_path.stem.split("_")[0]))

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
            _plot_burst_acfs(
                burst, plot_payloads[selected_n], figure_dir=output_dir / "figures", band=band
            )
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


def _artifact_control_summary(result: dict[str, Any]) -> str:
    """One-cell CHIME artifact-control verdict for the overview table."""
    ac = result.get("artifact_control")
    if not ac:
        return "measurement"
    status = ac.get("measurement_status", "measurement")
    if not ac.get("provenance", {}).get("is_chime"):
        return status  # DSA etc. — never demoted by the CHIME gate
    if status == guards.MEASUREMENT:
        return "measurement (CHIME, passed guards)"
    failed = ac.get("failed_checks", [])
    return "**diagnostic_only**<br>" + "; ".join(failed)


def _markdown_figure_path(figure_path: str, report_path: Path) -> Path:
    path = Path(figure_path)
    try:
        return path.resolve().relative_to(report_path.parent.resolve())
    except ValueError:
        return path


def _write_markdown(
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    path: Path,
    *,
    summary_figure_png: str | None = None,
) -> None:
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
        "### CHIME artifact-control guards",
        "",
        "CHIME upchannelized (gen-3) products carry instrumental structure that an",
        "ACF fit can mistake for scintillation (see",
        "`docs/rse/specs/experiment-freya-chime-instrumental-origin.md`). For",
        "`telescope: chime` this driver applies fail-closed guards and records them",
        "per sub-band in the JSON: (1) the coarse-channel **harmonic mask**",
        "(`analysis.fitting.harmonic_mask`) is applied to the fit-window ACF before",
        "the selector and the number of removed comb lag bins is recorded; (2) a",
        "**provenance gate** requires grid regularization, bandpass normalization,",
        "and the harmonic mask all be enabled; (3) an **off-pulse ACF null** refits",
        "burst-free noise slices on the identical sub-band channels and fails when",
        "they reproduce the on-pulse decorrelation scale; (4) a **low-lag excision**",
        "check refits after dropping the first few channel lags and fails when the",
        "width collapses (no resolved wing). The **harmonic-mask systematic** (fit",
        "with vs without the mask) is reported as a systematic band, not a",
        "correction. A CHIME burst is a `measurement` only if the provenance gate,",
        "the off-pulse null, and the low-lag stability all pass; otherwise it is",
        "`diagnostic_only`. DSA-band results are never demoted by these guards (no",
        "DSA config enables the harmonic mask, so the DSA fit is unchanged).",
        "",
        "## Burst Overview",
        "",
        "| burst | selected subbands | preferred n by subband | plurality n | median dnu by component (MHz) | status | selection note |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for result in results:
        usable = result.get("component_usable_median_dnu_mhz", {})
        if usable:
            med = ", ".join(f"c{k}={v:.4g}" for k, v in usable.items())
        else:
            med = "no unflagged components"
        lines.append(
            "| {burst} | {num_subbands} | {n_per_subband} | {burst_preferred_n} | {med} | {status} | {note} |".format(
                med=med or "-",
                status=_artifact_control_summary(result),
                note=_selection_summary(result),
                **result,
            )
        )

    if summary_figure_png:
        rel = _markdown_figure_path(summary_figure_png, path)
        lines.extend(
            [
                "",
                "## Paper Summary Figure",
                "",
                "The sample-level summary shows one bandwidth-scaling panel per",
                "burst. Filled circles are clean selected Lorentzian bandwidth",
                "measurements; dashed guides are shown only when at least two",
                "distinct clean sub-band frequencies anchor the fixed",
                "$\\gamma\\propto\\nu^4$ scaling. Selected components with quality",
                "flags remain in the tables and per-burst diagnostics.",
                "",
                f"![DSA Lorentzian bandwidth summary]({rel})",
            ]
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
            "Each burst figure follows the manuscript scintillation-summary",
            "layout: the left panel shows selected Lorentzian bandwidths versus",
            "DSA sub-band center frequency with a data-anchored reference",
            "$\\gamma\\propto\\nu^4$ curve where constrained, and the right column",
            "shows stacked frequency-lag ACF panels with the fitted total",
            "Lorentzian model overlaid.",
            "",
        ]
    )
    for result in results:
        figure_png = result.get("figure_png")
        if not figure_png:
            continue
        rel = _markdown_figure_path(figure_png, path)
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
    parser.add_argument(
        "--band",
        default="dsa",
        choices=("dsa", "chime"),
        help="Which band's configs/outputs to use ({burst}_{band}.yaml).",
    )
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
        config_path = Path("scintillation/configs/bursts") / f"{burst}_{args.band}.yaml"
        logging.info("Running %s from %s", burst, config_path)
        try:
            result = _fit_one_burst(
                config_path,
                output_dir=args.output_dir,
                max_components=args.max_components,
                make_figures=not args.no_figures,
                band=args.band,
            )
        except Exception as exc:
            logging.exception("%s failed", burst)
            failures.append({"burst": burst, "error": str(exc)})
            if not args.keep_going:
                raise
        else:
            results.append(result)
            burst_path = args.output_dir / f"{burst}_{args.band}_lorentzian_fits.json"
            burst_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))

    rows = _flatten_rows(results)
    summary_figures = {}
    if results and not args.no_figures:
        summary_figures = _plot_sample_summary(results, figure_dir=args.output_dir / "figures")
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
            **summary_figures,
            "notes": (
                "Fresh DSA ACFs from npz; YAML stored_fits and pkl ACF products are not read. "
                "Pipeline caches, diagnostic plots, MC noise templates, and 2D fits are disabled. "
                "When enabled, figures show a sample-level summary plus manuscript-style "
                "per-burst bandwidth-scaling and stacked ACF diagnostics."
            ),
        },
        "results": results,
    }
    (args.output_dir / f"{args.band}_lorentzian_fits.json").write_text(
        json.dumps(_jsonable(all_results), indent=2, sort_keys=True)
    )
    _write_csv(rows, args.output_dir / f"{args.band}_lorentzian_components.csv")
    _write_markdown(
        results,
        rows,
        args.output_dir / f"{args.band.upper()}_LORENTZIAN_FITS.md",
        summary_figure_png=summary_figures.get("summary_figure_png"),
    )

    if failures:
        logging.error("Completed with %d failures", len(failures))
        return 1
    logging.info("Completed %d bursts; wrote %s", len(results), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
