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


def _assign_gamma_tracks(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label Lorentzian scales by increasing width within each sub-band.

    ``gamma_1`` is always the narrowest fitted scale, ``gamma_2`` the next
    broader scale, and so on.  This deterministic ordering makes component
    membership explicit before either plotting or fitting the frequency law.
    """
    tracked = []
    for subband in sorted({int(row["subband"]) for row in component_rows}):
        rows = [row for row in component_rows if int(row["subband"]) == subband]
        for gamma_track, row in enumerate(
            sorted(rows, key=lambda item: float(item["dnu_mhz"])), start=1
        ):
            tracked.append({**row, "gamma_track": gamma_track})
    return tracked


def _fit_gamma_power_law(
    component_rows: list[dict[str, Any]], *, nu_ref_mhz: float = 1400.0
) -> dict[str, Any] | None:
    """Fit gamma = gamma_ref (nu / nu_ref)^alpha in log space."""
    usable = [
        row
        for row in component_rows
        if row.get("gamma_track") == 1
        and row["usable"]
        and np.isfinite(row["dnu_err_mhz"])
        and row["dnu_err_mhz"] > 0
        and row["dnu_mhz"] > 0
    ]
    if len(usable) < 2:
        return None
    nu = np.asarray([row["center_freq_mhz"] for row in usable], dtype=float)
    gamma = np.asarray([row["dnu_mhz"] for row in usable], dtype=float)
    gamma_err = np.asarray([row["dnu_err_mhz"] for row in usable], dtype=float)
    x = np.log(nu / nu_ref_mhz)
    y = np.log(gamma)
    sigma_y = gamma_err / gamma
    design = np.column_stack((np.ones_like(x), x))
    precision = 1.0 / np.square(sigma_y)
    normal = design.T @ (precision[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    coefficients = covariance @ (design.T @ (precision * y))
    return {
        "nu_ref_mhz": nu_ref_mhz,
        "gamma_ref_mhz": float(np.exp(coefficients[0])),
        "alpha": float(coefficients[1]),
        "covariance": covariance,
        "n_fit_components": len(usable),
        "selection_policy": "gamma_1_only",
        "included_tracks": [1],
    }


def _summary_value(summary: dict[str, Any], name: str) -> tuple[float, float, float] | None:
    """Return median, lower error, upper error from a joint-fit summary."""
    value = summary.get(name)
    if not isinstance(value, dict):
        value = summary.get("percentiles", {}).get(name)
    if not isinstance(value, dict) or value.get("median") is None:
        return None
    median = float(value["median"])
    err_minus = float(value.get("err_minus", median - float(value.get("lower", median))))
    err_plus = float(value.get("err_plus", float(value.get("upper", median)) - median))
    return median, max(err_minus, 0.0), max(err_plus, 0.0)


def _pbf_bandwidth_mhz(
    nu_mhz: np.ndarray, *, tau_1ghz_ms: float, alpha: float, c1: float
) -> np.ndarray:
    """Convert a PBF tau(nu) law to decorrelation bandwidth in MHz."""
    tau_nu_ms = tau_1ghz_ms * np.power(np.asarray(nu_mhz, dtype=float) / 1000.0, -alpha)
    return c1 / (2.0 * np.pi * tau_nu_ms * 1e3)


def _pbf_roster_entry_is_eligible(entry: dict[str, Any] | None) -> bool:
    """Fail closed unless the locked roster admits the joint-fit product."""
    return bool(
        entry
        and entry.get("gate_final") in {"PASS", "MARGINAL"}
        and entry.get("rail_class") in {"interior", "railed-hi"}
        and entry.get("fit_json")
    )


def _load_pbf_fit_for_burst(burst: str) -> dict[str, Any] | None:
    """Load the roster-adjudicated beta-coherent joint-fit summary."""
    roster_path = (
        REPO_ROOT
        / "analysis"
        / "scattering-refit-2026-06"
        / "citable_alpha_roster.json"
    )
    if not roster_path.exists():
        return None
    roster = json.loads(roster_path.read_text())
    entries = []
    for key in ("tier_a_fully_adjudicated", "tier_b_provisional_pending_s2"):
        entries.extend(roster.get(key, []))
    exemplar = roster.get("multiplicity_exemplar")
    if isinstance(exemplar, dict):
        entries.append(exemplar)
    entry = next((item for item in entries if item.get("nickname") == burst), None)
    if not _pbf_roster_entry_is_eligible(entry):
        return None
    fit_path = REPO_ROOT / str(entry["fit_json"])
    if not fit_path.is_file():
        return None
    payload = json.loads(fit_path.read_text())
    payload["_source"] = str(fit_path)
    payload["_roster_source"] = str(roster_path)
    payload["_roster_locked_utc"] = roster.get("locked_utc")
    payload["_roster_gate_final"] = entry.get("gate_final")
    payload["_roster_rail_class"] = entry.get("rail_class")
    return payload


def plot_burst_acf_diagnostic(
    burst: str,
    plot_subbands: list[dict[str, Any]],
    *,
    figure_dir: Path,
    band: str = "dsa",
    pbf_fit: dict[str, Any] | None = None,
    pbf_c1: float = 1.16,
) -> dict[str, Any]:
    """Render the canonical experiment-style per-burst ACF diagnostic.

    The structured ACF payload and tracked joint-PBF summary remain the
    scientific sources of truth. This function only renders them and never
    changes measurement eligibility.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: PLC0415

    from flits.plotting import use_flits_style  # noqa: PLC0415

    use_flits_style()
    plt.rcParams.update(
        {
            "axes.linewidth": 0.9,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "font.size": 9.5,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            # STIX math keeps a serif look consistent with the AASTeX manuscript
            # while emitting a correct (ToUnicode) PDF text layer; Computer-Modern
            # mathtext maps glyphs like gamma to codepoint 0xb0 and corrupts the
            # embedded text under PDF search/copy/accessibility tools.
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "xtick.direction": "in",
            "xtick.labelsize": 8.5,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": 8.5,
            "ytick.right": True,
        }
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    n_subbands = len(plot_subbands)
    n_acf_rows = max(1, n_subbands)
    fig = plt.figure(figsize=(13.0, max(6.5, 2.0 * n_acf_rows)), constrained_layout=True)
    gs = fig.add_gridspec(
        n_acf_rows,
        2,
        width_ratios=[1.15, 1.0],
        hspace=0.24,
        wspace=0.25,
    )
    left = gs[:, 0].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.10)
    ax_gamma1 = fig.add_subplot(left[0])
    ax_broad = fig.add_subplot(left[1], sharex=ax_gamma1)
    acf_colors = plt.get_cmap("plasma")(np.linspace(0.15, 0.75, max(n_subbands, 1)))

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

    component_rows = _assign_gamma_tracks(component_rows)
    gamma1_rows = [row for row in component_rows if row["gamma_track"] == 1]
    broad_rows = [row for row in component_rows if row["gamma_track"] > 1]

    def plot_track_rows(ax, rows, *, marker, color, label):
        if not rows:
            return
        x = [row["center_freq_mhz"] for row in rows]
        y = [row["dnu_mhz"] for row in rows]
        yerr = [
            row["dnu_err_mhz"]
            if np.isfinite(row["dnu_err_mhz"]) and row["dnu_err_mhz"] > 0
            else 0.0
            for row in rows
        ]
        if any(err > 0 for err in yerr):
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                alpha=0.9,
                zorder=1,
            )
        ax.scatter(
            x,
            y,
            marker=marker,
            s=24,
            color=color,
            edgecolors=[color if row["usable"] else "0.35" for row in rows],
            linewidths=0.7,
            alpha=[0.95 if row["usable"] else 0.45 for row in rows],
            label=label,
            zorder=3,
        )

    plot_track_rows(
        ax_gamma1,
        gamma1_rows,
        marker="o",
        color=MANUSCRIPT_PURPLE,
        label=r"$\gamma_1$ (narrow component; scaling-fit input)",
    )
    track_styles = {2: ("s", "#D55E00"), 3: ("D", "#0072B2")}
    for track in sorted({row["gamma_track"] for row in broad_rows}):
        marker, color = track_styles.get(track, ("P", "0.35"))
        plot_track_rows(
            ax_broad,
            [row for row in broad_rows if row["gamma_track"] == track],
            marker=marker,
            color=color,
            label=rf"$\gamma_{track}$ (excluded from scaling fit)",
        )

    freq_rows = component_rows
    nu = None
    if freq_rows:
        freqs = np.asarray([row["center_freq_mhz"] for row in freq_rows], dtype=float)
        nu = np.linspace(float(np.nanmin(freqs)) - 12.0, float(np.nanmax(freqs)) + 12.0, 320)

    gamma_fit = _fit_gamma_power_law(component_rows)
    gamma_envelope = None
    if gamma_fit is not None and nu is not None:
        log_nu = np.log(nu / gamma_fit["nu_ref_mhz"])
        design = np.column_stack((np.ones_like(log_nu), log_nu))
        log_gamma = np.log(gamma_fit["gamma_ref_mhz"]) + gamma_fit["alpha"] * log_nu
        log_sigma = np.sqrt(
            np.einsum("ij,jk,ik->i", design, gamma_fit["covariance"], design)
        )
        gamma_curve = np.exp(log_gamma)
        gamma_envelope = (
            np.exp(log_gamma - log_sigma),
            np.exp(log_gamma + log_sigma),
        )
        ax_gamma1.fill_between(
            nu,
            gamma_envelope[0],
            gamma_envelope[1],
            color=MANUSCRIPT_PURPLE,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax_gamma1.plot(
            nu,
            gamma_curve,
            color=MANUSCRIPT_PURPLE,
            lw=1.7,
            label=rf"$\gamma_1$ scaling fit: $\alpha_1={gamma_fit['alpha']:.2f}$",
            zorder=2,
        )
    elif component_rows:
        ax_gamma1.text(
            0.97,
            0.97,
            r"ACF bandwidth fit unavailable ($<2$ usable primary components)",
            transform=ax_gamma1.transAxes,
            fontsize=8.0,
            color="0.35",
            ha="right",
            va="top",
        )

    if pbf_fit is None:
        pbf_fit = _load_pbf_fit_for_burst(burst)
    tau_summary = _summary_value(pbf_fit, "tau_1ghz") if pbf_fit else None
    alpha_summary = _summary_value(pbf_fit, "alpha") if pbf_fit else None
    pbf_envelope = None
    if tau_summary is not None and alpha_summary is not None and nu is not None:
        tau_ref, tau_minus, tau_plus = tau_summary
        alpha_pbf, alpha_minus, alpha_plus = alpha_summary
        pbf_curve = _pbf_bandwidth_mhz(
            nu, tau_1ghz_ms=tau_ref, alpha=alpha_pbf, c1=pbf_c1
        )

        rng = np.random.default_rng(20260710)
        tau_samples = np.clip(
            rng.normal(tau_ref, 0.5 * (tau_minus + tau_plus), 4096),
            np.finfo(float).tiny,
            None,
        )
        alpha_samples = rng.normal(alpha_pbf, 0.5 * (alpha_minus + alpha_plus), 4096)
        sample_curves = _pbf_bandwidth_mhz(
            nu[None, :],
            tau_1ghz_ms=tau_samples[:, None],
            alpha=alpha_samples[:, None],
            c1=pbf_c1,
        )
        pbf_lo, pbf_hi = np.nanpercentile(sample_curves, [16.0, 84.0], axis=0)
        pbf_envelope = (pbf_lo, pbf_hi)
        ax_gamma1.fill_between(
            nu,
            pbf_lo,
            pbf_hi,
            color="#2ca25f",
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax_gamma1.plot(
            nu,
            pbf_curve,
            "-.",
            color="#238b45",
            lw=1.5,
            label=rf"PBF-derived (marginal approx.): $\alpha_\tau={alpha_pbf:.2f}$, $C_1={pbf_c1:g}$",
            zorder=2,
        )

    for ax, rows in ((ax_gamma1, gamma1_rows), (ax_broad, broad_rows)):
        ax.set_yscale("log")
        scale_values = []
        for row in rows:
            scale_values.append(row["dnu_mhz"])
            if np.isfinite(row["dnu_err_mhz"]) and row["dnu_err_mhz"] > 0:
                scale_values.append(max(row["dnu_mhz"] - row["dnu_err_mhz"], 0.0))
                scale_values.append(row["dnu_mhz"] + row["dnu_err_mhz"])
        for envelope in ((gamma_envelope, pbf_envelope) if ax is ax_gamma1 else ()):
            if envelope is not None:
                scale_values.extend((float(np.nanmin(envelope[0])), float(np.nanmax(envelope[1]))))
        finite_positive = np.asarray(
            [value for value in scale_values if np.isfinite(value) and value > 0], dtype=float
        )
        if finite_positive.size:
            y_lower = float(np.nanmin(finite_positive)) / 1.6
            y_upper = float(np.nanmax(finite_positive)) * 1.8
            ax.set_ylim(y_lower, y_upper)
            off_scale = [row for row in rows if not row["usable"] and row["dnu_mhz"] > y_upper]
            if off_scale:
                label = "; ".join(
                    f"SB {row['subband'] + 1}: {row['dnu_mhz']:.2g} MHz"
                    for row in off_scale
                )
                ax.text(
                    0.97,
                    0.88,
                    f"flagged, off scale: {label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    color="0.4",
                )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", color=MANUSCRIPT_GRID, alpha=0.55, lw=0.45)
        ax.tick_params(top=True, right=True, which="both", direction="in")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="upper left",
                frameon=True,
                framealpha=0.9,
                borderpad=0.3,
                handlelength=1.35,
            )
    if not broad_rows:
        ax_broad.text(
            0.5,
            0.5,
            r"No broader Lorentzian component selected",
            transform=ax_broad.transAxes,
            ha="center",
            va="center",
            color="0.4",
        )
    ax_gamma1.tick_params(labelbottom=False)
    ax_broad.set_xlabel("Center Frequency (MHz)")
    ax_gamma1.set_ylabel(r"Primary bandwidth, $\gamma_1$ (MHz)")
    ax_broad.set_ylabel(r"Broader scales (MHz)")

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

        xfit = np.linspace(-lag_zoom, lag_zoom, 900)
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
        data_idx = _decimated_indices(nonzero, max_points=450)
        order = data_idx[np.argsort(lags[data_idx])]
        ax_acf.plot(
            lags[order],
            acf[order],
            color=acf_colors[row_idx],
            lw=0.85,
            alpha=0.9,
            zorder=2,
        )

        ax_acf.plot(
            xfit,
            yfit,
            color="black",
            lw=1.65,
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
            fontsize=8.0,
            bbox={"facecolor": "white", "alpha": 0.75, "boxstyle": "round,pad=0.2"},
        )
        ax_acf.axhline(0.0, color="0.7", lw=0.55, zorder=1)
        if band == "chime":
            for harmonic in np.arange(0.390625, lag_zoom + 0.390625, 0.390625):
                ax_acf.axvline(harmonic, color="0.6", ls=":", lw=0.7, alpha=0.8)
                ax_acf.axvline(-harmonic, color="0.6", ls=":", lw=0.7, alpha=0.8)
        ax_acf.set_xlim(-lag_zoom, lag_zoom)
        ax_acf.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if row_idx == n_subbands - 1:
            ax_acf.set_xlabel("Frequency Lag (MHz)")
        if row_idx == max(0, n_subbands // 2 - 1):
            ax_acf.set_ylabel(r"ACF Power ($m^2$)")
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
    gamma_fit_record = None
    if gamma_fit is not None:
        gamma_fit_record = {
            "nu_ref_mhz": gamma_fit["nu_ref_mhz"],
            "gamma_ref_mhz": gamma_fit["gamma_ref_mhz"],
            "alpha": gamma_fit["alpha"],
            "covariance": np.asarray(gamma_fit["covariance"], dtype=float).tolist(),
            "n_fit_components": gamma_fit["n_fit_components"],
            "selection_policy": gamma_fit["selection_policy"],
            "included_tracks": gamma_fit["included_tracks"],
        }
    pbf_overlay = None
    if tau_summary is not None and alpha_summary is not None:
        pbf_overlay = {
            "source": pbf_fit.get("_source") if pbf_fit else None,
            "roster_source": pbf_fit.get("_roster_source") if pbf_fit else None,
            "roster_locked_utc": pbf_fit.get("_roster_locked_utc") if pbf_fit else None,
            "roster_gate_final": pbf_fit.get("_roster_gate_final") if pbf_fit else None,
            "roster_rail_class": pbf_fit.get("_roster_rail_class") if pbf_fit else None,
            "tau_1ghz_ms": tau_summary[0],
            "tau_1ghz_err_minus_ms": tau_summary[1],
            "tau_1ghz_err_plus_ms": tau_summary[2],
            "alpha": alpha_summary[0],
            "alpha_err_minus": alpha_summary[1],
            "alpha_err_plus": alpha_summary[2],
            "c1": pbf_c1,
            "relation": "delta_nu_mhz = C1 / (2 pi tau_ms 1e3)",
            "uncertainty_method": (
                "approximate independent Gaussian draws from marginal 16th-84th "
                "percentile summaries; tau-alpha covariance unavailable"
            ),
        }
    return {
        "figure_png": str(png),
        "figure_svg": str(svg),
        "figure_pdf": str(pdf),
        "gamma_power_law_fit": gamma_fit_record,
        "pbf_overlay": pbf_overlay,
    }


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
            plot_burst_acf_diagnostic(
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
            "Each burst figure follows the Freya instrumental-origin experiment's",
            "publication layout: a fitted bandwidth-frequency relation beside",
            "stacked symmetric-lag ACF panels, with the selected Lorentzian model",
            "overlaid in black. When available, the tracked time-frequency joint",
            "PBF fit supplies a second predicted bandwidth curve using C1=1.16.",
            "These figures remain diagnostic until the",
            "upstream Phase 0 producer/ACF/fitting validation passes.",
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
