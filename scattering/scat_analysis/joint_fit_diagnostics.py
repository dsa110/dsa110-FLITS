"""Deterministic diagnostics and fit-panel rendering for controlled joint fits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from flits.fitting.rails import classify_rail


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _parameter_name(percentiles: Mapping[str, Any], base: str, index: int) -> str:
    numbered = f"{base}{index}"
    if numbered in percentiles:
        return numbered
    if index == 1 and base in percentiles:
        return base
    raise ValueError(f"fit summary lacks component parameter {numbered}")


def _residual_diagnostics(grid: Mapping[str, np.ndarray], band: str) -> dict[str, Any]:
    valid = np.asarray(grid[f"valid{band}"], dtype=bool)
    data = np.asarray(grid[f"data{band}"], dtype=float)[valid]
    model = np.asarray(grid[f"model{band}"], dtype=float)[valid]
    noise = np.asarray(grid[f"noise{band}"], dtype=float)[valid]
    if data.shape != model.shape or data.shape[0] != noise.size or np.any(noise <= 0):
        raise ValueError(f"invalid {band} residual inputs")
    normalized = (data - model) / noise[:, None]
    profile = normalized.sum(axis=0) / np.sqrt(normalized.shape[0])
    return {
        "normalized_map_shape": list(normalized.shape),
        "normalized_map_sha256": array_sha256(normalized.astype("<f8")),
        "normalized_map_mean": float(normalized.mean()),
        "normalized_map_std": float(normalized.std()),
        "normalized_map_max_abs": float(np.max(np.abs(normalized))),
        "band_summed_profile_sha256": array_sha256(profile.astype("<f8")),
        "band_summed_profile": profile.tolist(),
        "profile_max_abs": float(np.max(np.abs(profile))),
        "profile_samples_over_abs_3": int(np.sum(np.abs(profile) > 3.0)),
        "residual_mean_square": float(grid[f"residual_mean_square{band}"]),
    }


def build_diagnostics(
    fit_summary: Mapping[str, Any],
    grid: Mapping[str, np.ndarray],
    *,
    samples: np.ndarray,
    weights: np.ndarray,
    param_names: list[str],
    broad_width_ratio: float = 5.0,
    low_fluence_fraction: float = 0.05,
) -> dict[str, Any]:
    """Reconstruct component and residual checks from the controlled outputs."""
    counts = {"C": int(fit_summary["components_C"]), "D": int(fit_summary["components_D"])}
    if counts != {"C": int(grid["nC"]), "D": int(grid["nD"])}:
        raise ValueError("component counts differ between fit summary and model grid")
    percentiles = fit_summary["percentiles"]
    components = {}
    degeneracies = []
    for band in ("C", "D"):
        time = np.asarray(grid[f"time{band}"], dtype=float)
        span = float(np.ptp(time))
        fluence = np.asarray(grid[f"fluence{band}"], dtype=float)
        if fluence.shape != (counts[band],) or not np.isfinite(fluence).all():
            raise ValueError(f"invalid {band} component fluence")
        total_fluence = float(fluence.sum())
        if total_fluence <= 0:
            raise ValueError(f"non-positive {band} modeled fluence")
        fractions = fluence / total_fluence
        rows = []
        for index in range(1, counts[band] + 1):
            t0_name = _parameter_name(percentiles, f"t0_{band}", index)
            width_name = _parameter_name(percentiles, f"zeta_{band}", index)
            arrival = percentiles[t0_name]
            width = percentiles[width_name]
            ratio = float(width["median"] / span)
            fraction = float(fractions[index - 1])
            row = {
                "component": index,
                "arrival_ms_16_50_84": [
                    float(arrival["lower"]),
                    float(arrival["median"]),
                    float(arrival["upper"]),
                ],
                "width_ms_16_50_84": [
                    float(width["lower"]),
                    float(width["median"]),
                    float(width["upper"]),
                ],
                "arrival_interval_inside_window": bool(
                    time.min() <= arrival["lower"] and arrival["upper"] <= time.max()
                ),
                "width_to_window_ratio": ratio,
                "modeled_fluence_fraction": fraction,
                "broad_vs_window": bool(ratio >= broad_width_ratio),
                "low_fluence": bool(fraction <= low_fluence_fraction),
            }
            rows.append(row)
            if row["broad_vs_window"] or row["low_fluence"]:
                degeneracies.append({"band": band, **row})
        components[band] = {
            "fitted_window_ms": [float(time.min()), float(time.max())],
            "components": rows,
        }

    if "beta" not in param_names:
        raise ValueError("weighted posterior lacks sampled beta")
    beta_samples = np.asarray(samples, dtype=float)[:, param_names.index("beta")]
    beta_lo, beta_hi = (float(value) for value in fit_summary["beta_bounds"])
    rail = classify_rail(
        lo=beta_lo,
        hi=beta_hi,
        samples=beta_samples,
        weights=np.asarray(weights, dtype=float),
    ).asdict()
    return {
        "schema": "flits-controlled-joint-fit-diagnostics/v1",
        "burst": fit_summary["burst"],
        "source_revision": fit_summary["source_revision"],
        "controlled_contract_sha256": fit_summary["controlled_contract_sha256"],
        "resolved_fit_identity_sha256": fit_summary["resolved_fit_identity_sha256"],
        "component_counts": counts,
        "component_diagnostics": components,
        "component_degeneracies": degeneracies,
        "residual_morphology": {band: _residual_diagnostics(grid, band) for band in ("C", "D")},
        "prior_rail": rail,
        "guard_contract": {
            "broad_width_to_window_ratio": broad_width_ratio,
            "low_fluence_fraction": low_fluence_fraction,
            "all_arrival_intervals_inside_fitted_windows": all(
                row["arrival_interval_inside_window"]
                for diagnostic in components.values()
                for row in diagnostic["components"]
            ),
            "component_degeneracy_flagged": bool(degeneracies),
        },
        "agent_recommendation": "pending_visual_assessment",
        "fit_value_trust": "pending",
    }


def write_diagnostics(path: Path, diagnostics: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_fit_panel(grid: Mapping[str, np.ndarray], output: Path) -> None:
    """Render CHIME/FRB and DSA-110 data, model, and normalized residuals."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    with mpl.rc_context(
        {
            **mpl.rcParamsDefault,
            "svg.hashsalt": "faber2026-controlled-joint-fit-v1",
            "font.size": 8,
            "axes.linewidth": 0.6,
        }
    ):
        figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
        for row, (band, label) in enumerate((("C", "CHIME/FRB"), ("D", "DSA-110"))):
            valid = np.asarray(grid[f"valid{band}"], dtype=bool)
            data = np.asarray(grid[f"data{band}"], dtype=float)[valid]
            model = np.asarray(grid[f"model{band}"], dtype=float)[valid]
            noise = np.asarray(grid[f"noise{band}"], dtype=float)[valid]
            residual = (data - model) / noise[:, None]
            frequency = np.asarray(grid[f"freq{band}"], dtype=float)[valid]
            time = np.asarray(grid[f"time{band}"], dtype=float)
            extent = [time.min(), time.max(), frequency.min(), frequency.max()]
            intensity_limit = float(np.nanpercentile(np.abs(np.concatenate((data, model))), 99.5))
            residual_limit = max(float(np.nanpercentile(np.abs(residual), 99.5)), 1.0)
            for column, (array, heading, limit, cmap) in enumerate(
                (
                    (data, "Data", intensity_limit, "viridis"),
                    (model, "Model", intensity_limit, "viridis"),
                    (residual, "Residual / noise", residual_limit, "RdBu_r"),
                )
            ):
                axes[row, column].imshow(
                    array,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    extent=extent,
                    cmap=cmap,
                    vmin=-limit if column == 2 else 0.0,
                    vmax=limit,
                    rasterized=True,
                )
                if row == 0:
                    axes[row, column].set_title(heading)
                axes[row, column].set_xlabel("Time (ms)")
                if column == 0:
                    axes[row, column].set_ylabel(f"{label}\nFrequency (GHz)")
                else:
                    axes[row, column].set_yticklabels([])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            format="svg",
            metadata={"Date": None, "Creator": "Faber2026 controlled joint-fit renderer"},
        )
        plt.close(figure)
