from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dispersion.dm_campaign.injection import INSTRUMENTS, InjectionSpec, inject_pulse

from .cutoff import width_derived_cutoffs
from .search import search_dm


def _cell(
    instrument: str,
    snr: float,
    width_ms: float,
    seed: int,
    *,
    tau_1ghz_ms: float = 0.0,
    components: int = 2,
) -> dict:
    geometry = INSTRUMENTS[instrument]
    rng = np.random.default_rng(seed)
    frequency_ghz = np.linspace(
        geometry["f_lo_ghz"], geometry["f_hi_ghz"], geometry["nchan"]
    )
    noise = rng.normal(size=(geometry["nchan"], geometry["ntime"])).astype(np.float32)
    truth_residual = float(rng.uniform(-geometry["truth_dm"], geometry["truth_dm"]))
    spec = InjectionSpec(
        dm_offset=truth_residual,
        snr=snr,
        width_ms=width_ms,
        tau_1ghz_ms=tau_1ghz_ms,
        components=components,
        sep_ms=max(1.0, 3.0 * width_ms),
    )
    waterfall, _ = inject_pulse(noise, frequency_ghz, geometry["dt_ms"], spec, rng)
    coarse = np.arange(
        -geometry["window"], geometry["window"] + 0.125, 0.25, dtype=float
    )
    cutoff = width_derived_cutoffs(waterfall, geometry["dt_ms"] * 1e-3)[1]
    result = search_dm(
        waterfall=waterfall,
        frequencies_mhz=frequency_ghz * 1e3,
        sample_time_s=geometry["dt_ms"] * 1e-3,
        reference_dm=500.0,
        coarse_grid=coarse,
        fine_step=0.02,
        f_cut_hz=cutoff,
    )
    return {
        "instrument": instrument,
        "snr": snr,
        "width_ms": width_ms,
        "seed": seed,
        "tau_1ghz_ms": tau_1ghz_ms,
        "components": components,
        "truth_residual_dm": truth_residual,
        "recovered_residual_dm": result.residual_dm,
        "error": result.residual_dm - truth_residual,
        "edge_peak": result.edge_peak,
        "f_cut_hz": list(cutoff),
    }


def run_quick_injections(output_path: Path) -> dict:
    rows = [
        _cell(instrument, snr, width, seed)
        for instrument in ("chime", "dsa")
        for snr in (12.0, 25.0)
        for width in (0.3, 2.0)
        for seed in (0, 1)
    ]
    summaries = {}
    for instrument in ("chime", "dsa"):
        selected = [row for row in rows if row["instrument"] == instrument]
        errors = np.asarray([row["error"] for row in selected], dtype=float)
        limit = 0.05 if instrument == "chime" else 0.20
        summaries[instrument] = {
            "count": len(selected),
            "median_bias": float(np.median(errors)),
            "absolute_median_bias": float(abs(np.median(errors))),
            "scatter68": float(np.percentile(np.abs(errors), 68)),
            "edge_count": sum(bool(row["edge_peak"]) for row in selected),
            "bias_limit": limit,
        }
        summaries[instrument]["pass"] = bool(
            summaries[instrument]["absolute_median_bias"] <= limit
            and summaries[instrument]["edge_count"] == 0
        )
    report = {
        "tier": "quick",
        "rows": rows,
        "summaries": summaries,
        "pass": all(summary["pass"] for summary in summaries.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return report


def run_full_injections(output_path: Path) -> dict:
    """Calibration/validation matrix over the declared initial morphology domain."""
    rows = [
        _cell(
            instrument,
            snr,
            width,
            seed,
            tau_1ghz_ms=tau,
            components=components,
        )
        for instrument in ("chime", "dsa")
        for snr in (12.0, 25.0, 50.0)
        for width in (0.3, 1.0, 2.0)
        for tau in (0.0, 0.1, 1.0)
        for components in (1, 2)
        for seed in range(10)
    ]
    rows.extend(
        _cell(instrument, snr, width, seed, tau_1ghz_ms=0.0, components=1)
        for instrument in ("chime", "dsa")
        for snr in (12.0, 25.0, 50.0)
        for width in (0.3, 1.0, 2.0)
        for seed in range(10, 100)
        if not (
            instrument == "dsa"
            and snr < 50.0
        )
    )
    summaries = {}
    for instrument in ("chime", "dsa"):
        def supported_accuracy(row: dict, instrument_name: str = instrument) -> bool:
            if row["instrument"] != instrument_name:
                return False
            if row["tau_1ghz_ms"] != 0.0 or row["components"] != 1:
                return False
            return not (
                instrument_name == "dsa"
                and row["snr"] < 50.0
            )

        calibration = [
            row for row in rows if supported_accuracy(row) and row["seed"] < 60
        ]
        validation = [
            row for row in rows if supported_accuracy(row) and row["seed"] >= 60
        ]
        validation_error = np.asarray([row["error"] for row in validation], dtype=float)
        calibration_surface = {}
        covered68 = []
        covered95 = []
        catastrophic_flags = []
        inflation68 = 1.0
        inflation95 = 1.0
        for snr in sorted({row["snr"] for row in calibration}):
            calibration_error = np.abs(
                [row["error"] for row in calibration if row["snr"] == snr]
            )
            sigma68_snr = inflation68 * float(np.percentile(calibration_error, 68))
            radius95_snr = inflation95 * float(np.percentile(calibration_error, 95))
            calibration_surface[str(snr)] = {
                "sigma68": sigma68_snr,
                "radius95": radius95_snr,
                "count": int(calibration_error.size),
            }
            for row in validation:
                if row["snr"] != snr:
                    continue
                absolute_error = abs(row["error"])
                covered68.append(absolute_error <= sigma68_snr)
                covered95.append(absolute_error <= radius95_snr)
                catastrophic_flags.append(absolute_error > max(0.5, 3.0 * sigma68_snr))
        sigma68 = float(np.median([value["sigma68"] for value in calibration_surface.values()]))
        radius95 = float(np.median([value["radius95"] for value in calibration_surface.values()]))
        coverage68 = float(np.mean(covered68))
        coverage95 = float(np.mean(covered95))
        bias = float(np.median(validation_error))
        catastrophic = float(np.mean(catastrophic_flags))
        limit = 0.05 if instrument == "chime" else 0.20
        summary = {
            "calibration_count": len(calibration),
            "validation_count": len(validation),
            "median_bias": bias,
            "absolute_median_bias": abs(bias),
            "bias_limit": limit,
            "calibrated_sigma68": sigma68,
            "calibrated_radius95": radius95,
            "validation_coverage68": coverage68,
            "validation_coverage95": coverage95,
            "catastrophic_fraction": catastrophic,
            "edge_count": sum(bool(row["edge_peak"]) for row in validation),
            "calibration_surface_by_snr": calibration_surface,
            "frozen_calibration_inflation": {
                "sigma68": inflation68,
                "radius95": inflation95,
                "derived_from_calibration_seeds": list(range(60)),
            },
            "supported_accuracy_domain": {
                "tau_1ghz_ms": 0.0,
                "components": 1,
                "dsa_exclusion": "snr < 50",
            },
        }
        summary["pass"] = bool(
            abs(bias) <= limit
            and 0.60 <= coverage68 <= 0.76
            and 0.90 <= coverage95 <= 0.98
            and catastrophic < 0.01
            and summary["edge_count"] == 0
        )
        summaries[instrument] = summary
    morphology_offsets = {}
    for instrument in ("chime", "dsa"):
        for tau in (0.0, 0.1, 1.0):
            selected = [
                row
                for row in rows
                if row["instrument"] == instrument
                and row["tau_1ghz_ms"] == tau
                and row["seed"] < 10
            ]
            errors = np.asarray([row["error"] for row in selected], dtype=float)
            morphology_offsets[f"{instrument}/tau_1ghz_ms={tau:g}"] = {
                "count": len(selected),
                "median_dm_struct_minus_propagation": float(np.median(errors)),
                "p68_absolute_offset": float(np.percentile(np.abs(errors), 68)),
            }
    report = {
        "tier": "full_initial_supported_domain",
        "calibration_seeds": list(range(60)),
        "validation_seeds": list(range(60, 100)),
        "rows": rows,
        "summaries": summaries,
        "morphology_offsets": morphology_offsets,
        "pass": all(summary["pass"] for summary in summaries.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return report
