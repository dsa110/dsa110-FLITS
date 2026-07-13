from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DSA_DT_S,
    _freq_grid_mhz,
    _orient_waterfall_to_ascending_frequency,
)

from .cutoff import estimate_fwhm_s, width_derived_cutoffs
from .model import ResolutionEvaluation
from .quality import (
    bootstrap_central_fraction,
    calibrated_injection_sigma,
    eligibility_reasons,
    peak_z_score,
    robust_profile_snr,
)
from .resolution import block_average, resolution_factors, select_resolution
from .search import search_dm
from .shifts import dedisperse_residual
from .uncertainty import channel_bootstrap

PILOT_BURSTS = frozenset({"casey", "mahi", "chromatica"})


def _crop_aligned(waterfall: np.ndarray, dt_s: float) -> tuple[np.ndarray, tuple[int, int]]:
    profile = np.nansum(
        waterfall - np.nanmedian(waterfall, axis=1)[:, None], axis=0
    )
    smooth = max(1, int(round(1e-4 / dt_s)))
    if smooth > 1:
        profile = np.convolve(profile, np.ones(smooth) / smooth, mode="same")
    peak = int(np.nanargmax(profile))
    fwhm = estimate_fwhm_s(waterfall, dt_s)
    half = max(int(round(0.012 / dt_s)), int(round(6.0 * fwhm / dt_s)))
    half = min(half, waterfall.shape[1] // 2)
    start = max(0, peak - half)
    stop = min(waterfall.shape[1], peak + half)
    if stop - start < 32:
        start, stop = 0, waterfall.shape[1]
    return waterfall[:, start:stop], (start, stop)


def _valid_channels(waterfall: np.ndarray) -> np.ndarray:
    finite = np.isfinite(waterfall).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mad = np.nanmedian(
            np.abs(waterfall - np.nanmedian(waterfall, axis=1)[:, None]), axis=1
        )
    return (finite >= 0.9) & np.isfinite(mad) & (mad > 0)


def _evaluate_resolution(
    waterfall: np.ndarray,
    frequency_mhz: np.ndarray,
    dt_s: float,
    reference_dm: float,
    preliminary_residual: float,
    frequency_factor: int,
    time_factor: int,
) -> tuple[ResolutionEvaluation, dict]:
    reduced, reduced_frequency = block_average(
        waterfall, frequency_mhz, frequency_factor, time_factor
    )
    selected_dt = dt_s * time_factor
    fwhm = estimate_fwhm_s(reduced, selected_dt)
    reasons = []
    minimum_channels = 64 if waterfall.shape[0] <= 1024 else 96
    is_dsa = waterfall.shape[0] > 1024
    if reduced.shape[0] < minimum_channels:
        reasons.append("valid_channels")
    if selected_dt > min(5e-4, fwhm / 4.0):
        reasons.append("time_resolution")
    local_grid = np.arange(-0.5, 0.5001, 0.10)
    cutoff_results = []
    cutoffs = width_derived_cutoffs(reduced, selected_dt)
    nyquist = 0.5 / selected_dt
    for low, high in cutoffs:
        high = min(high, 0.95 * nyquist)
        if high <= low:
            continue
        cutoff_results.append(
            search_dm(
                waterfall=reduced,
                frequencies_mhz=reduced_frequency,
                sample_time_s=selected_dt,
                reference_dm=reference_dm + preliminary_residual,
                coarse_grid=local_grid,
                fine_step=0.02,
                f_cut_hz=(low, high),
            )
        )
    if len(cutoff_results) < 3:
        reasons.append("cutoff_support")
        evaluation = ResolutionEvaluation(
            frequency_factor,
            time_factor,
            reduced.shape,
            None,
            None,
            robust_profile_snr(reduced),
            float("nan"),
            0.0,
            True,
            False,
            False,
            tuple(reasons),
        )
        return evaluation, {"cutoffs": [list(value) for value in cutoffs]}
    central = cutoff_results[1]
    residuals = np.asarray(
        [preliminary_residual + result.residual_dm for result in cutoff_results]
    )
    sigma = max(0.01, float(np.std(residuals, ddof=1)))
    cutoff_stable = bool(np.max(np.abs(residuals - residuals[1])) <= max(0.10, sigma))
    snr = robust_profile_snr(reduced)
    peak_z = peak_z_score(central.fine.score)
    reasons.extend(
        eligibility_reasons(
            profile_snr=snr,
            coherence_peak_z=peak_z,
            edge_peak=central.edge_peak,
            bootstrap_success_fraction=1.0,
            cutoff_stable=cutoff_stable,
            minimum_profile_snr=50.0 if is_dsa else 8.0,
        )
    )
    evaluation = ResolutionEvaluation(
        frequency_factor=frequency_factor,
        time_factor=time_factor,
        shape=reduced.shape,
        residual_dm=float(residuals[1]),
        sigma=sigma,
        profile_snr=snr,
        coherence_peak_z=peak_z,
        bootstrap_success_fraction=1.0,
        edge_peak=central.edge_peak,
        cutoff_stable=cutoff_stable,
        eligible=not reasons,
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )
    detail = {
        "cutoffs": [list(value) for value in cutoffs],
        "cutoff_residual_dms": residuals.tolist(),
        "fine_grid": central.fine.residual_dm_grid.tolist(),
        "fine_score": central.fine.score.tolist(),
        "fwhm_s": fwhm,
        "selected_dt_s": selected_dt,
    }
    return evaluation, detail


def measure_manifest_row(
    row: dict[str, str],
    data_dir: Path,
    *,
    input_sha256: str | None = None,
    n_bootstrap: int = 12,
) -> dict:
    telescope = row["telescope"]
    path = data_dir / row["filename"]
    raw = np.load(path, mmap_mode="r")
    waterfall = _orient_waterfall_to_ascending_frequency(raw, telescope)
    frequency = _freq_grid_mhz(telescope, waterfall.shape[0])
    valid = _valid_channels(waterfall)
    waterfall, frequency = waterfall[valid], frequency[valid]
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    coarse_ff = 16 if telescope == "chime" else 64
    coarse_tf = 256 if telescope == "chime" else 16
    coarse_waterfall, coarse_frequency = block_average(
        waterfall, frequency, coarse_ff, coarse_tf
    )
    window = 4.0 if telescope == "chime" else 5.0
    coarse_grid = np.arange(-window, window + 0.125, 0.25)
    coarse_cutoff = width_derived_cutoffs(coarse_waterfall, dt_s * coarse_tf)[1]
    coarse_cutoff = (coarse_cutoff[0], min(coarse_cutoff[1], 0.95 * 0.5 / (dt_s * coarse_tf)))
    preliminary = search_dm(
        waterfall=coarse_waterfall,
        frequencies_mhz=coarse_frequency,
        sample_time_s=dt_s * coarse_tf,
        reference_dm=float(row["dm_pc_cm3"]),
        coarse_grid=coarse_grid,
        fine_step=0.05,
        f_cut_hz=coarse_cutoff,
    )
    aligned = dedisperse_residual(
        np.asarray(waterfall, dtype=float),
        frequency,
        dt_s,
        preliminary.residual_dm,
    )
    cropped, crop = _crop_aligned(aligned, dt_s)
    evaluations = []
    details = []
    selected = None
    for frequency_factor, time_factor in resolution_factors(telescope):
        evaluation, detail = _evaluate_resolution(
            cropped,
            frequency,
            dt_s,
            float(row["dm_pc_cm3"]),
            preliminary.residual_dm,
            frequency_factor,
            time_factor,
        )
        evaluations.append(evaluation)
        details.append(detail)
        selected = select_resolution(evaluations)
        if selected is not None and len([item for item in evaluations if item.eligible]) >= 2:
            break
    bootstrap = None
    central_fraction = None
    if selected is not None:
        selected_waterfall, selected_frequency = block_average(
            cropped,
            frequency,
            selected.frequency_factor,
            selected.time_factor,
        )
        bootstrap = channel_bootstrap(
            waterfall=selected_waterfall,
            frequencies_mhz=selected_frequency,
            sample_time_s=dt_s * selected.time_factor,
            reference_dm=float(row["dm_pc_cm3"]) + preliminary.residual_dm,
            coarse_grid=np.arange(-0.5, 0.5001, 0.10),
            fine_step=0.02,
            f_cut_hz=width_derived_cutoffs(
                selected_waterfall, dt_s * selected.time_factor
            )[1],
            n_bootstrap=n_bootstrap,
            random_seed=sum(ord(char) for char in f"{row['burst']}:{telescope}"),
        )
        central_fraction = bootstrap_central_fraction(
            bootstrap.peaks,
            selected.residual_dm - preliminary.residual_dm,
            bootstrap.sigma,
        )
        bootstrap_ok = bool(
            bootstrap.success_fraction >= 0.90
            and np.isfinite(bootstrap.sigma)
            and bootstrap.sigma > 0
            and central_fraction >= 0.90
        )
        selected = replace(
            selected,
            sigma=max(selected.sigma or 0.0, bootstrap.sigma)
            if np.isfinite(bootstrap.sigma)
            else selected.sigma,
            bootstrap_success_fraction=bootstrap.success_fraction,
            eligible=bool(selected.eligible and bootstrap_ok),
            failure_reasons=(
                selected.failure_reasons
                if bootstrap_ok
                else tuple(
                    (*selected.failure_reasons, "bootstrap_multimodal")
                    if central_fraction < 0.90
                    else (*selected.failure_reasons, "bootstrap_success")
                )
            ),
        )
    status = "PASS" if selected is not None and selected.eligible else "UNCONSTRAINED"
    injection_sigma = (
        None
        if status != "PASS"
        else calibrated_injection_sigma(telescope, selected.profile_snr)
    )
    method_sigma = None if status != "PASS" else selected.sigma
    total_sigma = (
        None
        if method_sigma is None or injection_sigma is None
        else float(np.hypot(method_sigma, injection_sigma))
    )
    return {
        "burst": row["burst"],
        "telescope": telescope,
        "status": status,
        "input_path": str(path),
        "input_sha256": input_sha256,
        "dm_reference": float(row["dm_pc_cm3"]),
        "preliminary_residual_dm": preliminary.residual_dm,
        "dm_residual": None if status != "PASS" else selected.residual_dm,
        "dm_absolute": (
            None
            if status != "PASS"
            else float(row["dm_pc_cm3"]) + selected.residual_dm
        ),
        "sigma_method": method_sigma,
        "sigma_injection": injection_sigma,
        "sigma_total": total_sigma,
        "native_shape": list(raw.shape),
        "valid_channels": int(valid.sum()),
        "crop": list(crop),
        "selected": None if selected is None else asdict(selected),
        "bootstrap_peaks": None if bootstrap is None else bootstrap.peaks.tolist(),
        "bootstrap_central_fraction": central_fraction,
        "resolutions": [asdict(item) for item in evaluations],
        "resolution_details": details,
    }


def run_measurements(config_path: Path, *, pilot: bool) -> dict:
    config = yaml.safe_load(config_path.read_text())
    data_dir = Path(config["data_dir"]).expanduser()
    rows = list(csv.DictReader(Path(config["manifest"]).open()))
    if pilot:
        rows = [row for row in rows if row["burst"] in PILOT_BURSTS]
    output_dir = Path(config["output_dir"]) / "products"
    input_records_path = Path(config["output_dir"]) / "inputs.json"
    input_records = json.loads(input_records_path.read_text()) if input_records_path.exists() else []
    input_hashes = {Path(record["path"]).name: record.get("sha256") for record in input_records}
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    results = []
    for row in rows:
        result = measure_manifest_row(
            row,
            data_dir,
            input_sha256=input_hashes.get(row["filename"]),
        )
        result["code_commit"] = code_commit
        result["config_sha256"] = config_sha256
        product_dir = output_dir / row["burst"] / row["telescope"]
        product_dir.mkdir(parents=True, exist_ok=True)
        temporary = product_dir / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(product_dir / "result.json")
        results.append(result)
    summary = {
        "pilot": pilot,
        "count": len(results),
        "pass": sum(result["status"] == "PASS" for result in results),
        "unconstrained": sum(result["status"] == "UNCONSTRAINED" for result in results),
    }
    return summary
