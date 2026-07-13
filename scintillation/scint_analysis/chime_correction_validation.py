"""Fail-closed qualification for corrected CHIME scintillation products."""

from __future__ import annotations

import numpy as np

REQUIRED_CORRECTION_CHECKS = (
    "manifest_verification",
    "off_pulse_null",
    "injection_recovery",
    "low_lag_stability",
    "fit_window_stability",
    "split_time_stability",
    "split_band_stability",
    "comb_residual",
    "kernel_crosscheck",
    "manual_review",
)


def adjudicate_chime_result(checks: dict, *, fitted_dnu_mhz: float | None) -> dict:
    """Return correction and science status; missing checks are inconclusive."""
    failed = [name for name in REQUIRED_CORRECTION_CHECKS if checks.get(name, {}).get("pass") is False]
    pending = [name for name in REQUIRED_CORRECTION_CHECKS if checks.get(name, {}).get("pass") is not True]
    if failed:
        correction_status = "fail"
    elif pending:
        correction_status = "inconclusive"
    else:
        correction_status = "pass"

    valid_width = fitted_dnu_mhz is not None and np.isfinite(fitted_dnu_mhz) and fitted_dnu_mhz > 0
    if correction_status != "pass":
        science_status = "diagnostic_only"
    elif valid_width:
        science_status = "measurement"
    else:
        science_status = "upper_limit"
    return {
        "product_correction_status": correction_status,
        "science_status": science_status,
        "failed_checks": failed,
        "pending_checks": pending,
    }


def combine_science_status(*, artifact_status: str, correction_status: dict) -> str:
    """Correction is an additional gate and can never promote provenance."""
    if artifact_status != "measurement":
        return "diagnostic_only"
    return str(correction_status.get("science_status", "diagnostic_only"))


def injection_recovery_summary(
    injected_mhz: np.ndarray,
    recovered_mhz: np.ndarray,
    lower_mhz: np.ndarray,
    upper_mhz: np.ndarray,
    *,
    channel_width_mhz: float,
    min_trials_for_coverage: int = 20,
    coverage_tolerance: float = 0.15,
) -> dict:
    """Apply the specified width-bias and nominal-68%-coverage gates."""
    injected = np.asarray(injected_mhz, dtype=float)
    recovered = np.asarray(recovered_mhz, dtype=float)
    lower = np.asarray(lower_mhz, dtype=float)
    upper = np.asarray(upper_mhz, dtype=float)
    if not (injected.shape == recovered.shape == lower.shape == upper.shape) or injected.ndim != 1:
        raise ValueError("injection arrays must be matching one-dimensional arrays")
    finite = (
        np.isfinite(injected)
        & np.isfinite(recovered)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (injected > 0)
    )
    injected, recovered, lower, upper = (
        values[finite] for values in (injected, recovered, lower, upper)
    )
    if injected.size == 0:
        return {"pass": False, "n_trials": 0, "reason": "no finite injection trials"}

    absolute_bias = np.abs(recovered - injected)
    allowed_bias = np.maximum(0.1 * injected, 0.25 * float(channel_width_mhz))
    fractional_bias = absolute_bias / injected
    bias_pass = bool(np.all(absolute_bias < allowed_bias))
    covered = (lower <= injected) & (injected <= upper)
    coverage = float(np.mean(covered))
    coverage_pass = bool(
        injected.size >= min_trials_for_coverage and abs(coverage - 0.68) <= coverage_tolerance
    )
    return {
        "pass": bias_pass and coverage_pass,
        "n_trials": int(injected.size),
        "max_fractional_bias": float(np.max(fractional_bias)),
        "coverage_68": coverage,
        "bias_pass": bias_pass,
        "coverage_pass": coverage_pass,
        "bias_limit": "max(10 percent, 0.25 channel)",
    }


def kernel_crosscheck(
    uncorrected_acf: np.ndarray,
    corrected_acf: np.ndarray,
    off_pulse_kernel: np.ndarray,
    *,
    uncertainty: np.ndarray,
    max_abs_z: float = 3.0,
) -> dict:
    """Compare corrected ACF with an independently composed kernel prediction."""
    raw = np.asarray(uncorrected_acf, dtype=float)
    corrected = np.asarray(corrected_acf, dtype=float)
    kernel = np.asarray(off_pulse_kernel, dtype=float)
    sigma = np.asarray(uncertainty, dtype=float)
    if not (raw.shape == corrected.shape == kernel.shape == sigma.shape):
        raise ValueError("ACF and uncertainty arrays must have matching shapes")
    predicted = raw - kernel
    valid = np.isfinite(predicted) & np.isfinite(corrected) & np.isfinite(sigma) & (sigma > 0)
    if not valid.any():
        return {"pass": None, "max_abs_z": None, "reason": "no finite comparison bins"}
    z = (corrected[valid] - predicted[valid]) / sigma[valid]
    observed = float(np.max(np.abs(z)))
    return {
        "pass": bool(observed <= max_abs_z),
        "max_abs_z": observed,
        "n_bins": int(valid.sum()),
        "threshold": float(max_abs_z),
    }
