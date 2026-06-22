#
# Copyright 2024, by the California Institute of Technology.
# ALL RIGHTS RESERVED.
# United States Government sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# This software may be subject to U.S. export control laws and regulations.
# By accepting this document, the user agrees to comply with all applicable
# U.S. export laws and regulations. User has the responsibility to obtain
# export licenses, or other export authority as may be required before
# exporting such information to foreign countries or providing access to
# foreign persons.
"""
Core scientific calculations for joint analysis of scattering and scintillation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..fitting import VALIDATION_THRESHOLDS as VT

# Physical constants and reference frequencies
C_THIN_SCREEN = 1 / (2 * np.pi)  # ≈ 0.159
C_EXTENDED = 1.0
C_RANGE = (0.1, 2.0)  # Acceptable range for τ × Δν product

# Cordes & Rickett thin-Kolmogorov screen constant in 2π·Δν_d·τ = C1
# (scintillation/scint_analysis/physics.py:37). Turns a resolved Δν_d into the
# τ it would imply for that *same* screen.
C1_THIN_KOLMOGOROV = 1.16

FREQ_CHIME = 0.6  # Center of 400-800 MHz in GHz
FREQ_DSA = 1.4  # Center of 1.28-1.53 GHz in GHz


@dataclass
class ConsistencyResult:
    """Result from τ-Δν consistency check for a single burst."""

    burst_name: str
    telescope: str
    tau_1ghz_ms: float | None = None
    tau_1ghz_err: float | None = None
    delta_nu_mhz: float | None = None
    delta_nu_err: float | None = None
    tau_delta_nu_product: float | None = None
    tau_delta_nu_product_err: float | None = None
    scint_freq_ghz: float | None = None
    tau_at_scint_freq_ms: float | None = None
    is_consistent: bool = False
    consistency_sigma: float | None = None
    interpretation: str = ""
    quality_flag: str = "unknown"
    implied_tau_from_dnu_ms: float | None = None
    screen_verdict: str = "unknown"  # same_screen | different_screens | indeterminate


@dataclass
class FrequencyScalingResult:
    """Result from multi-frequency scaling analysis."""

    burst_name: str
    tau_chime_ms: float | None = None
    tau_chime_err: float | None = None
    delta_nu_chime_mhz: float | None = None
    delta_nu_chime_err: float | None = None
    tau_dsa_ms: float | None = None
    tau_dsa_err: float | None = None
    delta_nu_dsa_mhz: float | None = None
    delta_nu_dsa_err: float | None = None
    alpha_tau: float | None = None
    alpha_tau_err: float | None = None
    alpha_delta_nu: float | None = None
    alpha_delta_nu_err: float | None = None
    alpha_consistent: bool = False
    kolmogorov_consistent: bool = False
    interpretation: str = ""


def _validate_measurement(value, error, param_name="parameter", rel_err_threshold=0.5):
    """Validate a single measurement.

    Returns
    -------
    is_good : bool
    reason : str
    """
    if value is None or error is None:
        return False, f"{param_name} not measured"

    if value <= 0:
        return False, f"{param_name} <= 0 (unphysical)"

    rel_err = error / value if value != 0 else float("inf")
    if rel_err > 1.0:
        return False, f"{param_name} unconstrained (σ/value={rel_err:.2f})"
    elif rel_err > rel_err_threshold:
        return False, f"{param_name} poorly constrained (σ/value={rel_err:.2f})"
    else:
        return True, f"{param_name} well-constrained (σ/value={rel_err:.2f})"


def check_tau_deltanu_consistency(
    comparison_df: pd.DataFrame,
) -> list[ConsistencyResult]:
    """
    Check consistency between τ and Δν_dc from a comparison DataFrame.
    The product τ × Δν_dc should be approximately constant (0.1-1).
    """
    results = []
    for _, row in comparison_df.iterrows():
        burst_name = row["burst_name"]
        tel = row["telescope"]

        result = ConsistencyResult(burst_name=burst_name, telescope=tel)

        # Extract measurements from DataFrame
        result.tau_1ghz_ms = row.get("tau_1ghz")
        result.tau_1ghz_err = row.get("tau_1ghz_err")
        result.delta_nu_mhz = row.get("delta_nu_dc")
        result.delta_nu_err = row.get("delta_nu_dc_err")
        alpha = row.get("alpha", 4.0)  # Default to Kolmogorov
        if pd.isna(alpha):
            alpha = 4.0

        if tel == "chime":
            result.scint_freq_ghz = FREQ_CHIME
        elif tel == "dsa":
            result.scint_freq_ghz = FREQ_DSA

        # Compute product if both measurements available
        if (
            pd.notna(result.tau_1ghz_ms)
            and pd.notna(result.delta_nu_mhz)
            and result.scint_freq_ghz is not None
        ):
            freq_ratio = result.scint_freq_ghz / 1.0  # ν / 1 GHz
            result.tau_at_scint_freq_ms = result.tau_1ghz_ms * (freq_ratio ** (-alpha))

            # τ×Δν in SI: τ[ms]·Δν[MHz]·1e3 = τ[s]·Δν[Hz]. A single thin screen
            # gives ≈ C_THIN_SCREEN (1/2π ≈ 0.159); the prior ·1e-3 was 1e6 too small.
            product = result.tau_at_scint_freq_ms * result.delta_nu_mhz * 1e3
            result.tau_delta_nu_product = product

            # τ the resolved Δν_d would imply for its OWN screen (τ_ms = C1/(2π·Δν_MHz·1e3))
            result.implied_tau_from_dnu_ms = C1_THIN_KOLMOGOROV / (
                2 * np.pi * result.delta_nu_mhz * 1e3
            )

            # Propagate errors
            if pd.notna(result.tau_1ghz_err) and pd.notna(result.delta_nu_err):
                rel_err_tau = result.tau_1ghz_err / result.tau_1ghz_ms
                rel_err_nu = result.delta_nu_err / result.delta_nu_mhz
                result.tau_delta_nu_product_err = product * np.sqrt(rel_err_tau**2 + rel_err_nu**2)

            # Validate input measurements
            tau_valid, tau_msg = _validate_measurement(
                result.tau_1ghz_ms,
                result.tau_1ghz_err,
                "τ",
                rel_err_threshold=VT.PARAM_UNCERTAINTY_ACCEPTABLE_MAX,
            )
            nu_valid, nu_msg = _validate_measurement(
                result.delta_nu_mhz,
                result.delta_nu_err,
                "Δν",
                rel_err_threshold=VT.PARAM_UNCERTAINTY_ACCEPTABLE_MAX,
            )

            if not tau_valid or not nu_valid:
                result.is_consistent = False
                result.quality_flag = "poor_input_quality"
                reasons = []
                if not tau_valid:
                    reasons.append(tau_msg)
                if not nu_valid:
                    reasons.append(nu_msg)
                result.interpretation = "Measurements too uncertain: " + ", ".join(reasons)
                results.append(result)
                continue

            # Screen verdict from the single τ·Δν statistic (same screen ⟹
            # 2π·τ·Δν = C1 ⟹ product ≈ C_THIN_SCREEN…C_EXTENDED). A product ≫ range
            # means the resolved Δν_d is too large for the fitted pulse-broadening
            # τ — the scintillation samples a NEARER screen than the scattering one
            # (the wilhelm two-screen case). implied_tau_from_dnu_ms reports that
            # near screen's τ for the reader; it is the SAME statistic re-expressed,
            # not an independent probe.
            if C_RANGE[0] <= product <= C_RANGE[1]:
                result.is_consistent = True
                result.quality_flag = "good"
                result.screen_verdict = "same_screen"
                if abs(product - C_THIN_SCREEN) < abs(product - C_EXTENDED):
                    result.interpretation = f"Consistent with thin screen (C ≈ {C_THIN_SCREEN:.2f})"
                else:
                    result.interpretation = (
                        f"Consistent with extended medium (C ≈ {C_EXTENDED:.1f})"
                    )
            elif product > C_RANGE[1]:
                result.is_consistent = False
                result.quality_flag = "different_screens"
                result.screen_verdict = "different_screens"
                result.interpretation = (
                    f"Resolved Δν_d implies a near-screen τ≈{result.implied_tau_from_dnu_ms:.2e} ms "
                    f"≪ fitted pulse-broadening τ={result.tau_at_scint_freq_ms:.3f} ms "
                    "→ different screens (resolved near vs far scattering screen)"
                )
            else:
                result.is_consistent = False
                result.quality_flag = "inconsistent"
                result.screen_verdict = "indeterminate"
                # Opposite case to different_screens: here implied τ > fitted τ, i.e.
                # the resolved Δν_d would be a farther/stronger screen, or it is spurious.
                result.interpretation = (
                    "τ×Δν below expected range — resolved Δν_d implies τ larger than the "
                    "fitted scattering τ (farther/spurious screen); under-constrained"
                )

            # Sigma from expected (using geometric mean)
            if result.tau_delta_nu_product_err and result.tau_delta_nu_product_err > 0:
                expected = np.sqrt(C_THIN_SCREEN * C_EXTENDED)
                result.consistency_sigma = abs(product - expected) / result.tau_delta_nu_product_err

        results.append(result)

    return results


def analyze_frequency_scaling(
    comparison_df: pd.DataFrame,
) -> list[FrequencyScalingResult]:
    """
    Analyze frequency scaling for co-detected bursts from a comparison DataFrame.
    """
    results = []
    if comparison_df.empty:  # column-less frame (e.g. empty joint-fit dir) → no-op
        return results
    # Group by burst and check for co-detections
    for burst_name, group in comparison_df.groupby("burst_name"):
        if len(group["telescope"].unique()) < 2:
            continue  # Skip bursts not seen by multiple telescopes

        result = FrequencyScalingResult(burst_name=burst_name)

        chime_data = group[group["telescope"] == "chime"].iloc[0]
        dsa_data = group[group["telescope"] == "dsa"].iloc[0]

        # Populate scattering results
        result.tau_chime_ms = chime_data.get("tau_1ghz")
        result.tau_chime_err = chime_data.get("tau_1ghz_err")
        result.tau_dsa_ms = dsa_data.get("tau_1ghz")
        result.tau_dsa_err = dsa_data.get("tau_1ghz_err")

        # Populate scintillation results
        result.delta_nu_chime_mhz = chime_data.get("delta_nu_dc")
        result.delta_nu_chime_err = chime_data.get("delta_nu_dc_err")
        result.delta_nu_dsa_mhz = dsa_data.get("delta_nu_dc")
        result.delta_nu_dsa_err = dsa_data.get("delta_nu_dc_err")

        # --- Compute scaling indices ---

        # τ scaling index α_τ: the joint CHIME+DSA fit measures ONE shared α
        # across the band lever (burstfit_joint.py), so use it directly with its
        # own error. Averaging two copies of it (the old placeholder) fabricated
        # a √2 error shrink; that path survives only as a no-joint-fit fallback.
        alpha_joint = chime_data.get("alpha_joint")
        alpha_joint_err = chime_data.get("alpha_joint_err")
        if pd.notna(alpha_joint):
            result.alpha_tau = float(alpha_joint)
            result.alpha_tau_err = float(alpha_joint_err) if pd.notna(alpha_joint_err) else None
        elif pd.notna(chime_data.get("alpha")) and pd.notna(dsa_data.get("alpha")):
            # ponytail: fallback — per-telescope single-band α average (each
            # often fixed at 4). Scientifically weak; flagged, not load-bearing.
            alpha_c = chime_data.get("alpha")
            alpha_d = dsa_data.get("alpha")
            alpha_c_err = chime_data.get("alpha_err", 0)
            alpha_d_err = dsa_data.get("alpha_err", 0)
            weights = (
                [1 / alpha_c_err**2, 1 / alpha_d_err**2]
                if alpha_c_err > 0 and alpha_d_err > 0
                else [1, 1]
            )
            result.alpha_tau = np.average([alpha_c, alpha_d], weights=weights)
            result.alpha_tau_err = 1 / np.sqrt(np.sum(weights))

        # Δν scaling: Directly compute from the two data points
        if pd.notna(result.delta_nu_chime_mhz) and pd.notna(result.delta_nu_dsa_mhz):
            log_ratio_nu = np.log(result.delta_nu_dsa_mhz / result.delta_nu_chime_mhz)
            log_freq_ratio = np.log(FREQ_DSA / FREQ_CHIME)
            result.alpha_delta_nu = log_ratio_nu / log_freq_ratio

            # Error propagation
            if pd.notna(result.delta_nu_chime_err) and pd.notna(result.delta_nu_dsa_err):
                rel_err_c = result.delta_nu_chime_err / result.delta_nu_chime_mhz
                rel_err_d = result.delta_nu_dsa_err / result.delta_nu_dsa_mhz
                result.alpha_delta_nu_err = np.sqrt(rel_err_c**2 + rel_err_d**2) / abs(
                    log_freq_ratio
                )

        # --- Assess consistency ---
        interpretations = []
        if result.alpha_tau is not None:
            # Check if consistent with Kolmogorov theory (α=4)
            if 3.5 <= result.alpha_tau <= 4.5:
                result.kolmogorov_consistent = True
                interpretations.append(
                    f"τ-scaling (α_τ={result.alpha_tau:.2f}) consistent with Kolmogorov theory (α=4)."
                )
            else:
                interpretations.append(
                    f"τ-scaling (α_τ={result.alpha_tau:.2f}) deviates from Kolmogorov theory."
                )

        if result.alpha_delta_nu is not None and result.alpha_tau is not None:
            # Check for self-consistency between the two alpha estimates
            if np.isclose(
                result.alpha_delta_nu,
                result.alpha_tau,
                atol=max(result.alpha_delta_nu_err or 0, result.alpha_tau_err or 0),
            ):
                result.alpha_consistent = True
                interpretations.append("τ and Δν scaling indices are self-consistent.")
            else:
                interpretations.append("τ and Δν scaling indices are not self-consistent.")

        result.interpretation = (
            " ".join(interpretations)
            if interpretations
            else "Insufficient data for scaling analysis."
        )
        results.append(result)

    return results


def build_comparison_df_from_joint_fits(json_dir, scint_dnu: dict | None = None) -> pd.DataFrame:
    """Assemble a comparison DataFrame from joint-fit JSONs.

    Each ``*_joint_fit.json`` (one per sightline; produced by
    ``scattering/scat_analysis/burstfit_joint.py``) carries a *shared*
    ``tau_1ghz`` and ``alpha`` measured across the CHIME↔DSA band lever. This
    emits two rows per source (chime, dsa) carrying those shared joint values in
    both ``alpha`` and the explicit ``alpha_joint`` column, so
    :func:`analyze_frequency_scaling` uses the real joint α (not the placeholder
    per-telescope average) and :func:`check_tau_deltanu_consistency` can compare
    it against a per-telescope resolved Δν_d.

    Parameters
    ----------
    json_dir : path-like
        Directory of ``*_joint_fit.json`` files.
    scint_dnu : dict, optional
        ``{burst: {telescope: (delta_nu_dc_mhz, delta_nu_dc_err)}}`` resolved
        scintillation bandwidths to merge in (only sources with a measured ACF;
        the far pulse-broadening screen's Δν_d is sub-channel and stays NaN).
    """
    json_dir = Path(json_dir)
    rows = []
    for jf in sorted(json_dir.glob("*_joint_fit.json")):
        d = json.loads(jf.read_text())
        burst = d.get("burst", jf.name.split("_joint_fit")[0])
        tau, alpha = d["tau_1ghz"], d["alpha"]
        # ponytail: collapse the asymmetric ±err into one symmetric σ
        tau_err = (tau["err_plus"] + tau["err_minus"]) / 2
        a_med = alpha["median"]
        a_err = (alpha["err_plus"] + alpha["err_minus"]) / 2
        for tel in ("chime", "dsa"):
            dnu = dnu_err = np.nan
            if scint_dnu and burst in scint_dnu and tel in scint_dnu[burst]:
                dnu, dnu_err = scint_dnu[burst][tel]
            rows.append(
                {
                    "burst_name": burst,
                    "telescope": tel,
                    "tau_1ghz": tau["median"],
                    "tau_1ghz_err": tau_err,
                    "alpha": a_med,
                    "alpha_err": a_err,
                    "alpha_joint": a_med,
                    "alpha_joint_err": a_err,
                    "delta_nu_dc": dnu,
                    "delta_nu_dc_err": dnu_err,
                }
            )
    return pd.DataFrame(rows)
