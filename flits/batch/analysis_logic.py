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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Physical constants and reference frequencies
C_THIN_SCREEN = 1 / (2 * np.pi)  # ≈ 0.159
C_EXTENDED = 1.0
C_RANGE = (0.1, 2.0)  # Acceptable range for τ × Δν product

FREQ_CHIME = 0.6  # Center of 400-800 MHz in GHz
FREQ_DSA = 1.4    # Center of 1.28-1.53 GHz in GHz


@dataclass
class ConsistencyResult:
    """Result from τ-Δν consistency check for a single burst."""
    burst_name: str
    telescope: str
    tau_1ghz_ms: Optional[float] = None
    tau_1ghz_err: Optional[float] = None
    delta_nu_mhz: Optional[float] = None
    delta_nu_err: Optional[float] = None
    tau_delta_nu_product: Optional[float] = None
    tau_delta_nu_product_err: Optional[float] = None
    scint_freq_ghz: Optional[float] = None
    tau_at_scint_freq_ms: Optional[float] = None
    is_consistent: bool = False
    consistency_sigma: Optional[float] = None
    interpretation: str = ""
    quality_flag: str = "unknown"


@dataclass
class FrequencyScalingResult:
    """Result from multi-frequency scaling analysis."""
    burst_name: str
    tau_chime_ms: Optional[float] = None
    tau_chime_err: Optional[float] = None
    delta_nu_chime_mhz: Optional[float] = None
    delta_nu_chime_err: Optional[float] = None
    tau_dsa_ms: Optional[float] = None
    tau_dsa_err: Optional[float] = None
    delta_nu_dsa_mhz: Optional[float] = None
    delta_nu_dsa_err: Optional[float] = None
    alpha_tau: Optional[float] = None
    alpha_tau_err: Optional[float] = None
    alpha_delta_nu: Optional[float] = None
    alpha_delta_nu_err: Optional[float] = None
    alpha_consistent: bool = False
    kolmogorov_consistent: bool = False
    interpretation: str = ""


def check_tau_deltanu_consistency(
    comparison_df: pd.DataFrame,
) -> List[ConsistencyResult]:
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
            
            product = result.tau_at_scint_freq_ms * result.delta_nu_mhz * 1e-3
            result.tau_delta_nu_product = product
            
            # Propagate errors
            if pd.notna(result.tau_1ghz_err) and pd.notna(result.delta_nu_err):
                rel_err_tau = result.tau_1ghz_err / result.tau_1ghz_ms
                rel_err_nu = result.delta_nu_err / result.delta_nu_mhz
                result.tau_delta_nu_product_err = product * np.sqrt(rel_err_tau**2 + rel_err_nu**2)
            
            # --- START PATCH 4: Validate Measurements ---
            # Define helper if not exists (inline here or assume global)
            # Just implement logic directly for simplicity
            rel_err_tau = result.tau_1ghz_err / result.tau_1ghz_ms if result.tau_1ghz_err else 0
            rel_err_nu = result.delta_nu_err / result.delta_nu_mhz if result.delta_nu_err else 0
            
            tau_bad = rel_err_tau > 0.5
            nu_bad = rel_err_nu > 0.5
            
            if tau_bad or nu_bad:
                result.is_consistent = False
                result.quality_flag = "poor_input_quality"
                reasons = []
                if tau_bad: reasons.append(f"τ error > 50% ({rel_err_tau:.2f})")
                if nu_bad: reasons.append(f"Δν error > 50% ({rel_err_nu:.2f})")
                result.interpretation = "Measurements too uncertain: " + ", ".join(reasons)
                results.append(result)
                continue
            # --- END PATCH 4 ---

            
            # Assess consistency
            if C_RANGE[0] <= product <= C_RANGE[1]:
                result.is_consistent = True
                result.quality_flag = "good"
                if abs(product - C_THIN_SCREEN) < abs(product - C_EXTENDED):
                    result.interpretation = f"Consistent with thin screen (C ≈ {C_THIN_SCREEN:.2f})"
                else:
                    result.interpretation = f"Consistent with extended medium (C ≈ {C_EXTENDED:.1f})"
            else:
                result.is_consistent = False
                result.quality_flag = "inconsistent"
                result.interpretation = "τ×Δν outside expected range"

            # Sigma from expected (using geometric mean)
            if result.tau_delta_nu_product_err and result.tau_delta_nu_product_err > 0:
                expected = np.sqrt(C_THIN_SCREEN * C_EXTENDED)
                result.consistency_sigma = abs(product - expected) / result.tau_delta_nu_product_err

        results.append(result)
        
    return results


def analyze_frequency_scaling(
    comparison_df: pd.DataFrame,
) -> List[FrequencyScalingResult]:
    """
    Analyze frequency scaling for co-detected bursts from a comparison DataFrame.
    """
    results = []
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
        
        # τ scaling: This is a placeholder. A robust implementation would
        # re-fit τ vs ν across telescopes. Here, we average the individual α values
        # as a first-order approximation. This is scientifically questionable and
        # should be treated with caution.
        alpha_c = chime_data.get("alpha")
        alpha_d = dsa_data.get("alpha")
        alpha_c_err = chime_data.get("alpha_err", 0)
        alpha_d_err = dsa_data.get("alpha_err", 0)
        
        if pd.notna(alpha_c) and pd.notna(alpha_d):
            weights = [1/alpha_c_err**2, 1/alpha_d_err**2] if alpha_c_err > 0 and alpha_d_err > 0 else [1, 1]
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
                result.alpha_delta_nu_err = np.sqrt(rel_err_c**2 + rel_err_d**2) / abs(log_freq_ratio)

        # --- Assess consistency ---
        interpretations = []
        if result.alpha_tau is not None:
            # Check if consistent with Kolmogorov theory (α=4)
            if 3.5 <= result.alpha_tau <= 4.5:
                result.kolmogorov_consistent = True
                interpretations.append(f"τ-scaling (α_τ={result.alpha_tau:.2f}) consistent with Kolmogorov theory (α=4).")
            else:
                interpretations.append(f"τ-scaling (α_τ={result.alpha_tau:.2f}) deviates from Kolmogorov theory.")

        if result.alpha_delta_nu is not None and result.alpha_tau is not None:
            # Check for self-consistency between the two alpha estimates
            if np.isclose(result.alpha_delta_nu, result.alpha_tau, atol=max(result.alpha_delta_nu_err or 0, result.alpha_tau_err or 0)):
                result.alpha_consistent = True
                interpretations.append("τ and Δν scaling indices are self-consistent.")
            else:
                interpretations.append("τ and Δν scaling indices are not self-consistent.")

        result.interpretation = " ".join(interpretations) if interpretations else "Insufficient data for scaling analysis."
        results.append(result)
        
    return results
