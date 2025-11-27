"""
joint_analysis.py
=================

Cross-validate scattering and scintillation measurements using physical
relationships between propagation parameters.

Key Physics:
- Scattering time τ and scintillation bandwidth Δν_dc are related via:
  τ × Δν_dc ≈ C  (where C ~ 0.15–1 depending on screen geometry)
  
- For a thin screen: C ≈ 1/(2π) ≈ 0.16
- For extended medium: C ≈ 1

- Both should scale with frequency:
  τ ∝ ν^{-α}  (typically α ≈ 4 for Kolmogorov)
  Δν_dc ∝ ν^{+α}
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .results_db import ResultsDatabase, ScatteringResult, ScintillationResult

log = logging.getLogger(__name__)


# Physical constants
C_THIN_SCREEN = 1 / (2 * np.pi)  # ≈ 0.159
C_EXTENDED = 1.0
C_RANGE = (0.1, 2.0)  # Acceptable range for τ × Δν product


@dataclass
class ConsistencyResult:
    """Result from τ-Δν consistency check for a single burst."""
    burst_name: str
    telescope: str
    
    # Measured values
    tau_1ghz_ms: Optional[float] = None
    tau_1ghz_err: Optional[float] = None
    delta_nu_mhz: Optional[float] = None
    delta_nu_err: Optional[float] = None
    
    # Derived quantity
    tau_delta_nu_product: Optional[float] = None
    tau_delta_nu_product_err: Optional[float] = None
    
    # Frequency at which scintillation was measured
    scint_freq_ghz: Optional[float] = None
    
    # τ scaled to scintillation frequency (for direct comparison)
    tau_at_scint_freq_ms: Optional[float] = None
    
    # Consistency assessment
    is_consistent: bool = False
    consistency_sigma: Optional[float] = None  # How many σ from expected?
    interpretation: str = ""
    quality_flag: str = "unknown"


@dataclass
class FrequencyScalingResult:
    """Result from multi-frequency scaling analysis."""
    burst_name: str
    
    # CHIME results (400-800 MHz)
    tau_chime_ms: Optional[float] = None
    tau_chime_err: Optional[float] = None
    delta_nu_chime_mhz: Optional[float] = None
    delta_nu_chime_err: Optional[float] = None
    
    # DSA results (1.28-1.53 GHz)
    tau_dsa_ms: Optional[float] = None
    tau_dsa_err: Optional[float] = None
    delta_nu_dsa_mhz: Optional[float] = None
    delta_nu_dsa_err: Optional[float] = None
    
    # Scaling analysis
    alpha_tau: Optional[float] = None  # τ ∝ ν^{-α}
    alpha_tau_err: Optional[float] = None
    alpha_delta_nu: Optional[float] = None  # Δν ∝ ν^{+α}
    alpha_delta_nu_err: Optional[float] = None
    
    # Consistency
    alpha_consistent: bool = False  # Are α_τ and α_Δν compatible?
    kolmogorov_consistent: bool = False  # Is α ≈ 4?
    interpretation: str = ""


class JointAnalysis:
    """
    Perform joint analysis of scattering and scintillation measurements.
    
    Validates physical consistency between:
    1. τ × Δν_dc product (should be ~ 0.1-1)
    2. Frequency scaling (τ ∝ ν^{-α}, Δν ∝ ν^{+α})
    3. Cross-telescope comparison for co-detected bursts
    """
    
    # Reference frequencies (GHz)
    FREQ_CHIME = 0.6  # Center of 400-800 MHz
    FREQ_DSA = 1.4    # Center of 1.28-1.53 GHz
    
    def __init__(self, db: ResultsDatabase):
        """
        Initialize joint analysis.
        
        Args:
            db: Results database with scattering and scintillation measurements
        """
        self.db = db
        self.consistency_results: List[ConsistencyResult] = []
        self.scaling_results: List[FrequencyScalingResult] = []
        
    def check_tau_deltanu_consistency(
        self,
        burst_name: Optional[str] = None,
        telescope: Optional[str] = None,
    ) -> List[ConsistencyResult]:
        """
        Check consistency between τ and Δν_dc for each burst.
        
        The product τ × Δν_dc should be approximately constant (0.1-1)
        regardless of frequency, with the exact value depending on
        screen geometry.
        
        Args:
            burst_name: Optional filter by burst name
            telescope: Optional filter by telescope
            
        Returns:
            List of ConsistencyResult objects
        """
        scat_results = self.db.get_scattering_results(burst_name, telescope)
        scint_results = self.db.get_scintillation_results(burst_name, telescope)
        
        # Index by (burst_name, telescope)
        scat_dict = {(r.burst_name, r.telescope): r for r in scat_results}
        scint_dict = {(r.burst_name, r.telescope): r for r in scint_results}
        
        results = []
        
        for key in set(scat_dict.keys()) | set(scint_dict.keys()):
            burst, tel = key
            scat = scat_dict.get(key)
            scint = scint_dict.get(key)
            
            result = ConsistencyResult(burst_name=burst, telescope=tel)
            
            if scat and scat.tau_1ghz is not None:
                result.tau_1ghz_ms = scat.tau_1ghz
                result.tau_1ghz_err = scat.tau_1ghz_err
                
            if scint and scint.delta_nu_dc is not None:
                result.delta_nu_mhz = scint.delta_nu_dc
                result.delta_nu_err = scint.delta_nu_dc_err
                
                # Determine scintillation frequency
                if tel == "chime":
                    result.scint_freq_ghz = self.FREQ_CHIME
                elif tel == "dsa":
                    result.scint_freq_ghz = self.FREQ_DSA
                else:
                    result.scint_freq_ghz = 1.0  # Default
            
            # Compute product if both measurements available
            if result.tau_1ghz_ms is not None and result.delta_nu_mhz is not None:
                # Scale τ from 1 GHz to scintillation frequency
                alpha = scat.alpha if (scat and scat.alpha) else 4.0  # Default Kolmogorov
                freq_ratio = result.scint_freq_ghz / 1.0  # ν / 1 GHz
                result.tau_at_scint_freq_ms = result.tau_1ghz_ms * (freq_ratio ** (-alpha))
                
                # Product: τ (ms) × Δν (MHz) = τ (s) × Δν (Hz) × 10^{-3} × 10^{6} = τ × Δν × 10^3
                # But we want dimensionless: τ (s) × Δν (Hz)
                # τ_ms × Δν_MHz = τ_s × 10^3 × Δν_Hz × 10^{-6} = τ_s × Δν_Hz × 10^{-3}
                # So product_dimensionless = τ_ms × Δν_MHz × 10^{-3}
                product = result.tau_at_scint_freq_ms * result.delta_nu_mhz * 1e-3
                result.tau_delta_nu_product = product
                
                # Propagate errors
                if result.tau_1ghz_err and result.delta_nu_err:
                    rel_err_tau = result.tau_1ghz_err / result.tau_1ghz_ms
                    rel_err_nu = result.delta_nu_err / result.delta_nu_mhz
                    result.tau_delta_nu_product_err = product * np.sqrt(rel_err_tau**2 + rel_err_nu**2)
                
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
                    
                    if product < C_RANGE[0]:
                        result.interpretation = "τ×Δν too small: possible multiple screens or measurement bias"
                    else:
                        result.interpretation = "τ×Δν too large: possible unresolved scintillation or overestimated τ"
                
                # Sigma from expected (using geometric mean of thin/extended)
                expected = np.sqrt(C_THIN_SCREEN * C_EXTENDED)
                if result.tau_delta_nu_product_err and result.tau_delta_nu_product_err > 0:
                    result.consistency_sigma = abs(product - expected) / result.tau_delta_nu_product_err
            
            results.append(result)
            
        self.consistency_results = results
        return results
    
    def analyze_frequency_scaling(self) -> List[FrequencyScalingResult]:
        """
        Analyze frequency scaling for co-detected bursts (CHIME + DSA).
        
        For bursts detected by both telescopes, compute:
        - τ scaling index (should be ≈ -4 for Kolmogorov)
        - Δν scaling index (should be ≈ +4 for Kolmogorov)
        
        Returns:
            List of FrequencyScalingResult objects
        """
        scat_chime = {r.burst_name: r for r in self.db.get_scattering_results(telescope="chime")}
        scat_dsa = {r.burst_name: r for r in self.db.get_scattering_results(telescope="dsa")}
        scint_chime = {r.burst_name: r for r in self.db.get_scintillation_results(telescope="chime")}
        scint_dsa = {r.burst_name: r for r in self.db.get_scintillation_results(telescope="dsa")}
        
        # Find co-detected bursts
        all_bursts = set(scat_chime.keys()) | set(scat_dsa.keys()) | set(scint_chime.keys()) | set(scint_dsa.keys())
        co_detected = [b for b in all_bursts if b in scat_chime and b in scat_dsa]
        
        results = []
        
        for burst_name in co_detected:
            result = FrequencyScalingResult(burst_name=burst_name)
            
            # Scattering measurements
            sc_c, sc_d = scat_chime.get(burst_name), scat_dsa.get(burst_name)
            if sc_c:
                result.tau_chime_ms = sc_c.tau_1ghz
                result.tau_chime_err = sc_c.tau_1ghz_err
            if sc_d:
                result.tau_dsa_ms = sc_d.tau_1ghz
                result.tau_dsa_err = sc_d.tau_1ghz_err
            
            # Scintillation measurements
            si_c, si_d = scint_chime.get(burst_name), scint_dsa.get(burst_name)
            if si_c:
                result.delta_nu_chime_mhz = si_c.delta_nu_dc
                result.delta_nu_chime_err = si_c.delta_nu_dc_err
            if si_d:
                result.delta_nu_dsa_mhz = si_d.delta_nu_dc
                result.delta_nu_dsa_err = si_d.delta_nu_dc_err
            
            # Compute scaling indices
            # τ scaling: log(τ_C/τ_D) = -α × log(ν_C/ν_D)
            if result.tau_chime_ms and result.tau_dsa_ms:
                # Both are at 1 GHz reference, so we need to use the MEASURED values
                # Actually tau_1ghz is referenced to 1 GHz, so the ratio should be 1 if same screen
                # We need the actual measured τ at each frequency
                # For now, assume tau_1ghz is consistent and use α from individual fits
                if sc_c and sc_c.alpha and sc_d and sc_d.alpha:
                    result.alpha_tau = (sc_c.alpha + sc_d.alpha) / 2
                    if sc_c.alpha_err and sc_d.alpha_err:
                        result.alpha_tau_err = np.sqrt(sc_c.alpha_err**2 + sc_d.alpha_err**2) / 2
            
            # Δν scaling
            if result.delta_nu_chime_mhz and result.delta_nu_dsa_mhz:
                log_ratio_nu = np.log(result.delta_nu_dsa_mhz / result.delta_nu_chime_mhz)
                log_freq_ratio = np.log(self.FREQ_DSA / self.FREQ_CHIME)
                result.alpha_delta_nu = log_ratio_nu / log_freq_ratio
                
                # Error propagation
                if result.delta_nu_chime_err and result.delta_nu_dsa_err:
                    rel_err_c = result.delta_nu_chime_err / result.delta_nu_chime_mhz
                    rel_err_d = result.delta_nu_dsa_err / result.delta_nu_dsa_mhz
                    result.alpha_delta_nu_err = np.sqrt(rel_err_c**2 + rel_err_d**2) / abs(log_freq_ratio)
            
            # Assess consistency
            interpretations = []
            
            if result.alpha_tau is not None:
                if 3.5 <= result.alpha_tau <= 4.5:
                    result.kolmogorov_consistent = True
                    interpretations.append(f"τ scaling (α={result.alpha_tau:.1f}) consistent with Kolmogorov")
                else:
                    interpretations.append(f"τ scaling (α={result.alpha_tau:.1f}) deviates from Kolmogorov (α=4)")
            
            if result.alpha_delta_nu is not None:
                if result.alpha_tau is not None:
                    diff = abs(result.alpha_delta_nu - result.alpha_tau)
                    if diff < 1.0:  # Within reasonable agreement
                        result.alpha_consistent = True
                        interpretations.append("τ and Δν scaling are consistent")
                    else:
                        interpretations.append(f"τ and Δν scaling disagree by {diff:.1f}")
            
            result.interpretation = "; ".join(interpretations) if interpretations else "Insufficient data"
            results.append(result)
        
        self.scaling_results = results
        return results
    
    def generate_summary_plots(
        self,
        output_dir: Path,
        show: bool = True,
    ) -> List[Path]:
        """
        Generate summary plots for joint analysis.
        
        Args:
            output_dir: Directory to save plots
            show: Whether to display plots
            
        Returns:
            List of generated plot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = []
        
        # 1. τ × Δν consistency plot
        if self.consistency_results:
            fig = self._plot_tau_deltanu_consistency()
            path = output_dir / "tau_deltanu_consistency.pdf"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plots.append(path)
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 2. Frequency scaling plot
        if self.scaling_results:
            fig = self._plot_frequency_scaling()
            path = output_dir / "frequency_scaling.pdf"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plots.append(path)
            if show:
                plt.show()
            else:
                plt.close(fig)
        
        # 3. Cross-telescope comparison
        fig = self._plot_telescope_comparison()
        path = output_dir / "telescope_comparison.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plots.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        log.info(f"Generated {len(plots)} joint analysis plots in {output_dir}")
        return plots
    
    def _plot_tau_deltanu_consistency(self) -> plt.Figure:
        """Plot τ × Δν product for all bursts."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_results = [r for r in self.consistency_results if r.tau_delta_nu_product is not None]
        
        if not valid_results:
            ax.text(0.5, 0.5, "No valid τ × Δν measurements", ha="center", va="center")
            return fig
        
        # Prepare data
        names = [f"{r.burst_name}\n({r.telescope})" for r in valid_results]
        products = [r.tau_delta_nu_product for r in valid_results]
        errors = [r.tau_delta_nu_product_err or 0 for r in valid_results]
        colors = ["green" if r.is_consistent else "red" for r in valid_results]
        
        # Bar plot
        x = np.arange(len(names))
        bars = ax.bar(x, products, yerr=errors, capsize=3, color=colors, alpha=0.7, edgecolor="black")
        
        # Reference lines
        ax.axhline(C_THIN_SCREEN, color="blue", linestyle="--", label=f"Thin screen (C={C_THIN_SCREEN:.2f})")
        ax.axhline(C_EXTENDED, color="orange", linestyle="--", label=f"Extended (C={C_EXTENDED:.1f})")
        ax.axhspan(C_RANGE[0], C_RANGE[1], alpha=0.1, color="gray", label="Expected range")
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(r"$\tau \times \Delta\nu_{\rm dc}$ (dimensionless)", fontsize=12)
        ax.set_title("Scattering-Scintillation Consistency Check", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_ylim(0, max(products) * 1.3 if products else 2)
        
        plt.tight_layout()
        return fig
    
    def _plot_frequency_scaling(self) -> plt.Figure:
        """Plot frequency scaling analysis for co-detected bursts."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: τ comparison (CHIME vs DSA at 1 GHz reference)
        ax1 = axes[0]
        valid_tau = [r for r in self.scaling_results if r.tau_chime_ms and r.tau_dsa_ms]
        
        if valid_tau:
            chime_tau = [r.tau_chime_ms for r in valid_tau]
            dsa_tau = [r.tau_dsa_ms for r in valid_tau]
            names = [r.burst_name for r in valid_tau]
            
            ax1.scatter(chime_tau, dsa_tau, s=80, c="purple", alpha=0.7, edgecolors="black")
            for i, name in enumerate(names):
                ax1.annotate(name, (chime_tau[i], dsa_tau[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")
            
            # 1:1 line
            lims = [0, max(max(chime_tau), max(dsa_tau)) * 1.2]
            ax1.plot(lims, lims, "k--", alpha=0.5, label="1:1")
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
        
        ax1.set_xlabel(r"$\tau_{\rm 1\,GHz}$ (CHIME) [ms]", fontsize=11)
        ax1.set_ylabel(r"$\tau_{\rm 1\,GHz}$ (DSA) [ms]", fontsize=11)
        ax1.set_title(r"Scattering Time at 1 GHz Reference", fontsize=12, fontweight="bold")
        ax1.legend()
        
        # Panel 2: Δν comparison
        ax2 = axes[1]
        valid_nu = [r for r in self.scaling_results if r.delta_nu_chime_mhz and r.delta_nu_dsa_mhz]
        
        if valid_nu:
            chime_nu = [r.delta_nu_chime_mhz for r in valid_nu]
            dsa_nu = [r.delta_nu_dsa_mhz for r in valid_nu]
            names = [r.burst_name for r in valid_nu]
            
            ax2.scatter(chime_nu, dsa_nu, s=80, c="teal", alpha=0.7, edgecolors="black")
            for i, name in enumerate(names):
                ax2.annotate(name, (chime_nu[i], dsa_nu[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")
            
            # Expected scaling line (ν^4)
            if chime_nu:
                scale_factor = (self.FREQ_DSA / self.FREQ_CHIME) ** 4
                expected_dsa = [c * scale_factor for c in chime_nu]
                ax2.plot(chime_nu, expected_dsa, "g--", alpha=0.7, label=r"$\nu^4$ scaling")
        
        ax2.set_xlabel(r"$\Delta\nu_{\rm dc}$ (CHIME) [MHz]", fontsize=11)
        ax2.set_ylabel(r"$\Delta\nu_{\rm dc}$ (DSA) [MHz]", fontsize=11)
        ax2.set_title(r"Decorrelation Bandwidth", fontsize=12, fontweight="bold")
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def _plot_telescope_comparison(self) -> plt.Figure:
        """Plot side-by-side comparison of CHIME and DSA measurements."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        scat_df = self.db.to_dataframe("scattering")
        scint_df = self.db.to_dataframe("scintillation")
        
        # Panel 1: τ distribution by telescope
        ax = axes[0, 0]
        if not scat_df.empty and "tau_1ghz" in scat_df.columns:
            for tel, color in [("chime", "blue"), ("dsa", "red")]:
                data = scat_df[scat_df["telescope"] == tel]["tau_1ghz"].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=10, alpha=0.5, label=tel.upper(), color=color)
        ax.set_xlabel(r"$\tau_{\rm 1\,GHz}$ [ms]")
        ax.set_ylabel("Count")
        ax.set_title("Scattering Time Distribution")
        ax.legend()
        
        # Panel 2: Δν distribution by telescope
        ax = axes[0, 1]
        if not scint_df.empty and "delta_nu_dc" in scint_df.columns:
            for tel, color in [("chime", "blue"), ("dsa", "red")]:
                data = scint_df[scint_df["telescope"] == tel]["delta_nu_dc"].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=10, alpha=0.5, label=tel.upper(), color=color)
        ax.set_xlabel(r"$\Delta\nu_{\rm dc}$ [MHz]")
        ax.set_ylabel("Count")
        ax.set_title("Decorrelation Bandwidth Distribution")
        ax.legend()
        
        # Panel 3: τ vs burst (paired bars)
        ax = axes[1, 0]
        if not scat_df.empty:
            bursts = scat_df["burst_name"].unique()
            x = np.arange(len(bursts))
            width = 0.35
            
            chime_tau = [scat_df[(scat_df["burst_name"] == b) & (scat_df["telescope"] == "chime")]["tau_1ghz"].values for b in bursts]
            dsa_tau = [scat_df[(scat_df["burst_name"] == b) & (scat_df["telescope"] == "dsa")]["tau_1ghz"].values for b in bursts]
            
            chime_vals = [v[0] if len(v) > 0 else 0 for v in chime_tau]
            dsa_vals = [v[0] if len(v) > 0 else 0 for v in dsa_tau]
            
            ax.bar(x - width/2, chime_vals, width, label="CHIME", color="blue", alpha=0.7)
            ax.bar(x + width/2, dsa_vals, width, label="DSA", color="red", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(bursts, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(r"$\tau_{\rm 1\,GHz}$ [ms]")
            ax.set_title("Scattering Time by Burst")
            ax.legend()
        
        # Panel 4: χ² quality comparison
        ax = axes[1, 1]
        if not scat_df.empty and "chi2_reduced" in scat_df.columns:
            for tel, color, marker in [("chime", "blue", "o"), ("dsa", "red", "s")]:
                data = scat_df[scat_df["telescope"] == tel]
                if len(data) > 0:
                    ax.scatter(
                        range(len(data)),
                        data["chi2_reduced"].values,
                        c=color, marker=marker, alpha=0.7,
                        label=tel.upper(), s=60
                    )
            ax.axhline(1.0, color="green", linestyle="--", alpha=0.7, label="Ideal")
            ax.set_ylabel(r"$\chi^2_{\rm red}$")
            ax.set_xlabel("Burst index")
            ax.set_title("Fit Quality by Telescope")
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a text report summarizing joint analysis results.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        lines = [
            "=" * 70,
            "FLITS JOINT ANALYSIS REPORT",
            "=" * 70,
            "",
        ]
        
        # Consistency summary
        lines.append("1. SCATTERING-SCINTILLATION CONSISTENCY (τ × Δν)")
        lines.append("-" * 50)
        
        if self.consistency_results:
            n_consistent = sum(1 for r in self.consistency_results if r.is_consistent)
            n_total = len([r for r in self.consistency_results if r.tau_delta_nu_product is not None])
            lines.append(f"   Consistent: {n_consistent}/{n_total} bursts")
            lines.append("")
            
            for r in self.consistency_results:
                if r.tau_delta_nu_product is not None:
                    status = "✓" if r.is_consistent else "✗"
                    lines.append(f"   {status} {r.burst_name}/{r.telescope}:")
                    lines.append(f"      τ×Δν = {r.tau_delta_nu_product:.3f} ± {r.tau_delta_nu_product_err or 0:.3f}")
                    lines.append(f"      {r.interpretation}")
        else:
            lines.append("   No consistency results available")
        
        lines.append("")
        
        # Scaling summary
        lines.append("2. FREQUENCY SCALING (Co-detected bursts)")
        lines.append("-" * 50)
        
        if self.scaling_results:
            for r in self.scaling_results:
                lines.append(f"   {r.burst_name}:")
                if r.alpha_tau is not None:
                    lines.append(f"      α (scattering): {r.alpha_tau:.2f} ± {r.alpha_tau_err or 0:.2f}")
                if r.alpha_delta_nu is not None:
                    lines.append(f"      α (scintillation): {r.alpha_delta_nu:.2f} ± {r.alpha_delta_nu_err or 0:.2f}")
                lines.append(f"      {r.interpretation}")
        else:
            lines.append("   No scaling results available")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            log.info(f"Report saved to {output_path}")
        
        return report

