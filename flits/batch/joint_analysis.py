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
joint_analysis.py
=================

Orchestrates the cross-validation of scattering and scintillation measurements
by coordinating calls to analysis, plotting, and reporting modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .analysis_logic import (
    analyze_frequency_scaling,
    check_tau_deltanu_consistency,
    ConsistencyResult,
    FrequencyScalingResult,
)
from .joint_analysis_plots import generate_summary_plots
from .results_db import ResultsDatabase

log = logging.getLogger(__name__)


class JointAnalysis:
    """
    Orchestrates joint analysis of scattering and scintillation measurements.
    
    This class acts as a high-level controller that:
    1. Fetches a unified comparison DataFrame from the results database.
    2. Invokes scientific analysis functions from the `analysis_logic` module.
    3. Triggers plot generation from the `joint_analysis_plots` module.
    4. Generates a summary text report.
    """
    
    def __init__(self, db: ResultsDatabase):
        """
        Initialize the analysis orchestrator.
        
        Args:
            db: An initialized `ResultsDatabase` instance.
        """
        self.db = db
        self.comparison_df: Optional[pd.DataFrame] = None
        self.consistency_results: List[ConsistencyResult] = []
        self.scaling_results: List[FrequencyScalingResult] = []
        
    def run_analysis(self, output_dir: Path, show_plots: bool = True):
        """
        Execute the full joint analysis workflow.
        
        Args:
            output_dir: Directory to save plots and reports.
            show_plots: Whether to display plots interactively.
        """
        log.info("Starting joint analysis...")
        
        # 1. Get unified data table
        self.comparison_df = self.db.get_comparison_table()
        if self.comparison_df.empty:
            log.warning("Comparison DataFrame is empty. No analysis to perform.")
            return
            
        # 2. Perform scientific analysis
        log.info("Checking τ-Δν consistency...")
        self.consistency_results = check_tau_deltanu_consistency(self.comparison_df)
        
        log.info("Analyzing frequency scaling for co-detected bursts...")
        self.scaling_results = analyze_frequency_scaling(self.comparison_df)
        
        # 3. Generate plots
        log.info(f"Generating summary plots in {output_dir}...")
        generate_summary_plots(
            self.consistency_results,
            self.scaling_results,
            self.comparison_df,
            output_dir,
            show=show_plots,
        )
        
        # 4. Generate report
        log.info("Generating text report...")
        report_path = output_dir / "joint_analysis_report.txt"
        self.generate_report(report_path)
        
        log.info("Joint analysis complete.")

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a text report summarizing the joint analysis results.
        
        Args:
            output_path: Optional path to save the report file.
            
        Returns:
            The generated report as a string.
        """
        lines = [
            "=" * 70,
            "FLITS JOINT ANALYSIS REPORT",
            "=" * 70,
            "\n",
            "1. SCATTERING-SCINTILLATION CONSISTENCY (τ × Δν)",
            "-" * 50,
        ]
        
        if self.consistency_results:
            n_consistent = sum(1 for r in self.consistency_results if r.is_consistent)
            n_total = len([r for r in self.consistency_results if r.tau_delta_nu_product is not None])
            lines.append(f"   Summary: {n_consistent}/{n_total} bursts show consistent τ and Δν measurements.")
            lines.append("")
            
            for r in self.consistency_results:
                if r.tau_delta_nu_product is not None:
                    status = "✓" if r.is_consistent else "✗"
                    lines.append(f"   {status} {r.burst_name}/{r.telescope}:")
                    lines.append(f"      τ×Δν = {r.tau_delta_nu_product:.3f} ± {r.tau_delta_nu_product_err or 0:.3f} | {r.interpretation}")
        else:
            lines.append("   No consistency results available.")
        
        lines.extend([
            "\n",
            "2. FREQUENCY SCALING (Co-detected bursts)",
            "-" * 50,
        ])
        
        if self.scaling_results:
            for r in self.scaling_results:
                lines.append(f"   {r.burst_name}:")
                if r.alpha_tau is not None:
                    lines.append(f"      α (from τ): {r.alpha_tau:.2f} ± {r.alpha_tau_err or 0:.2f}")
                if r.alpha_delta_nu is not None:
                    lines.append(f"      α (from Δν): {r.alpha_delta_nu:.2f} ± {r.alpha_delta_nu_err or 0:.2f}")
                lines.append(f"      Interpretation: {r.interpretation}")
        else:
            lines.append("   No frequency scaling results available.")
            
        lines.extend(["\n", "=" * 70])
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
            log.info(f"Report saved to {output_path}")
            
        return report
