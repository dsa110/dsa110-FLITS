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
Plotting functions for joint analysis of scattering and scintillation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_logic import (
    C_EXTENDED, C_RANGE, C_THIN_SCREEN, FREQ_CHIME, FREQ_DSA,
    ConsistencyResult, FrequencyScalingResult
)

log = logging.getLogger(__name__)


def generate_summary_plots(
    consistency_results: List[ConsistencyResult],
    scaling_results: List[FrequencyScalingResult],
    comparison_df: pd.DataFrame,
    output_dir: Path,
    show: bool = True,
) -> List[Path]:
    """
    Generate and save all summary plots for the joint analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    
    # Create and save each plot
    plot_functions = {
        "tau_deltanu_consistency.pdf": _plot_tau_deltanu_consistency,
        "frequency_scaling.pdf": _plot_frequency_scaling,
        "telescope_comparison.pdf": _plot_telescope_comparison,
    }
    
    for filename, func in plot_functions.items():
        if "consistency" in filename and consistency_results:
            fig = func(consistency_results)
        elif "scaling" in filename and scaling_results:
            fig = func(scaling_results)
        elif "comparison" in filename and not comparison_df.empty:
            fig = func(comparison_df)
        else:
            continue
            
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plots.append(path)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
    log.info(f"Generated {len(plots)} joint analysis plots in {output_dir}")
    return plots


def _plot_tau_deltanu_consistency(
    consistency_results: List[ConsistencyResult],
) -> plt.Figure:
    """Plot τ × Δν product for all bursts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_results = [r for r in consistency_results if r.tau_delta_nu_product is not None]
    
    if not valid_results:
        ax.text(0.5, 0.5, "No valid τ × Δν measurements", ha="center", va="center")
        return fig
        
    names = [f"{r.burst_name}\n({r.telescope})" for r in valid_results]
    products = [r.tau_delta_nu_product for r in valid_results]
    errors = [r.tau_delta_nu_product_err or 0 for r in valid_results]
    colors = ["green" if r.is_consistent else "red" for r in valid_results]
    
    x = np.arange(len(names))
    ax.bar(x, products, yerr=errors, capsize=3, color=colors, alpha=0.7, edgecolor="black")
    
    ax.axhline(C_THIN_SCREEN, color="blue", linestyle="--", label=f"Thin screen (C={C_THIN_SCREEN:.2f})")
    ax.axhline(C_EXTENDED, color="orange", linestyle="--", label=f"Extended (C={C_EXTENDED:.1f})")
    ax.axhspan(C_RANGE[0], C_RANGE[1], alpha=0.1, color="gray", label="Expected range")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$\tau \times \Delta\nu_{\rm dc}$ (dimensionless)")
    ax.set_title("Scattering-Scintillation Consistency Check")
    ax.legend()
    ax.set_ylim(0, max(products) * 1.3 if products else 2)
    
    plt.tight_layout()
    return fig


def _plot_frequency_scaling(
    scaling_results: List[FrequencyScalingResult],
) -> plt.Figure:
    """Plot frequency scaling analysis for co-detected bursts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    # Panel 1: τ comparison (CHIME vs DSA at 1 GHz)
    valid_tau = [r for r in scaling_results if r.tau_chime_ms and r.tau_dsa_ms]
    if valid_tau:
        chime_tau = [r.tau_chime_ms for r in valid_tau]
        dsa_tau = [r.tau_dsa_ms for r in valid_tau]
        names = [r.burst_name for r in valid_tau]
        
        axes[0].scatter(chime_tau, dsa_tau, s=80, c="purple", alpha=0.7, edgecolors="black")
        for i, name in enumerate(names):
            axes[0].annotate(name, (chime_tau[i], dsa_tau[i]), fontsize=8, xytext=(5,5), textcoords="offset points")
        
        lims = [0, max(max(chime_tau), max(dsa_tau)) * 1.2]
        axes[0].plot(lims, lims, "k--", alpha=0.5, label="1:1")
        axes[0].set_xlim(lims)
        axes[0].set_ylim(lims)
    
    axes[0].set_xlabel(r"$\tau_{\rm 1\,GHz}$ (CHIME) [ms]")
    axes[0].set_ylabel(r"$\tau_{\rm 1\,GHz}$ (DSA) [ms]")
    axes[0].set_title(r"Scattering Time at 1 GHz Reference")
    axes[0].legend()

    # Panel 2: Δν comparison
    valid_nu = [r for r in scaling_results if r.delta_nu_chime_mhz and r.delta_nu_dsa_mhz]
    if valid_nu:
        chime_nu = [r.delta_nu_chime_mhz for r in valid_nu]
        dsa_nu = [r.delta_nu_dsa_mhz for r in valid_nu]
        names = [r.burst_name for r in valid_nu]
        
        axes[1].scatter(chime_nu, dsa_nu, s=80, c="teal", alpha=0.7, edgecolors="black")
        for i, name in enumerate(names):
            axes[1].annotate(name, (chime_nu[i], dsa_nu[i]), fontsize=8, xytext=(5,5), textcoords="offset points")
        
        if chime_nu:
            scale_factor = (FREQ_DSA / FREQ_CHIME) ** 4
            expected_dsa = [c * scale_factor for c in chime_nu]
            axes[1].plot(chime_nu, expected_dsa, "g--", alpha=0.7, label=r"$\nu^4$ scaling")

    axes[1].set_xlabel(r"$\Delta\nu_{\rm dc}$ (CHIME) [MHz]")
    axes[1].set_ylabel(r"$\Delta\nu_{\rm dc}$ (DSA) [MHz]")
    axes[1].set_title("Decorrelation Bandwidth")
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def _plot_telescope_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Plot side-by-side comparison of CHIME and DSA measurements."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograms for τ and Δν
    for ax, col, xlabel in zip(axes[0], ["tau_1ghz", "delta_nu_dc"], [r"$\tau_{\rm 1\,GHz}$ [ms]", r"$\Delta\nu_{\rm dc}$ [MHz]"]):
        for tel, color in [("chime", "blue"), ("dsa", "red")]:
            data = comparison_df[comparison_df["telescope"] == tel][col].dropna()
            if not data.empty:
                ax.hist(data, bins=10, alpha=0.5, label=tel.upper(), color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{xlabel} Distribution")
        ax.legend()
        
    # Bar plot for τ by burst
    ax = axes[1, 0]
    bursts = comparison_df["burst_name"].unique()
    x = np.arange(len(bursts))
    width = 0.35
    
    chime_tau = comparison_df[comparison_df["telescope"] == "chime"].set_index("burst_name")
    dsa_tau = comparison_df[comparison_df["telescope"] == "dsa"].set_index("burst_name")
    
    chime_vals = chime_tau.reindex(bursts)["tau_1ghz"].fillna(0)
    dsa_vals = dsa_tau.reindex(bursts)["tau_1ghz"].fillna(0)
    
    ax.bar(x - width/2, chime_vals, width, label="CHIME", color="blue", alpha=0.7)
    ax.bar(x + width/2, dsa_vals, width, label="DSA", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bursts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$\tau_{\rm 1\,GHz}$ [ms]")
    ax.set_title("Scattering Time by Burst")
    ax.legend()

    # Scatter plot for fit quality (chi-squared)
    ax = axes[1, 1]
    for tel, color, marker in [("chime", "blue", "o"), ("dsa", "red", "s")]:
        data = comparison_df[comparison_df["telescope"] == tel]
        if not data.empty:
            ax.scatter(
                range(len(data)),
                data["chi2_reduced"].dropna(),
                c=color, marker=marker, alpha=0.7, label=tel.upper(), s=60
            )
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.7, label="Ideal $\chi^2_{red}=1$")
    ax.set_ylabel(r"Reduced $\chi^2$")
    ax.set_xlabel("Burst Index")
    ax.set_title("Fit Quality")
    ax.legend()
    
    plt.tight_layout()
    return fig
