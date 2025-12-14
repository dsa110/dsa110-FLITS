"""
summary_plots.py
================

Generate publication-quality summary figures for batch analysis results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from .results_db import ResultsDatabase

log = logging.getLogger(__name__)


# Custom colormap for quality indicators
QUALITY_COLORS = {
    "good": "#2ecc71",      # Green
    "marginal": "#f39c12",  # Orange
    "bad": "#e74c3c",       # Red
    "unknown": "#95a5a6",   # Gray
}


def create_sample_overview(
    db: ResultsDatabase,
    output_path: Optional[Path] = None,
    title: str = "FLITS Sample Overview",
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive overview figure showing all bursts in the sample.
    
    Layout:
    - Top: Sample statistics and quality summary
    - Middle: Parameter distributions (τ, Δν, α)
    - Bottom: Burst-by-burst comparison grid
    
    Args:
        db: Results database
        output_path: Optional path to save figure
        title: Figure title
        show: Whether to display figure
        
    Returns:
        Matplotlib figure
    """
    scat_df = db.to_dataframe("scattering")
    scint_df = db.to_dataframe("scintillation")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1.5, 2])
    
    # -------------------------------------------------------------------------
    # Row 1: Statistics summary
    # -------------------------------------------------------------------------
    
    # Panel 1.1: Sample counts
    ax_counts = fig.add_subplot(gs[0, 0])
    _plot_sample_counts(ax_counts, scat_df, scint_df)
    
    # Panel 1.2: Quality distribution
    ax_quality = fig.add_subplot(gs[0, 1])
    _plot_quality_distribution(ax_quality, scat_df)
    
    # Panel 1.3: Text summary
    ax_text = fig.add_subplot(gs[0, 2])
    _plot_text_summary(ax_text, scat_df, scint_df)
    
    # -------------------------------------------------------------------------
    # Row 2: Parameter distributions
    # -------------------------------------------------------------------------
    
    # Panel 2.1: τ distribution
    ax_tau = fig.add_subplot(gs[1, 0])
    _plot_tau_distribution(ax_tau, scat_df)
    
    # Panel 2.2: Δν distribution  
    ax_deltanu = fig.add_subplot(gs[1, 1])
    _plot_deltanu_distribution(ax_deltanu, scint_df)
    
    # Panel 2.3: α distribution
    ax_alpha = fig.add_subplot(gs[1, 2])
    _plot_alpha_distribution(ax_alpha, scat_df)
    
    # -------------------------------------------------------------------------
    # Row 3: Burst-by-burst comparison
    # -------------------------------------------------------------------------
    
    ax_grid = fig.add_subplot(gs[2, :])
    _plot_burst_comparison_grid(ax_grid, scat_df, scint_df)
    
    # -------------------------------------------------------------------------
    # Final touches
    # -------------------------------------------------------------------------
    
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        log.info(f"Sample overview saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def _plot_sample_counts(ax, scat_df, scint_df):
    """Bar chart of sample counts by telescope and analysis type."""
    categories = ["CHIME\nScattering", "DSA\nScattering", "CHIME\nScintillation", "DSA\nScintillation"]
    counts = [
        len(scat_df[scat_df["telescope"] == "chime"]) if not scat_df.empty else 0,
        len(scat_df[scat_df["telescope"] == "dsa"]) if not scat_df.empty else 0,
        len(scint_df[scint_df["telescope"] == "chime"]) if not scint_df.empty else 0,
        len(scint_df[scint_df["telescope"] == "dsa"]) if not scint_df.empty else 0,
    ]
    colors = ["#3498db", "#e74c3c", "#3498db", "#e74c3c"]
    hatches = ["", "", "//", "//"]
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    ax.set_ylabel("Number of Bursts")
    ax.set_title("Sample Composition", fontweight="bold")
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")


def _plot_quality_distribution(ax, scat_df):
    """Pie chart of fit quality flags."""
    if scat_df.empty or "quality_flag" not in scat_df.columns:
        ax.text(0.5, 0.5, "No quality data", ha="center", va="center")
        ax.axis("off")
        return
    
    quality_counts = scat_df["quality_flag"].value_counts()
    
    colors = [QUALITY_COLORS.get(q, QUALITY_COLORS["unknown"]) for q in quality_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        quality_counts.values,
        labels=quality_counts.index,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        explode=[0.02] * len(quality_counts),
    )
    
    ax.set_title("Fit Quality Distribution", fontweight="bold")


def _plot_text_summary(ax, scat_df, scint_df):
    """Text panel with key statistics."""
    ax.axis("off")
    
    lines = []
    
    # Unique bursts
    all_bursts = set()
    if not scat_df.empty:
        all_bursts.update(scat_df["burst_name"].unique())
    if not scint_df.empty:
        all_bursts.update(scint_df["burst_name"].unique())
    lines.append(f"Unique bursts: {len(all_bursts)}")
    
    # Co-detected
    if not scat_df.empty:
        chime_bursts = set(scat_df[scat_df["telescope"] == "chime"]["burst_name"])
        dsa_bursts = set(scat_df[scat_df["telescope"] == "dsa"]["burst_name"])
        co_detected = chime_bursts & dsa_bursts
        lines.append(f"Co-detected: {len(co_detected)}")
    
    # Scattering stats
    if not scat_df.empty and "tau_1ghz" in scat_df.columns:
        tau_vals = scat_df["tau_1ghz"].dropna()
        if len(tau_vals) > 0:
            lines.append(f"\nScattering (τ₁ᴳʜᴢ):")
            lines.append(f"  Median: {tau_vals.median():.3f} ms")
            lines.append(f"  Range: [{tau_vals.min():.3f}, {tau_vals.max():.3f}] ms")
    
    # Scintillation stats
    if not scint_df.empty and "delta_nu_dc" in scint_df.columns:
        nu_vals = scint_df["delta_nu_dc"].dropna()
        if len(nu_vals) > 0:
            lines.append(f"\nScintillation (Δν_dc):")
            lines.append(f"  Median: {nu_vals.median():.3f} MHz")
            lines.append(f"  Range: [{nu_vals.min():.3f}, {nu_vals.max():.3f}] MHz")
    
    text = "\n".join(lines)
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10, 
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary Statistics", fontweight="bold")


def _plot_tau_distribution(ax, scat_df):
    """Histogram of scattering times."""
    if scat_df.empty or "tau_1ghz" not in scat_df.columns:
        ax.text(0.5, 0.5, "No scattering data", ha="center", va="center")
        return
    
    for tel, color, label in [("chime", "#3498db", "CHIME"), ("dsa", "#e74c3c", "DSA")]:
        data = scat_df[scat_df["telescope"] == tel]["tau_1ghz"].dropna()
        if len(data) > 0:
            ax.hist(data, bins=15, alpha=0.6, color=color, label=label, edgecolor="black")
    
    ax.set_xlabel(r"$\tau_{\rm 1\,GHz}$ [ms]", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Scattering Time Distribution", fontweight="bold")
    ax.legend()


def _plot_deltanu_distribution(ax, scint_df):
    """Histogram of decorrelation bandwidths."""
    if scint_df.empty or "delta_nu_dc" not in scint_df.columns:
        ax.text(0.5, 0.5, "No scintillation data", ha="center", va="center")
        return
    
    for tel, color, label in [("chime", "#3498db", "CHIME"), ("dsa", "#e74c3c", "DSA")]:
        data = scint_df[scint_df["telescope"] == tel]["delta_nu_dc"].dropna()
        if len(data) > 0:
            ax.hist(data, bins=15, alpha=0.6, color=color, label=label, edgecolor="black")
    
    ax.set_xlabel(r"$\Delta\nu_{\rm dc}$ [MHz]", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Decorrelation BW Distribution", fontweight="bold")
    ax.legend()


def _plot_alpha_distribution(ax, scat_df):
    """Histogram of frequency scaling indices."""
    if scat_df.empty or "alpha" not in scat_df.columns:
        ax.text(0.5, 0.5, "No α data", ha="center", va="center")
        return
    
    data = scat_df["alpha"].dropna()
    if len(data) == 0:
        ax.text(0.5, 0.5, "No α data", ha="center", va="center")
        return
    
    ax.hist(data, bins=15, alpha=0.7, color="#9b59b6", edgecolor="black")
    
    # Reference line for Kolmogorov
    ax.axvline(4.0, color="green", linestyle="--", linewidth=2, label="Kolmogorov (α=4)")
    ax.axvline(4.4, color="orange", linestyle=":", linewidth=2, label="Theoretical (α=4.4)")
    
    ax.set_xlabel(r"Scaling index $\alpha$", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Frequency Scaling Index", fontweight="bold")
    ax.legend(fontsize=9)


def _plot_burst_comparison_grid(ax, scat_df, scint_df):
    """Grid showing burst-by-burst comparison of measurements."""
    ax.axis("off")
    
    # Get all unique bursts
    all_bursts = sorted(set(
        list(scat_df["burst_name"].unique() if not scat_df.empty else []) +
        list(scint_df["burst_name"].unique() if not scint_df.empty else [])
    ))
    
    if not all_bursts:
        ax.text(0.5, 0.5, "No burst data available", ha="center", va="center", fontsize=14)
        return
    
    # Create data table
    rows = []
    for burst in all_bursts:
        row = {"Burst": burst}
        
        for tel in ["chime", "dsa"]:
            # Scattering
            if not scat_df.empty:
                scat_row = scat_df[(scat_df["burst_name"] == burst) & (scat_df["telescope"] == tel)]
                if len(scat_row) > 0:
                    tau = scat_row["tau_1ghz"].values[0]
                    row[f"τ ({tel.upper()})"] = f"{tau:.3f}" if pd.notna(tau) else "—"
                else:
                    row[f"τ ({tel.upper()})"] = "—"
            
            # Scintillation
            if not scint_df.empty:
                scint_row = scint_df[(scint_df["burst_name"] == burst) & (scint_df["telescope"] == tel)]
                if len(scint_row) > 0:
                    nu = scint_row["delta_nu_dc"].values[0]
                    row[f"Δν ({tel.upper()})"] = f"{nu:.3f}" if pd.notna(nu) else "—"
                else:
                    row[f"Δν ({tel.upper()})"] = "—"
        
        rows.append(row)
    
    # Create table
    df = pd.DataFrame(rows)
    
    # Display as matplotlib table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(df.columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#3498db")
            cell.set_text_props(color="white")
    
    ax.set_title("Burst-by-Burst Measurements", fontweight="bold", fontsize=12, pad=20)


def create_tau_deltanu_scatter(
    db: ResultsDatabase,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create scatter plot of τ vs Δν with theoretical relationships.
    
    Args:
        db: Results database
        output_path: Optional path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scat_df = db.to_dataframe("scattering")
    scint_df = db.to_dataframe("scintillation")
    
    if scat_df.empty or scint_df.empty:
        ax.text(0.5, 0.5, "Insufficient data for τ-Δν comparison", ha="center", va="center")
        return fig
    
    # Merge on burst_name and telescope
    merged = pd.merge(scat_df, scint_df, on=["burst_name", "telescope"], suffixes=("_scat", "_scint"))
    
    if merged.empty:
        ax.text(0.5, 0.5, "No matching scattering + scintillation measurements", ha="center", va="center")
        return fig
    
    # Plot by telescope
    for tel, color, marker in [("chime", "#3498db", "o"), ("dsa", "#e74c3c", "s")]:
        subset = merged[merged["telescope"] == tel]
        if len(subset) > 0:
            tau = subset["tau_1ghz"].values
            deltanu = subset["delta_nu_dc"].values
            names = subset["burst_name"].values
            
            ax.scatter(tau, deltanu, c=color, marker=marker, s=100, alpha=0.7, 
                      label=tel.upper(), edgecolors="black", linewidths=1)
            
            # Annotate
            for i, name in enumerate(names):
                ax.annotate(name, (tau[i], deltanu[i]), fontsize=8, 
                           xytext=(5, 5), textcoords="offset points")
    
    # Theoretical lines: τ × Δν = const
    tau_range = np.logspace(-2, 2, 100)
    
    # Thin screen: C = 1/(2π)
    ax.plot(tau_range, (1/(2*np.pi)) / tau_range, "g--", alpha=0.7, 
           label=r"Thin screen ($C = 1/2\pi$)")
    
    # Extended medium: C = 1
    ax.plot(tau_range, 1.0 / tau_range, "orange", linestyle="--", alpha=0.7,
           label=r"Extended ($C = 1$)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_{\rm 1\,GHz}$ [ms]", fontsize=12)
    ax.set_ylabel(r"$\Delta\nu_{\rm dc}$ [MHz]", fontsize=12)
    ax.set_title(r"Scattering vs Scintillation: $\tau \times \Delta\nu$ Relationship", 
                fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        log.info(f"τ-Δν scatter plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_all_summary_plots(
    db: ResultsDatabase,
    output_dir: Path,
    show: bool = False,
) -> List[Path]:
    """
    Generate all summary plots and save to output directory.
    
    Args:
        db: Results database
        output_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        List of generated plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = []
    
    # Sample overview
    path = output_dir / "sample_overview.pdf"
    create_sample_overview(db, path, show=show)
    plots.append(path)
    
    # τ-Δν scatter
    path = output_dir / "tau_deltanu_scatter.pdf"
    create_tau_deltanu_scatter(db, path, show=show)
    plots.append(path)
    
    log.info(f"Generated {len(plots)} summary plots in {output_dir}")
    return plots

