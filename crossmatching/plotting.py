"""Publication-quality plotting utilities for TOA crossmatch analysis.

This module provides improved visualization of the CHIME-DSA co-detection
results, replacing ad-hoc notebook plotting code with reusable functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Use FLITS style if available
try:
    from flits.plotting import use_flits_style
    use_flits_style()
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')


def load_crossmatch_results(json_path: str | Path) -> dict:
    """Load crossmatch results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_toa_residuals(
    results: dict,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
    show: bool = True,
) -> plt.Figure:
    """Create a publication-quality residual plot.
    
    The residual is defined as: (Measured Offset) - (Geometric Delay).
    A residual of zero indicates perfect agreement between the two telescopes
    after accounting for light travel time differences.
    
    Parameters
    ----------
    results : dict
        Crossmatch results dictionary (keyed by burst nickname).
    output_path : str or Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple
        Figure size in inches.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract data
    names = []
    residuals = []
    errors = []
    dms = []
    
    for nickname, burst in results.items():
        names.append(nickname.capitalize())
        
        # Compute residual
        measured = burst['measured_offset_ms']
        geometric = burst['geometric_delay_ms']
        residual = measured - geometric
        residuals.append(residual)
        
        # Combine errors in quadrature: DM uncertainty + FWHM (timing precision)
        dm_err = burst['combined_dm_uncertainty_ms']
        fwhm = burst.get('fwhm_ms', 0)  # FWHM as timing uncertainty
        total_err = np.sqrt(dm_err**2 + fwhm**2)
        errors.append(total_err)
        
        dms.append(burst['dm'])
    
    # Convert to arrays
    residuals = np.array(residuals)
    errors = np.array(errors)
    dms = np.array(dms)
    x_pos = np.arange(len(names))
    
    # Color by DM
    norm = plt.Normalize(dms.min(), dms.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(dms))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plot residuals with error bars
    for i, (x, y, err, c) in enumerate(zip(x_pos, residuals, errors, colors)):
        ax.errorbar(
            x, y, yerr=err,
            fmt='o',
            markersize=10,
            color=c,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            zorder=3,
        )
    
    # Reference line at zero (perfect agreement)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Perfect agreement', zorder=1)
    
    # Shaded region for typical uncertainty
    median_err = np.median(errors)
    ax.axhspan(-median_err, median_err, color='green', alpha=0.15,
               label=f'Median ±1σ ({median_err:.1f} ms)', zorder=0)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Residual (ms)', fontsize=12)
    ax.set_xlabel('Burst', fontsize=12)
    ax.set_title('TOA Residuals: Measured Offset − Geometric Delay', fontsize=14, fontweight='bold')
    
    # Colorbar for DM
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('DM (pc cm⁻³)', fontsize=11)
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.set_ylim(-max(abs(residuals.min()), abs(residuals.max())) * 1.5,
                max(abs(residuals.min()), abs(residuals.max())) * 1.5)
    
    # Grid
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_offset_comparison(
    results: dict,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (10, 8),
    show: bool = True,
) -> plt.Figure:
    """Create a comparison plot of measured offsets vs geometric delays.
    
    This visualization shows how well the measured TOA offsets match
    the predicted geometric delays based on observatory positions.
    
    Parameters
    ----------
    results : dict
        Crossmatch results dictionary.
    output_path : str or Path, optional
        Path to save the figure.
    figsize : tuple
        Figure size in inches.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract data
    names = []
    measured = []
    geometric = []
    errors = []
    
    for nickname, burst in results.items():
        names.append(nickname.capitalize())
        measured.append(burst['measured_offset_ms'])
        geometric.append(burst['geometric_delay_ms'])
        
        dm_err = burst['combined_dm_uncertainty_ms']
        fwhm = burst.get('fwhm_ms', 0)
        errors.append(np.sqrt(dm_err**2 + fwhm**2))
    
    measured = np.array(measured)
    geometric = np.array(geometric)
    errors = np.array(errors)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # --- Left panel: Scatter with 1:1 line ---
    ax1 = axes[0]
    
    ax1.errorbar(
        geometric, measured, yerr=errors,
        fmt='o', markersize=8, capsize=3, alpha=0.8,
        color='steelblue', ecolor='gray', label='Co-detected bursts'
    )
    
    # 1:1 line
    lims = [min(geometric.min(), measured.min()) - 1,
            max(geometric.max(), measured.max()) + 1]
    ax1.plot(lims, lims, 'k--', linewidth=1.5, label='1:1 line', zorder=0)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    
    ax1.set_xlabel('Geometric Delay (ms)', fontsize=12)
    ax1.set_ylabel('Measured Offset (ms)', fontsize=12)
    ax1.set_title('Offset vs. Geometric Delay', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(linestyle=':', alpha=0.5)
    
    # --- Right panel: Histogram of residuals ---
    ax2 = axes[1]
    
    residuals = measured - geometric
    
    ax2.hist(residuals, bins=8, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax2.axvline(np.mean(residuals), color='orange', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(residuals):.2f} ms')
    
    ax2.set_xlabel('Residual (ms)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    
    # Add statistics annotation
    stats_text = (
        f"N = {len(residuals)}\n"
        f"Mean = {np.mean(residuals):.2f} ms\n"
        f"Std = {np.std(residuals):.2f} ms\n"
        f"Median = {np.median(residuals):.2f} ms"
    )
    ax2.text(0.95, 0.65, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    """Generate all crossmatch plots from the results JSON."""
    results_path = Path(__file__).parent / 'toa_crossmatch_results.json'
    
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return
    
    results = load_crossmatch_results(results_path)
    print(f"Loaded {len(results)} co-detected bursts.")
    
    output_dir = Path(__file__).parent
    
    # Generate residual plot
    plot_toa_residuals(
        results,
        output_path=output_dir / 'toa_residuals.pdf',
        show=False,
    )
    
    # Generate comparison plot
    plot_offset_comparison(
        results,
        output_path=output_dir / 'toa_offset_comparison.pdf',
        show=False,
    )
    
    print("Done! Generated improved crossmatch figures.")


if __name__ == '__main__':
    main()
