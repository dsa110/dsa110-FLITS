"""Publication-quality plotting utilities for TOA crossmatch analysis.

This module provides improved visualization of the CHIME-DSA co-detection
results, replacing ad-hoc notebook plotting code with reusable functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use scienceplots for publication-quality plots
try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
    # Handle negative signs in case of font issues
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')


def load_crossmatch_results(json_path: str | Path) -> dict:
    """Load crossmatch results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return {}


def plot_toa_analysis(
    results: dict,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (16, 6),
    show: bool = True,
) -> plt.Figure:
    """Create a premium 2-panel TOA crossmatch analysis figure.
    
    Panel A: Residual (Measured - Geometric) vs Burst Nickname.
    Panel B: Residual vs DM, to check for systematic dispersion errors.
    
    Parameters
    ----------
    results : dict
        Crossmatch results dictionary.
    output_path : str or Path, optional
        Path to save the figure (supports .png, .pdf, .svg).
    figsize : tuple
        Figure size in inches.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not results:
        logger.warning("No results to plot.")
        return plt.figure()

    # Extract data
    nicknames = []
    residuals = []
    errors = []
    dms = []
    
    for nickname, burst in results.items():
        nicknames.append(nickname.capitalize())
        
        # Residual = Measured Offset - Predicted Geometric Delay
        # A residual of 0ms means the shift at 400MHz perfectly matches 
        # the geometric baseline delay.
        res = burst['measured_offset_ms'] - burst['geometric_delay_ms']
        residuals.append(res)
        
        # Quadrature sum of systematic (DM) and statistical (FWHM/timing) errors
        dm_err = burst['combined_dm_uncertainty_ms']
        fwhm = burst.get('fwhm_ms', 0)
        total_err = np.sqrt(dm_err**2 + fwhm**2)
        errors.append(total_err)
        
        dms.append(burst['dm'])
    
    # Convert to arrays
    nicknames = np.array(nicknames)
    residuals = np.array(residuals)
    errors = np.array(errors)
    dms = np.array(dms)
    x_pos = np.arange(len(nicknames))
    
    # Sort by DM for the DM panel
    sort_idx = np.argsort(dms)
    dms_sorted = dms[sort_idx]
    res_sorted = residuals[sort_idx]
    err_sorted = errors[sort_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]}, constrained_layout=True)
    
    # --- PANEL A: Residual vs Burst ---
    norm = plt.Normalize(dms.min(), dms.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(dms))
    
    for i, (x, y, err, c) in enumerate(zip(x_pos, residuals, errors, colors)):
        ax1.errorbar(
            x, y, yerr=err,
            fmt='o', markersize=9, color=c, linewidth=2,
            capsize=4, capthick=1.5, zorder=3
        )
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Theoretical Expectation', zorder=1)
    
    # Shaded confidence band (expected spread)
    med_err = np.median(errors)
    ax1.axhspan(-med_err, med_err, color='green', alpha=0.1, label=f'Median Uncertainty (±{med_err:.1f}ms)', zorder=0)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(nicknames, rotation=35, ha='right', fontsize=10)
    ax1.set_ylabel('Residual (Measured - Geo) [ms]', fontsize=12)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)
    
    # Add colorbar for Panel A
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.04)
    cbar.set_label('DM [pc cm⁻³]', rotation=270, labelpad=15)
    
    # --- PANEL B: Residual vs DM (Systematics Check) ---
    ax2.errorbar(
        dms_sorted, res_sorted, yerr=err_sorted,
        fmt='o', markersize=7, color='gray', alpha=0.6,
        linewidth=1.5, capsize=3, zorder=2
    )
    
    # Linear fit to check for DM-dependent bias
    if len(dms) > 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(dms, residuals)
        dm_range = np.linspace(dms.min() * 0.9, dms.max() * 1.1, 100)
        ax2.plot(dm_range, slope * dm_range + intercept, color='red', linestyle='-', alpha=0.7, 
                 label=f'Linear Fit (r={r_val:.2f})', zorder=1)
        
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('DM [pc cm⁻³]', fontsize=12)
    ax2.grid(linestyle=':', alpha=0.4)
    ax2.legend(loc='lower left', fontsize='small')
    
    # Set shared y-limits for easier comparison
    ymax = max(abs(residuals.min()), abs(residuals.max())) * 1.4
    ax1.set_ylim(-ymax, ymax)
    ax2.set_ylim(-ymax, ymax)

    ymax = max(abs(residuals.min()), abs(residuals.max())) * 1.4
    ax1.set_ylim(-ymax, ymax)
    ax2.set_ylim(-ymax, ymax)

    if output_path:
        fig.savefig(output_path, dpi=300)
        logger.info(f"Analysis plot saved to {output_path}")

    if show:
        plt.show()

    return fig


def plot_systematics_matrix(
    results: dict,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (14, 10),
    show: bool = True,
) -> plt.Figure:
    """Create a 4-panel systematics matrix checking for various correlations.
    
    Plots Residual vs:
    1. DM (Physical modeling)
    2. FWHM (Signal structure/Pulse width)
    3. Combined Uncertainty (Measurement quality)
    4. MJD (Temporal stability)
    """
    if not results:
        return plt.figure()

    # Extract all data
    residuals = []
    dms = []
    fwhms = []
    errs = []
    mjds = []
    
    for burst in results.values():
        residuals.append(burst['measured_offset_ms'] - burst['geometric_delay_ms'])
        dms.append(burst['dm'])
        fwhms.append(burst.get('fwhm_ms', 0))
        errs.append(np.sqrt(burst['combined_dm_uncertainty_ms']**2 + burst.get('fwhm_ms', 0)**2))
        mjds.append(burst.get('dm_mjd', 0))
        
    data = {
        'DM [pc cm⁻³]': np.array(dms),
        'FWHM [ms]': np.array(fwhms),
        'Combined Uncertainty [ms]': np.array(errs),
        'Time [MJD]': np.array(mjds)
    }
    residuals = np.array(residuals)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten()
    
    for ax, (label, x_data) in zip(axes_flat, data.items()):
        # Filter out 0/invalid MJDs if necessary
        mask = x_data > 0 if 'Time' in label else np.ones_like(x_data, dtype=bool)
        if not np.any(mask):
            ax.text(0.5, 0.5, f"No valid {label} data", transform=ax.transAxes, ha='center')
            continue
            
        x, y = x_data[mask], residuals[mask]
        
        ax.scatter(x, y, s=60, alpha=0.7, edgecolors='k', color='steelblue')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add correlation coefficient and trendline
        if len(x) > 2:
            r, p = stats.pearsonr(x, y)
            slope, intercept, _, _, _ = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, color='red', alpha=0.4, linestyle=':')
            
        ax.set_xlabel(label)
        ax.set_ylabel('Residual [ms]')
        ax.grid(linestyle=':', alpha=0.4)
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        logger.info(f"Systematics matrix saved to {output_path}")

    if show:
        plt.show()
    return fig


def main():
    """Generate the premium consolidated crossmatch figure."""
    results_path = Path(__file__).parent / 'toa_crossmatch_results.json'
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    results = load_crossmatch_results(results_path)
    output_dir = Path(__file__).parent
    
    # Generate premium 2-panel analysis
    plot_toa_analysis(results, output_path=output_dir / 'toa_crossmatch_analysis_premium.png', show=False)
    
    # Generate the new systematics matrix
    plot_systematics_matrix(results, output_path=output_dir / 'systematics_check_matrix.png', show=False)
    
    # Also save as PDFs for publication quality
    plot_toa_analysis(results, output_path=output_dir / 'toa_crossmatch_analysis_premium.pdf', show=False)
    plot_systematics_matrix(results, output_path=output_dir / 'systematics_check_matrix.pdf', show=False)
    
    logger.info("Done! Generated expanded systematic analysis figures.")


if __name__ == '__main__':
    main()
