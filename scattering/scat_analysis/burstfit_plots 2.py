"""
burstfit_plots.py
=================

Modular, focused plotting functions for FRB scattering analysis.
Replaces the unwieldy 16-panel omnibus plot with smaller, targeted visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from .burstfit import plot_dynamic


def plot_fit_summary(dataset, results, save_path=None, show=True):
    """
    Create a focused 2x2 plot showing the core fit results.
    
    Panels:
    - Top left: Observed data
    - Top right: Best-fit model
    - Bottom left: Residuals
    - Bottom right: Collapsed profiles (time + frequency)
    """
    best_key = results["best_key"]
    best_p = results["best_params"]
    model_instance = results["model_instance"]
    
    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dynamic spectrum plots
    vmin, vmax = np.percentile(dataset.data, [1, 99])
    
    # Data
    plot_dynamic(axes[0,0], dataset.data, dataset.time, dataset.freq, 
                 vmin=vmin, vmax=vmax, cmap='plasma')
    axes[0,0].set_title('Observed Data', fontsize=14, fontweight='bold')
    
    # Model
    plot_dynamic(axes[0,1], model_dyn, dataset.time, dataset.freq,
                 vmin=vmin, vmax=vmax, cmap='plasma')
    axes[0,1].set_title(f'Best-Fit Model ({best_key})', fontsize=14, fontweight='bold')
    
    # Residuals
    res_std = np.std(residual)
    plot_dynamic(axes[1,0], residual, dataset.time, dataset.freq,
                 vmin=-3*res_std, vmax=3*res_std, cmap='coolwarm')
    axes[1,0].set_title('Residuals (Data - Model)', fontsize=14, fontweight='bold')
    
    # Collapsed profiles
    axes[1,1].clear()
    axes[1,1].set_title('Collapsed Profiles', fontsize=14, fontweight='bold')
    
    # Time profile (inset on top)
    ax_time = fig.add_axes([0.57, 0.15, 0.35, 0.15])
    time_data = np.sum(dataset.data, axis=0)
    time_model = np.sum(model_dyn, axis=0)
    ax_time.plot(dataset.time, time_data, 'k-', lw=1.5, label='Data', alpha=0.7)
    ax_time.plot(dataset.time, time_model, 'm-', lw=2, label='Model')
    ax_time.set_xlabel('Time [ms]', fontsize=10)
    ax_time.set_ylabel('Intensity', fontsize=10)
    ax_time.legend(fontsize=9, loc='upper right')
    ax_time.grid(alpha=0.3)
    
    # Frequency spectrum (main plot)
    freq_data = np.sum(dataset.data, axis=1)
    freq_model = np.sum(model_dyn, axis=1)
    axes[1,1].plot(dataset.freq, freq_data, 'k-', lw=1.5, label='Data', alpha=0.7)
    axes[1,1].plot(dataset.freq, freq_model, 'm-', lw=2, label='Model')
    axes[1,1].set_xlabel('Frequency [GHz]', fontsize=11)
    axes[1,1].set_ylabel('Intensity', fontsize=11)
    axes[1,1].legend(fontsize=10, loc='best')
    axes[1,1].grid(alpha=0.3)
    
    # Add goodness-of-fit if available
    gof = results.get("goodness_of_fit")
    if gof:
        chi2 = gof.get('chi2_reduced', np.nan)
        axes[1,1].text(0.02, 0.98, f'χ²/dof = {chi2:.2f}', 
                      transform=axes[1,1].transAxes,
                      fontsize=11, va='top', ha='left',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved fit summary to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_residual_diagnostics(dataset, results, save_path=None, show=True):
    """
    Create a 2x2 plot focused on residual analysis.
    
    Panels:
    - Top left: Residual histogram vs Gaussian
    - Top right: Residual autocorrelation
    - Bottom left: Residuals vs time (collapsed)
    - Bottom right: Residuals vs frequency (collapsed)
    """
    best_key = results["best_key"]
    best_p = results["best_params"]
    model_instance = results["model_instance"]
    gof = results.get("goodness_of_fit", {})
    
    model_dyn = model_instance(best_p, best_key)
    residual = dataset.data - model_dyn
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Normalized residual histogram
    res_norm = residual / model_instance.noise_std[:, None]
    axes[0,0].hist(res_norm.flatten(), bins=100, density=True, 
                   color='gray', alpha=0.7, label='Residuals')
    x_pdf = np.linspace(-4, 4, 200)
    axes[0,0].plot(x_pdf, stats.norm.pdf(x_pdf), 'm-', lw=2.5, label='N(0,1)')
    axes[0,0].set_xlabel('Normalized Residual', fontsize=11)
    axes[0,0].set_ylabel('Probability Density', fontsize=11)
    axes[0,0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0,0].legend(fontsize=10)
    axes[0,0].grid(alpha=0.3)
    
    # Residual ACF
    if 'residual_autocorr' in gof:
        lags_ms = (np.arange(len(gof['residual_autocorr'])) - 
                   len(gof['residual_autocorr'])//2) * dataset.dt_ms
        axes[0,1].plot(lags_ms, gof['residual_autocorr'], 'k-', lw=1.5, label='Data')
        axes[0,1].axhline(0, color='gray', ls='--', alpha=0.5)
        axes[0,1].set_xlabel('Lag [ms]', fontsize=11)
        axes[0,1].set_ylabel('Autocorrelation', fontsize=11)
        axes[0,1].set_title('Residual Autocorrelation', fontsize=14, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'ACF not computed', 
                      ha='center', va='center', fontsize=12)
        axes[0,1].set_axis_off()
    
    # Residuals vs time
    resid_time = np.sum(residual, axis=0)
    axes[1,0].plot(dataset.time, resid_time, 'k-', lw=1, alpha=0.7)
    axes[1,0].axhline(0, color='m', ls='--', lw=2)
    axes[1,0].fill_between(dataset.time, 
                           -3*np.std(resid_time), 3*np.std(resid_time),
                           color='m', alpha=0.1)
    axes[1,0].set_xlabel('Time [ms]', fontsize=11)
    axes[1,0].set_ylabel('Residual (collapsed)', fontsize=11)
    axes[1,0].set_title('Time-Domain Residuals', fontsize=14, fontweight='bold')
    axes[1,0].grid(alpha=0.3)
    
    # Residuals vs frequency
    resid_freq = np.sum(residual, axis=1)
    axes[1,1].plot(dataset.freq, resid_freq, 'k-', lw=1, alpha=0.7)
    axes[1,1].axhline(0, color='m', ls='--', lw=2)
    axes[1,1].fill_between(dataset.freq,
                           -3*np.std(resid_freq), 3*np.std(resid_freq),
                           color='m', alpha=0.1)
    axes[1,1].set_xlabel('Frequency [GHz]', fontsize=11)
    axes[1,1].set_ylabel('Residual (collapsed)', fontsize=11)
    axes[1,1].set_title('Frequency-Domain Residuals', fontsize=14, fontweight='bold')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved residual diagnostics to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_parameter_summary(results, save_path=None, show=True):
    """
    Create a clean text summary of parameter posteriors.
    Can be displayed as a figure or printed.
    """
    param_names = results["param_names"]
    flat_chain = results["flat_chain"]
    best_p = results["best_params"]
    gof = results.get("goodness_of_fit", {})
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.axis('off')
    
    # Build summary text
    summary_lines = []
    summary_lines.append("="*50)
    summary_lines.append(f"  PARAMETER SUMMARY - {results['best_key']}")
    summary_lines.append("="*50)
    summary_lines.append("")
    
    # Goodness of fit
    if gof:
        chi2 = gof.get('chi2_reduced', np.nan)
        summary_lines.append(f"χ²/dof:  {chi2:.3f}")
        summary_lines.append("")
    
    # Parameter posteriors
    summary_lines.append("Parameters (median ± std):")
    summary_lines.append("-" * 50)
    for i, name in enumerate(param_names):
        median = np.median(flat_chain[:, i])
        std = np.std(flat_chain[:, i])
        lo = np.percentile(flat_chain[:, i], 16)
        hi = np.percentile(flat_chain[:, i], 84)
        summary_lines.append(f"  {name:12s}:  {median:.4f} ± {std:.4f}")
        summary_lines.append(f"               [{lo:.4f}, {hi:.4f}]  (16th, 84th %ile)")
    
    summary_lines.append("")
    summary_lines.append("="*50)
    
    summary_text = "\n".join(summary_lines)
    
    ax.text(0.1, 0.9, summary_text, 
            transform=ax.transAxes,
            fontfamily='monospace', fontsize=10,
            verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved parameter summary to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Also print to console
    print(summary_text)
    
    return fig


def plot_mcmc_diagnostics(results, save_path=None, show=True):
    """
    Create MCMC-specific diagnostics (traces, acceptance, etc.)
    
    Panels:
    - Trace plots for key parameters
    - Parameter correlations
    """
    sampler = results["sampler"]
    param_names = results["param_names"]
    flat_chain = results["flat_chain"]
    chain_stats = results.get("chain_stats", {})
    burn_in = chain_stats.get("burn_in", 0)
    
    chain = sampler.get_chain()
    n_params = len(param_names)
    
    # Create figure with trace plots
    n_show = min(6, n_params)  # Show up to 6 parameters
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2*n_show))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        # Plot every 10th walker to avoid clutter
        axes[i].plot(chain[:, ::10, i], 'k', alpha=0.2, lw=0.5)
        axes[i].axvline(burn_in, color='m', ls='--', lw=2, label=f'Burn-in ({burn_in})')
        axes[i].set_ylabel(param_names[i], fontsize=10)
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=9)
        if i == n_show - 1:
            axes[i].set_xlabel('Step', fontsize=10)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('MCMC Chain Traces', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MCMC diagnostics to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
