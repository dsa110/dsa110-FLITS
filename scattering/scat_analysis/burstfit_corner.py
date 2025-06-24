"""
burstfit_corner.py
==================

Utility for constructing and plotting corner
plots with posterior distributions.
"""

from __future__ import annotations

import logging
import pickle
import contextlib
from pathlib import Path
from typing import Any, Dict, Sequence

import corner
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from typing import List, Tuple, Dict, Any

from chainconsumer import ChainConsumer

def diagnose_sampler_convergence(sampler, param_names):
    """Diagnose MCMC convergence issues"""
    
    # Get the full chain
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    n_steps, n_walkers, n_params = chain.shape
    
    print(f"Chain shape: {chain.shape}")
    print(f"Total samples: {n_steps * n_walkers}")
    
    # 1. Check log probability evolution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Log probability traces
    axes[0].plot(log_prob[:, :], alpha=0.3, color='black')
    axes[0].set_ylabel('log(P)')
    axes[0].set_title('Log Probability Evolution')
    
    # Mean log prob
    mean_log_prob = np.mean(log_prob, axis=1)
    axes[1].plot(mean_log_prob, 'b-', linewidth=2)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Mean log(P)')
    axes[1].set_title('Mean Log Probability')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Parameter traces
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        # Plot a subset of walkers for clarity
        ax.plot(chain[:, ::5, i], alpha=0.5)
        ax.set_ylabel(name)
        ax.set_title(f'{name} chains')
    
    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.show()
    
    # 3. Autocorrelation analysis
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print("\nAutocorrelation times:")
        for name, t in zip(param_names, tau):
            print(f"  {name}: {t:.1f} steps")
        
        print(f"\nSuggested burn-in: {int(2 * np.max(tau))} steps")
        print(f"Suggested thinning: {int(0.5 * np.min(tau))}")
    except:
        print("Could not compute autocorrelation time (chain might be too short)")
    
    # 4. Gelman-Rubin statistic (R-hat)
    def gelman_rubin(chain):
        """Compute Gelman-Rubin statistic"""
        m, n = chain.shape[1], chain.shape[0]
        # Split chain into two halves
        chain1 = chain[:n//2, :]
        chain2 = chain[n//2:, :]
        
        # Within-chain variance
        W = 0.5 * (np.var(chain1, axis=0, ddof=1) + np.var(chain2, axis=0, ddof=1))
        
        # Between-chain variance
        mean1 = np.mean(chain1, axis=0)
        mean2 = np.mean(chain2, axis=0)
        B = n/2 * (mean1 - mean2)**2
        
        # Estimate of variance
        var_est = (1 - 1/n) * W + (1/n) * B
        
        # R-hat
        R_hat = np.sqrt(var_est / W)
        return R_hat
    
    print("\nGelman-Rubin R-hat (should be < 1.1):")
    for i, name in enumerate(param_names):
        r_hat = gelman_rubin(chain[:, :, i])
        status = "✓" if r_hat < 1.1 else "✗"
        print(f"  {name}: {r_hat:.3f} {status}")
    
    return chain, log_prob

def get_clean_samples(sampler, param_names, verbose=True):
    """Get properly processed samples for corner plot"""
    
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    n_steps, n_walkers, n_params = chain.shape
    
    # 1. Find where chains have converged using log probability
    mean_log_prob = np.mean(log_prob, axis=1)
    
    # Use a running mean to find where log prob stabilizes
    window = max(1, min(100, n_steps // 10))   # never smaller than 1
    running_mean = np.convolve(mean_log_prob, np.ones(window)/window, mode='valid')
    running_std = np.array([np.std(mean_log_prob[max(0,i-window):i+window]) 
                           for i in range(len(mean_log_prob))])
    
    # Find where standard deviation drops below threshold
    stable_idx = np.where(running_std[window:] < 0.1 * np.std(mean_log_prob[:window]))[0]
    
    if len(stable_idx) > 0:
        burn_in = stable_idx[0] + window
    else:
        burn_in = n_steps // 4  # Default fallback
    
    if verbose:
        print(f"Detected burn-in: {burn_in} steps")
    
    # 2. Compute autocorrelation time for thinning
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        thin = int(np.mean(tau) / 2)  # Half the autocorrelation time
        thin = max(1, min(thin, 20))  # Keep reasonable bounds
    except:
        thin = 5  # Default if autocorrelation fails
    
    if verbose:
        print(f"Using thinning: {thin}")
    
    # 3. Get the samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
    
    if verbose:
        print(f"Final samples: {flat_samples.shape[0]} (from {n_steps * n_walkers} total)")
    
    # 4. Remove outliers (optional but helpful for visualization)
    # Keep only samples within 99.9% of log probability range
    log_prob_threshold = np.percentile(flat_log_prob, 0.1)
    good_samples = flat_log_prob > log_prob_threshold
    
    if verbose:
        n_removed = np.sum(~good_samples)
        if n_removed > 0:
            print(f"Removed {n_removed} outlier samples")
    
    return flat_samples[good_samples]

def make_beautiful_corner(samples, param_names, best_params=None, title=""):
    """Create a well-formatted corner plot"""
    
    # Parameter labels with units
    label_map = {
        'c0': r'$c_0$ [a.u.]',
        't0': r'$t_0$ [ms]',
        'gamma': r'$\gamma$',
        'zeta': r'$\zeta$ [ms]',
        'tau_1ghz': r'$\tau_{\rm 1\,GHz}$ [ms]'
    }
    
    labels = [label_map.get(name, name) for name in param_names]
    
    # Compute quantiles for display
    quantiles = [0.16, 0.5, 0.84]
    
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=quantiles,
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={"fontsize": 14},
        label_kwargs={"fontsize": 14},
        # Better plot defaults
        plot_contours=True,
        plot_density=True,
        plot_datapoints=True,
        # Smoothing
        smooth=1.0,  # Slight smoothing for cleaner contours
        smooth1d=1.0,
        # Appearance
        bins=40,  # More bins for smoother histograms
        fill_contours=True,
        levels=(0.68, 0.95),  # 1 and 2 sigma contours
        color='purple',
        truth_color='magenta',
        # Data point appearance
        data_kwargs={
            'alpha': 0.5,
            'ms': 1.5,
            'color': 'black'
        },
        # Contour appearance  
        contour_kwargs={
            'colors': 'purple',
            'linewidths': 1.5
        },
        #contourf_kwargs={
        #    'colors': ['white', 'orchid', 'purple'],
        #    'alpha': 0.5
        #}
    )
    
    # Add true values if provided
    if best_params is not None:
        truths = [getattr(best_params, name) for name in param_names]
        
        # Extract axes
        axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
        
        # Add vertical lines for true values
        for i in range(len(param_names)):
            ax = axes[i, i]
            ax.axvline(truths[i], color='orchid', linestyle='--', linewidth=1.5)
            
        # Add lines in 2D plots
        for yi in range(len(param_names)):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truths[xi], color='orchid', linestyle='--', linewidth=1.5, alpha=0.75)
                ax.axhline(truths[yi], color='orchid', linestyle='--', linewidth=1.5, alpha=0.75)
                ax.plot(truths[xi], truths[yi], 'o', color='orchid', markersize=5)
    
    # Add title
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Adjust layout
    #plt.tight_layout()
    
    # Print summary statistics
    print("\nParameter Summary (median [16%, 84%]):")
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name}: {q50:.3f} [{q16:.3f}, {q84:.3f}]")
    
    return fig

def make_beautiful_corner_wide(samples, param_names, best_params=None, title=""):
    """Create a well-formatted corner plot"""
    
    # Parameter labels with units
    label_map = {
        'c0': r'$c_0$ [a.u.]',
        't0': r'$t_0$ [ms]',
        'gamma': r'$\gamma$',
        'zeta': r'$\zeta$ [ms]',
        'tau_1ghz': r'$\tau_{\rm 1\,GHz}$ [ms]'
    }
    
    labels = [label_map.get(name, name) for name in param_names]
    
    # Compute quantiles for display
    quantiles = [0.16, 0.5, 0.84]

    q   = np.percentile(samples, [2, 98], axis=0)      # loose 96 % band
    pad = 0.1 * (q[1] - q[0])                          # 10 % breathing room
    ranges = [(low - d, high + d) for (low, high), d in zip(q.T, pad)]
    
    fig = corner.corner(
        samples,
        labels=param_names,
        range=ranges,
        quantiles=quantiles,
        show_titles=True,
        title_fmt='.3f',
        max_n_ticks=4,
        title_kwargs={"fontsize": 14},
        label_kwargs={"fontsize": 14},
        # Better plot defaults
        plot_contours=True,
        plot_density=True,
        plot_datapoints=True,
        # Smoothing
        smooth=1.0,  # Slight smoothing for cleaner contours
        smooth1d=1.0,
        # Appearance
        bins=40,  # More bins for smoother histograms
        fill_contours=True,
        levels=(0.68, 0.95),  # 1 and 2 sigma contours
        color='purple',
        truth_color='magenta',
        # Data point appearance
        data_kwargs={
            'alpha': 0.5,
            'ms': 1.5,
            'color': 'black'
        },
        # Contour appearance  
        contour_kwargs={
            'colors': 'purple',
            'linewidths': 1.5
        },
        #contourf_kwargs={
        #    'colors': ['white', 'orchid', 'purple'],
        #    'alpha': 0.5
        #}
    )
    
    # Add true values if provided
    if best_params is not None:
        truths = [getattr(best_params, name) for name in param_names]
        
        # Extract axes
        axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
        
        # Add vertical lines for true values
        for i in range(len(param_names)):
            ax = axes[i, i]
            ax.axvline(truths[i], color='orchid', linestyle='--', linewidth=1.5)
            
        # Add lines in 2D plots
        for yi in range(len(param_names)):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truths[xi], color='orchid', linestyle='--', linewidth=1.5, alpha=0.75)
                ax.axhline(truths[yi], color='orchid', linestyle='--', linewidth=1.5, alpha=0.75)
                ax.plot(truths[xi], truths[yi], 'o', color='orchid', markersize=5)
    
    # Add title
    fig.suptitle(title, fontsize=16, y=1.02)
    
    # Adjust layout
    #plt.tight_layout()
    
    # Print summary statistics
    print("\nParameter Summary (median [16%, 84%]):")
    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name}: {q50:.3f} [{q16:.3f}, {q84:.3f}]")
    
    return fig
    
def make_chainconsumer_plot(samples, param_names, best_params=None):
    """Use ChainConsumer for publication-quality plots"""

    c = ChainConsumer()

    # Parameter labels
    label_map = {
        'c0': r'$c_0$',
        't0': r'$t_0$ [ms]',
        'gamma': r'$\gamma$',
        'zeta': r'$\zeta$ [ms]',
        'tau_1ghz': r'$\tau_{1\,\rm GHz}$ [ms]'
    }

    # Create parameter dictionary
    param_dict = {label_map.get(name, name): samples[:, i] 
                  for i, name in enumerate(param_names)}

    c.add_chain(param_dict, name="MCMC")

    if best_params is not None:
        truth_dict = {label_map.get(name, name): getattr(best_params, name) 
                     for name in param_names}
        c.add_truth(truth_dict, color='red')

    fig = c.plotter.plot(
        figsize=(10, 10),
        truth_alpha=0.8,
        diagonal_tick_labels=False
    )

    return fig

    
def quick_chain_check(sampler):
    """Quick check if chains are good enough for plotting"""
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    # Check 1: Log probability spread in last quarter
    last_quarter = log_prob[-len(log_prob)//4:]
    log_prob_spread = np.std(last_quarter.flatten())
    
    # Check 2: Parameter drift in last quarter  
    last_quarter_chain = chain[-len(chain)//4:]
    param_drift = np.std(np.mean(last_quarter_chain, axis=1), axis=0)
    initial_spread = np.std(chain[0], axis=0)
    relative_drift = param_drift / initial_spread
    
    print("Chain Health Check:")
    print(f"  Log-prob stability: {log_prob_spread:.2f} (want < 1.0)")
    print(f"  Parameter drift: {np.mean(relative_drift):.2f} (want < 0.1)")
    
    if log_prob_spread > 1.0 or np.mean(relative_drift) > 0.1:
        print("  ⚠️  Chains may need more steps!")
        print("  Consider running: sampler.run_mcmc(None, 1000, progress=True)")
    else:
        print("  ✓ Chains look converged")
    
    return log_prob_spread < 1.0 and np.mean(relative_drift) < 0.1
