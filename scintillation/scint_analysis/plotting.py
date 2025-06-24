# ==============================================================================
# File: scint_analysis/scint_analysis/plotting.py 
# ==============================================================================

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import SymLogNorm, LogNorm

from . import core

log = logging.getLogger(__name__)

def plot_dynamic_spectrum(spectrum_obj, ax=None, **kwargs):
    """
    Plots the 2D dynamic spectrum on a given axes object.
    If ax is None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    else:
        fig = ax.figure

    power_data = spectrum_obj.power
    if power_data.compressed().size > 0:
        # Use a percentile for the color limit to handle outliers gracefully
        vmax = np.percentile(power_data.compressed(), 99)
    else:
        vmax = 1.0

    im = ax.imshow(
        power_data,
        aspect='auto',
        origin='lower',
        extent=[
            spectrum_obj.times.min(), spectrum_obj.times.max(),
            spectrum_obj.frequencies.min(), spectrum_obj.frequencies.max()
        ],
        vmax=vmax,
        **kwargs
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    fig.colorbar(im, ax=ax, label="Power (arbitrary units)")
    
    # Set title if provided in kwargs
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    
    return fig, ax

def plot_pulse_window_diagnostic(spectrum_obj, title, save_path=None, **kwargs):
    """
    Generates a 2-panel diagnostic plot for a given time window of a dynamic spectrum.

    The top panel shows the 2D dynamic spectrum, and the bottom panel shows the
    frequency-averaged 1D time series.

    Args:
        spectrum_obj (core.DynamicSpectrum): A DynamicSpectrum object, typically
                                             containing a sliced portion of data.
        title (str): The title for the entire plot.
        save_path (str, optional): Path to save the figure to. Defaults to None.
    """
    log.info(f"Generating diagnostic plot: {title}")

    # Calculate the frequency-averaged time series from the input spectrum object
    time_series = np.ma.mean(spectrum_obj.power, axis=0)

    # Create the 2-panel figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=kwargs.get('figsize', (10, 8)),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(title, fontsize=16)

    # Panel 1: 2D Dynamic Spectrum
    plot_dynamic_spectrum(spectrum_obj, ax=ax1) # Use the existing function
    ax1.set_title("") # The main title is now the suptitle

    # Panel 2: 1D Time Series
    ax2.plot(spectrum_obj.times, time_series)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mean Power")
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(spectrum_obj.times.min(), spectrum_obj.times.max())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.info(f"Diagnostic plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save diagnostic plot to {save_path}: {e}")

    plt.show()
    plt.close(fig)


def plot_acf(acf_obj, fit_result=None, **kwargs):
    """
    Plots an ACF and optionally its Lorentzian fit.

    Args:
        acf_obj (ACF): The ACF object to plot.
        fit_result (lmfit.ModelResult, optional): The result from an lmfit run.
        **kwargs: Additional keyword arguments passed to plt.plot().
    """
    log.info("Generating ACF plot.")
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))
    
    ax.plot(acf_obj.lags, acf_obj.acf, 'k-', alpha=0.8, label="ACF Data", **kwargs)
    
    if fit_result:
        ax.plot(acf_obj.lags, fit_result.eval(x=acf_obj.lags), 'r--', label="Lorentzian Fit")
        gamma = fit_result.params['gamma1'].value
        ax.set_title(f"Decorrelation Bandwidth = {gamma*1000:.2f} kHz")
        #ax.set_xlim(-5 * gamma, 5 * gamma) # Auto-zoom to the feature
    
    ax.set_xlabel("Frequency Lag (MHz)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    
def plot_analysis_overview(
    analysis_results, 
    acf_results, 
    all_subband_fits, 
    all_powerlaw_fits,
    save_path=None,
    **kwargs
):
    """
    Generates a comprehensive, multi-panel summary plot of the entire 
    scintillation analysis, including goodness-of-fit diagnostics and
    dynamic ACF plot zoom.
    """
    log.info("Generating full analysis overview plot.")
    best_model_name = analysis_results['best_model']
    
    num_components = 1
    if '2c' in best_model_name: num_components = 2
    if '3c' in best_model_name: num_components = 3

    fig = plt.figure(figsize=kwargs.get('figsize', (12, 8 + 5 * num_components)))
    gs = fig.add_gridspec(3 + num_components, 2)
    fig.suptitle(f"Scintillation Analysis Summary (Best Model: {best_model_name})", fontsize=16)

    # --- Dynamically determine plot limits for the ACF plot ---
    max_bw = 0.0
    for component_data in analysis_results['components'].values():
        measurements = component_data.get('subband_measurements', [])
        for m in measurements:
            bw = m.get('bw')
            if bw is not None and bw > max_bw:
                max_bw = bw
    
    # Calculate the desired plot limit based on 3 * FWHM of the widest component
    desired_xlim = max_bw * 10 if max_bw > 0 else 1.0

    # Find the maximum possible lag from the data to avoid excessive empty space.
    max_available_lag = 0
    if acf_results['subband_lags_mhz'] and any(len(l) > 0 for l in acf_results['subband_lags_mhz']):
        max_available_lag = max(np.max(np.abs(lags)) for lags in acf_results['subband_lags_mhz'] if len(lags) > 0)

    # The final limit is the smaller of the desired zoom or the available data range.
    # This prevents zooming out too far and creating empty space.
    final_xlim = min(desired_xlim, max_available_lag) if max_available_lag > 0 else desired_xlim
    
    # Use a sensible default if the calculated limit is tiny
    if final_xlim < 0.1: final_xlim = 1.0

    # --- Panel 1: Stacked Sub-band ACFs with Fits ---
    #ax_acf = fig.add_subplot(gs[0:2, 0])
    #cmap = plt.get_cmap('plasma')
    #num_subbands = len(acf_results['subband_acfs'])
    #for i in range(num_subbands):
    #    offset = i * 1.5
    #    rgba = cmap(i*0.8 / (num_subbands - 1)) if num_subbands > 1 else cmap(0.5)
    #    lags = acf_results['subband_lags_mhz'][i]
    #    acf = acf_results['subband_acfs'][i]
    #    peak_val = np.max(acf)
    #    acf_normalized = acf / peak_val if peak_val > 0 else acf
    #    ax_acf.plot(lags, acf_normalized + offset, color=rgba)
    #    
    #    fit_obj = all_subband_fits[i].get(best_model_name)
    #    if fit_obj and fit_obj.success:
    #        fit_normalized = fit_obj.eval(x=lags) / peak_val
    #        ax_acf.plot(lags, fit_normalized + offset, 'k--', alpha=0.7, label='Best Fit' if i == 0 else "")
            
    # --- Panel 1: Stacked Sub-band ACFs with Fits ---
    ax_acf = fig.add_subplot(gs[0:2, 0])
    cmap = plt.get_cmap('plasma')
    num_subbands = len(acf_results['subband_acfs'])
    
    for i in range(num_subbands):
        offset = i * 1.5 # The vertical offset for stacking
        rgba = cmap(i*0.8 / (num_subbands - 1)) if num_subbands > 1 else cmap(0.5)
        
        lags = acf_results['subband_lags_mhz'][i]
        acf = acf_results['subband_acfs'][i]
        
        ### FIX: Normalize by the peak of the feature, not the noise spike ###
        
        # 1. Create a mask to exclude the zero-lag point
        plot_mask = (lags != 0)
        
        # 2. Find the peak value of the ACF *excluding* the zero-lag spike
        if np.any(acf[plot_mask]):
            peak_val = np.max(acf[plot_mask])
        else:
            peak_val = 1.0 # Fallback
            
        # 3. Normalize the entire ACF by this new peak value
        acf_normalized = acf / peak_val if peak_val > 0 else acf

        # 4. Plot the data using the mask
        ax_acf.plot(lags[plot_mask], acf_normalized[plot_mask] + offset, color=rgba)
        
        # 5. Normalize the fit by the same value for consistency
        fit_obj = all_subband_fits[i].get(best_model_name)
        if fit_obj and fit_obj.success:
            fit_lags = fit_obj.userkws['x']
            # Ensure the fit is normalized by the same peak_val as the data
            fit_normalized = fit_obj.best_fit / peak_val
            ax_acf.plot(fit_lags, fit_normalized + offset, 'k--', alpha=0.7, label='Best Fit' if i == 0 else "")
            
    ax_acf.set_yticks([(i * 1.5) for i in range(num_subbands)])
    ax_acf.set_yticklabels([f"{cf:.1f}" for cf in acf_results['subband_center_freqs_mhz']])
    ax_acf.set_title("Normalized Sub-band ACFs & Best Fit")
    ax_acf.set_xlabel("Frequency Lag (MHz)")
    ax_acf.set_ylabel("Center Freq. (MHz)")
    if num_subbands > 0: ax_acf.legend()
    # Apply the robustly calculated x-limit
    #ax_acf.set_xlim(-final_xlim, final_xlim)
    ax_acf.set_xlim(-10, 10)

    # ---- Panel 2 :  BIC model comparison ---------------------------------
    ax_bic = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------------------
    # 1) accumulate BIC totals and counts
    # ------------------------------------------------------------------
    bic_sum   = defaultdict(float)
    bic_count = defaultdict(int)

    for band in all_subband_fits:
        for name, fit in band.items():
            if fit and fit.success:
                bic_sum[name]   += fit.bic
                bic_count[name] += 1

    n_bands = len(all_subband_fits)

    # ------------------------------------------------------------------
    # 2) baseline names in the order you’d like to show
    # ------------------------------------------------------------------
    base_keys = [
        'fit_1c_lor', 'fit_2c_lor', 'fit_3c_lor',
        'fit_1c_gauss', 'fit_2c_gauss', 'fit_3c_gauss',
        'fit_2c_mixed', 'fit_2c_unresolved'
    ]

    labels, delta_bic = [], []
    best_total = np.inf

    for core in base_keys:                              # outer loop
        for prefix in ['', 'sn_tpl_', 'sn_', 'tpl_']:   # try each variant
            key = f'{prefix}{core}'
            if bic_count[key] == n_bands:               # keeps only complete models
                total = bic_sum[key]
                labels.append(key)
                delta_bic.append(total)                 # store raw; Δ later
                best_total = min(best_total, total)
                break                                   # stop at first valid variant

    # convert totals → ΔBIC relative to the best
    delta_bic = np.array(delta_bic) - best_total

    # ------------------------------------------------------------------
    # 3) plot
    # ------------------------------------------------------------------
    ax_bic.clear()
    ax_bic.barh(labels, delta_bic, color='skyblue')
    ax_bic.set_xlabel(r'$\Delta$BIC  (relative to best model)')
    ax_bic.set_title('Model Comparison (complete fits only)')
    ax_bic.invert_yaxis()
    ax_bic.grid(True, axis='x', alpha=0.3)

    # annotate exact ΔBIC on the bars (optional)
    for y, dx in enumerate(delta_bic):
        ax_bic.text(dx + 0.5, y, f'{dx:,.0f}', va='center')

    # --- Panel 3: Modulation Indices vs. Frequency ---
    ax_mod = fig.add_subplot(gs[1, 1])
    for name, component_data in analysis_results['components'].items():
        measurements = component_data.get('subband_measurements', [])
        if not measurements: continue
        freqs = np.array([m.get('freq_mhz') for m in measurements])
        mods = np.array([m.get('mod') for m in measurements])
        mod_errs = np.array([m.get('mod_err', 0) for m in measurements])
        ax_mod.errorbar(freqs, mods, yerr=mod_errs, fmt='o', capsize=5, label=name.replace('_', ' ').title())

    ax_mod.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax_mod.set_xlabel("Frequency (MHz)")
    ax_mod.set_ylabel("Modulation Index (m)")
    ax_mod.set_title("Modulation Index vs. Frequency")
    ax_mod.set_ylim(0,3)
    ax_mod.legend()
    ax_mod.grid(True, alpha=0.2)

    # --- Panel 4: Goodness-of-Fit (Reduced Chi-Squared) ---
    ax_gof = fig.add_subplot(gs[2, :])
    for name, component_data in analysis_results['components'].items():
        measurements = component_data.get('subband_measurements', [])
        if not measurements: continue
        
        if name == 'component_1' or name == 'scint_scale':
            freqs = [m.get('freq_mhz') for m in measurements]
            redchi_vals = [m.get('gof', {}).get('redchi') for m in measurements]
            ax_gof.plot(freqs, redchi_vals, 'o-', label=f'Best Model ({best_model_name})')
    
    ax_gof.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Ideal Fit ($\chi^2_\\nu=1$)')
    ax_gof.set_title("Goodness-of-Fit Diagnostic")
    ax_gof.set_xlabel("Frequency (MHz)")
    ax_gof.set_ylabel("Reduced Chi-Squared ($\\chi^2_\\nu$)")
    #ax_gof.set_yscale('log')
    ax_gof.legend()
    ax_gof.grid(True, alpha=0.3)

    # --- Panels 5+: Power-Law Fits for Each Component ---
    for comp_idx, (name, component_data) in enumerate(analysis_results['components'].items()):
        ax_plaw = fig.add_subplot(gs[3 + comp_idx, :])
        
        measurements = component_data.get('subband_measurements', [])
        if not measurements:
            ax_plaw.text(0.5, 0.5, "Power-law fit failed or not performed.", ha='center', va='center')
            ax_plaw.set_title(f"Power-Law Fit: {name.replace('_', ' ').title()}")
            continue
            
        freqs = np.array([m.get('freq_mhz') for m in measurements])
        bws = np.array([m.get('bw') for m in measurements])
        fit_errs = np.array([m.get('bw_err', 0) for m in measurements])
        finite_errs = np.array([m.get('finite_err', 0) for m in measurements])
        total_errs = np.sqrt(np.nan_to_num(fit_errs)**2 + np.nan_to_num(finite_errs)**2)
        redchi_vals = np.array([m.get('gof', {}).get('redchi', 1.0) for m in measurements])
        
        sc = ax_plaw.scatter(freqs, bws, c=redchi_vals, cmap='coolwarm', vmin=0.5, vmax=2.0, zorder=10, label='Sub-band Measurements')
        ax_plaw.errorbar(freqs, bws, yerr=total_errs, fmt='none', ecolor='gray', capsize=5, zorder=5)
        fig.colorbar(sc, ax=ax_plaw, label='$\\chi^2_\\nu$ of ACF Fit')

        fit_output = all_powerlaw_fits.get(name)
        if fit_output:
            c, n = fit_output.beta
            freq_model = np.linspace(min(freqs), max(freqs), 100)
            scint_model = c * (freq_model ** n)
            ax_plaw.plot(freq_model, scint_model, 'k--', label=f'Power-Law Fit ($\\alpha={n:.2f}$)')

        interpretation_text = component_data.get('scaling_interpretation', '')
        ax_plaw.set_title(f"Power-Law Fit: {name.replace('_', ' ').title()}\n{interpretation_text}")
        ax_plaw.set_xlabel("Frequency (MHz)")
        ax_plaw.set_ylabel("Decorrelation BW (MHz)")
        ax_plaw.legend()
        ax_plaw.grid(True, alpha=0.2)
        ax_plaw.set_ylim(0, np.max(bws))
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Analysis overview plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save plot to {save_path}: {e}")
            
    plt.show()
    
def plot_noise_distribution(spectrum_obj, downsample_factor=8, save_path=None, **kwargs):
    """
    Calculates and plots the distribution of the noise in the time series.

    This function isolates the noise using sigma-clipping and compares its
    distribution to a Gaussian, which is a key assumption for many
    statistical analyses.

    Args:
        spectrum_obj (DynamicSpectrum): The dynamic spectrum object to analyze.
        downsample_factor (int): The factor by which to downsample the time series.
        save_path (str, optional): Path to save the figure to. Defaults to None.
        **kwargs: Additional keyword arguments passed to plt.hist().
    """
    log.info("Generating noise distribution plot.")
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))

    # 1. Get and downsample the time series profile
    prof = spectrum_obj.get_profile().compressed()
    if downsample_factor > 1:
        n = prof.size - (prof.size % downsample_factor)
        if n > 0:
            prof = prof[:n].reshape(-1, downsample_factor).mean(axis=1)

    # 2. Isolate the noise using sigma-clipping
    noise_data = sigma_clip(prof, sigma=3, maxiters=5, masked=True)

    # 3. Calculate robust statistics from the noise
    mu = np.ma.median(noise_data)
    sigma = np.ma.std(noise_data)
    
    # Get the unmasked noise values for histogramming
    noise_values = noise_data.compressed()

    # 4. Plot the normalized histogram of the noise
    ax.hist(noise_values, bins='auto', density=True, alpha=0.7,
            label='Noise Distribution', **kwargs)

    # 5. Overlay the best-fit Gaussian curve
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, 'r-', lw=2, label='Gaussian Fit')

    ax.set_title(f"Noise Distribution ($\\mu={mu:.2e}$, $\\sigma={sigma:.2e}$)")
    ax.set_xlabel("Flux (arbitrary units)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save the figure if a path is provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            log.info(f"Noise distribution plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save plot to {save_path}: {e}")

    plt.show()

def plot_intra_pulse_evolution(
    intra_pulse_results,
    on_pulse_profile,
    on_pulse_times,
    save_path=None,
    **kwargs
):
    """
    Plots the evolution of scintillation parameters across the burst profile.
    The left panel shows the ACFs as a 2D heatmap over time, and the right
    panels show the evolution of the fitted parameters.

    Args:
        intra_pulse_results (list): The output from analyze_intra_pulse_scintillation.
        on_pulse_profile (np.ndarray): The frequency-averaged time series of the burst.
        on_pulse_times (np.ndarray): The time axis for the on_pulse_profile.
        save_path (str, optional): Path to save the figure to. Defaults to None.
        **kwargs: Additional keyword arguments.
    """
    if not intra_pulse_results:
        log.warning("Intra-pulse results are empty. Skipping evolution plot.")
        return

    log.info("Generating intra-pulse evolution plot with 2D ACF heatmap.")

    # --- 1. Set up the Figure Layout ---
    fig = plt.figure(figsize=kwargs.get('figsize', (16, 10)))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    fig.suptitle("Intra-Pulse Scintillation Evolution", fontsize=16, y=0.98)

    ax_acf = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 1], sharex=ax_prof)
    ax_mod = fig.add_subplot(gs[2, 1], sharex=ax_prof)
    
    plt.setp(ax_prof.get_xticklabels(), visible=False)
    plt.setp(ax_bw.get_xticklabels(), visible=False)

    # --- 2. Prepare data for the 2D ACF Heatmap ---
    # Ensure all ACFs have the same length by taking the minimum length
    min_len = min(len(res['acf_data']) for res in intra_pulse_results)
    lags = intra_pulse_results[0]['acf_lags'][:min_len]
    
    # Create the 2D array for the image
    acf_image = np.array([res['acf_data'][:min_len] for res in intra_pulse_results])
    
    # Get the time extent for the y-axis
    times_axis = [res['time_s'] for res in intra_pulse_results]

    # --- 3. Plot the 2D ACF Heatmap (Left Side) ---
    # Use a symmetric log scale for color to handle peaks and troughs, with a linear range near zero
    norm = SymLogNorm(linthresh=0.05, vmin=np.min(acf_image), vmax=np.max(acf_image))
    
    im = ax_acf.imshow(
        acf_image,
        aspect='auto',
        origin='lower',
        extent=[lags.min(), lags.max(), times_axis[0], times_axis[-1]],
        cmap='plasma',
        norm=norm
    )
    fig.colorbar(im, ax=ax_acf, label="ACF Amplitude")
    
    # Overplot the fitted decorrelation bandwidth (HWHM) on the heatmap
    for res in intra_pulse_results:
        if res.get('fit_success', False):
            t = res['time_s']
            bw = res['bw']
            # Plot white lines to indicate the +/- HWHM of the fit
            ax_acf.plot([-bw, -bw], [t-0.00005, t+0.00005], color='w', lw=1.5, alpha=0.8)
            ax_acf.plot([bw, bw], [t-0.00005, t+0.00005], color='w', lw=1.5, alpha=0.8, label='Fit HWHM' if 'HWHM' not in ax_acf.get_legend_handles_labels()[1] else '')

    ax_acf.set_title("ACF vs. Time")
    ax_acf.set_xlabel("Frequency Lag (MHz)")
    ax_acf.set_ylabel("Time in Burst (s)")
    if any(res.get('fit_success', False) for res in intra_pulse_results):
        ax_acf.legend(loc='upper right')
    
    # --- 4. Plot the Parameter Evolution Panels (Right Side) ---
    times = np.array([res['time_s'] for res in intra_pulse_results])
    bws = np.array([res['bw'] for res in intra_pulse_results])
    bw_errs = np.array([res.get('bw_err', 0) for res in intra_pulse_results])
    mods = np.array([res['mod'] for res in intra_pulse_results])
    mod_errs = np.array([res.get('mod_err', 0) for res in intra_pulse_results])

    ax_prof.plot(on_pulse_times, on_pulse_profile, color='k', alpha=0.8)
    ax_prof.set_ylabel("Mean Power")
    ax_prof.set_title("Burst Profile")
    
    ax_bw.errorbar(times, bws, yerr=bw_errs, fmt='o', capsize=5, color='C0')
    ax_bw.set_ylabel("Decorrelation BW (MHz)")
    ax_bw.set_ylim(bottom=0)

    ax_mod.errorbar(times, mods, yerr=mod_errs, fmt='s', capsize=5, color='C1')
    ax_mod.set_xlabel("Time (s)")
    ax_mod.set_ylabel("Modulation Index (m)")
    ax_mod.set_ylim(0, max(1.2, np.nanmax(mods) * 1.1) if np.any(~np.isnan(mods)) else 1.2)

    for ax in [ax_prof, ax_bw, ax_mod]:
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            log.info(f"Intra-pulse evolution plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save plot to {save_path}: {e}")
            
    plt.show()
    
def plot_intra_pulse_evolution_stackacfs(
    intra_pulse_results,
    on_pulse_profile,
    on_pulse_times,
    save_path=None,
    **kwargs
):
    """
    Plots the evolution of scintillation parameters across the burst profile,
    including a panel showing the individual ACFs and their fits.

    Args:
        intra_pulse_results (list): The output from analyze_intra_pulse_scintillation.
        on_pulse_profile (np.ndarray): The frequency-averaged time series of the burst.
        on_pulse_times (np.ndarray): The time axis for the on_pulse_profile.
        save_path (str, optional): Path to save the figure to. Defaults to None.
        **kwargs: Additional keyword arguments.
    """
    if not intra_pulse_results:
        log.warning("Intra-pulse results are empty. Skipping evolution plot.")
        return

    log.info("Generating intra-pulse evolution plot with ACF panel.")

    # --- 1. Set up the Figure Layout ---
    fig = plt.figure(figsize=kwargs.get('figsize', (16, 10)))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    fig.suptitle("Intra-Pulse Scintillation Evolution", fontsize=16, y=0.98)

    ax_acf = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 1], sharex=ax_prof)
    ax_mod = fig.add_subplot(gs[2, 1], sharex=ax_prof)
    
    plt.setp(ax_prof.get_xticklabels(), visible=False)
    plt.setp(ax_bw.get_xticklabels(), visible=False)

    # --- 2. Plot the new ACF Panel (Left Side) ---
    cmap = plt.get_cmap('plasma')
    num_results = len(intra_pulse_results)
    # find max acf value for offsetting, handle case where all fits might fail
    successful_results = [res for res in intra_pulse_results if res['fit_success']]
    if not successful_results:
        log.warning("No successful fits in intra-pulse analysis, ACF panel may be incomplete.")
        max_acf_val = 1.0 # Default offset
    else:
        max_acf_val = max(res['acf_data'].max() for res in successful_results)
    
    for i, res in enumerate(intra_pulse_results):
        offset = i * max_acf_val * 1.1 
        color = cmap(i / num_results)
        
        ax_acf.plot(res['acf_lags'], res['acf_data']*1.5 + offset, color=color, alpha=0.9)
        
        if res['fit_success']:
            ### FIX: Use 'acf_fit_lags' which has the same dimension as 'acf_fit_best'
            ax_acf.plot(res['acf_fit_lags'], res['acf_fit_best']*1.5 + offset, 'k--', alpha=0.7, lw=1.5)

    ax_acf.set_title("ACF Evolution (Earliest to Latest)")
    ax_acf.set_xlabel("Frequency Lag (MHz)")
    ax_acf.set_ylabel("Stacked & Offset ACFs")
    ax_acf.grid(True, linestyle=':', alpha=0.5)
    ax_acf.set_yticks([])
    
    avg_bw_list = [res['bw'] for res in successful_results if 'bw' in res and not np.isnan(res['bw'])]
    if avg_bw_list:
        avg_bw = np.nanmean(avg_bw_list)
        ax_acf.set_xlim(-8 * avg_bw, 8 * avg_bw)

    # --- 3. Plot the Parameter Evolution Panels (Right Side) ---
    # (This section remains unchanged)
    times = np.array([res['time_s'] for res in intra_pulse_results])
    bws = np.array([res['bw'] for res in intra_pulse_results])
    bw_errs = np.array([res.get('bw_err', 0) for res in intra_pulse_results])
    mods = np.array([res['mod'] for res in intra_pulse_results])
    mod_errs = np.array([res.get('mod_err', 0) for res in intra_pulse_results])

    ax_prof.plot(on_pulse_times, on_pulse_profile, color='k', alpha=0.8)
    ax_prof.set_ylabel("Mean Power")
    ax_prof.set_title("Burst Profile & Measurement Times")
    for t in times:
        ax_prof.axvline(t, color='red', linestyle='--', alpha=0.4, lw=1)
    
    ax_bw.errorbar(times, bws, yerr=bw_errs, fmt='o', capsize=5, color='C0')
    ax_bw.set_ylabel("Decorrelation BW (MHz)")
    ax_bw.set_ylim(0, 5)

    ax_mod.errorbar(times, mods, yerr=mod_errs, fmt='s', capsize=5, color='C1')
    ax_mod.set_xlabel("Time (s)")
    ax_mod.set_ylabel("Modulation Index (m)")
    ax_mod.set_ylim(0, min(1.2, np.nanmax(mods) * 1.1) if np.any(~np.isnan(mods)) else 1.2)

    for ax in [ax_prof, ax_bw, ax_mod]:
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            log.info(f"Intra-pulse evolution plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save plot to {save_path}: {e}")
            
    plt.show()
    
def plot_baseline_fit(
    off_pulse_spectrum,
    fitted_baseline,
    frequencies,
    poly_order,
    save_path=None,
    **kwargs
):
    """
    Generates a diagnostic plot showing the off-pulse spectrum, the fitted
    polynomial baseline, and the residuals after subtraction.

    Args:
        off_pulse_spectrum (np.ma.MaskedArray): The original 1D off-pulse spectrum.
        fitted_baseline (np.ndarray): The 1D array of the fitted baseline model.
        frequencies (np.ndarray): The frequency axis in MHz.
        poly_order (int): The order of the polynomial that was fit.
        save_path (str, optional): Path to save the figure to. Defaults to None.
    """
    log.info("Generating baseline fit diagnostic plot.")
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

    # 1. Plot the original off-pulse data (only unmasked points)
    valid_mask = ~off_pulse_spectrum.mask
    ax.plot(
        frequencies[valid_mask],
        off_pulse_spectrum.compressed(),
        '.',
        color='C0',
        alpha=0.5,
        label="Off-Pulse Spectrum Data"
    )

    # 2. Plot the fitted polynomial baseline
    ax.plot(
        frequencies,
        fitted_baseline,
        'r--',
        lw=2,
        label=f"Polynomial Fit (order={poly_order})"
    )

    # 3. Plot the residuals after subtraction
    residuals = off_pulse_spectrum - fitted_baseline
    ax.plot(
        frequencies[valid_mask],
        residuals.compressed(),
        color='gray',
        alpha=0.8,
        label="Residuals (Data - Fit)"
    )
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='k', linestyle=':', alpha=0.6)

    ax.set_title("Baseline Subtraction Diagnostic")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (arbitrary units)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.info(f"Baseline diagnostic plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Failed to save baseline plot to {save_path}: {e}")
    
    plt.show()
    plt.close(fig)