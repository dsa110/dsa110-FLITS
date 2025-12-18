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
    ax_acf.set_xlim(-final_xlim, final_xlim)
    #ax_acf.set_xlim(-10, 10)

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

# ==============================================================================
# Publication-Quality ACF Plotting (Added from notebook refactoring)
# ==============================================================================

def format_error(value, error):
    """
    Format a value and its error using compact notation X(err).
    
    Examples: 17.3 ± 1.5 → "17.3(15)"
              0.3346 ± 0.0429 → "0.335(43)"
    
    Parameters
    ----------
    value : float
        The central value
    error : float or None
        The uncertainty
    
    Returns
    -------
    str
        Formatted string with compact error notation
    """
    if error is None or not np.isfinite(error) or error <= 0:
        return f"{value:.3f}"
    
    # Determine number of decimal places based on error magnitude
    decimals = -int(np.floor(np.log10(error))) + 1
    decimals = max(0, decimals)
    
    value_str = f"{value:.{decimals}f}"
    error_int = int(round(error * (10**decimals)))
    
    return f"{value_str}({error_int})"


def plot_publication_acf(
    acf_obj,
    best_fit_curve,
    component_curves,
    params,
    fit_range_mhz,
    redchi=None,
    zoom_lag_range_mhz=(-20, 20),
    center_freq_ghz=1.4,
    save_path=None
):
    """
    Generate a publication-quality 3-panel ACF plot.
    
    Creates a comprehensive visualization with:
    - Panel a: Full ACF range with fit and zoom indicator
    - Panel b: Zoomed view showing component breakdown and parameters
    - Panel c: Fit residuals with chi-squared statistic
    
    Parameters
    ----------
    acf_obj : ACF
        ACF object containing lags, acf, and optionally err arrays
    best_fit_curve : np.ndarray
        Best-fit composite model evaluated on acf_obj.lags
    component_curves : dict
        Dictionary of component curves, e.g. {'l_1_': array, 'g_2_': array, 'c_': array}
    params : dict
        Best-fit parameters, format: {name: {'value': float, 'stderr': float}}
    fit_range_mhz : list or tuple
        [min_lag, max_lag] range used for fitting
    redchi : float, optional
        Reduced chi-squared of the fit
    zoom_lag_range_mhz : tuple, optional
        Lag range for zoomed panel b. Default is (-20, 20) MHz
    center_freq_ghz : float, optional
        Center frequency for plot title. Default is 1.4 GHz
    save_path : str or Path, optional
        If provided, save the figure to this path
    """
    plot_mask = (acf_obj.lags != 0)
    
    # Convert to modulation index space (m^2 → m)
    m_data = np.sqrt(np.maximum(0, acf_obj.acf))
    m_fit_curve = np.sqrt(np.maximum(0, best_fit_curve))
    m_component_curves = {
        key: np.sqrt(np.maximum(0, curve))
        for key, curve in component_curves.items()
    }
    
    # Propagate errors: σ_m = σ_{m^2} / (2*m)
    m_err = None
    if acf_obj.err is not None:
        m_err = np.divide(acf_obj.err, 2 * m_data, out=np.zeros_like(m_data), where=(m_data != 0))
    
    residuals = m_data - m_fit_curve
    full_lag_range_mhz = (acf_obj.lags.min(), acf_obj.lags.max())
    
    # Generate legend label
    shape_component_keys = sorted([k for k in component_curves if 'c_' not in k])
    label_parts = []
    model_map = {'l_': 'L', 'g_': 'G', 'lg_': 'L_{\\mathrm{gen}}', 'p_': 'P'}
    
    for prefix in shape_component_keys:
        base_prefix = prefix.split('_')[0] + '_'
        index = prefix.split('_')[1]
        symbol = model_map.get(base_prefix, '?')
        label_parts.append(f"$\\mathcal{{{symbol}}}_{{{index}}}$")
    
    if 'c_' in component_curves:
        label_parts.append("$c$")
    
    composite_label = "Fit: " + " + ".join(label_parts)
    
    # Create figure
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=False,
        gridspec_kw={'height_ratios': [2, 3, 1.5]},
        constrained_layout=True
    )
    
    colors = plt.get_cmap('plasma')(np.linspace(0.25, 0.75, 6))
    
    # Panel a: Wide View
    if acf_obj.err is not None:
        ax_a.errorbar(acf_obj.lags[plot_mask], acf_obj.acf[plot_mask],
                     yerr=acf_obj.err[plot_mask], fmt='none', capsize=2,
                     ecolor='lightgrey', alpha=1)
    ax_a.plot(acf_obj.lags[plot_mask], acf_obj.acf[plot_mask],
             color=colors[0], alpha=0.6, lw=1)
    ax_a.plot(acf_obj.lags, best_fit_curve, 'k-', lw=1.5, label=composite_label)
    ax_a.axvspan(zoom_lag_range_mhz[0], zoom_lag_range_mhz[1],
                color=colors[1], alpha=0.3, hatch='//', zorder=-1,
                label='Zoom In ($\\mathbf{b}$)')
    
    ax_a.set_xlim(acf_obj.lags.min(), acf_obj.lags.max())
    ax_a.set_ylabel("ACF$~$ ($m^{2}$)")
    ax_a.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8,
               facecolor='white', edgecolor='black', borderpad=0.3, fontsize='small')
    ax_a.text(0.02, 0.9, "a", transform=ax_a.transAxes, fontsize=20,
             fontweight='bold', va='top')
    
    mask_a_zoom = (acf_obj.lags >= full_lag_range_mhz[0]) & (acf_obj.lags <= full_lag_range_mhz[1])
    if np.any(mask_a_zoom):
        min_val_a = np.min(acf_obj.acf[mask_a_zoom]) / 2
        max_val_a = np.max(acf_obj.acf[mask_a_zoom]) / 2
        padding_a = (max_val_a - min_val_a) * 0.1
        ax_a.set_ylim(min_val_a - padding_a, max_val_a + padding_a)
    
    # Panel b: Zoomed View
    if acf_obj.err is not None:
        ax_b.errorbar(acf_obj.lags[plot_mask], acf_obj.acf[plot_mask],
                     yerr=acf_obj.err[plot_mask], fmt='none', capsize=2,
                     ecolor='lightgrey', alpha=1)
    ax_b.plot(acf_obj.lags[plot_mask], acf_obj.acf[plot_mask],
             color=colors[2], alpha=0.6, lw=1)
    ax_b.plot(acf_obj.lags, best_fit_curve, 'k-', lw=2, label=composite_label)
    
    log.info("--- Fitted Decorrelation Bandwidths ---")
    for i, prefix in enumerate(shape_component_keys):
        color = colors[i]
        width_param_name = None
        for pname in params:
            if pname.startswith(prefix) and ('gamma' in pname or 'sigma' in pname):
                width_param_name = pname
                break
        
        if width_param_name and width_param_name in params:
            param_info = params[width_param_name]
            gamma_val = param_info['value']
            gamma_err = param_info['stderr']
            
            gamma_val_khz = gamma_val * 1000
            gamma_err_khz = gamma_err * 1000 if gamma_err is not None else 0.0
            label_text = f"$\\gamma_{i+1}$ = {format_error(gamma_val_khz, gamma_err_khz)} kHz"
            
            ax_b.plot(acf_obj.lags, m_component_curves[prefix],
                     color=color, lw=2.0, linestyle='--', label=label_text)
            log.info(f"  {label_text}")
    
    ax_b.axvspan(fit_range_mhz[0], fit_range_mhz[1], color=colors[3],
                alpha=0.3, hatch='//', zorder=-1, label='$\\chi^2_r$ Fit Range')
    ax_b.set_xlim(fit_range_mhz[0]-5, fit_range_mhz[1]+5)
    ax_b.set_ylabel("ACF$~$ ($m^{2}$)")
    ax_b.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8,
               facecolor='white', edgecolor='black', borderpad=0.3, fontsize='small')
    ax_b.text(0.02, 0.95, "b", transform=ax_b.transAxes, fontsize=20,
             fontweight='bold', va='top')
    
    mask_b_zoom = (acf_obj.lags >= zoom_lag_range_mhz[0]) & (acf_obj.lags <= zoom_lag_range_mhz[1])
    if np.any(mask_b_zoom):
        min_val = np.min(acf_obj.acf[mask_b_zoom & plot_mask])
        max_val = np.max(acf_obj.acf[mask_b_zoom & plot_mask])
        padding = (max_val - min_val) * 0.1
        ax_b.set_ylim(min_val - padding, max_val + padding)
    
    # Panel c: Residuals
    ax_c.plot(acf_obj.lags[plot_mask], residuals[plot_mask],
             color=colors[4], alpha=0.8, lw=1)
    ax_c.axhline(0, color='k', linestyle='--', lw=1)
    ax_c.set_xlim(zoom_lag_range_mhz)
    
    mask_c_zoom = (acf_obj.lags >= zoom_lag_range_mhz[0]) & (acf_obj.lags <= zoom_lag_range_mhz[1])
    if np.any(mask_c_zoom):
        max_abs_resid = np.max(np.abs(residuals[mask_c_zoom & plot_mask]))
        ax_c.set_ylim(-max_abs_resid * 1.2, max_abs_resid * 1.2)
    
    if redchi is not None:
        ax_c.text(0.825, 0.9, f"$\\chi_r^2$= {redchi:.2f}",
                 transform=ax_c.transAxes, ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    ax_c.set_xlabel("Frequency lag (MHz)")
    ax_c.set_ylabel("Residuals")
    ax_c.text(0.02, 0.95, "c", transform=ax_c.transAxes, fontsize=20,
             fontweight='bold', va='top')
    
    plt.suptitle(f"$\\nu_{{c}}$ = {center_freq_ghz:.2f} GHz",
                fontsize=20, y=1.05, x=0.55)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        log.info(f"Publication ACF plot saved to: {save_path}")
    
    plt.show()


# ==============================================================================
# 2D Scintillation Fitting Plots
# ==============================================================================

def plot_2d_fit_overview(
    acf_results: dict,
    fit_result,  # Scintillation2DResult from fitting_2d
    fit_range_mhz: float = 25.0,
    save_path: str = None,
    figsize: tuple = (14, 10),
):
    """
    Generate comprehensive 2D fit overview plot.
    
    Creates a multi-panel figure showing:
    - Panel (a): All ACFs with global fit overlaid
    - Panel (b): γ vs frequency with power-law fit
    - Panel (c): Residuals by sub-band
    - Panel (d): Corner plot of γ₀ and α (if MCMC available)
    
    Parameters
    ----------
    acf_results : dict
        Dictionary from ScintillationAnalysis.acf_results
    fit_result : Scintillation2DResult
        Result from fit_2d_scintillation()
    fit_range_mhz : float
        Fit range used (for visual indicator)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from .fitting_2d import lorentzian_acf, gen_lorentzian_acf, gaussian_acf
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax_acfs = fig.add_subplot(gs[0, 0])
    ax_scaling = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_corner = fig.add_subplot(gs[1, 1])
    
    # Extract data
    lags_list = acf_results['subband_lags_mhz']
    acfs_list = acf_results['subband_acfs']
    errs_list = acf_results.get('subband_acfs_err', [None] * len(lags_list))
    center_freqs = np.array(acf_results['subband_center_freqs_mhz'])
    n_subbands = len(center_freqs)
    
    # Color map for sub-bands
    cmap = plt.cm.viridis
    colors = [cmap(i / (n_subbands - 1)) for i in range(n_subbands)]
    
    # Panel (a): ACFs with fits
    for i, (lags, acf, nu_c, color) in enumerate(
        zip(lags_list, acfs_list, center_freqs, colors)
    ):
        mask = np.abs(lags) <= fit_range_mhz * 2
        
        # Data
        ax_acfs.plot(lags[mask], acf[mask], 'o', color=color, 
                     alpha=0.5, ms=3, label=f'{nu_c:.0f} MHz')
        
        # Model
        gamma = fit_result.gamma_0 * (nu_c / fit_result.nu_ref) ** fit_result.alpha
        m = fit_result.m_0
        
        lags_fine = np.linspace(lags[mask].min(), lags[mask].max(), 200)
        if fit_result.model_type == 'lorentzian':
            model = lorentzian_acf(lags_fine, gamma, m)
        elif fit_result.model_type == 'gen_lorentzian':
            eta = fit_result.params.get('eta', 2.0)
            if hasattr(eta, 'value'):
                eta = eta.value
            model = gen_lorentzian_acf(lags_fine, gamma, m, eta)
        else:
            model = gaussian_acf(lags_fine, gamma, m)
        
        ax_acfs.plot(lags_fine, model, '-', color=color, lw=2)
    
    ax_acfs.axvline(-fit_range_mhz, color='gray', ls='--', alpha=0.5)
    ax_acfs.axvline(fit_range_mhz, color='gray', ls='--', alpha=0.5)
    ax_acfs.set_xlabel('Frequency lag (MHz)')
    ax_acfs.set_ylabel('ACF')
    ax_acfs.set_title('(a) ACFs with global 2D fit')
    ax_acfs.legend(fontsize=8, loc='upper right')
    ax_acfs.set_xlim(-fit_range_mhz * 1.5, fit_range_mhz * 1.5)
    
    # Panel (b): γ vs frequency scaling
    ax_scaling.errorbar(
        center_freqs, fit_result.subband_gamma,
        yerr=fit_result.subband_gamma_err,
        fmt='o', color='C0', ms=8, capsize=3,
        label='2D fit'
    )
    
    # Power-law fit line
    nu_fine = np.linspace(center_freqs.min() * 0.95, center_freqs.max() * 1.05, 100)
    gamma_fit = fit_result.gamma_0 * (nu_fine / fit_result.nu_ref) ** fit_result.alpha
    ax_scaling.plot(nu_fine, gamma_fit, 'C1-', lw=2,
                    label=f'$\\gamma \\propto \\nu^{{{fit_result.alpha:.2f}}}$')
    
    # Add reference lines for known scalings
    gamma_ref = fit_result.gamma_0
    gamma_thin = gamma_ref * (nu_fine / fit_result.nu_ref) ** 4.0
    gamma_kolm = gamma_ref * (nu_fine / fit_result.nu_ref) ** 4.4
    ax_scaling.plot(nu_fine, gamma_thin, 'k--', alpha=0.3, label='Thin screen (α=4)')
    ax_scaling.plot(nu_fine, gamma_kolm, 'k:', alpha=0.3, label='Kolmogorov (α=4.4)')
    
    ax_scaling.set_xlabel('Frequency (MHz)')
    ax_scaling.set_ylabel('Scintillation bandwidth γ (MHz)')
    ax_scaling.set_title('(b) Frequency scaling')
    ax_scaling.legend(fontsize=8)
    ax_scaling.set_xscale('log')
    ax_scaling.set_yscale('log')
    
    # Panel (c): Residuals
    all_residuals = []
    all_freqs = []
    for i, (lags, acf, err, nu_c) in enumerate(
        zip(lags_list, acfs_list, errs_list, center_freqs)
    ):
        mask = np.abs(lags) <= fit_range_mhz
        gamma = fit_result.gamma_0 * (nu_c / fit_result.nu_ref) ** fit_result.alpha
        m = fit_result.m_0
        
        if fit_result.model_type == 'lorentzian':
            model = lorentzian_acf(lags[mask], gamma, m)
        elif fit_result.model_type == 'gen_lorentzian':
            eta = fit_result.params.get('eta', 2.0)
            if hasattr(eta, 'value'):
                eta = eta.value
            model = gen_lorentzian_acf(lags[mask], gamma, m, eta)
        else:
            model = gaussian_acf(lags[mask], gamma, m)
        
        if err is not None:
            resid = (acf[mask] - model) / err[mask]
        else:
            resid = acf[mask] - model
        
        all_residuals.extend(resid)
        all_freqs.extend([nu_c] * len(resid))
    
    ax_resid.scatter(all_freqs, all_residuals, alpha=0.3, s=10, c='C0')
    ax_resid.axhline(0, color='k', ls='--')
    ax_resid.axhline(2, color='r', ls=':', alpha=0.5)
    ax_resid.axhline(-2, color='r', ls=':', alpha=0.5)
    ax_resid.set_xlabel('Frequency (MHz)')
    ax_resid.set_ylabel('Normalized residuals')
    ax_resid.set_title(f'(c) Fit residuals (χ²_red = {fit_result.redchi:.2f})')
    
    # Panel (d): Parameter summary
    ax_corner.axis('off')
    
    summary_text = (
        f"2D Scintillation Fit Results\n"
        f"{'─' * 35}\n"
        f"Model: {fit_result.model_type.title()}\n"
        f"Reference freq: {fit_result.nu_ref:.0f} MHz\n\n"
        f"γ₀ = {fit_result.gamma_0:.3f} ± {fit_result.gamma_0_err:.3f} MHz\n"
        f"α  = {fit_result.alpha:.3f} ± {fit_result.alpha_err:.3f}\n"
        f"m₀ = {fit_result.m_0:.3f} ± {fit_result.m_0_err:.3f}\n\n"
        f"χ²_red = {fit_result.redchi:.2f}\n"
        f"N_free = {fit_result.nfree}\n\n"
        f"{'─' * 35}\n"
        f"Physical interpretation:\n"
    )
    
    # Interpret alpha
    if 3.5 <= fit_result.alpha <= 4.5:
        summary_text += f"α ≈ 4 → Consistent with thin screen\n"
    elif 4.0 <= fit_result.alpha <= 5.0:
        summary_text += f"α ≈ 4.4 → Kolmogorov turbulence\n"
    elif fit_result.alpha < 3.5:
        summary_text += f"α < 3.5 → Extended scattering medium\n"
    else:
        summary_text += f"α > 5 → Steep spectrum (unusual)\n"
    
    ax_corner.text(0.1, 0.95, summary_text, transform=ax_corner.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('2D Scintillation Analysis', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        log.info(f"2D fit overview saved to: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_gamma_scaling(
    fit_result,  # Scintillation2DResult
    gamma_1d: np.ndarray = None,
    gamma_1d_err: np.ndarray = None,
    save_path: str = None,
    figsize: tuple = (8, 6),
):
    """
    Publication-quality γ vs ν scaling plot.
    
    Parameters
    ----------
    fit_result : Scintillation2DResult
        Result from 2D fit
    gamma_1d : np.ndarray, optional
        1D fit results for comparison
    gamma_1d_err : np.ndarray, optional
        Errors on 1D fits
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    center_freqs = fit_result.center_freqs
    
    # Plot 2D fit results
    ax.errorbar(
        center_freqs, fit_result.subband_gamma,
        yerr=fit_result.subband_gamma_err,
        fmt='s', color='C0', ms=10, capsize=4, capthick=2,
        label='2D global fit', zorder=3
    )
    
    # Plot 1D comparison if provided
    if gamma_1d is not None:
        if gamma_1d_err is not None:
            ax.errorbar(
                center_freqs, gamma_1d, yerr=gamma_1d_err,
                fmt='o', color='C1', ms=8, capsize=3, alpha=0.7,
                label='1D sub-band fits', zorder=2
            )
        else:
            ax.plot(center_freqs, gamma_1d, 'o', color='C1', ms=8, 
                    alpha=0.7, label='1D sub-band fits', zorder=2)
    
    # Power-law fit line
    nu_fine = np.linspace(center_freqs.min() * 0.9, center_freqs.max() * 1.1, 100)
    gamma_fit = fit_result.gamma_0 * (nu_fine / fit_result.nu_ref) ** fit_result.alpha
    
    # Uncertainty band
    gamma_upper, gamma_lower = [], []
    for nu in nu_fine:
        g, g_err = fit_result.gamma_at_freq(nu)
        gamma_upper.append(g + g_err)
        gamma_lower.append(g - g_err)
    
    ax.fill_between(nu_fine, gamma_lower, gamma_upper, 
                    color='C0', alpha=0.2, zorder=1)
    ax.plot(nu_fine, gamma_fit, 'C0-', lw=2, zorder=2,
            label=f'$\\gamma = \\gamma_0 (\\nu/\\nu_{{ref}})^\\alpha$')
    
    # Annotation
    text = (
        f"$\\gamma_0 = {fit_result.gamma_0:.2f} \\pm {fit_result.gamma_0_err:.2f}$ MHz\n"
        f"$\\alpha = {fit_result.alpha:.2f} \\pm {fit_result.alpha_err:.2f}$\n"
        f"$\\nu_{{ref}} = {fit_result.nu_ref:.0f}$ MHz"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Scintillation bandwidth $\\gamma$ (MHz)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        log.info(f"Gamma scaling plot saved to: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_2d_acf_grid(
    acf_results: dict,
    fit_result,  # Scintillation2DResult
    fit_range_mhz: float = 25.0,
    ncols: int = 2,
    save_path: str = None,
    figsize_per_panel: tuple = (5, 4),
):
    """
    Grid of ACF panels with 2D model overlay.
    
    Parameters
    ----------
    acf_results : dict
        ACF results dictionary
    fit_result : Scintillation2DResult
        2D fit result
    fit_range_mhz : float
        Fit range for visual indicator
    ncols : int
        Number of columns in grid
    save_path : str, optional
        Path to save figure
    figsize_per_panel : tuple
        Size of each panel
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .fitting_2d import lorentzian_acf, gen_lorentzian_acf, gaussian_acf
    
    lags_list = acf_results['subband_lags_mhz']
    acfs_list = acf_results['subband_acfs']
    errs_list = acf_results.get('subband_acfs_err', [None] * len(lags_list))
    center_freqs = np.array(acf_results['subband_center_freqs_mhz'])
    n_subbands = len(center_freqs)
    
    nrows = int(np.ceil(n_subbands / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i, (lags, acf, err, nu_c) in enumerate(
        zip(lags_list, acfs_list, errs_list, center_freqs)
    ):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        # Data
        mask = np.abs(lags) <= fit_range_mhz * 2
        if err is not None:
            ax.errorbar(lags[mask], acf[mask], yerr=err[mask],
                       fmt='o', color='C0', ms=3, alpha=0.6, capsize=2)
        else:
            ax.plot(lags[mask], acf[mask], 'o', color='C0', ms=3, alpha=0.6)
        
        # Model
        gamma = fit_result.gamma_0 * (nu_c / fit_result.nu_ref) ** fit_result.alpha
        m = fit_result.m_0
        
        lags_fine = np.linspace(lags[mask].min(), lags[mask].max(), 200)
        if fit_result.model_type == 'lorentzian':
            model = lorentzian_acf(lags_fine, gamma, m)
        elif fit_result.model_type == 'gen_lorentzian':
            eta = fit_result.params.get('eta', 2.0)
            if hasattr(eta, 'value'):
                eta = eta.value
            model = gen_lorentzian_acf(lags_fine, gamma, m, eta)
        else:
            model = gaussian_acf(lags_fine, gamma, m)
        
        ax.plot(lags_fine, model, 'C1-', lw=2)
        
        # Fit range indicator
        ax.axvline(-fit_range_mhz, color='gray', ls='--', alpha=0.4)
        ax.axvline(fit_range_mhz, color='gray', ls='--', alpha=0.4)
        
        ax.set_title(f'{nu_c:.0f} MHz\n$\\gamma$ = {gamma:.2f} MHz', fontsize=10)
        ax.set_xlabel('Lag (MHz)', fontsize=9)
        ax.set_ylabel('ACF', fontsize=9)
        ax.tick_params(labelsize=8)
    
    # Hide empty subplots
    for i in range(n_subbands, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis('off')
    
    plt.suptitle(
        f'2D Fit: $\\gamma_0$ = {fit_result.gamma_0:.2f} MHz, '
        f'$\\alpha$ = {fit_result.alpha:.2f}',
        fontsize=12, y=1.02
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        log.info(f"ACF grid saved to: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_scat_scint_consistency(
    scint_results: dict,
    scat_results: dict,
    c_factor: float = 1.16,
    save_path: str = None,
    figsize: tuple = (10, 8)
):
    """
    Generate a consistency plot comparing measured scintillation bandwidths (gamma)
    with predictions derived from the scattering timescale (tau).
    
    The prediction follows: delta_nu_d = C / (2 * pi * tau)
    
    Parameters
    ----------
    scint_results : dict
        Dictionary containing 'subband_center_freqs_mhz' and 'subband_gamma' (or from 2D fit)
    scat_results : dict
        Dictionary containing scattering best_params (tau_1ghz, alpha, etc.)
    c_factor : float, optional
        The proportionality constant C in delta_nu_d * tau = C / (2*pi).
        Default is 1.16 (Kolmogorov thin screen).
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Extract Scintillation Data
    # Handle both sub-band results and 2D fit results
    nu_scint = np.array(scint_results.get('subband_center_freqs_mhz', [])) / 1000.0 # to GHz
    gamma_scint = np.array(scint_results.get('subband_gamma', []))
    gamma_err = np.array(scint_results.get('subband_gamma_err', np.zeros_like(gamma_scint)))
    
    if len(nu_scint) == 0:
        log.error("No scintillation frequency data found.")
        return fig

    # 2. Extract Scattering Data
    params = scat_results.get('best_params', {})
    tau_1ghz = params.get('tau_1ghz')
    alpha_scat = params.get('alpha', 4.0)
    
    if tau_1ghz is None:
        log.error("No scattering tau_1ghz found.")
        return fig

    # 3. Calculate Predicted Bandwidth
    # tau(nu) = tau_1ghz * (nu / 1.0)^-alpha
    # delta_nu_pred = C / (2 * pi * tau(nu))
    nu_fine = np.logspace(np.log10(nu_scint.min()*0.9), np.log10(nu_scint.max()*1.1), 100)
    tau_fine = tau_1ghz * (nu_fine / 1.0)**(-alpha_scat)
    # Convert tau from ms to s for the relation? No, delta_nu is in MHz, tau in ms.
    # 1 MHz * 1 ms = 1. Correct.
    gamma_pred = c_factor / (2 * np.pi * tau_fine)

    # 4. Plot consistency
    ax1.errorbar(nu_scint, gamma_scint, yerr=gamma_err, fmt='o', color='C0', 
                 ms=8, capsize=5, label='Measured Scintillation ($\\gamma$)')
    
    ax1.plot(nu_fine, gamma_pred, 'm-', lw=2.5, 
             label=f'Predicted $\\Delta\\nu_d$ ($C={c_factor:.2f}$, $\\alpha_{{scat}}={alpha_scat:.2f}$)')

    ax1.set_yscale('log')
    ax1.set_ylabel('Scintillation Bandwidth (MHz)', fontsize=12)
    ax1.set_title('Scintillation-Scattering Consistency', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, which='both', alpha=0.3)

    # 5. Residuals (Measured / Predicted)
    # Interpolate predicted gamma to measured frequencies
    gamma_pred_interp = np.interp(nu_scint, nu_fine, gamma_pred)
    ratio = gamma_scint / gamma_pred_interp
    ratio_err = gamma_err / gamma_pred_interp
    
    ax2.errorbar(nu_scint, ratio, yerr=ratio_err, fmt='o', color='gray')
    ax2.axhline(1.0, color='m', ls='--', alpha=0.7)
    ax2.set_xlabel('Frequency (GHz)', fontsize=12)
    ax2.set_ylabel('Ratio (Meas/Pred)', fontsize=10)
    ax2.set_ylim(0.1, 10.0)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', alpha=0.3)

    # Add summary text
    avg_ratio = np.nanmean(ratio)
    summary_text = (
        f"$\\langle \\gamma_{{meas}} / \\Delta\\nu_{{pred}} \\rangle = {avg_ratio:.2f}$\n"
        f"Scattering $\\tau_{{1GHz}} = {tau_1ghz:.3f}$ ms"
    )
    ax1.text(0.05, 0.05, summary_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='bottom')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        log.info(f"Consistency plot saved to {save_path}")
    
    return fig
