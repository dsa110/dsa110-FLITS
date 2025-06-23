# ==============================================================================
# File: scint_analysis/scint_analysis/plotting.py (NEW FILE)
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import logging

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
    ax_acf = fig.add_subplot(gs[0:2, 0])
    cmap = plt.get_cmap('plasma')
    num_subbands = len(acf_results['subband_acfs'])
    for i in range(num_subbands):
        offset = i * 1.5
        rgba = cmap(i*0.8 / (num_subbands - 1)) if num_subbands > 1 else cmap(0.5)
        lags = acf_results['subband_lags_mhz'][i]
        acf = acf_results['subband_acfs'][i]
        peak_val = np.max(acf)
        acf_normalized = acf / peak_val if peak_val > 0 else acf
        ax_acf.plot(lags, acf_normalized + offset, color=rgba)
        
        fit_obj = all_subband_fits[i].get(best_model_name)
        if fit_obj and fit_obj.success:
            fit_normalized = fit_obj.eval(x=lags) / peak_val
            ax_acf.plot(lags, fit_normalized + offset, 'k--', alpha=0.7, label='Best Fit' if i == 0 else "")

    ax_acf.set_yticks([(i * 1.5) for i in range(num_subbands)])
    ax_acf.set_yticklabels([f"{cf:.1f}" for cf in acf_results['subband_center_freqs_mhz']])
    ax_acf.set_title("Normalized Sub-band ACFs & Best Fit")
    ax_acf.set_xlabel("Frequency Lag (MHz)")
    ax_acf.set_ylabel("Center Freq. (MHz)")
    if num_subbands > 0: ax_acf.legend()
    # Apply the robustly calculated x-limit
    ax_acf.set_xlim(-final_xlim, final_xlim)

    # --- Panel 2: Complete BIC Model Comparison ---
    ax_bic = fig.add_subplot(gs[0, 1])
    model_names = ['fit_1c_lor', 'fit_2c_lor', 'fit_3c_lor', 'fit_1c_gauss', 'fit_2c_gauss', 'fit_3c_gauss', 'fit_2c_mixed']
    bic_data = {name: [] for name in model_names}
    bic_freqs = acf_results['subband_center_freqs_mhz']

    for i in range(num_subbands):
        fits = all_subband_fits[i]
        for name in model_names:
            fit_result = fits.get(name)
            bic_val = fit_result.bic if (fit_result and fit_result.success) else np.nan
            bic_data[name].append(bic_val)
    
    for name in model_names:
        if not np.isnan(bic_data[name]).all():
            ax_bic.plot(bic_freqs, bic_data[name], 'o-', label=name.replace('fit_',''), alpha=0.8, markersize=4)

    ax_bic.set_ylabel("BIC (Lower is Better)")
    ax_bic.set_xlabel("Frequency (MHz)")
    ax_bic.set_title("Model Comparison")
    ax_bic.legend(fontsize='small')
    ax_bic.grid(True, alpha=0.3)

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

        ax_plaw.set_title(f"Power-Law Fit: {name.replace('_', ' ').title()}")
        ax_plaw.set_xlabel("Frequency (MHz)")
        ax_plaw.set_ylabel("Decorrelation BW (MHz)")
        ax_plaw.legend()
        ax_plaw.grid(True, alpha=0.2)
        ax_plaw.set_ylim(0, 5)
        
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
