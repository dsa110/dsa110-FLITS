# ==============================================================================
# File: scint_analysis/scint_analysis/analysis.py
# ==============================================================================
import numpy as np
import logging
from .core import ACF
from lmfit import Model
from tqdm import tqdm
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# -------------------------
# --- Model Definitions ---
# -------------------------

def lorentzian_model_1_comp(x, gamma1, m1, c1):
    """A single Lorentzian model with a constant offset."""
    return (m1**2 / (1 + (x / gamma1)**2)) + c1

def lorentzian_model_2_comp(x, gamma1, m1, gamma2, m2, c2):
    """A model of two Lorentzians with a shared constant offset."""
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    return lor1 + lor2 + c2

def power_law_model(x, c, n):
    """A simple power-law model: y = c * x^n."""
    return c * (x**n)

# ----------------------------------------------
# --- Core Calculation and Fitting Functions ---
# ----------------------------------------------

def calculate_acf(spectrum_1d, channel_width_mhz, off_burst_spectrum_mean=None, max_lag_bins=None):
    """
    Calculates the one-sided autocorrelation function of a spectrum using
    efficient NumPy operations.

    Parameters
    ----------
    spectrum_1d : np.ma.MaskedArray
        The 1D spectrum to autocorrelate. Must be a masked array.
    channel_width_mhz : float
        The channel width in MHz.
    off_burst_spectrum_mean : float, optional
        The mean of the off-burst spectrum, used for normalization.
    max_lag_bins : int, optional
        The maximum number of bins to compute the ACF out to.

    Returns
    -------
    ACF: object
    """
    log.debug(f"Calculating ACF for a spectrum of length {len(spectrum_1d)}.")
    valid_spec = spectrum_1d.compressed()
    if valid_spec.size < 10: return None
    
    mean_on = np.mean(valid_spec)
    
    # Define the normalization denominator for measuring the modulation index
    denom = (mean_on - off_burst_spectrum_mean)**2 if off_burst_spectrum_mean is not None else mean_on**2
    if denom == 0: denom = 1.0

    # Prepare the mean-subtracted spectrum, using NaN for masked values
    x = spectrum_1d.filled(np.nan) - mean_on
    n_chan = len(x)
    if max_lag_bins is None: max_lag_bins = n_chan
    
    lags = np.arange(1, max_lag_bins)
    acf_vals = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Create two shifted versions of the array
        v1, v2 = x[:-lag], x[lag:]
        # The product will be NaN if either element was originally masked
        prod = v1 * v2
        # Count only the pairs where both elements were valid
        num_valid = np.sum(~np.isnan(prod))
        if num_valid > 1:
            # Sum only the valid (non-NaN) products in numerator
            acf_vals[i] = np.nansum(prod) / (num_valid * denom)
            
    pos_lags_mhz = lags * channel_width_mhz
    full_acf = np.concatenate((acf_vals[::-1], acf_vals))
    full_lags = np.concatenate((-pos_lags_mhz[::-1], pos_lags_mhz))
    
    return ACF(full_acf, full_lags)

def calculate_acfs_for_subbands(masked_spectrum, config):
    """
    Takes a masked DynamicSpectrum and calculates the ACF for each sub-band.
    """
    log.info("Starting sub-band ACF calculations.")
    analysis_config = config.get('analysis', {})
    acf_config = analysis_config.get('acf', {})
    rfi_config = analysis_config.get('rfi_masking', {})
    
    num_subbands = acf_config.get('num_subbands', 8)
    use_snr_subbanding = acf_config.get('use_snr_subbanding', False)
    max_lag_mhz = acf_config.get('max_lag_mhz', 1.0)
    if max_lag_mhz is None: max_lag_mhz = 45.0 # Default to 100 if null

    # Prepare off-burst spectrum for normalization
    burst_lims = masked_spectrum.find_burst_envelope(thres=rfi_config.get('find_burst_thres', 5))
    
    fig = plt.figure()
    plt.plot(masked_spectrum.get_profile((burst_lims[0], burst_lims[1])))
    plt.title('On Burst Timeseries')
    plt.show()
    
    # Symmetric noise window
    if rfi_config.get('use_symmetric_noise_window', False):
        on_burst_duration = burst_lims[1] - burst_lims[0]
        off_burst_end = burst_lims[0]
        off_burst_start = off_burst_end - on_burst_duration
        log.info(f"Using symmetric noise window of duration {on_burst_duration} bins.")
    else:
        # Original logic using a fixed buffer
        off_burst_end = burst_lims[0] - rfi_config.get('off_burst_buffer', 100)
        off_burst_start = 0
        
    # Ensure the window is valid
    if off_burst_start < 0:
        log.warning(f"Calculated noise window start is before data start. Clipping to 0.")
        off_burst_start = 0
    
    off_burst_spec = masked_spectrum.get_spectrum((off_burst_start, off_burst_end))
             
    fig = plt.figure()
    plt.plot(masked_spectrum.get_spectrum((burst_lims[0], burst_lims[1])))
    plt.title('On Burst Spectrum')
    plt.show()
              
    fig = plt.figure()
    plt.plot(off_burst_spec)
    plt.title('Off Burst Spectrum')
    plt.show()


    results = {
        'subband_acfs': [], 'subband_lags_mhz': [], 
        'subband_center_freqs_mhz': [], 'subband_channel_widths_mhz': [],
        'subband_num_channels': []
    }

    start_idx = 0
    burst_spectrum_full = masked_spectrum.get_spectrum(burst_lims)
    total_signal = np.sum(burst_spectrum_full.compressed())
    
    for i in tqdm(range(num_subbands), desc="Calculating sub-band ACFs"):
        sub_len = masked_spectrum.num_channels // num_subbands
        if not use_snr_subbanding:
            end_idx = start_idx + sub_len
        else:
            target_signal = total_signal / num_subbands
            cumulative_signal = 0
            end_idx = start_idx
            while cumulative_signal < target_signal and end_idx < masked_spectrum.num_channels:
                # Check the mask of the parent array before accessing data
                if not burst_spectrum_full.mask[end_idx]:
                    cumulative_signal += burst_spectrum_full.data[end_idx]
                end_idx += 1
        
        if i == num_subbands - 1: end_idx = masked_spectrum.num_channels

        sub_spectrum = burst_spectrum_full[start_idx:end_idx]
        sub_off_mean = np.ma.mean(off_burst_spec[start_idx:end_idx])
        sub_freqs = masked_spectrum.frequencies[start_idx:end_idx]
        
        channel_width = np.abs(np.mean(np.diff(sub_freqs))) if len(sub_freqs) > 1 else 0
        if channel_width == 0: 
            continue

        # Cap max_lag if it exceeds bandwidth of the sub-band
        sub_bandwidth = (sub_spectrum.count() * channel_width)
        current_max_lag = max_lag_mhz
        if current_max_lag is None or current_max_lag > sub_bandwidth:
            current_max_lag = sub_bandwidth
            
        max_lag_bins_sub = int(current_max_lag / channel_width)
        
        acf_obj = calculate_acf(
            sub_spectrum, 
            channel_width, 
            off_burst_spectrum_mean=sub_off_mean, 
            max_lag_bins=max_lag_bins_sub)
        
        if acf_obj:
            results['subband_acfs'].append(acf_obj.acf)
            results['subband_lags_mhz'].append(acf_obj.lags)
            results['subband_center_freqs_mhz'].append(np.mean(sub_freqs))
            results['subband_channel_widths_mhz'].append(np.abs(np.mean(np.diff(sub_freqs))))
            results['subband_num_channels'].append(sub_spectrum.count()) # Count of unmasked channels
        
        start_idx = end_idx
        
    return results

def _fit_lorentzian_models_to_acf(acf_object, fit_lagrange_mhz):
    """Internal helper to fit both 1- and 2-component models to a single ACF."""
    fit_results = {}
    
    # Max fit lag range cannot exceed ACF width
    max_available_lag = np.max(np.abs(acf_object.lags))
    current_fit_lagrange = min(fit_lagrange_mhz, max_available_lag)
    
    fit_mask = np.abs(acf_object.lags) <= current_fit_lagrange
    
    try:
        model1 = Model(lorentzian_model_1_comp)
        params1 = model1.make_params(gamma1=0.05, m1=0.8, c1=0)
        params1['gamma1'].set(min=1e-6); params1['m1'].set(min=0)
        fit_results['fit_1_comp'] = model1.fit(acf_object.acf[fit_mask], params1, x=acf_object.lags[fit_mask])
    except Exception as e:
        log.warning(f"1-component fit failed: {e}")
        fit_results['fit_1_comp'] = None

    try:
        model2 = Model(lorentzian_model_2_comp)
        params2 = model2.make_params(gamma1=0.01, m1=0.5, gamma2=0.1, m2=0.5, c2=0)
        params2['gamma1'].set(min=1e-6); params2['gamma2'].set(min=1e-6)
        params2['m1'].set(min=0); params2['m2'].set(min=0)
        fit_results['fit_2_comp'] = model2.fit(acf_object.acf[fit_mask], params2, x=acf_object.lags[fit_mask])
    except Exception as e:
        log.warning(f"2-component fit failed: {e}")
        fit_results['fit_2_comp'] = None
        
    return fit_results

def _select_overall_best_model(all_subband_fits):
    """Internal helper to determine the best overall model using BIC."""
    votes = {1: 0, 2: 0}
    for fits in all_subband_fits:
        fit1, fit2 = fits.get('fit_1_comp'), fits.get('fit_2_comp')
        if fit1 and fit2:
            votes[1 if fit1.bic < fit2.bic else 2] += 1
        elif fit1: votes[1] += 1
        elif fit2: votes[2] += 1
    return 1 if votes[1] >= votes[2] else 2

def analyze_scintillation_from_acfs(acf_results, config):
    """
    Main analysis orchestrator. Analyzes a set of ACFs to derive scintillation parameters.
    
    Args:
        acf_results (dict): A dictionary containing ACF data.
        config (dict): The merged configuration dictionary for the analysis.

    Returns:
        dict: A dictionary containing the final derived scintillation parameters.
    """
    fit_config = config.get('analysis', {}).get('fitting', {})
    fit_lagrange_mhz = fit_config.get('fit_lagrange_mhz', 0.5)
    ref_freq = fit_config.get('reference_frequency_mhz', 600.0)
    show_subband_plots = fit_config.get('show_subband_fit_plots', False)

    log.info("Fitting Lorentzian models to all sub-band ACFs...")
    all_fits = []
    for i, (acf_data, lags) in enumerate(zip(acf_results['subband_acfs'], acf_results['subband_lags_mhz'])):
        sub_bandwidth = (acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i])
        current_fit_lagrange = min(fit_lagrange_mhz, sub_bandwidth / 2.0)
        
        fit_result = _fit_lorentzian_models_to_acf(ACF(acf_data, lags), current_fit_lagrange)
        all_fits.append(fit_result)

        if show_subband_plots:
            center_freq = acf_results['subband_center_freqs_mhz'][i]
            plt.figure(figsize=(10, 6))
            plt.plot(lags, acf_data, 'k.', markersize=4, label='ACF Data')
            fit1 = fit_result.get('fit_1_comp')
            if fit1 and fit1.success:
                plt.plot(lags, fit1.eval(x=lags), 'b--', alpha=0.8, label=f'1-Comp Fit (BIC: {fit1.bic:.1f})')
            fit2 = fit_result.get('fit_2_comp')
            if fit2 and fit2.success:
                plt.plot(lags, fit2.eval(x=lags), 'r-', alpha=0.7, label=f'2-Comp Fit (BIC: {fit2.bic:.1f})')
            
            plt.title(f'Diagnostic Fit for Sub-band @ {center_freq:.1f} MHz')
            plt.xlabel('Frequency Lag (MHz)')
            plt.ylabel('Autocorrelation')
            plt.xlim(-current_fit_lagrange, current_fit_lagrange)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

    best_model = _select_overall_best_model(all_fits)
    log.info(f"Model selection complete. Best overall model: {best_model} component(s).")
    
    params_per_comp = [[] for _ in range(best_model)]
    
    for i, fits in enumerate(all_fits):
        fit_obj = fits.get(f'fit_{best_model}_comp')
        if not (fit_obj and fit_obj.success):
            for comp_list in params_per_comp: comp_list.append({})
            continue

        p = fit_obj.params
        sub_bw = acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i]

        def get_err(param):
            return float(param.stderr) if param.stderr is not None else np.nan

        if best_model == 1:
            bw = p['gamma1'].value
            num_scintles = max(1, sub_bw / bw) if bw > 0 else 1
            finite_err = bw / (2 * np.sqrt(num_scintles))
            params_per_comp[0].append({'bw': bw, 'mod': p['m1'].value, 'bw_err': get_err(p['gamma1']), 'finite_err': finite_err})
        else:
            components = sorted([(p[f'gamma{j}'].value, p[f'm{j}'].value, get_err(p[f'gamma{j}'])) for j in [1, 2]])
            for comp_idx, (bw, mod, err) in enumerate(components):
                num_scintles = max(1, sub_bw / bw) if bw > 0 else 1
                finite_err = bw / (2 * np.sqrt(num_scintles))
                params_per_comp[comp_idx].append({'bw': bw, 'mod': mod, 'bw_err': err, 'finite_err': finite_err})

    # 4. Perform Power-Law fitting for each component
    final_results = {'best_model': best_model, 'components': {}}
    all_powerlaw_fits = {}
    
    for i, params_list in enumerate(params_per_comp):
        name = f'component_{i+1}' if best_model > 1 else 'scint_scale'
        freqs = np.array(acf_results['subband_center_freqs_mhz'])
        bws = np.array([p.get('bw') for p in params_list])
        
        # Build fitting arrays from the parameter list ***
        # This ensures that only data from successful fits are included.
        freqs_for_fit = np.array([acf_results['subband_center_freqs_mhz'][j] for j, p in enumerate(params_list) if 'bw' in p])
        bws_for_fit = np.array([p.get('bw') for p in params_list if 'bw' in p])
        
        fit_errs_list = [p.get('bw_err') for p in params_list if 'bw' in p]
        fit_errs = np.array([err if err is not None else np.nan for err in fit_errs_list], dtype=float)
        
        finite_errs = np.array([p.get('finite_err') for p in params_list if 'bw' in p])
        
        total_errs = np.sqrt(np.nan_to_num(fit_errs)**2 + np.nan_to_num(finite_errs)**2)
        
        valid = ~np.isnan(bws_for_fit) & ~np.isnan(total_errs) & (total_errs > 0)
        # If not enough valid data points, add placeholder to results and continue
        if np.sum(valid) < 2:
            log.warning(f"Skipping power-law fit for {name}: not enough valid data points.")
            final_results['components'][name] = {'power_law_fit_report': 'Fit failed: Not enough data'}
            continue
        
        #try:
        #    bws = np.array([p.get('bw') for p in params_list])
        #    fit_errs = np.array([p.get('bw_err', np.nan) for p in params_list])
        #    finite_errs = np.array([p.get('finite_err', np.nan) for p in params_list])

        #    total_errs = np.sqrt(np.nan_to_num(fit_errs)**2 + np.nan_to_num(finite_errs)**2)

        #    valid = ~np.isnan(bws) & ~np.isnan(total_errs) & (total_errs > 0)
        #    
        #    if np.sum(valid) < 2:
        #        log.warning(f"Skipping power-law fit for {name}: not enough valid data points.")
        #        continue
        #
        #except:
        #    import uncertainties.unumpy as unp
        #    bws       = unp.nominal_values(bws)        # strip uncertainties
        #    fit_errs  = unp.nominal_values(fit_errs)
        #    finite_errs = unp.nominal_values(finite_errs)

        #    total_errs = np.sqrt(np.nan_to_num(fit_errs)**2 + np.nan_to_num(finite_errs)**2)

        #    valid = ~np.isnan(bws) & ~np.isnan(total_errs) & (total_errs > 0)
        #
        #    if np.sum(valid) < 2:
        #        log.warning(f"Skipping power-law fit for {name}: not enough valid data points.")
        #        continue
            
        freqs_for_fit = np.array([acf_results['subband_center_freqs_mhz'][j] for j, p in enumerate(params_list) if 'bw' in p])
        bws_for_fit = np.array([p.get('bw') for p in params_list if 'bw' in p])
        
        #powlaw_fit_result = Model(power_law_model).fit(bws_for_fit[valid], x=freqs_for_fit[valid], weights=1/total_errs[valid], c=1, n=4)
        #c, n = powlaw_fit_result.params['c'].value, powlaw_fit_result.params['n'].value
        
        #try:
        # Define power-law model: b = A * nu**alpha
        def f(B, x):
            return B[0] * x**B[1]

        from scipy.odr import RealData, ODR
        from scipy.odr import Model as ModelODR
        model   = ModelODR(f)
        data    = RealData(freqs_for_fit, bws_for_fit)
        odr     = ODR(data, model, beta0=[1e-8, 4.0])   # initial guess [A, alpha]
        out     = odr.run()

        A_fit, alpha_fit = out.beta
        print(f"α = {alpha_fit:.2f},  A = {A_fit:.3g},  residual variance = {out.res_var:.3f}")
        A_err, alpha_err = out.sd_beta

        cov = out.cov_beta * out.res_var       # full covariance matrix
        ref = ref_freq                           # e.g. 1400.0 for 1400 MHz

        # partial derivatives
        dAdA   = ref**alpha_fit               # ∂b/∂A
        dAda   = A_fit * (ref**alpha_fit)*np.log(ref)  # ∂b/∂α

        # build gradient and compute variance
        grad = np.array([dAdA, dAda])
        var_b_ref = grad @ cov @ grad
        b_ref     = A_fit * ref**alpha_fit
        b_ref_err = np.sqrt(var_b_ref)

        # Store the successful fit object
        all_powerlaw_fits[name] = out

        # Populate the results for this component
        subband_measurements = []
        valid_indices = np.where(valid)[0]
        for j in valid_indices:
            p = params_list[j]
            measurement = {
                'freq_mhz': freqs[j], 'bw': p.get('bw'), 'mod': p.get('mod'),
                'bw_err': p.get('bw_err'), 'finite_err': p.get('finite_err')
            }
            subband_measurements.append(measurement)

        final_results['components'][name] = {
            'power_law_fit_report': [A_fit, alpha_fit],
            'scaling_index': alpha_fit, 
            'scaling_index_err': alpha_err,
            'bw_at_ref_mhz': A_fit*(ref_freq ** alpha_fit),
            'bw_at_ref_mhz_err': b_ref_err,
            'subband_measurements': subband_measurements
            }
        
        return final_results, all_fits, all_powerlaw_fits  #, 

        #except:

        #    return final_results, all_fits, None #, powlaw_fit_result.params
