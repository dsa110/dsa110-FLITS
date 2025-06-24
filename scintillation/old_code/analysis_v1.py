# ==============================================================================
# File: scint_analysis/scint_analysis/analysis.py
# ==============================================================================
import numpy as np
import logging
from .core import ACF
from lmfit import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.odr import RealData, ODR, Model as ModelODR

log = logging.getLogger(__name__)

# -------------------------
# --- Model Definitions ---
# -------------------------

def lorentzian_model_1_comp(x, gamma1, m1, c1):
    return (m1**2 / (1 + (x / gamma1)**2)) + c1

def lorentzian_model_2_comp(x, gamma1, m1, gamma2, m2, c2):
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    return lor1 + lor2 + c2

def lorentzian_model_3_comp(x, gamma1, m1, gamma2, m2, gamma3, m3, c3):
    lor1 = m1**2 / (1 + (x / gamma1)**2)
    lor2 = m2**2 / (1 + (x / gamma2)**2)
    lor3 = m3**2 / (1 + (x / gamma3)**2)
    return lor1 + lor2 + lor3 + c3

def gaussian_model_1_comp(x, sigma1, m1, c1):
    return (m1**2 * np.exp(-0.5 * (x / sigma1)**2)) + c1

def gaussian_model_2_comp(x, sigma1, m1, sigma2, m2, c2):
    gauss1 = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    gauss2 = m2**2 * np.exp(-0.5 * (x / sigma2)**2)
    return gauss1 + gauss2 + c2

def gaussian_model_3_comp(x, sigma1, m1, sigma2, m2, sigma3, m3, c3):
    gauss1 = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    gauss2 = m2**2 * np.exp(-0.5 * (x / sigma2)**2)
    gauss3 = m3**2 * np.exp(-0.5 * (x / sigma3)**2)
    return gauss1 + gauss2 + gauss3 + c3

def gauss_plus_lor_model(x, sigma1, m1, gamma2, m2, c):
    gauss = m1**2 * np.exp(-0.5 * (x / sigma1)**2)
    lor = m2**2 / (1 + (x / gamma2)**2)
    return gauss + lor + c

def two_screen_unresolved_model(x, gamma1, m1, gamma2, m2, c):
    """ A physically-motivated model for two unresolved screens. """
    lor1 = (m1**2) / (1 + (x / gamma1)**2)
    lor2 = (m2**2) / (1 + (x / gamma2)**2)
    return lor1 + lor2 + (lor1 * lor2) + c

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

def calculate_acfs_for_subbands(masked_spectrum, config, burst_lims, noise_desc=None):
    """
    Takes a masked DynamicSpectrum and calculates the ACF for each sub-band.

    This function uses a pre-calculated burst envelope and an optional noise
    descriptor for robust ACF normalization.

    Args:
        masked_spectrum (DynamicSpectrum): The processed (RFI-masked, baseline-subtracted) spectrum.
        config (dict): The analysis configuration dictionary.
        burst_lims (tuple): The (start, end) time bins of the on-pulse region.
        noise_desc (NoiseDescriptor, optional): A pre-calculated noise descriptor.
    """
    log.info("Starting sub-band ACF calculations.")
    analysis_config = config.get('analysis', {})
    acf_config = analysis_config.get('acf', {})
    
    num_subbands = acf_config.get('num_subbands', 8)
    use_snr_subbanding = acf_config.get('use_snr_subbanding', False)
    max_lag_mhz = acf_config.get('max_lag_mhz', 45.0)

    # --- Noise handling ---
    # This section is now simplified as the noise window is also derived from burst_lims
    if noise_desc is None:
        log.warning("No noise descriptor provided. Using legacy method for noise estimation.")
        rfi_config = analysis_config.get('rfi_masking', {})
        if rfi_config.get('use_symmetric_noise_window', False):
            on_burst_duration = burst_lims[1] - burst_lims[0]
            off_burst_end = burst_lims[0]
            off_burst_start = off_burst_end - on_burst_duration
        else:
            off_burst_end = burst_lims[0] - rfi_config.get('off_burst_buffer', 100)
            off_burst_start = 0
        if off_burst_start < 0: off_burst_start = 0
        
        off_burst_spec = masked_spectrum.get_spectrum((off_burst_start, off_burst_end))
    else:
        log.info("Using robust noise parameters from provided NoiseDescriptor.")
        off_burst_spec = None

    results = {
        'subband_acfs': [], 'subband_lags_mhz': [], 
        'subband_center_freqs_mhz': [], 'subband_channel_widths_mhz': [],
        'subband_num_channels': []
    }

    # Get the on-pulse spectrum using the provided burst_lims
    burst_spectrum_full = masked_spectrum.get_spectrum(burst_lims)
    
    start_idx = 0
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
                if not burst_spectrum_full.mask[end_idx]:
                    cumulative_signal += burst_spectrum_full.data[end_idx]
                end_idx += 1
        if i == num_subbands - 1: end_idx = masked_spectrum.num_channels
        sub_spectrum = burst_spectrum_full[start_idx:end_idx]
        sub_freqs = masked_spectrum.frequencies[start_idx:end_idx]
        
        # --- USE THE NOISE DESCRIPTOR FOR NORMALIZATION ---
        if noise_desc:
            # If the noise is intensity-like, its mean is mu.
            # If it's flux-like (mean-subtracted), the baseline is effectively 0.
            sub_off_mean = noise_desc.mu if noise_desc.kind == "intensity" else 0.0
        else:
            # Fallback to the legacy method if no descriptor was provided
            sub_off_mean = np.ma.mean(off_burst_spec[start_idx:end_idx])
        
        channel_width = np.abs(np.mean(np.diff(sub_freqs))) if len(sub_freqs) > 1 else 0
        if channel_width == 0: 
            start_idx = end_idx
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

def _fit_acf_models(acf_object, fit_lagrange_mhz):
    """
    Internal helper to fit all candidate models to a single ACF.
    """
    fit_results = {}
    fit_mask = np.abs(acf_object.lags) <= fit_lagrange_mhz
    x_data = acf_object.lags[fit_mask]
    y_data = acf_object.acf[fit_mask]

    # --- Fit 1-Comp Models ---
    try:
        model1L = Model(lorentzian_model_1_comp, prefix='l1_')
        params1L = model1L.make_params(l1_gamma1=0.05, l1_m1=0.8, l1_c1=0)
        params1L['l1_gamma1'].set(min=1e-6); params1L['l1_m1'].set(min=0)
        fit_results['fit_1c_lor'] = model1L.fit(y_data, params1L, x=x_data)
    except Exception: fit_results['fit_1c_lor'] = None
    
    try:
        model1G = Model(gaussian_model_1_comp, prefix='g1_')
        params1G = model1G.make_params(g1_sigma1=0.05, g1_m1=0.8, g1_c1=0)
        params1G['g1_sigma1'].set(min=1e-6); params1G['g1_m1'].set(min=0)
        fit_results['fit_1c_gauss'] = model1G.fit(y_data, params1G, x=x_data)
    except Exception: fit_results['fit_1c_gauss'] = None

    # --- Fit 2-Comp Additive Models ---
    try:
        model2L = Model(lorentzian_model_2_comp, prefix='l2_')
        params2L = model2L.make_params(l2_gamma1=0.01, l2_m1=0.5, l2_gamma2=0.1, l2_m2=0.5, l2_c2=0)
        params2L['l2_gamma1'].set(min=1e-6); params2L['l2_gamma2'].set(min=1e-6)
        params2L['l2_m1'].set(min=0); params2L['l2_m2'].set(min=0)
        fit_results['fit_2c_lor'] = model2L.fit(y_data, params2L, x=x_data)
    except Exception as e:
        log.debug(f"2-comp Lorentzian fit failed: {e}")
        fit_results['fit_2c_lor'] = None

    try:
        model2G = Model(gaussian_model_2_comp, prefix='g2_')
        params2G = model2G.make_params(g2_sigma1=0.01, g2_m1=0.5, g2_sigma2=0.1, g2_m2=0.5, g2_c2=0)
        params2G['g2_sigma1'].set(min=1e-6); params2G['g2_sigma2'].set(min=1e-6)
        params2G['g2_m1'].set(min=0); params2G['g2_m2'].set(min=0)
        fit_results['fit_2c_gauss'] = model2G.fit(y_data, params2G, x=x_data)
    except Exception as e:
        log.debug(f"2-comp Gaussian fit failed: {e}")
        fit_results['fit_2c_gauss'] = None

    # --- Fit 1-Comp Gaussian + 1-Comp Lorentzian ---
    try:
        modelGL = Model(gauss_plus_lor_model, prefix='gl_')
        paramsGL = modelGL.make_params(gl_sigma1=0.01, gl_m1=0.5, gl_gamma2=0.1, gl_m2=0.5, gl_c=0)
        paramsGL['gl_sigma1'].set(min=1e-6)
        paramsGL['gl_gamma2'].set(min=1e-6)
        paramsGL['gl_m1'].set(min=0)
        paramsGL['gl_m2'].set(min=0)
        fit_results['fit_2c_mixed'] = modelGL.fit(y_data, paramsGL, x=x_data)
    except Exception as e:
        log.debug(f"Gauss+Lor fit failed: {e}")
        fit_results['fit_2c_mixed'] = None
        
    try:
        model2U = Model(two_screen_unresolved_model, prefix='tsu_')
        params2U = model2U.make_params(tsu_gamma1=0.01, tsu_m1=0.5, tsu_gamma2=0.1, tsu_m2=0.5, tsu_c=0)
        params2U['tsu_gamma1'].set(min=1e-6); params2U['tsu_gamma2'].set(min=1e-6)
        params2U['tsu_m1'].set(min=0); params2U['tsu_m2'].set(min=0)
        fit_results['fit_2c_unresolved'] = model2U.fit(y_data, params2U, x=x_data)
    except Exception as e:
        log.debug(f"2-comp unresolved screen fit failed: {e}")
        fit_results['fit_2c_unresolved'] = None
        
    # --- Fit 3-Comp Models ---
    try:
        model3L = Model(lorentzian_model_3_comp, prefix='l3_')
        params3L = model3L.make_params(l3_gamma1=0.01, l3_m1=0.3, l3_gamma2=0.1, l3_m2=0.3, l3_gamma3=0.5, l3_m3=0.3, c3=0)
        params3L['l3_gamma1'].set(min=1e-6); params3L['l3_gamma2'].set(min=1e-6); params3L['l3_gamma3'].set(min=1e-6)
        params3L['l3_m1'].set(min=0); params3L['l3_m2'].set(min=0); params3L['l3_m3'].set(min=0)
        fit_results['fit_3c_lor'] = model3L.fit(y_data, params3L, x=x_data)
    except Exception as e:
        log.debug(f"3-comp Lorentzian fit failed: {e}")
        fit_results['fit_3c_lor'] = None

    try:
        model3G = Model(gaussian_model_3_comp, prefix='g3_')
        params3G = model3G.make_params(g3_sigma1=0.01, g3_m1=0.3, g3_sigma2=0.1, g3_m2=0.3, g3_sigma3=0.5, g3_m3=0.3, c3=0)
        params3G['g3_sigma1'].set(min=1e-6); params3G['g3_sigma2'].set(min=1e-6); params3G['g3_sigma3'].set(min=1e-6)
        params3G['g3_m1'].set(min=0); params3G['g3_m2'].set(min=0); params3G['g3_m3'].set(min=0)
        fit_results['fit_3c_gauss'] = model3G.fit(y_data, params3G, x=x_data)
    except Exception as e:
        log.debug(f"3-comp Gaussian fit failed: {e}")
        fit_results['fit_3c_gauss'] = None

    return fit_results
    
def _select_overall_best_model(all_subband_fits):
    """
    Determines the best overall model by summing the BIC across all sub-bands
    for each model type and selecting the one with the lowest total BIC.
    """
    # Use a dictionary to store total BICs and fit counts for each model
    model_bics = {
        'fit_1c_lor': {'total_bic': 0.0, 'count': 0},
        'fit_2c_lor': {'total_bic': 0.0, 'count': 0},
        'fit_3c_lor': {'total_bic': 0.0, 'count': 0},
        'fit_1c_gauss': {'total_bic': 0.0, 'count': 0},
        'fit_2c_gauss': {'total_bic': 0.0, 'count': 0},
        'fit_3c_gauss': {'total_bic': 0.0, 'count': 0},
        'fit_2c_mixed': {'total_bic': 0.0, 'count': 0},
        'fit_2c_unresolved': {'total_bic': 0.0, 'count': 0},
    }
    
    for fits in all_subband_fits:
        for model_name, fit_result in fits.items():
            if fit_result and fit_result.success:
                model_bics[model_name]['total_bic'] += fit_result.bic
                model_bics[model_name]['count'] += 1

    log.info("--- Model Comparison (Lowest Total BIC is Best) ---")
    
    best_model = None
    min_bic = float('inf')

    for model_name, results in model_bics.items():
        if results['count'] > 0:
            log.info(f"Model '{model_name}': Total BIC = {results['total_bic']:.2f} (from {results['count']} fits)")
            if results['total_bic'] < min_bic:
                min_bic = results['total_bic']
                best_model = model_name
        else:
            log.info(f"Model '{model_name}': No successful fits.")
    
    if best_model is None:
        log.warning("No successful fits for any model. Defaulting to 'fit_1c_lor'.")
        return 'fit_1c_lor'

    log.info(f"==> Best overall model selected: {best_model}")
    return best_model

def analyze_scintillation_from_acfs(acf_results, config):
    """
    Main analysis orchestrator. Fits multiple ACF models, selects the best one,
    and derives scintillation parameters, including goodness-of-fit checks.
    """
    fit_config = config.get('analysis', {}).get('fitting', {})
    fit_lagrange_mhz = fit_config.get('fit_lagrange_mhz', 45.0)
    ref_freq = fit_config.get('reference_frequency_mhz', 600.0)

    log.info("Fitting all ACF models to all sub-band ACFs...")
    all_fits = []
    for i in tqdm(range(len(acf_results['subband_acfs'])), desc="Fitting Sub-band ACFs"):
        acf_data = acf_results['subband_acfs'][i]
        lags = acf_results['subband_lags_mhz'][i]
        sub_bandwidth = (acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i])
        current_fit_lagrange = min(fit_lagrange_mhz, sub_bandwidth / 2.0)
        fit_result = _fit_acf_models(ACF(acf_data, lags), current_fit_lagrange)
        all_fits.append(fit_result)

    # 1. Get the automatically selected best model via BIC as a default.
    auto_best_model = _select_overall_best_model(all_fits)
    
    # 2. Check the config for a user-forced model.
    forced_model = fit_config.get('force_model')
    
    if forced_model:
        # Check if the forced model is a valid option
        valid_models = all_fits[0].keys() if all_fits else []
        if forced_model in valid_models:
            log.warning(f"OVERRIDE: User has forced the model to '{forced_model}'. Bypassing BIC selection.")
            best_model_name = forced_model
        else:
            log.error(f"Invalid model '{forced_model}' specified in config. Falling back to automatic BIC selection.")
            log.info(f"Valid model names are: {list(valid_models)}")
            best_model_name = auto_best_model
    else:
        # If no model is forced, use the automatic selection.
        best_model_name = auto_best_model
    
    # Logic for determining the number of components was not robust.
    if '3c' in best_model_name:
        num_comps = 3
    elif '2c' in best_model_name or 'unresolved' in best_model_name:
        num_comps = 2
    else:
        num_comps = 1
    
    params_per_comp = [[] for _ in range(num_comps)]
    
    for i, fits in enumerate(all_fits):
        fit_obj = fits.get(best_model_name)
        
        if not (fit_obj and fit_obj.success):
            for comp_list in params_per_comp: comp_list.append({})
            continue

        p = fit_obj.params
        sub_bw = acf_results['subband_num_channels'][i] * acf_results['subband_channel_widths_mhz'][i]
        gof_metrics = {'bic': fit_obj.bic, 'redchi': fit_obj.redchi}

        def get_bw_params(param_name, is_gauss):
            val = p[param_name].value
            err = p[param_name].stderr if p[param_name].stderr is not None else np.nan
            if is_gauss:
                hwhm_factor = np.sqrt(2 * np.log(2))
                return val * hwhm_factor, err * hwhm_factor
            return val, err
        
        def get_mod_err(param_name):
            param = p.get(param_name)
            return param.stderr if param is not None and param.stderr is not None else np.nan

        component_params = []
        if '1c' in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g1_' if is_gauss else 'l1_'
            p_root = 'sigma' if is_gauss else 'gamma'
            bw, bw_err = get_bw_params(f'{prefix}{p_root}1', is_gauss)
            mod = p[f'{prefix}m1'].value
            mod_err = get_mod_err(f'{prefix}m1')
            component_params.append((bw, mod, bw_err, mod_err))

        elif '2c' in best_model_name and 'mixed' not in best_model_name and 'unresolved' not in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g2_' if is_gauss else 'l2_'
            p_root = 'sigma' if is_gauss else 'gamma'
            for j in range(1, 3):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))
        
        # Added case for the two_screen_unresolved_model
        elif 'unresolved' in best_model_name:
            prefix = 'tsu_'
            p_root = 'gamma'
            for j in range(1, 3):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss=False)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))

        elif '3c' in best_model_name:
            is_gauss = 'gauss' in best_model_name
            prefix = 'g3_' if is_gauss else 'l3_'
            p_root = 'sigma' if is_gauss else 'gamma'
            for j in range(1, 4):
                bw, bw_err = get_bw_params(f'{prefix}{p_root}{j}', is_gauss)
                mod = p[f'{prefix}m{j}'].value
                mod_err = get_mod_err(f'{prefix}m{j}')
                component_params.append((bw, mod, bw_err, mod_err))

        elif best_model_name == 'fit_2c_mixed':
            bw_g, bw_err_g = get_bw_params('gl_sigma1', is_gauss=True)
            bw_l, bw_err_l = get_bw_params('gl_gamma2', is_gauss=False)
            mod_g, mod_l = p['gl_m1'].value, p['gl_m2'].value
            mod_err_g, mod_err_l = get_mod_err('gl_m1'), get_mod_err('gl_m2')
            component_params.extend([(bw_g, mod_g, bw_err_g, mod_err_g), (bw_l, mod_l, bw_err_l, mod_err_l)])
        
        # Sort components by bandwidth (narrowest first) before assigning
        for comp_idx, (bw, mod, bw_err, mod_err) in enumerate(sorted(component_params)):
            num_scintles = max(1, sub_bw / bw) if bw > 0 else 1
            finite_err = bw / (2 * np.sqrt(num_scintles))
            param_dict = {'bw': bw, 'mod': mod, 'bw_err': bw_err, 'mod_err': mod_err, 'finite_err': finite_err}
            if comp_idx < len(params_per_comp):
                if comp_idx == 0: param_dict['gof'] = gof_metrics
                params_per_comp[comp_idx].append(param_dict)

    final_results = {'best_model': best_model_name, 'components': {}}
    all_powerlaw_fits = {}
    
    for i, params_list in enumerate(params_per_comp):
        name = f'component_{i+1}' if num_comps > 1 else 'scint_scale'
        measurements = [p for p in params_list if 'bw' in p]
        
        if len(measurements) < 2:
            log.warning(f"Skipping power-law fit for {name}: not enough valid data points.")
            final_results['components'][name] = {'power_law_fit_report': 'Fit failed: Not enough data'}
            continue

        freqs = np.array([acf_results['subband_center_freqs_mhz'][j] for j, p in enumerate(params_list) if 'bw' in p])
        bws = np.array([p.get('bw') for p in measurements])
        bw_errs = np.array([p.get('bw_err') for p in measurements])
        finite_errs = np.array([p.get('finite_err') for p in measurements])
        total_errs = np.sqrt(np.nan_to_num(bw_errs)**2 + np.nan_to_num(finite_errs)**2)

        def f(B, x): return B[0] * x**B[1]
        model = ModelODR(f)
        data = RealData(freqs, bws, sy=total_errs)
        odr = ODR(data, model, beta0=[1e-8, 4.0])
        out = odr.run()

        A_fit, alpha_fit = out.beta
        A_err, alpha_err = out.sd_beta
        
        # ================================================================= #
        # Use the fitted alpha and its error to suggest a 
        # physical scenario based on the findings from Pradeep et al. (2025)
        # and Nimmo et al. (2025).

        interpretation = "Undetermined"
        # Check for consistency within 3-sigma of the theoretical values
        if alpha_err is not None and not np.isnan(alpha_err):
            if abs(alpha_fit - 4.0) < 3 * alpha_err:
                # α ≈ 4 is expected for an unresolved point source 
                interpretation = "Consistent with unresolved point source (α ≈ 4)"
            elif abs(alpha_fit - 3.0) < 3 * alpha_err:
                # α ≈ 3 is expected for a screen resolving an incoherent emission region 
                interpretation = "Consistent with resolved emission region (α ≈ 3)"
            elif abs(alpha_fit - 1.0) < 3 * alpha_err:
                # α ≈ 1 is the flattened slope for two fully resolving screens 
                interpretation = "Consistent with resolved screens (α ≈ 1)"
        # ================================================================= #
        
        cov = out.cov_beta * out.res_var
        ref = ref_freq
        dAdA, dAda = ref**alpha_fit, A_fit * (ref**alpha_fit) * np.log(ref)
        grad = np.array([dAdA, dAda])
        var_b_ref = grad @ cov @ grad
        b_ref, b_ref_err = A_fit * ref**alpha_fit, np.sqrt(var_b_ref)
        
        all_powerlaw_fits[name] = out

        subband_measurements = []
        for j, p_dict in enumerate(measurements):
            measurement = {
                'freq_mhz': freqs[j], 'bw': p_dict.get('bw'), 'mod': p_dict.get('mod'),
                'bw_err': p_dict.get('bw_err'), 'mod_err': p_dict.get('mod_err'),
                'finite_err': p_dict.get('finite_err'), 'gof': p_dict.get('gof', {})
            }
            subband_measurements.append(measurement)

        final_results['components'][name] = {
            'power_law_fit_report': [A_fit, alpha_fit],
            'scaling_index': alpha_fit, 'scaling_index_err': alpha_err,
            'scaling_interpretation': interpretation,
            'bw_at_ref_mhz': b_ref, 'bw_at_ref_mhz_err': b_ref_err,
            'subband_measurements': subband_measurements
        }
    
    return final_results, all_fits, all_powerlaw_fits

def analyze_intra_pulse_scintillation(masked_spectrum, burst_lims, config, noise_desc):
    """
    Analyzes the evolution of scintillation parameters across the burst profile.

    This function divides the on-pulse data into time slices, calculates the ACF
    for each, and fits a model to track the evolution of the decorrelation
    bandwidth and modulation index.

    Args:
        masked_spectrum (DynamicSpectrum): The processed dynamic spectrum.
        burst_lims (tuple): The (start, end) time bins of the on-pulse region.
        config (dict): The analysis configuration dictionary.
        noise_desc (NoiseDescriptor): A pre-calculated noise descriptor for ACF normalization.

    Returns:
        list: A list of dictionaries, where each dictionary contains the fitted
              parameters ('time_s', 'bw', 'bw_err', 'mod', 'mod_err') for one time slice.
              Returns an empty list if the analysis cannot be run.
    """
    log.info("Starting intra-pulse scintillation analysis...")
    acf_config = config.get('analysis', {}).get('acf', {})
    fit_config = config.get('analysis', {}).get('fitting', {})

    num_time_bins = acf_config.get('intra_pulse_time_bins', 10)
    # Allow user to specify which model to fit, defaulting to a simple 1-comp Lorentzian
    # This is useful for tracking the evolution of the dominant (likely broader) component.
    model_to_fit = fit_config.get('intra_pulse_fit_model', 'fit_1c_lor')
    
    # Check if the chosen model is valid for parameter extraction
    if '1c' not in model_to_fit:
        log.error(f"Model '{model_to_fit}' is not a 1-component model. Intra-pulse analysis requires a simple model to track evolution. Aborting.")
        return []

    results = []
    
    on_pulse_start, on_pulse_end = burst_lims
    total_duration_bins = on_pulse_end - on_pulse_start
    slice_width_bins = total_duration_bins // num_time_bins

    if slice_width_bins < 2:
        log.warning("Burst duration is too short for the number of requested time slices. Skipping intra-pulse analysis.")
        return []

    for i in tqdm(range(num_time_bins), desc="Analyzing ACF vs. Time"):
        start_bin = on_pulse_start + (i * slice_width_bins)
        end_bin = start_bin + slice_width_bins

        # Extract the 1D spectrum for this time slice
        sub_spectrum = masked_spectrum.power[:, start_bin:end_bin].mean(axis=1)
        if sub_spectrum.count() < 10:  # Check if there's enough unmasked data
            continue

        # Get noise mean for proper normalization
        sub_off_mean = noise_desc.mu if noise_desc and noise_desc.kind == "intensity" else 0.0

        # Calculate ACF
        acf_obj = calculate_acf(
            sub_spectrum,
            masked_spectrum.channel_width_mhz,
            off_burst_spectrum_mean=sub_off_mean
        )
        if not acf_obj:
            continue

        # Fit models to the ACF
        fit_results = _fit_acf_models(acf_obj, fit_lagrange_mhz=fit_config.get('fit_lagrange_mhz', 45.0))
        
        fit_obj = fit_results.get(model_to_fit)
        if not (fit_obj and fit_obj.success):
            continue
            
        # Determine the exact lags used for the fit
        fit_lagrange = fit_config.get('fit_lagrange_mhz', 45.0)
        fit_mask = np.abs(acf_obj.lags) <= fit_lagrange
        fit_lags = acf_obj.lags[fit_mask] # These are the lags matching the best_fit array

        # Extract parameters from the 1-component fit
        p = fit_obj.params
        is_gauss = 'gauss' in model_to_fit
        prefix = 'g1_' if is_gauss else 'l1_'
        p_root = 'sigma' if is_gauss else 'gamma'
        
        bw_val = p[f'{prefix}{p_root}1'].value
        bw_err = p[f'{prefix}{p_root}1'].stderr if p[f'{prefix}{p_root}1'].stderr is not None else np.nan
        
        # Convert Gaussian sigma to HWHM if necessary
        if is_gauss:
            hwhm_factor = np.sqrt(2 * np.log(2))
            bw_val *= hwhm_factor
            if bw_err: bw_err *= hwhm_factor
            
        mod_val = p[f'{prefix}m1'].value
        mod_err = p[f'{prefix}m1'].stderr if p[f'{prefix}m1'].stderr is not None else np.nan

        # Calculate the central time of the bin
        center_time = np.mean(masked_spectrum.times[start_bin:end_bin])

        results.append({
            'time_s': center_time,
            'bw': bw_val,
            'bw_err': bw_err,
            'mod': mod_val,
            'mod_err': mod_err,
            'acf_lags': acf_obj.lags,      # Full lags for the raw ACF data
            'acf_data': acf_obj.acf,      # Raw ACF data
            'acf_fit_lags': fit_lags,     # Lags corresponding to the fit
            'acf_fit_best': fit_obj.best_fit, # The best-fit line
            'fit_success': fit_obj.success
        })


    log.info(f"Intra-pulse analysis complete. Found results for {len(results)} time slices.")
    return results

