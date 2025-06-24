# ------------------------------------------------------------
# -- Backup Calculation and Fitting Functions (analysis.py) --
# ------------------------------------------------------------

def calculate_acf_chunking(spectrum_1d, channel_width_mhz, off_burst_spectrum_mean=None, max_lag_bins=None, num_chunks=8):
    """
    Calculates the one-sided ACF and its uncertainty using a chunking method.
    """
    n_unmasked = spectrum_1d.count()
    log.debug(f"Calculating ACF for a spectrum with {n_unmasked} unmasked channels.")
    
    # --- FIX: More robust initial checks ---
    if n_unmasked < 20:
        log.warning(f"Not enough data ({n_unmasked} points) to calculate a reliable ACF. Skipping.")
        return None
        
    if max_lag_bins is None:
        max_lag_bins = n_unmasked // (num_chunks * 2)

    if max_lag_bins <= 1:
        log.warning("max_lag_bins is too small to calculate ACF. Skipping.")
        return None

    if n_unmasked < max_lag_bins * num_chunks:
        log.warning("Not enough data for robust chunked ACF calculation. Errors may be unreliable.")
        num_chunks = max(2, n_unmasked // max_lag_bins)
    
    # Ensure there's enough data for at least minimal chunking
    if spectrum_1d.count() < 20: # Arbitrary small number
        log.warning("Not enough data to calculate a reliable ACF.")
        return None
        
    if max_lag_bins is None:
        # Default max_lag_bins if not provided
        max_lag_bins = spectrum_1d.count() // (num_chunks * 2)

    if spectrum_1d.count() < max_lag_bins * num_chunks:
        log.warning("Not enough data for robust chunked ACF calculation. Errors may be unreliable.")
        num_chunks = max(2, spectrum_1d.count() // max_lag_bins)

    # Prepare the full mean-subtracted spectrum
    mean_on = np.ma.mean(spectrum_1d)
    denom = (mean_on - off_burst_spectrum_mean)**2 if off_burst_spectrum_mean is not None else mean_on**2
    if denom == 0: denom = 1.0
    x = spectrum_1d.filled(np.nan) - mean_on
    n_chan = len(x)
    
    lags = np.arange(1, max_lag_bins)
    
    # Calculate ACF for each chunk to estimate variance
    chunk_size = n_chan // num_chunks
    chunk_acfs = []
    if chunk_size < max_lag_bins:
        log.warning("Chunk size is smaller than max_lag_bins, ACF will be truncated.")
        return None

    for i in range(num_chunks):
        chunk_x = x[i*chunk_size : (i+1)*chunk_size]
        chunk_acf_vals = np.zeros(len(lags))
        for j, lag in enumerate(lags):
            v1, v2 = chunk_x[:-lag], chunk_x[lag:]
            prod = v1 * v2
            num_valid = np.sum(~np.isnan(prod))
            if num_valid > 1:
                chunk_acf_vals[j] = np.nansum(prod) / (num_valid * denom)
        chunk_acfs.append(chunk_acf_vals)
        
    acf_vals = np.mean(chunk_acfs, axis=0)
    acf_errs = np.std(chunk_acfs, axis=0) / np.sqrt(num_chunks)

    ### FIX: Correctly create the two-sided, symmetrical ACF, lags, and errors ###
    
    # 1. Create the array of positive lags in physical units
    pos_lags_mhz = lags * channel_width_mhz
    
    # 2. Create the full, symmetric arrays for ACF, lags, and errors
    # The structure is: [reversed_negative_side, zero_point, positive_side]
    full_acf = np.concatenate((acf_vals[::-1], [1.0], acf_vals))
    full_lags = np.concatenate((-pos_lags_mhz[::-1], [0.0], pos_lags_mhz))
    full_err = np.concatenate((acf_errs[::-1], [1e-9], acf_errs))

    return ACF(full_acf, full_lags, full_err)

# ----------------------------------------------
# --- Old Calculation and Fitting Functions ----
# ----------------------------------------------

def _fit_acf_models_v0(acf_object, fit_lagrange_mhz):
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

def _fit_acf_models_v1(acf_object, fit_lagrange_mhz):
    """
    Internal helper to fit all candidate models to a single ACF using
    constrained fitting to ensure component separation and stability.
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

    # --- Fit 2-Comp Models (Constrained) ---
    try:
        model2L = Model(lorentzian_model_2_comp, prefix='l2_')
        params2L = model2L.make_params(l2_gamma1=0.01, l2_m1=0.5, l2_c2=0)
        params2L.add('l2_gamma_factor', value=5, min=1.01)
        params2L['l2_gamma2'].set(expr='l2_gamma1 * l2_gamma_factor')
        params2L['l2_m2'].set(value=0.5, min=0)
        fit_results['fit_2c_lor'] = model2L.fit(y_data, params2L, x=x_data)
    except Exception: fit_results['fit_2c_lor'] = None
        
    try:
        model2G = Model(gaussian_model_2_comp, prefix='g2_')
        params2G = model2G.make_params(g2_sigma1=0.01, g2_m1=0.5, g2_c2=0)
        params2G.add('g2_sigma_factor', value=5, min=1.01)
        params2G['g2_sigma2'].set(expr='g2_sigma1 * g2_sigma_factor')
        params2G['g2_m2'].set(value=0.5, min=0)
        fit_results['fit_2c_gauss'] = model2G.fit(y_data, params2G, x=x_data)
    except Exception: fit_results['fit_2c_gauss'] = None

    # --- Fit Mixed & Unresolved Models (Constrained where appropriate) ---
    try:
        # No ordering constraint needed here as the functional forms are different
        modelGL = Model(gauss_plus_lor_model, prefix='gl_')
        paramsGL = modelGL.make_params(gl_sigma1=0.01, gl_m1=0.5, gl_gamma2=0.1, gl_m2=0.5, gl_c=0)
        paramsGL['gl_sigma1'].set(min=1e-6); paramsGL['gl_m1'].set(min=0)
        paramsGL['gl_gamma2'].set(min=1e-6); paramsGL['gl_m2'].set(min=0)
        fit_results['fit_2c_mixed'] = modelGL.fit(y_data, paramsGL, x=x_data)
    except Exception: fit_results['fit_2c_mixed'] = None
        
    try:
        model2U = Model(two_screen_unresolved_model, prefix='tsu_')
        params2U = model2U.make_params(tsu_gamma1=0.01, tsu_m1=0.5, tsu_c=0)
        params2U.add('tsu_gamma_factor', value=5, min=1.01)
        params2U['tsu_gamma2'].set(expr='tsu_gamma1 * tsu_gamma_factor')
        params2U['tsu_m2'].set(value=0.5, min=0)
        fit_results['fit_2c_unresolved'] = model2U.fit(y_data, params2U, x=x_data)
    except Exception: fit_results['fit_2c_unresolved'] = None

    # --- Fit 3-Comp Models (Constrained) ---
    try:
        model3L = Model(lorentzian_model_3_comp, prefix='l3_')
        params3L = model3L.make_params(l3_gamma1=0.01, l3_m1=0.3, l3_c3=0)
        params3L.add('l3_gamma_factor2', value=5, min=1.01)
        params3L.add('l3_gamma_factor3', value=5, min=1.01)
        params3L['l3_gamma2'].set(expr='l3_gamma1 * l3_gamma_factor2')
        params3L['l3_gamma3'].set(expr='l3_gamma2 * l3_gamma_factor3')
        params3L['l3_m2'].set(value=0.3, min=0)
        params3L['l3_m3'].set(value=0.3, min=0)
        fit_results['fit_3c_lor'] = model3L.fit(y_data, params3L, x=x_data)
    except Exception: fit_results['fit_3c_lor'] = None

    try:
        model3G = Model(gaussian_model_3_comp, prefix='g3_')
        params3G = model3G.make_params(g3_sigma1=0.01, g3_m1=0.3, g3_c3=0)
        params3G.add('g3_sigma_factor2', value=5, min=1.01)
        params3G.add('g3_sigma_factor3', value=5, min=1.01)
        params3G['g3_sigma2'].set(expr='g3_sigma1 * g3_sigma_factor2')
        params3G['g3_sigma3'].set(expr='g3_sigma2 * g3_sigma_factor3')
        params3G['g3_m2'].set(value=0.3, min=0)
        params3G['g3_m3'].set(value=0.3, min=0)
        fit_results['fit_3c_gauss'] = model3G.fit(y_data, params3G, x=x_data)
    except Exception: fit_results['fit_3c_gauss'] = None
        
    return fit_results

def analyze_intra_pulse_scintillation_v0(masked_spectrum, burst_lims, config, noise_desc):
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

def calculate_acfs_for_subbands_v0(masked_spectrum, config, burst_lims, noise_desc=None):
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
        
        # USE THE NOISE DESCRIPTOR FOR NORMALIZATION
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

        # The actual (unmasked) bandwidth of this sub-band
        available_bandwidth = sub_spectrum.count() * channel_width
        
        # The requested max lag cannot exceed the available bandwidth
        current_max_lag_mhz = min(max_lag_mhz, available_bandwidth)
        
        # Convert the capped max lag to bins
        max_lag_bins_sub = int(current_max_lag_mhz / channel_width)
        
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
                          
def _fit_acf_models_v0(acf_object, fit_lagrange_mhz):
    """
    Internal helper to fit all candidate models to a single ACF using
    constrained fitting and proper uncertainties (weights).

    Args:
        acf_object (ACF): An ACF object that must contain data, lags, and errors.
        fit_lagrange_mhz (float): The frequency lag range over which to perform the fit.

    Returns:
        dict: A dictionary containing the fit results for all attempted models.
    """
    fit_results = {}
    
    # Create a mask that selects the fit range and explicitly excludes the zero-lag point.
    fit_mask = (np.abs(acf_object.lags) <= fit_lagrange_mhz) & (acf_object.lags != 0)
    x_data = acf_object.lags[fit_mask]
    y_data = acf_object.acf[fit_mask]
    
    # Use the calculated ACF errors to create weights for the fit.
    # A small floor is added to the errors to prevent division by zero.
    if acf_object.err is not None:
        y_errors = acf_object.err[fit_mask]
        weights = 1.0 / np.maximum(y_errors, 1e-9)
    else:
        # Fallback if no errors are provided (not recommended)
        weights = None
        
    # --- Fit 1-Comp Models ---
    try:
        model1L = Model(lorentzian_model_1_comp, prefix='l1_')
        params1L = model1L.make_params(l1_gamma1=0.05, l1_m1=0.8, l1_c1=0)
        params1L['l1_gamma1'].set(min=1e-6); params1L['l1_m1'].set(min=0)
        fit_results['fit_1c_lor'] = model1L.fit(y_data, params1L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"1-comp Lorentzian fit failed: {e}")
        fit_results['fit_1c_lor'] = None
    
    try:
        model1G = Model(gaussian_model_1_comp, prefix='g1_')
        params1G = model1G.make_params(g1_sigma1=0.05, g1_m1=0.8, g1_c1=0)
        params1G['g1_sigma1'].set(min=1e-6); params1G['g1_m1'].set(min=0)
        fit_results['fit_1c_gauss'] = model1G.fit(y_data, params1G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"1-comp Gaussian fit failed: {e}")
        fit_results['fit_1c_gauss'] = None

    # --- Fit 2-Comp Models (Constrained) ---
    try:
        model2L = Model(lorentzian_model_2_comp, prefix='l2_')
        params2L = model2L.make_params(l2_gamma1=0.01, l2_m1=0.5, l2_c2=0)
        params2L['l2_gamma1'].set(min=1e-6); params2L['l2_m1'].set(min=0)
        params2L['l2_m2'].set(value=0.5, min=0)
        params2L.add('l2_gamma_factor', value=5, min=1.01) # Must be > 1
        params2L['l2_gamma2'].set(expr='l2_gamma1 * l2_gamma_factor')
        fit_results['fit_2c_lor'] = model2L.fit(y_data, params2L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp Lorentzian fit failed: {e}")
        fit_results['fit_2c_lor'] = None
        
    try:
        model2G = Model(gaussian_model_2_comp, prefix='g2_')
        params2G = model2G.make_params(g2_sigma1=0.01, g2_m1=0.5, g2_c2=0)
        params2G['g2_sigma1'].set(min=1e-6); params2G['g2_m1'].set(min=0)
        params2G['g2_m2'].set(value=0.5, min=0)
        params2G.add('g2_sigma_factor', value=5, min=1.01)
        params2G['g2_sigma2'].set(expr='g2_sigma1 * g2_sigma_factor')
        fit_results['fit_2c_gauss'] = model2G.fit(y_data, params2G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp Gaussian fit failed: {e}")
        fit_results['fit_2c_gauss'] = None

    # --- Fit Mixed & Unresolved Models ---
    try:
        # No ordering constraint needed; functional forms are different, preventing degeneracy.
        modelGL = Model(gauss_plus_lor_model, prefix='gl_')
        paramsGL = modelGL.make_params(gl_sigma1=0.01, gl_m1=0.5, gl_gamma2=0.1, gl_m2=0.5, gl_c=0)
        paramsGL['gl_sigma1'].set(min=1e-6); paramsGL['gl_m1'].set(min=0)
        paramsGL['gl_gamma2'].set(min=1e-6); paramsGL['gl_m2'].set(min=0)
        fit_results['fit_2c_mixed'] = modelGL.fit(y_data, paramsGL, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"Gauss+Lor fit failed: {e}")
        fit_results['fit_2c_mixed'] = None
        
    try:
        model2U = Model(two_screen_unresolved_model, prefix='tsu_')
        params2U = model2U.make_params(tsu_gamma1=0.01, tsu_m1=0.5, tsu_c=0)
        params2U['tsu_gamma1'].set(min=1e-6); params2U['tsu_m1'].set(min=0)
        params2U['tsu_m2'].set(value=0.5, min=0)
        params2U.add('tsu_gamma_factor', value=5, min=1.01)
        params2U['tsu_gamma2'].set(expr='tsu_gamma1 * tsu_gamma_factor')
        fit_results['fit_2c_unresolved'] = model2U.fit(y_data, params2U, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp unresolved screen fit failed: {e}")
        fit_results['fit_2c_unresolved'] = None

    # --- Fit 3-Comp Models (Constrained) ---
    try:
        model3L = Model(lorentzian_model_3_comp, prefix='l3_')
        params3L = model3L.make_params(l3_gamma1=0.01, l3_m1=0.3, l3_c3=0)
        params3L['l3_gamma1'].set(min=1e-6); params3L['l3_m1'].set(min=0)
        params3L['l3_m2'].set(value=0.3, min=0)
        params3L['l3_m3'].set(value=0.3, min=0)
        params3L.add('l3_gamma_factor2', value=5, min=1.01)
        params3L.add('l3_gamma_factor3', value=5, min=1.01)
        params3L['l3_gamma2'].set(expr='l3_gamma1 * l3_gamma_factor2')
        params3L['l3_gamma3'].set(expr='l3_gamma2 * l3_gamma_factor3')
        fit_results['fit_3c_lor'] = model3L.fit(y_data, params3L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"3-comp Lorentzian fit failed: {e}")
        fit_results['fit_3c_lor'] = None

    try:
        model3G = Model(gaussian_model_3_comp, prefix='g3_')
        params3G = model3G.make_params(g3_sigma1=0.01, g3_m1=0.3, g3_c3=0)
        params3G['g3_sigma1'].set(min=1e-6); params3G['g3_m1'].set(min=0)
        params3G['g3_m2'].set(value=0.3, min=0)
        params3G['g3_m3'].set(value=0.3, min=0)
        params3G.add('g3_sigma_factor2', value=5, min=1.01)
        params3G.add('g3_sigma_factor3', value=5, min=1.01)
        params3G['g3_sigma2'].set(expr='g3_sigma1 * g3_sigma_factor2')
        params3G['g3_sigma3'].set(expr='g3_sigma2 * g3_sigma_factor3')
        fit_results['fit_3c_gauss'] = model3G.fit(y_data, params3G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"3-comp Gaussian fit failed: {e}")
        fit_results['fit_3c_gauss'] = None
        
    return fit_results

def _fit_acf_models_v2(acf_object, fit_lagrange_mhz):
    """
    Internal helper to fit all candidate models to a single ACF using
    constrained fitting and proper uncertainties (weights).

    Args:
        acf_object (ACF): An ACF object that must contain data, lags, and errors.
        fit_lagrange_mhz (float): The frequency lag range over which to perform the fit.

    Returns:
        dict: A dictionary containing the fit results for all attempted models.
    """
    fit_results = {}
    
    # Create a mask that selects the fit range and explicitly excludes the zero-lag point.
    fit_mask = (np.abs(acf_object.lags) <= fit_lagrange_mhz) & (acf_object.lags != 0)
    x_data = acf_object.lags[fit_mask]
    y_data = acf_object.acf[fit_mask]
    
    # Use the calculated ACF errors to create weights for the fit.
    # A small floor is added to the errors to prevent division by zero.
    if acf_object.err is not None:
        y_errors = acf_object.err[fit_mask]
        weights = 1.0 / np.maximum(y_errors, 1e-9)
    else:
        # Fallback if no errors are provided (not recommended)
        weights = None
        
    has_sn   = sigma_self_mhz is not None
    if has_sn:
        sn = Model(gauss_fixed_width, prefix='sn_')  # width frozen
        p_sn = sn.make_params(sn_sigma_self=sigma_self_mhz, vary=False,
                              sn_m_self=0.3,            min=0, max=np.sqrt(3),
                              sn_c=0.0)

    # --- Fit 1-Comp Models ---
    try:
        if has_sn:
            try:
                # Additive composite model: Gself(Δν) + L1(Δν)
                modelSN1L  = sn + model1L
                paramsSN1L = (p_sn +  # ⟵ self-noise block
                              model1L.make_params( # ⟵ new copy of Lorentzian parms
                                  l1_gamma1=0.05,  min=1e-6,
                                  l1_m1   =0.8,    min=0,
                                  l1_c1   =0.0))

                fit_results['fit_sn_1c_lor'] = modelSN1L.fit(
                    y_data, paramsSN1L, x=x_data, weights=weights)
            except Exception as e:
                log.debug(f\"Self-noise+1-comp Lorentzian fit failed: {e}\")
                fit_results['fit_sn_1c_lor'] = None
                
        else:
            model1L = Model(lorentzian_model_1_comp, prefix='l1_')
            params1L = model1L.make_params(l1_gamma1=0.05, l1_m1=0.8, l1_c1=0)
            params1L['l1_gamma1'].set(min=1e-6); params1L['l1_m1'].set(min=0)
            fit_results['fit_1c_lor'] = model1L.fit(y_data, params1L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"1-comp Lorentzian fit failed: {e}")
        fit_results['fit_1c_lor'] = None
    
    try:
        model1G = Model(gaussian_model_1_comp, prefix='g1_')
        params1G = model1G.make_params(g1_sigma1=0.05, g1_m1=0.8, g1_c1=0)
        params1G['g1_sigma1'].set(min=1e-6); params1G['g1_m1'].set(min=0)
        fit_results['fit_1c_gauss'] = model1G.fit(y_data, params1G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"1-comp Gaussian fit failed: {e}")
        fit_results['fit_1c_gauss'] = None

    # --- Fit 2-Comp Models (Constrained) ---
    try:
        model2L = Model(lorentzian_model_2_comp, prefix='l2_')
        params2L = model2L.make_params(l2_gamma1=0.01, l2_m1=0.5, l2_c2=0)
        params2L['l2_gamma1'].set(min=1e-6); params2L['l2_m1'].set(min=0)
        params2L['l2_m2'].set(value=0.5, min=0)
        params2L.add('l2_gamma_factor', value=5, min=1.01) # Must be > 1
        params2L['l2_gamma2'].set(expr='l2_gamma1 * l2_gamma_factor')
        fit_results['fit_2c_lor'] = model2L.fit(y_data, params2L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp Lorentzian fit failed: {e}")
        fit_results['fit_2c_lor'] = None
        
    try:
        model2G = Model(gaussian_model_2_comp, prefix='g2_')
        params2G = model2G.make_params(g2_sigma1=0.01, g2_m1=0.5, g2_c2=0)
        params2G['g2_sigma1'].set(min=1e-6); params2G['g2_m1'].set(min=0)
        params2G['g2_m2'].set(value=0.5, min=0)
        params2G.add('g2_sigma_factor', value=5, min=1.01)
        params2G['g2_sigma2'].set(expr='g2_sigma1 * g2_sigma_factor')
        fit_results['fit_2c_gauss'] = model2G.fit(y_data, params2G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp Gaussian fit failed: {e}")
        fit_results['fit_2c_gauss'] = None

    # --- Fit Mixed & Unresolved Models ---
    try:
        # No ordering constraint needed; functional forms are different, preventing degeneracy.
        modelGL = Model(gauss_plus_lor_model, prefix='gl_')
        paramsGL = modelGL.make_params(gl_sigma1=0.01, gl_m1=0.5, gl_gamma2=0.1, gl_m2=0.5, gl_c=0)
        paramsGL['gl_sigma1'].set(min=1e-6); paramsGL['gl_m1'].set(min=0)
        paramsGL['gl_gamma2'].set(min=1e-6); paramsGL['gl_m2'].set(min=0)
        fit_results['fit_2c_mixed'] = modelGL.fit(y_data, paramsGL, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"Gauss+Lor fit failed: {e}")
        fit_results['fit_2c_mixed'] = None
        
    try:
        model2U = Model(two_screen_unresolved_model, prefix='tsu_')
        params2U = model2U.make_params(tsu_gamma1=0.01, tsu_m1=0.5, tsu_c=0)
        params2U['tsu_gamma1'].set(min=1e-6); params2U['tsu_m1'].set(min=0)
        params2U['tsu_m2'].set(value=0.5, min=0)
        params2U.add('tsu_gamma_factor', value=5, min=1.01)
        params2U['tsu_gamma2'].set(expr='tsu_gamma1 * tsu_gamma_factor')
        fit_results['fit_2c_unresolved'] = model2U.fit(y_data, params2U, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"2-comp unresolved screen fit failed: {e}")
        fit_results['fit_2c_unresolved'] = None

    # --- Fit 3-Comp Models (Constrained) ---
    try:
        model3L = Model(lorentzian_model_3_comp, prefix='l3_')
        params3L = model3L.make_params(l3_gamma1=0.01, l3_m1=0.3, l3_c3=0)
        params3L['l3_gamma1'].set(min=1e-6); params3L['l3_m1'].set(min=0)
        params3L['l3_m2'].set(value=0.3, min=0)
        params3L['l3_m3'].set(value=0.3, min=0)
        params3L.add('l3_gamma_factor2', value=5, min=1.01)
        params3L.add('l3_gamma_factor3', value=5, min=1.01)
        params3L['l3_gamma2'].set(expr='l3_gamma1 * l3_gamma_factor2')
        params3L['l3_gamma3'].set(expr='l3_gamma2 * l3_gamma_factor3')
        fit_results['fit_3c_lor'] = model3L.fit(y_data, params3L, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"3-comp Lorentzian fit failed: {e}")
        fit_results['fit_3c_lor'] = None

    try:
        model3G = Model(gaussian_model_3_comp, prefix='g3_')
        params3G = model3G.make_params(g3_sigma1=0.01, g3_m1=0.3, g3_c3=0)
        params3G['g3_sigma1'].set(min=1e-6); params3G['g3_m1'].set(min=0)
        params3G['g3_m2'].set(value=0.3, min=0)
        params3G['g3_m3'].set(value=0.3, min=0)
        params3G.add('g3_sigma_factor2', value=5, min=1.01)
        params3G.add('g3_sigma_factor3', value=5, min=1.01)
        params3G['g3_sigma2'].set(expr='g3_sigma1 * g3_sigma_factor2')
        params3G['g3_sigma3'].set(expr='g3_sigma2 * g3_sigma_factor3')
        fit_results['fit_3c_gauss'] = model3G.fit(y_data, params3G, x=x_data, weights=weights)
    except Exception as e:
        log.debug(f"3-comp Gaussian fit failed: {e}")
        fit_results['fit_3c_gauss'] = None
        
    return fit_results