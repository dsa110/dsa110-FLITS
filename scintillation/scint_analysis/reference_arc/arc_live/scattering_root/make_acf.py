import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from tqdm import tqdm

# --- Core Lorentzian Models ---
def lorentzian_model_with_offset(x, decorrelation_bandwidth_mhz, modulation_index, constant_offset):
    """
    A single Lorentzian model with a constant offset.
    
    Parameters are named for clarity in fitting reports.
    """
    return modulation_index**2 / (1 + (x / decorrelation_bandwidth_mhz)**2) + constant_offset


def calculate_acf(spectrum_1d, off_burst_spectrum_mean=None, max_lag_bins=None):
    """
    Calculates the one-sided autocorrelation function of a spectrum using
    efficient NumPy operations.

    Parameters
    ----------
    spectrum_1d : np.ma.MaskedArray
        The 1D spectrum to autocorrelate. Must be a masked array.
    off_burst_spectrum_mean : float, optional
        The mean of the off-burst spectrum, used for normalization.
    max_lag_bins : int, optional
        The maximum number of bins to compute the ACF out to.

    Returns
    -------
    acf : np.ndarray
        The one-sided autocorrelation function (from lag 1 onwards).
    """
    if not isinstance(spectrum_1d, np.ma.MaskedArray):
        raise TypeError("Input 'spectrum_1d' must be a NumPy masked array.")

    valid_spectrum_values = spectrum_1d.compressed()
    if valid_spectrum_values.size == 0:
        return np.array([])

    mean_on_burst = np.mean(valid_spectrum_values)
    
    # Define the normalization denominator for measuring the modulation index
    if off_burst_spectrum_mean is not None:
        normalization_denominator = (mean_on_burst - off_burst_spectrum_mean)**2
    else:
        normalization_denominator = mean_on_burst**2

    if normalization_denominator == 0:
        normalization_denominator = 1.0

    # Prepare the mean-subtracted spectrum, using NaN for masked values
    mean_subtracted_spec = spectrum_1d.filled(np.nan)
    mean_subtracted_spec -= mean_on_burst
    
    num_channels = len(mean_subtracted_spec)
    if max_lag_bins is None:
        max_lag_bins = num_channels
    
    # Vectorized ACF calculation loop
    lags_to_compute = np.arange(1, max_lag_bins)
    acf = np.zeros(len(lags_to_compute))

    for i, current_lag in enumerate(tqdm(lags_to_compute, desc="Calculating ACF", leave=False)):
        # Create two shifted versions of the array
        original_segment = mean_subtracted_spec[:-current_lag]
        lagged_segment = mean_subtracted_spec[current_lag:]
        
        # The product will be NaN if either element was originally masked
        product_array = original_segment * lagged_segment
        
        # Sum only the valid (non-NaN) products
        numerator = np.nansum(product_array)
        # Count only the pairs where both elements were valid
        num_valid_pairs = np.sum(~np.isnan(product_array))
        
        if num_valid_pairs > 0:
            acf[i] = numerator / (num_valid_pairs * normalization_denominator)
            
    return acf

def compute_and_fit_acf(
    dynamic_spectrum, 
    frequencies_mhz,
    off_burst_spectrum_mean=None,
    on_burst_time_bins=None,
    max_lag_mhz=None,
    fit_lagrange_mhz=10.0,
    show_diagnostic_plot=False
):
    """
    A wrapper that prepares a spectrum, computes its ACF, and fits a Lorentzian.

    Parameters
    ----------
    dynamic_spectrum : np.ndarray or np.ma.MaskedArray
        Input data, can be 1D (already a spectrum) or 2D (dynamic spectrum).
    frequencies_mhz : np.ndarray
        Array of channel frequencies in MHz.
    off_burst_spectrum_mean : float, optional
        Mean of the off-burst noise, for proper normalization.
    on_burst_time_bins : tuple, optional
        A tuple (start_bin, end_bin) required if input is a 2D dynamic spectrum.
    max_lag_mhz : float, optional
        The maximum lag to compute the ACF out to, in MHz.
    fit_lagrange_mhz : float
        The range of lags around zero (in MHz) to use for the model fit.
    show_diagnostic_plot : bool
        If True, displays a plot of the ACF and its fit.

    Returns
    -------
    tuple: (full_acf, lags_mhz, lmfit_result)
    """
    # 1. Prepare the 1D Spectrum
    if dynamic_spectrum.ndim == 1:
        spectrum_1d = dynamic_spectrum
    elif dynamic_spectrum.ndim == 2:
        if on_burst_time_bins is None:
            raise ValueError("'on_burst_time_bins' must be provided for 2D data.")
        spectrum_1d = np.ma.mean(dynamic_spectrum[:, on_burst_time_bins[0]:on_burst_time_bins[1]], axis=1)
    else:
        raise ValueError("Input 'dynamic_spectrum' must be a 1D or 2D array.")
        
    if not isinstance(spectrum_1d, np.ma.MaskedArray):
        spectrum_1d = np.ma.masked_where(spectrum_1d == 0, spectrum_1d)

    # 2. Compute the ACF
    channel_width_mhz = np.abs(frequencies_mhz[1] - frequencies_mhz[0])
    max_lag_bins = int(max_lag_mhz / channel_width_mhz) if max_lag_mhz is not None else None
    
    one_sided_acf = calculate_acf(
        spectrum_1d, 
        off_burst_spectrum_mean=off_burst_spectrum_mean, 
        max_lag_bins=max_lag_bins
    )
    
    # 3. Create a two-sided ACF for fitting and plotting
    positive_lags_mhz = np.arange(1, len(one_sided_acf) + 1) * channel_width_mhz
    full_acf = np.concatenate((one_sided_acf[::-1], one_sided_acf))
    lags_mhz = np.concatenate((-positive_lags_mhz[::-1], positive_lags_mhz))

    # 4. Fit the Lorentzian model
    fit_result = None
    try:
        lorentzian_model = Model(lorentzian_model_with_offset)
        fit_mask = np.abs(lags_mhz) <= fit_lagrange_mhz
        
        result = lorentzian_model.fit(
            full_acf[fit_mask], 
            x=lags_mhz[fit_mask], 
            decorrelation_bandwidth_mhz=0.01,
            modulation_index=1.0,
            constant_offset=0.0
        )
        fit_result = result
    except Exception as e:
        print(f"Warning: Lorentzian fit failed. {e}")

    # 5. Optional Diagnostic Plotting
    if show_diagnostic_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(lags_mhz, full_acf, 'k-', alpha=0.7, label='ACF Data')
        if fit_result:
            fit_params = fit_result.params
            gamma_val = fit_params['decorrelation_bandwidth_mhz'].value
            plt.plot(lags_mhz, fit_result.eval(x=lags_mhz), 'r--', label='Lorentzian Fit')
            plt.title(f"Scintillation Bandwidth = {gamma_val*1000:.2f} kHz")
            plt.xlim(-5 * gamma_val, 5 * gamma_val)
        plt.xlabel("Frequency Lag (MHz)")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return full_acf, lags_mhz, fit_result


def analyze_acf_per_subband(
    full_spectrum,
    full_frequencies_mhz,
    off_burst_spectrum=None,
    num_subbands=8,
    use_snr_subbanding=False,
    max_lag_mhz=1.0,
    output_plot_path=None
):
    """
    Divides a spectrum into sub-bands and analyzes the ACF for each one.
    
    Returns a dictionary of results for easier handling downstream.
    """
    if not isinstance(full_spectrum, np.ma.MaskedArray):
        full_spectrum = np.ma.masked_where(full_spectrum == 0, full_spectrum)

    analysis_results = {
        'subband_acfs': [],
        'subband_lags_mhz': [],
        'subband_fits': [],
        'subband_center_freqs_mhz': [],
        'subband_masked_fraction': []
    }
    
    total_signal = np.sum(full_spectrum.compressed())
    
    # --- Sub-banding Logic ---
    start_idx = 0
    print(f"Dividing spectrum into {num_subbands} sub-bands...")
    for i in range(num_subbands):
        sub_len = len(full_spectrum) // num_subbands
        if not use_snr_subbanding:
            end_idx = start_idx + sub_len
        else:
            # Determine end_idx based on equal S/N
            cumulative_signal = 0
            target_signal = total_signal / num_subbands
            end_idx = start_idx
            while cumulative_signal < target_signal and end_idx < len(full_spectrum):
                if not full_spectrum.mask[end_idx]:
                    cumulative_signal += full_spectrum.data[end_idx]
                end_idx += 1

        if i == num_subbands - 1: # Ensure the last subband includes all remaining channels
            end_idx = len(full_spectrum)
        
        subband_spectrum = full_spectrum[start_idx:end_idx]
        subband_frequencies = full_frequencies_mhz[start_idx:end_idx]
        sub_off_mean = np.ma.mean(off_burst_spectrum[start_idx:end_idx]) if off_burst_spectrum is not None else None
        
        print(f"  Sub-band {i+1}: Freq Range {subband_frequencies.min():.1f} - {subband_frequencies.max():.1f} MHz")
        
        # --- Call the unified function for the current sub-band ---
        acf, lags, fit_result = compute_and_fit_acf(
            subband_spectrum,
            subband_frequencies,
            off_burst_spectrum_mean=sub_off_mean,
            on_burst_time_bins=[0, 1], # Dummy value, as spectrum is already 1D
            max_lag_mhz=max_lag_mhz
        )
        
        analysis_results['subband_acfs'].append(acf)
        analysis_results['subband_lags_mhz'].append(lags)
        analysis_results['subband_fits'].append(fit_result)
        analysis_results['subband_center_freqs_mhz'].append(np.mean(subband_frequencies))
        analysis_results['subband_masked_fraction'].append(np.sum(subband_spectrum.mask) / subband_spectrum.size)

        start_idx = end_idx

    # --- Plotting ---
    if output_plot_path:
        plt.figure(figsize=(8, 10))
        cmap = plt.get_cmap('plasma')
        for i in range(num_subbands):
            rgba = cmap(i / (num_subbands - 1))
            offset = i * 1.5
            plt.plot(analysis_results['subband_lags_mhz'][i], analysis_results['subband_acfs'][i] + offset, color=rgba, alpha=0.8)
            fit = analysis_results['subband_fits'][i]
            if fit:
                plt.plot(analysis_results['subband_lags_mhz'][i], fit.eval(x=analysis_results['subband_lags_mhz'][i]) + offset, 'k--', alpha=0.6)

        plt.yticks([(i * 1.5) for i in range(num_subbands)], [f"{cf:.1f}" for cf in analysis_results['subband_center_freqs_mhz']])
        plt.ylabel("Center Frequency (MHz)")
        plt.xlabel("Frequency Lag (MHz)")
        plt.xlim(-max_lag_mhz, max_lag_mhz)
        plt.title("ACF per Sub-band")
        plt.savefig(output_plot_path)
        plt.show()

    return analysis_results
