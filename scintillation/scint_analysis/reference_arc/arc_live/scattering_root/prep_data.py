import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

def _get_main_peak_lim(timeseries, threshold):
    """
    Finds the start and end indices of the most significant contiguous
    region in a 1D profile above a given threshold.
    
    Significance is defined by the integrated S/N (the sum of the
    profile values within the contiguous region).

    Args:
        timeseries (np.ndarray): 1D array of the time series or spectrum.
        threshold (float): The floor level above which to search for peaks.

    Returns:
        tuple: A tuple containing the start and end indices (t0, t1) of the
               main peak. Returns (None, None) if no peak is found.
    """
    # Find all regions above the threshold using scipy's labeling function
    labeled_array, num_features = label(timeseries > threshold)
    
    if num_features == 0:
        return None, None

    # Find the "brightest" feature (peak) by summing the profile values
    # within each labeled region.
    peak_fluences = [np.sum(timeseries[labeled_array == i]) for i in range(1, num_features + 1)]
    brightest_feature_label = np.argmax(peak_fluences) + 1
    
    # Get the start and end indices corresponding to the brightest feature
    peak_indices = np.where(labeled_array == brightest_feature_label)[0]
    
    # Return the start index and the index *after* the end of the peak
    return peak_indices[0], peak_indices[-1] + 1


def get_burst_envelope(dynamic_spectrum, threshold=5, pad=0.0, downsample_factor=1, diagnostic_plots=False):
    """
    Generalized implementation to find the burst envelope from a 2D power array
    (frequency, time) using only NumPy and SciPy.

    Parameters
    ----------
    dynamic_spectrum: np.ndarray
       2D Power array (dynamic spectrum) with shape (frequency, time).
    threshold: float
       The S/N threshold for identifying signal regions.
    pad: float
       Fractional padding to add to the start and end of the envelope.
    downsample_factor: int
        Factor by which to downsample the time axis for analysis.
    diagnostic_plots: bool
       If True, generates a plot of the profile and the identified envelope.

    Returns
    -------
    lims: list
       A list containing the start and end time bins of the burst envelope.
    """
    # 1. Create the 1D burst profile from the 2D power array.
    if dynamic_spectrum.ndim != 2:
        raise ValueError("Input 'dynamic_spectrum' array must be 2-dimensional (frequency, time)")
        
    # Downsample in time if required.
    if downsample_factor > 1:
        remainder = dynamic_spectrum.shape[1] % downsample_factor
        if remainder > 0:
            I_for_scrunch = dynamic_spectrum[:, :-remainder]
        else:
            I_for_scrunch = dynamic_spectrum
        
        # Reshape and average over the new time bin axis
        I_scrunched = np.nanmean(
            I_for_scrunch.reshape(I_for_scrunch.shape[0], I_for_scrunch.shape[1] // downsample_factor, downsample_factor),
            axis=-1
        )
        prof = np.nanmean(I_scrunched, axis=0)
    else:
        prof = np.nanmean(dynamic_spectrum, axis=0)

    # 2. Iteratively find and remove peaks to define the noise floor.
    floor = prof.copy()
    
    # Normalize the profile to calculate S/N
    prof_median = np.nanmedian(floor)
    prof_std = np.nanstd(floor)
    
    snr_prof = (prof - prof_median) / prof_std
    floor = (floor - prof_median) / prof_std
    
    while True:
        # Find the main peak in the current floor
        peak_t0, peak_t1 = _get_main_peak_lim(floor, threshold=threshold)
        
        if peak_t0 is None:  # No more significant peaks found
            break
            
        # Blank the identified peak region by setting it to NaN
        floor[peak_t0:peak_t1] = np.nan
        
        # Check if any data remains to prevent infinite loops
        if np.isnan(floor).all():
            break

    # 3. Define the envelope as the full extent of the blanked (NaN) regions.
    idx_is_signal = np.isnan(floor)
    if not np.any(idx_is_signal):
        print("Warning: No burst envelope found above threshold.")
        return [0, 0]
        
    try:
        lims = np.array([
            np.where(idx_is_signal)[0].min(),
            np.where(idx_is_signal)[0].max()
        ])
    except ValueError:
        return [0, len(prof)]
        
    # 4. Apply padding to the limits.
    pad_samples = int((lims[1] - lims[0]) * pad)
    lims[0] = max(0, lims[0] - pad_samples)
    lims[1] = min(len(prof), lims[1] + pad_samples)
    
    # 5. Rescale limits back to the original time resolution.
    final_lims = lims * downsample_factor

    # 6. Generate diagnostic plots if requested.
    if diagnostic_plots:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(snr_prof)) * downsample_factor, snr_prof, label='S/N Profile')
        plt.axvline(final_lims[0], c="k", ls="--", label='Envelope')
        plt.axvline(final_lims[1], c="k", ls="--")
        plt.xlabel(f"Time Bins (Original Resolution)")
        plt.ylabel("Signal-to-Noise Ratio")
        plt.title("Burst Envelope Identification")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return final_lims.tolist()

def _get_detrended_snr(stat_series, threshold=5.0):
    """
    Helper function to normalize, detrend, and re-normalize a 1D array of statistics.

    Args:
        stat_series (np.ndarray): A 1D array of statistics (e.g., mean or std dev per channel).
        threshold (float): The sigma level used to exclude outliers for the initial linear fit.

    Returns:
        np.ndarray: The final, detrended S/N series.
    """
    # Create a copy to avoid modifying the original array
    series = np.copy(stat_series)
    
    # First normalization to identify a clean subset for fitting
    median_val = np.nanmedian(series)
    std_val = np.nanstd(series)
    if std_val == 0: return np.zeros_like(series) # Avoid division by zero
    
    snr_series = (series - median_val) / std_val
    
    # Find a "clean" subset of the data that doesn't include strong outliers
    clean_indices = np.where(np.abs(snr_series) < threshold)[0]
    
    # If not enough clean points, cannot detrend, so return initial SNR
    if len(clean_indices) < 10:
        return snr_series
        
    # Perform a linear fit on the clean subset to model the trend
    x_axis = np.arange(len(series))
    coeffs = np.polyfit(x_axis[clean_indices], series[clean_indices], 1)
    linear_trend = np.polyval(coeffs, x_axis)
    
    # Subtract the trend and re-normalize to get the final S/N
    detrended_series = series - linear_trend
    final_std = np.nanstd(detrended_series)
    if final_std == 0: return np.zeros_like(detrended_series)

    final_snr = detrended_series / final_std
    return final_snr


def mask_dynamic_spectrum_rfi(
    dynamic_spectrum, 
    off_burst_indices, 
    thres_mean=5.0, 
    thres_std=5.0,
    thres_time=7.0,
    show_diagnostic_plots=False
):
    """
    A generalized function to mask RFI from a 2D real-valued dynamic spectrum.
    This replaces the need for `data_dedisp_derip_filled_masked` by focusing
    solely on masking, with no external library dependencies.

    Parameters
    ----------
    dynamic_spectrum : np.ndarray
        2D dynamic spectrum of shape (frequency, time).
    off_burst_indices : tuple
        A tuple (start_bin, end_bin) defining the noise-only region.
    thres_mean : float
        The sigma threshold for flagging channels based on their mean.
    thres_std : float
        The sigma threshold for flagging channels based on their standard deviation.
    thres_time : float
        The sigma threshold for flagging impulsive RFI in the time domain.
    show_diagnostic_plots : bool
        If True, displays plots of the masks and data.

    Returns
    -------
    np.ma.MaskedArray
        The RFI-masked 2D dynamic spectrum.
    """
    if dynamic_spectrum.ndim != 2:
        raise ValueError("Input 'dynamic_spectrum' must be a 2D array (frequency, time).")

    # --- 1. Frequency Domain RFI Flagging (Channel Masking) ---
    # Use the off-burst section to characterize the noise properties of each channel
    noise_data = dynamic_spectrum[:, off_burst_indices[0]:off_burst_indices[1]]
    
    channel_mask = np.zeros(dynamic_spectrum.shape[0], dtype=bool)
    
    # Iteratively find and flag bad channels
    for _ in range(5): # Iterate a few times to catch progressively fainter RFI
        # Temporarily mask already-flagged channels
        noise_data_masked = np.ma.masked_array(noise_data, mask=np.tile(channel_mask, (noise_data.shape[1], 1)).T)

        # Calculate statistics only on unmasked data
        channel_means = np.ma.mean(noise_data_masked, axis=1).filled(np.nan)
        channel_stds = np.ma.std(noise_data_masked, axis=1).filled(np.nan)

        # Get detrended S/N for both mean and std
        snr_means = _get_detrended_snr(channel_means, threshold=thres_mean)
        snr_stds = _get_detrended_snr(channel_stds, threshold=thres_std)

        # Flag channels that are outliers in either metric
        newly_flagged = (np.abs(snr_means) > thres_mean) | (np.abs(snr_stds) > thres_std)
        
        if not np.any(newly_flagged): # Stop if no new RFI is found
            break
        channel_mask |= newly_flagged

    # --- 2. Time Domain RFI Flagging (Impulsive RFI) ---
    # Create a time series by averaging over the RFI-free channels
    freq_masked_data = np.ma.masked_array(dynamic_spectrum, mask=np.tile(channel_mask, (dynamic_spectrum.shape[1], 1)).T)
    time_series = np.ma.mean(freq_masked_data, axis=0).filled(np.nan)
    
    # Normalize the time series using a robust statistic (median absolute deviation)
    # This is better for data with strong, impulsive outliers.
    ts_median = np.nanmedian(time_series)
    ts_mad = median_abs_deviation(time_series, nan_policy='omit')
    if ts_mad == 0: ts_mad = 1e-9 # Avoid division by zero
    
    # Calculate robust z-score
    robust_z_score = 0.6745 * (time_series - ts_median) / ts_mad
    time_mask = np.abs(robust_z_score) > thres_time

    # --- 3. Combine Masks and Create Final Masked Array ---
    # Use broadcasting to combine the 1D channel and time masks into a 2D mask
    final_mask_2d = np.logical_or(channel_mask[:, np.newaxis], time_mask[np.newaxis, :])
    masked_power = np.ma.masked_array(dynamic_spectrum, mask=final_mask_2d)

    # --- 4. Diagnostic Plots ---
    if show_diagnostic_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot time domain mask
        axes[0, 0].plot(time_series, 'k-', label='Frequency-Averaged Profile')
        axes[0, 0].scatter(np.where(time_mask)[0], time_series[time_mask], color='r', label='Flagged Time Bins')
        axes[0, 0].set_title("Time Domain RFI Flagging")
        axes[0, 0].set_xlabel("Time Bin")
        axes[0, 0].legend()

        # Plot channel domain mask
        axes[0, 1].plot(channel_means, 'k-')
        axes[0, 1].scatter(np.where(channel_mask)[0], channel_means[channel_mask], color='r')
        axes[0, 1].set_title("Frequency Domain RFI Flagging")
        axes[0, 1].set_xlabel("Channel Index")
        axes[0, 1].set_ylabel("Mean (Off-Burst)")

        # Plot original and masked dynamic spectra
        vmax = np.nanpercentile(dynamic_spectrum, 99) # Consistent color scale
        axes[1, 0].imshow(dynamic_spectrum, aspect='auto', origin='lower', vmax=vmax, cmap='viridis')
        axes[1, 0].set_title("Original Dynamic Spectrum")
        
        axes[1, 1].imshow(masked_power, aspect='auto', origin='lower', vmax=vmax, cmap='viridis')
        axes[1, 1].set_title("Masked Dynamic Spectrum")

        plt.tight_layout()
        plt.show()

    return masked_power
