# -*- coding: utf-8 -*-
"""
Functions for analyzing scintillation in Fast Radio Bursts (FRBs),
including upchannelization, ACF calculation, and model fitting.
"""

import numpy as np
import math
import scipy.constants as cons
from scipy.fft import fft, fftshift
from lmfit import minimize, Parameters, Model, Minimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# Define constants for CHIME frequency band (can be adjusted if needed)
FREQ_TOP_MHZ = 800.1953125
FREQ_BOTTOM_MHZ = 400.1953125
TOTAL_CHANNELS = 1024 # Standard number of CHIME channels before upchannelization

def upchannel(wfall, freq_id, fftsize=32, downfreq=2):
    """
    Upchannelizes baseband voltage data using an FFT-based method.

    This function takes a time-frequency data array (waterfall), performs
    an FFT along the time axis for segments of size `fftsize`, shifts the
    frequency components, optionally averages (`downfreq`), and rearranges
    the output into a higher frequency resolution array.

    Note: A similar function might exist in `kenzie_functions.py`.

    Parameters
    ----------
    wfall : np.ndarray
        Input complex voltage data array. Expected shape [freq, pol, time].
    freq_id : np.ndarray
        1D array containing the original frequency channel IDs corresponding
        to the first axis of `wfall`.
    fftsize : int, optional
        Size of the FFT window along the time axis. Default is 32.
    downfreq : int, optional
        Factor by which to average frequency channels *after* the FFT shift.
        Effectively reduces the final number of upchannels per original channel.
        Default is 2.

    Returns
    -------
    spec : np.ndarray
        Upchannelized complex voltage data. Shape [pol, nblock, nchan_up].
        `nblock` is the number of time blocks after processing.
        `nchan_up` is the total number of upchannelized frequency channels.
    f_upchan_final : np.ndarray
        1D array of the central frequencies (in MHz) for the upchannelized
        channels present in the output `spec`.
    chan_id_upchan_final : np.ndarray
        1D array of the integer channel IDs for the upchannelized channels,
        relative to a full `TOTAL_CHANNELS * upchan` grid.
    """
    # Input validation
    if wfall.ndim != 3:
        raise ValueError(f"Input wfall must be 3D (freq, pol, time), got shape {wfall.shape}")
    if freq_id.ndim != 1:
        raise ValueError(f"Input freq_id must be 1D, got shape {freq_id.shape}")
    if wfall.shape[0] != len(freq_id):
         raise ValueError(f"Dimension mismatch: wfall.shape[0] ({wfall.shape[0]}) != len(freq_id) ({len(freq_id)})")
    if fftsize <= 0 or not isinstance(fftsize, int):
         raise ValueError("fftsize must be a positive integer.")
    if downfreq <= 0 or not isinstance(downfreq, int):
         raise ValueError("downfreq must be a positive integer.")
    if fftsize % downfreq != 0:
        raise ValueError("fftsize must be divisible by downfreq.")

    # Swap axes ordering to (pol, time, chan) for easier processing
    wfall_proc = np.swapaxes(wfall, 0, 1) # Now [pol, freq, time]
    wfall_proc = np.swapaxes(wfall_proc, 1, 2) # Now [pol, time, freq]

    npol, nsamp, nchan_in = wfall_proc.shape

    # --- Parameters ---
    # No time averaging of complex voltages is performed here.
    downtime = 1
    # Upchannelization factor per original channel
    upchan = fftsize // downfreq
    # Number of time blocks after upchannelization
    # Integer division truncates any partial final block
    nblock = nsamp // (fftsize * downtime)
    # Total number of upchannelized frequency channels in the output
    nchan_up = nchan_in * upchan

    # --- Frequency Calculation ---
    # Create frequency array for the full potential upchannelized band
    f_upchan_bandtot = np.linspace(
        FREQ_TOP_MHZ, FREQ_BOTTOM_MHZ, upchan * TOTAL_CHANNELS
    )

    # --- Initialization ---
    spec = np.zeros((npol, nblock, nchan_up), dtype=np.complex64)
    # Store the final upchannel IDs corresponding to the input freq_id
    chan_id_upchan_map = np.zeros((nchan_in, upchan), dtype=int)

    # --- Upchannelization Loop ---
    for pol in range(npol):
        for bi in range(nblock):
            # Process block by block, channel by channel
            # Start index for time samples in this block
            time_start = bi * fftsize * downtime # downtime is 1
            time_end = time_start + fftsize

            for chidx_in in range(nchan_in):
                # Extract the time series segment for this block and input channel
                ts_segment = wfall_proc[pol, time_start:time_end, chidx_in].copy()

                # Perform FFT along the time segment
                ft = fft(ts_segment)

                # Shift zero-frequency component to center
                ft_shifted = fftshift(ft)

                # Downsample (average) in the upchannelized frequency dimension
                # Reshape groups `downfreq` channels together, then average
                ft_downsampled = ft_shifted.reshape(upchan, downfreq).mean(axis=1)

                # Assign the result to the correct position in the output spectrum
                spec_chan_start = chidx_in * upchan
                spec_chan_end = spec_chan_start + upchan
                spec[pol, bi, spec_chan_start:spec_chan_end] = ft_downsampled

                # Calculate and store the corresponding upchannel IDs (only needs doing once)
                if pol == 0 and bi == 0:
                    original_chan_id = freq_id[chidx_in]
                    chan_id_upchan_map[chidx_in, :] = np.arange(
                        upchan * original_chan_id, upchan * original_chan_id + upchan, 1
                    )

    # Flatten the channel ID map and select the corresponding frequencies
    chan_id_upchan_final = chan_id_upchan_map.ravel()
    # Ensure indices are within bounds of the full frequency array
    valid_indices = chan_id_upchan_final < len(f_upchan_bandtot)
    chan_id_upchan_final = chan_id_upchan_final[valid_indices]
    f_upchan_final = f_upchan_bandtot[chan_id_upchan_final]

    # Adjust spec array if some indices were invalid (shouldn't happen with standard CHIME IDs)
    if not np.all(valid_indices):
         print("Warning: Some calculated upchannel IDs were out of bounds.")
         # This part needs careful handling if invalid indices occur.
         # For now, assume valid indices cover the relevant part of spec.
         # A more robust solution might involve masking or resizing spec.
         pass # Placeholder

    return spec, f_upchan_final, chan_id_upchan_final


def make_scallop_model(off_data, fftsize, downfreq):
    """
    Creates a model of the instrumental bandpass ripple ("scallop") using
    off-burst (noise) data that has been upchannelized. Also identifies
    channels with strong RFI spikes in the noise.

    Parameters
    ----------
    off_data : np.ndarray
        Upchannelized complex voltage array containing *off-burst* data.
        Expected shape: [pol, time_block, freq_upchan].
    fftsize : int
        FFT size used during upchannelization.
    downfreq : int
        Downsampling factor used during upchannelization.

    Returns
    -------
    model : np.ndarray
        1D array representing the average scallop shape, tiled to match the
        full upchannelized frequency axis dimension of `off_data`.
    inds_rfi : np.ndarray
        1D array of indices corresponding to frequency channels identified
        as having strong RFI spikes (above 3-sigma) in the noise spectrum.
    """
    if off_data.ndim != 3:
        raise ValueError(f"Input off_data must be 3D (pol, time, freq), got shape {off_data.shape}")

    # Calculate average power spectrum from off-burst data
    noise_power = np.abs(off_data**2)
    # Average over polarization, then transpose to [freq, time]
    I_noise = np.mean(noise_power, axis=0).T
    # Average over time blocks to get the noise spectrum
    spec_noise = np.nanmean(I_noise, axis=1)

    # --- RFI Identification ---
    # Calculate robust statistics (median/MAD) if available, otherwise mean/std
    try:
        from scipy.stats import median_abs_deviation
        noise_median = np.nanmedian(spec_noise)
        # scale=1.4826 for normality
        noise_mad = median_abs_deviation(spec_noise, nan_policy='omit', scale=1.4826)
        if noise_mad == 0: # Handle case where MAD is zero
             noise_mad = np.nanstd(spec_noise) # Fallback to std dev
        spec_noise_norm = (spec_noise - noise_median) / noise_mad
    except ImportError:
        print("Warning: scipy.stats.median_abs_deviation not found. Using mean/std for RFI flagging.")
        noise_mean = np.nanmean(spec_noise)
        noise_std = np.nanstd(spec_noise)
        if noise_std == 0: # Handle case with no variation
             noise_std = 1.0
        spec_noise_norm = (spec_noise - noise_mean) / noise_std

    # Identify channels exceeding a 3-sigma threshold
    inds_rfi = np.where(np.abs(spec_noise_norm) > 3)[0]

    # --- Scallop Model Creation ---
    # Temporarily mask identified RFI channels before modeling scallop
    spec_noise_masked = np.ma.masked_where(np.isnan(spec_noise), spec_noise)
    spec_noise_masked[inds_rfi] = np.ma.masked # Mask RFI

    # Reshape the spectrum according to the upchannelization factor
    upchan = fftsize // downfreq
    nchan_in_orig = spec_noise_masked.shape[0] // upchan # Infer original number of channels
    # Ensure the reshaping is possible
    if spec_noise_masked.shape[0] % upchan != 0:
        # This might happen if the input `off_data` didn't cover the full band initially
        print(f"Warning: Noise spectrum length ({spec_noise_masked.shape[0]}) "
              f"not divisible by upchannel factor ({upchan}). Truncating.")
        trunc_len = (spec_noise_masked.shape[0] // upchan) * upchan
        spec_noise_masked_reshape = spec_noise_masked[:trunc_len].reshape(nchan_in_orig, upchan)
    else:
         spec_noise_masked_reshape = spec_noise_masked.reshape(nchan_in_orig, upchan)


    # Calculate the average shape within each original channel's upchan block
    model_scallop_single = np.ma.mean(spec_noise_masked_reshape, axis=0)

    # Tile this single scallop shape across the entire band
    # Use the original inferred number of channels for tiling
    model = np.tile(model_scallop_single.data, nchan_in_orig) # Use .data to get array from masked array

    # Handle potential length mismatch if truncation occurred
    if len(model) != len(spec_noise):
        print(f"Warning: Tiled model length ({len(model)}) differs from "
              f"original spectrum length ({len(spec_noise)}). Adjusting model length.")
        if len(model) > len(spec_noise):
            model = model[:len(spec_noise)]
        else:
            # Pad model if it's shorter (less likely but possible)
            padding = np.zeros(len(spec_noise) - len(model))
            model = np.concatenate((model, padding)) # Or use np.pad

    # Optional: Return the corrected noise spectrum (noise divided by model)
    # spec_noise_masked_corr = spec_noise_masked / model # Element-wise division
    # spec_noise_masked_corr = np.ma.masked_where(spec_noise_masked_corr == 0, spec_noise_masked_corr)

    return model, inds_rfi


def shift(v, i, nchan):
    """
    Helper function to circularly shift a 1D array `v` by `i` positions,
    padding with zeros for ACF calculation.

    Parameters
    ----------
    v : np.ndarray
        1D array to shift.
    i : int
        Shift amount (can be positive or negative).
    nchan : int
        Original number of channels (length of `v`), used for calculating
        the effective shift index within the padded array.

    Returns
    -------
    r : np.ndarray
        Zero-padded array of size 3*n containing `v` shifted by `i`.
    """
    n = len(v)
    if n != nchan:
        # This check is important if called with slices of original data
        # print(f"Warning: Length of input vector v ({n}) does not match nchan ({nchan}).")
        pass # Allow mismatch for flexibility, but be aware

    r = np.zeros(3 * n)
    # Calculate the starting index in the padded array `r`
    # Adding (nchan - 1) centers the unshifted array `v` in the middle third
    # effectively handling negative lags correctly.
    start_index = int(i + nchan - 1)

    # Place the shifted array `v` into `r`
    # Ensure indices stay within the bounds of `r`
    end_index = start_index + n
    v_start = 0
    v_end = n

    # Adjust if shift pushes data outside the 3*n buffer
    if start_index < 0:
        v_start = -start_index
        start_index = 0
    if end_index > 3 * n:
        v_end = n - (end_index - 3 * n)
        end_index = 3 * n

    if start_index < end_index and v_start < v_end: # Check if there's anything to copy
        r[start_index:end_index] = v[v_start:v_end]

    return r

def autocorr(spec, v=None, zerolag=False, maxlag=None, offspec_mean=None):
    """
    Calculates the Auto-Correlation Function (ACF) of a 1D spectrum.

    Handles masking of invalid data points. The ACF is normalized by the
    variance of the signal (mean-subtracted spectrum).

    Note: A similar function might exist in `kenzie_functions.py`.

    Parameters
    ----------
    spec : np.ndarray or np.ma.MaskedArray
        1D input spectrum (intensity vs. frequency channel).
    v : np.ndarray, optional
        Mask array (0 for masked, 1 for valid). If None, assumes all
        data in `spec` is valid unless `spec` is already a MaskedArray.
    zerolag : bool, optional
        If True, include the zero-lag value in the output ACF.
        If False (default), exclude the zero-lag (often dominated by noise).
    maxlag : int, optional
        Maximum lag (in number of channels) to compute the ACF up to.
        If None (default), compute for all possible lags up to nchan-1.
    offspec_mean : float, optional
        If provided, use (mean(spec) - offspec_mean)^2 for normalization
        denominator, otherwise use mean(spec)^2. This is relevant if
        `spec` has already had an off-burst mean subtracted but still
        needs normalization relative to the *original* signal variance.

    Returns
    -------
    ACF : np.ndarray
        The calculated auto-correlation function for positive lags.
        Length is `maxlag` or `nchan` (if `zerolag` is True) or
        `nchan-1` (if `zerolag` is False).
    """
    nchan = len(spec)

    # --- Handle Input Spectrum and Mask ---
    if isinstance(spec, np.ma.MaskedArray):
        if v is not None:
            print("Warning: Both MaskedArray spec and mask v provided. Using spec.mask.")
        mask = ~spec.mask # Invert mask: True=valid (1), False=masked (0)
        x = spec.data # Use underlying data array
    else:
        x = np.copy(spec)
        if v is None:
            mask = np.ones(nchan, dtype=bool) # Assume all valid
        else:
            mask = v.astype(bool) # Ensure boolean mask

    # Apply mask to data (set masked values to NaN for calculations)
    x[~mask] = np.nan

    # --- Mean and Normalization ---
    x_mean_valid = np.nanmean(x) # Mean of only valid points

    if np.isnan(x_mean_valid):
        print("Warning: Mean calculation resulted in NaN. Check input data/mask.")
        return np.zeros(maxlag if maxlag is not None else (nchan if zerolag else nchan - 1)) * np.nan

    # Denominator for normalization (related to variance)
    if offspec_mean is not None:
        denom = (x_mean_valid - offspec_mean)**2
    else:
        denom = x_mean_valid**2

    if denom == 0:
        print("Warning: Normalization denominator is zero.")
        # Avoid division by zero; return NaN ACF
        return np.zeros(maxlag if maxlag is not None else (nchan if zerolag else nchan - 1)) * np.nan


    # Subtract mean from valid data points *before* ACF calculation
    x_meansub = np.copy(x)
    x_meansub[mask] -= x_mean_valid

    # --- ACF Calculation ---
    if maxlag is None:
        num_lags_calc = nchan # Calculate up to nchan-1 lag
    else:
        # Ensure maxlag is within bounds
        maxlag = min(int(maxlag), nchan)
        num_lags_calc = maxlag

    # Determine the size and start index of the output ACF array
    if zerolag:
        acf_len = num_lags_calc
        lag_start_idx = 0 # Start calculation from lag 0
    else:
        if num_lags_calc == 0: return np.array([]) # Handle edge case maxlag=0
        acf_len = num_lags_calc -1 # Exclude zero lag
        lag_start_idx = 1 # Start calculation from lag 1

    ACF = np.zeros(acf_len)

    # Convert boolean mask to float (1.0 for valid, 0.0 for masked) for multiplication
    mask_float = mask.astype(float)

    # Loop through lags
    # tqdm provides a progress bar
    for i in tqdm(range(lag_start_idx, num_lags_calc), desc="Calculating ACF", leave=False):
        # Shift the mask and mean-subtracted data
        shifted_mask = shift(mask_float, i, nchan)
        shifted_x = shift(x_meansub, i, nchan)

        # Get the unshifted versions within the padded array
        # (shift by 0 centers it in the middle)
        unshifted_mask = shift(mask_float, 0, nchan)
        unshifted_x = shift(x_meansub, 0, nchan)

        # Calculate overlap mask (product is 1 only where both are valid)
        overlap_mask = unshifted_mask * shifted_mask

        # Calculate ACF numerator: sum of product where both shifted and unshifted are valid
        numerator = np.nansum(unshifted_x * shifted_x * overlap_mask)

        # Calculate ACF denominator: product of number of valid overlapping points and variance term
        num_overlap_points = np.sum(overlap_mask)

        if num_overlap_points > 0:
            acf_index = i - lag_start_idx # Index in the output ACF array
            ACF[acf_index] = numerator / (num_overlap_points * denom)
        else:
            # Handle cases with no overlap (should be rare for small lags)
            ACF[acf_index] = np.nan # Or 0, depending on desired behavior

    return ACF


def acf_scint_plot(ds, freq_ids, freqs, time_range, lagrange_for_fit=10.,
                   diagnostic_plots=True, maxlag=None, offspec_mean=None):
    """
    Calculates the ACF of a spectrum (or dynamic spectrum averaged in time),
    fits a Lorentzian model to the central part, and optionally plots results.

    Note: A similar function might exist in `kenzie_functions.py`.

    Parameters
    ----------
    ds : np.ndarray or np.ma.MaskedArray
        Input data. Can be 1D spectrum [freq] or 2D dynamic spectrum [freq, time].
    freq_ids : np.ndarray
        1D array of frequency channel IDs corresponding to the freq axis of `ds`.
    freqs : np.ndarray
        1D array of central frequencies (MHz) corresponding to the freq axis of `ds`.
    time_range : list or tuple
        [start_bin, end_bin]. Time bins to average over if `ds` is 2D. Ignored if `ds` is 1D.
    lagrange_for_fit : float, optional
        Frequency lag range (in MHz, symmetric around 0) used for fitting the
        Lorentzian model. Default is 10 MHz.
    diagnostic_plots : bool, optional
        If True, generate plots of the ACF and the fit. Default is True.
    maxlag : float, optional
        Maximum frequency lag (in MHz) to compute the ACF out to. If None,
        computes up to half the bandwidth. Default is None.
    offspec_mean : float, optional
        Mean of an off-burst spectrum, used for ACF normalization (see `autocorr`).
        Default is None.

    Returns
    -------
    acf_full : np.ndarray
        The full calculated ACF (positive and negative lags).
    lags_mhz : np.ndarray
        The frequency lags (in MHz) corresponding to `acf_full`.
    result : lmfit.model.ModelResult, optional
        The result object from the Lorentzian fit. Returned only if fit is successful.
        Returns None if fitting fails or is skipped.
    """
    # --- Input Validation and Preparation ---
    if ds.ndim not in [1, 2]:
        raise ValueError("Input ds must be 1D or 2D.")
    if freqs.shape != freq_ids.shape or freqs.ndim != 1:
        raise ValueError("freqs and freq_ids must be 1D and have the same shape.")
    if ds.shape[0] != len(freqs):
        raise ValueError("Frequency dimension mismatch between ds and freqs/freq_ids.")

    # Calculate frequency resolution (MHz per channel)
    # Assumes regularly spaced channels, uses first two valid channels
    valid_freq_indices = np.where(np.diff(freq_ids) > 0)[0]
    if len(valid_freq_indices) < 1:
        # Fallback if only one channel or irregular spacing detected early
        if len(freqs) > 1:
            f_res = np.abs(freqs[1] - freqs[0])
            print(f"Warning: Could not reliably determine f_res from freq_ids. Using diff(freqs): {f_res:.5f} MHz")
        else:
            f_res = 0.390625 / TOTAL_CHANNELS # Default CHIME fine channel width if only one channel
            print(f"Warning: Only one frequency channel provided. Assuming default f_res: {f_res:.5f} MHz")
    else:
        idx0 = valid_freq_indices[0]
        num_chan_diff = int(freq_ids[idx0 + 1] - freq_ids[idx0])
        freq_diff = np.abs(freqs[idx0 + 1] - freqs[idx0])
        if num_chan_diff <= 0:
             f_res = freq_diff # Handle case where freq_ids might not be incremental
             print(f"Warning: Non-positive channel difference in freq_ids. Using diff(freqs): {f_res:.5f} MHz")
        else:
             f_res = freq_diff / float(num_chan_diff)

    print(f"Frequency resolution: {f_res:.5f} MHz/channel")
    if f_res <= 0:
        print("Error: Calculated frequency resolution is not positive. Cannot proceed.")
        return None, None, None

    # Prepare the 1D spectrum
    if ds.ndim == 2:
        # Average over the specified time range
        spec_1d = np.ma.mean(ds[:, int(time_range[0]):int(time_range[1])], axis=1)
    else:
        spec_1d = np.ma.masked_array(ds) # Ensure it's a masked array

    # Ensure masking is handled correctly (mask NaNs and zeros)
    spec_1d = np.ma.masked_where((spec_1d == 0) | np.isnan(spec_1d), spec_1d)
    mask_1d = ~spec_1d.mask # 1 for valid, 0 for masked

    # --- ACF Calculation ---
    if maxlag is None:
        maxlag_bin = None # Compute all lags in autocorr
    else:
        maxlag_bin = int(maxlag / f_res)
        if maxlag_bin <= 0:
            print("Warning: maxlag is smaller than frequency resolution. Setting maxlag_bin to 1.")
            maxlag_bin = 1

    # Calculate ACF for positive lags (excluding zero lag)
    acf_pos = autocorr(spec_1d, v=mask_1d, zerolag=False,
                       maxlag=maxlag_bin, offspec_mean=offspec_mean)

    if len(acf_pos) == 0 or np.all(np.isnan(acf_pos)):
        print("ACF calculation failed or resulted in NaNs.")
        return None, None, None

    # Create full ACF (positive and negative lags)
    # Note: ACF is symmetric for real signals, acf(-lag) = acf(lag)
    acf_full = np.concatenate((acf_pos[::-1], acf_pos))
    # Create corresponding lags in MHz
    lags_pos_bins = np.arange(1, len(acf_pos) + 1)
    lags_mhz = np.concatenate((-lags_pos_bins[::-1], lags_pos_bins)) * f_res

    # --- Lorentzian Fitting ---
    fit_result = None
    try:
        gmodel = Model(lorentz_w_c) # Use the model with constant offset
        # Select data for fitting based on lagrange_for_fit
        fit_mask = (lags_mhz >= -lagrange_for_fit) & (lags_mhz <= lagrange_for_fit)
        lags_for_fit = lags_mhz[fit_mask]
        acf_for_fit = acf_full[fit_mask]

        if len(lags_for_fit) < 3: # Need at least 3 points for fit
             raise ValueError("Not enough points within lagrange_for_fit for Lorentzian fit.")

        # Estimate initial parameters
        initial_gamma = lagrange_for_fit / 4.0 # Guess: width is ~1/4 of fit range
        initial_m = np.sqrt(np.max(acf_for_fit)) if np.max(acf_for_fit) > 0 else 1.0
        initial_c = np.min(acf_for_fit)

        # Perform the fit
        fit_result = gmodel.fit(acf_for_fit, x=lags_for_fit,
                                gamma=initial_gamma, m=initial_m, c=initial_c)

        if diagnostic_plots:
            print("\nLorentzian Fit Report:")
            print(fit_report(fit_result))

    except Exception as e:
        print(f"Could not fit a Lorentzian model: {e}")
        if diagnostic_plots: plt.legend() # Add legend even if fit fails

    # --- Plotting ---
    if diagnostic_plots:
        plt.figure(figsize=(10, 6))
        # Plot the calculated ACF
        plt.plot(lags_mhz, acf_full, drawstyle='steps-mid', color='k',
                 linewidth=1.0, label=f"ACF (res: {f_res:.3f} MHz)")

        if fit_result:
            # Plot the fitted Lorentzian model
            gamma_fit = fit_result.params['gamma'].value
            plt.plot(lags_mhz, fit_result.eval(x=lags_mhz), color='orange',
                     label=f'Lorentzian Fit ($\\gamma$ = {gamma_fit:.3f} MHz)')
            # Zoom in on the central part for clarity
            plot_xlim = max(np.abs(gamma_fit) * 10, lagrange_for_fit * 1.5)
            plt.xlim(-plot_xlim, plot_xlim)
        else:
            # Default zoom if fit failed
            plot_xlim_default = maxlag if maxlag is not None else lags_mhz[-1]
            plt.xlim(-plot_xlim_default, plot_xlim_default)


        plt.xlabel("Frequency Lag [MHz]")
        plt.ylabel("Normalized ACF")
        plt.title("Auto-Correlation Function")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return acf_full, lags_mhz, fit_result


def acf_per_subband(spec, freqs, freqids, num_subbands=2, savefig=None,
                    plot_fit=False, maxlag=None, snsubband=False, offspec=None):
    """
    Divides a spectrum into subbands and calculates the ACF for each.

    Optionally plots the ACFs and fits Lorentzians (fitting part seems missing
    in the original implementation, relies on `acf_scint_plot`'s potential fit).

    Parameters
    ----------
    spec : np.ndarray or np.ma.MaskedArray
        1D input spectrum [freq].
    freqs : np.ndarray
        1D array of central frequencies (MHz).
    freqids : np.ndarray
        1D array of frequency channel IDs.
    num_subbands : int, optional
        Number of subbands to divide the spectrum into. Default is 2.
    savefig : str, optional
        File path to save the plot of overlaid ACFs. If None, plot is not saved
        (but might be shown by acf_scint_plot if its diagnostics are on).
        Default is None.
    plot_fit : bool, optional
        If True, attempts to plot fits (relies on `acf_scint_plot` returning
        fit results). Also triggers saving a secondary plot of scint bw vs freq.
        Default is False.
    maxlag : float, optional
        Maximum frequency lag (MHz) for ACF calculation in each subband.
        Default is None (uses `acf_scint_plot` default).
    snsubband : bool, optional
        If True, divide into subbands based on equal signal-to-noise (flux)
        rather than equal number of channels. Default is False.
    offspec : np.ndarray or np.ma.MaskedArray, optional
        1D off-burst spectrum, used for normalization in `acf_scint_plot`.
        Must have the same shape and mask as `spec`. Default is None.

    Returns
    -------
    all_acfs : list
        List containing the ACF array for each subband.
    all_fcents : list
        List of central frequencies (MHz) for each subband.
    all_lags : list
        List containing the frequency lag array (MHz) for each subband's ACF.
    sub_sn : list
        List of total flux (signal) in each subband.
    sub_mask_count : list
        List of the number of masked channels in each subband.
    spec_lens : list
        List of the number of channels (length) in each subband.
    """
    # --- Input Preparation ---
    spec = np.ma.masked_where((spec == 0) | np.isnan(spec) | (spec.mask if hasattr(spec, 'mask') else False), spec, copy=True)
    if offspec is not None:
        offspec = np.ma.masked_where(spec.mask, offspec, copy=True) # Apply same mask

    nchan = len(spec)
    mask_bool = ~spec.mask # True where valid

    # --- Subband Division ---
    all_acfs = []
    all_lags = []
    all_fcents = []
    sub_sn = []
    sub_mask_count = []
    spec_lens = []

    # Calculate total valid flux if using snsubband
    total_valid_flux = 0
    if snsubband:
        total_valid_flux = np.sum(spec.data[mask_bool])
        if total_valid_flux <= 0:
            print("Warning: Total valid flux is not positive. Cannot use snsubband. Falling back to equal channels.")
            snsubband = False # Fallback

    flux_per_subband = total_valid_flux / float(num_subbands) if snsubband else 0
    current_flux_in_sub = 0
    start_idx = 0

    for sub in range(num_subbands):
        end_idx = nchan # Default end

        if snsubband:
            # Find end index based on flux
            if sub == num_subbands - 1:
                end_idx = nchan # Last subband takes the rest
            else:
                current_flux_in_sub = 0
                idx = start_idx
                while current_flux_in_sub < flux_per_subband and idx < nchan:
                    if mask_bool[idx]:
                        current_flux_in_sub += spec.data[idx]
                    idx += 1
                end_idx = idx
        else:
            # Find end index based on equal channels
            end_idx = start_idx + (nchan // num_subbands)
            if sub == num_subbands - 1:
                end_idx = nchan # Last subband takes the rest to avoid rounding issues

        # Ensure indices are valid
        end_idx = min(end_idx, nchan)
        if start_idx >= end_idx:
            print(f"Warning: Subband {sub+1} has zero length. Skipping.")
            start_idx = end_idx
            continue # Skip to next subband


        # --- Extract Subband Data ---
        spec_sub = spec[start_idx:end_idx]
        freqs_sub = freqs[start_idx:end_idx]
        freqids_sub = freqids[start_idx:end_idx]
        offspec_sub = offspec[start_idx:end_idx] if offspec is not None else None

        sub_len = len(spec_sub)
        spec_lens.append(sub_len)
        sub_mask_count.append(np.sum(spec_sub.mask))
        sub_sn.append(np.sum(spec_sub.data[~spec_sub.mask])) # Sum of valid flux

        # Calculate central frequency of the subband
        if sub_len > 0:
             fcent_sub = freqs_sub[~spec_sub.mask].mean() if np.any(~spec_sub.mask) else (freqs_sub[0] + freqs_sub[-1]) / 2.0
        else:
             fcent_sub = np.nan # Should not happen with check above

        all_fcents.append(fcent_sub)
        print(f"Subband {sub+1}: Freqs {freqs_sub[0]:.2f}-{freqs_sub[-1]:.2f} MHz (Cent: {fcent_sub:.2f} MHz), Chans {start_idx}-{end_idx-1}")

        # --- Calculate ACF for Subband ---
        # Determine lag range for fitting within this subband
        # Use a fraction of the maxlag or a fixed value
        lagrange_fit_sub = 5.0 # Default fixed value in MHz
        if maxlag is not None and maxlag < 10.0:
             lagrange_fit_sub = maxlag / 2.0

        # Call acf_scint_plot (disable its internal plotting)
        acf_result = acf_scint_plot(spec_sub, freqids_sub, freqs_sub, [0, 1], # time_range dummy
                                    lagrange_for_fit=lagrange_fit_sub,
                                    diagnostic_plots=False, # Control plotting externally
                                    maxlag=maxlag,
                                    offspec_mean=np.ma.mean(offspec_sub) if offspec_sub is not None else None)

        if acf_result is None or acf_result[0] is None:
            print(f"ACF calculation failed for subband {sub+1}. Appending NaNs.")
            all_acfs.append(np.array([np.nan]))
            all_lags.append(np.array([np.nan]))
            # Store fit results if needed (currently acf_result[2] holds it)
        else:
            all_acfs.append(acf_result[0])
            all_lags.append(acf_result[1])
            # Store fit results if needed

        # Update start index for next subband
        start_idx = end_idx

    # --- Plotting Overlaid ACFs ---
    if savefig is not None and len(all_acfs) > 0:
        plt.figure(figsize=(10, 8))
        cmap = matplotlib.cm.get_cmap('plasma')
        max_acf_val = 0
        min_acf_val = 0
        for i in range(len(all_fcents)):
            if len(all_acfs[i]) > 1: # Check if ACF is valid
                rgba = cmap(i / float(len(all_fcents)))
                offset = 1.0 * i # Vertical offset for clarity
                plt.plot(all_lags[i], all_acfs[i] + offset, drawstyle='steps-mid',
                         color=rgba, linewidth=1.5, alpha=0.8,
                         label=f'{all_fcents[i]:.1f} MHz')
                max_acf_val = max(max_acf_val, np.nanmax(all_acfs[i] + offset))
                min_acf_val = min(min_acf_val, np.nanmin(all_acfs[i] + offset))
            # Add plotting of fits here if `plot_fit` is True and fit results were stored

        plt.xlabel('Frequency Lag [MHz]')
        plt.ylabel('ACF + Offset')
        plt.title(f'ACF per Subband ({num_subbands} subbands)')
        # Adjust xlim based on maxlag or data range
        if maxlag:
             plt.xlim(-maxlag, maxlag)
        elif len(all_lags) > 0 and len(all_lags[0]) > 1:
             plt.xlim(all_lags[0][0], all_lags[0][-1]) # Use range of first valid ACF

        # Adjust ylim based on plotted data
        plt.ylim(min_acf_val - 0.5, max_acf_val + 0.5)

        plt.legend(loc='upper left', fontsize='small')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(savefig, format='pdf')
        plt.close() # Close plot after saving

    # --- Plot Scintillation Bandwidth vs Frequency (if plot_fit) ---
    # This part requires extracting the fitted 'gamma' values from acf_scint_plot results
    if plot_fit and savefig is not None:
        print("Warning: 'plot_fit' functionality requires extracting fit results "
              "from acf_scint_plot, which is not fully implemented here.")
        # Placeholder: Assumes 'sub_scint' list was populated with gamma values
        # sub_cent = all_fcents
        # plt.figure()
        # plt.scatter(sub_cent, sub_scint, marker='x', color='k')
        # # Add power-law fit line if desired
        # # plt.plot(freqs, some_model(freqs), color='r')
        # plt.xlabel('Frequency [MHz]')
        # plt.ylabel('Scintillation Bandwidth [MHz]')
        # plt.title('Scintillation Bandwidth vs Frequency')
        # plt.grid(True, linestyle=':', alpha=0.6)
        # plt.savefig(savefig.replace('.pdf', '_scintbw.pdf'), format='pdf')
        # plt.close()


    return all_acfs, all_fcents, all_lags, sub_sn, sub_mask_count, spec_lens


# --- Model Definitions for Fitting ---

def lorentz(x, gamma1, m1):
    """ Lorentzian function without constant offset. """
    return m1**2 / (1 + (x / gamma1)**2)

def lorentz_w_c(x, gamma1, m1, c):
    """ Lorentzian function with constant offset 'c'. """
    return m1**2 / (1 + (x / gamma1)**2) + c

def doublelorentz_w_c(x, gamma1, m1, gamma2, m2, c):
    """ Sum of two Lorentzian functions with a shared constant offset 'c'. """
    return (m1**2 / (1 + (x / gamma1)**2)) + (m2**2 / (1 + (x / gamma2)**2)) + c

def triplelorentz(x, gamma1, m1, gamma2, m2, gamma3, m3):
    """ Sum of three Lorentzian functions without constant offset. """
    return (m1**2 / (1 + (x / gamma1)**2)) + \
           (m2**2 / (1 + (x / gamma2)**2)) + \
           (m3**2 / (1 + (x / gamma3)**2))

# --- Minimizer Functions for lmfit ---
# These functions define the residual (data - model) / error for lmfit minimizers.

def lorentz_withc_min(params, x, y, err):
    """ Residual function for lorentz_w_c model. """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    c = params['c'].value
    model = lorentz_w_c(x, gamma1, m1, c)
    return (model - y) / err

def doublelorentz_withc_min(params, x, y, err):
    """ Residual function for doublelorentz_w_c model. """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    gamma2 = params['gamma2'].value
    m2 = params['m2'].value
    c = params['c'].value
    model = doublelorentz_w_c(x, gamma1, m1, gamma2, m2, c)
    return (model - y) / err

def triplelorentz_min(params, x, y, err):
    """ Residual function for triplelorentz model. """
    gamma1 = params['gamma1'].value
    m1 = params['m1'].value
    gamma2 = params['gamma2'].value
    m2 = params['m2'].value
    gamma3 = params['gamma3'].value
    m3 = params['m3'].value
    model = triplelorentz(x, gamma1, m1, gamma2, m2, gamma3, m3)
    return (model - y) / err

# --- Scintillation Bandwidth vs. Frequency Relation ---

def scint_freq_relation(v, c, n):
    """ Power-law model for scintillation bandwidth: bw = c * v^n. """
    # Ensure v is positive for power law
    v_safe = np.maximum(v, 1e-9) # Avoid log(0) or negative base issues if n is not integer
    return c * (v_safe**n)

def scint_freq_relation_min(params, x, y, err):
    """ Residual function for scint_freq_relation model. """
    c = params['c'].value
    n = params['n'].value
    model = scint_freq_relation(x, c, n)
    return (model - y) / err

# --- Linear Model ---

def lin(x, grad, c):
    """ Linear model: y = grad * x + c. """
    return grad * x + c

def linmin(params, x, y, errs):
    """ Residual function for linear model. """
    grad = params['grad'].value
    c = params['c'].value
    model = lin(x, grad, c)
    return (model - y) / errs


# --- Utility Functions ---

def scrunch(wfall, tscrunch, fscrunch):
    """
    Rebins (averages) a 2D array along time and frequency axes.

    Parameters
    ----------
    wfall: ndarray (2D)
        Array to be rebinned [freq, time].
    tscrunch: int
        Scrunching factor along time axis (axis 1).
    fscrunch: int
        Scrunching factor along frequency axis (axis 0).

    Returns
    -------
    rebinned_array: array
        The rebinned array.
    """
    if wfall.ndim != 2:
        raise ValueError("Input wfall for scrunch must be 2D.")
    if tscrunch <= 0 or fscrunch <= 0:
        raise ValueError("Scrunch factors must be positive.")

    nchan, nbins = wfall.shape

    # Scrunch time axis
    if tscrunch > 1:
        remainder_t = nbins % tscrunch
        if remainder_t != 0:
            wfall = wfall[:, : nbins - remainder_t] # Truncate
        # Reshape, mean, and handle potential NaNs
        wfall = np.nanmean(
            wfall.reshape(nchan, nbins // tscrunch, tscrunch), axis=2
        )

    # Scrunch frequency axis
    if fscrunch > 1:
        remainder_f = nchan % fscrunch
        if remainder_f != 0:
            # Raise error instead of truncating frequency
            raise ValueError(f"Number of channels ({nchan}) not an integer "
                              f"factor of fscrunch ({fscrunch}).")
        # Reshape, mean, and handle potential NaNs
        wfall = np.nanmean(
            wfall.reshape(nchan // fscrunch, fscrunch, wfall.shape[1]), axis=1
        )

    return wfall


def weighted_avg_and_std(values, weights):
    """
    Calculates the weighted average and weighted standard deviation.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    weights : np.ndarray
        Array of weights, same shape as `values`.

    Returns
    -------
    tuple
        (weighted_average, weighted_standard_deviation)
    """
    if values.shape != weights.shape:
        raise ValueError("Shapes of values and weights must match.")
    if np.sum(weights) == 0:
        print("Warning: Sum of weights is zero.")
        return np.nan, np.nan

    average = np.average(values, weights=weights)
    # Variance calculation: weighted average of squared deviations
    variance = np.average((values - average)**2, weights=weights)

    # Return average and sqrt(variance)
    # Handle potential negative variance due to floating point errors
    return average, math.sqrt(max(0, variance))


# --- Physical Parameter Estimations ---

def res(lens_dist, lda, scat_lens):
    """
    Estimates the physical resolution of a thin scattering screen.

    Parameters
    ----------
    lens_dist : float
        Distance between source and scattering lens [kpc].
    lda : float
        Wavelength of observation [m].
    scat_lens : float
        Scattering timescale imparted by the screen [ms].

    Returns
    -------
    float
        Physical resolution of the lens [km].
    """
    if lens_dist <= 0 or lda <= 0 or scat_lens <= 0:
        print("Warning: Input distances, wavelength, and scattering time must be positive.")
        return np.nan

    lens_dist_m = lens_dist * cons.parsec * 1000.0 # Convert kpc to m
    scat_lens_s = scat_lens / 1000.0 # Convert ms to s

    # Formula for Fresnel scale (related to resolution)
    # r_F = sqrt(lambda * D / (2 * pi)) for vacuum
    # Here, adapted for scattering: relates resolution to scattering time.
    # The exact formula derivation might depend on screen model assumptions.
    # The formula used here seems related to diffractive scale.
    # Check original source/paper for derivation if needed.
    # Original comment mentioned uncertainty about a factor of 2.
    resolution_m = (lda / np.pi) * np.sqrt(lens_dist_m / (4.0 * cons.c * scat_lens_s))

    return resolution_m / 1000.0 # Convert m to km


def emission_size(phys_res, mod_ind):
    """
    Estimates the physical size of the emission region based on lens
    resolution and scintillation modulation index.

    Assumes Gaussian source brightness distribution and relates modulation
    index to the ratio of source size to lens resolution.

    Parameters
    ----------
    phys_res : float
        Physical resolution of the lens [km] (e.g., from `res` function).
    mod_ind : float
        Modulation index of scintillation (e.g., sqrt(ACF[0]) or std(spec)/mean(spec)).
        Should be <= 1 for physical interpretation here.

    Returns
    -------
    float
        Estimated physical size (sigma of Gaussian) of the emission region [km].
    """
    if phys_res <= 0:
        print("Warning: Physical resolution must be positive.")
        return np.nan
    if not (0 < mod_ind <= 1):
        print(f"Warning: Modulation index ({mod_ind:.3f}) is outside the "
              "expected range (0, 1]. Result may be unphysical.")
        # Handle cases leading to sqrt of negative or division by zero
        if mod_ind <= 0: return np.nan
        # If mod_ind > 1, the term inside sqrt becomes negative.

    # Formula derived from relationship between modulation index (m) and
    # source size (sigma_s) relative to resolution (r_res):
    # m^2 approx 1 / (1 + (sigma_s / r_res)^2)  (for Gaussian source/lens?)
    # Rearranging gives: (sigma_s / r_res)^2 = (1 / m^2) - 1
    # sigma_s = r_res * sqrt((1 / m^2) - 1)
    # The factor of 4 in the original code (sqrt(...)/4) seems incorrect based on this common formula.
    # Let's use the standard formula:
    term_inside_sqrt = (1.0 / mod_ind**2) - 1.0
    if term_inside_sqrt < 0:
        print("Warning: (1/mod_ind^2 - 1) is negative. Cannot calculate real emission size.")
        return np.nan

    sigma_source = phys_res * np.sqrt(term_inside_sqrt)

    # Original code had: sigma = np.sqrt((1/(float(mod_ind)**2) - 1)/4.)
    # This implies sigma_source = phys_res * sqrt(...) / 2
    # Keeping the standard formula unless the factor of 1/2 (or 1/4 under sqrt)
    # has a specific justification in the context of this code's origin.

    return sigma_source

