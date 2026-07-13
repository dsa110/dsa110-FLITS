# -*- coding: utf-8 -*-
"""
Refactored scintillation analysis tools based on scinttools_new.py.

This module provides functions to calculate Auto-Correlation Functions (ACFs)
from spectra, fit Lorentzian models to measure scintillation bandwidth
and modulation index vs frequency, analyze modulation index vs time,
and generate diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from tqdm.auto import tqdm
import warnings
import math # For calculating plot grid size

# --- Configuration ---
# Scintillation bandwidth definition: FWHM of the Lorentzian fit to the ACF.
# The 'gamma' parameter in the Lorentzian model corresponds to HWHM.
# We will fit for HWHM (gamma) and report FWHM = 2 * gamma.
# The 'amplitude' parameter (A) in the fit corresponds to m^2.
# We fit for m = sqrt(A) and report m.

# --- Lorentzian Model Definitions ---

def lorentzian_model(x, amplitude, center, gamma):
    """Standard Lorentzian function, parameterized for lmfit."""
    # gamma is HWHM (Half-Width at Half-Maximum)
    # amplitude is the peak height (related to m^2)
    return amplitude / (1.0 + ((x - center) / gamma)**2)

def lorentzian_model_with_offset(x, amplitude, center, gamma, offset):
    """Lorentzian function with a constant vertical offset."""
    return lorentzian_model(x, amplitude, center, gamma) + offset

def double_lorentzian_model_with_offset(x, amp1, cen1, gam1, amp2, cen2, gam2, offset):
    """Double Lorentzian function with a constant vertical offset."""
    return (lorentzian_model(x, amp1, cen1, gam1) +
            lorentzian_model(x, amp2, cen2, gam2) +
            offset)

# --- Core ACF and Fitting Functions (Frequency Domain) ---

def calculate_acf(spectrum, mask=None, max_lag_bins=None, mean_subtract=True,
                  normalize=True, offspec_mean=0.0):
    """
    Calculates the Auto-Correlation Function (ACF) of a 1D spectrum.

    Args:
        spectrum (np.ndarray): 1D array containing the spectrum data.
        mask (np.ndarray, optional): Boolean mask for the spectrum (True means masked).
                                     Defaults to None (no mask).
        max_lag_bins (int, optional): Maximum lag (in frequency bins) to compute.
                                     Defaults to None (compute for all lags).
        mean_subtract (bool, optional): Subtract the mean before calculating ACF.
                                        Defaults to True.
        normalize (bool, optional): Normalize ACF such that ACF(0) ~ m^2 = Var(I)/<I>^2.
                                    Defaults to True.
        offspec_mean (float, optional): Mean of off-burst spectrum, used in normalization
                                         if provided. Defaults to 0.0.

    Returns:
        tuple: (lags_bins, acf)
            - lags_bins (np.ndarray): Lags in units of frequency bins (0, 1, 2,...).
            - acf (np.ndarray): Calculated auto-correlation function. Returns None if
              calculation fails (e.g., insufficient unmasked data).
    """
    if not isinstance(spectrum, np.ma.MaskedArray):
        spec_ma = np.ma.masked_array(spectrum, mask=mask)
    else:
        # Ensure input mask is combined with internal masking if any
        spec_ma = np.ma.masked_array(spectrum.data, mask=np.logical_or(spectrum.mask, mask if mask is not None else False))

    if spec_ma.count() < 2: # Need at least 2 unmasked points
        warnings.warn("Insufficient unmasked data points to calculate ACF.")
        return np.array([]), None # Return empty array for lags, None for acf

    # Use only unmasked data for calculations
    valid_spec = spec_ma.compressed()
    if valid_spec.size < 2:
        warnings.warn("Insufficient unmasked data points after compression.")
        return np.array([]), None

    mean_spec = np.mean(valid_spec)

    if mean_subtract:
        # Create working array, subtract mean only from unmasked elements
        work_spec = spec_ma.astype(float, copy=True) # Work with floats
        work_spec[~spec_ma.mask] -= mean_spec
        # Ensure masked values are 0 for correlation calculation
        work_spec.fill_value = 0.0
        data_for_corr = work_spec.filled()
    else:
        work_spec = spec_ma.astype(float, copy=True)
        work_spec.fill_value = 0.0
        data_for_corr = work_spec.filled()


    n_chan = len(spectrum)
    if max_lag_bins is None:
        max_lag_bins = n_chan -1
    else:
        max_lag_bins = min(max_lag_bins, n_chan - 1)

    # Use numpy.correlate for efficiency
    # 'valid' mode computes correlation only where signals fully overlap
    # Need 'full' mode and then select lags, or implement manually for mask handling
    # Manual implementation for precise mask handling and normalization:

    lags_bins = np.arange(max_lag_bins + 1)
    acf = np.zeros(max_lag_bins + 1, dtype=float)
    counts = np.zeros(max_lag_bins + 1, dtype=int)

    # Pre-calculate which elements are valid (not masked)
    valid_indices = ~spec_ma.mask

    for lag in lags_bins: # No need for tqdm usually, it's fast
        if lag == 0:
            valid_pair_indices = valid_indices
            if np.sum(valid_pair_indices) > 0:
                acf[lag] = np.sum(work_spec[valid_pair_indices]**2)
                counts[lag] = np.sum(valid_pair_indices)
            else:
                acf[lag] = np.nan # Avoid division by zero later if count is 0
                counts[lag] = 0
            continue

        # Indices for lag > 0
        idx1 = np.arange(n_chan - lag)
        idx2 = np.arange(lag, n_chan)

        # Check mask for both elements in the pair
        valid_pair_indices = valid_indices[idx1] & valid_indices[idx2]

        if np.sum(valid_pair_indices) > 0:
            acf[lag] = np.sum(work_spec[idx1[valid_pair_indices]] * work_spec[idx2[valid_pair_indices]])
            counts[lag] = np.sum(valid_pair_indices)
        else:
            acf[lag] = np.nan # Avoid division by zero if count is 0
            counts[lag] = 0


    # --- Normalization ---
    # Avoid division by zero where counts are zero
    valid_counts = counts > 0
    if np.any(valid_counts): # Proceed only if there's something to normalize
        acf[valid_counts] /= counts[valid_counts]
    else: # If all counts are zero, acf calculation failed earlier or data is fully masked
         return lags_bins, None

    if normalize:
        # Normalize by variance proxy: <I>^2 or (<I> - <I_off>)^2
        # This makes ACF(0) approx Var(I)/<I>^2 = m^2
        effective_mean = mean_spec - offspec_mean
        if effective_mean != 0:
            # Check variance: acf[0] should be Var(I) if mean_subtract=True
            variance_est = acf[0] if mean_subtract else np.var(valid_spec)
            norm_factor = effective_mean**2
            if norm_factor > 1e-15: # Avoid division by zero or tiny numbers
                 acf /= norm_factor
            else:
                 warnings.warn(f"Normalization factor (effective_mean^2 = {norm_factor:.2e}) is close to zero. Skipping ACF normalization by power.")
                 # Fallback: normalize by variance estimate if possible
                 if variance_est > 1e-15:
                     acf /= variance_est # Makes ACF(0) approx 1
                     warnings.warn("Normalizing ACF by variance estimate instead.")
                 else:
                      warnings.warn("Variance estimate also near zero. ACF remains unnormalized by power.")
                      acf = np.full_like(acf, np.nan) # Cannot reliably normalize
        else:
             warnings.warn("Effective mean is zero. Cannot normalize ACF by power.")
             # Normalize by variance? Or return unnormalized?
             # Let's normalize by variance if possible, making ACF(0) approx 1
             variance_est = acf[0] if mean_subtract else np.var(valid_spec)
             if variance_est > 1e-15:
                 acf /= variance_est
                 warnings.warn("Normalizing ACF by variance estimate instead.")
             else:
                  warnings.warn("Variance estimate also near zero. ACF remains unnormalized.")
                  acf = np.full_like(acf, np.nan) # Cannot reliably normalize


    if np.all(np.isnan(acf)):
        return lags_bins, None

    return lags_bins, acf


def fit_acf_model(lags_bins, acf, freq_res_mhz, model_type='single',
                  fit_lag_range_mhz=None, initial_gamma_mhz=0.1):
    """
    Fits a Lorentzian model to the ACF.

    Args:
        lags_bins (np.ndarray): Lags in units of frequency bins (0, 1, 2,...).
        acf (np.ndarray): Auto-correlation function values.
        freq_res_mhz (float): Frequency resolution in MHz per bin.
        model_type (str, optional): Type of model ('single', 'double').
                                    Defaults to 'single'.
        fit_lag_range_mhz (float, optional): Fit only lags up to this value [MHz].
                                             Defaults to None (fit all lags).
        initial_gamma_mhz (float, optional): Initial guess for HWHM gamma [MHz].
                                              Defaults to 0.1 MHz.

    Returns:
        lmfit.model.ModelResult or None: The fit result object, or None if fit fails.
    """
    if acf is None or len(acf) < 2 or np.all(np.isnan(acf)):
        warnings.warn("ACF is None or too short or all NaN, cannot fit.")
        return None

    lags_mhz = lags_bins * freq_res_mhz

    # Select data range for fitting
    if fit_lag_range_mhz is not None:
        fit_indices = np.where(lags_mhz <= fit_lag_range_mhz)[0]
        if len(fit_indices) < 3: # Need points for fit
             warnings.warn(f"Less than 3 data points within fit range {fit_lag_range_mhz} MHz. Cannot fit.")
             return None
        fit_lags_mhz = lags_mhz[fit_indices]
        fit_acf = acf[fit_indices]
    else:
        fit_lags_mhz = lags_mhz
        fit_acf = acf

    # Remove NaNs from fitting data
    nan_mask = np.isnan(fit_acf)
    if np.all(nan_mask):
        warnings.warn("All ACF values in the fitting range are NaN. Cannot fit.")
        return None
    fit_lags_mhz = fit_lags_mhz[~nan_mask]
    fit_acf = fit_acf[~nan_mask]

    if len(fit_acf) < 3: # Check again after NaN removal
        warnings.warn(f"Less than 3 valid data points remaining for fit. Cannot fit.")
        return None


    # --- Set up lmfit model ---
    if model_type == 'single':
        model = Model(lorentzian_model_with_offset)
        params = model.make_params()
        # Initial guesses and bounds
        initial_amplitude = fit_acf[0] if fit_acf[0] > 0 else 1e-3 # Should be m^2
        params['amplitude'].set(value=initial_amplitude, min=1e-9) # m^2 > 0
        params['center'].set(value=0.0, vary=False) # Assume peak at zero lag
        params['gamma'].set(value=initial_gamma_mhz, min=1e-6) # HWHM > 0 [MHz]
        # Improved offset guess: median of last half, bounded reasonably
        offset_guess = np.nanmedian(fit_acf[len(fit_acf)//2:])
        params['offset'].set(value=offset_guess if not np.isnan(offset_guess) else 0.0, min=-1.0, max=1.0)


    elif model_type == 'double':
         model = Model(double_lorentzian_model_with_offset)
         params = model.make_params()
         initial_amplitude = fit_acf[0] if fit_acf[0] > 0 else 1e-3
         # Guesses might need refinement based on data
         params['amp1'].set(value=initial_amplitude * 0.7, min=1e-9)
         params['cen1'].set(value=0.0, vary=False)
         params['gam1'].set(value=initial_gamma_mhz, min=1e-6)
         params['amp2'].set(value=initial_amplitude * 0.3, min=1e-9)
         params['cen2'].set(value=0.0, vary=False)
         params['gam2'].set(value=initial_gamma_mhz * 5, min=1e-6) # Guess broader component
         offset_guess = np.nanmedian(fit_acf[len(fit_acf)//2:])
         params['offset'].set(value=offset_guess if not np.isnan(offset_guess) else 0.0, min=-1.0, max = 1.0)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Perform the fit
    try:
        # Add weights? Penalize points far from zero lag? Use ACF variance? For now, no weights.
        fit_result = model.fit(fit_acf, params, x=fit_lags_mhz, nan_policy='omit')
        if not fit_result.success:
             warnings.warn(f"lmfit optimization failed: {fit_result.message}")
             # Optionally return the failed fit result for inspection
             # return fit_result
             return None
        return fit_result
    except Exception as e:
        warnings.warn(f"lmfit fitting process raised an exception: {e}")
        return None


def extract_scint_params(fit_result):
    """
    Extracts key scintillation parameters from an lmfit ModelResult.

    Args:
        fit_result (lmfit.model.ModelResult): Successful result from fit_acf_model.

    Returns:
        dict: Dictionary containing parameters like 'fwhm_mhz', 'fwhm_mhz_err',
              'mod_index', 'mod_index_err', 'offset', 'offset_err', and potentially
              parameters for a second component if model was 'double'. Returns None
              if input is None or extraction fails.
    """
    if fit_result is None:
        return None

    params = fit_result.params
    extracted = {}

    try:
        # Component 1
        amp1 = params['amplitude'].value if 'amplitude' in params else params['amp1'].value
        gam1 = params['gamma'].value if 'gamma' in params else params['gam1'].value
        amp1_err = params['amplitude'].stderr if 'amplitude' in params else params['amp1'].stderr
        gam1_err = params['gamma'].stderr if 'gamma' in params else params['gam1'].stderr

        # FWHM = 2 * HWHM (gamma)
        extracted['fwhm_mhz'] = 2.0 * gam1
        extracted['fwhm_mhz_err'] = 2.0 * gam1_err if gam1_err is not None else np.nan # Use NaN for missing errors

        # Mod index m = sqrt(Amplitude)
        extracted['mod_index'] = np.sqrt(amp1) if amp1 > 0 else 0.0
        # Error propagation: err(sqrt(A)) = err(A) / (2*sqrt(A))
        if amp1 > 1e-9 and amp1_err is not None:
             # Check if mod_index is also non-zero before division
             if extracted['mod_index'] > 1e-9 :
                 extracted['mod_index_err'] = abs(amp1_err / (2.0 * extracted['mod_index']))
             else:
                 extracted['mod_index_err'] = np.nan
        else:
             extracted['mod_index_err'] = np.nan


        # Offset
        extracted['offset'] = params['offset'].value
        extracted['offset_err'] = params['offset'].stderr if params['offset'].stderr is not None else np.nan

        # Component 2 (if double fit)
        if 'amp2' in params:
            amp2 = params['amp2'].value
            gam2 = params['gam2'].value
            amp2_err = params['amp2'].stderr
            gam2_err = params['gam2'].stderr

            extracted['fwhm_mhz_2'] = 2.0 * gam2
            extracted['fwhm_mhz_err_2'] = 2.0 * gam2_err if gam2_err is not None else np.nan

            extracted['mod_index_2'] = np.sqrt(amp2) if amp2 > 0 else 0.0
            if amp2 > 1e-9 and amp2_err is not None:
                 if extracted['mod_index_2'] > 1e-9:
                    extracted['mod_index_err_2'] = abs(amp2_err / (2.0 * extracted['mod_index_2']))
                 else:
                    extracted['mod_index_err_2'] = np.nan
            else:
                extracted['mod_index_err_2'] = np.nan

    except KeyError as e:
        warnings.warn(f"Parameter extraction failed: Missing key {e} in fit result.")
        return None
    except TypeError as e:
        # Handles cases where stderr might be None or other type issues
        warnings.warn(f"Parameter extraction failed due to type error: {e}. Returning partial results.")
        # Return partially filled dict, errors already default to NaN or handled

    # Final check for NaNs in essential parameters
    if np.isnan(extracted.get('fwhm_mhz', np.nan)) or np.isnan(extracted.get('mod_index', np.nan)):
        warnings.warn("Essential parameters (FWHM or ModIndex) could not be extracted or are NaN.")
        # Decide whether to return partial or None based on needs
        # return None # Stricter: fail if key params missing
        pass # Lenient: return what could be extracted


    return extracted

# --- Workflow Functions (Frequency Domain) ---

def analyze_spectrum(spectrum, freqs_mhz, mask=None, max_lag_mhz=10.0,
                     model_type='single', fit_lag_range_mhz=None,
                     offspec_mean=0.0, initial_gamma_mhz=0.1):
    """
    Performs full ACF analysis (calculation, fitting, parameter extraction)
    on a single 1D spectrum.

    Args:
        spectrum (np.ndarray): 1D array containing the spectrum data.
        freqs_mhz (np.ndarray): Corresponding frequencies in MHz.
        mask (np.ndarray, optional): Boolean mask for the spectrum (True means masked).
        max_lag_mhz (float, optional): Max lag for ACF calculation [MHz]. Defaults to 10.0.
        model_type (str, optional): 'single' or 'double' Lorentzian fit. Defaults to 'single'.
        fit_lag_range_mhz (float, optional): Max lag for fitting range [MHz]. Defaults to None (use max_lag_mhz).
        offspec_mean (float, optional): Mean of off-burst spectrum for normalization. Defaults to 0.0.
        initial_gamma_mhz (float, optional): Initial guess for HWHM gamma [MHz]. Defaults to 0.1.


    Returns:
        dict: A dictionary containing results:
            'lags_mhz': Lags in MHz (symmetric: -max to +max).
            'acf': ACF values (symmetric).
            'fit_result': The lmfit ModelResult object (or None).
            'params': Dictionary of extracted parameters (or None).
            'freq_res_mhz': Frequency resolution.
            'status': 'OK' or 'Error: specific message'.
    """
    results = {'status': 'OK'}
    if len(freqs_mhz) < 2:
        results['status'] = 'Error: Need at least 2 frequency channels.'
        return results

    # Determine frequency resolution (handle potential non-uniformity)
    freq_diffs = np.diff(freqs_mhz)
    if len(freq_diffs)==0: # Handle case of only 2 channels
        freq_res_mhz = np.abs(freqs_mhz[1] - freqs_mhz[0])
    elif np.allclose(freq_diffs, freq_diffs[0]):
        freq_res_mhz = np.abs(freq_diffs[0])
    else:
        # Use average resolution if not uniform, with a warning
        freq_res_mhz = np.abs(np.mean(freq_diffs))
        warnings.warn("Frequency channels are not uniformly spaced. Using average resolution.")
    results['freq_res_mhz'] = freq_res_mhz

    if freq_res_mhz <= 0:
        results['status'] = 'Error: Invalid frequency resolution <= 0.'
        return results


    # Calculate ACF
    max_lag_bins = int(np.ceil(max_lag_mhz / freq_res_mhz))
    lags_bins, acf_raw = calculate_acf(spectrum, mask=mask, max_lag_bins=max_lag_bins,
                                    mean_subtract=True, normalize=True,
                                    offspec_mean=offspec_mean)

    if acf_raw is None:
        results['status'] = 'Error: ACF calculation failed.'
        # Provide empty placeholders for consistent output structure
        results['lags_mhz'] = np.array([])
        results['acf'] = np.array([])
        results['fit_result'] = None
        results['params'] = None
        return results

    # Make symmetric for output/plotting convenience (fitting uses positive lags)
    lags_mhz_sym = np.concatenate((-lags_bins[1:][::-1] * freq_res_mhz, lags_bins * freq_res_mhz))
    acf_sym = np.concatenate((acf_raw[1:][::-1], acf_raw))
    results['lags_mhz'] = lags_mhz_sym
    results['acf'] = acf_sym

    # Fit ACF Model
    fit_range = fit_lag_range_mhz if fit_lag_range_mhz is not None else max_lag_mhz
    # Pass freq_res_mhz to fit_acf_model if needed for initial guesses based on bins
    fit_result = fit_acf_model(lags_bins, acf_raw, freq_res_mhz,
                               model_type=model_type,
                               fit_lag_range_mhz=fit_range,
                               initial_gamma_mhz=initial_gamma_mhz)
    # Store freq_res in userkws for potential later use in plotting/analysis
    if fit_result is not None:
        fit_result.userkws['freq_res_mhz'] = freq_res_mhz


    results['fit_result'] = fit_result

    # Extract Parameters
    scint_params = extract_scint_params(fit_result)
    results['params'] = scint_params

    if fit_result is None or scint_params is None:
         current_status = results.get('status', 'OK')
         if current_status == 'OK': # Don't overwrite ACF calc error
             results['status'] = 'Warning: ACF fitting or parameter extraction failed.'
         # Keep going, but params/fit_result might be None


    return results


def analyze_subbands(spectrum, freqs_mhz, num_subbands=8, mask=None,
                     divide_method='equal_freq', **kwargs):
    """
    Divides a spectrum into sub-bands and analyzes each using analyze_spectrum.

    Args:
        spectrum (np.ndarray): 1D array containing the full spectrum data.
        freqs_mhz (np.ndarray): Corresponding frequencies in MHz.
        num_subbands (int, optional): Number of sub-bands to divide into. Defaults to 8.
        mask (np.ndarray, optional): Boolean mask for the spectrum (True means masked).
        divide_method (str, optional): How to divide: 'equal_freq' (equal frequency range)
                                      or 'equal_snr' (approx. equal integrated intensity).
                                      Defaults to 'equal_freq'.
        **kwargs: Additional keyword arguments passed directly to analyze_spectrum
                  (e.g., max_lag_mhz, model_type, offspec_mean, etc.).

    Returns:
        list[dict]: A list of result dictionaries, one for each sub-band.
                    Each dictionary contains:
                    'subband_index': 0-based index.
                    'freq_center_mhz': Central frequency of the sub-band.
                    'freq_range_mhz': (min_freq, max_freq) of the sub-band.
                    'analysis_results': The dictionary returned by analyze_spectrum.
    """
    if not isinstance(spectrum, np.ma.MaskedArray):
        spec_ma = np.ma.masked_array(spectrum, mask=mask)
    else:
        spec_ma = np.ma.masked_array(spectrum.data, mask=np.logical_or(spectrum.mask, mask if mask is not None else False))


    n_chan = len(spec_ma)
    all_indices = np.arange(n_chan)
    subband_results = []

    if n_chan < num_subbands * 2 : # Ensure at least 2 channels per subband on average
        warnings.warn(f"Number of channels ({n_chan}) is low for {num_subbands} subbands. Reducing num_subbands.")
        num_subbands = max(1, n_chan // 2)
        if num_subbands == 0: # Handle edge case of very few channels
             warnings.warn(f"Fewer than 2 channels available. Cannot perform subband analysis.")
             return subband_results


    if divide_method == 'equal_freq':
        # Ensure splits result in at least 1 channel per subband, preferably more
        indices_per_subband = np.array_split(all_indices, num_subbands)
        # Get start index of subbands > 0; handle case where array_split might return empty arrays if num_subbands > n_chan
        split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]


    elif divide_method == 'equal_snr':
         # Calculate cumulative intensity (use fill_value=0 for masked points)
         valid_spec = spec_ma.compressed()
         if len(valid_spec) == 0:
              warnings.warn("Cannot use equal_snr division: No unmasked data. Falling back to 'equal_freq'.")
              indices_per_subband = np.array_split(all_indices, num_subbands)
              split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]
         else:
              mean_signal = np.mean(valid_spec) # Use mean of valid points
              # Get offspec mean if provided, otherwise assume 0 baseline
              offspec_mean_val = 0.0
              if 'offspec_spectrum' in kwargs and kwargs['offspec_spectrum'] is not None:
                   # Ensure offspec_spectrum is a masked array for calculation
                   offspec_ma = np.ma.masked_array(kwargs['offspec_spectrum'])
                   offspec_mean_val = np.ma.mean(offspec_ma) if offspec_ma.count() > 0 else 0.0
              elif 'offspec_mean' in kwargs:
                   offspec_mean_val = kwargs['offspec_mean']

              signal_est = np.maximum(0, spec_ma.filled(0.0) - offspec_mean_val) # Approx signal above baseline
              cumul_signal = np.cumsum(signal_est)
              total_signal = cumul_signal[-1] if len(cumul_signal) > 0 else 0.0

              if total_signal <=0:
                   warnings.warn("Total estimated signal is non-positive. Cannot use 'equal_snr'. Falling back to 'equal_freq'.")
                   indices_per_subband = np.array_split(all_indices, num_subbands)
                   split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]
              else:
                   target_signal_per_subband = total_signal / num_subbands
                   split_indices = []
                   current_target = target_signal_per_subband
                   last_idx = 0
                   # Find indices where cumulative signal crosses target boundaries
                   for i in range(num_subbands - 1):
                        # searchsorted finds insertion point; ensures index > last_idx
                        found_idx = np.searchsorted(cumul_signal, current_target, side='left')
                        # Ensure index advances and doesn't leave too few points at the end
                        found_idx = max(found_idx, last_idx + 1)
                        # Ensure enough channels left for remaining subbands (at least 1 each)
                        found_idx = min(found_idx, n_chan - (num_subbands - 1 - i))
                        split_indices.append(found_idx)
                        last_idx = found_idx
                        current_target += target_signal_per_subband
    else:
        raise ValueError(f"Unknown divide_method: {divide_method}")

    # Split the indices using the calculated split points
    subband_indices = np.split(all_indices, split_indices)

    # --- Analyze each sub-band ---
    for i, indices in enumerate(subband_indices):
        if len(indices) < 2:
            warnings.warn(f"Sub-band {i} has less than 2 channels, skipping analysis.")
            continue

        sub_spec = spec_ma[indices]
        sub_freqs = freqs_mhz[indices]
        sub_mask = spec_ma.mask[indices] if np.ma.is_masked(spec_ma) else None

        # Calculate offspec_mean specific to this sub-band if offspec_spectrum provided
        current_kwargs = kwargs.copy()
        if 'offspec_spectrum' in current_kwargs:
             offspec_full = current_kwargs.pop('offspec_spectrum') # Remove from kwargs passed down
             if offspec_full is not None and len(offspec_full) == n_chan:
                  # Assume offspec_full is 1D array or masked array
                  sub_offspec_data = offspec_full[indices]
                  if isinstance(sub_offspec_data, np.ma.MaskedArray):
                       # Calculate mean only if there are unmasked points
                       current_kwargs['offspec_mean'] = np.ma.mean(sub_offspec_data) if sub_offspec_data.count() > 0 else 0.0
                  else: # Assume simple numpy array
                       # Ensure it's not empty before taking mean
                       current_kwargs['offspec_mean'] = np.mean(sub_offspec_data) if len(sub_offspec_data) > 0 else 0.0
             elif 'offspec_mean' not in current_kwargs:
                  # If full offspec not provided or wrong shape, use global if passed, else 0
                  current_kwargs['offspec_mean'] = kwargs.get('offspec_mean', 0.0)


        print(f"--- Analyzing Sub-band {i} ({sub_freqs[0]:.1f} - {sub_freqs[-1]:.1f} MHz) ---")
        analysis = analyze_spectrum(sub_spec.data, sub_freqs, mask=sub_mask, **current_kwargs)

        sub_results = {
            'subband_index': i,
            'freq_center_mhz': np.mean(sub_freqs),
            'freq_range_mhz': (sub_freqs[0], sub_freqs[-1]),
            'num_channels': len(indices),
            'analysis_results': analysis
        }
        subband_results.append(sub_results)

    return subband_results

# --- Time Domain Analysis ---

def analyze_modulation_over_time(dynamic_spectrum, times_sec, burst_indices,
                                 time_chunk_size_bins, time_overlap_bins=0,
                                 freqs_mhz=None, freq_range_mhz=None):
    """
    Calculates the modulation index (std/mean) over time chunks.

    Args:
        dynamic_spectrum (np.ma.MaskedArray): 2D dynamic spectrum (freq, time).
        times_sec (np.ndarray): 1D array of time coordinates for each bin [seconds].
        burst_indices (tuple): Start and end bin index of the burst (start_bin, end_bin).
        time_chunk_size_bins (int): Number of time bins per analysis chunk.
        time_overlap_bins (int, optional): Number of overlapping bins between chunks. Defaults to 0.
        freqs_mhz (np.ndarray, optional): 1D array of frequencies. Required if freq_range_mhz is used.
        freq_range_mhz (tuple, optional): (min_freq, max_freq) to average over.
                                          Defaults to None (use all frequencies).

    Returns:
        list[dict]: List of dictionaries, one per time chunk, containing:
            'time_center_sec': Center time of the chunk.
            'time_range_sec': (start_time, end_time) of the chunk.
            'mod_index': Calculated modulation index (std/mean).
            'mean': Mean intensity in the chunk.
            'std_dev': Standard deviation in the chunk.
            'num_points': Number of unmasked data points used.
            'status': 'OK' or 'Error: message'.
    """
    results_list = []
    if not isinstance(dynamic_spectrum, np.ma.MaskedArray):
         dynamic_spectrum = np.ma.masked_array(dynamic_spectrum) # Ensure it's masked

    n_freq, n_time = dynamic_spectrum.shape
    burst_start, burst_end = burst_indices
    # Adjust burst_end to be exclusive, matching Python slicing
    burst_end_exclusive = min(burst_end, n_time)
    burst_duration_bins = burst_end_exclusive - burst_start

    if burst_duration_bins <= 0:
        warnings.warn("Burst duration is non-positive based on indices.")
        return results_list

    if time_chunk_size_bins <= 1:
        warnings.warn("time_chunk_size_bins must be > 1 to calculate standard deviation.")
        return results_list

    # Determine frequency range indices
    if freq_range_mhz is not None:
        if freqs_mhz is None:
             raise ValueError("freqs_mhz must be provided if freq_range_mhz is set.")
        freq_min, freq_max = min(freq_range_mhz), max(freq_range_mhz) # Ensure order
        # Find indices corresponding to the frequency range
        if freqs_mhz[0] > freqs_mhz[-1]: # Frequencies decreasing
             freq_mask = (freqs_mhz <= freq_max) & (freqs_mhz >= freq_min)
        else: # Frequencies increasing
             freq_mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freq_indices = np.where(freq_mask)[0]

        if len(freq_indices) == 0:
             warnings.warn(f"No frequency channels found in range {freq_range_mhz} MHz.")
             return results_list
    else:
        freq_indices = np.arange(n_freq)

    # Calculate step size for non-overlapping part of chunks
    time_step_bins = time_chunk_size_bins - time_overlap_bins
    if time_step_bins <= 0:
        warnings.warn("time_overlap_bins >= time_chunk_size_bins, results in zero or negative step. Setting overlap to 0.")
        time_step_bins = time_chunk_size_bins
        time_overlap_bins = 0


    # Iterate through time chunks within the burst duration
    # range(start, stop, step) - stop is exclusive
    for t_start in range(burst_start, burst_end_exclusive, time_step_bins):
        t_end = t_start + time_chunk_size_bins
        # Ensure chunk doesn't exceed burst end
        t_end = min(t_end, burst_end_exclusive)
        actual_chunk_size = t_end - t_start

        if actual_chunk_size < 2 : continue # Need at least 2 points for std dev

        time_slice_indices = np.arange(t_start, t_end)

        # Extract the 2D data chunk (freq x time) using advanced indexing
        data_chunk_2d = dynamic_spectrum[freq_indices[:, np.newaxis], time_slice_indices]

        # Average over frequency first to get a 1D time series for the chunk
        # Only include time bins where *at least one* frequency channel is unmasked
        # Axis 0 is frequency. Keepdims=True helps with broadcasting later if needed.
        time_series_chunk = np.ma.mean(data_chunk_2d, axis=0)

        # Calculate stats on the 1D time series
        chunk_mean = np.ma.mean(time_series_chunk)
        chunk_std = np.ma.std(time_series_chunk)
        # Count valid points in the *time series* after frequency averaging
        chunk_count = time_series_chunk.count()

        chunk_result = {
            'time_center_sec': np.mean(times_sec[time_slice_indices]),
            'time_range_sec': (times_sec[t_start], times_sec[t_end-1]),
            'mod_index': np.nan,
            'mean': chunk_mean if chunk_mean is not np.ma.masked else np.nan,
            'std_dev': chunk_std if chunk_std is not np.ma.masked else np.nan,
            'num_points': chunk_count,
            'status': 'OK'
        }

        if chunk_count < 2:
            chunk_result['status'] = 'Error: < 2 valid points in time chunk'
        elif np.ma.is_masked(chunk_mean) or np.isnan(chunk_mean) or chunk_mean == 0:
            # Handle mean=0 specifically, m=inf might be valid in noiseless case, but nan is safer
            chunk_result['status'] = 'Error: Mean is zero, masked, or NaN'
            chunk_result['mod_index'] = np.nan
        elif np.ma.is_masked(chunk_std) or np.isnan(chunk_std):
             chunk_result['status'] = 'Error: Std deviation is masked or NaN'
        else:
            # Ensure mean isn't dangerously small before division
            if abs(chunk_mean) < 1e-12:
                 chunk_result['status'] = 'Warning: Mean is very close to zero'
                 chunk_result['mod_index'] = np.inf # Or NaN? Inf seems more indicative
            else:
                 chunk_result['mod_index'] = chunk_std / chunk_mean

        results_list.append(chunk_result)

    return results_list


# --- Plotting Functions ---

def plot_acf_fit(ax, lags_mhz, acf, fit_result, params, title="ACF Fit", fontsize=8):
    """
    Plots ACF and fit on a given matplotlib axes object.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        lags_mhz (np.ndarray): Symmetric frequency lags [MHz].
        acf (np.ndarray): Symmetric ACF values.
        fit_result (lmfit.model.ModelResult): lmfit result object.
        params (dict): Dictionary of extracted parameters.
        title (str): Title for the subplot.
        fontsize (int): Fontsize for title and labels.
    """
    # Check if acf contains any valid data before plotting
    if acf is None or len(lags_mhz) == 0 or np.all(np.isnan(acf)):
        ax.text(0.5, 0.5, 'No valid ACF data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=fontsize-1)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
        return

    # Plot data
    ax.plot(lags_mhz, acf, drawstyle='steps-mid', label='ACF Data', color='k', lw=0.8)

    # Plot fit if available
    if fit_result is not None and params is not None:
        plot_lags = np.linspace(lags_mhz.min(), lags_mhz.max(), 200) # Fewer points for multiplot
        fit_line = fit_result.model.eval(params=fit_result.params, x=plot_lags)
        fwhm_val = params.get("fwhm_mhz", np.nan)
        mod_idx_val = params.get("mod_index", np.nan)
        # Short label for subplot
        fit_label = f'F={fwhm_val:.2f}, m={mod_idx_val:.2f}'
        ax.plot(plot_lags, fit_line, label=fit_label, color='r', alpha=0.8, lw=1.0)
        ax.legend(fontsize=fontsize-2, loc='upper right') # Smaller legend

    # --- Axis limits and labels ---
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-1)


    # Sensible X limits, e.g., based on FWHM or max lag used
    max_lag_plot = lags_mhz[-1]
    if params and 'fwhm_mhz' in params and params['fwhm_mhz'] is not None and not np.isnan(params['fwhm_mhz']):
         try:
              freq_res = fit_result.userkws.get('freq_res_mhz', 0.01) # Get freq_res if stored
              # Zoom in, but show at least +/- 2*FWHM or +/- 5*freq_res
              plot_lim_x = max(5 * freq_res, min(5 * params['fwhm_mhz'], max_lag_plot))
              ax.set_xlim(-plot_lim_x, plot_lim_x)
         except Exception as e:
              warnings.warn(f"Could not set ACF plot x-limits for subplot '{title}': {e}")
              ax.set_xlim(-max_lag_plot, max_lag_plot) # Fallback
    else:
        ax.set_xlim(-max_lag_plot, max_lag_plot) # Fallback to full range if no fit


    # Sensible Y limits
    min_acf_val = np.nanmin(acf) if not np.all(np.isnan(acf)) else -0.1
    max_acf_val = np.nanmax(acf) if not np.all(np.isnan(acf)) else 1.1
    # Add padding, ensure range is reasonable
    y_min = min(min_acf_val if not np.isnan(min_acf_val) else -0.1, -0.1) - 0.05
    y_max = max(max_acf_val if not np.isnan(max_acf_val) else 1.1, 0.1) + 0.05
    # Prevent excessively large y-range if offset is weird
    y_range = y_max - y_min
    if y_range > 5.0: # Arbitrary sanity check limit on range
        y_max = y_min + 5.0
    ax.set_ylim(y_min , y_max)


def plot_subband_summary(subband_results_list, param='fwhm_mhz'):
    """Plots a chosen parameter (e.g., FWHM) vs frequency across sub-bands."""
    freq_centers = []
    values = []
    errors = []
    param_err_key = param + "_err"

    for result in subband_results_list:
        freq_centers.append(result['freq_center_mhz'])
        analysis = result['analysis_results']
        # Check status and if params dict exists and is not None
        if analysis.get('status', 'Error') in ['OK', 'Warning: ACF fitting or parameter extraction failed.'] and analysis.get('params') is not None:
            param_val = analysis['params'].get(param, np.nan) # Use get for safety
            err_val = analysis['params'].get(param_err_key, np.nan) # Use get for safety
            values.append(param_val)
            errors.append(err_val)
        else:
            values.append(np.nan)
            errors.append(np.nan)

    values = np.array(values)
    errors = np.array(errors)
    freq_centers = np.array(freq_centers)

    # Filter out NaN values for plotting robustness
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        print(f"No valid data points found for parameter '{param}' to plot.")
        # Still create plot, but indicate no data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f'No valid data for {param}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"Scintillation Parameter vs. Frequency")
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(f"{param} " + ("[MHz]" if "mhz" in param else ""))
        return fig, ax # Return figure and axes

    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Filter errors as well - plot only points with valid values, errorbar handles NaN errors gracefully
        ax.errorbar(freq_centers[valid_mask], values[valid_mask], yerr=errors[valid_mask],
                    fmt='o', capsize=3, label=param, ecolor='gray', alpha=0.75)

        # Optional: Add theoretical scaling line (e.g., nu^4 for Kolmogorov)
        if param == 'fwhm_mhz' and np.any(valid_mask):
            try:
                # Fit power law or just plot nu^4 from highest freq point
                valid_freqs = freq_centers[valid_mask]
                valid_vals = values[valid_mask]
                # Find highest frequency point with valid data
                idx_high_freq = np.argmax(valid_freqs)
                ref_freq = valid_freqs[idx_high_freq]
                ref_val = valid_vals[idx_high_freq]
                if ref_freq > 0 and not np.isnan(ref_freq) and ref_val > 0 and not np.isnan(ref_val): # Need positive ref values for scaling
                    scaling_freqs = np.linspace(np.min(valid_freqs), np.max(valid_freqs), 100)
                    # nu^4 scaling for FWHM
                    scaled_vals = ref_val * (scaling_freqs / ref_freq)**(4.0)
                    ax.plot(scaling_freqs, scaled_vals, ls='--', color='r', label=r'$\propto \nu^{4.0}$ scaling')
            except Exception as e:
                print(f"Could not plot scaling law: {e}")


    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(f"{param} " + ("[MHz]" if "mhz" in param else ""))
    ax.set_title(f"Scintillation Parameter vs. Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param == 'mod_index':
         ax.set_ylim(bottom=0) # Modulation index >= 0
    fig.tight_layout()
    return fig, ax # Return figure and axes
    # plt.show() # Or save figure


def plot_modulation_vs_time(time_analysis_results):
    """Plots the modulation index vs time."""
    times = []
    mod_indices = []
    means = []

    for result in time_analysis_results:
        if result['status'] == 'OK':
            times.append(result['time_center_sec'])
            mod_indices.append(result['mod_index'])
            means.append(result['mean'])
        else:
             # Optionally plot failed points differently or skip
             # times.append(result['time_center_sec'])
             # mod_indices.append(np.nan)
             # means.append(np.nan)
             pass # Skip failed points for now


    fig, ax1 = plt.subplots(figsize=(10, 5))
    if not times:
        print("No valid time analysis results to plot.")
        ax1.text(0.5, 0.5, 'No valid time analysis data', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title("Modulation Index vs. Time")
        return fig, ax1

    times = np.array(times)
    mod_indices = np.array(mod_indices)
    means = np.array(means)


    color = 'tab:red'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Modulation Index (std/mean)', color=color)
    ax1.plot(times, mod_indices, marker='o', ls='-', color=color, label='Modulation Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0) # Modulation index >= 0

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Mean Intensity (arb. units)', color=color)
    ax2.plot(times, means, marker='.', ls=':', color=color, alpha=0.6, label='Mean Intensity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)

    fig.suptitle("Modulation Index and Mean Intensity vs. Time")
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    return fig, ax1
    # plt.show() # Or save figure


# --- NEW: Multi-panel ACF Plot ---
def plot_all_subband_acf_fits(subband_results_list, **kwargs):
    """
    Creates a multi-panel plot showing ACF and fit for each sub-band.

    Args:
        subband_results_list (list[dict]): The list of results from analyze_subbands.
        **kwargs: Additional keyword arguments passed to fig.suptitle.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    num_subbands = len(subband_results_list)
    if num_subbands == 0:
        print("No subband results to plot.")
        return plt.figure() # Return empty figure

    # Determine grid size (e.g., aiming for roughly square)
    ncols = math.ceil(math.sqrt(num_subbands))
    nrows = math.ceil(num_subbands / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                             sharex=False, sharey=False, squeeze=False) # Don't share axes initially
    axes_flat = axes.flatten()

    # Determine common x and y limits across valid plots for potential sharing
    valid_lags = [res['analysis_results']['lags_mhz'] for res in subband_results_list if res['analysis_results']['status'] != 'Error: ACF calculation failed.' and len(res['analysis_results']['lags_mhz']) > 0]
    valid_acfs = [res['analysis_results']['acf'] for res in subband_results_list if res['analysis_results']['status'] != 'Error: ACF calculation failed.' and len(res['analysis_results']['acf']) > 0]

    common_xlim = None
    common_ylim = None

    if valid_lags:
         max_lag_all = max(lag[-1] for lag in valid_lags)
         # Potentially base xlim on median FWHM or max lag? For now, use max lag found.
         median_fwhm = np.nanmedian([res['analysis_results']['params'].get('fwhm_mhz', np.nan)
                                     for res in subband_results_list if res['analysis_results'].get('params')])
         if not np.isnan(median_fwhm):
              common_xlim_val = max(5 * 0.01, min(10 * median_fwhm, max_lag_all)) # Heuristic similar to single plot
              common_xlim = (-common_xlim_val, common_xlim_val)
         else:
             common_xlim = (-max_lag_all, max_lag_all)


    if valid_acfs:
        all_min = min(np.nanmin(acf) for acf in valid_acfs if not np.all(np.isnan(acf)))
        all_max = max(np.nanmax(acf) for acf in valid_acfs if not np.all(np.isnan(acf)))
        y_min = min(all_min if not np.isnan(all_min) else -0.1, -0.1) - 0.05
        y_max = max(all_max if not np.isnan(all_max) else 1.1, 0.1) + 0.05
        # Prevent excessive range
        y_range = y_max - y_min
        if y_range > 5.0: y_max = y_min + 5.0
        common_ylim = (y_min, y_max)


    for i, sub_result in enumerate(subband_results_list):
        ax = axes_flat[i]
        analysis = sub_result['analysis_results']
        freq_range = sub_result['freq_range_mhz']
        title = f"Sub {i}: {freq_range[0]:.0f}-{freq_range[1]:.0f} MHz"

        plot_acf_fit(ax=ax,
                     lags_mhz=analysis.get('lags_mhz', np.array([])),
                     acf=analysis.get('acf', None),
                     fit_result=analysis.get('fit_result', None),
                     params=analysis.get('params', None),
                     title=title,
                     fontsize=9) # Slightly larger font for multi-plot

        # Optionally set common limits if determined
        if common_xlim is not None: ax.set_xlim(common_xlim)
        if common_ylim is not None: ax.set_ylim(common_ylim)

        # Axis labels only on outer plots
        row, col = divmod(i, ncols)
        if row == nrows - 1: # Bottom row
            ax.set_xlabel("Lag [MHz]", fontsize=9)
        else:
            ax.set_xlabel("")
            #ax.tick_params(axis='x', labelbottom=False) # Keep ticks for reference?

        if col == 0: # Leftmost column
            ax.set_ylabel("Norm. ACF", fontsize=9)
        else:
            ax.set_ylabel("")
            #ax.tick_params(axis='y', labelleft=False)


    # Hide unused axes
    for j in range(num_subbands, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("ACF Fits per Sub-band", fontsize=12, **kwargs)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    return fig
