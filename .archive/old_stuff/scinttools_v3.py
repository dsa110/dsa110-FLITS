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
    return (
        lorentzian_model(x, amp1, cen1, gam1) +
        lorentzian_model(x, amp2, cen2, gam2) +
        offset
    )


def calculate_acf(spectrum, mask=None, max_lag_bins=None, mean_subtract=True,
                  normalize=True, offspec_mean=0.0):
    """
    Calculates the Auto-Correlation Function (ACF) of a 1D spectrum.
    
    Args:
        spectrum (np.ndarray): 1D array containing the spectrum data.
        mask (np.ndarray, optional): Boolean mask for the spectrum. Defaults to None.
        max_lag_bins (int, optional): Maximum lag (in frequency bins) to compute.
        mean_subtract (bool, optional): Subtract the mean before calculating ACF.
        normalize (bool, optional): Normalize such that ACF(0) ~ m^2.
        offspec_mean (float, optional): Mean of off-burst spectrum.
        
    Returns:
        tuple: (lags_bins, acf) or (np.array([]), None) on failure.
    """
    if not isinstance(spectrum, np.ma.MaskedArray):
        spec_ma = np.ma.masked_array(spectrum, mask=mask)
    else:
        spec_ma = np.ma.masked_array(
            spectrum.data,
            mask=np.logical_or(spectrum.mask, mask if mask is not None else False)
        )

    if spec_ma.count() < 2:
        warnings.warn("Insufficient unmasked data points to calculate ACF.")
        return np.array([]), None

    valid_spec = spec_ma.compressed()
    if valid_spec.size < 2:
        warnings.warn("Insufficient unmasked data points after compression.")
        return np.array([]), None

    mean_spec = np.mean(valid_spec)

    if mean_subtract:
        work_spec = spec_ma.astype(float, copy=True)
        work_spec[~spec_ma.mask] -= mean_spec
        work_spec.fill_value = 0.0
        data_for_corr = work_spec.filled()
    else:
        work_spec = spec_ma.astype(float, copy=True)
        work_spec.fill_value = 0.0
        data_for_corr = work_spec.filled()

    n_chan = len(spectrum)
    if max_lag_bins is None:
        max_lag_bins = n_chan - 1
    else:
        max_lag_bins = min(max_lag_bins, n_chan - 1)

    lags_bins = np.arange(max_lag_bins + 1)
    acf = np.zeros(max_lag_bins + 1, dtype=float)
    counts = np.zeros(max_lag_bins + 1, dtype=int)

    valid_indices = ~spec_ma.mask

    for lag in lags_bins:
        if lag == 0:
            valid_pair_indices = valid_indices
            if np.sum(valid_pair_indices) > 0:
                acf[lag] = np.sum(work_spec[valid_pair_indices] ** 2)
                counts[lag] = np.sum(valid_pair_indices)
            else:
                acf[lag] = np.nan
                counts[lag] = 0
            continue

        idx1 = np.arange(n_chan - lag)
        idx2 = np.arange(lag, n_chan)

        valid_pair_indices = valid_indices[idx1] & valid_indices[idx2]

        if np.sum(valid_pair_indices) > 0:
            acf[lag] = np.sum(
                work_spec[idx1[valid_pair_indices]] *
                work_spec[idx2[valid_pair_indices]]
            )
            counts[lag] = np.sum(valid_pair_indices)
        else:
            acf[lag] = np.nan
            counts[lag] = 0

    valid_counts = counts > 0
    if np.any(valid_counts):
        acf[valid_counts] /= counts[valid_counts]
    else:
        return lags_bins, None

    if normalize:
        effective_mean = mean_spec - offspec_mean
        if effective_mean != 0:
            variance_est = acf[0] if mean_subtract else np.var(valid_spec)
            norm_factor = effective_mean ** 2
            if norm_factor > 1e-15:
                acf /= norm_factor
            else:
                warnings.warn(f"Normalization factor (effective_mean^2 = {norm_factor:.2e}) is close to zero. Skipping ACF normalization by power.")
                if variance_est > 1e-15:
                    acf /= variance_est
                    warnings.warn("Normalizing ACF by variance estimate instead.")
                else:
                    warnings.warn("Variance estimate also near zero. ACF remains unnormalized by power.")
                    acf = np.full_like(acf, np.nan)
        else:
            warnings.warn("Effective mean is zero. Cannot normalize ACF by power.")
            variance_est = acf[0] if mean_subtract else np.var(valid_spec)
            if variance_est > 1e-15:
                acf /= variance_est
                warnings.warn("Normalizing ACF by variance estimate instead.")
            else:
                warnings.warn("Variance estimate also near zero. ACF remains unnormalized.")
                acf = np.full_like(acf, np.nan)

    if np.all(np.isnan(acf)):
        return lags_bins, None

    return lags_bins, acf


def fit_acf_model(lags_bins, acf, freq_res_mhz, model_type='single',
                  fit_lag_range_mhz=None, initial_gamma_mhz=0.1):
    """
    Fits a Lorentzian model to the ACF.
    
    Args:
        lags_bins (np.ndarray): Lags in frequency bins.
        acf (np.ndarray): Auto-correlation function values.
        freq_res_mhz (float): Frequency resolution in MHz per bin.
        model_type (str): 'single' or 'double'. Defaults to 'single'.
        fit_lag_range_mhz (float, optional): Maximum lag for fitting.
        initial_gamma_mhz (float, optional): Initial guess for HWHM gamma in MHz.
        
    Returns:
        lmfit.model.ModelResult or None
    """
    if acf is None or len(acf) < 2 or np.all(np.isnan(acf)):
        warnings.warn("ACF is None or too short or all NaN, cannot fit.")
        return None

    lags_mhz = lags_bins * freq_res_mhz

    if fit_lag_range_mhz is not None:
        fit_indices = np.where(lags_mhz <= fit_lag_range_mhz)[0]
        if len(fit_indices) < 3:
            warnings.warn(f"Less than 3 data points within fit range {fit_lag_range_mhz} MHz. Cannot fit.")
            return None
        fit_lags_mhz = lags_mhz[fit_indices]
        fit_acf = acf[fit_indices]
    else:
        fit_lags_mhz = lags_mhz
        fit_acf = acf

    nan_mask = np.isnan(fit_acf)
    if np.all(nan_mask):
        warnings.warn("All ACF values in the fitting range are NaN. Cannot fit.")
        return None

    fit_lags_mhz = fit_lags_mhz[~nan_mask]
    fit_acf = fit_acf[~nan_mask]

    if len(fit_acf) < 3:
        warnings.warn("Less than 3 valid data points remaining for fit. Cannot fit.")
        return None

    if model_type == 'single':
        model = Model(lorentzian_model_with_offset)
        params = model.make_params()
        initial_amplitude = fit_acf[0] if fit_acf[0] > 0 else 1e-3
        params['amplitude'].set(value=initial_amplitude, min=1e-9)
        params['center'].set(value=0.0, vary=False)
        params['gamma'].set(value=initial_gamma_mhz, min=1e-6)
        offset_guess = np.nanmedian(fit_acf[len(fit_acf) // 2:])
        params['offset'].set(value=offset_guess if not np.isnan(offset_guess) else 0.0, min=-1.0, max=1.0)
    elif model_type == 'double':
        model = Model(double_lorentzian_model_with_offset)
        params = model.make_params()
        initial_amplitude = fit_acf[0] if fit_acf[0] > 0 else 1e-3
        params['amp1'].set(value=initial_amplitude * 0.7, min=1e-9)
        params['cen1'].set(value=0.0, vary=False)
        params['gam1'].set(value=initial_gamma_mhz, min=1e-6)
        params['amp2'].set(value=initial_amplitude * 0.3, min=1e-9)
        params['cen2'].set(value=0.0, vary=False)
        params['gam2'].set(value=initial_gamma_mhz * 5, min=1e-6)
        offset_guess = np.nanmedian(fit_acf[len(fit_acf) // 2:])
        params['offset'].set(value=offset_guess if not np.isnan(offset_guess) else 0.0, min=-1.0, max=1.0)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    try:
        fit_result = model.fit(fit_acf, params, x=fit_lags_mhz, nan_policy='omit')
        if not fit_result.success:
            warnings.warn(f"lmfit optimization failed: {fit_result.message}")
            return None
        return fit_result
    except Exception as e:
        warnings.warn(f"lmfit fitting process raised an exception: {e}")
        return None


def extract_scint_params(fit_result):
    """
    Extracts key scintillation parameters from an lmfit ModelResult.
    
    Args:
        fit_result (lmfit.model.ModelResult): Successful fit result.
        
    Returns:
        dict: Parameters such as 'fwhm_mhz', 'mod_index', etc., or None on failure.
    """
    if fit_result is None:
        return None

    params = fit_result.params
    extracted = {}

    try:
        amp1 = params['amplitude'].value if 'amplitude' in params else params['amp1'].value
        gam1 = params['gamma'].value if 'gamma' in params else params['gam1'].value
        amp1_err = params['amplitude'].stderr if 'amplitude' in params else params['amp1'].stderr
        gam1_err = params['gamma'].stderr if 'gamma' in params else params['gam1'].stderr

        extracted['fwhm_mhz'] = 2.0 * gam1
        extracted['fwhm_mhz_err'] = 2.0 * gam1_err if gam1_err is not None else np.nan

        extracted['mod_index'] = np.sqrt(amp1) if amp1 > 0 else 0.0
        if amp1 > 1e-9 and amp1_err is not None:
            if extracted['mod_index'] > 1e-9:
                extracted['mod_index_err'] = abs(amp1_err / (2.0 * extracted['mod_index']))
            else:
                extracted['mod_index_err'] = np.nan
        else:
            extracted['mod_index_err'] = np.nan

        extracted['offset'] = params['offset'].value
        extracted['offset_err'] = params['offset'].stderr if params['offset'].stderr is not None else np.nan

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
        warnings.warn(f"Parameter extraction failed due to type error: {e}. Returning partial results.")

    if np.isnan(extracted.get('fwhm_mhz', np.nan)) or np.isnan(extracted.get('mod_index', np.nan)):
        warnings.warn("Essential parameters (FWHM or ModIndex) could not be extracted or are NaN.")

    return extracted


def analyze_spectrum(spectrum, freqs_mhz, mask=None, max_lag_mhz=10.0,
                     model_type='single', fit_lag_range_mhz=None,
                     offspec_mean=0.0, initial_gamma_mhz=0.1):
    """
    Performs full ACF analysis on a single 1D spectrum.
    
    Returns a dict with keys:
    'lags_mhz', 'acf', 'fit_result', 'params', 'freq_res_mhz', 'status'
    """
    results = {'status': 'OK'}
    if len(freqs_mhz) < 2:
        results['status'] = 'Error: Need at least 2 frequency channels.'
        return results

    freq_diffs = np.diff(freqs_mhz)
    if len(freq_diffs) == 0:
        freq_res_mhz = np.abs(freqs_mhz[1] - freqs_mhz[0])
    elif np.allclose(freq_diffs, freq_diffs[0]):
        freq_res_mhz = np.abs(freq_diffs[0])
    else:
        freq_res_mhz = np.abs(np.mean(freq_diffs))
        warnings.warn("Frequency channels are not uniformly spaced. Using average resolution.")
    results['freq_res_mhz'] = freq_res_mhz

    if freq_res_mhz <= 0:
        results['status'] = 'Error: Invalid frequency resolution <= 0.'
        return results

    max_lag_bins = int(np.ceil(max_lag_mhz / freq_res_mhz))
    lags_bins, acf_raw = calculate_acf(spectrum, mask=mask, max_lag_bins=max_lag_bins,
                                       mean_subtract=True, normalize=True,
                                       offspec_mean=offspec_mean)

    if acf_raw is None:
        results['status'] = 'Error: ACF calculation failed.'
        results['lags_mhz'] = np.array([])
        results['acf'] = np.array([])
        results['fit_result'] = None
        results['params'] = None
        return results

    lags_mhz_sym = np.concatenate((-lags_bins[1:][::-1] * freq_res_mhz, lags_bins * freq_res_mhz))
    acf_sym = np.concatenate((acf_raw[1:][::-1], acf_raw))
    results['lags_mhz'] = lags_mhz_sym
    results['acf'] = acf_sym

    fit_range = fit_lag_range_mhz if fit_lag_range_mhz is not None else max_lag_mhz
    fit_result = fit_acf_model(lags_bins, acf_raw, freq_res_mhz,
                               model_type=model_type,
                               fit_lag_range_mhz=fit_range,
                               initial_gamma_mhz=initial_gamma_mhz)
    if fit_result is not None:
        fit_result.userkws['freq_res_mhz'] = freq_res_mhz

    results['fit_result'] = fit_result

    scint_params = extract_scint_params(fit_result)
    results['params'] = scint_params

    if fit_result is None or scint_params is None:
        current_status = results.get('status', 'OK')
        if current_status == 'OK':
            results['status'] = 'Warning: ACF fitting or parameter extraction failed.'
    return results


def analyze_subbands(spectrum, freqs_mhz, num_subbands=8, mask=None,
                     divide_method='equal_freq', **kwargs):
    """
    Divides a spectrum into sub-bands and analyzes each using analyze_spectrum.
    
    Returns a list of dicts for each sub-band.
    """
    if not isinstance(spectrum, np.ma.MaskedArray):
        spec_ma = np.ma.masked_array(spectrum, mask=mask)
    else:
        spec_ma = np.ma.masked_array(
            spectrum.data,
            mask=np.logical_or(spectrum.mask, mask if mask is not None else False)
        )

    n_chan = len(spec_ma)
    all_indices = np.arange(n_chan)
    subband_results = []

    if n_chan < num_subbands * 2:
        warnings.warn(f"Number of channels ({n_chan}) is low for {num_subbands} subbands. Reducing num_subbands.")
        num_subbands = max(1, n_chan // 2)
        if num_subbands == 0:
            warnings.warn("Fewer than 2 channels available. Cannot perform subband analysis.")
            return subband_results

    if divide_method == 'equal_freq':
        indices_per_subband = np.array_split(all_indices, num_subbands)
        split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]
    elif divide_method == 'equal_snr':
        valid_spec = spec_ma.compressed()
        if len(valid_spec) == 0:
            warnings.warn("Cannot use equal_snr division: No unmasked data. Falling back to 'equal_freq'.")
            indices_per_subband = np.array_split(all_indices, num_subbands)
            split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]
        else:
            mean_signal = np.mean(valid_spec)
            offspec_mean_val = 0.0
            if 'offspec_spectrum' in kwargs and kwargs['offspec_spectrum'] is not None:
                offspec_ma = np.ma.masked_array(kwargs['offspec_spectrum'])
                offspec_mean_val = np.ma.mean(offspec_ma) if offspec_ma.count() > 0 else 0.0
            elif 'offspec_mean' in kwargs:
                offspec_mean_val = kwargs['offspec_mean']
            signal_est = np.maximum(0, spec_ma.filled(0.0) - offspec_mean_val)
            cumul_signal = np.cumsum(signal_est)
            total_signal = cumul_signal[-1] if len(cumul_signal) > 0 else 0.0

            if total_signal <= 0:
                warnings.warn("Total estimated signal is non-positive. Cannot use 'equal_snr'. Falling back to 'equal_freq'.")
                indices_per_subband = np.array_split(all_indices, num_subbands)
                split_indices = [indices[0] for indices in indices_per_subband[1:] if len(indices) > 0]
            else:
                target_signal_per_subband = total_signal / num_subbands
                split_indices = []
                current_target = target_signal_per_subband
                last_idx = 0
                for i in range(num_subbands - 1):
                    found_idx = np.searchsorted(cumul_signal, current_target, side='left')
                    found_idx = max(found_idx, last_idx + 1)
                    found_idx = min(found_idx, n_chan - (num_subbands - 1 - i))
                    split_indices.append(found_idx)
                    last_idx = found_idx
                    current_target += target_signal_per_subband
    else:
        raise ValueError(f"Unknown divide_method: {divide_method}")

    subband_indices = np.split(all_indices, split_indices)

    for i, indices in enumerate(subband_indices):
        if len(indices) < 2:
            warnings.warn(f"Sub-band {i} has less than 2 channels, skipping analysis.")
            continue

        sub_spec = spec_ma[indices]
        sub_freqs = freqs_mhz[indices]
        sub_mask = spec_ma.mask[indices] if np.ma.is_masked(spec_ma) else None

        current_kwargs = kwargs.copy()
        if 'offspec_spectrum' in current_kwargs:
            offspec_full = current_kwargs.pop('offspec_spectrum')
            if offspec_full is not None and len(offspec_full) == n_chan:
                sub_offspec_data = offspec_full[indices]
                if isinstance(sub_offspec_data, np.ma.MaskedArray):
                    current_kwargs['offspec_mean'] = np.ma.mean(sub_offspec_data) if sub_offspec_data.count() > 0 else 0.0
                else:
                    current_kwargs['offspec_mean'] = np.mean(sub_offspec_data) if len(sub_offspec_data) > 0 else 0.0
            elif 'offspec_mean' not in current_kwargs:
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


def analyze_modulation_over_time(dynamic_spectrum, times_sec, burst_indices,
                                 time_chunk_size_bins, time_overlap_bins=0,
                                 freqs_mhz=None, freq_range_mhz=None):
    """
    Calculates the modulation index (std/mean) over time chunks.
    
    Returns a list of dictionaries with analysis results per time chunk.
    """
    results_list = []
    if not isinstance(dynamic_spectrum, np.ma.MaskedArray):
        dynamic_spectrum = np.ma.masked_array(dynamic_spectrum)

    n_freq, n_time = dynamic_spectrum.shape
    burst_start, burst_end = burst_indices
    burst_end_exclusive = min(burst_end, n_time)
    burst_duration_bins = burst_end_exclusive - burst_start

    if burst_duration_bins <= 0:
        warnings.warn("Burst duration is non-positive based on indices.")
        return results_list

    if time_chunk_size_bins <= 1:
        warnings.warn("time_chunk_size_bins must be > 1 to calculate standard deviation.")
        return results_list

    if freq_range_mhz is not None:
        if freqs_mhz is None:
            raise ValueError("freqs_mhz must be provided if freq_range_mhz is set.")
        freq_min, freq_max = min(freq_range_mhz), max(freq_range_mhz)
        if freqs_mhz[0] > freqs_mhz[-1]:
            freq_mask = (freqs_mhz <= freq_max) & (freqs_mhz >= freq_min)
        else:
            freq_mask = (freqs_mhz >= freq_min) & (freqs_mhz <= freq_max)
        freq_indices = np.where(freq_mask)[0]
        if len(freq_indices) == 0:
            warnings.warn(f"No frequency channels found in range {freq_range_mhz} MHz.")
            return results_list
    else:
        freq_indices = np.arange(n_freq)

    time_step_bins = time_chunk_size_bins - time_overlap_bins
    if time_step_bins <= 0:
        warnings.warn("time_overlap_bins >= time_chunk_size_bins, results in zero or negative step. Setting overlap to 0.")
        time_step_bins = time_chunk_size_bins
        time_overlap_bins = 0

    for t_start in range(burst_start, burst_end_exclusive, time_step_bins):
        t_end = t_start + time_chunk_size_bins
        t_end = min(t_end, burst_end_exclusive)
        actual_chunk_size = t_end - t_start

        if actual_chunk_size < 2:
            continue

        time_slice_indices = np.arange(t_start, t_end)
        data_chunk_2d = dynamic_spectrum[freq_indices[:, np.newaxis], time_slice_indices]
        time_series_chunk = np.ma.mean(data_chunk_2d, axis=0)
        chunk_mean = np.ma.mean(time_series_chunk)
        chunk_std = np.ma.std(time_series_chunk)
        chunk_count = time_series_chunk.count()

        chunk_result = {
            'time_center_sec': np.mean(times_sec[time_slice_indices]),
            'time_range_sec': (times_sec[t_start], times_sec[t_end - 1]),
            'mod_index': np.nan,
            'mean': chunk_mean if chunk_mean is not np.ma.masked else np.nan,
            'std_dev': chunk_std if chunk_std is not np.ma.masked else np.nan,
            'num_points': chunk_count,
            'status': 'OK'
        }

        if chunk_count < 2:
            chunk_result['status'] = 'Error: < 2 valid points in time chunk'
        elif np.ma.is_masked(chunk_mean) or np.isnan(chunk_mean) or chunk_mean == 0:
            chunk_result['status'] = 'Error: Mean is zero, masked, or NaN'
            chunk_result['mod_index'] = np.nan
        elif np.ma.is_masked(chunk_std) or np.isnan(chunk_std):
            chunk_result['status'] = 'Error: Std deviation is masked or NaN'
        else:
            if abs(chunk_mean) < 1e-12:
                chunk_result['status'] = 'Warning: Mean is very close to zero'
                chunk_result['mod_index'] = np.inf
            else:
                chunk_result['mod_index'] = chunk_std / chunk_mean

        results_list.append(chunk_result)

    return results_list


def plot_acf_fit(ax, lags_mhz, acf, fit_result, params, title="ACF Fit", fontsize=8):
    """
    Plots ACF and fit on a given matplotlib axes object.
    """
    if acf is None or len(lags_mhz) == 0 or np.all(np.isnan(acf)):
        ax.text(0.5, 0.5, 'No valid ACF data', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=fontsize - 1)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)
        return

    ax.plot(lags_mhz, acf, drawstyle='steps-mid', label='ACF Data', color='k', lw=0.8)

    if fit_result is not None and params is not None:
        plot_lags = np.linspace(lags_mhz.min(), lags_mhz.max(), 200)
        fit_line = fit_result.model.eval(params=fit_result.params, x=plot_lags)
        fwhm_val = params.get("fwhm_mhz", np.nan)
        mod_idx_val = params.get("mod_index", np.nan)
        fit_label = f'F={fwhm_val:.2f}, m={mod_idx_val:.2f}'
        ax.plot(plot_lags, fit_line, label=fit_label, color='r', alpha=0.8, lw=1.0)
        ax.legend(fontsize=fontsize - 2, loc='upper right')

    ax.set_title(title, fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    max_lag_plot = lags_mhz[-1]
    if params and 'fwhm_mhz' in params and params['fwhm_mhz'] is not None and not np.isnan(params['fwhm_mhz']):
        try:
            freq_res = fit_result.userkws.get('freq_res_mhz', 0.01)
            plot_lim_x = max(5 * freq_res, min(5 * params['fwhm_mhz'], max_lag_plot))
            ax.set_xlim(-plot_lim_x, plot_lim_x)
        except Exception as e:
            warnings.warn(f"Could not set ACF plot x-limits for subplot '{title}': {e}")
            ax.set_xlim(-max_lag_plot, max_lag_plot)
    else:
        ax.set_xlim(-max_lag_plot, max_lag_plot)

    min_acf_val = np.nanmin(acf) if not np.all(np.isnan(acf)) else -0.1
    max_acf_val = np.nanmax(acf) if not np.all(np.isnan(acf)) else 1.1
    y_min = min(min_acf_val if not np.isnan(min_acf_val) else -0.1, -0.1) - 0.05
    y_max = max(max_acf_val if not np.isnan(max_acf_val) else 1.1, 0.1) + 0.05
    y_range = y_max - y_min
    if y_range > 5.0:
        y_max = y_min + 5.0
    ax.set_ylim(y_min, y_max)


def plot_subband_summary(subband_results_list, param='fwhm_mhz'):
    """
    Plots a chosen parameter (e.g., FWHM) vs frequency across sub-bands.
    """
    freq_centers = []
    values = []
    errors = []
    param_err_key = param + "_err"

    for result in subband_results_list:
        freq_centers.append(result['freq_center_mhz'])
        analysis = result['analysis_results']
        if analysis.get('status', 'Error') in ['OK', 'Warning: ACF fitting or parameter extraction failed.'] and analysis.get('params') is not None:
            param_val = analysis['params'].get(param, np.nan)
            err_val = analysis['params'].get(param_err_key, np.nan)
            values.append(param_val)
            errors.append(err_val)
        else:
            values.append(np.nan)
            errors.append(np.nan)

    values = np.array(values)
    errors = np.array(errors)
    freq_centers = np.array(freq_centers)

    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        print(f"No valid data points found for parameter '{param}' to plot.")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f'No valid data for {param}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("Scintillation Parameter vs. Frequency")
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(f"{param} " + ("[MHz]" if "mhz" in param else ""))
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(freq_centers[valid_mask], values[valid_mask], yerr=errors[valid_mask],
                    fmt='o', capsize=3, label=param, ecolor='gray', alpha=0.75)

        if param == 'fwhm_mhz' and np.any(valid_mask):
            try:
                valid_freqs = freq_centers[valid_mask]
                valid_vals = values[valid_mask]
                idx_high_freq = np.argmax(valid_freqs)
                ref_freq = valid_freqs[idx_high_freq]
                ref_val = valid_vals[idx_high_freq]
                if ref_freq > 0 and not np.isnan(ref_freq) and ref_val > 0 and not np.isnan(ref_val):
                    scaling_freqs = np.linspace(np.min(valid_freqs), np.max(valid_freqs), 100)
                    scaled_vals = ref_val * (scaling_freqs / ref_freq) ** 4.0
                    ax.plot(scaling_freqs, scaled_vals, ls='--', color='r', label=r'$\propto \nu^{4.0}$ scaling')
            except Exception as e:
                print(f"Could not plot scaling law: {e}")

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(f"{param} " + ("[MHz]" if "mhz" in param else ""))
    ax.set_title("Scintillation Parameter vs. Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param == 'mod_index':
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig, ax


def plot_modulation_vs_time(time_analysis_results):
    """
    Plots the modulation index vs time.
    """
    times = []
    mod_indices = []
    means = []

    for result in time_analysis_results:
        if result['status'] == 'OK':
            times.append(result['time_center_sec'])
            mod_indices.append(result['mod_index'])
            means.append(result['mean'])

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
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean Intensity (arb. units)', color=color)
    ax2.plot(times, means, marker='.', ls=':', color=color, alpha=0.6, label='Mean Intensity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)

    fig.suptitle("Modulation Index and Mean Intensity vs. Time")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, ax1


def plot_all_subband_acf_fits(subband_results_list, **kwargs):
    """
    Creates a multi-panel plot showing ACF and fit for each sub-band.
    """
    num_subbands = len(subband_results_list)
    if num_subbands == 0:
        print("No subband results to plot.")
        return plt.figure()

    ncols = math.ceil(math.sqrt(num_subbands))
    nrows = math.ceil(num_subbands / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                             sharex=False, sharey=False, squeeze=False)
    axes_flat = axes.flatten()

    valid_lags = [res['analysis_results']['lags_mhz']
                  for res in subband_results_list
                  if res['analysis_results']['status'] != 'Error: ACF calculation failed.' and len(res['analysis_results']['lags_mhz']) > 0]
    valid_acfs = [res['analysis_results']['acf']
                  for res in subband_results_list
                  if res['analysis_results']['status'] != 'Error: ACF calculation failed.' and len(res['analysis_results']['acf']) > 0]

    common_xlim = None
    common_ylim = None

    if valid_lags:
        max_lag_all = max(lag[-1] for lag in valid_lags)
        median_fwhm = np.nanmedian([
            res['analysis_results']['params'].get('fwhm_mhz', np.nan)
            for res in subband_results_list
            if res['analysis_results'].get('params') is not None
        ])
        if not np.isnan(median_fwhm):
            common_xlim_val = max(5 * 0.01, min(10 * median_fwhm, max_lag_all))
            common_xlim = (-common_xlim_val, common_xlim_val)
        else:
            common_xlim = (-max_lag_all, max_lag_all)

    if valid_acfs:
        all_min = min(np.nanmin(acf)
                      for acf in valid_acfs
                      if not np.all(np.isnan(acf)))
        all_max = max(np.nanmax(acf)
                      for acf in valid_acfs
                      if not np.all(np.isnan(acf)))
        y_min = min(all_min if not np.isnan(all_min) else -0.1, -0.1) - 0.05
        y_max = max(all_max if not np.isnan(all_max) else 1.1, 0.1) + 0.05
        y_range = y_max - y_min
        if y_range > 5.0:
            y_max = y_min + 5.0
        common_ylim = (y_min, y_max)

    for i, sub_result in enumerate(subband_results_list):
        ax = axes_flat[i]
        analysis = sub_result['analysis_results']
        freq_range = sub_result['freq_range_mhz']
        title = f"Sub {i}: {freq_range[0]:.0f}-{freq_range[1]:.0f} MHz"

        plot_acf_fit(
            ax=ax,
            lags_mhz=analysis.get('lags_mhz', np.array([])),
            acf=analysis.get('acf', None),
            fit_result=analysis.get('fit_result', None),
            params=analysis.get('params', None),
            title=title,
            fontsize=9
        )

        if common_xlim is not None:
            ax.set_xlim(common_xlim)
        if common_ylim is not None:
            ax.set_ylim(common_ylim)

        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel("Lag [MHz]", fontsize=9)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("Norm. ACF", fontsize=9)
        else:
            ax.set_ylabel("")

    for j in range(num_subbands, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("ACF Fits per Sub-band", fontsize=12, **kwargs)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig