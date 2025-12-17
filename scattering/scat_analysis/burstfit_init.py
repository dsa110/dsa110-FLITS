"""
burstfit_init.py
================

Data-driven initial parameter estimation for FRB scattering analysis.

Instead of using hardcoded guesses, this module extracts parameter estimates
directly from the dynamic spectrum properties:

**Parameters Estimated:**
- `c0`: Total burst amplitude from integrated profile
- `t0`: Peak time from profile maximum
- `gamma`: Spectral index from frequency-resolved flux
- `zeta`: Intrinsic width from deconvolved pulse width
- `tau_1ghz`: Scattering timescale from exponential tail fit
- `alpha`: Scattering spectral index from frequency scaling

**Methods:**
1. `estimate_spectral_index()` - Log-linear fit to frequency-resolved flux
2. `estimate_pulse_width()` - Gaussian fit to dedispersed profile
3. `estimate_scattering_from_tail()` - Exponential fit to trailing edge
4. `estimate_scattering_frequency_scaling()` - α from multi-band widths
5. `data_driven_initial_guess()` - Full parameter estimation

Usage
-----
```python
from burstfit_init import data_driven_initial_guess

init_params = data_driven_initial_guess(
    data=waterfall,
    freq=frequencies,
    time=time_axis,
    dm=500.0,
)
print(init_params)
# FRBParams(c0=1234.5, t0=12.3, gamma=-1.8, zeta=0.15, tau_1ghz=0.23, ...)
```
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.special import erfc

from .burstfit import FRBParams

log = logging.getLogger(__name__)

__all__ = [
    "data_driven_initial_guess",
    "estimate_spectral_index",
    "estimate_spectral_index",
    "estimate_pulse_width",
    "estimate_profile_emg",
    "estimate_scattering_from_tail",
    "estimate_scattering_frequency_scaling",
    "estimate_params_from_subbands",
    "InitialGuessResult",
]


@dataclass
class InitialGuessResult:
    """Container for initial guess with diagnostics."""
    
    params: FRBParams
    diagnostics: Dict[str, Any]
    
    def __repr__(self) -> str:
        p = self.params
        return (
            f"InitialGuessResult(\n"
            f"  c0={p.c0:.2f}, t0={p.t0:.3f} ms\n"
            f"  gamma={p.gamma:.2f} (spectral index)\n"
            f"  zeta={p.zeta:.3f} ms (intrinsic width)\n"
            f"  tau_1ghz={p.tau_1ghz:.3f} ms (scattering @ 1 GHz)\n"
            f"  alpha={p.alpha:.2f} (scattering scaling)\n"
            f")"
        )


def _gaussian(t: NDArray, amp: float, mu: float, sigma: float, offset: float) -> NDArray:
    """Gaussian function for profile fitting."""
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2) + offset


def _exponential_tail(t: NDArray, amp: float, tau: float, offset: float) -> NDArray:
    """Exponential decay for scattering tail fitting."""
    return amp * np.exp(-t / tau) + offset


def _scattered_gaussian(
    t: NDArray, amp: float, mu: float, sigma: float, tau: float, offset: float
) -> NDArray:
    """Convolution of Gaussian with exponential scattering kernel.
    
    This is the analytic solution for scattered Gaussian pulse.
    """
    if tau < 1e-6:
        return _gaussian(t, amp, mu, sigma, offset)
    
    # Analytic convolution: Gaussian * Exponential
    # Result is proportional to exp((t-mu)/tau) * erfc((t-mu)/(sqrt(2)*sigma) + sigma/(sqrt(2)*tau))
    
    arg1 = (t - mu) / tau + (sigma ** 2) / (2 * tau ** 2)
    arg2 = (t - mu) / (np.sqrt(2) * sigma) + sigma / (np.sqrt(2) * tau)
    
    # Safe computation
    with np.errstate(over='ignore', invalid='ignore'):
        result = amp * 0.5 * np.exp(arg1) * erfc(arg2)
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result + offset


def estimate_spectral_index(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    burst_lims: Optional[Tuple[int, int]] = None,
    min_flux_threshold: float = 0.1,
) -> Tuple[float, float]:
    """Estimate spectral index γ from frequency-resolved flux.
    
    Fits log(S) = γ * log(ν) + const to the spectrum.
    
    Parameters
    ----------
    data : array (nfreq, ntime)
        Dynamic spectrum
    freq : array (nfreq,)
        Frequencies in GHz
    burst_lims : (start, end), optional
        Time indices for burst window. If None, uses full data.
    min_flux_threshold : float
        Minimum fraction of max flux to include in fit
        
    Returns
    -------
    gamma : float
        Spectral index (typically negative, -1 to -3)
    gamma_err : float
        Uncertainty on gamma
    """
    if burst_lims is not None:
        spectrum = np.nansum(data[:, burst_lims[0]:burst_lims[1]], axis=1)
    else:
        spectrum = np.nansum(data, axis=1)
    
    # Normalize to reference frequency
    ref_freq = np.median(freq)
    
    # Filter to positive flux above threshold
    flux_threshold = min_flux_threshold * np.nanmax(spectrum)
    mask = (spectrum > flux_threshold) & np.isfinite(spectrum) & (freq > 0)
    
    if mask.sum() < 3:
        log.warning("Not enough valid channels for spectral fit, using default γ=-1.6")
        return -1.6, 0.5
    
    # Log-log fit
    log_freq = np.log(freq[mask])
    log_flux = np.log(spectrum[mask])
    
    try:
        # Weighted fit (higher flux = higher weight)
        weights = spectrum[mask] / np.max(spectrum[mask])
        coeffs, cov = np.polyfit(log_freq, log_flux, 1, w=weights, cov=True)
        gamma = coeffs[0]
        gamma_err = np.sqrt(cov[0, 0])
        
        # Sanity check: typical FRB γ is between -5 and +2
        if not (-5 < gamma < 2):
            log.warning(f"Unusual spectral index γ={gamma:.2f}, clipping to [-5, 2]")
            gamma = np.clip(gamma, -5, 2)
        
        return float(gamma), float(gamma_err)
        
    except Exception as e:
        log.warning(f"Spectral index fit failed: {e}. Using default γ=-1.6")
        return -1.6, 0.5


def estimate_pulse_width(
    data: NDArray[np.floating],
    time: NDArray[np.floating],
    burst_lims: Optional[Tuple[int, int]] = None,
    smooth_bins: int = 3,
) -> Tuple[float, float, float]:
    """Estimate pulse width and peak time from profile.
    
    Fits a Gaussian to the frequency-integrated profile.
    
    Parameters
    ----------
    data : array (nfreq, ntime)
        Dynamic spectrum
    time : array (ntime,)
        Time axis in ms
    burst_lims : (start, end), optional
        Time indices to search within
    smooth_bins : int
        Smoothing width in bins
        
    Returns
    -------
    t0 : float
        Peak time (ms)
    width : float
        FWHM width (ms)
    width_err : float
        Uncertainty on width
    """
    profile = np.nansum(data, axis=0)
    
    if burst_lims is not None:
        t_slice = slice(burst_lims[0], burst_lims[1])
        profile_window = profile[t_slice]
        time_window = time[t_slice]
    else:
        profile_window = profile
        time_window = time
    
    # Smooth to reduce noise
    if smooth_bins > 1:
        profile_smooth = gaussian_filter1d(profile_window, smooth_bins)
    else:
        profile_smooth = profile_window
    
    # Find peak
    peak_idx = np.nanargmax(profile_smooth)
    t0 = time_window[peak_idx]
    
    # Remove baseline
    baseline = np.nanpercentile(profile_window, 10)
    profile_sub = profile_window - baseline
    
    # Initial width estimate from second moment
    weights = np.maximum(profile_sub, 0)
    weights /= np.sum(weights) + 1e-30
    
    t_mean = np.sum(time_window * weights)
    t_var = np.sum((time_window - t_mean) ** 2 * weights)
    width_init = 2.355 * np.sqrt(max(t_var, 1e-6))  # FWHM = 2.355 * sigma
    
    # Try Gaussian fit for refinement
    try:
        amp_init = np.max(profile_sub)
        p0 = [amp_init, t0, width_init / 2.355, baseline]
        
        # Bounds
        bounds = (
            [0, time_window[0], 0.001, -np.inf],
            [10 * amp_init, time_window[-1], time_window[-1] - time_window[0], np.inf]
        )
        
        popt, pcov = curve_fit(
            _gaussian, time_window, profile_window,
            p0=p0, bounds=bounds, maxfev=1000
        )
        
        t0 = popt[1]
        sigma = abs(popt[2])
        width = 2.355 * sigma  # Convert sigma to FWHM
        width_err = 2.355 * np.sqrt(pcov[2, 2]) if pcov[2, 2] > 0 else width * 0.1
        
    except Exception as e:
        log.debug(f"Gaussian fit failed: {e}. Using moment estimate.")
        width = width_init
        width_err = width * 0.2
    
    return float(t0), float(width), float(width_err)


def estimate_profile_emg(
    data: NDArray[np.floating],
    time: NDArray[np.floating],
    freq: Optional[NDArray[np.floating]] = None,
    burst_lims: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float]:
    """Estimate pulse parameters using 1D EMG fit on low-frequency band.

    Fits a Pulse-Broadened Gaussian (EMG) to the profile. 
    If freq is provided, uses lowest 25% of band where scattering is strongest.

    Returns
    -------
    t0 : float
        Peak time (ms)
    zeta : float
        Intrinsic width (sigma, ms)
    tau : float
        Scattering timescale (effective at band center/low, ms)
    """
    # 1. Get Gaussian guess
    t0_init, width_init, _ = estimate_pulse_width(data, time, burst_lims)
    sigma_init = width_init / 2.355

    # Select low-frequency band if freq provided
    if freq is not None:
        freq_lo = np.percentile(freq, 0)
        freq_hi = np.percentile(freq, 25)
        freq_mask = (freq >= freq_lo) & (freq <= freq_hi)
        if freq_mask.sum() < 3:
             freq_mask = np.ones(len(freq), dtype=bool)
        profile = np.nansum(data[freq_mask, :], axis=0)
    else:
        profile = np.nansum(data, axis=0)

    if burst_lims:
        sl = slice(burst_lims[0], burst_lims[1])
        t_win = time[sl]
        p_win = profile[sl]

    else:
        t_win = time
        p_win = profile

    # Sub baseline
    base = np.nanpercentile(p_win, 10)
    p_sub = p_win - base
    amp_init = np.max(p_sub)

    # Strategy: Try two initial guesses.
    # 1. Balanced: sigma ~ width, tau ~ small (or balanced)
    # 2. Scattering Dominant: sigma ~ small, tau ~ width
    
    # Guess 1: Balanced (Original)
    tau_init_1 = width_init * 0.4
    t0_guess_1 = t0_init - tau_init_1
    sigma_init_1 = width_init / 2.355
    p0_1 = [amp_init, t0_guess_1, sigma_init_1, tau_init_1, base]
    
    # Guess 2: Scattering Dominant
    tau_init_2 = width_init * 0.9
    t0_guess_2 = t0_init - tau_init_2
    sigma_init_2 = width_init * 0.1 # Very narrow intrinsic
    p0_2 = [amp_init, t0_guess_2, sigma_init_2, tau_init_2, base]

    # Bounds: amp>0, mu in window, sigma>0, tau>=0
    bounds = (
        [0, t_win[0], 0.001, 0, -np.inf],
        [np.inf, t_win[-1], (t_win[-1]-t_win[0]), (t_win[-1]-t_win[0]), np.inf]
    )

    best_popt = None
    best_chi2 = np.inf
    
    for p0 in [p0_1, p0_2]:
        try:
            popt, _ = curve_fit(
                _scattered_gaussian, t_win, p_win, p0=p0, bounds=bounds, maxfev=2000
            )
            # Calc chi2
            model = _scattered_gaussian(t_win, *popt)
            chi2 = np.sum((p_win - model)**2)
            
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_popt = popt
        except Exception:
            continue

    if best_popt is not None:
        t0_fit = best_popt[1]
        zeta_fit = best_popt[2]
        tau_fit = best_popt[3]
        return float(t0_fit), float(zeta_fit), float(tau_fit)
    else:
        log.warning("EMG fit failed for all guesses. Falling back.")
        return t0_init, sigma_init, 0.05


def estimate_params_from_subbands(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    burst_lims: Optional[Tuple[int, int]] = None,
    n_bands: int = 4,
) -> Tuple[float, float, float, float]:
    """Estimate parameters by splitting band into n_bands and fitting.

    Returns
    -------
    t0 : float (ms)
    zeta : float (ms) 
    tau_1ghz : float (ms)
    alpha : float (scattering index)
    """
    
    # Divide band into N chunks
    f_min, f_max = freq.min(), freq.max()
    edges = np.linspace(f_min, f_max, n_bands + 1)
    
    subband_results = []
    
    for i in range(n_bands):
        f_lo, f_hi = edges[i], edges[i+1]
        f_center = (f_lo + f_hi) / 2
        
        # Fit EMG to this subband
        # Note: estimate_profile_emg handles the masking internally if freq provided
        # But we want to call it with a specific mask or freq subset?
        # estimate_profile_emg currently takes 'freq' and selects lowest 25%.
        # We should modify it or just manually subset here.
        # Manual subsetting is safer.
        
        mask = (freq >= f_lo) & (freq <= f_hi)
        if mask.sum() < 2:
            continue
            
        data_sub = data[mask, :]
        
        try:
            # First get width/peak for this band
            t0_sub, w_sub, _ = estimate_pulse_width(data_sub, time, burst_lims=burst_lims)
            
            # Use tail estimator for tau (more robust than EMG for faint tails)
            # data_sub has shape (freqs, time). mask is already applied?
            # estimate_scattering_from_tail expects (data, time, freq, t0, width)
            # We pass the full data frequency array subset? No, we need to pass matching shapes.
            # But data_sub is already subsetted.
            # We can pass freq=None/dummy? 
            # estimate_scattering_from_tail uses freq to select low band. 
            # We want to use the WHOLE subband.
            # So we pass freq_band=(min, max) covering the whole subband.
            
            # Construct a dummy freq array for the subband
            freq_sub = freq[mask]
            
            tau_s, _ = estimate_scattering_from_tail(
                data_sub, time, freq_sub, t0_sub, w_sub, freq_band=(f_lo, f_hi)
            )
            
            # Approximate zeta as width (since tau is from tail). 
            # If tau is large, zeta ~ sqrt(w^2 - tau^2).
            zeta_s = w_sub / 2.355 
            if tau_s < w_sub:
                 zeta_s = np.sqrt(max(0, (w_sub/2.355)**2 - (tau_s/2.355)**2)) # Crude approx

            subband_results.append({
                'freq': f_center,
                't0': t0_sub,
                'zeta': zeta_s,
                'tau': tau_s
            })
            log.info(f"  Subband {i} ({f_center:.3f} GHz): t0={t0_sub:.3f}, width={w_sub:.3f}, tau={tau_s:.3f}")
        except Exception:
            continue
            
    if len(subband_results) < 2:
        return 0.0, 0.0, 0.0, 4.0 # Failed
        
    # Fit Power Law to Tau: tau = A * freq^-alpha
    freqs = np.array([r['freq'] for r in subband_results])
    taus = np.array([r['tau'] for r in subband_results])
    
    # Filter out bad fits (tau ~ 0 or unreasonable)
    valid = (taus > 0.001) & (taus < 100)
    if valid.sum() < 2:
         # Fallback
         best_band = subband_results[0] # Usually lowest freq has strongest signal/tau
         return best_band['t0'], best_band['zeta'], 0.0, 4.0

    try:
        log_f = np.log(freqs[valid])
        log_tau = np.log(taus[valid])
        
        coeffs = np.polyfit(log_f, log_tau, 1)
        alpha_fit = -coeffs[0]
        
        # tau(1GHz) -> log(tau) = -alpha * log(1) + C => C = log(tau_1ghz)
        # But our fit is log(tau) = -alpha * log(f) + C
        # So exp(C) is tau at 1 GHz (if f is in GHz).
        tau_1ghz_fit = np.exp(coeffs[1])
        
        # Clip alpha to reasonable physics
        alpha_final = np.clip(alpha_fit, 2.0, 6.0)
        
        # Re-estimate tau_1ghz if we clipped alpha?
        # No, just keep the fitted value or re-project from mean.
        # Let's re-project from the most reliable band (lowest freq)
        idx_low = np.argmin(freqs[valid])
        f_ref = freqs[valid][idx_low]
        tau_ref = taus[valid][idx_low]
        tau_1ghz_final = tau_ref * (f_ref / 1.0)**alpha_final
        
    except Exception:
        alpha_final = 4.0
        tau_1ghz_final = 1.0 # Placeholder
        
    # Intrinsic parameters (zeta, t0) usually best from highest frequency (least scattering)
    # But if high freq has low SNR, middle is better.
    # Let's take the mean of the top 50% frequency bands
    high_freq_indices = np.argsort(freqs)[-max(1, len(freqs)//2):]
    
    avg_t0 = np.mean([subband_results[i]['t0'] for i in high_freq_indices])
    avg_zeta = np.mean([subband_results[i]['zeta'] for i in high_freq_indices])

    return avg_t0, avg_zeta, tau_1ghz_final, alpha_final


def estimate_scattering_from_tail(
    data: NDArray[np.floating],
    time: NDArray[np.floating],
    freq: NDArray[np.floating],
    t0: float,
    width: float,
    freq_band: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Estimate scattering timescale from exponential tail.
    
    Fits an exponential decay to the trailing edge of the pulse.
    
    Parameters
    ----------
    data : array (nfreq, ntime)
        Dynamic spectrum
    time : array (ntime,)
        Time axis in ms
    freq : array (nfreq,)
        Frequencies in GHz
    t0 : float
        Peak time (ms)
    width : float
        Pulse FWHM (ms)
    freq_band : (lo, hi), optional
        Frequency range to use. If None, uses lowest 25% of band.
        
    Returns
    -------
    tau : float
        Scattering timescale at measured frequency (ms)
    tau_err : float
        Uncertainty on tau
    """
    # Select low-frequency band where scattering is strongest
    if freq_band is None:
        freq_lo = np.percentile(freq, 0)
        freq_hi = np.percentile(freq, 25)
    else:
        freq_lo, freq_hi = freq_band
    
    freq_mask = (freq >= freq_lo) & (freq <= freq_hi)
    if freq_mask.sum() < 3:
        freq_mask = np.ones(len(freq), dtype=bool)
    
    # Average profile in low-frequency band
    profile_lo = np.nanmean(data[freq_mask, :], axis=0)
    
    # Define tail region: start 1 FWHM after peak, extend 5 FWHM
    tail_start = t0 + width
    tail_end = t0 + 6 * width
    
    tail_mask = (time >= tail_start) & (time <= tail_end) & np.isfinite(profile_lo)
    
    if tail_mask.sum() < 5:
        log.warning("Not enough tail samples for scattering fit")
        return 0.1, 0.1
    
    t_tail = time[tail_mask] - t0
    profile_tail = profile_lo[tail_mask]
    
    # Baseline
    baseline = np.nanpercentile(profile_tail, 10)
    profile_tail_sub = profile_tail - baseline
    
    # Initial tau guess from e-folding
    above_half = profile_tail_sub > 0.5 * np.max(profile_tail_sub)
    if above_half.sum() > 0:
        tau_init = t_tail[above_half][-1] - t_tail[above_half][0]
    else:
        tau_init = width
    
    try:
        p0 = [np.max(profile_tail_sub), max(tau_init, 0.01), 0]
        bounds = ([0, 0.001, -np.inf], [np.inf, 20 * width, np.inf])
        
        popt, pcov = curve_fit(
            _exponential_tail, t_tail, profile_tail_sub,
            p0=p0, bounds=bounds, maxfev=500
        )
        
        tau = abs(popt[1])
        tau_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else tau * 0.2
        
    except Exception as e:
        log.debug(f"Exponential tail fit failed: {e}. Using half-width estimate.")
        tau = max(width * 0.5, 0.1)
        tau_err = tau * 0.5
    
    return float(tau), float(tau_err)


def estimate_scattering_frequency_scaling(
    data: NDArray[np.floating],
    time: NDArray[np.floating],
    freq: NDArray[np.floating],
    t0: float,
    n_bands: int = 4,
) -> Tuple[float, float, float]:
    """Estimate scattering spectral index α from multi-band widths.
    
    Measures pulse width in frequency sub-bands and fits τ ∝ ν^(-α).
    
    Parameters
    ----------
    data : array (nfreq, ntime)
        Dynamic spectrum
    time : array (ntime,)
        Time axis in ms
    freq : array (nfreq,)
        Frequencies in GHz
    t0 : float
        Peak time (ms)
    n_bands : int
        Number of frequency sub-bands
        
    Returns
    -------
    alpha : float
        Scattering spectral index
    alpha_err : float
        Uncertainty on alpha
    tau_1ghz : float
        Scattering timescale at 1 GHz (ms)
    """
    freq_edges = np.linspace(freq.min(), freq.max(), n_bands + 1)
    
    band_centers = []
    band_widths = []
    band_width_errs = []
    
    for i in range(n_bands):
        f_lo, f_hi = freq_edges[i], freq_edges[i + 1]
        mask = (freq >= f_lo) & (freq < f_hi)
        
        if mask.sum() < 2:
            continue
        
        # Profile in this band
        profile_band = np.nansum(data[mask, :], axis=0)
        
        # Quick width estimate
        try:
            _, width, width_err = estimate_pulse_width(
                data[mask, :], time, smooth_bins=2
            )
            
            band_centers.append((f_lo + f_hi) / 2)
            band_widths.append(width)
            band_width_errs.append(width_err)
            
        except Exception:
            continue
    
    if len(band_centers) < 3:
        log.warning("Not enough bands for α estimation, using default α=4.0")
        return 4.0, 0.5, 0.1
    
    band_centers = np.array(band_centers)
    band_widths = np.array(band_widths)
    band_width_errs = np.array(band_width_errs)
    
    # Fit: log(width) = -α * log(freq) + const
    # (scattering dominates at low freq, so width ≈ τ ∝ ν^-α)
    log_freq = np.log(band_centers)
    log_width = np.log(np.maximum(band_widths, 1e-6))
    
    try:
        # Weighted fit
        weights = 1.0 / (band_width_errs / band_widths + 0.1) ** 2
        coeffs, cov = np.polyfit(log_freq, log_width, 1, w=weights, cov=True)
        
        alpha = -coeffs[0]  # Note the negative sign
        alpha_err = np.sqrt(cov[0, 0])
        
        # Derive tau at 1 GHz
        tau_1ghz = np.exp(coeffs[1])  # intercept when log(freq)=0
        
        # Sanity check
        if not (2.0 < alpha < 6.0):
            log.warning(f"Unusual α={alpha:.2f}, clipping to [2, 6]")
            alpha = np.clip(alpha, 2.0, 6.0)
        
        return float(alpha), float(alpha_err), float(tau_1ghz)
        
    except Exception as e:
        log.warning(f"α fit failed: {e}. Using default α=4.0")
        return 4.0, 0.5, 0.1


def data_driven_initial_guess(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm: float = 0.0,
    burst_lims: Optional[Tuple[int, int]] = None,
    min_scattering: float = 0.01,
    ne2001_fallback: bool = True,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    verbose: bool = True,
) -> InitialGuessResult:
    """Generate data-driven initial parameter estimates.
    
    Extracts all FRBParams directly from data properties instead of
    using hardcoded values.
    
    Parameters
    ----------
    data : array (nfreq, ntime)
        Dynamic spectrum (frequency × time)
    freq : array (nfreq,)
        Frequencies in GHz (or MHz if > 100)
    time : array (ntime,)
        Time axis in ms
    dm : float
        Dispersion measure (pc/cm³)
    burst_lims : (start, end), optional
        Time indices containing the burst. If None, auto-detected.
    min_scattering : float
        Minimum scattering timescale (ms)
    ne2001_fallback : bool
        If data-driven scattering estimate fails, use NE2001 prediction
    ra_deg, dec_deg : float, optional
        Sky position for NE2001 fallback
    verbose : bool
        Print progress
        
    Returns
    -------
    InitialGuessResult
        Contains FRBParams and diagnostic information
        
    Examples
    --------
    >>> result = data_driven_initial_guess(waterfall, freq, time, dm=500)
    >>> print(result.params)
    >>> fitter = FRBFitter(model, priors, initial_guess=result.params)
    """
    diagnostics = {}
    
    # Ensure frequency is in GHz
    if np.median(freq) > 100:
        freq = freq / 1000.0  # Convert MHz to GHz
        if verbose:
            log.info("Converted frequencies from MHz to GHz")
    
    # Auto-detect burst if not specified
    if burst_lims is None:
        profile = np.nansum(data, axis=0)
        profile_smooth = gaussian_filter1d(profile, 5)
        noise_level = np.nanpercentile(profile_smooth, 25)
        threshold = noise_level + 3 * np.nanstd(profile_smooth[:len(profile)//4])
        above_thresh = profile_smooth > threshold
        
        if above_thresh.sum() > 0:
            indices = np.where(above_thresh)[0]
            margin = max(10, int(0.1 * len(profile)))
            burst_lims = (max(0, indices[0] - margin), 
                         min(len(time), indices[-1] + margin))
        else:
            burst_lims = (0, len(time))
        
        diagnostics['auto_burst_lims'] = burst_lims
    
    # 1. Amplitude (c0)
    # 1. Amplitude (c0)
    # FRBModel treats c0 as per-channel fluence (approx).
    # nansum(data) is total flux. We must divide by n_freq.
    n_freq = data.shape[0]
    if burst_lims:
        c0_total = np.nansum(data[:, burst_lims[0]:burst_lims[1]])
    else:
        c0_total = np.nansum(data)
        
    c0 = c0_total / max(n_freq, 1)
    
    diagnostics['c0_method'] = 'integrated_flux_per_channel'
    
    # 2. Peak time and width
    t0, width, width_err = estimate_pulse_width(data, time, burst_lims)
    diagnostics['t0_method'] = 'gaussian_fit'
    diagnostics['observed_width'] = width
    diagnostics['width_err'] = width_err
    
    if verbose:
        log.info(f"Peak time: t0 = {t0:.3f} ms")
        log.info(f"Observed width: {width:.3f} ± {width_err:.3f} ms")
    
    # 3. Spectral index
    gamma, gamma_err = estimate_spectral_index(data, freq, burst_lims)
    diagnostics['gamma'] = gamma
    diagnostics['gamma_err'] = gamma_err
    
    if verbose:
        log.info(f"Spectral index: γ = {gamma:.2f} ± {gamma_err:.2f}")

    # 4. Scattering from tail (Legacy method, moved up for fallback availability)
    tau_meas, tau_err = estimate_scattering_from_tail(
        data, time, freq, t0, width
    )
    diagnostics['tau_measured'] = tau_meas
    diagnostics['tau_err'] = tau_err
    
    # Measure at band center, scale to 1 GHz
    freq_center = np.median(freq)

    # NEW: Subband-based Initialization
    # This is superior to single-band EMG or tail fitting.
    t0_sb, zeta_sb, tau_sb, alpha_sb = estimate_params_from_subbands(data, freq, time, burst_lims)
    
    # Check if subband fit was reliable
    # If alpha is hitting bounds (2.0 or 6.0), the spectral fit failed (e.g. inverted spectrum).
    # In that case, tau_sb is likely wrong (extrapolated from bad index).
    subband_reliable = (tau_sb > min_scattering) and (2.01 < alpha_sb < 5.99)
    
    if subband_reliable:
        diagnostics['method'] = 'subband_tail_fit'
        t0 = t0_sb
        zeta = zeta_sb
        tau_1ghz = tau_sb
        alpha = alpha_sb
        
        if verbose:
            log.info(f"Subband Init -> t0={t0:.3f}, ζ={zeta:.3f}, τ(1GHz)={tau_1ghz:.3f}, α={alpha:.2f}")
    else:
        # Fallback to single-band methods
        log.info(f"Subband init unreliable (α={alpha_sb:.2f}, τ={tau_sb:.3f}), falling back.")
        
        # Prefer the legacy tail fit (tau_meas) if available and reasonable
        # Prefer the legacy tail fit (tau_meas) if available and reasonable
        if tau_meas > min_scattering:
             tau_1ghz = tau_meas * (freq_center / 1.0) ** 4.0 # Assume alpha=4
             diagnostics['method'] = 'legacy_tail_fit'
             
             # Estimate zeta from width and tau
             # width roughly sqrt(zeta_fwhm^2 + tau^2) + ...
             # Just use width/2.35 as upper bound or crude estimate
             zeta = max(0.01, (width / 2.355))
        else:
             # Fallback to single-band EMG
             t0_emg, zeta_emg, tau_emg_obs = estimate_profile_emg(data, time, freq, burst_lims)
             
             # Project tau observed (at low band center) to 1 GHz
             freq_lo_mean = np.mean(freq[freq <= np.percentile(freq, 25)])
             tau_1ghz = tau_emg_obs * (freq_lo_mean / 1.0) ** 4.0
             
             t0 = t0_emg
             zeta = zeta_emg
             diagnostics['method'] = 'single_band_emg'
        
        # Default alpha
        alpha = 4.0

    # Diagnostics storage
    diagnostics['alpha'] = alpha
    diagnostics['tau_1ghz_estimate'] = tau_1ghz
    diagnostics['zeta_estimate'] = zeta
    diagnostics['t0_estimate'] = t0
    
    if verbose:
        log.info(f"Scattering τ(1GHz) = {tau_1ghz:.3f} ms, α={alpha:.2f}")

    # Legacy tail fit for comparison/logging only
    try:
         estimate_scattering_from_tail(data, time, freq, t0, width)
    except:
         pass
    
    # 6. NE2001 fallback/comparison
    if ne2001_fallback and ra_deg is not None and dec_deg is not None:
        try:
            from .priors_physical import get_ne2001_scattering
            tau_ne2001, _ = get_ne2001_scattering(ra_deg, dec_deg, dm, freq_mhz=1000)
            diagnostics['tau_ne2001'] = tau_ne2001
            
            # If data estimate is unreasonable, use NE2001
            if tau_1ghz < 0.001 or tau_1ghz > 100:
                if verbose:
                    log.info(f"Using NE2001 fallback: τ = {tau_ne2001:.4f} ms")
                tau_1ghz = tau_ne2001
                
        except Exception as e:
            log.debug(f"NE2001 fallback failed: {e}")
    
    # 7. Intrinsic width (Use EMG result directly)
    zeta = max(zeta, 0.01)
    # Check if zeta is reliable? If EMG tau >> zeta, zeta might be very small.
    # Keep it.
    diagnostics['zeta_estimate'] = zeta
    
    if verbose:
        log.info(f"Intrinsic width: ζ = {zeta:.3f} ms")
    
    # Build FRBParams
    params = FRBParams(
        c0=float(c0),
        t0=float(t0),
        gamma=float(gamma),
        zeta=float(zeta),
        tau_1ghz=float(tau_1ghz),
        alpha=float(alpha),
        delta_dm=0.0,  # No residual DM by default
    )
    
    if verbose:
        log.info(f"\n=== Data-Driven Initial Guess ===")
        log.info(f"  c0      = {params.c0:.2f}")
        log.info(f"  t0      = {params.t0:.3f} ms")
        log.info(f"  γ       = {params.gamma:.2f}")
        log.info(f"  ζ       = {params.zeta:.3f} ms")
        log.info(f"  τ(1GHz) = {params.tau_1ghz:.3f} ms")
        log.info(f"  α       = {params.alpha:.2f}")
    
    return InitialGuessResult(params=params, diagnostics=diagnostics)


# Convenience function for pipeline integration
def quick_initial_guess(
    data: NDArray[np.floating],
    freq: NDArray[np.floating],
    time: NDArray[np.floating],
    dm: float = 0.0,
) -> FRBParams:
    """Simple wrapper returning just FRBParams.
    
    For backward compatibility and quick usage.
    """
    result = data_driven_initial_guess(data, freq, time, dm, verbose=False)
    return result.params
