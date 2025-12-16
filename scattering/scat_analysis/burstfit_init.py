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

from .burstfit import FRBParams

log = logging.getLogger(__name__)

__all__ = [
    "data_driven_initial_guess",
    "estimate_spectral_index",
    "estimate_pulse_width",
    "estimate_scattering_from_tail",
    "estimate_scattering_frequency_scaling",
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
    from scipy.special import erfc
    
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
    if burst_lims:
        c0 = np.nansum(data[:, burst_lims[0]:burst_lims[1]])
    else:
        c0 = np.nansum(data)
    diagnostics['c0_method'] = 'integrated_flux'
    
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
    
    # 4. Scattering from tail
    tau_meas, tau_err = estimate_scattering_from_tail(
        data, time, freq, t0, width
    )
    diagnostics['tau_measured'] = tau_meas
    diagnostics['tau_err'] = tau_err
    
    # Measure at band center, scale to 1 GHz
    freq_center = np.median(freq)
    alpha_init = 4.0  # Initial guess for scaling
    
    # 5. Scattering frequency scaling
    alpha, alpha_err, tau_1ghz_from_scaling = estimate_scattering_frequency_scaling(
        data, time, freq, t0
    )
    diagnostics['alpha'] = alpha
    diagnostics['alpha_err'] = alpha_err
    
    if verbose:
        log.info(f"Scattering α = {alpha:.2f} ± {alpha_err:.2f}")
    
    # Scale measured tau to 1 GHz
    tau_1ghz = tau_meas * (freq_center / 1.0) ** alpha
    tau_1ghz = max(tau_1ghz, min_scattering)
    
    # Use average of both methods
    if tau_1ghz_from_scaling > 0:
        tau_1ghz = np.sqrt(tau_1ghz * tau_1ghz_from_scaling)  # Geometric mean
    
    diagnostics['tau_1ghz_estimate'] = tau_1ghz
    
    if verbose:
        log.info(f"Scattering τ(1GHz) = {tau_1ghz:.3f} ms")
    
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
    
    # 7. Intrinsic width (deconvolve scattering)
    # Observed width² ≈ intrinsic² + scattering²
    tau_at_center = tau_1ghz * (freq_center / 1.0) ** (-alpha)
    width_squared = max(width ** 2 - tau_at_center ** 2, 0.01 ** 2)
    zeta = np.sqrt(width_squared)
    
    # Minimum intrinsic width
    zeta = max(zeta, 0.01)
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
