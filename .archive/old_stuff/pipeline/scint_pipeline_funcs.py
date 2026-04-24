# scint_pipeline_funcs.py
# Refactored and generalized scintillation analysis functions

import math
import numpy as np
import scipy.constants as cons
from scipy.fft import fft, fftshift, fftfreq, rfft, irfft
from scipy.signal import correlate, correlation_lags
from scipy.optimize import curve_fit
from lmfit import minimize, Parameters, Model
from astropy import units as u
from astropy.stats import weighted_mean
from typing import Tuple, Optional, Callable, Dict, Any

# --- Constants ---
C_MPS = cons.c  # Speed of light in m/s
PARSEC_M = cons.parsec # Parsec in meters

# --- Data Handling & Preprocessing ---

def scrunch(data: np.ndarray, t_scrunch: int = 1, f_scrunch: int = 1) -> np.ndarray:
    """
    Scrunch data in time and frequency by averaging blocks.

    Args:
        data: Input data array (time, freq).
        t_scrunch: Factor to scrunch in time.
        f_scrunch: Factor to scrunch in frequency.

    Returns:
        Scrunched data array.

    Raises:
        ValueError: If scrunch factors do not evenly divide dimensions.
    """
    nt, nf = data.shape
    if nt % t_scrunch != 0:
        raise ValueError(f"Time dimension ({nt}) not divisible by t_scrunch ({t_scrunch})")
    if nf % f_scrunch != 0:
        raise ValueError(f"Frequency dimension ({nf}) not divisible by f_scrunch ({f_scrunch})")

    scrunched_data = data.reshape(nt // t_scrunch, t_scrunch, nf // f_scrunch, f_scrunch)
    scrunched_data = scrunched_data.mean(axis=(1, 3)) # Average over t_scrunch and f_scrunch axes
    return scrunched_data

def upchannelize(
    intensity: np.ndarray,
    fftsize: int = 32,
    downfreq: int = 2,
    downtime: int = 1,
    ) -> Tuple[np.ndarray, int]:
    """
    Upchannelize a dynamic spectrum (Intensity only).

    Adapted from the original `upchannel` function, assuming input is
    intensity (Stokes I) with shape (frequency, time).
    Performs an FFT along the time axis for blocks of `fftsize`,
    optionally averages in time and frequency post-FFT.

    Args:
        intensity: Dynamic spectrum intensity array (frequency, time).
        fftsize: FFT length for the time axis transform.
        downfreq: Downsampling factor in frequency after upchannelization.
        downtime: Downsampling factor in time after upchannelization.

    Returns:
        Tuple containing:
            - upchann_spec (np.ndarray): Upchannelized spectrum (new_freq, new_time).
                                        Frequency order corresponds to FFT output.
            - upchan_factor (int): The effective upchannelization factor (fftsize // downfreq).

    Raises:
        ValueError: If dimensions are incompatible with parameters.
    """
    if intensity.ndim != 2:
        raise ValueError("Input intensity array must be 2D (frequency, time).")

    nchan, nsamp = intensity.shape

    if nsamp % (fftsize * downtime) != 0:
        raise ValueError(f"Time samples ({nsamp}) not divisible by fftsize*downtime ({fftsize*downtime})")
    if fftsize % downfreq != 0:
        raise ValueError(f"fftsize ({fftsize}) must be divisible by downfreq ({downfreq})")

    # Upchannelization factor
    upchan_factor = fftsize // downfreq
    nchan_up = nchan * upchan_factor

    # Number of time blocks
    nblock = nsamp // (fftsize * downtime)

    # Reshape for block processing: (nchan, nblock, downtime, fftsize)
    wfall_block = intensity.reshape(nchan, nblock, downtime, fftsize)

    # Initialize output array
    # The FFT output will have fftsize freq bins for each original channel
    # Shape before downsampling: (nchan, nblock, downtime, fftsize) -> FFT -> (nchan, nblock, downtime, fftsize)
    # Shape after downsampling: (nchan * upchan_factor, nblock) = (nchan_up, nblock)
    upchann_spec = np.zeros((nchan_up, nblock), dtype=np.float64) # Store power |FFT|^2

    # Process blocks
    for i in range(nblock):
        # FFT along the last axis (fftsize)
        # Using rfft for real-valued input
        # The output has fftsize // 2 + 1 complex values
        # We take the magnitude squared for power
        # Note: Original code used full FFT, implying complex input?
        # Assuming intensity input, rfft is more appropriate.
        # If complex voltage data is needed later, this function needs revision.
        # block_fft = np.fft.fft(wfall_block[:, i, :, :], axis=-1) # Original used full FFT
        block_fft = np.fft.rfft(wfall_block[:, i, :, :], axis=-1) # Shape (nchan, downtime, fftsize//2+1)
        block_power = np.abs(block_fft)**2

        # Average over downtime samples
        if downtime > 1:
            block_power = np.mean(block_power, axis=1) # Shape (nchan, fftsize//2+1)
        else:
             block_power = block_power[:, 0, :] # Remove the downtime dimension

        # Downsample (average) in frequency (within the FFT output)
        # The rfft output bins correspond to frequencies 0, df, 2*df, ..., (N/2)*df
        # Need to reshape and average to match `downfreq` parameter logic.
        # This part is tricky to interpret from the original code without knowing
        # the exact desired output frequency mapping. Assuming simple averaging
        # over `downfreq` bins of the RFFT output *magnitude squared*.
        # The original code's `upchan = fftsize // downfreq` suggests splitting the
        # `fftsize` output bins into `downfreq` groups and averaging, resulting in
        # `upchan` effective channels per original channel.

        # Let's try to match the original output shape (nchan * upchan_factor, nblock)
        # rfft output length is fftsize//2 + 1. This doesn't easily map to upchan_factor.
        # Reverting to full FFT as in original code to maintain structure,
        # but applying it to intensity data which is unusual.
        # User should be warned if using this with intensity.

        block_fft_full = np.fft.fft(wfall_block[:, i, :, :], axis=-1) # Shape (nchan, downtime, fftsize)
        block_power_full = np.abs(block_fft_full)**2

        if downtime > 1:
             block_power_full = np.mean(block_power_full, axis=1) # Shape (nchan, fftsize)
        else:
             block_power_full = block_power_full[:, 0, :] # Remove the downtime dimension

        # Reshape and average over `downfreq` bins
        # (nchan, fftsize) -> (nchan, upchan_factor, downfreq)
        try:
            power_reshaped = block_power_full.reshape(nchan, upchan_factor, downfreq)
        except ValueError as e:
             raise ValueError(f"Cannot reshape power array with shape {(nchan, fftsize)} "
                              f"into {(nchan, upchan_factor, downfreq)}. Check fftsize ({fftsize}), "
                              f"downfreq ({downfreq}), upchan_factor ({upchan_factor}).") from e

        # Average over the `downfreq` dimension
        power_downsampled = np.mean(power_reshaped, axis=2) # Shape (nchan, upchan_factor)

        # Assign to the output spectrum, reshaping to (nchan * upchan_factor)
        upchann_spec[:, i] = power_downsampled.ravel() # Order: [chan0_up0, chan0_up1..., chan1_up0...]

    # The output frequency axis needs careful interpretation. It's a mix of
    # original channels and the new FFT-based channels.
    # Output shape is (nchan_up, nblock) = (nchan * upchan_factor, nsamp / (fftsize * downtime))
    # Transpose to match (time, freq) convention if desired downstream:
    # upchann_spec = upchann_spec.T # (new_time, new_freq)

    return upchann_spec, upchan_factor


# --- ACF Calculation and Fitting ---

def calculate_acf_1d(
    series: np.ndarray,
    norm: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the 1D Autocorrelation Function (ACF) of a time series.

    Args:
        series: 1D numpy array.
        norm: Normalize ACF to have max value of 1.

    Returns:
        Tuple containing:
            - lags (np.ndarray): Lags for the ACF.
            - acf (np.ndarray): Calculated ACF.
    """
    n = len(series)
    valid_series = series[~np.isnan(series)] # Ignore NaNs
    if len(valid_series) < 2:
        return np.arange(-n//2 + 1, n//2 + 1), np.full(n, np.nan) # Handle case with too few points

    # Detrend by subtracting mean
    detrended_series = valid_series - np.mean(valid_series)

    # Use scipy.signal.correlate for ACF
    # mode='full' gives lags from -(n-1) to +(n-1)
    # mode='same' gives lags centered around 0, size n
    acf = correlate(detrended_series, detrended_series, mode='same')
    lags = correlation_lags(len(detrended_series), len(detrended_series), mode='same')

    if norm:
        acf = acf / acf[n // 2] # Normalize by zero lag (center element)

    # Pad with NaNs if original series had NaNs or for consistency?
    # For now, return ACF based on valid points only. Size matches 'same' mode.

    return lags, acf


def calculate_acf_2d(
    dyn_spec: np.ndarray,
    axis: int = 1,
    norm: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the 2D Autocorrelation Function (ACF) along a specified axis.

    Averages ACFs calculated for each slice along the *other* axis.

    Args:
        dyn_spec: 2D dynamic spectrum (time, freq).
        axis: Axis along which to calculate the ACF (0 for time, 1 for freq).
        norm: Normalize individual ACFs before averaging.

    Returns:
        Tuple containing:
            - lags (np.ndarray): Lags for the ACF along the specified axis.
            - avg_acf (np.ndarray): Averaged ACF.
    """
    if dyn_spec.ndim != 2:
        raise ValueError("Input dynamic spectrum must be 2D.")

    n_slices = dyn_spec.shape[1 - axis] # Number of slices along the other axis
    acf_len = dyn_spec.shape[axis]
    all_acfs = []
    common_lags = None

    for i in range(n_slices):
        series = dyn_spec[i, :] if axis == 1 else dyn_spec[:, i]
        lags, acf = calculate_acf_1d(series, norm=norm)

        if common_lags is None:
            # Ensure lags are consistent size, centered correctly
            mid_point = len(lags)//2
            target_lags = np.arange(-acf_len // 2 + 1, acf_len // 2 +1) # Assuming centered lags needed
            # Crude alignment: Adjust if needed based on calculate_acf_1d output
            # This needs refinement if calculate_acf_1d returns variable length/centers
            aligned_acf = np.full(len(target_lags), np.nan)
            # Basic centering assumption for now
            start_idx_target = max(0, len(target_lags)//2 - len(lags)//2)
            start_idx_acf = max(0, len(lags)//2 - len(target_lags)//2)
            len_overlap = min(len(lags) - start_idx_acf, len(target_lags) - start_idx_target)
            aligned_acf[start_idx_target:start_idx_target+len_overlap] = acf[start_idx_acf:start_idx_acf+len_overlap]

            common_lags = target_lags # Use the target centered lags
            all_acfs.append(aligned_acf)
        elif len(lags) == len(common_lags): # Basic check
            all_acfs.append(acf)
        else:
            # Handle cases where slice lengths differ (e.g. due to NaNs) - needs robust alignment
            print(f"Warning: Slice {i} ACF length mismatch. Skipping.") # Simple approach for now
            # Or implement padding/interpolation

    if not all_acfs:
        return np.arange(-acf_len // 2 + 1, acf_len // 2 +1), np.full(acf_len, np.nan)

    # Average the ACFs, ignoring NaNs
    avg_acf = np.nanmean(np.array(all_acfs), axis=0)

    return common_lags, avg_acf


# --- Model Fitting Functions (using lmfit style where available) ---

# --- Lorentzian Models ---
def lorentzian(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
    """ Standard Lorentzian function. wid = FWHM/2 """
    return amp * wid**2 / ((x - cen)**2 + wid**2)

def lorentzian_with_const(x: np.ndarray, amp: float, cen: float, wid: float, c: float) -> np.ndarray:
    """ Lorentzian plus a constant offset. """
    return lorentzian(x, amp, cen, wid) + c

def fit_lorentzian_acf(
    lags: np.ndarray,
    acf: np.ndarray,
    errs: Optional[np.ndarray] = None,
    center_guess: float = 0.0,
    const_offset: bool = True
    ) -> Tuple[Optional[Parameters], Optional[Model], Optional[Any]]:
    """
    Fits a Lorentzian (optionally with constant) to the central part of an ACF.

    Args:
        lags: ACF lags (should be centered around 0).
        acf: ACF values.
        errs: Optional errors for weighted fitting.
        center_guess: Initial guess for the Lorentzian center.
        const_offset: Whether to include a constant offset in the fit.

    Returns:
        Tuple containing:
        - Best-fit parameters (lmfit Parameters object) or None if fit fails.
        - lmfit Model object or None.
        - lmfit fit result object or None.
    """
    # Select data near the peak (e.g., central 50% of lags or where acf > 0.1?)
    # Heuristic: fit where ACF is positive and relatively large
    center_idx = np.argmax(acf) # Should be near the middle for ACF
    fit_mask = acf > 0.1 * acf[center_idx]
    # Ensure mask is contiguous around the peak if needed?
    # Or just fit where mask is true.

    if not np.any(fit_mask):
        print("Warning: No suitable data points found for Lorentzian fit.")
        return None, None, None

    x_fit = lags[fit_mask]
    y_fit = acf[fit_mask]
    weights = None
    if errs is not None:
        errs_fit = errs[fit_mask]
        weights = 1.0 / errs_fit**2 # lmfit uses weights=1/sigma^2

    if const_offset:
        model = Model(lorentzian_with_const)
        params = Parameters()
        params.add('amp', value=np.max(y_fit), min=0)
        params.add('cen', value=center_guess, vary=True) # Center might not be exactly 0
        params.add('wid', value=np.std(x_fit), min=1e-9) # Guess width from data spread
        params.add('c', value=np.min(y_fit))
    else:
        model = Model(lorentzian)
        params = Parameters()
        params.add('amp', value=np.max(y_fit), min=0)
        params.add('cen', value=center_guess, vary=True)
        params.add('wid', value=np.std(x_fit), min=1e-9)

    try:
        result = model.fit(y_fit, params, x=x_fit, weights=weights)
        return result.params, model, result
    except Exception as e:
        print(f"Error during Lorentzian fit: {e}")
        return None, None, None

# --- Double Lorentzian Models (Example - can add Triple etc. if needed) ---
def double_lorentzian_with_const(x: np.ndarray,
                                  amp1: float, cen1: float, wid1: float,
                                  amp2: float, cen2: float, wid2: float,
                                  c: float) -> np.ndarray:
    """ Sum of two Lorentzians plus a constant offset. """
    return lorentzian(x, amp1, cen1, wid1) + lorentzian(x, amp2, cen2, wid2) + c

# --- Power Law Models ---
def power_law(x: np.ndarray, amp: float, index: float) -> np.ndarray:
    """ Power law function: amp * x ** index """
    # Avoid issues with log(0) or negative bases if index is non-integer
    # Assume x represents frequency, which should be positive.
    return amp * np.power(x, index)

def fit_scint_bandwidth_freq_relation(
    freqs: np.ndarray,
    scint_widths: np.ndarray,
    errs: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Parameters], Optional[Model], Optional[Any]]:
    """
    Fits Δν_d ∝ ν^α power law relationship.

    Args:
        freqs: Frequencies (e.g., central frequency of subbands).
        scint_widths: Measured scintillation bandwidths (e.g., HWHM from Lorentzian fits).
        errs: Optional errors for weighted fitting.

    Returns:
        Tuple containing:
        - Best-fit parameters (lmfit Parameters object) or None if fit fails.
        - lmfit Model object or None.
        - lmfit fit result object or None.
    """
    if len(freqs) != len(scint_widths):
        raise ValueError("Frequency and scintillation width arrays must have the same length.")

    weights = None
    if errs is not None:
        if len(errs) != len(freqs):
            raise ValueError("Error array must have the same length as frequency array.")
        weights = 1.0 / errs**2

    # Ensure positivity for power law fit
    valid_mask = (freqs > 0) & (scint_widths > 0) & (~np.isnan(freqs)) & (~np.isnan(scint_widths))
    if errs is not None:
        valid_mask &= (~np.isnan(errs)) & (errs > 0)

    if np.sum(valid_mask) < 2:
        print("Warning: Fewer than 2 valid points for power law fit.")
        return None, None, None

    x_fit = freqs[valid_mask]
    y_fit = scint_widths[valid_mask]
    weights_fit = weights[valid_mask] if weights is not None else None

    model = Model(power_law)
    params = Parameters()
    # Initial guess: Use log-log linear fit
    try:
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)
        coeffs = np.polyfit(log_x, log_y, 1, w=y_fit*weights_fit if weights_fit is not None else None) # Weighted by y if errors provided
        alpha_guess = coeffs[0]
        amp_guess = np.exp(coeffs[1])
    except Exception:
        alpha_guess = 4.0 # Default guess if log-log fails
        amp_guess = np.mean(y_fit / (x_fit**alpha_guess))

    params.add('amp', value=amp_guess, min=1e-12)
    params.add('index', value=alpha_guess) # alpha (α)

    try:
        result = model.fit(y_fit, params, x=x_fit, weights=weights_fit)
        return result.params, model, result
    except Exception as e:
        print(f"Error during power law fit: {e}")
        return None, None, None


# --- Secondary Spectrum Calculation ---

def calculate_secondary_spectrum(
    dyn_spec: np.ndarray,
    time_res_s: float,
    freq_res_hz: float,
    subtract_mean: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the secondary spectrum (conjugate spectrum) of a dynamic spectrum.

    Args:
        dyn_spec: 2D dynamic spectrum (time, freq).
        time_res_s: Time resolution (sampling interval) in seconds.
        freq_res_hz: Frequency resolution (channel width) in Hz.
        subtract_mean: Subtract mean from each frequency channel before FFT.

    Returns:
        Tuple containing:
            - secondary_spec (np.ndarray): Magnitude squared of the 2D FFT,
                                           shifted so zero frequency/delay is center.
            - fd_axis (np.ndarray): Conjugate frequency (Doppler shift) axis in Hz.
            - tau_axis (np.ndarray): Conjugate time (delay) axis in seconds.
    """
    if dyn_spec.ndim != 2:
        raise ValueError("Input dynamic spectrum must be 2D (time, freq).")

    nt, nf = dyn_spec.shape
    proc_spec = dyn_spec.copy()

    # Check for and handle NaNs/Infs
    if np.any(~np.isfinite(proc_spec)) :
        print("Warning: Non-finite values found in dynamic spectrum. Replacing with median.")
        median_val = np.nanmedian(proc_spec)
        proc_spec[~np.isfinite(proc_spec)] = median_val

    # Subtract mean from each channel (frequency slice)
    if subtract_mean:
        proc_spec = proc_spec - np.mean(proc_spec, axis=0, keepdims=True)

    # Perform 2D FFT
    # FFT along time (axis 0), then frequency (axis 1)
    fft_res = np.fft.fft2(proc_spec)

    # Calculate magnitude squared and shift origin to center
    secondary_spec = np.abs(fftshift(fft_res))**2

    # Calculate conjugate axes
    # Delay axis (conjugate to frequency)
    tau_axis = fftshift(fftfreq(nf, d=freq_res_hz)) # Units: s
    # Doppler frequency axis (conjugate to time)
    fd_axis = fftshift(fftfreq(nt, d=time_res_s)) # Units: Hz

    return secondary_spec, fd_axis, tau_axis


# --- Physical Parameter Derivations ---

def scintillation_bandwidth_to_timescale(
    delta_nu_d_hz: float,
    freq_mhz: float,
    alpha: float = 4.0
    ) -> float:
    """
    Estimate scintillation timescale (τ_d) from bandwidth (Δν_d) using scaling relation.
    Assumes τ_d * Δν_d ≈ C / (2π), where C depends on the structure function.
    For Kolmogorov (alpha=4), C ≈ 1.16 is sometimes used for thin screen exponential ACF.
    Ref: Cordes & Lazio 1991 (ApJ 376), Gwinn et al. 1998.
    A simpler C=1 is often used as an approximation.

    Args:
        delta_nu_d_hz: Scintillation bandwidth (HWHM or FWHM - specify interpretation) in Hz.
                       Assuming this corresponds to the characteristic bandwidth.
        freq_mhz: Observing frequency in MHz (used for context, not direct calculation here).
        alpha: Power-law index of frequency scaling (Δν_d ∝ ν^α). Used for context.

    Returns:
        Estimated scintillation timescale (τ_d) in seconds.
        NOTE: This is an *estimate* based on scaling, not a direct measurement.
              The constant factor can vary. We use C=1 here.
    """
    if delta_nu_d_hz <= 0:
        return np.nan
    # Using the simple relation 2π * Δν_d * τ_d ≈ 1
    tau_d_s = 1.0 / (2.0 * np.pi * delta_nu_d_hz)
    return tau_d_s


def effective_velocity(
    lens_dist_kpc: float,
    source_dist_kpc: float,
    tau_d_ms: float, # Scintillation timescale
    delta_nu_d_khz: float, # Scintillation bandwidth
    freq_ghz: float, # Observation frequency
    is_earth_term_dominant: bool = False # If True, assumes screen is close to Earth
    ) -> float:
    """
    Estimate effective transverse velocity from scintillation parameters.
    Uses relation V_eff = sqrt(A_iss * D_eff) / tau_d
    where A_iss = lambda^2 / (2 * pi * theta_s^2), theta_s ~ 1 / (2 * pi * tau_d * nu) ? No, uses scattering angle.
    Alternative: V_eff ~ r_F / tau_d where r_F is Fresnel scale.
    r_F = sqrt(lambda * D_eff / 2*pi)
    D_eff = D_L * D_LS / D_S (effective distance)

    Let's use the formula from Cordes & Rickett 1998, Eq 5:
    V_eff = sqrt(2 * c * D_eff / freq) * sqrt(delta_nu_d / freq) / (2 * pi * tau_d * delta_nu_d)
    V_eff = sqrt(c * D_eff / (2 * pi^2 * freq * tau_d^2 * delta_nu_d)) --- Check this derivation
    Or simpler: V_eff = sqrt(D_eff * lambda / (2*pi)) / tau_d = FresnelScale / tau_d

    Args:
        lens_dist_kpc: Distance to the scattering screen (lens) in kpc.
        source_dist_kpc: Distance to the source (FRB) in kpc.
        tau_d_ms: Scintillation timescale in milliseconds.
        delta_nu_d_khz: Scintillation bandwidth in kHz.
        freq_ghz: Observing frequency in GHz.
        is_earth_term_dominant: If True, assumes D_eff ~ D_L (screen near observer).

    Returns:
        Effective transverse velocity in km/s.
    """
    if any(val <= 0 for val in [lens_dist_kpc, source_dist_kpc, tau_d_ms, delta_nu_d_khz, freq_ghz]):
        return np.nan

    D_S = source_dist_kpc * 1000 * PARSEC_M # Source distance in m
    D_L = lens_dist_kpc * 1000 * PARSEC_M # Lens distance in m
    D_LS = D_S - D_L # Lens-Source distance in m

    if D_LS <= 0:
      print("Warning: Lens distance >= Source distance. Check inputs.")
      return np.nan

    D_eff = (D_L * D_LS) / D_S if not is_earth_term_dominant else D_L # Effective distance in m

    lambda_m = C_MPS / (freq_ghz * 1e9) # Wavelength in m
    tau_d_s = tau_d_ms / 1000.0 # Timescale in s
    #delta_nu_d_hz = delta_nu_d_khz * 1000.0 # Bandwidth in Hz

    # Fresnel scale calculation
    r_F = np.sqrt(lambda_m * D_eff / (2.0 * np.pi)) # Fresnel scale in m

    # Effective velocity
    V_eff_mps = r_F / tau_d_s # Velocity in m/s

    return V_eff_mps / 1000.0 # Convert to km/s


def screen_distance_from_curvature(
    curvature: float, # Measured from secondary spectrum arc fit (s^3)
    freq_ghz: float, # Center frequency of observation
    source_dist_mpc: Optional[float] = None # Source distance (optional, for D_eff calc)
    ) -> float:
    """
    Estimate screen distance using the curvature of the scintillation arc.
    Curvature eta = D_eff * lambda^2 / (2 * c) = D_eff * c / (2 * freq^2)
    => D_eff = 2 * c * curvature / lambda^2 = 2 * freq_ghz^2 * curvature / c

    Args:
        curvature: Arc curvature (eta) in s^3 (or 1/Hz^2). Check units!
                   If fit is tau = eta * f_D^2, eta has units s / Hz^2 = s^3.
        freq_ghz: Central observing frequency in GHz.
        source_dist_mpc: Optional source distance in Mpc. If provided, can estimate
                         D_L; otherwise returns D_eff.

    Returns:
        Effective distance D_eff in pc, or lens distance D_L in pc if source_dist_mpc given.
    """
    if curvature <= 0 or freq_ghz <= 0:
        return np.nan

    freq_hz = freq_ghz * 1e9
    #lambda_m = C_MPS / freq_hz # Wavelength in m

    # Calculate D_eff from curvature
    # D_eff = eta * (2 * c * freq_hz**2)  --- Check formula source
    # Stinebring et al. 2001 Eq 1: eta = lambda^2 * D_eff / (2 * c)
    # D_eff = eta * 2 * c / lambda^2 = eta * 2 * c / (c/freq_hz)^2 = eta * 2 * c * freq_hz^2 / c^2
    D_eff_m = curvature * 2 * freq_hz**2 / C_MPS # Effective distance in meters

    D_eff_pc = D_eff_m / PARSEC_M # Effective distance in pc

    if source_dist_mpc is None:
        return D_eff_pc # Return effective distance in pc
    else:
        # Calculate D_L from D_eff = D_L * (1 - D_L/D_S)
        # D_eff = D_L - D_L^2 / D_S
        # (1/D_S) * D_L^2 - D_L + D_eff = 0
        # Quadratic formula for D_L
        D_S_pc = source_dist_mpc * 1e6 # Source distance in pc
        a = 1.0 / D_S_pc
        b = -1.0
        c_quad = D_eff_pc

        discriminant = b**2 - 4*a*c_quad
        if discriminant < 0:
            print("Warning: No real solution for D_L (discriminant < 0).")
            return np.nan

        # Two solutions for D_L:
        D_L1 = (-b + np.sqrt(discriminant)) / (2*a)
        D_L2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Physical solution must be 0 < D_L < D_S
        physical_DL = [d for d in [D_L1, D_L2] if 0 < d < D_S_pc]

        if len(physical_DL) == 1:
            return physical_DL[0] # Return lens distance D_L in pc
        elif len(physical_DL) == 0:
            print("Warning: No physical solution for D_L found (0 < D_L < D_S).")
            return np.nan
        else:
            # This case (two valid solutions) implies D_eff > D_S/4, screen near midpoint
            print(f"Warning: Two possible solutions for D_L: {D_L1:.2f} pc, {D_L2:.2f} pc")
            # Return the closer one? Or average? Or NaN? Returning closer one for now.
            return min(physical_DL)


# --- Miscellaneous Utilities ---

def weighted_avg_and_std(
    values: np.ndarray,
    weights: np.ndarray
    ) -> Tuple[float, float]:
    """
    Return the weighted average and weighted standard deviation.

    Args:
        values: NumPy ndarray of values.
        weights: NumPy ndarray of weights (same shape as values).

    Returns:
        Tuple containing:
            - Weighted average.
            - Weighted standard deviation.

    Raises:
        ValueError: If shapes mismatch or weights sum to zero.
    """
    if values.shape != weights.shape:
        raise ValueError("Shapes of values and weights must match.")
    if np.sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero.")

    average = np.average(values, weights=weights)
    # Weighted variance: sum(weights * (values - average)**2) / sum(weights)
    variance = np.average((values - average)**2, weights=weights)

    # Apply Bessel's correction for unbiased estimator if desired?
    # For weighted standard deviation, Bessel's correction is more complex:
    # V1 = sum(weights)
    # V2 = sum(weights**2)
    # corrected_variance = variance * V1 / (V1 - V2/V1)
    # For simplicity, return the population standard deviation sqrt(variance)
    return average, math.sqrt(variance)


# --- Plotting Utilities (Optional - can be kept separate) ---
# Placeholder for plotting functions if needed within the pipeline library itself.
# Usually better to keep plotting logic separate in the analysis script/notebook.