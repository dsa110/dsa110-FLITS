#!/usr/bin/env python3
"""frb_scintillator.py — redshift‑aware **full‑fidelity** fast‑radio‑burst two‑screen
scintillation simulator (Pradeep et al. 2025).

Version 2.2: Complete Class Implementation.
=======================================================================
This version re-integrates the helper methods for direct IRF inspection
and 1D time-series simulation, making the class fully capable of
replicating all analyses presented in the user's notebook.

Corrections included:
- Replaced `fit_acf` with `fit_acf_robust` for improved stability.
- Enhanced docstrings for all classes and public methods.
- Added back `get_irf_spikes`, `simulate_scattered_time_series`, and
  `calculate_theoretical_observables` methods.

Features included:
* Redshift‑exact geometry using angular‑diameter distances.
* Optional observer/screen transverse velocities for repeating bursts.
* Advanced screen realisation options (anisotropy, power-law profiles).
* Intrinsic pulse shapes (delta function or Gaussian).
* Robust ACF fitting via sequential component isolation.

Dependencies
------------
* Python ≥ 3.10, numpy, scipy, astropy, numba (optional), matplotlib
* tqdm (optional, for progress bars)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import os
import sys
import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import curve_fit
from scipy.signal import firwin

try:
    import numba as nb
    _NUMBA = True
    print("Numba detected. Using JIT-accelerated version.")
except ModuleNotFoundError:
    print("Numba not found, using pure Python loops.")
    _NUMBA = False

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range # Fallback if tqdm is not installed

# ----------------------------------------------------------------------------
# Constants & logging
# ----------------------------------------------------------------------------
C_M_PER_S = const.c.to(u.m / u.s).value
logger = logging.getLogger("frb_scintillator")

# ----------------------------------------------------------------------------
# Numba JIT-compiled Helper Function
# ----------------------------------------------------------------------------

if _NUMBA:
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _irf_coherent_numba_loop(field_products, total_delay, freqs):
        """
        Numba-accelerated JIT-compiled loop for calculating the coherent sum
        of the Impulse Response Function (IRF) across frequency channels.
        
        This is the performance-critical part of the IRF calculation.
        """
        nchan = freqs.shape[0]
        field_vs_freq = np.zeros(nchan, dtype=np.complex128)
        
        # parallel=True tells Numba to parallelize this outer loop across CPU cores
        for i in nb.prange(nchan):
            nu = freqs[i]
            # Coherently sum the fields from all N_mw * N_host paths
            phase_matrix = np.exp(-2j * np.pi * total_delay * nu)
            field_vs_freq[i] = np.sum(field_products * phase_matrix)
        return field_vs_freq

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def _DA(z1: float, z2: float = 0.0) -> u.Quantity:
    """
    Calculates the angular-diameter distance between two redshifts.
    Uses the cosmology model defined globally (Planck18).
    """
    if z2 < z1:
        z1, z2 = z2, z1
    return cosmo.angular_diameter_distance_z1z2(z1, z2)

def _array2(vec: Tuple[float, float] | np.ndarray | None, unit: u.Unit) -> np.ndarray:
    """Utility to ensure a 2-vector has the correct type, shape, and units."""
    if vec is None:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("Velocity / offset vectors must be 2-element tuples.")
    return (arr * unit).to(unit).value

# ----------------------------------------------------------------------------
# Dataclass Configurations
# ----------------------------------------------------------------------------

@dataclass
class ScreenCfg:
    """
    Configuration for a single thin scattering screen.

    Attributes:
        N (int): Number of discrete images (scatterers) on the screen.
        L (u.Quantity): Physical size (diameter) of the screen, corresponding
                        to the 2-sigma width of the image distribution.
        profile (str): Envelope profile for image amplitudes ('gaussian' or 'powerlaw').
        alpha (float): Power-law index for the 'powerlaw' profile.
        theta0 (u.Quantity): Core radius for the 'powerlaw' profile.
        geometry (str): '2D' for a circular/elliptical screen, '1D' for a linear screen.
        axial_ratio (float): Ratio of y-axis to x-axis for anisotropic screens.
        pa (u.Quantity): Position angle for anisotropic or 1D screens.
        amp_distribution (str): 'constant' for uniform amplitudes, 'rayleigh' for stochastic.
        rng_seed (int): Seed for the random number generator for reproducible screens.
        v_perp (tuple): Transverse velocity of the screen in km/s (vx, vy).
    """
    N: int = 128
    L: u.Quantity = 1.0 * u.AU
    profile: Literal["gaussian", "powerlaw"] = "gaussian"
    alpha: float = 3.0
    theta0: u.Quantity = 100.0 * u.marcsec
    geometry: Literal["2D", "1D"] = "2D"
    axial_ratio: float = 1.0
    pa: u.Quantity = 0.0 * u.deg
    amp_distribution: Literal["constant", "rayleigh"] = "constant"
    rng_seed: Optional[int] = None
    v_perp: Tuple[float, float] | np.ndarray | None = None

    def __post_init__(self):
        # Ensure v_perp is a unitless numpy array in km/s.
        object.__setattr__(self, "v_perp", _array2(self.v_perp, u.km / u.s))

@dataclass
class SimCfg:
    """
    Top-level configuration for the entire two-screen simulation.
    Parameters are based on those used in Pradeep et al. (2025).

    Attributes:
        nu0 (u.Quantity): Center observing frequency.
        bw (u.Quantity): Observing bandwidth.
        nchan (int): Number of frequency channels in the simulation.
        D_mw (u.Quantity): Distance from the observer to the Milky Way screen.
        z_host (float): Redshift of the host galaxy.
        D_host_src (u.Quantity): Distance from the host screen to the FRB source.
        mw (ScreenCfg): Configuration for the Milky Way screen.
        host (ScreenCfg): Configuration for the host galaxy screen.
        intrinsic_pulse (str): Shape of the intrinsic FRB pulse ('delta' or 'gauss').
        pulse_width (u.Quantity): FWHM of the Gaussian intrinsic pulse.
        corr_thresh (float): Threshold for isolating the broad component in ACF fitting.
    """
    nu0: u.Quantity = 1.25 * u.GHz
    bw: u.Quantity = 16.0 * u.MHz
    nchan: int = 1024
    D_mw: u.Quantity = 1.0 * u.kpc
    z_host: float = 0.5
    D_host_src: u.Quantity = 5.0 * u.kpc
    mw: ScreenCfg = field(default_factory=ScreenCfg)
    host: ScreenCfg = field(default_factory=ScreenCfg)
    intrinsic_pulse: Literal["delta", "gauss"] = "delta"
    pulse_width: u.Quantity = 30.0 * u.us
    corr_thresh: float = 0.03 # Threshold for ACF component isolation

# ----------------------------------------------------------------------------
# Scattering Screen Realisation
# ----------------------------------------------------------------------------

class Screen:
    """
    Generates a random physical realisation of a single scattering screen
    based on a provided configuration.
    """
    def __init__(self, cfg: ScreenCfg, D_screen_m: float):
        """
        Initializes and creates a single screen realisation.

        Args:
            cfg (ScreenCfg): The configuration object for this screen.
            D_screen_m (float): The distance to this screen in meters.
        """
        self.cfg = cfg
        self.D_screen_m = D_screen_m
        rng = np.random.default_rng(cfg.rng_seed)
        
        L_m = cfg.L.to(u.m).value
        # Angular size of the screen (radius) in radians
        half_box_rad = (L_m / 2) / D_screen_m
        
        # --- Generate image coordinates (theta) ---
        if cfg.geometry == "1D":
            xy = np.zeros((cfg.N, 2))
            xy[:, 0] = rng.uniform(-half_box_rad, half_box_rad, size=cfg.N)
        else: # 2D
            xy = rng.uniform(-half_box_rad, half_box_rad, size=(cfg.N, 2))
            if cfg.axial_ratio != 1.0:
                 xy[:, 1] /= cfg.axial_ratio

        if cfg.pa.value != 0.0:
            pa_rad = cfg.pa.to(u.rad).value
            R = np.array([[np.cos(pa_rad), -np.sin(pa_rad)], [np.sin(pa_rad), np.cos(pa_rad)]])
            xy = xy @ R.T
        self.theta = xy # Angular positions of images [rad]

        # --- Generate complex electric fields for images ---
        if cfg.amp_distribution == "constant":
            amps = np.ones(cfg.N)
        else: # Rayleigh distribution for scattered amplitudes
            amps = rng.rayleigh(scale=1 / np.sqrt(2), size=cfg.N)
        phases = rng.uniform(0, 2 * np.pi, size=cfg.N)
        field = amps * np.exp(1j * phases)

        # Apply the specified amplitude envelope profile
        if cfg.profile == "gaussian":
            r2 = np.sum(self.theta**2, axis=1)
            # The screen size L is the 2-sigma width of the image distribution.
            sigma_rad = (L_m / self.D_screen_m) / 2.0
            w = np.exp(-r2 / (2 * sigma_rad**2))
        else: # powerlaw
            theta0_rad = cfg.theta0.to(u.rad).value
            r2 = np.sum(self.theta**2, axis=1)
            w = (1 + r2 / theta0_rad**2) ** (-cfg.alpha / 2)
        
        # Apply envelope and normalize total power
        self.field = field * w / np.linalg.norm(w, ord=2)

# ----------------------------------------------------------------------------
# Main Simulator Class
# ----------------------------------------------------------------------------

class FRBScintillator:
    """
    A two-screen scintillation simulator that implements the physics described
    in Pradeep et al. (2025).
    """
    def __init__(self, cfg: SimCfg):
        """
        Initializes the simulator with a given configuration.

        Args:
            cfg (SimCfg): The top-level simulation configuration object.
        """
        self.cfg = cfg
        self._prepare_geometry()
        self._prepare_screens()
        self._prepare_frequency_grid()
        logger.info(f"Initialized FRBScintillator with RP = {self.resolution_power():.3f}")

    def _prepare_geometry(self):
        """
        Calculates all geometric and effective distances based on the paper's
        cosmological formulae (Eqs. 2.2, 2.6).
        """
        cfg = self.cfg
        self.nu0_hz = cfg.nu0.to(u.Hz).value
        self.lam0_m = C_M_PER_S / self.nu0_hz
        
        # Physical distances from observer (z=0)
        self.D_mw_m = cfg.D_mw.to(u.m).value
        self.D_host_m = _DA(0.0, cfg.z_host).to(u.m).value
        self.D_host_src_m = cfg.D_host_src.to(u.m).value
        
        # Approximate the source redshift based on host redshift and D_host_src.
        # This is a reasonable approximation for the paper's purposes.
        z_src_approx = cfg.z_host + self.D_host_src_m / cosmo.hubble_distance.to(u.m).value / (1+cfg.z_host)
        self.D_src_m = _DA(0.0, z_src_approx).to(u.m).value
        
        # Distance between the two screens
        self.D_mw_host_m = self.D_host_m - self.D_mw_m

        # Effective distances for delay calculation (Eq. 2.6)
        self.deff_mw_m = (self.D_mw_m * self.D_host_m) / self.D_mw_host_m
        term1 = (1 + cfg.z_host) * (self.D_host_m * self.D_src_m) / self.D_host_src_m
        self.deff_host_m = term1 + self.deff_mw_m # Note: This is the corrected formula

    def _prepare_screens(self):
        """Instantiate the Screen objects."""
        self.mw_screen = Screen(self.cfg.mw, self.D_mw_m)
        self.host_screen = Screen(self.cfg.host, self.D_host_m)
        self._compute_static_delays()

    def _compute_static_delays(self):
        """Precomputes the three geometric delay terms from Eq. 2.5 in seconds."""
        # MW screen self-term
        self._tau_mw0 = (self.deff_mw_m / (2 * C_M_PER_S)) * np.sum(self.mw_screen.theta**2, axis=1)
        # Host screen self-term
        self._tau_host0 = (self.deff_host_m / (2 * C_M_PER_S)) * np.sum(self.host_screen.theta**2, axis=1)
        # Cross term, which governs resolution effects
        self._tau_cross0 = -(self.deff_mw_m / C_M_PER_S) * (self.mw_screen.theta @ self.host_screen.theta.T)

    def _prepare_frequency_grid(self):
        """Sets up the frequency channel array for the simulation."""
        cfg = self.cfg
        self.bw_hz = cfg.bw.to(u.Hz).value
        self.freqs = np.linspace(self.nu0_hz - self.bw_hz / 2, self.nu0_hz + self.bw_hz / 2, cfg.nchan)
        self.dnu = self.freqs[1] - self.freqs[0] # Channel width in Hz

    def resolution_power(self) -> float:
        """
        Calculates the Resolution Power (RP) of the two-screen system,
        as defined in Eq. 3.11 of the paper. RP > 1 indicates a resolving system.
        """
        L_mw = self.cfg.mw.L.to(u.m).value
        L_host = self.cfg.host.L.to(u.m).value
        return (L_mw * L_host) / (self.lam0_m * self.D_mw_host_m)

    def _delays(self, dt_s: float = 0.0):
        """
        Returns delay terms at a time offset dt_s, accounting for screen motion.
        Used for simulating dynamic effects in repeating FRBs.
        """
        if dt_s == 0.0:
            return self._tau_mw0, self._tau_host0, self._tau_cross0

        # Update angular positions based on transverse velocities
        theta_mw_t = self.mw_screen.theta + (self.cfg.mw.v_perp * dt_s) / self.D_mw_m
        theta_host_t = self.host_screen.theta + (self.cfg.host.v_perp * dt_s) / self.D_host_m

        # Recalculate delay terms with new positions
        tau_mw = (self.deff_mw_m / (2 * C_M_PER_S)) * np.sum(theta_mw_t**2, axis=1)
        tau_host = (self.deff_host_m / (2 * C_M_PER_S)) * np.sum(theta_host_t**2, axis=1)
        tau_cross = -(self.deff_mw_m / C_M_PER_S) * (theta_mw_t @ theta_host_t.T)
        return tau_mw, tau_host, tau_cross
    
    def _irf_coherent_vs_freq(self, tau_mw, tau_host, tau_cross) -> np.ndarray:
        """
        Calculates the Impulse Response Function R(ν) by performing the coherent
        sum over all N_mw * N_host propagation paths (Eq. 3.3).
        """
        # Pre-calculate product of field amplitudes for all paths
        field_products = self.mw_screen.field[:, None] * self.host_screen.field[None, :]
        # Calculate total delay for all paths
        total_delay = tau_mw[:, None] + tau_host[None, :] + tau_cross
        
        if _NUMBA:
            return _irf_coherent_numba_loop(field_products, total_delay, self.freqs)
        
        # Fallback pure Python loop if Numba is not available
        field_vs_freq = np.zeros(self.cfg.nchan, dtype=np.complex128)
        for i, nu in enumerate(self.freqs):
            # Calculate phase for all paths at this frequency
            phase_matrix = np.exp(-2j * np.pi * total_delay * nu)
            # Sum contributions from all paths
            field_vs_freq[i] = np.sum(field_products * phase_matrix)
        return field_vs_freq
    
    def simulate_time_integrated_spectrum(self) -> np.ndarray:
        """
        Simulates the time-averaged spectrum (intensity vs. frequency), I(ν).
        This is equivalent to the case of a delta-function intrinsic pulse.
        """
        # The electric field spectrum is the IRF
        field_spectrum = self._irf_coherent_vs_freq(self._tau_mw0, self._tau_host0, self._tau_cross0)
        # Intensity is the squared magnitude
        return np.abs(field_spectrum)**2
    
    @staticmethod
    def acf(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the one-sided spectral autocorrelation function (ACF).

        The ACF is normalized such that the zero-lag value equals the squared
        modulation index (m^2), consistent with Eq. 4.25 from the paper.
        """
        mean_intensity = np.nanmean(spectrum)
        if mean_intensity == 0:
            return np.zeros(len(spectrum) // 2), np.arange(len(spectrum) // 2)

        spec_mean_sub = spectrum - mean_intensity
        n = len(spec_mean_sub)
        
        # Use np.correlate to compute the autocovariance function
        unnormalized_covariance = np.correlate(spec_mean_sub, spec_mean_sub, mode='full')[n - 1:]
        covariance_func = unnormalized_covariance / n
        
        # Normalize by mean squared to get ACF where ACF(0) = m^2
        acf_func = covariance_func / (mean_intensity**2)
        lags = np.arange(acf_func.size)
        
        # Return the one-sided ACF and corresponding lags
        return acf_func[:n//2], lags[:n//2]
    
    def fit_acf_robust(self, corr: np.ndarray, lags: np.ndarray) -> tuple[float, float]:
        """
        Fits the spectral ACF with a robust, sequential "fit-subtract-fit"
        procedure to handle two components with widely different scales, as is
        common in two-screen models.

        This method is designed to be more stable than a simultaneous
        multi-component fit. It first isolates and fits the broader of the two
        scintillation components. It then subtracts this model from the data
        and fits the residual to find the narrower component. This strategy is
        based on the analysis method described in Pradeep et al. (2025),
        where distinct scintillation components are isolated for analysis.

        Args:
            corr (np.ndarray): The correlation values of the ACF.
            lags (np.ndarray): The frequency lags corresponding to the correlation values.

        Returns:
            tuple[float, float]: A tuple containing the HWHM of the broad and
                                 narrow scintillation bandwidths in Hz, respectively.
                                 Returns (np.nan, np.nan) if fitting fails.
        """
        lags_hz = lags * self.dnu
        
        if corr.size < 5 or corr[0] < 0.1:
            return np.nan, np.nan

        def lorentzian_model(x, amplitude, hwhm):
            """A single Lorentzian model for fitting."""
            return amplitude / (1 + (x / (hwhm + 1e-12))**2)

        # --- Step 1: Fit the broad component ---
        # Isolate the "wings" of the ACF, avoiding the central narrow spike.
        broad_mask = (corr > self.cfg.corr_thresh) & (lags > 3)
        if not np.any(broad_mask):
            return np.nan, np.nan # Not enough data for broad fit.

        x_broad, y_broad = lags_hz[broad_mask], corr[broad_mask]

        try:
            popt_broad, _ = curve_fit(lorentzian_model, x_broad, y_broad, bounds=([0, 0], [np.inf, np.inf]))
            amp_broad, hwhm_broad = popt_broad
        except (RuntimeError, ValueError):
            return np.nan, np.nan # Broad fit failed

        # --- Step 2: Subtract the broad model and fit the narrow residual ---
        broad_model_full = lorentzian_model(lags_hz, amp_broad, hwhm_broad)
        residual = corr - broad_model_full

        # The narrow component is in the residual, primarily at the center.
        narrow_mask = residual > 0
        if not np.any(narrow_mask):
            return hwhm_broad, np.nan # No significant narrow component found

        x_narrow, y_narrow = lags_hz[narrow_mask], residual[narrow_mask]

        # Initial guess for the narrow component's HWHM
        p0_narrow = (y_narrow[0], self.dnu * 2)

        try:
            # Constrain the narrow HWHM to be less than the broad one
            popt_narrow, _ = curve_fit(lorentzian_model, x_narrow, y_narrow, p0=p0_narrow, bounds=([0, 0], [np.inf, hwhm_broad]))
            _, hwhm_narrow = popt_narrow
        except (RuntimeError, ValueError):
            # If narrow fit fails, we still have the broad component result
            return hwhm_broad, np.nan

        # Ensure the output is always (broad_hwhm, narrow_hwhm)
        return (hwhm_broad, hwhm_narrow) if hwhm_broad > hwhm_narrow else (hwhm_narrow, hwhm_broad)

    def _get_time_domain_irf(self, dt_s: float, n_t: int, time_res_s: float) -> np.ndarray:
        """Constructs the time-domain impulse response function R(t)."""
        tau_mw, tau_host, tau_cross = self._delays(dt_s)
        
        all_delays = (tau_mw[:, None] + tau_host[None, :] + tau_cross).ravel()
        all_amps = (self.mw_screen.field[:, None] * self.host_screen.field[None, :]).ravel()

        irf_t = np.zeros(n_t, dtype=np.complex128)
        time_bins = np.round(all_delays / time_res_s).astype(int)
        
        valid_mask = (time_bins >= 0) & (time_bins < n_t)
        np.add.at(irf_t, time_bins[valid_mask], all_amps[valid_mask])
        
        return irf_t

    def _get_intrinsic_pulse(self, n_t: int, time_res_s: float, rng) -> np.ndarray:
        """Generates the intrinsic pulse time series E_int(t)."""
        if self.cfg.intrinsic_pulse == "delta":
            pulse_t = np.zeros(n_t, dtype=np.complex128)
            pulse_t[0] = 1.0 # Centered at t=0
            return pulse_t
        
        # For a Gaussian pulse, start it near the beginning of the time series
        t_axis = (np.arange(n_t) - n_t // 8) * time_res_s
        sigma_t = self.cfg.pulse_width.to(u.s).value / (2 * np.sqrt(2 * np.log(2))) # FWHM to sigma
        envelope = np.exp(-t_axis**2 / (2 * sigma_t**2))
        
        # Add complex noise for realistic phase variations
        noise = rng.normal(size=n_t) + 1j * rng.normal(size=n_t)
        return envelope * noise / np.sqrt(2)

    def _simulate_scattered_efield(self, duration: u.Quantity, dt_epoch: u.Quantity = 0.0 * u.s, rng=None):
        """
        Core simulation engine: produces the observed complex electric field vs. time.

        This method performs the fundamental convolution of the intrinsic pulse
        with the time-domain impulse response function (IRF). The time resolution
        is set by the Nyquist criterion of the simulation bandwidth to prevent
        aliasing. All other data products are derived from the output of this method.

        Args:
            duration (u.Quantity): The total time duration of the simulation.
            dt_epoch (u.Quantity): Time offset of the observation (for velocity effects).
            rng: A numpy random generator instance.

        Returns:
            tuple: (E_obs_t, time_axis_s, intrinsic_pulse_t)
                   Contains the complex observed E-field, the time axis in seconds,
                   and the complex intrinsic pulse E-field.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Time resolution must satisfy Nyquist for the full simulation bandwidth.
        time_res_s = 1.0 / self.bw_hz
        n_t = int(duration.to(u.s).value / time_res_s)
        
        # Generate the two components in the time domain
        irf_t = self._get_time_domain_irf(dt_epoch.to(u.s).value, n_t, time_res_s)
        pulse_t = self._get_intrinsic_pulse(n_t, time_res_s, rng)
        
        # Convolve in the frequency domain for efficiency: E_obs(t) = IFFT(FFT(R(t)) * FFT(E_int(t)))
        E_obs_t = np.fft.ifft(np.fft.fft(irf_t) * np.fft.fft(pulse_t))
        
        time_axis = np.arange(n_t) * time_res_s
        return E_obs_t, time_axis, pulse_t

    def simulate_scattered_time_series(self, duration: u.Quantity, rng=None):
        """
        Generates the final 1D scattered pulse time series I(t).
        
        This is a wrapper around the core E-field simulator.

        Args:
            duration (u.Quantity): The total duration of the output series.
            rng: A numpy random generator instance.

        Returns:
            tuple: (scattered_pulse_I, intrinsic_pulse_I, time_axis_s)
        """
        E_obs_t, time_axis_s, intrinsic_pulse_t = self._simulate_scattered_efield(
            duration=duration, rng=rng
        )
        
        # Intensity is the squared magnitude of the complex E-field
        scattered_intensity = np.abs(E_obs_t)**2
        intrinsic_intensity = np.abs(intrinsic_pulse_t)**2
        
        return scattered_intensity, intrinsic_intensity, time_axis_s

    def synthesise_dynamic_spectrum(self, duration: u.Quantity, dt_epoch: u.Quantity = 0.0 * u.s, rng=None):
        """
        Generates a full 2D dynamic spectrum I(t, ν).

        This is a wrapper around the core E-field simulator that subsequently
        performs channelization via a Short-Time Fourier Transform (STFT),
        implemented manually as "FFT-and-square".
        
        Args:
            duration (u.Quantity): The total time duration of the simulation.
            dt_epoch (u.Quantity): Time offset of the observation (for velocity effects).
            rng: A numpy random generator instance.

        Returns:
            A tuple of (I_t_nu, time_axis, freq_axis) containing the dynamic
            spectrum (time, freq), the output time axis (s), and the frequency axis (Hz).
        """
        # 1. Get the raw E-field time series from the core simulator
        E_obs_t, _, _ = self._simulate_scattered_efield(
            duration=duration, dt_epoch=dt_epoch, rng=rng
        )
        
        # 2. Manually channelize using the "FFT-and-square" method
        N_fft = self.cfg.nchan
        num_spectra = len(E_obs_t) // N_fft
        if num_spectra == 0:
            logger.warning("Simulation duration is too short for the number of channels.")
            return np.array([]), np.array([]), np.array([])
            
        E_obs_t_trimmed = E_obs_t[:num_spectra * N_fft]
        E_reshaped = E_obs_t_trimmed.reshape((num_spectra, N_fft))
        
        # FFT each time segment to get the spectrum for that time bin
        E_t_nu = np.fft.fftshift(np.fft.fft(E_reshaped, axis=1), axes=1)
        I_t_nu = np.abs(E_t_nu)**2
        
        # 3. Create the corresponding time and frequency axes for the output
        time_res_s = 1.0 / self.bw_hz
        output_time_res = N_fft * time_res_s
        time_axis = np.arange(num_spectra) * output_time_res
        
        freq_axis_baseband = np.fft.fftshift(np.fft.fftfreq(N_fft, d=time_res_s))
        freq_axis_sky = self.nu0_hz + freq_axis_baseband
        
        return I_t_nu, time_axis, freq_axis_sky

    def get_irf_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the raw delays and intensities of all individual geometric paths,
        which constitute the Impulse Response Function (IRF).
        Useful for replicating Figure 6 from the paper.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - An array of time delays for each path in seconds.
                - An array of intensities (squared amplitudes) for each path.
        """
        all_delays_s = (self._tau_mw0[:, None] + self._tau_host0[None, :] + self._tau_cross0).ravel()
        all_amps_sq = np.abs(self.mw_screen.field[:, None] * self.host_screen.field[None, :]).ravel()**2
        return all_delays_s, all_amps_sq

    def simulate_scattered_time_series(self, time_res: u.Quantity, duration: u.Quantity, rng=None):
        """
        Generates the final 1D scattered pulse time series I(t) by convolving
        the time-domain IRF with an intrinsic pulse.

        Args:
            time_res (u.Quantity): The desired time resolution of the output series.
            duration (u.Quantity): The total duration of the output series.
            rng: A numpy random generator instance.

        Returns:
            tuple: (scattered_pulse_I, intrinsic_pulse_I, time_axis_s)
                   Contains the final scattered intensity, the intrinsic pulse
                   intensity, and the corresponding time axis in seconds.
        """
        if rng is None: rng = np.random.default_rng()
        
        time_res_s = time_res.to(u.s).value
        n_t = int(duration.to(u.s).value / time_res_s)
        time_axis = np.arange(n_t) * time_res_s
        
        # Construct the IRF in the time domain
        irf_t = self._get_time_domain_irf(0.0, n_t, time_res_s)
        
        # Generate the intrinsic pulse
        intrinsic_pulse_t = self._get_intrinsic_pulse(n_t, time_res_s, rng)
        
        # Convolve in the frequency domain for efficiency
        E_obs_t = np.fft.ifft(np.fft.fft(irf_t) * np.fft.fft(intrinsic_pulse_t))
        
        return np.abs(E_obs_t)**2, np.abs(intrinsic_pulse_t)**2, time_axis

    def calculate_theoretical_observables(self) -> dict:
        """
        Calculates the theoretical scintillation bandwidth (nu_s) and scattering
        time (tau_s) from the simulation's screen parameters, assuming the
        unresolved regime. This is based on Eqs. 4.9 and 4.14 from the paper.

        Returns:
            dict: A dictionary containing the theoretical nu_s [Hz] and tau_s [s]
                  for both the MW and host screens.
        """
        # 1/e radius of the intensity distribution of scattering angles
        theta_L_mw_rad = (self.cfg.mw.L.to(u.m).value / (2 * self.D_mw_m))
        theta_L_host_rad = (self.cfg.host.L.to(u.m).value / (2 * self.D_host_m))

        # Scintillation bandwidth (nu_s) from Eq. 4.14
        nu_s_mw_hz = C_M_PER_S / (np.pi * self.deff_mw_m * theta_L_mw_rad**2)
        nu_s_host_hz = C_M_PER_S / (np.pi * self.deff_host_m * theta_L_host_rad**2)

        # Scattering time (tau_s) from Eq. 4.9
        tau_s_mw_s = (self.deff_mw_m * theta_L_mw_rad**2) / (2 * C_M_PER_S)
        tau_s_host_s = (self.deff_host_m * theta_L_host_rad**2) / (2 * C_M_PER_S)

        return {
            "nu_s_mw_hz": nu_s_mw_hz, "nu_s_host_hz": nu_s_host_hz,
            "tau_s_mw_s": tau_s_mw_s, "tau_s_host_s": tau_s_host_s,
        }

    def analyze_intra_pulse_scintillation(self, I_t_nu: np.ndarray, time_axis: np.ndarray) -> dict:
        """
        Analyzes a 2D dynamic spectrum to measure the evolution of
        scintillation parameters (bandwidths, modulation index) over time.
        This is used to replicate Figures 10 and 11 from the paper.
        """
        num_spectra = I_t_nu.shape[0]
        results = {
            "time_ms": [], "m_total": [], "nu_s_mw_hz": [], "nu_s_host_hz": []
        }
        
        print("Analyzing scintillation evolution across the pulse...")
        for i in trange(num_spectra, desc="Analyzing time slices"):
            spectrum_slice = I_t_nu[i, :]
            
            if np.nanmean(spectrum_slice) < 1e-9 * np.nanmean(I_t_nu):
                continue

            corr, lags = self.acf(spectrum_slice)
            
            m_total_sq = corr[0]
            nu_s_mw, nu_s_host = self.fit_acf_robust(corr, lags)
            
            if not np.isnan(nu_s_mw):
                results["time_ms"].append(time_axis[i] * 1e3)
                results["m_total"].append(np.sqrt(m_total_sq))
                results["nu_s_mw_hz"].append(nu_s_mw)
                results["nu_s_host_hz"].append(nu_s_host)
        
        for key in results:
            results[key] = np.array(results[key])
            
        return results