from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import curve_fit

try:
    import numba as nb
    _NUMBA = True
    print("Numba detected. Using JIT-accelerated version.")
except ModuleNotFoundError:
    _NUMBA = False
    print("Numba not found, using pure Python loops.")

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range  # Fallback if tqdm is not installed

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:
    Parallel = None
    delayed = None

from geometry import _DA
from screen import ScreenCfg, Screen
from instrument import InstrumentalCfg

C_M_PER_S = const.c.to(u.m / u.s).value
logger = logging.getLogger("frb_scintillator")

if _NUMBA:
    @nb.njit(cache=True, fastmath=True)
    def _irf_coherent_numba_loop(field_products, total_delay, freqs):
        """Vectorised IRF calculation."""
        phase = np.exp(-2j * np.pi * total_delay[:, None] * freqs[None, :])
        field_mtx = field_products[:, None] * phase
        return np.sum(field_mtx, axis=0)


@dataclass
class SimCfg:
    """
    Top-level configuration for the entire two-screen simulation.
    Parameters are based on those used in Pradeep et al. (2025).

    Attributes:
        peak_flux (u.Quantity): Peak flux density of the FRB. 
                                This refers to the peak of the *intrinsic* pulse.
        nu0 (u.Quantity): Center observing frequency.
        bw (u.Quantity): Observing bandwidth.
        nchan (int): Number of frequency channels in the simulation output.
        D_mw (u.Quantity): Distance from the observer to the Milky Way screen.
        z_host (float): Redshift of the host galaxy.
        D_host_src (u.Quantity): Distance from the host screen to the FRB source.
        mw (ScreenCfg): Configuration for the Milky Way screen.
        host (ScreenCfg): Configuration for the host galaxy screen.
        intrinsic_pulse (str): Shape of the intrinsic FRB pulse ('delta' or 'gauss').
        pulse_width (u.Quantity): FWHM of the Gaussian intrinsic pulse.
        corr_thresh (float): Threshold for isolating the broad component in ACF fitting.
        instrument (InstrumentalCfg): Instrumental settings for noise.
    """
    peak_flux: u.Quantity = 1.0 * u.Jy
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
    instrument: InstrumentalCfg = field(default_factory=InstrumentalCfg)

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
        self.n_chan = self.cfg.nchan
        self.C_M_PER_S = C_M_PER_S

        self._prepare_geometry()
        self._prepare_screens()
        self._precompute_doppler_terms()
        self._prepare_frequency_grid()
        
        self._validate_resolution()
        
        logging.info(f"Initialized FRBScintillator with RP = {self.resolution_power():.3f}")

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
        self.deff_host_m = term1 + self.deff_mw_m

    def _compute_static_delays(self):
        """
        Precomputes the three geometric delay terms from Eq. 2.5 in seconds.
        """
        # Apply anisotropy scaling before squaring theta for the self-terms.
        theta_mw_scaled = self.mw_screen.theta * self.mw_screen.anisotropy_scaling
        theta_host_scaled = self.host_screen.theta * self.host_screen.anisotropy_scaling
        
        # MW screen self-term (anisotropic)
        self._tau_mw0 = (self.deff_mw_m / (2 * self.C_M_PER_S)) * np.sum(theta_mw_scaled**2, axis=1)
        # Host screen self-term (anisotropic)
        self._tau_host0 = (self.deff_host_m / (2 * self.C_M_PER_S)) * np.sum(theta_host_scaled**2, axis=1)
        
        # The cross term mixes geometries. For now, we use the standard isotropic
        # dot product, which is a reasonable approximation.
        self._tau_cross0 = -(self.deff_mw_m / self.C_M_PER_S) * (self.mw_screen.theta @ self.host_screen.theta.T)

    def _prepare_screens(self):
        """
        Instantiate the Screen objects.
        """
        self.mw_screen = Screen(self.cfg.mw, self.D_mw_m)
        self.host_screen = Screen(self.cfg.host, self.D_host_m)
        self._compute_static_delays()
        
    def _precompute_doppler_terms(self):
        """
        Pre-calculates the coefficients for the time-evolution of the delay,
        now fully accounting for screen anisotropy.
        """
        self._tau_static = (self._tau_mw0[:, None] + self._tau_host0[None, :] + self._tau_cross0)

        v_mw_mps = self.cfg.mw.v_perp * 1000
        v_host_mps = self.cfg.host.v_perp * 1000
        
        # --- FIX #4: Anisotropy Implementation ---
        # Apply anisotropy scaling to both theta and velocity vectors
        theta_mw_scaled = self.mw_screen.theta * self.mw_screen.anisotropy_scaling
        v_mw_scaled = v_mw_mps * self.mw_screen.anisotropy_scaling
        
        theta_host_scaled = self.host_screen.theta * self.host_screen.anisotropy_scaling
        v_host_scaled = v_host_mps * self.host_screen.anisotropy_scaling
        
        # B term: The linear (Doppler) coefficient
        dtau_dv_mw = (self.deff_mw_m / self.C_M_PER_S) * (theta_mw_scaled @ v_mw_scaled.T)
        dtau_dv_host = (self.deff_host_m / self.C_M_PER_S) * (theta_host_scaled @ v_host_scaled.T)
        dtau_dv_cross = -(self.deff_mw_m / self.C_M_PER_S) * (v_mw_mps @ self.host_screen.theta.T + self.mw_screen.theta @ v_host_mps.T)
        self._tau_linear_coeff = dtau_dv_mw[:, None] + dtau_dv_host[None, :] + dtau_dv_cross

        # A term: The quadratic (fringe acceleration) coefficient
        d2tau_dv2_mw = (self.deff_mw_m / (2 * self.C_M_PER_S)) * np.sum(v_mw_scaled**2)
        d2tau_dv2_host = (self.deff_host_m / (2 * self.C_M_PER_S)) * np.sum(v_host_scaled**2)
        d2tau_dv2_cross = -(self.deff_mw_m / self.C_M_PER_S) * np.sum(v_mw_mps * v_host_mps)
        self._tau_quad_coeff = d2tau_dv2_mw + d2tau_dv2_host + d2tau_dv2_cross
        
    def _prepare_frequency_grid(self):
        """
        Sets up the frequency channel array for the simulation.
        """
        # Use a single, consistent definition for the frequency grid.
        self.bw_hz = self.cfg.bw.to(u.Hz).value
        self.dnu_hz = self.bw_hz / self.n_chan
        # Define channel centers relative to nu0
        freq_offsets = (np.arange(self.n_chan) - (self.n_chan - 1) / 2) * self.dnu_hz
        self.freqs_hz = self.nu0_hz + freq_offsets

    def _delays(self, dt_s: float = 0.0, speed: str = 'fast'):
        """
        Calculate geometric delays for all propagation paths.

        Parameters
        ----------
        dt_s : float
            Time offset in seconds (for dynamic spectra of repeaters)

        Returns
        -------
        tau_mw, tau_host, tau_cross : ndarray
            Delay components in seconds
        """
        if dt_s == 0.0:
            return self._tau_mw0, self._tau_host0, self._tau_cross0
        
        if speed == 'fast':
            return self._tau_static + self._tau_linear_coeff * dt_s + self._tau_quad_coeff * dt_s**2

        else:
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
        field_products = (self.mw_screen.field[:, None] * self.host_screen.field[None, :]).ravel()
        # Calculate total delay for all paths
        total_delay = (tau_mw[:, None] + tau_host[None, :] + tau_cross).ravel()
        
        if _NUMBA:
            return _irf_coherent_numba_loop(field_products, total_delay, self.freqs_hz)
    
        # Fallback pure Python loop if Numba is not available
        field_vs_freq = np.zeros(self.n_chan, dtype=np.complex128)
        for i, nu in enumerate(self.freqs_hz):
            # Calculate phase for all paths at this frequency
            phase_matrix = np.exp(-2j * np.pi * total_delay * nu)
            # Sum contributions from all paths
            field_vs_freq[i] = np.sum(field_products * phase_matrix)
        return field_vs_freq

    def _simulate_scattered_efield(self, duration: u.Quantity, dt_epoch: u.Quantity = 0.0 * u.s, rng=None):
        """
        Core simulation engine: produces the observed complex electric field vs. time.

        This method performs the fundamental convolution of the intrinsic pulse
        with the time-domain impulse response function (IRF), and adds instrumental
        noise. The time resolution is set by the Nyquist criterion of the simulation
        bandwidth to prevent aliasing.

        Args:
            duration (u.Quantity): The total time duration of the simulation.
            dt_epoch (u.Quantity): Time offset of the observation (for velocity effects).
            rng: A numpy random generator instance.

        Returns:
            tuple: (E_obs_t, time_axis)
                   Contains the complex observed E-field in sqrt(Jy) and the time axis in seconds.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Nyquist sampling for the full bandwidth
        time_res_s = 1.0 / self.cfg.bw.to_value(u.Hz)
        n_t = int(duration.to_value(u.s) / time_res_s)
        
        # 1. Get the unitless time-domain IRF 
        irf_t = self._get_time_domain_irf(dt_epoch.to_value(u.s), n_t, time_res_s)
        
        # 2. Get the physically-scaled intrinsic pulse E-field 
        pulse_t_phys = self._get_intrinsic_pulse(n_t, time_res_s, rng)
        
        # 3. Convolve to get the noise-free scattered signal in physical units
        E_signal_phys = np.fft.ifft(np.fft.fft(irf_t) * np.fft.fft(pulse_t_phys))
        
        # 4. Add physically modeled thermal noise 
        sefd_jy = self.cfg.instrument.get_sefd_jy()
        if sefd_jy is None:
            # If no instrument is defined, return the noise-free signal
            E_obs_t = E_signal_phys
        else:
            # Use the rigorously correct noise variance derivation.
            # The variance of each complex voltage component is half the total noise power
            # := 0.5 x SEFD (which is a power spectral density, Jy) divided by n_pol.
            n_pol = self.cfg.instrument.n_pol
            variance_component = sefd_jy / (2.0 * n_pol)
            sigma_component_phys = np.sqrt(variance_component)

            noise = rng.normal(scale=sigma_component_phys, size=n_t) + \
                    1j * rng.normal(scale=sigma_component_phys, size=n_t)
            
            E_obs_t = E_signal_phys + noise
            
        time_axis = np.arange(n_t) * time_res_s
        return E_obs_t, time_axis
    
    def _get_time_domain_irf(self, dt_s: float, n_t: int, time_res_s: float) -> np.ndarray:
        """
        Constructs the time-domain impulse response function R(t).
        """
        tau_mw, tau_host, tau_cross = self._delays(dt_s)
        
        all_delays = (tau_mw[:, None] + tau_host[None, :] + tau_cross).ravel()
        all_amps = (self.mw_screen.field[:, None] * self.host_screen.field[None, :]).ravel()
        
        irf_t = np.zeros(n_t, dtype=np.complex128)
        time_bins = np.round(all_delays / time_res_s).astype(int)
        
        valid_mask = (time_bins >= 0) & (time_bins < n_t)
        np.add.at(irf_t, time_bins[valid_mask], all_amps[valid_mask])
        
        return irf_t
    
    def _get_intrinsic_pulse(self, n_t: int, time_res_s: float, rng) -> np.ndarray:
        """
        Generates the physically-scaled intrinsic pulse time series E_int(t).
        The pulse is scaled such that its peak intensity corresponds to cfg.peak_flux.
        """
        # --- Generate the unitless complex pulse shape ---
        if self.cfg.intrinsic_pulse == "delta":
            # For a delta function, the E-field is concentrated in one sample.
            # Its intensity is normalized to 1 for scaling purposes.
            pulse_t_unscaled = np.zeros(n_t, dtype=np.complex128)
            pulse_t_unscaled[n_t // 8] = 1.0 # Start pulse 1/8th of the way in
        else: # Gaussian pulse
            # This follows the paper's description of a complex random field
            # with a Gaussian envelope.
            t_axis = (np.arange(n_t) - n_t // 8) * time_res_s
            # Convert FWHM to sigma for the Gaussian envelope
            sigma_t = self.cfg.pulse_width.to_value(u.s) / (2 * np.sqrt(2 * np.log(2)))
            envelope = np.exp(-t_axis**2 / (2 * sigma_t**2))
            
            # Complex white noise provides the stochastic phase
            noise = rng.normal(size=n_t) + 1j * rng.normal(size=n_t)
            pulse_t_unscaled = envelope * noise / np.sqrt(2) # Normalize noise power

        # --- Scale the pulse to physical units (sqrt(Jy)) ---
        # Find the peak intensity of the unscaled, intrinsic pulse
        peak_I_unscaled = np.max(np.abs(pulse_t_unscaled)**2)
        
        # Get the desired intrinsic peak flux in Jy
        peak_flux_jy = self.cfg.peak_flux.to_value(u.Jy)
        
        # Calculate the scale factor to match the desired peak flux.
        # E_phys = E_unscaled * scale_factor  =>  I_phys = I_unscaled * scale_factor**2
        scale_factor = np.sqrt(peak_flux_jy / peak_I_unscaled) if peak_I_unscaled > 0 else 0.0
        
        # Return the E-field in physical units of sqrt(Jy)
        return pulse_t_unscaled * scale_factor
    
    def _get_intrinsic_pulse_freq_domain(self, rng) -> np.ndarray:
        """
        Generates the intrinsic pulse in the time domain and returns its
        Fourier transform, E_int(ν).
        """
        # We must generate the time-domain pulse at the Nyquist resolution
        # of the full bandwidth to avoid aliasing before the FFT.
        time_res_s = 1.0 / self.bw_hz
        # Use nchan as the number of time samples for the FFT. This makes the
        # resulting E_int(ν) have the same shape as our frequency grid.
        n_t = self.n_chan
        
        # Get the physically-scaled time-domain pulse E_int(t)
        pulse_t_phys = self._get_intrinsic_pulse(n_t, time_res_s, rng)
        
        # Return its Fourier transform, E_int(ν)
        return np.fft.fft(pulse_t_phys)
    
    def _validate_resolution(self):
        """
        Check if frequency resolution is sufficient for the narrowest scintillation.
        """
        theo = self.calculate_theoretical_observables()
        # Use the scalar channel width `dnu_hz` for the comparison
        if self.dnu_hz > theo['nu_s_host_hz'] / 10:
            logger.warning(
                f"Channel width ({self.dnu_hz/1e3:.1f} kHz) may be too coarse "
                f"to resolve host scintillation ({theo['nu_s_host_hz']/1e3:.1f} kHz)"
            )

    def resolution_power(self) -> float:
        """
        Calculates the Resolution Power (RP) of the two-screen system,
        as defined in Eq. 3.11 of the paper. RP > 1 indicates a resolving system.
        """
        L_mw = self.cfg.mw.L.to(u.m).value
        L_host = self.cfg.host.L.to(u.m).value
        return (L_mw * L_host) / (self.lam0_m * self.D_mw_host_m)
    
    def simulate_time_integrated_spectrum(self, dt_epoch: u.Quantity = 0.0 * u.s, rng=None):
        """
        Simulates a time-integrated spectrum |R(ν)|² for a delta-function pulse.
        This is the fundamental observable for analyzing scintillation.
        """
        tau_mw, tau_host, tau_cross = self._delays(dt_epoch.to_value(u.s))
        irf_freq = self._irf_coherent_vs_freq(tau_mw, tau_host, tau_cross)

        # For noise-free case, return |R(ν)|²
        spectrum = np.abs(irf_freq)**2

        # Add noise if instrumental parameters are specified
        if self.cfg.instrument.get_sefd_jy() is not None:
            spectrum = self._add_spectral_noise(spectrum, rng)

        return spectrum
    
    def simulate_1d_time_series(self, duration: u.Quantity, rng=None):
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
        E_obs_t, time_axis_s = self._simulate_scattered_efield(duration=duration, rng=rng)
        return np.abs(E_obs_t)**2, time_axis_s
    
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
            A tuple of (I_t_nu, time_axis, freq_axis_sky) containing the dynamic
            spectrum (time, freq), the output time axis (s), and the frequency axis (Hz).
        """
        # Get the raw E-field time series from the core simulator.
        # This now uses the correct baseband time resolution.
        E_obs_t, time_axis_raw = self._simulate_scattered_efield(
            duration=duration, dt_epoch=dt_epoch, rng=rng
        )
        
        time_res_s_raw = time_axis_raw[1] - time_axis_raw[0]

        # The integration time for each spectrum in the dynamic spectrum
        N_fft = self.cfg.nchan
        tint_s = N_fft * time_res_s_raw
        
        # Manually channelize using the "FFT-and-square" method
        num_spectra = len(E_obs_t) // N_fft
        if num_spectra == 0:
            logger.warning("Simulation duration is too short for the number of channels.")
            return np.array([]), np.array([]), np.array([])
        
        # FFT each time segment to get the spectrum for that time bin
        E_reshaped = E_obs_t[:num_spectra * N_fft].reshape((num_spectra, N_fft))
        E_t_nu = np.fft.fftshift(np.fft.fft(E_reshaped, axis=1), axes=1)
        
        # Power Spectral Density is proportional to |FFT(V)|^2. The 1/N_fft factor
        # is a convention for Parseval's theorem for discrete transforms.
        I_t_nu = np.abs(E_t_nu)**2 / N_fft

        # The output time axis corresponds to the start of each integrated spectrum
        time_axis = np.arange(num_spectra) * tint_s
        
        # First we define the *baseband* frequency axis, e.g., from -12.5 MHz to +12.5 MHz
        # this just needs to spand the bandwidth of the celestial signal, it's simulated around
        # 0 MHz central frequency for computational efficiency (allows for coarser sampling)
        # and then shifted up to the actual bandpass range of the instrument / signal
        # NB: The frequency axis calculation MUST use the time resolution of the input data.
        freq_axis_baseband = np.fft.fftshift(np.fft.fftfreq(N_fft, d=time_res_s_raw))
        # The final "sky" (observed) frequency axis, e.g., from 787.5 MHz to 812.5 MHz
        freq_axis_sky = self.cfg.nu0.to_value(u.Hz) + freq_axis_baseband
        
        return I_t_nu, time_axis, freq_axis_sky
    
    def synthesise_dynamic_spectrum_2d(self, duration: u.Quantity, time_res: u.Quantity, rng=None):
        """
        Generates a full 2D dynamic spectrum I(t, ν) using the "True 2D Engine".

        This high-fidelity method avoids STFT approximations by calculating the
        exact spectrum for each time step. It is more computationally expensive
        but more physically accurate.

        Args:
            duration (u.Quantity): The total time duration of the simulation.
            time_res (u.Quantity): The desired time resolution of the output dynamic spectrum.
            rng: A numpy random generator instance.

        Returns:
            A tuple of (I_t_nu, time_axis, freq_axis_sky) containing the dynamic
            spectrum, time axis, and frequency axis.
        """
        if rng is None:
            rng = np.random.default_rng()

        # 1. Define the output time and frequency axes
        time_res_s = time_res.to_value(u.s)
        time_axis = np.arange(0, duration.to_value(u.s), time_res_s)
        num_time_steps = len(time_axis)
        freq_axis_sky = self.freqs_hz

        # 2. Get the intrinsic pulse spectrum, E_int(ν)
        E_int_nu = self._get_intrinsic_pulse_freq_domain(rng)

        # 3. Loop over time to calculate the time-variable IRF spectrum, R(t, ν)
        print("Running True 2D Engine (this may take a while)...")
        # Array to store the noise-free E-field spectrum at each time step
        E_signal_t_nu = np.zeros((num_time_steps, self.n_chan), dtype=np.complex128)

        for i in trange(num_time_steps, desc="Calculating spectra per time step"):
            dt_s = time_axis[i]
            # Calculate delays for this specific time step (includes velocity effects)
            tau_mw, tau_host, tau_cross = self._delays(dt_s)
            # Calculate the IRF spectrum R(ν) for this time step
            R_nu = self._irf_coherent_vs_freq(tau_mw, tau_host, tau_cross)
            # Convolve in the frequency domain (i.e., multiply)
            E_signal_t_nu[i, :] = R_nu * E_int_nu

        # 4. Add thermal noise in the frequency domain
        sefd_jy = self.cfg.instrument.get_sefd_jy()
        if sefd_jy is None:
            E_obs_t_nu = E_signal_t_nu
        else:
            # Variance of noise in a single freq channel of width dnu_hz
            n_pol = self.cfg.instrument.n_pol
            variance_component = (sefd_jy * self.dnu_hz) / (2.0 * n_pol)
            sigma_component = np.sqrt(variance_component)
            
            noise_shape = (num_time_steps, self.n_chan)
            noise = rng.normal(scale=sigma_component, size=noise_shape) + \
                    1j * rng.normal(scale=sigma_component, size=noise_shape)
            
            E_obs_t_nu = E_signal_t_nu + noise

        # 5. Calculate the final intensity
        I_t_nu = np.abs(E_obs_t_nu)**2
        
        return I_t_nu, time_axis, freq_axis_sky
    
    def synthesise_dynamic_spectrum_2d_parallel(self, duration: u.Quantity, time_res: u.Quantity, rng=None):
        """
        Generates a full 2D dynamic spectrum I(t, ν), based on synthesise_dynamic_spectrum_2d().
        This version is parallelized to use multiple CPU cores for significant speedup.
        """
        if rng is None:
            rng = np.random.default_rng()
        if Parallel is None:
            raise ModuleNotFoundError("joblib is required for parallel execution")

        time_res_s = time_res.to_value(u.s)
        time_axis = np.arange(0, duration.to_value(u.s), time_res_s)
        num_time_steps = len(time_axis)
        freq_axis_sky = self.freqs_hz

        E_int_nu = self._get_intrinsic_pulse_freq_domain(rng)

        # Define a helper function for the parallel loop
        def _calculate_spectrum_for_timestep(dt_s):
            # The _delays method is now very fast due to pre-computation
            total_delay = self._delays(dt_s)
            # The _irf_coherent_vs_freq method is the main workload here
            R_nu = self._irf_coherent_vs_freq(total_delay)
            return R_nu * E_int_nu

        print("Running Parallel True 2D Engine...")
        # Use joblib.Parallel to run the loop across multiple CPUs
        # n_jobs=-1 uses all available cores.
        E_signal_t_nu_list = Parallel(n_jobs=-1)(
            delayed(_calculate_spectrum_for_timestep)(dt_s) for dt_s in time_axis
        )

        E_signal_t_nu = np.array(E_signal_t_nu_list)

        sefd_jy = self.cfg.instrument.get_sefd_jy()
        if sefd_jy is None:
            E_obs_t_nu = E_signal_t_nu
        else:
            n_pol = self.cfg.instrument.n_pol
            variance_component = (sefd_jy * self.dnu_hz) / (2.0 * n_pol)
            sigma_component = np.sqrt(variance_component)
            
            noise_shape = (num_time_steps, self.n_chan)
            noise = rng.normal(scale=sigma_component, size=noise_shape) + \
                    1j * rng.normal(scale=sigma_component, size=noise_shape)
            
            E_obs_t_nu = E_signal_t_nu + noise

        I_t_nu = np.abs(E_obs_t_nu)**2
        
        return I_t_nu, time_axis, freq_axis_sky
    
    @staticmethod
    def acf(spectrum: np.ndarray, norm: str = "m2") -> tuple[np.ndarray, np.ndarray]:
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
        # or by the mean to get ACF where ACF(0) = m
        if norm == "m2":        # ACF(0) = m²  (paper Eq. 4.25)
            acf_func = covariance_func / (mean_intensity**2)
        elif norm == "m":       # ACF(0) = m   (often handier downstream)
            acf_func = np.sign(covariance_func) * np.sqrt(np.abs(covariance_func)) / mean_intensity
        else:
            raise ValueError("norm must be 'm' or 'm2'")
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
        lags_hz = lags * self.dnu_hz
        
        if corr.size < 5 or corr[0] < 0.1:
            return np.nan, np.nan

        def lorentzian_model(x, amplitude, hwhm, eps=1e-12):
            """A single Lorentzian model for fitting."""
            return amplitude / (1 + (x / (hwhm + eps))**2)

        # 1. Fit the broad component
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

        # 2. Subtract the broad model and fit the narrow residual
        broad_model_full = lorentzian_model(lags_hz, amp_broad, hwhm_broad)
        residual = corr - broad_model_full

        # The narrow component is in the residual, primarily at the center.
        narrow_mask = residual > 0
        if not np.any(narrow_mask):
            return hwhm_broad, np.nan # No significant narrow component found

        x_narrow, y_narrow = lags_hz[narrow_mask], residual[narrow_mask]

        # Initial guess for the narrow component's HWHM
        p0_narrow = (y_narrow[0], self.dnu_hz * 2)

        try:
            # Constrain the narrow HWHM to be less than the broad one
            popt_narrow, _ = curve_fit(lorentzian_model, x_narrow, y_narrow, p0=p0_narrow, bounds=([0, 0], [np.inf, hwhm_broad]))
            _, hwhm_narrow = popt_narrow
        except (RuntimeError, ValueError):
            # If narrow fit fails, we still have the broad component result
            return hwhm_broad, np.nan

        # Ensure the output is always (broad_hwhm, narrow_hwhm)
        return (hwhm_broad, hwhm_narrow) if hwhm_broad > hwhm_narrow else (hwhm_narrow, hwhm_broad)
    
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
            
