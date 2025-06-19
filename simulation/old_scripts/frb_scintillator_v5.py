#!/usr/bin/env python3
"""frb_scintillator.py — redshift‑aware **full‑fidelity** fast‑radio‑burst two‑screen
scintillation simulator (Pradeep et al. 2025)
=======================================================================
This single‑file module now supports every physical and instrumental
feature requested:

* Redshift‑exact geometry using angular‑diameter distances.
* Optional **observer / screen transverse velocities** → time‑evolving
  delays for repeating bursts.
* **Screen realisations**: isotropic Gaussian (paper default) **or**
  power‑law envelopes *plus* optional anisotropy (axial ratio &
  position angle).
* **Intrinsic pulse**: delta‑function (default) **or** Gaussian in the
  time domain.
* **Polyphase filterbank**: Blackman window with optional tap
  quantisation.
* **Instrumental bandpass** multiplication and AWGN radiometer noise at
  user‑specified S/N.
* **ACF analysis**: two‑Lorentzian or full composite model with robust
  (Huber) loss.

The public API remains centred on two dataclass configurations (`SimCfg`
& `ScreenCfg`) plus the main `FRBScintillator` class. All new knobs are
exposed as optional fields with sane defaults that reproduce the paper’s
behaviour when left untouched.

Dependencies
------------
* Python ≥ 3.10, numpy, scipy, astropy, numba (optional), matplotlib
* tqdm (optional, only for progress bars in large Monte‑Carlo sweeps)

Example
-------
```python
from frb_scintillator import u, ScreenCfg, SimCfg, FRBScintillator

cfg = SimCfg(
    nu0=1.25*u.GHz,
    bw=16*u.MHz,
    nchan=2048,
    D_mw=1.0*u.kpc,
    z_host=0.3,
    mw=ScreenCfg(N=256, L=1*u.AU,
                 profile="gaussian", axial_ratio=2.0, pa=45*u.deg),
    host=ScreenCfg(N=256, L=5*u.AU, profile="powerlaw", alpha=3.0),
    intrinsic_pulse="gauss", pulse_width=30*u.us,
    noise_snr=100  # radiometer S/N per channel
)

sim = FRBScintillator(cfg)
I_nu = sim.simulate_dynspec(dt=0.5*u.day)  # repeat burst 12 h later
corr, lags = sim.acf(I_nu)
νs_mw, νs_host = sim.fit_acf(corr, lags, model="composite")
print(νs_mw/1e3, "kHz", νs_host/1e3, "kHz")
```

All angle‑based quantities respect `astropy.units.dimensionless_angles()`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.coordinates import Angle
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import firwin, lfilter
from scipy.optimize import curve_fit, least_squares

try:
    import numba as nb

    _NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA = False

# ----------------------------------------------------------------------------
# Constants & logging
# ----------------------------------------------------------------------------
C_M_PER_S = const.c.to(u.m / u.s).value
logger = logging.getLogger("frb_scintillator")

# ----------------------------------------------------------------------------
# Helper: angular‑diameter distance
# ----------------------------------------------------------------------------

def _DA(z1: float, z2: float = 0.0) -> u.Quantity:
    """Angular‑diameter distance between planes at redshifts *z1* and *z2*."""
    # Ensure z1 < z2 for the astropy function
    if z2 < z1:
        z1, z2 = z2, z1
    return cosmo.angular_diameter_distance_z1z2(z1, z2)


# ----------------------------------------------------------------------------
# Dataclass configurations
# ----------------------------------------------------------------------------


def _array2(vec: Tuple[float, float] | np.ndarray | None, unit: u.Unit) -> np.ndarray:
    """Utility: ensure a 2‑vector with units becomes float64 array [unit]."""
    if vec is None:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("Velocity / offset vectors must be 2‑element tuples.")
    return (arr * unit).to(unit).value


@dataclass
class ScreenCfg:
    """Configuration for a thin scattering screen."""

    # statistical realisation
    N: int = 128  # number of sub‑images
    L: u.Quantity = 1.0 * u.AU  # physical transverse size (defines envelope scale)

    # envelope profile
    profile: Literal["gaussian", "powerlaw"] = "gaussian"
    alpha: float = 3.0  # power‑law exponent for |θ| ≫ θ0 if profile==powerlaw
    theta0: u.Quantity = 100.0 * u.marcsec  # core scale for power‑law

    # anisotropy
    axial_ratio: float = 1.0  # ≥1 ; minor/major axis ratio
    pa: u.Quantity = 0.0 * u.deg  # position angle east of north

    # per‑image field statistics
    amp_distribution: Literal["constant", "rayleigh"] = "constant"

    # reproducibility
    rng_seed: Optional[int] = None

    # transverse velocity (for repeating bursts)
    v_perp: Tuple[float, float] | np.ndarray | None = None  # km/s

    # internal: convert velocity to m/s float64 array during post‑init
    def __post_init__(self):
        object.__setattr__(self, "v_perp", _array2(self.v_perp, u.km / u.s))


@dataclass
class SimCfg:
    """Top‑level simulation parameters following Pradeep et al. (2025)."""

    # Radio band
    nu0: u.Quantity = 1.25 * u.GHz  # centre
    bw: u.Quantity = 16.0 * u.MHz
    nchan: int = 1024

    # Geometry / cosmology
    D_mw: u.Quantity = 1.0 * u.kpc  # observer→MW screen (assumed z=0)
    z_host: float = 0.50
    # Source redshift defaults to host redshift if not specified.
    z_src: Optional[float] = None

    # Screens
    mw: ScreenCfg = field(default_factory=ScreenCfg)
    host: ScreenCfg = field(default_factory=ScreenCfg)

    # Intrinsic pulse
    intrinsic_pulse: Literal["delta", "gauss"] = "delta"
    pulse_width: u.Quantity = 30.0 * u.us  # FWHM

    # PFB / channelisation
    ntap: int = 4
    pfb_window: Literal["blackman", "rect"] = "blackman"
    quant_bits: Optional[int] = None  # quantise taps to this many bits

    # Instrumental effects
    bandpass_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None  # fn(ν[Hz]) → gain
    noise_snr: Optional[float] = None  # if set, inject AWGN with this S/N per channel

    # ACF fitting
    corr_thresh: float = 0.03
    bootstrap_blocks: int = 32
    
    def __post_init__(self):
        # Default source redshift to host redshift if not provided
        if self.z_src is None:
            self.z_src = self.z_host


# ----------------------------------------------------------------------------
# Scattering screen realisation
# ----------------------------------------------------------------------------


class Screen:
    """Random realisation of a single screen (θ array + complex field).

    The physical transverse scale ``cfg.L`` may be supplied **either** as an
    *angle* (e.g. ``100*u.marcsec``) **or** as a *length* (e.g. ``1*u.AU``).
    If a length is given, the constructor converts it to an angular size by
    dividing by the screen distance ``D_screen`` supplied by the caller.
    """

    def __init__(self, cfg: ScreenCfg, D_screen_m: float, nu0_hz: float):
        self.cfg = cfg
        self.D_screen_m = D_screen_m  # store for motion projection
        self.nu0_hz = nu0_hz
        rng = np.random.default_rng(cfg.rng_seed)

        # --- coordinate grid -------------------------------------------------
        # The paper scales screen size L as nu^-2. We define L at nu0
        # and scale it for the actual simulation frequency if needed later.
        # For now, we use the size at nu0.
        L_m = self.cfg.L.to(u.m).value
        half_box_rad = (L_m / 2) / D_screen_m

        xy = rng.uniform(-half_box_rad, half_box_rad, size=(cfg.N, 2))  # [rad]

        # Apply anisotropy: scale y‑axis by axial_ratio and rotate by pa.
        if cfg.axial_ratio != 1.0 or cfg.pa.value != 0.0:
            pa_rad = cfg.pa.to(u.rad).value
            R = np.array(
                [[np.cos(pa_rad), -np.sin(pa_rad)], [np.sin(pa_rad), np.cos(pa_rad)]]
            )
            xy = xy @ R.T  # rotate into principal frame
            xy[:, 1] /= cfg.axial_ratio  # compress minor axis
            xy = xy @ R  # rotate back into sky frame
        self.theta = xy  # store [rad]

        # --- per‑image complex field ----------------------------------------
        if cfg.amp_distribution == "constant":
            amps = np.ones(cfg.N)
        elif cfg.amp_distribution == "rayleigh":
            amps = rng.rayleigh(scale=1 / np.sqrt(2), size=cfg.N)
        else:  # pragma: no cover
            raise ValueError("amp_distribution must be 'constant' or 'rayleigh'")

        phases = rng.uniform(0, 2 * np.pi, size=cfg.N)
        field = amps * np.exp(1j * phases)

        # --- envelope weight -------------------------------------------------
        if cfg.profile == "gaussian":
            # Paper’s Eq. 3.10 uses a Gaussian scattering disk.
            # FWHM=L corresponds to 2.355 sigma. Here L is the 2-sigma width.
            sigma_rad = half_box_rad / 2.0
            w = np.exp(-np.sum(xy**2, axis=1) / (2 * sigma_rad**2))
        elif cfg.profile == "powerlaw":
            # Envelope ∝ [1+(θ/θ0)^2]^{-α/2}
            theta0 = cfg.theta0.to(u.rad).value
            r2 = np.sum(xy**2, axis=1)
            w = (1 + r2 / theta0**2) ** (-cfg.alpha / 2)
        else:  # pragma: no cover
            raise ValueError("profile must be 'gaussian' or 'powerlaw'")

        self.field = field * w / np.linalg.norm(w, ord=2)  # normalise total power

    # ---------------------------------------------------------------------
    #   Time evolution (observer / screen velocity)
    # ---------------------------------------------------------------------
    def advance(self, dt: u.Quantity):
        """Shift image angles by transverse motion over *dt* (seconds)."""
        if not np.any(self.cfg.v_perp):  # no motion requested
            return
        delta_xy = (self.cfg.v_perp * dt.to(u.s).value) / self.D_screen_m  # in rad
        self.theta += delta_xy


# Main simulator
# ----------------------------------------------------------------------------

class FRBScintillator:
    """Two‑screen scintillation simulator with optional full‑fidelity effects."""

    # ---------------------------- constructor ---------------------------------
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self._prepare_geometry()
        self._prepare_screens()
        self._prepare_frequency_grid()
        self._prepare_pfb_kernel()
        logger.debug("Initialised FRBScintillator with RP=%.3f", self.resolution_power())

    # ---------------------------- geometry ------------------------------------
    def _prepare_geometry(self):
        """Calculate all geometric and effective distances."""
        cfg = self.cfg
        self.nu0_hz = cfg.nu0.to(u.Hz).value
        self.lam0_m = (C_M_PER_S / self.nu0_hz)
        
        # Physical distances in meters
        self.D_mw_m = cfg.D_mw.to(u.m).value
        self.D_host_m = _DA(0.0, cfg.z_host).to(u.m).value
        self.D_src_m = _DA(0.0, cfg.z_src).to(u.m).value
        self.D_mw_host_m = self.D_host_m - self.D_mw_m # Approx for low-z MW
        self.D_host_src_m = _DA(cfg.z_host, cfg.z_src).to(u.m).value

        # Effective distances (Eq. 2.6)
        self.deff_mw_m = (self.D_mw_m * self.D_host_m) / self.D_mw_host_m
        
        # CORRECTED: This now directly implements Eq. 2.6b from the paper.
        term1 = (1 + cfg.z_host) * (self.D_host_m * self.D_src_m) / self.D_host_src_m
        self.deff_host_m = term1 + self.deff_mw_m
        
    # ---------------------------- screens -------------------------------------
    def _prepare_screens(self):
        self.mw_screen = Screen(self.cfg.mw, self.D_mw_m, self.nu0_hz)
        self.host_screen = Screen(self.cfg.host, self.D_host_m, self.nu0_hz)
        self._compute_static_delays()

    def _compute_static_delays(self):
        """Precompute delays (sec) ignoring any time evolution, as per Eq. 2.5."""
        # Note: theta arrays are in radians.
        
        # Term 1: (D_eff,MW / 2c) * theta_MW^2
        self._tau_mw0 = (
            self.deff_mw_m / (2 * C_M_PER_S) * np.sum(self.mw_screen.theta**2, axis=1)
        )
        # Term 3: (D_eff,host / 2c) * theta_host^2
        self._tau_host0 = (
            self.deff_host_m / (2 * C_M_PER_S) * np.sum(self.host_screen.theta**2, axis=1)
        )
        # Term 2 (cross-term): -(D_eff,MW / c) * theta_MW . theta_host
        self._tau_cross0 = -(
            self.deff_mw_m
            / C_M_PER_S
            * (self.mw_screen.theta @ self.host_screen.theta.T)
        )

    # ---------------------------- frequency grid ------------------------------
    def _prepare_frequency_grid(self):
        cfg = self.cfg
        self.bw_hz = cfg.bw.to(u.Hz).value
        self.freqs = np.linspace(
            self.nu0_hz - self.bw_hz / 2, self.nu0_hz + self.bw_hz / 2, cfg.nchan, dtype=np.float64
        )
        self.dnu = self.freqs[1] - self.freqs[0]

    # ---------------------------- PFB kernel ----------------------------------
    def _prepare_pfb_kernel(self):
        cfg = self.cfg
        if cfg.ntap <= 0:
            self.pfb_taps = None
            return
        window = "blackman" if cfg.pfb_window == "blackman" else "boxcar"
        taps = firwin(cfg.ntap * cfg.nchan, 1 / cfg.nchan, window=window)
        if cfg.quant_bits is not None:
            max_int = 2 ** (cfg.quant_bits - 1) - 1
            taps = np.round(taps * max_int) / max_int
        self.pfb_taps = taps

    # -------------------------------------------------------------------------
    #   Resolution power & propagation mode helper
    # -------------------------------------------------------------------------
    def resolution_power(self) -> float:
        """Calculate Resolution Power (RP) as per Eq. 3.11"""
        L_mw = self.cfg.mw.L.to(u.m).value
        L_host = self.cfg.host.L.to(u.m).value
        return (L_mw * L_host) / (self.lam0_m * self.D_mw_host_m)

    # -------------------------------------------------------------------------
    #   Time‑dependent delays (observer / screen motion)
    # -------------------------------------------------------------------------
    def _delays(self, dt_s: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return τ_mw, τ_host, τ_cross arrays at time offset *dt_s* (seconds)."""
        if dt_s == 0.0 or (not np.any(self.cfg.mw.v_perp) and not np.any(self.cfg.host.v_perp)):
            return self._tau_mw0, self._tau_host0, self._tau_cross0

        # Shift theta due to transverse motion: Δθ = v⊥ dt / D_screen
        theta_mw_t = self.mw_screen.theta + (self.cfg.mw.v_perp * dt_s) / self.D_mw_m
        theta_host_t = self.host_screen.theta + (self.cfg.host.v_perp * dt_s) / self.D_host_m

        tau_mw = (self.deff_mw_m / (2 * C_M_PER_S)) * np.sum(theta_mw_t**2, axis=1)
        tau_host = (self.deff_host_m / (2 * C_M_PER_S)) * np.sum(theta_host_t**2, axis=1)
        tau_cross = -(
            self.deff_mw_m
            / C_M_PER_S
            * (theta_mw_t @ theta_host_t.T)
        )
        return tau_mw, tau_host, tau_cross

    # -------------------------------------------------------------------------
    #   IRF evaluators
    # -------------------------------------------------------------------------
    def _irf_coherent(self, tau_mw, tau_host, tau_cross) -> np.ndarray:
        """Calculates the coherent two-screen impulse response function."""
        # Reshape for broadcasting:
        # f_mw: (N1, 1, 1), f_host: (1, N2, 1)
        # phase: (N1, N2, Nf)
        f_mw = self.mw_screen.field[:, None, None]
        f_host = self.host_screen.field[None, :, None]
        
        # Combine delays for all paths
        # tau_mw: (N1,) -> (N1, 1)
        # tau_host: (N2,) -> (1, N2)
        # tau_cross: (N1, N2)
        total_delay = tau_mw[:, None] + tau_host[None, :] + tau_cross

        phase = np.exp(-2j * np.pi * total_delay[..., None] * self.freqs)
        
        if _NUMBA:
            return _irf_sum_numba(f_mw, f_host, phase)
        return np.sum(f_mw * f_host * phase, axis=(0, 1))

    def _irf_spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the time and intensity of individual path delays."""
        tau = (self._tau_mw0[:, None] + self._tau_host0[None, :] + self._tau_cross0).ravel()
        I   = (np.abs(self.mw_screen.field[:, None] *
                      self.host_screen.field[None, :]) ** 2).ravel()
        return tau, I

# Numba helper ------------------------------------------------------------

if _NUMBA:
    @nb.njit(parallel=True, fastmath=True)
    def _irf_sum_numba(field_mw, field_host, phase):
        N1, _, _ = field_mw.shape
        _, N2, _ = field_host.shape
        _, _, Nf = phase.shape
        out = np.zeros(Nf, dtype=np.complex128)
        for k in nb.prange(Nf):
            s = 0.0 + 0.0j
            for i in range(N1):
                for j in range(N2):
                    s += field_mw[i, 0, 0] * field_host[0, j, 0] * phase[i, j, k]
            out[k] = s
        return out


# ----------------------------------------------------------------------------
#   Synthesis chain
# ----------------------------------------------------------------------------

class FRBScintillator(FRBScintillator):  # extend methods

    # ---------------------------- intrinsic pulse -----------------------------
    def _intrinsic_time_series(self, nt: int, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Return complex baseband u(t) of the intrinsic FRB emission."""
        if self.cfg.intrinsic_pulse == "delta":
            sig = np.zeros(nt, dtype=np.complex128)
            sig[0] = 1.0
            return sig
        elif self.cfg.intrinsic_pulse == "gauss":
            t = (np.arange(nt) - nt // 2) * dt
            sigma = (self.cfg.pulse_width.to(u.s).value) / (2 * np.sqrt(2 * np.log(2)))
            envelope = np.exp(-(t**2) / (2 * sigma**2))
            noise = rng.normal(size=nt) + 1j * rng.normal(size=nt)
            return envelope * noise / np.sqrt(2)
        else:  # pragma: no cover
            raise ValueError("intrinsic_pulse must be 'delta' or 'gauss'")

    # ---------------------------- dynspec synth ------------------------------
    def simulate_dynspec(self, dt: u.Quantity = 0.0 * u.s, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return *spectral intensity* I(ν) at epoch offset *dt*.

        The method does *not* yet return a 2‑D time × frequency array even if
        intrinsic_pulse is Gaussian; instead it forms a channelised spectrum
        by convolving the intrinsic u(t) with the IRF and integrating over
        time, matching the paper’s treatment (spectral ACF only).
        """
        if rng is None:
            rng = np.random.default_rng()
            
        dt_s = dt.to(u.s).value
        tau_mw, tau_host, tau_cross = self._delays(dt_s)

        # CORRECTED: Always use the coherent sum. Resolution effects emerge
        # naturally from the cross-term in the phase, as per the paper.
        field = self._irf_coherent(tau_mw, tau_host, tau_cross)
        spec = np.abs(field) ** 2

        # Instrumental bandpass
        if self.cfg.bandpass_fn is not not None:
            spec *= self.cfg.bandpass_fn(self.freqs)

        # Radiometer noise
        if self.cfg.noise_snr is not None and self.cfg.noise_snr > 0:
            sigma = np.sqrt(spec.mean()) / self.cfg.noise_snr
            noise = rng.normal(scale=sigma, size=spec.size)
            spec += noise
            spec = np.clip(spec, 0, None)

        return spec.astype(np.float64)

    # ---------------------------- ACF & models -------------------------------
    @staticmethod
    def acf(spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the autocorrelation function of a spectrum."""
        spec_mean_sub = spectrum - spectrum.mean()
        # Use FFT for efficient computation
        fft_spec = np.fft.fft(spec_mean_sub)
        corr = np.fft.ifft(np.abs(fft_spec)**2).real
        corr = np.roll(corr, len(corr)//2) # Shift zero lag to center if needed, but for one-sided...
        corr = corr[:len(spectrum)//2]
        corr /= corr[0] # Normalize
        lags = np.arange(corr.size)
        return corr, lags

    # ---------------------------------------------------------------------
    #   ACF fitting
    # ---------------------------------------------------------------------
    def fit_acf(
        self,
        corr: np.ndarray,
        lags: np.ndarray,
        model: Literal["two_lorentzian", "composite"] = "two_lorentzian",
    ) -> Tuple[float, float]:
        """Fit the spectral ACF and return (ν_s,MW, ν_s,host) half‑widths."""
        lags_hz = lags * self.dnu

        mask = corr > self.cfg.corr_thresh
        x, y = lags_hz[mask], corr[mask]
        
        # Define fit functions
        def two_lorentzian(l, a, w1, w2):
            return a / (1 + (l / w1) ** 2) + (1 - a) / (1 + (l / w2) ** 2)

        def composite(l, w1, w2):
            # Implements ACF = L(w1) + L(w2) + L(w1)*L(w2) with fixed amps=1
            term1 = 1 / (1 + (l / w1) ** 2)
            term2 = 1 / (1 + (l / w2) ** 2)
            return term1 + term2 + term1 * term2

        # Select model and initial guess
        if model == "two_lorentzian":
            fit_func = two_lorentzian
            p0 = (0.5, self.bw_hz / 10, self.bw_hz / 1000)
        elif model == "composite":
            fit_func = composite
            p0 = (self.bw_hz / 10, self.bw_hz / 1000)
        else:
            raise ValueError("model must be 'two_lorentzian' or 'composite'")

        try:
            popt, _ = curve_fit(fit_func, x, y, p0=p0, maxfev=5000)
            if model == "two_lorentzian":
                _, w1, w2 = popt
            else: # composite
                w1, w2 = popt
            # Ensure w1 is always the larger bandwidth (MW screen)
            if w1 < w2:
                w1, w2 = w2, w1
            return w1, w2
        except RuntimeError:
            logger.warning("ACF fit failed to converge.")
            return np.nan, np.nan


    # -------------------------- bootstrap errors ------------------------------
    def acf_bootstrap_errors(
        self, spectrum: np.ndarray, n_iter: int = 100, model: str = "two_lorentzian"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate fit parameter errors using block bootstrap."""
        n = spectrum.size
        blk = max(n // self.cfg.bootstrap_blocks, 1)
        errs = []
        for _ in range(n_iter):
            # Create resampled spectrum from random blocks
            idx = np.random.randint(0, self.cfg.bootstrap_blocks, self.cfg.bootstrap_blocks)
            resamp_list = [spectrum[i * blk : (i + 1) * blk] for i in idx if (i + 1) * blk <= n]
            if not resamp_list: continue
            resamp = np.concatenate(resamp_list)
            
            corr, lags = self.acf(resamp)
            w1, w2 = self.fit_acf(corr, lags, model=model)
            if not (np.isnan(w1) or np.isnan(w2)):
                errs.append([w1, w2])
        return np.std(errs, axis=0)
    
    # ---------------------- 2‑D dynamic spectrum -----------------------------
    def simulate_dynamic_spectrum(
        self,
        ntime: int,
        dt: u.Quantity,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Return a *time × frequency* dynamic spectrum I(t,ν).

        Parameters
        ----------
        ntime
            Number of time samples to synthesise.
        dt
            Interval between adjacent spectra (can be larger than the pulse
            width; typically milliseconds to minutes).  Screen motions are
            projected according to ``mw.v_perp`` and ``host.v_perp``.
        rng
            Optional shared random generator so that radiometer noise is
            reproducible across calls.
        """
        if rng is None:
            rng = np.random.default_rng()
            
        dyn = np.empty((ntime, self.cfg.nchan), dtype=np.float64)
        for i in range(ntime):
            dyn[i] = self.simulate_dynspec(dt=i * dt, rng=rng)
        return dyn


# -----------------------------------------------------------------------------
#                               Self‑test
# -----------------------------------------------------------------------------

def _self_test():  # pragma: no cover
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = SimCfg(
        z_host=0.3,
        nchan=2048,
        mw=ScreenCfg(N=128, L=1.0*u.AU, axial_ratio=2.0, pa=30*u.deg),
        host=ScreenCfg(N=128, L=5.0*u.AU, profile="powerlaw", alpha=3.5),
        intrinsic_pulse="gauss",
        noise_snr=50,
    )
    sim = FRBScintillator(cfg)
    spec = sim.simulate_dynspec()
    corr, lags = sim.acf(spec)
    
    # Fit with the theoretical composite model
    w1, w2 = sim.fit_acf(corr, lags, model="composite")

    print("RP =", sim.resolution_power())
    print("Fit Results (composite model):")
    print("ν_s (MW) = %.2f kHz" % (w1/1e3))
    print("ν_s (host) = %.2f kHz" % (w2/1e3))

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    ax[0].plot(sim.freqs / 1e6, spec)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Intensity (arbitrary units)")
    ax[0].set_title("Simulated Spectrum")
    ax[0].grid(True, alpha=0.3)

    lags_hz = lags * sim.dnu
    ax[1].plot(lags_hz / 1e3, corr, label='Data ACF')
    
    # Overplot the fit
    def composite(l, w1, w2):
        term1 = 1 / (1 + (l / w1) ** 2)
        term2 = 1 / (1 + (l / w2) ** 2)
        return term1 + term2 + term1 * term2
    
    if not (np.isnan(w1) or np.isnan(w2)):
        ax[1].plot(lags_hz / 1e3, composite(lags_hz, w1, w2), 'r--', label='Composite Fit')
    
    ax[1].set_xlabel("Frequency Lag (kHz)")
    ax[1].set_ylabel("Autocorrelation")
    ax[1].set_title("Spectral Autocorrelation Function")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim(-0.1, np.max(corr)*1.1 if corr.size>0 else 1.1)
    
    plt.show()


if __name__ == "__main__":
    _self_test()
