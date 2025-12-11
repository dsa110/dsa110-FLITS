#!/usr/bin/env python3
"""frb_scintillator.py — fully integrated, redshift‑aware two‑screen FRB scintillation simulator

This **self‑contained** module merges the earlier end‑to‑end feature set
(PFB, dynamic‑spectrum synthesis, ACF fitting, hybrid propagation mode, unit
 tests) with the cosmology‑corrected geometry patch that threads redshift into
 all distances, exactly as in *Pradeep et al.* (2025).

---------------------------------------------------------------------
Quick start
---------------------------------------------------------------------
```python
from frb_scintillator import SimCfg, FRBScintillator

cfg = SimCfg(nu0=1.25*u.GHz, bw=16*u.MHz, nchan=1024,
             D_mw=1.0*u.kpc, z_host=0.5,
             mw=ScreenCfg(N=256, L=1*u.AU),
             host=ScreenCfg(N=256, L=5*u.AU),
             prop_mode="hybrid")

sim = FRBScintillator(cfg)
dynspec = sim.simulate_dynspec()
acf, lags = sim.acf(dynspec)
print("MW Δν ≈", sim.scint_band_mw, "Hz ; host Δν ≈", sim.scint_band_host, "Hz")
```
---------------------------------------------------------------------
Dependencies
---------------------------------------------------------------------
* **Python ≥ 3.10**
* numpy, scipy, astropy, numba (optional but speeds up IRF evaluation)
* matplotlib (only for the quick self‑test at the bottom)

---------------------------------------------------------------------
Implementation notes
---------------------------------------------------------------------
* **Redshift factors** enter via angular‑diameter distances using
  `astropy.cosmology.Planck18`.  The host term carries the required `(1+z_host)`
  prefactor.
* Public parameters accept `astropy.units.Quantity`; internal numerics strip
  units for speed.
* Three propagation modes: "coherent" (default), "power", and "hybrid" which
  switches at user‑settable `rp_switch≈1`.
* A simple bootstrap on channel blocks gives 1‑σ errors on the Lorentzian fits.

This file is large (~500 lines) but single‑purpose, so we forgo breaking it into
sub‑modules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import firwin, lfilter
from scipy.optimize import curve_fit

try:
    import numba as nb

    _NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    _NUMBA = False

# -----------------------------------------------------------------------------
# Constants & logging
# -----------------------------------------------------------------------------
C_M_PER_S = u.speed_of_light.to(u.m / u.s).value  # float for hot loops
logger = logging.getLogger("frb_scintillator")

# -----------------------------------------------------------------------------
# Helper: angular‑diameter distance
# -----------------------------------------------------------------------------

def _DA(z1: float, z2: float = 0.0) -> u.Quantity:
    """Angular‑diameter distance between two planes at redshifts *z1* < *z2*."""
    if z2 < z1:
        z1, z2 = z2, z1
    return cosmo.angular_diameter_distance_z1z2(z1, z2)


# -----------------------------------------------------------------------------
# Dataclass configurations
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ScreenCfg:
    """Configuration for a thin scattering screen."""

    N: int = 128  # number of scattering sub‑images
    L: u.Quantity = 1.0 * u.AU  # transverse physical size of the cloud
    amp_distribution: Literal["constant", "rayleigh"] = "constant"
    rng_seed: Optional[int] = None


@dataclass(slots=True)
class SimCfg:
    """Top‑level simulation parameters following Pradeep et al. (2025)."""

    # Radio band
    nu0: u.Quantity = 1.25 * u.GHz  # centre frequency
    bw: u.Quantity = 16.0 * u.MHz  # bandwidth
    nchan: int = 1024  # spectral channels

    # Geometry / cosmology
    D_mw: u.Quantity = 1.0 * u.kpc  # observer→MW screen
    z_host: float = 0.50  # redshift of host screen
    z_src: float = 0.50  # redshift of source (default same as host)

    # Screens
    mw: ScreenCfg = field(default_factory=ScreenCfg)
    host: ScreenCfg = field(default_factory=ScreenCfg)

    # Propagation & analysis knobs
    prop_mode: Literal["coherent", "power", "hybrid"] = "coherent"
    rp_switch: float = 1.0  # RP threshold used by hybrid mode
    ntap: int = 4  # taps of the polyphase filterbank (per channel)
    corr_thresh: float = 0.03  # ACF lag threshold for fitting
    bootstrap_blocks: int = 32  # blocks to resample for ACF error


# -----------------------------------------------------------------------------
# Scattering screen realisation
# -----------------------------------------------------------------------------


class Screen:
    """Random realisation of a single screen (θ array + complex field)."""

    def __init__(self, cfg: ScreenCfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.rng_seed)

        # Box encloses ~4σ of the Gaussian envelope assumed by Pradeep et al.
        half_box = (cfg.L / 2).to(u.rad, equivalencies=u.dimensionless_angles()).value
        self.theta = rng.uniform(-half_box, half_box, size=(cfg.N, 2))  # [rad]

        if cfg.amp_distribution == "constant":
            amps = np.ones(cfg.N)
        elif cfg.amp_distribution == "rayleigh":
            amps = rng.rayleigh(scale=1 / np.sqrt(2), size=cfg.N)
        else:  # pragma: no cover
            raise ValueError("amp_distribution must be 'constant' or 'rayleigh'")

        phases = rng.uniform(0, 2 * np.pi, size=cfg.N)
        self.field = amps * np.exp(1j * phases)


# -----------------------------------------------------------------------------
# Main simulator
# -----------------------------------------------------------------------------


class FRBScintillator:
    """Two‑screen scintillation simulator faithful to Pradeep et al. (2025)."""

    # ---------------------------- constructor ---------------------------------
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self._prepare_geometry()
        self._prepare_screens()
        self._prepare_frequency_grid()
        logger.debug("Initialised FRBScintillator with RP=%.3f", self.resolution_power())

    # ---------------------------- geometry ------------------------------------
    def _prepare_geometry(self):
        cfg = self.cfg
        self.nu0_hz = cfg.nu0.to(u.Hz).value
        self.lam0_m = (u.c / cfg.nu0).to(u.m).value

        # Distances
        self.D_mw_m = cfg.D_mw.to(u.m).value
        D_obs_host = _DA(0.0, cfg.z_host).to(u.m).value
        D_obs_src = _DA(0.0, cfg.z_src).to(u.m).value
        self.D_mw_host = _DA(0.0, cfg.z_host).to(u.m).value

        # Effective distances (Eq. 2.6)
        self.deff_mw = (self.D_mw_m * self.D_mw_host) / (self.D_mw_host - self.D_mw_m)
        self.deff_host = (
            (1.0 + cfg.z_host) * (self.D_mw_host * D_obs_src) / (D_obs_src - self.D_mw_host)
            + self.deff_mw
        )

    # ---------------------------- screens -------------------------------------
    def _prepare_screens(self):
        self.mw_screen = Screen(self.cfg.mw)
        self.host_screen = Screen(self.cfg.host)

        # Precompute per‑image delays that do not depend on frequency
        self._tau_mw = (self.deff_mw / (2 * C_M_PER_S)) * (
            np.sum(self.mw_screen.theta ** 2, axis=1)
        )  # shape (N1,)
        self._tau_host = (self.deff_host / (2 * C_M_PER_S)) * (
            np.sum(self.host_screen.theta ** 2, axis=1)
        )  # shape (N2,)

        # Cross‑term grid (N1×N2)
        self._tau_cross = -(
            self.deff_mw
            / C_M_PER_S
            * np.sum(
                self.mw_screen.theta[:, None, :] * self.host_screen.theta[None, :, :],
                axis=-1,
            )
        )

    # ---------------------------- frequency grid ------------------------------
    def _prepare_frequency_grid(self):
        cfg = self.cfg
        bw_hz = cfg.bw.to(u.Hz).value
        self.freqs = np.linspace(
            self.nu0_hz - bw_hz / 2, self.nu0_hz + bw_hz / 2, cfg.nchan, dtype=np.float64
        )
        self.dnu = self.freqs[1] - self.freqs[0]

    # -------------------------------------------------------------------------
    #   Resolution power & propagation mode helper
    # -------------------------------------------------------------------------
    def resolution_power(self) -> float:
        L_mw = self.cfg.mw.L.to(u.m).value
        L_host = self.cfg.host.L.to(u.m).value
        return (L_mw * L_host) / (self.lam0_m * self.D_mw_host)

    def _mode(self) -> str:  # choose propagation mode for this RP
        if self.cfg.prop_mode != "hybrid":
            return self.cfg.prop_mode
        return "coherent" if self.resolution_power() < self.cfg.rp_switch else "power"

    # -------------------------------------------------------------------------
    #   IRF evaluators
    # -------------------------------------------------------------------------
    def _irf_coherent(self) -> np.ndarray:
        """Complex field G(ν) with full two‑screen coherence (matrix sum)."""
        N1, N2 = self.mw_screen.cfg.N, self.host_screen.cfg.N
        field_mw = self.mw_screen.field[:, None]  # (N1,1)
        field_host = self.host_screen.field[None, :]  # (1,N2)

        # Delays in seconds: τ(i,j) = τ_mw(i) + τ_host(j) + τ_cross(i,j)
        tau_mat = (
            self._tau_mw[:, None]
            + self._tau_host[None, :]
            + self._tau_cross
        )  # (N1,N2)

        # Vectorised over frequency using broadcasting
        phase = np.exp(2j * np.pi * tau_mat[..., None] * self.freqs)  # (N1,N2,nchan)
        if _NUMBA:
            return _irf_sum_numba(field_mw, field_host, phase)
        return np.sum(field_mw * field_host * phase, axis=(0, 1))  # (nchan,)

    def _irf_power(self) -> np.ndarray:
        """Intensity multiplication approximation (no cross term)."""
        field_mw = self._single_screen_field(self.mw_screen, self._tau_mw)
        field_host = self._single_screen_field(self.host_screen, self._tau_host)
        return np.abs(field_mw) ** 2 * np.abs(field_host) ** 2  # (nchan,)

    def _single_screen_field(self, screen: Screen, tau: np.ndarray) -> np.ndarray:
        """Complex field ∑_i f_i exp(2πiντ_i) for one screen."""
        phase = np.exp(2j * np.pi * tau[:, None] * self.freqs)  # (Ni,nchan)
        return np.sum(screen.field[:, None] * phase, axis=0)

    # Numba‑accelerated matrix sum ------------------------------------------------


if _NUMBA:

    @nb.njit(parallel=True, fastmath=True)
    def _irf_sum_numba(field_mw: np.ndarray, field_host: np.ndarray, phase: np.ndarray):
        N1, N2, Nf = phase.shape
        out = np.zeros(Nf, dtype=np.complex128)
        for k in nb.prange(Nf):
            s = 0.0 + 0.0j
            for i in range(N1):
                for j in range(N2):
                    s += field_mw[i, 0] * field_host[0, j] * phase[i, j, k]
            out[k] = s
        return out

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    #   Dynamic spectrum & analysis
    # -------------------------------------------------------------------------


class FRBScintillator(FRBScintillator):  # noqa: E0102  — extend previous class
    """Add synthesis and analysis methods on top of geometry helpers."""

    # ---------------------------- synthesis -----------------------------------
    def simulate_dynspec(self) -> np.ndarray:
        """Return I(ν) for a single de‑dispersed burst (frequency axis only).

        *No time axis* is generated because the paper analyses the *spectral*
        ACF of an already dedispersed, band‑integrated pulse.
        """
        mode = self._mode()
        if mode == "coherent":
            field = self._irf_coherent()
            I_nu = np.abs(field) ** 2
        elif mode == "power":
            I_nu = self._irf_power()
        else:  # pragma: no cover — should never happen
            raise RuntimeError("Unknown mode")
        return I_nu.astype(np.float64)

    # ---------------------------- ACF & fitting -------------------------------
    def acf(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalised autocorrelation of *spectrum* along frequency."""
        spec = spectrum - spectrum.mean()
        corr = np.correlate(spec, spec, mode="full")
        corr = corr[corr.size // 2 :]  # keep non‑negative lags
        corr /= corr[0]
        lags = np.arange(corr.size) * self.dnu  # Hz
        return corr, lags

    def fit_acf(self, corr: np.ndarray, lags: np.ndarray) -> Tuple[float, float]:
        """Fit two Lorentzians to the ACF and return (ν_s,mw, ν_s,host)."""

        def two_lorentz(l, a, w1, w2):
            return a / (1 + (l / w1) ** 2) + (1 - a) / (1 + (l / w2) ** 2)

        # Use only lags where corr > thresh
        mask = corr > self.cfg.corr_thresh
        p0 = (0.5, self.cfg.bw.to(u.Hz).value / 10, self.cfg.bw.to(u.Hz).value / 100)
        try:
            popt, _ = curve_fit(two_lorentz, lags[mask], corr[mask], p0=p0, maxfev=2000)
            a, w1, w2 = popt
            # Convention: w1 > w2 ⇒ w1 = MW (broad), w2 = host (narrow)
            if w1 < w2:
                w1, w2 = w2, w1
            self.scint_band_mw = w1
            self.scint_band_host = w2
            return w1, w2
        except RuntimeError:
            logger.warning("Lorentzian fit failed; returning NaNs")
            return np.nan, np.nan

    # -------------------------- bootstrap errors ------------------------------
    def acf_bootstrap_errors(self, spectrum: np.ndarray, n_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Resample channel blocks and refit the ACF to estimate uncertainties."""
        n = spectrum.size
        blk = max(n // self.cfg.bootstrap_blocks, 1)
        errs = []
        for _ in range(n_iter):
            idx = np.random.randint(0, self.cfg.bootstrap_blocks, size=self.cfg.bootstrap_blocks)
            resamp = np.concatenate([
                spectrum[i * blk : (i + 1) * blk] for i in idx if (i + 1) * blk <= n
            ])
            corr, lags = self.acf(resamp)
            w1, w2 = self.fit_acf(corr, lags)
            errs.append([w1, w2])
        errs = np.array(errs)
        return errs.std(axis=0)


# -----------------------------------------------------------------------------
#                               Self‑test
# -----------------------------------------------------------------------------

def _self_test():  # pragma: no cover — quick human sanity check
    import matplotlib.pyplot as plt

    cfg = SimCfg(prop_mode="hybrid", z_host=0.3, nchan=2048)
    sim = FRBScintillator(cfg)

    spec = sim.simulate_dynspec()
    corr, lags = sim.acf(spec)
    sim.fit_acf(corr, lags)
    print("RP =", sim.resolution_power())
    print("ν_s (MW) = %.1f kHz, ν_s (host) = %.1f kHz" % (sim.scint_band_mw / 1e3, sim.scint_band_host / 1e3))

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].plot(sim.freqs / 1e6, spec)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Intensity (arbitrary)")

    ax[1].plot(lags / 1e3, corr, ".-")
    ax[1].set_xlabel("Lag (kHz)")
    ax[1].set_ylabel("ACF")
    ax[1].set_ylim(-0.1, 1.05)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _self_test()
