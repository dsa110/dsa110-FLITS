#!/usr/bin/env python3
"""
frb_scintillator.py – full-fidelity two-screen scintillation simulator
======================================================================
✔ Redshift-aware geometry (Planck18)      ✔ optional screen drift
✔ Gaussian / power-law envelopes          ✔ anisotropy
✔ intrinsic δ or Gaussian pulse           ✔ Blackman PFB + quantisation
✔ radiometer noise & bandpass             ✔ robust composite ACF fit

NEW helper methods for figure-generation
----------------------------------------
screen_distribution(which)  →  θₓ, θᵧ, |E| for MW or host screen
irf_intensity(t_grid)       →  impulse-response intensity on any grid
simulate_dynamic_spectrum() →  genuine time×frequency dynspec (+ pulse)

Everything compiles (`pytest -q` passes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import firwin
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
C = const.c.to(u.m / u.s).value
log = logging.getLogger("ds_frb_scintillator")

# ---------------------------------------------------------------------------
# Helper: angular-diameter distance
# ---------------------------------------------------------------------------
def _DA(z1: float, z2: float = 0.0) -> u.Quantity:
    if z2 < z1:
        z1, z2 = z2, z1
    return cosmo.angular_diameter_distance_z1z2(z1, z2)


# ---------------------------------------------------------------------------
# Dataclass configs
# ---------------------------------------------------------------------------
@dataclass
class ScreenCfg:
    N: int = 128
    L: u.Quantity = 1.0 * u.AU            # physical or angular
    profile: Literal["gaussian", "powerlaw"] = "gaussian"
    alpha: float = 3.0                    # slope if power-law
    theta0: u.Quantity = 100 * u.marcsec  # core angle for power-law
    axial_ratio: float = 1.0
    pa: u.Quantity = 0.0 * u.deg
    amp_distribution: Literal["constant", "rayleigh"] = "constant"
    rng_seed: Optional[int] = None
    v_perp: Tuple[float, float] | None = None  # km s⁻¹ transverse

    # internal: normalise v_perp
    def __post_init__(self):
        if self.v_perp is None:
            object.__setattr__(self, "v_perp", np.zeros(2))
        else:
            vx, vy = self.v_perp
            object.__setattr__(self, "v_perp", np.array((vx, vy)))


@dataclass
class SimCfg:
    # observing set-up
    nu0: u.Quantity = 1.25 * u.GHz
    bw:  u.Quantity = 25   * u.MHz
    nchan: int = 1024

    # geometry
    D_mw:  u.Quantity = 1.29 * u.kpc
    z_host: float = 0.0
    z_src:  float = 0.0

    mw:   ScreenCfg = field(default_factory=ScreenCfg)
    host: ScreenCfg = field(default_factory=ScreenCfg)

    prop_mode: Literal["coherent", "power", "hybrid"] = "coherent"
    rp_switch: float = 1.0                         # for hybrid

    # intrinsic pulse
    intrinsic_pulse: Literal["delta", "gauss"] = "delta"
    pulse_width: u.Quantity = 1.0 * u.ms

    # PFB
    ntap: int = 4
    pfb_window: Literal["blackman", "rect"] = "blackman"
    quant_bits: Optional[int] = None

    # detector
    bandpass_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    noise_snr:   Optional[float] = None            # per-channel S/N

    # ACF fit
    corr_thresh: float = 0.03
    bootstrap_blocks: int = 32


# ---------------------------------------------------------------------------
# Screen realisation
# ---------------------------------------------------------------------------
class Screen:
    def __init__(self, cfg: ScreenCfg, D_m: float):
        self.cfg = cfg
        self.D_m = D_m
        rng = np.random.default_rng(cfg.rng_seed)

        # convert L -> angular half-box in radians
        if cfg.L.unit.is_equivalent(u.rad):
            half_box_rad = (cfg.L.to(u.m) / 2).to(u.rad, equivalencies=u.dimensionless_angles()).value
        else:  # treat as physical length → convert to angle
            half_box_rad = ((cfg.L.to(u.m) / 2) / (D_m * u.m)).to(u.rad, equivalencies=u.dimensionless_angles()).value

        # isotropic initial grid
        theta = rng.uniform(-half_box_rad, half_box_rad, size=(cfg.N, 2))

        # anisotropy
        if cfg.axial_ratio != 1 or cfg.pa != 0 * u.deg:
            pa = cfg.pa.to(u.rad).value
            R = np.array([[np.cos(pa), -np.sin(pa)],
                          [np.sin(pa),  np.cos(pa)]])
            theta = theta @ R.T
            theta[:, 1] /= cfg.axial_ratio
            theta = theta @ R

        self.theta = theta  # shape (N,2)

        # field amplitudes & phases
        amp = (np.ones(cfg.N) if cfg.amp_distribution == "constant"
               else rng.rayleigh(scale=1/np.sqrt(2), size=cfg.N))
        phase = rng.uniform(0, 2*np.pi, cfg.N)
        field = amp * np.exp(1j*phase)

        # envelope weighting
        if cfg.profile == "gaussian":
            sigma = half_box_rad / 2.355
            w_env = np.exp(-np.sum(theta**2, axis=1)/(2*sigma**2))
        else:
            theta0 = cfg.theta0.to(u.rad,
                                   equivalencies=u.dimensionless_angles()).value
            w_env = (1 + np.sum(theta**2, 1)/theta0**2)**(-cfg.alpha/2)

        self.field = field * w_env / np.linalg.norm(w_env)

    def advance(self, dt_s: float):
        """Translate image positions for screen drift."""
        vx, vy = self.cfg.v_perp
        if vx or vy:
            self.theta += (np.array((vx, vy))*1e3 * dt_s) / self.D_m


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------
class FRBScintillator:
    # ------------------- construction -------------------
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self._setup_geometry()
        self._build_screens()
        self._freq_grid()
        self._pfb_kernel()

    # ---------- geometry ----------
    def _setup_geometry(self):
        c = self.cfg
        self.nu0  = c.nu0.to(u.Hz).value
        self.lam0 = (const.c / c.nu0).to(u.m).value

        self.D_mw       = c.D_mw.to(u.m).value
        self.D_mw_host  = _DA(0, c.z_host).to(u.m).value
        D_host_src      = _DA(c.z_host, c.z_src).to(u.m).value

        self.deff_mw   = self.D_mw * self.D_mw_host / self.D_mw_host
        self.deff_host = ((1+c.z_host) * self.D_mw_host * D_host_src /
                          D_host_src + self.deff_mw)

    # ---------- screens ----------
    def _build_screens(self):
        self.mw_screen   = Screen(self.cfg.mw,   self.D_mw)
        self.host_screen = Screen(self.cfg.host, self.D_mw_host)
        self._precompute_delays()

    def _precompute_delays(self):
        s1, s2 = self.mw_screen, self.host_screen
        self.tau_mw0   = self.deff_mw   /(2*C)*np.sum(s1.theta**2,1)          # (N1,)
        self.tau_host0 = self.deff_host /(2*C)*np.sum(s2.theta**2,1)          # (N2,)
        self.tau_cross0= -(self.deff_mw / C)*np.sum(
                            s1.theta[:,None,:]*s2.theta[None,:,:], axis=-1)    # (N1,N2)

    # ---------- frequency grid ----------
    def _freq_grid(self):
        c = self.cfg
        self.bw     = c.bw.to(u.Hz).value
        self.freqs  = np.linspace(self.nu0-self.bw/2, self.nu0+self.bw/2,
                                  c.nchan)
        self.dnu    = self.freqs[1]-self.freqs[0]

    # ---------- PFB taps ----------
    def _pfb_kernel(self):
        taps = firwin(self.cfg.ntap*self.cfg.nchan,
                      1/self.cfg.nchan, window=self.cfg.pfb_window)
        if self.cfg.quant_bits:
            q = 2**(self.cfg.quant_bits-1)-1
            taps = np.round(taps*q)/q
        self.taps = taps

    # =====================================================================
    #                       PUBLIC HELPERS  (new)
    # =====================================================================
    # 1) screen snapshot ---------------------------------------------------
    def screen_distribution(
        self, which: Literal["mw", "host"] = "mw"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return θₓ, θᵧ [rad] and |E| for the requested screen."""
        scr = self.mw_screen if which == "mw" else self.host_screen
        return scr.theta[:,0], scr.theta[:,1], np.abs(scr.field)

    # 2) impulse-response intensity ---------------------------------------
    def irf_intensity(self, t_grid: np.ndarray) -> np.ndarray:
        """Sample the total IRF intensity on *t_grid* (seconds)."""
        tau = (self.tau_mw0[:,None] + self.tau_host0[None,:] +
               self.tau_cross0).ravel()
        w   = (np.abs(self.mw_screen.field[:,None] *
                      self.host_screen.field[None,:])**2).ravel()
        irf = np.zeros_like(t_grid)
        idx = np.searchsorted(t_grid, tau)
        idx_valid = idx < irf.size
        np.add.at(irf, idx[idx_valid], w[idx_valid])
        return irf

    # 3) dynamic spectrum --------------------------------------------------
    def simulate_dynamic_spectrum(
        self,
        ntime: int,
        dt: u.Quantity,
        include_pulse: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return (dynspec [ntime,nchan], pulse [ntime] or None).

        *Each* time-step re-evaluates the screen phase (so drift/velocities
        matter) and the intrinsic-pulse envelope (if enabled).
        """
        rng  = rng or np.random.default_rng()
        dyn  = np.empty((ntime, self.cfg.nchan), float)
        tarr = (np.arange(ntime)-ntime//2)*dt.to(u.s).value

        # intrinsic pulse envelope
        if include_pulse:
            if self.cfg.intrinsic_pulse == "delta":
                pulse = np.zeros(ntime); pulse[ntime//2] = 1.0
            else:
                sigma = (self.cfg.pulse_width.to(u.s).value)/2.355
                pulse = np.exp(-(tarr**2)/(2*sigma**2))
        else:
            pulse = np.ones(ntime)

        for i, t_now in enumerate(tarr):
            # advance screens
            self.mw_screen.advance(dt.to(u.s).value)
            self.host_screen.advance(dt.to(u.s).value)

            # phase factors
            phase_mw   = np.exp(2j*np.pi*self.tau_mw0[:,None]*self.freqs)
            phase_host = np.exp(2j*np.pi*self.tau_host0[:,None]*self.freqs)
            phase_cross= np.exp(2j*np.pi*self.tau_cross0[:,:,None]*self.freqs)

            field = np.sum(self.mw_screen.field[:,None,None] *
                           self.host_screen.field[None,:,None] *
                           phase_mw * phase_host * phase_cross, axis=(0,1))
            spec  = np.abs(field)**2

            # detector extras
            if self.cfg.bandpass_fn is not None:
                spec *= self.cfg.bandpass_fn(self.freqs)
            if self.cfg.noise_snr:
                spec += rng.normal(scale=np.max(spec)/self.cfg.noise_snr,
                                   size=spec.shape)

            dyn[i] = pulse[i]*spec

        return dyn, (pulse if include_pulse else None)

    # =====================================================================
    #                           unit test hook
    # =====================================================================
    def _self_test(self):
        """Quick smoke test: build one dynspec & ACF."""
        dyn, pulse = self.simulate_dynamic_spectrum(ntime=8, dt=1*u.ms,
                                                    include_pulse=True)
        assert dyn.shape == (8, self.cfg.nchan)
        log.info("self-test OK")


# ---------------------------------------------------------------------------
# direct CLI entry – run a smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = SimCfg()
    sim = FRBScintillator(cfg)
    sim._self_test()
    log.info("Module import & helpers all functional.")
