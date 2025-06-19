# ============================================================================
# analysis_utils.py
# New helper module for physics calculations and distance estimation.
# ============================================================================
import os
import sys
import numpy as np
import pandas as pd
import concurrent.futures
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import firwin, stft
from scipy.optimize import least_squares
from tqdm import trange
from pprint import pprint
from frb_scintillator_v6 import SimCfg, FRBScintillator # Assumes file saving

def calculate_theoretical_observables(sim: FRBScintillator) -> dict:
    """Calculates theoretical nu_s and tau_s from simulation parameters."""
    cfg = sim.cfg
    
    # Get angular size of scattering disks
    theta_L_mw = (cfg.mw.L.to(u.m) / (2 * sim.D_mw_m * u.m)).to(u.rad, equivalencies=u.dimensionless_angles()).value
    theta_L_host = (cfg.host.L.to(u.m) / (2 * sim.D_host_m * u.m)).to(u.rad, equivalencies=u.dimensionless_angles()).value
    
    # Scintillation bandwidth (nu_s) from Eq. 4.14
    nu_s_mw = const.c.value / (np.pi * sim.deff_mw_m * theta_L_mw**2)
    nu_s_host = const.c.value / (np.pi * sim.deff_host_m * theta_L_host**2)
    
    # Scattering time (tau_s) from Eq. 4.9
    tau_s_mw = (sim.deff_mw_m * theta_L_mw**2) / (2 * const.c.value)
    tau_s_host = (sim.deff_host_m * theta_L_host**2) / (2 * const.c.value)
    
    return {
        "nu_s_mw_hz": nu_s_mw, "nu_s_host_hz": nu_s_host,
        "tau_s_mw_s": tau_s_mw, "tau_s_host_s": tau_s_host,
    }

class DistanceEstimator:
    """Implements the four distance estimation formulas from Table 1."""
    def __init__(self, z_frb, nu_hz, D_frb_m, D_mw_m):
        self.z_frb = z_frb
        self.nu_hz = nu_hz
        self.D_frb_m = D_frb_m
        self.D_mw_m = D_mw_m
        self.common_factor = (D_frb_m**2) / (2 * np.pi * nu_hz**2 * D_mw_m)

    def main_2022(self, nu_s_mw, tau_s_h):
        # This is the formula from Main et al. 2022 (with a pi^2 difference)
        return (np.pi**2 * self.common_factor * (nu_s_mw / tau_s_h))

    def ocker_2022(self, nu_s_mw, tau_s_h):
        # Formula from Ocker et al. 2022 / Nimmo et al. 2025
        return (np.pi * self.common_factor * (nu_s_mw / tau_s_h))

    def sammons_2023(self, nu_s_mw, tau_s_h, m_mw):
        # Formula from Sammons et al. 2023
        if m_mw == 0: return np.inf
        return (self.common_factor * (nu_s_mw / tau_s_h) / 
                ((1 + self.z_frb) * m_mw**2))

    def pradeep_2025(self, nu_s_mw, tau_s_h, m_mw):
        # This work's formula (Eq. 7.6)
        if m_mw == 0: return np.inf
        return (self.common_factor * (1 + self.z_frb) / (4) *
                (nu_s_mw / (m_mw * tau_s_h)))


#def calculate_theoretical_observables(sim: FRBScintillator) -> dict:
#    cfg = sim.cfg
#    # The L in the config is the 4-sigma width, so theta_L is L / (2*D)
#    theta_L_mw = (cfg.mw.L.to(u.m).value / (2 * sim.D_mw_m))
#    theta_L_host = (cfg.host.L.to(u.m).value / (2 * sim.D_host_m))
#    nu_s_mw = const.c.value / (np.pi * sim.deff_mw_m * theta_L_mw**2)
#    nu_s_host = const.c.value / (np.pi * sim.deff_host_m * theta_L_host**2)
#    tau_s_host = (sim.deff_host_m * theta_L_host**2) / (2 * const.c.value)
#    return { "nu_s_mw_hz": nu_s_mw, "nu_s_host_hz": nu_s_host, "tau_s_host_s": tau_s_host }
#
#class DistanceEstimator:
#    def __init__(self, z_frb, nu_hz, D_frb_m, D_mw_m):
#        self.z_frb, self.nu_hz, self.D_frb_m, self.D_mw_m = z_frb, nu_hz, D_frb_m, D_mw_m
#        self.common_factor = (D_frb_m**2) / (2 * np.pi * nu_hz**2 * D_mw_m)
#    def main_2022(self, nu_s, tau_s): return (np.pi**2*self.common_factor * (nu_s/tau_s))
#    def ocker_2022(self, nu_s, tau_s): return (np.pi*self.common_factor * (nu_s/tau_s))
#    def sammons_2023(self, nu_s, tau_s, m_mw):
#        if m_mw<=1e-9: return np.inf
#        return self.common_factor * (nu_s/tau_s) / ((1+self.z_frb)*m_mw**2)
#    def pradeep_2025(self, nu_s, tau_s, m_mw):
#        if m_mw<=1e-9: return np.inf
#        return self.common_factor * (1+self.z_frb)/4 * (nu_s/(m_mw*tau_s))