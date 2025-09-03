# ============================================================================
# monte_carlo.py
# New script to run the Monte Carlo simulation.
# ============================================================================
import os
import gc
import sys
import numpy as np
import pandas as pd
import concurrent.futures
from collections import Counter
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import firwin, stft
from scipy.optimize import least_squares
from tqdm import tqdm, trange
from pprint import pprint
from geometry import _DA
from screen import ScreenCfg
from engine import SimCfg, FRBScintillator
from analysis_utils import calculate_theoretical_observables, DistanceEstimator

def run_single_trial(trial_index: int):
    """
    Encapsulates the logic for a single Monte Carlo trial.
    Uses importance sampling to generate observable systems.
    """
    rng = np.random.default_rng(trial_index)
    
    # 1. Define FIXED instrument parameters from Appendix B
    nu0 = 1.4 * u.GHz
    bw = (500) * u.MHz
    nchan = (2**13) # 8192 channels
    dnu = (bw / nchan).to(u.Hz).value
    
    # 2. Define the "observable window" for nu_s_mw based on filters
    min_nu_s_mw = 30 * dnu
    max_nu_s_mw = bw.to(u.Hz).value / 100
    if min_nu_s_mw >= max_nu_s_mw:
        return "Rejected: Instrument parameters create impossible filters"

    # 3. Generate physical parameters using Importance Sampling
    z_frb = rng.uniform(0.01, 0.5)
    D_mw_kpc = rng.uniform(0.1, 5.0)
    
    # Generate an observable nu_s_mw and work backwards to L_mw
    target_nu_s_mw = rng.uniform(min_nu_s_mw, max_nu_s_mw)
    D_host_m = _DA(0.0, z_frb).to(u.m).value
    D_mw_m = (D_mw_kpc * u.kpc).to(u.m).value
    if D_host_m <= D_mw_m: return "Rejected: Non-physical geometry"
    D_mw_host_m = D_host_m - D_mw_m
    deff_mw_m = (D_mw_m * D_host_m) / D_mw_host_m
    theta_L_mw_sq = const.c.value / (np.pi * deff_mw_m * target_nu_s_mw)
    L_mw_m = 2 * D_mw_m * np.sqrt(theta_L_mw_sq)

    # Now generate the host screen parameters
    target_rp = rng.uniform(0.1, 15.0)
    lam0_m = (const.c / nu0).to(u.m).value
    L_host_m = (target_rp * lam0_m * D_mw_host_m) / L_mw_m
    D_host_src_kpc = rng.uniform(0.1, 100.0)
    
    # 4. Configure simulation
    cfg = SimCfg(
        nu0=nu0, bw=bw, nchan=nchan, z_host=z_frb,
        D_mw=D_mw_kpc*u.kpc, D_host_src=D_host_src_kpc*u.kpc,
        mw=ScreenCfg(N=1000, L=L_mw_m*u.m, rng_seed=rng.integers(1e6)),
        host=ScreenCfg(N=1000, L=L_host_m*u.m, rng_seed=rng.integers(1e6))
    )
    sim = FRBScintillator(cfg)
    
    # 5. Final filter: check for scale separation
    theo_obs = calculate_theoretical_observables(sim)
    if not (theo_obs['nu_s_mw_hz'] > 8 * theo_obs['nu_s_host_hz']):
        return "Rejected: Scales not separated enough"
        
    # 6. Analyze the output
    spectrum = sim.simulate_time_integrated_spectrum()
    corr, lags = sim.acf(spectrum)
    if corr.size == 0 or np.isnan(corr[0]): return "Rejected: ACF calculation failed"
    m_mw_quenched = 1.0 / np.sqrt(1 + (np.pi**2 / 4**3) * sim.resolution_power()**2)
    nu_s_mw_fit, _ = sim.fit_acf(corr, lags)
    if np.isnan(nu_s_mw_fit): return "Rejected: ACF fit failed"
        
    # 7. Apply distance estimators
    estimator = DistanceEstimator(z_frb, sim.nu0_hz, sim.D_src_m, sim.D_mw_m)
    tau_s_h = theo_obs['tau_s_host_s']
    
    return {
        "injected_D_h_src_kpc": D_host_src_kpc, "rp": sim.resolution_power(),
        "est_main_kpc": (estimator.main_2022(nu_s_mw_fit, tau_s_h)*u.m).to(u.kpc).value,
        "est_ocker_kpc": (estimator.ocker_2022(nu_s_mw_fit, tau_s_h)*u.m).to(u.kpc).value,
        "est_sammons_kpc": (estimator.sammons_2023(nu_s_mw_fit, tau_s_h, m_mw_quenched)*u.m).to(u.kpc).value,
        "est_pradeep_kpc": (estimator.pradeep_2025(nu_s_mw_fit, tau_s_h, m_mw_quenched)*u.m).to(u.kpc).value,
    }

def run_monte_carlo_parallel(n_trials=1000):
    results, rejections = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_trial, i): i for i in range(n_trials)}
        pbar = tqdm(total=n_trials, desc="Running Monte Carlo Trials")
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, dict): results.append(result)
            else: rejections.append(result)
            pbar.update(1)
        pbar.close()

    print("\n--- Monte Carlo Run Summary ---")
    total_succeeded = len(results)
    print(f"Total trials attempted: {n_trials}")
    print(f"Successful trials: {total_succeeded} ({100*total_succeeded/n_trials:.1f}%)")
    print("\nRejection reasons:")
    counts = Counter(rejections)
    for reason, count in counts.items():
        print(f"- {reason}: {count} trials ({100*count/n_trials:.1f}%)")
    print("--------------------------------\n")
    
    return pd.DataFrame(results)