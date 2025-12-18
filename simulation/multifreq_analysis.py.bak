#!/usr/bin/env python3
# ============================================================================
# multifreq_analysis.py
#
# This script performs a broadband (multi-frequency) scintillation analysis
# to replicate the results of Figure 16 from Pradeep et al. (2025).
# It demonstrates how the scintillation bandwidth (nu_s) evolves with
# observing frequency, revealing the transition from an unresolved to a
# resolved two-screen system.
#
# Key Features:
# - Uses the `frb_scintillator` module for the core simulation.
# - Implements Monte Carlo averaging by running multiple trials per frequency
#   to obtain statistically robust measurements and error bars.
# - Correctly scales screen sizes with frequency based on the nu^-2 law.
# - Fits power-law models to different resolution regimes (unresolved,
#   partially resolved, highly resolved) to measure the spectral index alpha.
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
from tqdm import trange

# Assumes the updated frb_scintillator.py is in the same directory or path
from screen import ScreenCfg
from engine import SimCfg, FRBScintillator

def run_broadband_analysis(n_trials: int = 30):
    """
    Performs a broadband analysis to replicate Figure 16 from Pradeep et al. (2025).

    This function simulates a two-screen system over a range of observing
    frequencies. It includes Monte Carlo averaging over multiple random screen
    realisations at each frequency to obtain statistically robust results
    and error bars, as was done in the paper.

    Args:
        n_trials (int): The number of random simulations to average per
                        frequency point. The paper uses 30.

    Returns:
        dict: A dictionary containing the aggregated results of the analysis,
              including frequencies, mean scintillation bandwidths, standard
              errors, and mean resolution powers.
    """
    
    # --- 1. Define Reference Configuration ---
    # Screen sizes are defined at a single reference frequency (nu_ref).
    # They will be scaled appropriately for each simulated frequency.
    nu_ref = 1.0 * u.GHz
    L_mw_ref = 2.0 * u.AU
    L_host_ref = 10.0 * u.AU
    
    base_cfg_dict = {
        "peak_flux": 5 * u.Jy, "D_mw": 1.0 * u.kpc, "z_host": 0.05, "D_host_src": 4.0 * u.kpc,
        "nchan": 2048, "bw": 2.0 * u.MHz, "intrinsic_pulse": "delta"
    }
    
    # --- 2. Define Simulation Frequencies and Prepare Results Dictionary ---
    sim_freqs = np.linspace(300, 1400, 20) * u.MHz
    results = {
        "freqs_mhz": [], "nu_s_mw_mean_hz": [], "nu_s_mw_err_hz": [], "rp_mean": []
    }
    
    print(f"Running broadband analysis with {n_trials} trials per frequency...")
    
    # --- 3. Outer Loop: Iterate Over Frequencies ---
    for freq in trange(sim_freqs.size, desc="Simulating Frequencies"):
        nu0 = sim_freqs[freq]
        
        trial_nu_s_mw = []
        trial_rp = []
        
        # --- 4. Inner Loop: Monte Carlo Trials ---
        for i in range(n_trials):
            # The angular size of scattering theta_L scales as nu^-2.
            # Physical reasoning: The scattering angle θ_L ∝ λ = c/ν
            # For thin screens, θ_L ∝ ν^(-2) (Eq. 3.7)
            # Since L = D × θ_L, and D is fixed, L ∝ ν^(-2)
            scale_factor = (nu0 / nu_ref).value**-2
            L_mw_scaled = L_mw_ref * scale_factor
            L_host_scaled = L_host_ref * scale_factor
            
            # Use a unique seed for each trial to ensure random screen realisations
            seed = int(nu0.to(u.Hz).value) + i
            
            cfg = SimCfg(
                **base_cfg_dict,
                nu0=nu0,
                mw=ScreenCfg(N=200, L=L_mw_scaled, rng_seed=seed),
                host=ScreenCfg(N=200, L=L_host_scaled, rng_seed=seed + n_trials)
            )
            sim = FRBScintillator(cfg)
            
            # Run simulation and analysis
            spec = sim.simulate_time_integrated_spectrum()
            corr, lags = sim.acf(spec)
            nu_s_mw, _ = sim.fit_acf_robust(corr, lags)
            
            if not np.isnan(nu_s_mw):
                trial_nu_s_mw.append(nu_s_mw)
                trial_rp.append(sim.resolution_power())
        
        # --- 5. Aggregate and Store Results from Trials ---
        if trial_nu_s_mw:
            results["freqs_mhz"].append(nu0.to(u.MHz).value)
            results["nu_s_mw_mean_hz"].append(np.mean(trial_nu_s_mw))
            # Calculate standard error of the mean for error bars
            results["nu_s_mw_err_hz"].append(np.std(trial_nu_s_mw) / np.sqrt(len(trial_nu_s_mw)))
            results["rp_mean"].append(np.mean(trial_rp))
            
    # Convert result lists to numpy arrays for easier processing
    for key in results:
        results[key] = np.array(results[key])

    return results

def plot_broadband_results(results: dict):
    """
    Plots the results from the broadband analysis, replicating Figure 16.
    
    Args:
        results (dict): The dictionary of results from run_broadband_analysis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data for plotting
    freqs_mhz = results["freqs_mhz"]
    nu_s_mhz_mean = results["nu_s_mw_mean_hz"] / 1e6
    nu_s_mhz_err = results["nu_s_mw_err_hz"] / 1e6
    rps = results["rp_mean"]

    # Plot data points with error bars
    ax.errorbar(freqs_mhz, nu_s_mhz_mean, yerr=nu_s_mhz_err, fmt='o', color='k', 
                capsize=4, elinewidth=1.5, label="Simulation (Mean & Std. Error)")
    
    # --- Power-Law Fits: nu_s = a * nu^alpha ---
    def power_law(x, a, b):
        return a * x**b
    
    # Fit Unresolved Regime (RP < 1)
    unresolved_mask = rps < 1
    if np.sum(unresolved_mask) > 1:
        # Provide a good initial guess for alpha = 4
        p0 = (nu_s_mhz_mean[unresolved_mask][0] / freqs_mhz[unresolved_mask][0]**4, 4.0)
        popt, _ = curve_fit(power_law, freqs_mhz[unresolved_mask], nu_s_mhz_mean[unresolved_mask], p0=p0, maxfev=5000)
        ax.plot(freqs_mhz[unresolved_mask], power_law(freqs_mhz[unresolved_mask], *popt),
                'g--', lw=2.5, label=f'Unresolved fit ($\\alpha \\approx {popt[1]:.2f}$)')

    # Fit Highly Resolved Regime (RP > 3)
    resolved_mask = rps > 3
    if np.sum(resolved_mask) > 1:
        # Provide a good initial guess for alpha = 1
        p0 = (nu_s_mhz_mean[resolved_mask][0] / freqs_mhz[resolved_mask][0]**1, 1.0)
        popt, _ = curve_fit(power_law, freqs_mhz[resolved_mask], nu_s_mhz_mean[resolved_mask], p0=p0, maxfev=5000)
        ax.plot(freqs_mhz[resolved_mask], power_law(freqs_mhz[resolved_mask], *popt),
                'b--', lw=2.5, label=f'Highly Resolved fit ($\\alpha \\approx {popt[1]:.2f}$)')

    # --- Add RP lines for context ---
    sorted_indices = np.argsort(rps)
    sorted_rps, sorted_freqs = rps[sorted_indices], freqs_mhz[sorted_indices]
    if len(sorted_rps) > 1:
        rp1_freq = np.interp(1.0, sorted_rps, sorted_freqs)
        rp3_freq = np.interp(3.0, sorted_rps, sorted_freqs)
        ax.axvline(rp1_freq, color='dimgray', ls='--', label='RP=1')
        ax.axvline(rp3_freq, color='dimgray', ls=':', label='RP=3')
        ax.axvspan(300, rp1_freq, alpha=0.2, color='green', label='Unresolved (RP<1)')
        ax.axvspan(rp1_freq, rp3_freq, alpha=0.2, color='orange', label='Partially resolved')
        ax.axvspan(rp3_freq, 1400, alpha=0.2, color='red', label='Highly resolved (RP>3)')

    # --- Final Plot Formatting ---
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Observing Frequency (MHz)", fontsize=14)
    ax.set_ylabel("Scintillation Bandwidth, $\\nu_{s,MW}$ (MHz)", fontsize=14)
    ax.set_title("Broadband Scintillation Analysis (Fig. 16 Replication)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    
    # Set nice tick labels for log scale
    from matplotlib.ticker import ScalarFormatter
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Run the full analysis and generate the plot
    analysis_results = run_broadband_analysis(n_trials=30)
    plot_broadband_results(analysis_results)
