#!/usr/bin/env python3
# ============================================================================
# recreate_figures.py
#
# A dedicated script to reproduce the key figures from the paper
# "Scintillometry of Fast Radio Bursts: Resolution effects in two-screen models"
# by Pradeep et al. (2025).
#
# This script uses the `frb_scintillator` module to run simulations with
# parameters specified in the paper's appendices and generates plots
# matching the publication.
#
# Usage:
#   python recreate_figures.py --figure [figure_number]
#   Example:
#   python recreate_figures.py --figure 7 8 15
#   (This will generate Figures 7, 8, and 15)
# ============================================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Assumes the updated frb_scintillator.py (v2.3) is in the same directory or path
from frb_scintillator import SimCfg, ScreenCfg, FRBScintillator

# ============================================================================
# 1. CENTRALIZED CONFIGURATIONS
# ============================================================================
# All simulation parameters are defined here, based on Appendix B of the paper.
# This makes it easy to find and verify the settings for each figure.

CONFIGS = {
    # --- Configs for Figure 7 & 8: Dynamic Spectra and ACFs ---
    'fig7_8_unresolved': SimCfg(
        nu0=800 * u.MHz, bw=25.0 * u.MHz, nchan=4096,
        z_host=0.192, D_mw=2.3 * u.kpc, D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=200, L=3.5 * u.AU, rng_seed=1234),
        host=ScreenCfg(N=200, L=20.0 * u.AU, rng_seed=5678),
        intrinsic_pulse="gauss", pulse_width=5.0 * u.ms
    ),
    'fig7_8_just_resolved': SimCfg(
        nu0=800 * u.MHz, bw=25.0 * u.MHz, nchan=2048,
        z_host=0.192, D_mw=1.29 * u.kpc, D_host_src=4.0 * u.kpc,
        mw=ScreenCfg(N=200, L=4.12 * u.AU, rng_seed=2023),
        host=ScreenCfg(N=200, L=82.5 * u.AU, rng_seed=2024),
        intrinsic_pulse="gauss", pulse_width=0.1 * u.ms
    ),
    'fig7_8_resolved': SimCfg(
        nu0=800 * u.MHz, bw=25 * u.MHz, nchan=1024,
        D_mw=1.29 * u.kpc, z_host=0.0337, D_host_src=3.0 * u.kpc,
        mw=ScreenCfg(N=200, L=4.12 * u.AU, rng_seed=2025),
        host=ScreenCfg(N=200, L=165 * u.AU, rng_seed=2026),
        intrinsic_pulse="gauss", pulse_width=0.1 * u.ms,
    ),
    # --- Configs for Figure 15: 1D Screens ---
    'fig15_parallel': SimCfg(
        nu0=800 * u.MHz, bw=1.0 * u.MHz, nchan=2048,
        D_mw=1.0 * u.kpc, z_host=0.03, D_host_src=5.0 * u.kpc,
        mw=ScreenCfg(N=400, L=10 * u.AU, geometry="1D", rng_seed=2025),
        host=ScreenCfg(N=400, L=50 * u.AU, geometry="1D", rng_seed=2026)
    ),
    'fig15_perpendicular': SimCfg(
        nu0=800 * u.MHz, bw=1.0 * u.MHz, nchan=2048,
        D_mw=1.0 * u.kpc, z_host=0.03, D_host_src=5.0 * u.kpc,
        mw=ScreenCfg(N=400, L=10 * u.AU, geometry="1D", rng_seed=2025),
        host=ScreenCfg(N=400, L=50 * u.AU, geometry="1D", pa=90*u.deg, rng_seed=2026)
    ),
}

# ============================================================================
# 2. FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_theoretical_plots():
    """Generates plots for Figures 1 and 4, which are purely theoretical."""
    
    # --- Figure 1: RP vs. Redshift ---
    print("--- Generating theoretical plot: Figure 1 (RP vs. z) ---")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    L_values_AU = np.array([5, 10, 20, 30, 50]) * u.AU
    lam_m = const.c / (1.25 * u.GHz)
    z_grid = np.linspace(0.001, 0.3, 200)
    D_grid_m = cosmo.angular_diameter_distance(z_grid)
    
    for L in L_values_AU:
        RP = (L**2) / (lam_m * D_grid_m)
        ax1.plot(z_grid, RP, label=f"L = {L.value:.0f} AU")

    ax1.axhline(1.0, linestyle="--", color="k", label="RP = 1")
    ax1.set(xlabel="Host Redshift (z)", ylabel="Resolution Power (RP)",
            title="Figure 1: Evolution of Resolution Power with Redshift",
            xlim=(0, 0.3), ylim=(0, 10))
    ax1.legend()
    fig1.savefig("figure_1_replication.png", dpi=150)

    # --- Figure 4: Theoretical ACF Components ---
    print("--- Generating theoretical plot: Figure 4 (ACF Components) ---")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    vs_mw, vs_host = 200.0, 20.0
    lag_khz = np.linspace(-400, 400, 1601)
    dv = np.abs(lag_khz)
    lorentz = lambda dv, vs: 1 / (1 + (dv / vs)**2)
    
    ACF_MW = lorentz(dv, vs_mw)
    ACF_host = lorentz(dv, vs_host)
    ACF_full = ACF_MW * ACF_host + ACF_MW + ACF_host
    
    ax4.plot(lag_khz, ACF_MW, label=r"$\mathrm{ACF}_{MW}$")
    ax4.plot(lag_khz, ACF_host, label=r"$\mathrm{ACF}_{host}$")
    ax4.plot(lag_khz, ACF_full, lw=2, label="Correct Two-Screen ACF")
    ax4.set(xlabel=r"Frequency lag $\delta \nu$ (kHz)", ylabel="Normalized ACF",
            title="Figure 4: Two-Screen ACF Decomposition", ylim=(0, 3.1))
    ax4.legend()
    fig4.savefig("figure_4_replication.png", dpi=150)
    plt.show()

def generate_figure_6():
    """Generates plot for Figure 6, showing the simulation components."""
    print("--- Generating plot for Figure 6 (Simulation Components) ---")
    cfg = CONFIGS['fig6']
    sim = FRBScintillator(cfg)

    # --- Get Data ---
    irf_delays, irf_intensities = sim.get_irf_spikes()
    duration = (np.max(irf_delays) + 5 * cfg.pulse_width.to(u.s).value) * u.s
    
    scattered_pulse, intrinsic_pulse, time_axis = sim.simulate_scattered_time_series(
        duration=duration
    )
    
    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Figure 6: Simulation Components", fontsize=16)

    axes[0].stem(irf_delays * 1e3, irf_intensities, linefmt='C0-', markerfmt=' ', basefmt=" ")
    axes[0].set_ylabel("|IRF|²")
    axes[0].set_title("Top: Impulse Response Function")

    axes[1].plot(time_axis * 1e3, intrinsic_pulse, color='C1')
    axes[1].set_ylabel("$I_{intr}$")
    axes[1].set_title("Middle: Intrinsic Pulse Intensity", loc='left')

    axes[2].plot(time_axis * 1e3, scattered_pulse, color='k')
    axes[2].set_ylabel("$I_{meas}$")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Bottom: Scattered Pulse Intensity", loc='left')
    
    peak_time_ms = time_axis[np.argmax(scattered_pulse)] * 1e3
    window_ms = 8 * cfg.pulse_width.to(u.ms).value
    axes[0].set_xlim(peak_time_ms - window_ms, peak_time_ms + window_ms)
    
    fig.savefig("figure_6_replication.png", dpi=150)
    plt.show()

def generate_figure_7_and_8():
    """Generates plots for Figures 7 (Dynamic Spectra) and 8 (ACFs)."""
    
    fig7, axes7 = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)
    fig8, axes8 = plt.subplots(3, 1, figsize=(10, 15), sharex=True, constrained_layout=True)
    
    cases = {
        'Unresolved (RP < 1)': 'fig7_8_unresolved',
        'Just Resolved (RP ≈ 1)': 'fig7_8_just_resolved',
        'Completely Resolved (RP > 1)': 'fig7_8_resolved',
    }
    
    for i, (title, cfg_key) in enumerate(cases.items()):
        print(f"--- Generating plots for case: {title} ---")
        cfg = CONFIGS[cfg_key]
        sim = FRBScintillator(cfg)
        
        # --- Simulate Data ---
        # Use a longer duration for cases with large scattering tails
        theo_obs = sim.calculate_theoretical_observables()
        duration = (theo_obs['tau_s_host_s'] * 1.5 + 8 * cfg.pulse_width.to(u.s).value) * u.s
        
        I_t_nu, time_axis, freq_axis = sim.synthesise_dynamic_spectrum(duration=duration)
        
        # --- Figure 7: Dynamic Spectrum ---
        ax = axes7[i]
        pulse_profile = I_t_nu.mean(axis=1)
        
        # Use imshow for the dynamic spectrum
        ax.imshow(I_t_nu.T, aspect='auto', origin='lower',
                  extent=[time_axis.min()*1e3, time_axis.max()*1e3, 
                          freq_axis.min()/1e6, freq_axis.max()/1e6],
                  cmap='viridis')
        ax.set_title(f"Dynamic Spectrum: {title}")
        ax.set_ylabel("Frequency (MHz)")
        if i == len(cases) - 1:
            ax.set_xlabel("Time (ms)")

        # --- Figure 8: Autocorrelation Function ---
        ax = axes8[i]
        # Calculate ACF from the time-averaged spectrum
        spectrum_avg = I_t_nu.mean(axis=0)
        corr, lags = sim.acf(spectrum_avg)
        lags_khz = lags * sim.dnu / 1e3

        # Plot the main ACF
        ax.plot(lags_khz, corr, 'k-', lw=1, label=f'Simulated ACF (RP={sim.resolution_power():.2f})')
        
        # Create inset for the narrow component
        inset = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        inset.plot(lags_khz, corr, 'r.-', lw=0.8, ms=2)
        
        # Set limits for main plot and inset
        max_lag_main = (lags_khz[-1] / 5)
        max_lag_inset = theo_obs['nu_s_host_hz'] * 5 / 1e3 # 5x HWHM
        
        ax.set_xlim(0, max_lag_main)
        inset.set_xlim(0, max_lag_inset)
        ax.indicate_inset_zoom(inset, edgecolor="black")

        ax.set_title(f"Spectral ACF: {title}")
        ax.axhline(1.0, color='blue', ls=':', label='m=1 (Single Screen)')
        ax.axhline(3.0, color='magenta', ls='--', label='m²=3 (Two Screens, Unresolved)')
        
        if i == len(cases) - 1:
            ax.set_xlabel("Frequency Lag (kHz)")
        ax.set_ylabel("Normalized Correlation")
        ax.legend()

    fig7.suptitle("Figure 7: Dynamic Spectra Under Different Resolution Regimes", fontsize=16)
    fig8.suptitle("Figure 8: Spectral ACFs Under Different Resolution Regimes", fontsize=16)
    
    fig7.savefig("figure_7_replication.png", dpi=150)
    fig8.savefig("figure_8_replication.png", dpi=150)
    plt.show()


def generate_figure_15():
    """Generates plot for Figure 15, comparing parallel and perpendicular 1D screens."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    
    # --- Case 1: Parallel Screens ---
    print("--- Simulating parallel 1D screens (Fig 15 Left) ---")
    sim_para = FRBScintillator(CONFIGS['fig15_parallel'])
    spec_para = sim_para.simulate_time_integrated_spectrum()
    acf_para, lags_para = sim_para.acf(spec_para)
    
    # --- Case 2: Perpendicular Screens ---
    print("--- Simulating perpendicular 1D screens (Fig 15 Right) ---")
    sim_perp = FRBScintillator(CONFIGS['fig15_perpendicular'])
    spec_perp = sim_perp.simulate_time_integrated_spectrum()
    acf_perp, lags_perp = sim_perp.acf(spec_perp)

    # --- Plotting ---
    # Plot screen images
    for ax, sim, title in [(axes[0,0], sim_para, "Parallel"), (axes[0,1], sim_perp, "Perpendicular")]:
        mw_mas = sim.mw_screen.theta * u.rad.to(u.mas)
        host_mas = sim.host_screen.theta * u.rad.to(u.mas)
        ax.plot(mw_mas[:,0], mw_mas[:,1], '.', ms=3, label=f'MW Screen (L={sim.cfg.mw.L.value:.0f} AU)')
        ax.plot(host_mas[:,0], host_mas[:,1], 'x', ms=3, label=f'Host Screen (L={sim.cfg.host.L.value:.0f} AU)')
        ax.set_title(f"{title} Alignment (RP={sim.resolution_power():.1f})")
        ax.set_xlabel("$\\theta_x$ (mas)"); ax.set_ylabel("$\\theta_y$ (mas)")
        ax.set_aspect('equal')
        ax.legend()

    # Plot ACFs
    axes[1,0].plot(lags_para * sim_para.dnu / 1e3, acf_para, 'k-')
    axes[1,0].set_title(f"ACF (Parallel) - Peak Corr = {acf_para[0]:.2f}")
    
    axes[1,1].plot(lags_perp * sim_perp.dnu / 1e3, acf_perp, 'k-')
    axes[1,1].set_title(f"ACF (Perpendicular) - Peak Corr = {acf_perp[0]:.2f}")

    for ax in [axes[1,0], axes[1,1]]:
        ax.set_xlabel("Frequency Lag (kHz)")
        ax.set_ylabel("Normalized Correlation")
        ax.set_ylim(-0.2, max(acf_para[0], acf_perp[0]) * 1.1)

    fig.suptitle("Figure 15: Scintillation Quenching for 1D Screens", fontsize=16)
    fig.savefig("figure_15_replication.png", dpi=150)
    plt.show()

def generate_figure_noisy_acf():
    """
    Demonstrates the effect of noise on the ACF and the correct analysis
    procedure to recover the true scintillation modulation index.
    """
    print("--- Demonstrating Noisy ACF Analysis ---")
    
    # Configure a simple single-screen simulation with noise
    
    chime_instrument = InstrumentalCfg(
        t_sys=50 * u.K,
        a_eff=2500 * u.m**2 # Approximate, adjust as needed
    )
    
    cfg = SimCfg(
        peak_flux=5*u.Jy, # Set the physical peak flux of the pulse
        nu0=800 * u.MHz,
        bw=100 * u.MHz,
        nchan=4096,
        mw=ScreenCfg(N=200, L=2 * u.AU),
        host=ScreenCfg(N=1, L=0.001 * u.AU), # Effectively a single screen
        intrinsic_pulse="delta",
    )
    sim = FRBScintillator(cfg)
    
    # --- Simulate and Analyze ---
    # The simulated spectrum now includes noise
    noisy_spectrum = sim.simulate_time_integrated_spectrum()
    corr, lags = sim.acf(noisy_spectrum)
    lags_khz = lags * sim.dnu / 1e3
    
    # The zero-lag value is now biased by noise
    biased_m_squared = corr[0]

    # --- Perform Noise-Corrected Fit ---
    def lorentzian_model(x, amplitude, hwhm):
        return amplitude / (1 + (x / hwhm)**2)
    
    # Fit the model, EXCLUDING the zero-lag point (lag > 0)
    try:
        popt, _ = curve_fit(
            lorentzian_model,
            lags_khz[1:], # Use lags > 0
            corr[1:],     # Use corr > 0
            p0=(1.0, 100) # Initial guess for (amplitude, HWHM in kHz)
        )
        noise_corrected_m_squared = popt[0] # The fitted amplitude is the true m^2
        fit_curve = lorentzian_model(lags_khz, *popt)
    except RuntimeError:
        print("Fit failed. Cannot perform noise-corrected analysis.")
        noise_corrected_m_squared = np.nan
        fit_curve = np.full_like(corr, np.nan)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot the raw, noisy ACF data
    ax.plot(lags_khz, corr, 'o', color='gray', alpha=0.6, label='Noisy ACF Data')
    
    # Highlight the biased zero-lag point
    ax.plot(0, biased_m_squared, 'x', color='red', markersize=12, mew=2,
            label=f'Biased ACF(0) = {biased_m_squared:.2f}')

    # Plot the noise-corrected fit
    ax.plot(lags_khz, fit_curve, 'b-', lw=2,
            label=f'Fit to Lags > 0')
            
    # Highlight the extrapolated, true m^2
    ax.plot(0, noise_corrected_m_squared, '*', color='blue', markersize=15,
            label=f'Noise-Corrected $m^2_{{scint}}$ = {noise_corrected_m_squared:.2f}')

    ax.set_title(f"ACF Analysis with Noise (SNR = {cfg.noise_snr})", fontsize=16)
    ax.set_xlabel("Frequency Lag (kHz)", fontsize=12)
    ax.set_ylabel("Normalized Correlation", fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Theoretical expectation for a single screen is m^2 = 1
    ax.axhline(1.0, color='k', ls=':', label='Theoretical $m^2=1$ (Noise-Free)')
    
    ax.set_xlim(-50, lags_khz[-1])
    ax.set_ylim(0, biased_m_squared * 1.2)
    
    plt.tight_layout()
    fig.savefig("figure_noisy_acf_demo.png", dpi=150)
    plt.show()
    
# ============================================================================
# 3. MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recreate figures from Pradeep et al. (2025).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--figure',
        nargs='+',
        type=int,
        required=True,
        help='The figure number(s) to generate (e.g., 7, 8, 15).'
    )
    args = parser.parse_args()

    # Dictionary mapping figure numbers to their generation functions
    FIGURE_GENERATORS = {
        1: generate_theoretical_plots,
        4: generate_theoretical_plots,
        6: generate_figure_6,
        7: generate_figure_7_and_8,
        8: generate_figure_7_and_8,
        15: generate_figure_15,
        # 16 is in its own script (multifreq_analysis.py) because it is slow
        99: generate_figure_noisy_acf
    }
    
    # Keep track of which figures have been generated
    generated = []
    
    for fig_num in args.figure:
        if fig_num in generated:
            continue
            
        if fig_num in FIGURE_GENERATORS:
            print(f"\n==================== Generating Figure {fig_num} ====================")
            FIGURE_GENERATORS[fig_num]()
            generated.append(fig_num)
            if fig_num == 7: generated.append(8) # 7 and 8 are generated together
            if fig_num == 8: generated.append(7)
        else:
            print(f"WARNING: No generator found for Figure {fig_num}. Skipping.")
