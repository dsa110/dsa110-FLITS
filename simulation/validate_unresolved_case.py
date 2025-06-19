#!/usr/bin/env python3
# ============================================================================
# validate_unresolved_case.py
#
# This script serves as a validation test for the frb_scintillator module.
# It simulates the "unresolved" two-screen case from Pradeep et al. (2025)
# and verifies that the total modulation index squared (the peak of the ACF)
# is close to the theoretical value of 3, as predicted by Eq. 4.26 in the
# paper.
#
# A successful run of this script provides confidence that the core physics
# of the coherent summation across two screens is implemented correctly.
#
# Usage:
#   python validate_unresolved_case.py
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import sys

# Add the parent directory to the path to find the frb_scintillator module
# This allows the script to be run from within the validation/ directory
sys.path.append('..')
from frb_scintillator import SimCfg, ScreenCfg, FRBScintillator

def run_validation():
    """
    Configures and runs the simulation for the unresolved case and checks
    the result against theoretical predictions.
    """
    
    # --- Configuration ---
    # Parameters are set to match the "unresolved" case from Table B.1
    # in Pradeep et al. (2025). This should yield RP << 1.
    cfg = SimCfg(
        peak_flux=5*u.Jy, # Set the physical peak flux of the pulse
        nu0=800 * u.MHz,
        bw=25.0 * u.MHz,
        nchan=4096,  # Increased for better ACF statistics
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(
            N=200,
            L=3.5 * u.AU,
            rng_seed=1234
        ),
        host=ScreenCfg(
            N=200,
            L=20.0 * u.AU,
            rng_seed=5678
        ),
        intrinsic_pulse="delta" # Use delta function to avoid self-noise effects
    )

    # --- Simulation and Analysis ---
    print("--- Running Unresolved Regime Validation ---")
    sim = FRBScintillator(cfg)
    print(f"Simulating with Resolution Power (RP) = {sim.resolution_power():.3f}")
    
    # Simulate the time-integrated spectrum (equivalent to a delta pulse)
    spectrum = sim.simulate_time_integrated_spectrum()
    
    # Calculate the spectral autocorrelation function
    corr, lags = sim.acf(spectrum)
    peak_acf_value = corr[0]
    
    # --- Verification ---
    print("\n" + "="*43)
    print(f"Expected theoretical peak ACF value (m_total^2): 3.0")
    print(f"Simulated peak ACF value: {peak_acf_value:.4f}")

    # Check if the result is close to the theory within a reasonable tolerance
    if np.isclose(peak_acf_value, 3.0, atol=0.2):
        print("✅ SUCCESS: The result is consistent with the theoretical prediction.")
    else:
        print("❌ FAILED: The result deviates significantly from the theory.")
    print("="*43 + "\n")

    # --- Visualization ---
    # Plot the ACF for visual confirmation of the result.
    fig, ax = plt.subplots(figsize=(10, 6))
    lags_khz = lags * sim.dnu / 1e3
    
    ax.plot(lags_khz, corr, 'k-', lw=1.5, label=f'Simulated ACF (Peak = {peak_acf_value:.3f})')
    ax.axhline(3.0, color='r', ls='--', label='Theoretical Peak for Two Unresolved Screens (m²=3)')
    
    ax.set_xlabel("Frequency Lag (kHz)", fontsize=12)
    ax.set_ylabel("Normalized Correlation", fontsize=12)
    ax.set_title("Validation of Unresolved Case ACF", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim(left=0, right=lags_khz[len(lags_khz)//10]) # Zoom in on the central part
    
    plt.tight_layout()
    plt.savefig("validation_unresolved_case.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    run_validation()
