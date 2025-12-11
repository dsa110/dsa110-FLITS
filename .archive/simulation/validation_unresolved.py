# ============================================================================
# PART 2: validation_unresolved.py
# (New script to test the unresolved regime)
# ============================================================================
#!/usr/bin/env python3
"""validation_unresolved.py - Test the unresolved two-screen regime.

This script uses the FRBScintillator to simulate a two-screen system
with a low Resolution Power (RP < 1). It then calculates the spectral
autocorrelation function (ACF) and checks if its peak value is close
to the theoretical prediction of 3, as derived in Pradeep et al. (2025).
"""
import matplotlib.pyplot as plt
from frb_scintillator_v6 import u, SimCfg, ScreenCfg, FRBScintillator # Assuming saved as file

def validate_unresolved_regime():
    """Run simulation and validation for a low-RP system."""
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Configuration ---
    # Parameters are set to match the "unresolved" case from Table B.1
    # in Pradeep et al. (2025), which should yield RP ≈ 0.2.
    cfg_unresolved = SimCfg(
        nu0=800 * u.MHz,
        bw=25.0 * u.MHz,
        nchan=4096,  # Increased for better ACF statistics
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(
            N=200,  # Number of images
            L=3.5 * u.AU,
            rng_seed=1234
        ),
        host=ScreenCfg(
            N=200, # Number of images
            L=20.0 * u.AU,
            rng_seed=5678
        ),
        intrinsic_pulse="delta", # Use delta function to avoid self-noise
        noise_snr=None # No noise for pure theoretical test
    )

    # --- Simulation ---
    sim = FRBScintillator(cfg_unresolved)
    spectrum = sim.simulate_dynspec()
    
    # --- Analysis ---
    corr, lags = sim.acf(spectrum)
    peak_acf_value = corr[0]
    
    # --- Report Results ---
    print("\n--- Unresolved Regime Validation ---")
    print(f"Expected theoretical peak ACF value: 3.0 (since m_total^2 = (sqrt(3))^2)")
    print(f"Simulated peak ACF value: {peak_acf_value:.4f}")
    
    # Check if the result is close to the theory
    if np.isclose(peak_acf_value, 3.0, atol=0.2):
        print("✅ SUCCESS: The result is consistent with the theoretical prediction.")
    else:
        print("❌ FAILED: The result deviates significantly from the theory.")
    print("------------------------------------\n")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle("Validation of the Unresolved Regime (RP < 1)", fontsize=16)

    # Plot Spectrum
    ax1.plot(sim.freqs / 1e6, spectrum, color='C0', lw=0.8)
    ax1.set_title("Simulated Spectrum")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Intensity (arbitrary units)")
    ax1.grid(True, alpha=0.3)

    # Plot ACF
    lags_khz = lags * sim.dnu / 1e3
    ax2.plot(lags_khz, corr, '.-', color='C1', label=f'Simulated ACF (Peak = {peak_acf_value:.3f})')
    ax2.axhline(3.0, color='k', ls='--', label='Theoretical Peak (m²=3)')
    ax2.set_title("Spectral Autocorrelation Function (ACF)")
    ax2.set_xlabel("Frequency Lag (kHz)")
    ax2.set_ylabel("Normalized Correlation")
    ax2.set_xlim(0, lags_khz[-1] / 10) # Zoom in on the central part
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == '__main__':
    # To run this script, you would save the simulator code above as
    # frb_scintillator_v2_1.py and then run this file.
    # For this interactive environment, we just call the function directly.
    validate_unresolved_regime()

