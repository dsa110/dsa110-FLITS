#!/usr/bin/env python3
"""
validate_doppler_rate.py: Validates the Doppler rate implementation in the
FRBScintillator against the theoretical formulae in Pradeep et al. (2025), Appendix A.

This script simulates a dynamic spectrum with a moving screen, computes the
secondary spectrum, and compares the resulting Doppler shifts to the theoretical
maximum.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Import the simulator classes from your script
from frb_scintillator import SimCfg, ScreenCfg, FRBScintillator

def calculate_theoretical_max_doppler(sim: FRBScintillator) -> float:
    """
    Calculates the maximum expected Doppler rate based on Appendix A.
    """
    cfg = sim.cfg
    
    # Effective velocities from Eq. A.5
    v_obs_kms = np.zeros(2) # Assume observer is stationary
    v_mw_kms = cfg.mw.v_perp
    # Relative velocity between host screen and source (assume source is stationary)
    v_host_rel_kms = cfg.host.v_perp
    
    z = cfg.z_host
    D_mw = sim.D_mw_m
    D_host = sim.D_host_m
    D_src = sim.D_src_m
    D_mw_host = sim.D_mw_host_m
    D_host_src = sim.D_host_src_m

    # Note: v_perp is in km/s, so we convert to m/s
    v_mw_mps = v_mw_kms * 1000
    v_host_rel_mps = v_host_rel_kms * 1000

    Veff_mw = v_obs_kms - (D_host / D_mw_host) * v_mw_mps + (D_mw / ((1 + z) * D_mw_host)) * v_host_rel_mps
    Veff_host = (D_host / D_mw_host) * v_mw_mps - ((D_mw / ((1 + z) * D_mw_host)) + (D_src / D_host_src)) * v_host_rel_mps

    # Max Doppler rate comes from the largest theta on each screen
    # The characteristic angular size (1-sigma radius)
    theta_L_mw = cfg.mw.L.to(u.m).value / D_mw
    theta_L_host = cfg.host.L.to(u.m).value / D_host

    # Max Doppler rate contribution from each screen (Eq. A.4)
    fD_max_mw = (sim.nu0_hz / sim.C_M_PER_S) * np.linalg.norm(Veff_mw) * theta_L_mw
    fD_max_host = (sim.nu0_hz / sim.C_M_PER_S) * np.linalg.norm(Veff_host) * theta_L_host

    # The maximum total Doppler rate is the sum of the maximums
    return fD_max_mw + fD_max_host


def main():
    """Main function to run the validation."""
    print("Setting up simulation to validate Doppler rates...")

    # 1. Configure a simulation with a high-velocity screen
    mw_screen_cfg = ScreenCfg(
        N=128,
        L=0.5 * u.AU,
        v_perp=(500.0, 0.0), # High velocity in x-direction [km/s]
        rng_seed=42
    )
    host_screen_cfg = ScreenCfg(N=128, L=1.0 * u.AU, rng_seed=101)

    sim_config = SimCfg(
        nu0=800 * u.MHz,
        bw=10 * u.MHz,
        nchan=512,
        D_mw=1.0 * u.kpc,
        z_host=0.1,
        D_host_src=2.0 * u.kpc,
        mw=mw_screen_cfg,
        host=host_screen_cfg,
        intrinsic_pulse="delta" # Use delta function to isolate IRF
    )

    # 2. Initialize simulator and calculate theoretical max Doppler rate
    sim = FRBScintillator(sim_config)
    fD_max_theory = calculate_theoretical_max_doppler(sim)
    print(f"Theoretical Maximum Doppler Rate: {fD_max_theory:.4f} Hz")

    # 3. Simulate a dynamic spectrum over a duration long enough to resolve fringes
    # A duration of a few minutes should be sufficient
    duration = 5 * u.min
    print(f"Simulating dynamic spectrum over {duration}...")
    dyn_spec, time_axis, freq_axis = sim.synthesise_dynamic_spectrum(duration)
    
    if dyn_spec.size == 0:
        print("Simulation failed to produce a dynamic spectrum.")
        return

    # 4. Compute the secondary spectrum
    print("Computing secondary spectrum...")
    # Subtract mean from each channel before FFT
    dyn_spec_mean_sub = dyn_spec - np.mean(dyn_spec, axis=0, keepdims=True)
    secondary_spec = np.abs(np.fft.fftshift(np.fft.fft(dyn_spec_mean_sub, axis=0), axes=0))**2
    
    # Sum over frequency channels to get 1D Doppler power spectrum
    doppler_power = np.sum(secondary_spec, axis=1)

    # Create the Doppler frequency axis
    time_res = time_axis[1] - time_axis[0]
    doppler_axis_hz = np.fft.fftshift(np.fft.fftfreq(dyn_spec.shape[0], d=time_res))

    # 5. Plot the results for validation
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(doppler_axis_hz, doppler_power, color='darkcyan', label='Simulated Doppler Power')
    ax.axvline(fD_max_theory, color='crimson', linestyle='--', label=f'Theoretical Max f_D (+{fD_max_theory:.2f} Hz)')
    ax.axvline(-fD_max_theory, color='crimson', linestyle='--', label=f'Theoretical Max f_D (-{fD_max_theory:.2f} Hz)')
    
    ax.set_xlabel("Doppler Frequency (Hz)")
    ax.set_ylabel("Power (Arbitrary Units)")
    ax.set_title("Doppler Rate Validation")
    ax.legend()
    ax.set_xlim(-2 * fD_max_theory, 2 * fD_max_theory)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()