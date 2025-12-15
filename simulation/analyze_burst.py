
import argparse
import yaml
import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

from simulation.wave_optics import WaveOpticsSimCfg, WaveOpticsScreenCfg, WaveOpticsScintillator

def estimate_r0_from_tau(tau_s, D_eff, nu):
    """
    Invert Eq 4.9 from Pradeep 2025 to estimate r0 from tau.
    tau ~ D_eff * theta_scatt^2 / 2c
    theta_scatt ~ lambda / r0
    => r0 ~ lambda / sqrt(2 * c * tau / D_eff)
    """
    c = const.c # Astropy constant
    lam = c / nu
    
    # theta^2 = 2 * c * tau / D_eff
    # Ensure all are quantities
    val = (2 * c * tau_s / D_eff).to(u.dimensionless_unscaled)
    theta = np.sqrt(val)
    
    # r0 = lambda / theta
    r0 = lam / theta
    return r0.to(u.m)

def analyze_burst(burst_name, bursts_file="bursts.yaml"):
    # 1. Load Burst Data
    with open(bursts_file, "r") as f:
        data = yaml.safe_load(f)
    
    if burst_name not in data['bursts']:
        print(f"Error: Burst '{burst_name}' not found in {bursts_file}")
        return

    burst = data['bursts'][burst_name]
    print(f"Analyzing Burst: {burst_name.upper()}")
    
    # Get parameters (handle missing values with defaults)
    dm = burst.get('dm', 500.0)
    # Estimate redshift roughly from DM (z approx DM/1000)
    z_approx = dm / 1000.0
    print(f"  DM: {dm}, Approx z: {z_approx:.3f}")
    
    scat = burst.get('scattering')
    if scat:
        tau_ms = scat.get('tau_1ghz_ms', 0.0)
        print(f"  Observed Scattering: {tau_ms} ms @ 1 GHz")
    else:
        tau_ms = 0.1 # Default small scattering if null
        print(f"  No scattering fit found. Assuming toy tau={tau_ms} ms.")

    # 2. Config Simulation
    nu0 = 1.25 * u.GHz
    
    # Setup Geometry
    # MW Screen: Fixed at 1 kpc
    D_mw = 1.0 * u.kpc
    
    # Host Screen: Fixed near source (total distance approx D_A(z))
    # Note: For wave optics code, we need D_host parameter relative to observer
    # But code handles cosmological DA internally if we pass z_host.
    # We'll assume the host screen is AT the host galaxy (z_approx).
    
    # SCALING STRATEGY
    # ----------------
    # Direct simulation of ms-scale scattering requires ~10^7 pixels (Fresnel number ~ c*tau/lambda).
    # To test the physics (Modulation Index vs Resolution Power), we simulate a "Scaled System".
    # We maintain the "Resolution Power" topology but scale tau down to ~0.5 us.
    # This allows checking if the MW screen resolves the Host screen.
    
    target_tau_us = 0.1 
    scale_factor = target_tau_us / (tau_ms * 1e3) 
    print(f"  Scaling Simulation: Reducing tau by factor {scale_factor:.1e} to {target_tau_us} us for computational feasibility.")
    
    # We use the Toy D_eff_sim but derive r0 to match this SCALED tau.
    D_eff_sim = 1.0 * u.kpc
    tau_sim = target_tau_us * 1e-6 * u.s
    r0_sim = estimate_r0_from_tau(tau_sim, D_eff_sim, nu0)
    
    print(f"  Scaled Simulation Params: D={D_eff_sim}, Derived r0={r0_sim:.2e}")

    # Config Objects
    # Ensure screens are resolved (L > scattering disk)
    lam = (const.c / nu0).to(u.m)
    theta_scatt = lam / r0_sim
    L_disk = D_eff_sim * theta_scatt
    
    # Critical: For valid wave optics, we need dx < r0.
    # L_grid = N * dx.
    # We set L_grid to cover the disk (e.g. 4 * L_disk).
    # Then we check if N=512 gives dx < r0.
    L_grid = L_disk * 4.0
    dx = L_grid / 512.0
    
    if dx > r0_sim:
        print(f"  WARNING: Grid undersampled (dx={dx:.2e} > r0={r0_sim:.2e}). Increasing L/r0 ratio requires more pixels.")
        # Clamp L to ensure resolution, even if we clip tails? 
        # Better to just use a smaller tau for validation if needed.
        # or increase N?
    else:
        print(f"  Grid Samling OK: dx={dx:.2e} < r0={r0_sim:.2e}")

    print(f"  Scattering Disk Size: {L_disk.to(u.AU):.2f}. Grid Size: {L_grid.to(u.AU):.2f}")
    
    screen_cfg = WaveOpticsScreenCfg(
        N=512, 
        L=L_grid,
        r0_ref=r0_sim,
        nu_ref=nu0
    )
    
    sim_cfg = WaveOpticsSimCfg(
        nu0=nu0,
        bw=32.0 * u.MHz, 
        nchan=64, 
        mw=screen_cfg, 
        host=screen_cfg,
        D_mw=0.1 * u.kpc, # Scaled D
        z_host=0.0, 
        D_host_src=0.1 * u.kpc
    )
    
    # 3. Run Simulation
    print("  Running Wave Optics Simulator...")
    scint = WaveOpticsScintillator(sim_cfg)
    I_nu, freqs = scint.simulate_dynamic_spectrum()
    
    # 4. Analyze
    obs = scint.compute_observables(I_nu)
    print(f"  Simulation Result: m = {obs['modulation_index']:.3f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs/1e9, I_nu, label='Simulated Spectrum')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Intensity")
    plt.title(f"Wave Optics Re-analysis: {burst_name.upper()}\n(tau={tau_ms}ms, m={obs['modulation_index']:.2f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    outfile = f"analysis_{burst_name}_waveoptics.png"
    plt.savefig(outfile)
    print(f"  Saved analysis plot to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("burst", help="Name of burst in bursts.yaml (e.g., freya)")
    args = parser.parse_args()
    
    analyze_burst(args.burst)
