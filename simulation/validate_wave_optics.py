
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from simulation.wave_optics import WaveOpticsSimCfg, WaveOpticsScintillator, WaveOpticsScreenCfg

def validate():
    # Setup parameters for a regime with RP << 1 (Unresolved)
    # This should yield modulation index m ~ sqrt(3) ~ 1.73 for two strong screens
    
    # MW Screen: Tuned for valid sampling on 1024 grid
    # D = 0.5 kpc -> rF ~ 1.7e9 m
    # We choose r0 ~ 5e8 m (Weak/Moderate scattering)
    # Scattering angle theta ~ lam/r0 ~ 0.2/5e8 = 4e-10
    # Disk size ~ theta * D ~ 4e-10 * 1.5e19 ~ 6e9 m
    # We set L = 4e10 m (approx 25 * rF, captures disk)
    # dx = 4e10 / 1024 ~ 4e7 m.
    # dx < r0 (4e7 < 5e8). Resolved!
    mw_cfg = WaveOpticsScreenCfg(
        N=1024,
        L=0.3 * u.AU, # ~4.5e10 m
        r0_ref=5e8 * u.m, 
        nu_ref=1.25 * u.GHz
    )
    
    # Host Screen
    host_cfg = WaveOpticsScreenCfg(
        N=1024,
        L=0.3 * u.AU,
        r0_ref=5e8 * u.m, 
        nu_ref=1.25 * u.GHz
    )
    
    # "Toy" Cosmology: 0.5 kpc and 1.0 kpc (Local/Pulsar-like scales)
    # This avoids Gpc projection issues for validation
    sim_cfg = WaveOpticsSimCfg(
        nu0=1.25 * u.GHz,
        bw=10.0 * u.MHz,
        nchan=50, 
        mw=mw_cfg,
        host=host_cfg,
        D_mw=0.5 * u.kpc, # Screen 1
        z_host=0.0,
        D_host_src=0.5 * u.kpc # Source 0.5kpc behind Host (Total 1.5?)
    )
    # Actually, let's interpret D parameters consistently.
    # D_mw is distance to first screen.
    # D_host is distance to second screen.
    # D_host_src is distance from second screen to source.
    # We'll calculate z_host=0 for local.
    
    # We need to manually patch the D_host_m interpretation in the class if z=0
    # Because _DA(0,0) is 0. 
    # But WaveOpticsScintillator uses _DA.
    # We should stick to the class logic.
    # If z_host > 0, D_host is large.
    # To mimic a "Toy" 2-screen system, we might need a subclass or just use the z_host logic 
    # but acknowledge we only test the "Strong + Strong" limit.
    
    # Let's keep the original "Gpc" setup but increase simulation box L?
    # No, L=10^22 m is 1 Gpc. We can't simulate a grid that large with 1024 pixels (dx would be MPc).
    # Wave optics requires dx < r0 (approx). 
    # If r0 ~ 1000m, we need 10^20 pixels.
    # IMPOSSIBLE for Gpc simulation with realistic r0.
    
    # Conclusion: Full wave optics simulation of IGM scattering (Gpc) is only possible if
    # the scattering is extremely weak (r0 large) OR if we use "effective" scaled models.
    # Pradeep 2025 likely simulates a "scaled" version or focuses on the diffractive limit near caustics.
    # Given the user says "Extending FLITS... significantly improve accuracy in certain regimes",
    # and "validating geometric approximation", it's likely intended for the regime where it IS feasible.
    # (e.g. Plasma lensing, small D).
    
    # So I will assert the "Toy Model" (Pulsar-like distances) is the correct validation case.
    # To fix "z_host=0" issue, I will just set params such that D_host is calculable.
    # Actually, let's just use z_host=0.000001 (very close).
    
    sim_cfg.z_host = 2.5e-7 # -> ~1 kpc distance? 
    # D_A ~ c*z/H0. 3e5 * 1e-7 ~ 0.03 km. Too small.
    # 1 kpc = 3e19 m. c/H0 = 4000 Mpc.
    # z = 1 kpc / 4 Gpc ~ 2.5e-7.
    
    # Let's simple hardcode distances in the object if needed, but let's try z ~ 2.5e-7.
    # Wait, 1 kpc / 4 Gpc is 2.5e-7.

    
    scint = WaveOpticsScintillator(sim_cfg)
    
    # Run simulation
    print("Running Wave Optics Simulation...")
    intensities, freqs = scint.simulate_dynamic_spectrum()
    
    obs = scint.compute_observables(intensities)
    print("Observables:", obs)
    
    # Check Modulation Index
    # For two screens in strong scattering (unresolved), m should be sqrt(2^2 - 1)? 
    # Or sum of variances?
    # Pradeep Eq 4.34: m = sqrt(3) ~ 1.732
    # Single screen strong scattering -> m = 1 (exponential statistics)
    
    m = obs['modulation_index']
    print(f"Measured m: {m:.3f}")
    
    # Plot
    plt.figure()
    plt.plot(freqs/1e9, intensities)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Intensity")
    plt.savefig("validate_wave_optics_spectrum.png")
    print("Saved plot to validate_wave_optics_spectrum.png")

if __name__ == "__main__":
    validate()
