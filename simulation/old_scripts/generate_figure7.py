```python
# generate_figure7.py

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from frb_scintillator_v3 import SimCfg, ScreenCfg, FRBScintillator

def generate_figure7_plot():
    """
    Generates and plots a 2D dynamic spectrum similar to the
    bottom-right panel of Figure 7 in Pradeep et al. (2025).
    This uses the "completely resolved" scenario.
    """
    
    # Configuration based on the "completely resolved" case from Table B.1
    cfg_resolved = SimCfg(
        nu0=800 * u.MHz,
        bw=25 * u.MHz,
        nchan=1024, # Will be determined by STFT nperseg
        D_mw=1.29 * u.kpc,
        z_host=0.0337,
        D_host_src=3.0 * u.kpc,
        mw=ScreenCfg(N=200, L=4.12 * u.AU, rng_seed=2025),
        host=ScreenCfg(N=200, L=165 * u.AU, rng_seed=2026),
        intrinsic_pulse="gauss",
        pulse_width=0.1 * u.ms,
    )

    sim = FRBScintillator(cfg_resolved)
    
    # Define simulation time parameters
    # The duration needs to be long enough to capture the full scattering tail
    # Max delay is approx tau_host ~ D_eff_host * theta_host^2 / 2c
    # theta_host ~ L_host / D_host. Let's estimate and add buffer.
    max_delay_est = sim.deff_host_m * (cfg_resolved.host.L.to(u.m).value / sim.D_host_m)**2 / (2*3e8)
    sim_duration = (max_delay_est * 1.5 + 5 * cfg_resolved.pulse_width.to(u.s).value) * u.s
    time_resolution = cfg_resolved.pulse_width / 5 

    print("Synthesising dynamic spectrum...")
    I_t_nu, time_axis, freq_axis = sim.synthesise_dynamic_spectrum(
        time_res=time_resolution,
        duration=sim_duration
    )
    print("Synthesis complete.")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True, 
        gridspec_kw={'height_ratios': [1, 3]}
    )
    fig.suptitle("Dynamic Spectrum (Resolved Case, RP > 1)", fontsize=16)

    # Top panel: Time series (pulse profile)
    pulse_profile = I_t_nu.mean(axis=1)
    ax1.plot(time_axis * 1e3, pulse_profile / pulse_profile.max(), color='k')
    ax1.set_ylabel("Normalized Intensity")
    ax1.set_xlim(0, time_axis.max() * 1e3)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Dynamic spectrum
    # Use logarithmic scale to see faint structures
    vmin = np.percentile(I_t_nu[I_t_nu > 0], 5)
    vmax = np.percentile(I_t_nu, 99.5)
    
    im = ax2.imshow(
        I_t_nu.T,
        aspect='auto',
        origin='lower',
        extent=[
            time_axis.min() * 1e3, time_axis.max() * 1e3,
            freq_axis.min() / 1e6, freq_axis.max() / 1e6
        ],
        cmap='viridis',
        norm=plt.cm.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")
    fig.colorbar(im, ax=ax2, label="Intensity (log scale)")
    
    plt.show()

if __name__ == '__main__':
    generate_figure7_plot()