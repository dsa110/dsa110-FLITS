# ============================================================================
# generate_figure16.py
# New script to perform broadband analysis and replicate Figure 16.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
from tqdm import trange
from frb_scintillator_v6 import SimCfg, ScreenCfg, FRBScintillator

def broadband_analysis():
    """
    Performs a broadband analysis to replicate Figure 16 from the paper,
    showing the evolution of scintillation bandwidth with frequency.
    """
    
    # --- Define Reference Configuration ---
    # We define the screen sizes at a reference frequency.
    nu_ref = 1.0 * u.GHz
    L_mw_ref = 2.0 * u.AU
    L_host_ref = 10.0 * u.AU
    
    base_cfg_dict = {
        "D_mw": 1.0 * u.kpc, "z_host": 0.05, "D_host_src": 4.0 * u.kpc,
        "nchan": 2048, "bw": 2.0 * u.MHz
    }
    
    # --- Define Simulation Frequencies ---
    sim_freqs = np.linspace(300, 1400, 20) * u.MHz
    results = {"freqs": [], "nu_s_mw": [], "rp": []}
    
    print("Running broadband analysis...")
    for nu_sim in trange(len(sim_freqs), desc="Simulating frequencies"):
        nu0 = sim_freqs[nu_sim]
        
        # --- Scale Screen Sizes ---
        # L(nu) = L_ref * (nu / nu_ref)^-2
        scale_factor = (nu0 / nu_ref).value**-2
        L_mw_scaled = L_mw_ref * scale_factor
        L_host_scaled = L_host_ref * scale_factor
        
        # --- Create and Run Simulation for this Frequency ---
        cfg = SimCfg(
            **base_cfg_dict,
            nu0=nu0,
            mw=ScreenCfg(N=300, L=L_mw_scaled, rng_seed=int(nu0.to(u.Hz).value)),
            host=ScreenCfg(N=300, L=L_host_scaled, rng_seed=int(nu0.to(u.Hz).value)+1)
        )
        sim = FRBScintillator(cfg)

        spec = sim.simulate_time_integrated_spectrum()
        corr, lags = sim.acf(spec)

        # We only care about the broad component for this analysis
        nu_s_mw, _ = sim.fit_acf(corr, lags)

        if not np.isnan(nu_s_mw):
            results["freqs"].append(nu0.to(u.MHz).value)
            # The result is already in Hz, no need to divide by 1e6 here
            results["nu_s_mw"].append(nu_s_mw) 
            results["rp"].append(sim.resolution_power())
            
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    freqs_mhz = np.array(results["freqs"])
    nu_s_mhz = np.array(results["nu_s_mw"]) / 1e6
    rps = np.array(results["rp"])

    ax.loglog(freqs_mhz, nu_s_mhz, 'o', color='k', label="Simulation values")
    
    # --- Power-Law Fits ---
    def power_law(x, a, b):
        return a * x**b

    # Unresolved regime (high frequency, low RP)
    unresolved_mask = rps < 1
    if np.sum(unresolved_mask) > 1:
        popt, _ = curve_fit(power_law, freqs_mhz[unresolved_mask], nu_s_mhz[unresolved_mask])
        ax.plot(freqs_mhz[unresolved_mask], power_law(freqs_mhz[unresolved_mask], *popt),
                'g--', lw=2, label=f'Unresolved fit ($\\alpha \\approx {popt[1]:.2f}$)')

    # Highly resolved regime (low frequency, high RP)
    resolved_mask = rps > 3
    if np.sum(resolved_mask) > 1:
        popt, _ = curve_fit(power_law, freqs_mhz[resolved_mask], nu_s_mhz[resolved_mask])
        ax.plot(freqs_mhz[resolved_mask], power_law(freqs_mhz[resolved_mask], *popt),
                'b--', lw=2, label=f'Highly Resolved fit ($\\alpha \\approx {popt[1]:.2f}$)')

    # Add RP lines for context
    rp1_freq = np.interp(1.0, rps[::-1], freqs_mhz[::-1])
    rp3_freq = np.interp(3.0, rps[::-1], freqs_mhz[::-1])
    ax.axvline(rp1_freq, color='gray', ls='--', label='RP=1')
    ax.axvline(rp3_freq, color='gray', ls=':', label='RP=3')

    ax.set_xlabel("Observing Frequency (MHz)")
    ax.set_ylabel("Scintillation Bandwidth, $\\nu_{s,MW}$ (MHz)")
    ax.set_title("Broadband Scintillation Analysis ($\nu_s \propto \\nu^\\alpha$)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.show()