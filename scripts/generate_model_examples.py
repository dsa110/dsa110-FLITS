
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure flits is in path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except ImportError:
    print("SciencePlots not found, using default style")

from scattering.scat_analysis.burstfit import FRBModel, FRBParams

def generate_model_examples():
    # 1. Setup Grid (DSA-110-like)
    n_freq = 128
    n_time = 256
    freq = np.linspace(1.3, 1.5, n_freq)  # GHz
    time = np.linspace(-5, 15, n_time)    # ms
    
    # DM for smearing calculation
    dm_val = 500.0
    
    # Initialize Model Wrapper
    # noise_std is needed for likelihood but not for raw simulation, but constructor might need it
    # We can pass None for simulation if we are careful, or dummy noise
    model = FRBModel(
        time=time, 
        freq=freq, 
        dm_init=dm_val,
        df_MHz=(freq[1]-freq[0])*1000.0,
        noise_std=np.ones((n_freq, n_time)) # Dummy
    )

    # 2. Define Parameters for Single Models
    # Shared basics
    c0 = 10.0
    t0 = 2.0
    gamma = -1.5 # Typical astrophysical spectral index
    
    # M0: Unresolved (Pure Smearing)
    # zeta=0, tau=0
    p_m0 = FRBParams(c0=c0, t0=t0, gamma=gamma, zeta=0.0, tau_1ghz=0.0, alpha=4.4, delta_dm=0.0)
    
    # M1: Resolved (Intrinsic Width)
    # zeta=2.0 ms
    p_m1 = FRBParams(c0=c0, t0=t0, gamma=gamma, zeta=2.0, tau_1ghz=0.0, alpha=4.4, delta_dm=0.0)
    
    # M2: Scattered Unresolved (Fixed Alpha=4.0 for demo)
    # zeta=0, tau=2.5 ms (Requested update)
    p_m2 = FRBParams(c0=c0, t0=t0, gamma=gamma, zeta=0.0, tau_1ghz=2.5, alpha=4.0, delta_dm=0.0)
    
    # M3: Scattered Resolved (Free Alpha=2.5 for demo contrast)
    # zeta=0.75 ms (1/2 of previous 1.5), tau=5.0 ms (Requested update)
    p_m3 = FRBParams(c0=c0, t0=t0, gamma=gamma, zeta=0.75, tau_1ghz=5.0, alpha=2.5, delta_dm=0.0)

    # 3. Generate Single Model Data
    data_m0 = model(p_m0, "M0")
    data_m1 = model(p_m1, "M1")
    data_m2 = model(p_m2, "M2")
    data_m3 = model(p_m3, "M3")

    # 4. Generate Mixed Multi-Component Model
    # Component 1: M1 (Early, Resolved) at t=0
    p_c1 = FRBParams(c0=10.0, t0=0.0, gamma=0.0, zeta=1.0, tau_1ghz=0.0, alpha=4.4, delta_dm=0.0)
    # Component 2: M3 (Middle, Scat+Res) at t=5
    p_c2 = FRBParams(c0=15.0, t0=5.0, gamma=0.0, zeta=1.0, tau_1ghz=2.0, alpha=4.0, delta_dm=0.0)
    # Component 3: M2 (Late, Scat Unres) at t=10
    p_c3 = FRBParams(c0=8.0, t0=10.0, gamma=0.0, zeta=0.0, tau_1ghz=4.0, alpha=4.0, delta_dm=0.0)
    
    data_mixed = model(p_c1, "M1") + model(p_c2, "M3") + model(p_c3, "M2")

    # 5. Plotting
    # We need a layout that accommodates marginals for each of the 5 panels.
    # GridSpec approach: 5 main columns. Each column is a 2x2 grid (Main, Right; Top, Empty).
    
    fig = plt.figure(figsize=(24, 5))
    
    # Outer GridSpec: 1 row, 5 columns
    outer_gs = fig.add_gridspec(1, 5, wspace=0.3)
    
    models = [
        (data_m0, "M0: Unresolved\n(DM Smear Only)"),
        (data_m1, "M1: Resolved\n(Intrinsic Width)"),
        (data_m2, "M2: Scattered\nUnresolved"),
        (data_m3, "M3: Scattered\nResolved"),
        (data_mixed, "Mixed Model\n(M1 + M3 + M2)")
    ]
    
    extent = [time[0], time[-1], freq[0], freq[-1]]
    
    for i, (data, title) in enumerate(models):
        # Normalize
        d_norm = data / np.max(data)
        
        # Inner GridSpec for this panel: 2 rows (Top, Main), 2 cols (Main, Right)
        # Height ratios: 1 (Top) : 4 (Main)
        # Width ratios: 4 (Main) : 1 (Right)
        inner_gs = outer_gs[i].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        
        ax_main = fig.add_subplot(inner_gs[1, 0])
        ax_top = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)
        
        # Plot Main Waterfall
        im = ax_main.imshow(d_norm, aspect='auto', origin='lower', extent=extent, cmap='magma', vmin=0, vmax=1)
        
        # Plot Top Marginal (Time Series) - Sum over freq
        ts = np.sum(d_norm, axis=0)
        ts /= np.max(ts)
        ax_top.plot(time, ts, 'k-', lw=1.5)
        ax_top.set_ylim(0, 1.1)
        ax_top.axis('off')
        ax_top.set_title(title, fontsize=12, pad=10)
        
        # Plot Right Marginal (Spectrum) - Sum over time
        spec = np.sum(d_norm, axis=1)
        spec /= np.max(spec)
        ax_right.plot(spec, freq, 'k-', lw=1.5)
        ax_right.set_xlim(0, 1.1)
        ax_right.axis('off')
        
        # Axis Labels
        ax_main.set_xlabel("Time (ms)")
        if i == 0:
            ax_main.set_ylabel("Frequency (GHz)")
        else:
            ax_main.set_yticklabels([])
            
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
    # cbar.set_label("Normalized Intensity")
    
    # plt.tight_layout() # constrained_layout is better for nested gridspecs usually, but manual is safer here
    out_path = "model_examples.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    generate_model_examples()
