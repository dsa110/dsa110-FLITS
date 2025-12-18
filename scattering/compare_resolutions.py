#!/usr/bin/env python3
"""
compare_resolutions.py
======================

Compare ULTRA_FAST fit parameters against high-resolution data.

This script:
1. Loads best-fit parameters from the ULTRA_FAST run (low-resolution fit)
2. Loads high-resolution data (t_factor=4, f_factor=32)
3. Generates model using ULTRA_FAST parameters on high-res grid
4. Creates diagnostic plots comparing the low-res fit against high-res reality
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scattering.scat_analysis.burstfit_pipeline import BurstDataset
from scattering.scat_analysis.burstfit import FRBModel, FRBParams
from scattering.scat_analysis.config_utils import load_telescope_block

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "chime" / "casey_chime_I_491_2085_32000b_cntr_bpc.npy"
    output_dir = Path(__file__).parent / "test_output"
    
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    # Check for ULTRA_FAST results
    results_file = output_dir / "ultra_fast_test_fit_results.json"
    if not results_file.exists():
        print(f"ERROR: ULTRA_FAST results not found at {results_file}")
        print("Please run simple_runtime_test.py first to generate the fit.")
        sys.exit(1)
    
    print("="*60)
    print("Comparing Low-Res Fit vs High-Res Data")
    print("="*60)
    
    # Load ULTRA_FAST results
    print(f"\n1. Loading ULTRA_FAST fit results from {results_file.name}...")
    with open(results_file, 'r') as f:
        ultra_fast_results = json.load(f)
    
    best_params_dict = ultra_fast_results['best_params']
    ultra_fast_params = FRBParams(**best_params_dict)
    
    print(f"   Parameters from low-res fit:")
    print(f"     τ@1GHz = {ultra_fast_params.tau_1ghz:.4f} ms")
    print(f"     α = {ultra_fast_params.alpha:.3f}")
    print(f"     ζ = {ultra_fast_params.zeta:.4f} ms")
    print(f"     γ = {ultra_fast_params.gamma:.3f}")
    print(f"     t0 = {ultra_fast_params.t0:.3f} ms")
    
    # Load HIGH-RES data
    print(f"\n2. Loading HIGH-RESOLUTION data (t_factor=4, f_factor=32)...")
    
    # Load telescope config with proper path
    telcfg_path = base_dir / "scattering" / "configs" / "telescopes.yaml"
    telescope = load_telescope_block(telcfg_path, "chime")
    
    highres_dataset = BurstDataset(
        inpath=data_path,
        outpath=output_dir,
        name="highres_comparison",
        telescope=telescope,
        t_factor=4,
        f_factor=32,
    )
    
    print(f"   High-res data shape: {highres_dataset.data.shape}")
    print(f"   Freq range: {highres_dataset.freq[0]:.4f} - {highres_dataset.freq[-1]:.4f} GHz")
    print(f"   Time range: {highres_dataset.time[0]:.3f} - {highres_dataset.time[-1]:.3f} ms")
    
    # Generate model on high-res grid using ULTRA_FAST parameters
    print(f"\n3. Generating high-res model using low-res fit parameters...")
    highres_model_obj = FRBModel(
        time=highres_dataset.time,
        freq=highres_dataset.freq,
        data=highres_dataset.data,
        df_MHz=highres_dataset.df_MHz,
        dm_init=0.0,
    )
    
    # Generate model
    model_highres = highres_model_obj(ultra_fast_params, "M3")
    
    # Calculate residuals
    residual = highres_dataset.data - model_highres
    
    # Compute goodness of fit
    from scattering.scat_analysis.burstfit import goodness_of_fit
    gof = goodness_of_fit(
        highres_dataset.data,
        model_highres,
        highres_model_obj.noise_std,
        n_params=7,
    )
    
    print(f"\n4. Goodness of Fit on High-Res Data:")
    print(f"   χ²/dof = {gof['chi2_reduced']:.2f}")
    print(f"   R² = {gof['r_squared']:.3f}")
    print(f"   Quality: {gof['quality_flag']}")
    
    # Create diagnostic plot
    print(f"\n5. Creating diagnostic plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Common setup
    time = highres_dataset.time
    freq = highres_dataset.freq
    extent = [time[0], time[-1], freq[0], freq[-1]]
    
    # Panel 1: Data
    im1 = axes[0, 0].imshow(highres_dataset.data, extent=extent, aspect='auto', 
                            origin='lower', cmap='plasma')
    axes[0, 0].set_title('High-Res Data', fontsize=14, weight='bold')
    axes[0, 0].set_ylabel('Frequency [GHz]')
    plt.colorbar(im1, ax=axes[0, 0], label='Intensity')
    
    # Panel 2: Model (from low-res fit)
    im2 = axes[0, 1].imshow(model_highres, extent=extent, aspect='auto',
                            origin='lower', cmap='plasma')
    axes[0, 1].set_title('Model (Low-Res Fit Params)', fontsize=14, weight='bold')
    plt.colorbar(im2, ax=axes[0, 1], label='Intensity')
    
    # Panel 3: Residual
    vmax_res = np.nanpercentile(np.abs(residual), 99)
    im3 = axes[0, 2].imshow(residual, extent=extent, aspect='auto',
                            origin='lower', cmap='coolwarm', 
                            vmin=-vmax_res, vmax=vmax_res)
    axes[0, 2].set_title('Residual', fontsize=14, weight='bold')
    plt.colorbar(im3, ax=axes[0, 2], label='Intensity')
    
    # Panel 4: Time profiles
    data_profile = np.nansum(highres_dataset.data, axis=0)
    model_profile = np.nansum(model_highres, axis=0)
    
    axes[1, 0].plot(time, data_profile, 'k-', alpha=0.6, lw=1.5, label='Data')
    axes[1, 0].plot(time, model_profile, 'r-', lw=2, label='Model')
    axes[1, 0].set_xlabel('Time [ms]')
    axes[1, 0].set_ylabel('Integrated Flux')
    axes[1, 0].set_title('Time Profile', fontsize=14, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Panel 5: Frequency profiles
    data_spectrum = np.nansum(highres_dataset.data, axis=1)
    model_spectrum = np.nansum(model_highres, axis=1)
    
    axes[1, 1].plot(freq, data_spectrum, 'k-', alpha=0.6, lw=1.5, label='Data')
    axes[1, 1].plot(freq, model_spectrum, 'r-', lw=2, label='Model')
    axes[1, 1].set_xlabel('Frequency [GHz]')
    axes[1, 1].set_ylabel('Integrated Flux')
    axes[1, 1].set_title('Frequency Profile', fontsize=14, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Panel 6: Residual histogram
    res_flat = residual.flatten()
    res_flat = res_flat[np.isfinite(res_flat)]
    
    axes[1, 2].hist(res_flat / highres_model_obj.noise_std.mean(), 
                    bins=100, density=True, alpha=0.7, color='gray', label='Residuals')
    
    # Overlay Gaussian
    from scipy import stats
    x = np.linspace(-5, 5, 100)
    axes[1, 2].plot(x, stats.norm.pdf(x), 'r--', lw=2, label='N(0,1)')
    axes[1, 2].set_xlabel('Normalized Residual')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Residual Distribution', fontsize=14, weight='bold')
    axes[1, 2].legend()
    axes[1, 2].set_xlim(-5, 5)
    axes[1, 2].grid(alpha=0.3)
    
    # Add overall title with fit info
    fig.suptitle(
        f'Low-Res Fit (16×400) Applied to High-Res Data (31×521)\\n'
        f'χ²/dof = {gof["chi2_reduced"]:.2f}, R² = {gof["r_squared"]:.3f}, '
        f'τ@1GHz = {ultra_fast_params.tau_1ghz:.4f} ms, α = {ultra_fast_params.alpha:.2f}',
        fontsize=16, weight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_file = output_dir / "lowres_fit_highres_data_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to: {output_file}")
    
    # Also create a difference plot showing what was missed
    print(f"\n6. Creating detail comparison plot...")
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Zoom into burst region
    peak_idx = np.argmax(data_profile)
    window = 50  # samples around peak
    t_start, t_end = max(0, peak_idx - window), min(len(time), peak_idx + window)
    
    time_zoom = time[t_start:t_end]
    data_zoom = highres_dataset.data[:, t_start:t_end]
    model_zoom = model_highres[:, t_start:t_end]
    residual_zoom = residual[:, t_start:t_end]
    
    extent_zoom = [time_zoom[0], time_zoom[-1], freq[0], freq[-1]]
    
    im1 = axes2[0].imshow(data_zoom, extent=extent_zoom, aspect='auto',
                          origin='lower', cmap='plasma')
    axes2[0].set_title('Data (Zoomed)', fontsize=14, weight='bold')
    axes2[0].set_xlabel('Time [ms]')
    axes2[0].set_ylabel('Frequency [GHz]')
    plt.colorbar(im1, ax=axes2[0])
    
    im2 = axes2[1].imshow(model_zoom, extent=extent_zoom, aspect='auto',
                          origin='lower', cmap='plasma')
    axes2[1].set_title('Model from Low-Res Fit', fontsize=14, weight='bold')
    axes2[1].set_xlabel('Time [ms]')
    plt.colorbar(im2, ax=axes2[1])
    
    vmax_res = np.nanpercentile(np.abs(residual_zoom), 99)
    im3 = axes2[2].imshow(residual_zoom, extent=extent_zoom, aspect='auto',
                          origin='lower', cmap='coolwarm',
                          vmin=-vmax_res, vmax=vmax_res)
    axes2[2].set_title('Residual (Fine Structure)', fontsize=14, weight='bold')
    axes2[2].set_xlabel('Time [ms]')
    plt.colorbar(im3, ax=axes2[2])
    
    fig2.suptitle('Zoomed View: What Fine Structure Is Missed?',
                  fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file2 = output_dir / "lowres_fit_highres_data_zoom.png"
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Zoom plot saved to: {output_file2}")
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"\nKey Findings:")
    print(f"  • Low-res fit data: 16 freq × 400 time bins")
    print(f"  • High-res eval data: {highres_dataset.data.shape[0]} freq × {highres_dataset.data.shape[1]} time bins")
    print(f"  • χ²/dof on high-res: {gof['chi2_reduced']:.2f} (vs {ultra_fast_results.get('goodness_of_fit', {}).get('chi2_reduced', 'N/A')} on low-res)")
    print(f"  • R² on high-res: {gof['r_squared']:.3f}")
    
    if gof['chi2_reduced'] > 10:
        print(f"\n⚠️  High χ² suggests the low-res fit misses important structure!")
        print(f"    Fine spectral/temporal features are not captured.")
    elif gof['chi2_reduced'] > 2:
        print(f"\n⚠️  Moderate χ² - some structure may be missed.")
    else:
        print(f"\n✓ Good fit! Low-res parameters capture high-res structure well.")

if __name__ == "__main__":
    main()
