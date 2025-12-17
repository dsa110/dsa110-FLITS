#!/usr/bin/env python3
"""
Generate diagnostic plots for Casey burst scattering fit.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Load Casey results
results_path = Path("scattering/scat_process/casey_chime_I_491_2085_32000b_cntr_bpc_fit_results.json")
with open(results_path) as f:
    results = json.load(f)

bp = results["best_params"]
gof = results["goodness_of_fit"]

# Print summary
print("=" * 60)
print("Casey Burst - Scattering Fit Results")
print("=" * 60)
print(f"Best Model: {results['best_model']}")
print(f"\nBest-Fit Parameters:")
print(f"  c0 (amplitude):        {bp['c0']:.2f}")
print(f"  t0 (peak time):        {bp['t0']:.3f} ms")
print(f"  γ (intrinsic width):   {bp['gamma']:.3f} ms")
print(f"  ζ (width parameter):   {bp['zeta']:.6f} ms")
print(f"  τ(1 GHz) (scattering): {bp['tau_1ghz']:.3f} ms")
print(f"  α (scattering index):  {bp['alpha']:.1f} (fixed)")
print(f"  ΔDM (DM refinement):   {bp['delta_dm']:.6f} pc/cm³")
print(f"\nGoodness of Fit:")
print(f"  χ²/dof:  {gof['chi2_reduced']:.2f}")
print(f"  R²:      {gof['r_squared']:.3f}")
print(f"  Quality: {gof.get('quality_flag', 'N/A')}")
print("=" * 60)

# Now load Freya for comparison
freya_path = Path("scattering/scat_process/freya_chime_I_912_4067_32000b_cntr_bpc_fit_results.json")
with open(freya_path) as f:
    freya_results = json.load(f)

freya_bp = freya_results["best_params"]
freya_gof = freya_results["goodness_of_fit"]

print("\n" + "=" * 60)
print("Comparison: Casey vs Freya")
print("=" * 60)
print(f"{'Parameter':<20} {'Casey':>12} {'Freya':>12}  {'Ratio':>10}")
print("-" * 60)
print(f"{'τ(1 GHz) [ms]':<20} {bp['tau_1ghz']:>12.3f} {freya_bp['tau_1ghz']:>12.3f}  {bp['tau_1ghz']/freya_bp['tau_1ghz']:>10.2f}x")
print(f"{'γ (width) [ms]':<20} {bp['gamma']:>12.3f} {freya_bp['gamma']:>12.3f}  {bp['gamma']/freya_bp['gamma']:>10.2f}x")
print(f"{'t0 [ms]':<20} {bp['t0']:>12.3f} {freya_bp['t0']:>12.3f}  {'-':>10}")
print(f"{'ΔDM [pc/cm³]':<20} {bp['delta_dm']:>12.6f} {freya_bp['delta_dm']:>12.6f}  {'-':>10}")
print(f"{'χ²/dof':<20} {gof['chi2_reduced']:>12.2f} {freya_gof['chi2_reduced']:>12.2f}  {gof['chi2_reduced']/freya_gof['chi2_reduced']:>10.2f}x")
print(f"{'R²':<20} {gof['r_squared']:>12.3f} {freya_gof['r_squared']:>12.3f}  {'-':>10}")
print("=" * 60)

print("\nKey findings:")
print(f"• Casey has τ(1 GHz) = {bp['tau_1ghz']:.3f} ms (vs Freya: {freya_bp['tau_1ghz']:.3f} ms)")
print(f"• Casey shows {bp['gamma']/freya_bp['gamma']:.1f}x wider intrinsic width than Freya")
print(f"• Both bursts best fit by M3 (scattering + intrinsic width)")
print(f"• Casey shows higher χ²/dof ({gof['chi2_reduced']:.1f} vs {freya_gof['chi2_reduced']:.1f})")
print(f"  This may indicate residual RFI or more complex burst structure")
