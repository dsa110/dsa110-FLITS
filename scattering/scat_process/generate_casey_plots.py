#!/usr/bin/env python3
"""
Generate diagnostic plots for Casey burst scattering fit.
Matches the plotting style used for Freya.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scattering.scat_analysis.burstfit import FRBModel, FRBParams, downsample
from scipy.ndimage import gaussian_filter1d

# Load Casey results
print("Loading Casey fit results...")
results_path = Path("scattering/scat_process/casey_chime_I_491_2085_32000b_cntr_bpc_fit_results.json")
with open(results_path) as f:
    results = json.load(f)

bp = results["best_params"]
best_model = results["best_model"]

# Load telescope config
with open("scattering/configs/telescopes.yaml") as f:
    config = yaml.safe_load(f)["chime"]

# Load and preprocess data (match pipeline settings)
print("Loading and preprocessing data...")
raw = np.load("data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy")
raw = np.nan_to_num(raw.astype(np.float64))

# Bandpass correction
n_t_raw = raw.shape[1]
q = n_t_raw // 4
off_pulse_idx = np.r_[0:q, -q:0]
mu = np.nanmean(raw[:, off_pulse_idx], axis=1, keepdims=True)
sig = np.nanstd(raw[:, off_pulse_idx], axis=1, keepdims=True)
sig[sig < 1e-9] = np.nan
raw_corr = np.nan_to_num((raw - mu) / sig, nan=0.0)

# Downsample (match pipeline t_factor, f_factor)
t_factor, f_factor = 4, 32
print(f"Downsampling by factors: time={t_factor}, freq={f_factor}")
data = downsample(raw_corr, f_factor, t_factor)

# Apply same trim as pipeline (outer_trim=0.45)
outer_trim = 0.45
n_trim = int(outer_trim * data.shape[1])
data = data[:, n_trim:-n_trim] if n_trim > 0 else data

# Build axes
n_ch, n_t = data.shape
dt_ms = config["dt_ms_raw"] * t_factor
freq = np.linspace(config["f_min_GHz"], config["f_max_GHz"], n_ch)
time = np.arange(n_t) * dt_ms

print(f"Data shape: {data.shape}")
print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} ms, dt={dt_ms:.4f} ms")
print(f"Freq range: {freq[0]:.3f} to {freq[-1]:.3f} GHz")

# Center burst
prof = np.sum(data, axis=0)
sigma_samps = (0.1 / 2.355) / dt_ms
burst_idx = np.argmax(gaussian_filter1d(prof, sigma=sigma_samps))
shift = n_t // 2 - burst_idx
data = np.roll(data, shift, axis=1)

print(f"Burst centered at sample {n_t // 2}")

# Generate model
print(f"Generating {best_model} model...")
model = FRBModel(time=time, freq=freq, data=data, df_MHz=config["df_MHz_raw"] * f_factor)
p = FRBParams(**bp)
model_dyn = model(p, best_model)

# Scale for visualization
scale = np.max(np.sum(data, axis=0)) / np.max(np.sum(model_dyn, axis=0))
model_scaled = model_dyn * scale

print(f"Model scaling factor: {scale:.3f}")

# Calculate residuals
residual = data - model_scaled

# Generate time profiles
data_prof = np.sum(data, axis=0)
model_prof = np.sum(model_scaled, axis=0)
residual_prof = data_prof - model_prof

# Create four-panel diagnostic plot
print("Creating four-panel diagnostic plot...")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel 1: Data dynamic spectrum
ax1 = fig.add_subplot(gs[0, 0])
vmax = np.percentile(data, 99.5)
vmin = np.percentile(data, 0.5)
im1 = ax1.imshow(data, aspect='auto', origin='lower', 
                 extent=[time[0], time[-1], freq[0], freq[-1]],
                 vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
ax1.set_xlabel('Time (ms)', fontsize=11)
ax1.set_ylabel('Frequency (GHz)', fontsize=11)
ax1.set_title('Data', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Intensity')

# Panel 2: Model dynamic spectrum
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(model_scaled, aspect='auto', origin='lower',
                 extent=[time[0], time[-1], freq[0], freq[-1]],
                 vmin=vmin, vmax=vmax, cmap='viridis', interpolation='nearest')
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylabel('Frequency (GHz)', fontsize=11)
ax2.set_title(f'Model ({best_model})', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Intensity')

# Panel 3: Residual dynamic spectrum
ax3 = fig.add_subplot(gs[1, 0])
res_vmax = np.percentile(np.abs(residual), 99)
im3 = ax3.imshow(residual, aspect='auto', origin='lower',
                 extent=[time[0], time[-1], freq[0], freq[-1]],
                 vmin=-res_vmax, vmax=res_vmax, cmap='RdBu_r', interpolation='nearest')
ax3.set_xlabel('Time (ms)', fontsize=11)
ax3.set_ylabel('Frequency (GHz)', fontsize=11)
ax3.set_title('Residual (Data - Model)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='Intensity')

# Panel 4: Time profiles
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time, data_prof, 'k-', linewidth=1.5, label='Data', alpha=0.7)
ax4.plot(time, model_prof, 'r-', linewidth=2, label='Model', alpha=0.8)
ax4.plot(time, residual_prof, 'b-', linewidth=1, label='Residual', alpha=0.6)
ax4.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax4.set_xlabel('Time (ms)', fontsize=11)
ax4.set_ylabel('Intensity', fontsize=11)
ax4.set_title('Frequency-Averaged Profiles', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)

# Panel 5: Parameters and fit quality
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Create text summary
gof = results["goodness_of_fit"]
text_info = f"""
Best Model: {best_model}

Best-Fit Parameters:
  • Amplitude (c₀):          {bp['c0']:.2f}
  • Peak time (t₀):          {bp['t0']:.4f} ms
  • Intrinsic width (γ):     {bp['gamma']:.4f} ms
  • Width parameter (ζ):     {bp['zeta']:.6f} ms
  • Scattering τ(1 GHz):     {bp['tau_1ghz']:.4f} ms
  • Scattering index (α):    {bp['alpha']:.2f} (fixed)
  • DM refinement (ΔDM):     {bp['delta_dm']:.6f} pc/cm³

Goodness of Fit:
  • χ²/dof:     {gof['chi2_reduced']:.2f}
  • R²:         {gof['r_squared']:.4f}
  • Quality:    {gof.get('quality_flag', 'N/A')}
  
Processing:
  • Downsampling: {t_factor}× (time), {f_factor}× (freq)
  • Likelihood: Student-t
  • Fitting method: Nested sampling
"""

ax5.text(0.05, 0.95, text_info, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Casey Burst - Scattering Analysis Diagnostics', 
             fontsize=14, fontweight='bold', y=0.995)

# Save figure
output_path = Path("scattering/scat_process/casey_fit_diagnostic.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved diagnostic plot to: {output_path}")

print(f"\nPlot generation complete!")
print(f"Resolution: 150 DPI")
print(f"Figure size: 14 x 10 inches")
