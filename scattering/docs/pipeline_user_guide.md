# FLITS Scattering Pipeline: User Guide

A step-by-step guide for running the scattering analysis pipeline on FRB data.

## Prerequisites

1. **Environment**: Create and activate the `flits` conda environment:
   ```bash
   cd /path/to/FLITS
   conda env create -f environment.yml
   conda activate flits
   ```
2. **Data**: `.npy` file containing the dynamic spectrum (freq × time)
3. **Telescope config**: Entry in `scattering/configs/telescopes.yaml`

---

## Quick Start (Command Line)

```bash
# Navigate to FLITS root
cd /path/to/FLITS

# Run the pipeline
python3 -m scattering.scat_analysis.burstfit_pipeline \
    data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy \
    --outpath ./scattering/scat_process/ \
    --telescope chime \
    --t_factor 4 \
    --f_factor 32 \
    --likelihood studentt \
    --alpha-fixed 4.0 \
    --fitting-method nested \
    --no-plot
```

---

## Command Arguments Explained

| Argument           | Description                            | Recommended Value                  |
| ------------------ | -------------------------------------- | ---------------------------------- |
| `data_path`        | Path to the `.npy` data file           | Required                           |
| `--outpath`        | Output directory for results           | `./scattering/scat_process/`       |
| `--telescope`      | Telescope name (must match YAML entry) | `chime`, `dsa`                     |
| `--t_factor`       | Time downsampling factor               | 4 (adjust for data size)           |
| `--f_factor`       | Frequency downsampling factor          | 32 (adjust for data size)          |
| `--likelihood`     | Likelihood function                    | `studentt` (robust to RFI)         |
| `--alpha-fixed`    | Fix scattering index                   | `4.0` (thin screen) or omit to fit |
| `--fitting-method` | Sampler choice                         | `nested` (recommended)             |
| `--no-plot`        | Skip plotting (faster)                 | Use for initial runs               |

---

## Typical Run Output

```
[BurstFit] detected 12 logical CPUs. Use how many workers? [default 11, 0 = serial] » 4
[BurstFit] starting Pool(4)
[INFO | burstfit.pipeline] Finding data-driven initial guess for MCMC...
[INFO | scattering.scat_analysis.burstfit_init] Peak time: t0 = 4.287 ms
[INFO | scattering.scat_analysis.burstfit_init] Scattering τ(1GHz) = 0.674 ms
[INFO | scattering.scat_analysis.burstfit_init] Scattering α = 3.67 ± 0.02
...
==================================================
Model Comparison Summary
==================================================
M0: log(Z) = -4162.49 ± 0.01  (ΔlnZ = -12568.8)
M1: log(Z) =  2151.57 ± 0.26  (ΔlnZ = -6254.7)
M2: log(Z) = -4162.49 ± 0.01  (ΔlnZ = -12568.8)
M3: log(Z) =  8406.30 ± 0.44  ← BEST

→ Best model by evidence: M3

[INFO | burstfit.pipeline] Best model: M3 | χ²/dof = 3.90
[INFO | burstfit.pipeline] Saved fit results to scattering/scat_process/freya_..._fit_results.json
```

---

## Understanding the Models

| Model  | Description                                       | When It Wins       |
| ------ | ------------------------------------------------- | ------------------ |
| **M0** | Gaussian only (no scattering, no intrinsic width) | Baseline           |
| **M1** | Gaussian + intrinsic width (no scattering)        | Unscattered bursts |
| **M2** | Gaussian + scattering (no intrinsic width)        | Heavily scattered  |
| **M3** | Gaussian + scattering + intrinsic width           | **Most FRBs**      |

**Model selection**: The model with highest `log(Z)` is selected. A difference of `ΔlnZ > 5` is considered "strong evidence."

---

## Output Files

| File                 | Description                                |
| -------------------- | ------------------------------------------ |
| `*_fit_results.json` | Best-fit parameters and validation metrics |
| `*_four_panel.pdf`   | Diagnostic plot (if `--no-plot` omitted)   |

### JSON Structure

```json
{
  "best_model": "M3",
  "best_params": {
    "c0": 83.99,
    "t0": 3.85,
    "gamma": 1.6,
    "zeta": 0.0004,
    "tau_1ghz": 0.168,
    "alpha": 4.0,
    "delta_dm": 0.019
  },
  "goodness_of_fit": {
    "chi2_reduced": 3.9,
    "r_squared": 0.68,
    "quality_flag": "FAIL"
  }
}
```

---

## Validation Metrics

| Metric         | Good Range                 | Interpretation                     |
| -------------- | -------------------------- | ---------------------------------- |
| `chi2_reduced` | 0.5 - 5.0                  | <1 = overfit, >5 = poor fit or RFI |
| `r_squared`    | > 0.5                      | Fraction of variance explained     |
| `quality_flag` | `PASS`, `MARGINAL`, `FAIL` | Overall quality assessment         |

> **Note**: `quality_flag = FAIL` with good `chi2_reduced` (3-5) usually indicates residual RFI or unmodeled burst structure, not a fundamental failure.

---

## Generating Diagnostic Plots

After the fit completes, generate plots using the saved parameters:

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scattering.scat_analysis.burstfit import FRBModel, FRBParams, downsample
from scipy.ndimage import gaussian_filter1d

# Load results
with open("scattering/scat_process/freya_..._fit_results.json") as f:
    results = json.load(f)
bp = results["best_params"]

# Load config
with open("scattering/configs/telescopes.yaml") as f:
    config = yaml.safe_load(f)["chime"]

# Load and preprocess data (match pipeline settings)
raw = np.load("data/chime/freya_chime_I_912_4067_32000b_cntr_bpc.npy")
raw = np.nan_to_num(raw.astype(np.float64))

# Bandpass correct
n_t_raw = raw.shape[1]
q = n_t_raw // 4
off_pulse_idx = np.r_[0:q, -q:0]
mu = np.nanmean(raw[:, off_pulse_idx], axis=1, keepdims=True)
sig = np.nanstd(raw[:, off_pulse_idx], axis=1, keepdims=True)
sig[sig < 1e-9] = np.nan
raw_corr = np.nan_to_num((raw - mu) / sig, nan=0.0)

# Downsample (match pipeline t_factor, f_factor)
t_factor, f_factor = 4, 32
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

# Center burst
prof = np.sum(data, axis=0)
sigma_samps = (0.1 / 2.355) / dt_ms
burst_idx = np.argmax(gaussian_filter1d(prof, sigma=sigma_samps))
shift = n_t // 2 - burst_idx
data = np.roll(data, shift, axis=1)

# Generate model
model = FRBModel(time=time, freq=freq, data=data, df_MHz=config["df_MHz_raw"] * f_factor)
p = FRBParams(**bp)
model_dyn = model(p, results["best_model"])

# Scale for visualization
scale = np.max(np.sum(data, axis=0)) / np.max(np.sum(model_dyn, axis=0))
model_scaled = model_dyn * scale

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... (add plotting code as needed)
plt.savefig("diagnostic_plot.png", dpi=150)
```

---

## Troubleshooting

| Symptom                              | Cause                         | Solution                        |
| ------------------------------------ | ----------------------------- | ------------------------------- |
| `chi2_reduced > 10^10`               | Masked channels in validation | Fixed in code                   |
| `alpha` not matching `--alpha-fixed` | Parameter injection bug       | Fixed in code                   |
| Model at wrong time                  | Time axis mismatch            | Check `dt_ms_raw` in YAML       |
| Burst upside-down                    | Frequency axis flipped        | Check `freq_descending` in YAML |
| M0/M2 "plateau"                      | Prior too wide for data       | Expected behavior               |

---

## Adding a New Telescope

1. Add entry to `scattering/configs/telescopes.yaml`:

   ```yaml
   new_telescope:
     df_MHz_raw: 0.5 # Channel width in MHz
     dt_ms_raw: 0.001 # Time sample in ms
     f_min_GHz: 1.0 # Bottom of band in GHz
     f_max_GHz: 2.0 # Top of band in GHz
     freq_descending: false # true if data[0] = high freq
   ```

2. Verify with a test run on your data.

---

## Example: Freya Burst Results

| Parameter  | Value                             |
| ---------- | --------------------------------- |
| Best Model | M3 (Scattering + Intrinsic Width) |
| log(Z)     | +8406.30                          |
| α          | 4.0 (fixed)                       |
| τ(1 GHz)   | 0.168 ms                          |
| t₀         | 3.85 ms                           |
| χ²/dof     | 3.90                              |
| R²         | 0.68                              |
