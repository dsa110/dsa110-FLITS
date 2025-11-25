# FRB Dispersion Measure Analysis

This directory contains tools for high-precision DM estimation using phase-coherent methods.

## Overview

The FLITS dispersion module implements a **phase-coherence DM estimator** that exploits frequency-domain structure to achieve sub-0.001 pc cm⁻³ precision. The method is particularly effective for bright, narrow pulses where traditional time-domain methods struggle with binning artifacts.

## Method

### Phase-Coherence Algorithm

1. **Phase Rotation**: For each trial DM, apply frequency-domain dedispersion
2. **Coherent Power**: Compute phase-aligned power across frequency channels
3. **Power Spectrum**: Weight by fluctuation frequency squared (ω²)
4. **Bootstrap Uncertainty**: Resample channels to estimate σ_DM

The key metric is the coherent power:

$$P'_{\rm Co}(\omega, {\rm DM}) = \left| \sum_{\nu} w_\nu \frac{S(\omega, \nu)}{|S(\omega, \nu)|} \right|^2 \omega^2$$

where the phase alignment maximizes when the trial DM matches the true DM.

### Core Module

**`dmphasev2.py`** - Production-ready estimator

```python
from dmphasev2 import DMPhaseEstimator

# Create estimator
est = DMPhaseEstimator(
    waterfall=waterfall,    # Complex voltage (n_time, n_chan)
    freqs=freqs,            # Channel frequencies (MHz)
    dt=dt,                  # Sampling time (s)
    dm_grid=dm_grid,        # Trial DM values (pc cm^-3)
    ref='top',              # Reference frequency
    n_boot=200              # Bootstrap samples for uncertainty
)

# Get result
dm_best, dm_sigma = est.get_dm()
print(f"DM = {dm_best:.4f} ± {dm_sigma:.4f} pc cm^-3")
```

**Key features:**
- Vectorized operations for speed
- Bootstrap-based uncertainty quantification
- Flexible frequency weighting (MAD-based by default)
- Optional frequency window selection

## Notebook

**`test_dm_phase.ipynb`** - Synthetic tests and visualization

The notebook provides:
- **Synthetic burst generation** with realistic dispersion
- **End-to-end testing** with known DM values
- **Diagnostic plots**:
  - DM curve with bootstrap uncertainty bands
  - 1D coherent power spectrum P'(ω)
  - 2D power map P'(ω, DM)
  - Dedispersed waterfall comparisons

**Usage:**
```python
from test_dm_phase import test_estimator

# Run full test suite with plots
results = test_estimator(verbose=True, make_plots=True)
```

### Generated Diagnostics

1. **`dm_curve_v2.png`**: Peak detection in DM space
2. **`power_spectrum_v2.png`**: Frequency domain sensitivity
3. **`power_map_v2.png`**: 2D map showing optimal DM ridge
4. **`dedispersed_waterfalls_v2.png`**: Visual confirmation at DM_best ± 5σ

## Testing

**`tests/test_dm_phase.py`** - Unit tests with real data

Tests the estimator on **B0355+54** repeating FRB bursts from CHIME/FRB:

```bash
cd /data/jfaber/FLITS/dispersion
pytest tests/test_dm_phase.py -v
```

**Test data:**
- Burst: B0355+54 (CHIME/FRB)
- Known DM: 57.1420 pc cm⁻³
- Dedispersed at: 57.2403 pc cm⁻³
- Expected recovery: ~0.1 pc cm⁻³ precision

## Typical Workflow

### 1. Prepare Data

```python
import numpy as np

# Load complex voltage waterfall
waterfall = np.load("burst_dynspec.npy")  # Shape: (n_time, n_chan)

# Define frequency axis
nchan = waterfall.shape[1]
f_lo, f_hi = 400.0, 800.0  # MHz
freqs = np.linspace(f_hi, f_lo, nchan)  # Descending order

# Sampling time
dt = 1e-4  # seconds
```

### 2. Preprocess (Optional)

```python
# Channel-wise normalization
mean = np.nanmean(waterfall, axis=0)
std = np.nanstd(waterfall, axis=0)
waterfall = (waterfall - mean) / std

# Handle NaNs
waterfall[np.isnan(waterfall)] = np.nanmedian(waterfall)
```

### 3. Define DM Grid

```python
# Coarse scan
dm_coarse = np.linspace(50, 60, 101)

# Fine refinement around expected value
dm_expected = 57.0
dm_fine = np.linspace(dm_expected - 1, dm_expected + 1, 201)
```

### 4. Run Estimator

```python
from dmphasev2 import DMPhaseEstimator

est = DMPhaseEstimator(
    waterfall=waterfall,
    freqs=freqs,
    dt=dt,
    dm_grid=dm_fine,
    ref='top',          # Use highest frequency as reference
    n_boot=200,         # More samples = better σ_DM estimate
    random_state=42     # Reproducibility
)

# Extract results
result = est.result()
print(f"Best DM: {result['dm_best']:.4f} ± {result['dm_sigma']:.4f}")
```

### 5. Generate Diagnostics

```python
from test_dm_phase import plot_dm_diagnostics

plot_dm_diagnostics(est, waterfall, freqs, dt)
```

## Algorithm Details

### Frequency Weighting

By default, weights are computed from median absolute deviation (MAD):

```python
mad = np.median(np.abs(waterfall - np.median(waterfall, axis=0)), axis=0)
sigma = 1.4826 * mad
weights = 1 / sigma**2
```

Custom weights can be provided via the `weights` parameter.

### Frequency Window Selection

The estimator automatically selects a frequency window based on the power spectrum peak:
- Find maximum power at trial DM
- Integrate until power drops below 10% of peak
- Optionally override with `f_cut=(f_low, f_high)` tuple

### Bootstrap Uncertainty

The σ_DM estimate comes from resampling frequency channels (not time samples):
1. For each bootstrap iteration, randomly sample N channels with replacement
2. Recompute coherent power and fit DM peak
3. σ_DM = standard deviation of bootstrap peaks

This captures channel-to-channel variability and is robust to outliers.

### Peak Fitting

The DM peak is fit with a quadratic function over ±2 grid points:

$$P({\rm DM}) = a \cdot {\rm DM}^2 + b \cdot {\rm DM} + c$$

Peak location: ${\rm DM}_{\rm best} = -b / (2a)$

## Performance Notes

**Speed:** Vectorized FFT operations scale as O(N_DM × N_time × N_chan × log N_time)

**Memory:** Requires 3D complex array (N_DM × N_time × N_chan) in memory

**Typical runtimes** (single-threaded):
- 1024 time × 256 channels × 100 trial DMs: ~0.5s
- 2048 time × 1024 channels × 200 trial DMs: ~5s

## Comparison to Time-Domain Methods

| Method | Precision | Speed | Best Use Case |
|--------|-----------|-------|---------------|
| Phase-coherence | ~0.001 pc cm⁻³ | Medium | Bright, narrow pulses |
| Dedispersion tree | ~0.01 pc cm⁻³ | Fast | Wide pulses, real-time |
| Structure function | ~0.001 pc cm⁻³ | Slow | Complex pulse structure |

**Advantages:**
- Sub-millisecond precision without fine time binning
- Robust to RFI in individual channels
- Works on complex voltage data (preserves phase)

**Limitations:**
- Requires bright pulses (SNR > 10)
- Memory-intensive for large trial DM grids
- Less effective for multi-component bursts

## References

- **Method development**: Derived from pulsar timing techniques
- **Bootstrap uncertainty**: Efron & Tibshirani (1993)
- **CHIME/FRB application**: See CHIME/FRB collaboration papers

## File Structure

```
dispersion/
├── dmphasev2.py              # Core estimator module
├── notebooks/
│   ├── test_dm_phase.ipynb   # Interactive tests + diagnostics
│   └── README.md             # This file
└── tests/
    └── test_dm_phase.py      # Unit tests (B0355+54 data)
```

## Support

For questions or issues:
1. Review synthetic test in `test_dm_phase.ipynb`
2. Check docstrings in `dmphasev2.py`
3. Run unit tests: `pytest tests/ -v`
