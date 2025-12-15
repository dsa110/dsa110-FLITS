# FLITS Scattering Broadening Integration

## Architecture Overview

This document summarizes the integrated scatter-broadening architecture folded into the FLITS codebase with optional enhancements (physical priors).

### Core Components

1. **Extended Parameters** (`flits/params.py`)

   - `FRBParams` now includes:
     - `tau_ms` (float): Scattering timescale at 1 GHz (milliseconds); 0.0 disables scattering.
     - `tau_alpha` (float): Frequency scaling exponent (default 4.4 for Kolmogorov turbulence).
   - Backward compatible: existing code works unchanged.

2. **Scattering Utilities** (`flits/scattering/broaden.py`)

   - `scatter_broaden()`: Applies exponential broadening via FFT convolution.
     - Supports 1D and 2D (per-frequency) signals.
     - Normalized kernel preserves total flux.
     - Causal or symmetric options.
   - `tau_per_freq()`: Computes per-frequency timescale via power-law scaling.
     - τ(ν) = τ_ref × (ν_ref / ν)^α
   - **Physical priors**:
     - `log_normal_prior()`: Log-normal distribution for tau_ms.
     - `gaussian_prior()`: Gaussian distribution for alpha.

3. **Simulation** (`flits/models.py`)

   - `FRBModel.simulate()` now includes:
     - Optional `tau_sc_ms` parameter (direct override for tau_ms).
     - Optional `tau_alpha` parameter (direct override for freq scaling exponent).
     - Automatic frequency-dependent τ(ν) if both tau_ms > 0 and tau_alpha > 0.
     - Uses `scatter_broaden()` for efficient kernel convolution.
   - Example:

     ```python
     from flits.params import FRBParams
     from flits.models import FRBModel
     import numpy as np

     # Create params with scatter
     params = FRBParams(
         dm=500.0,
         amplitude=1.0,
         t0=50.0,
         width=1.0,
         tau_ms=1.5,        # Scattering at 1 GHz
         tau_alpha=4.4,     # Kolmogorov
     )

     model = FRBModel(params)
     t = np.linspace(0, 200, 256)
     freqs = np.linspace(1280, 1530, 64)
     dynspec = model.simulate(t, freqs)  # Dispersed + scatter-broadened
     ```

4. **Fitting** (`scattering/scat_analysis/burstfit.py`)
   - Enhanced `FRBModel` class (existing, now with utilities):
     - Model key `"M3"` includes `tau_1ghz` and `alpha` parameters.
     - Uses `scatter_broaden()` from shared utilities for consistency.
   - New `apply_physical_priors()` function:
     - Applies log-normal prior on `tau_1ghz`.
     - Applies Gaussian prior on `alpha`.
     - Called by `_log_prob_wrapper()` in MCMC sampling.
   - Updated `_log_prob_wrapper()` signature:
     - New parameter: `tau_prior=(mu, sigma)` for log-normal prior on tau.
     - Backward compatible: existing calls work unchanged.
   - Example (with physical priors):

     ```python
     from scattering.scat_analysis.burstfit import FRBModel, apply_physical_priors

     # Initialize fit model
     fit = FRBModel(time, freq, data=data, dm_init=500.0)

     # Fit with Kolmogorov priors
     tau_prior = (np.log(1.0), 0.5)      # log-normal: log(τ) ~ N(-0.3, 0.25)
     alpha_prior = (4.4, 0.2)             # Gaussian: α ~ N(4.4, 0.04)

     # Pass to sampler for MCMC
     sampler = fit.sample(..., tau_prior=tau_prior, alpha_prior=alpha_prior)
     ```

### Backward Compatibility

- All existing code continues to work:
  - `FRBModel` without scattering: set `tau_ms=0` (default).
  - Old fitting calls: tau and alpha parameters are optional.
  - No breaking changes to APIs.

### Physical Priors (Enhancements)

#### Log-Normal Prior on Scattering Timescale

```python
from flits.scattering import log_normal_prior
import numpy as np

# τ ~ LogNormal(μ=log(1 ms), σ=0.5)
logp = log_normal_prior(x=tau_ms_value, mu=np.log(1.0), sigma=0.5)
```

- Typical for instrument/ISM timescales.
- Prevents unphysical negative tau values.

#### Gaussian Prior on Frequency Exponent

```python
from flits.scattering import gaussian_prior

# α ~ N(4.4, 0.2²)  [Kolmogorov default]
logp = gaussian_prior(x=alpha_value, mu=4.4, sigma=0.2)
```

- Constrains α to physically reasonable range: 4.0 (thin screen) to 4.4 (Kolmogorov).

### Usage Examples

#### 1. Simulate Broadened FRB (No Scintillation)

```python
from flits.params import FRBParams
from flits.models import FRBModel
import numpy as np
import matplotlib.pyplot as plt

# Ground truth: dispersed + scattered pulse
params = FRBParams(
    dm=500.0,
    amplitude=1.0,
    t0=50.0,
    width=1.0,
    tau_ms=2.0,      # 2 ms scattering timescale
    tau_alpha=4.4,   # Frequency-dependent scaling
)

model = FRBModel(params)
t = np.linspace(0, 200, 512)
freqs = np.linspace(1280, 1530, 256)
dynspec = model.simulate(t, freqs)

# Plot
plt.imshow(dynspec, aspect='auto', origin='lower', extent=[t.min(), t.max(), freqs.min(), freqs.max()])
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (MHz)')
plt.title(f'Dispersed + Scattered FRB (τ=2 ms, α=4.4)')
plt.show()
```

#### 2. Fit with Physical Priors (sketch)

```python
from scattering.scat_analysis.burstfit import FRBModel as FitModel
import numpy as np

# Data
fit_model = FitModel(time, freq, data=obs_dynspec, dm_init=dm_est)

# Fit (simplified: see scattering/scripts/verify_scattering.py for full example)
# With Kolmogorov + log-normal tau priors
tau_prior = (np.log(1.5), 0.3)    # log(τ) ~ N(log(1.5), 0.09)
alpha_prior = (4.4, 0.2)          # α ~ N(4.4, 0.04)

# Pass to sampler
# sampler = fit_model.sample(..., tau_prior=tau_prior, alpha_prior=alpha_prior)
```

#### 3. Per-Frequency τ Scaling

```python
from flits.scattering import tau_per_freq
import numpy as np

# Compute τ(ν) across band
freqs = np.linspace(1280, 1530, 256)  # MHz
tau_ref = 1.5                          # ms at 1 GHz
alpha = 4.4                            # Kolmogorov

tau_array = tau_per_freq(tau_ref, freqs, alpha)
# tau_array[i] is the scattering timescale at freqs[i]
```

### Testing

Run integration tests:

```bash
cd /Users/jakobfaber/Documents/research/caltech/ovro/dsa110/FLITS
python3 -c "
from flits.params import FRBParams
from flits.models import FRBModel
from flits.scattering import scatter_broaden, tau_per_freq, log_normal_prior, gaussian_prior
import numpy as np

# Test all components
params = FRBParams(dm=500, amplitude=1.0, t0=50, width=1.0, tau_ms=1.5, tau_alpha=4.4)
model = FRBModel(params)
t = np.linspace(0, 200, 256)
freqs = np.linspace(1280, 1530, 32)
dynspec = model.simulate(t, freqs)
print('✓ Full stack works:', dynspec.shape)
"
```

See also: `scattering/scripts/verify_scattering.py` for a complete synthetic data generation + fitting example.

### Files Modified/Created

- **Modified**:

  - `flits/params.py`: Added `tau_ms`, `tau_alpha` to `FRBParams`.
  - `flits/models.py`: Updated `FRBModel.simulate()` to use scattering utilities.
  - `scattering/scat_analysis/burstfit.py`: Added `apply_physical_priors()`, updated imports and `_log_prob_wrapper()`.
  - `flits/scattering/__init__.py`: Exports scattering utilities.

- **Created**:
  - `flits/scattering/broaden.py`: Scattering utilities (kernel, priors).
  - `scattering/scripts/verify_scattering.py`: Integration verification script.

### Design Principles

1. **Minimal invasiveness**: Existing code unchanged; new features are opt-in.
2. **Reusable kernels**: Single `scatter_broaden()` utility used by both simulation and fitting.
3. **Physical motivation**: Priors tied to observed ISM physics (Kolmogorov, log-normal timescales).
4. **Frequency awareness**: Native support for τ(ν) scaling without code duplication.
5. **MCMC-ready**: Priors integrate seamlessly into Bayesian sampling via `apply_physical_priors()`.
