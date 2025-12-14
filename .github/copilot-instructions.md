# FLITS | FRB Intensity Analysis Pipeline

FLITS is a lightweight, modular, telescope-agnostic toolkit for fitting pulse-broadening and scintillation in Fast Radio Burst (FRB) dynamic spectra, and instrumental effects.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- **CRITICAL**: Set PYTHONPATH for all Python operations:
  ```bash
  export PYTHONPATH=/home/runner/work/FLITS/FLITS:/home/runner/work/FLITS/FLITS/scattering:/home/runner/work/FLITS/FLITS/simulation
  ```
- Install dependencies: `pip install -r requirements.txt` -- takes 3-5 minutes. NEVER CANCEL.
- Python 3.8+ required. Uses scientific stack: numpy, scipy, matplotlib, emcee, astropy.

### Build and Test
- **NO BUILD STEP REQUIRED** -- This is a pure Python package with no compilation.
- Run core tests:
  ```bash
  PYTHONPATH=/home/runner/work/FLITS/FLITS python tests/test_models.py
  PYTHONPATH=/home/runner/work/FLITS/FLITS python tests/test_sampler.py
  ```
- Test basic functionality:
  ```bash
  PYTHONPATH=/home/runner/work/FLITS/FLITS python -c "
  import numpy as np
  from flits import FRBModel, FRBParams, FRBFitter
  params = FRBParams(dm=50.0, amplitude=1.0, t0=0.0, width=1.0)
  model = FRBModel(params)
  freqs = np.linspace(1300, 1500, 100)
  times = np.linspace(-10, 10, 200)
  data = model.simulate(times, freqs)
  print('✓ Core FLITS functionality works')
  "
  ```

### MCMC Fitting and Timing
- **NEVER CANCEL MCMC OPERATIONS** -- Fitting can take significant time.
- Typical timing for MCMC fits:
  - Quick test (100 steps): ~10 seconds
  - Standard fit (1000 steps): ~100 seconds (1.5 minutes). NEVER CANCEL. Set timeout to 200+ seconds.
  - Full analysis (5000+ steps): 10-30 minutes. NEVER CANCEL. Set timeout to 60+ minutes.
- Rate: approximately 0.1 seconds per step with realistic data (200×500 samples).

### Main Analysis Pipelines

#### Core FLITS Usage
```bash
PYTHONPATH=/home/runner/work/FLITS/FLITS python -c "
# Basic FRB modeling and fitting
from flits import FRBModel, FRBParams, FRBFitter
import numpy as np

# Create parameters and model
params = FRBParams(dm=50.0, amplitude=1.0, t0=0.0, width=2.0)
model = FRBModel(params)

# Generate or load data
freqs = np.linspace(1300, 1500, 200)  # MHz
times = np.linspace(-50, 50, 500)     # ms
data = model.simulate(times, freqs)

# Fit with MCMC
fitter = FRBFitter(times, freqs, data, noise_std=0.1)
initial = np.array([45.0, 0.8])  # [dm, amplitude] initial guess
sampler = fitter.sample(initial, nwalkers=32, nsteps=1000)  # Takes ~100s
"
```

#### Scattering Analysis Pipeline
- Navigate to scattering directory: `cd /home/runner/work/FLITS/FLITS/scattering`
- Run with configuration: 
  ```bash
  PYTHONPATH=/home/runner/work/FLITS/FLITS:/home/runner/work/FLITS/FLITS/scattering python run_scat_analysis.py configs/bursts/dsa/casey_dsa.yaml
  ```
- **NOTE**: Requires actual .npy data files. Config files point to specific data paths that may not exist in test environment.

#### Simulation Engine
```bash
cd /home/runner/work/FLITS/FLITS/simulation
PYTHONPATH=/home/runner/work/FLITS/FLITS:/home/runner/work/FLITS/FLITS/simulation python -c "
import engine
import numpy as np
print('✓ Simulation engine with numba acceleration works')
"
```

## Validation

### Always Test After Changes
- **MANDATORY**: Run the core functionality test above after any changes to the `flits/` package.
- **MCMC Validation**: Always test MCMC fitting with synthetic data:
  ```bash
  PYTHONPATH=/home/runner/work/FLITS/FLITS python -c "
  import time, numpy as np
  from flits import FRBModel, FRBParams, FRBFitter
  params = FRBParams(dm=50.0, amplitude=1.0, t0=0.0, width=2.0)
  model = FRBModel(params)
  freqs, times = np.linspace(1300, 1500, 50), np.linspace(-10, 10, 100)
  data = model.simulate(times, freqs) + 0.1 * np.random.randn(50, 100)
  fitter = FRBFitter(times, freqs, data, noise_std=0.1)
  start = time.time()
  sampler = fitter.sample(np.array([45.0, 0.8]), nwalkers=8, nsteps=50)
  print(f'✓ MCMC test completed in {time.time()-start:.1f}s')
  "
  ```
- **End-to-End**: Create synthetic FRB data, add noise, fit parameters, verify recovery.

### Module Status
- ✅ **WORKING**: `flits/` (core models), `scattering/` (pipeline), `simulation/` (engine), `dispersion/`
- ❌ **BROKEN**: `scintillation/` (import error), `crossmatching/` (missing dependencies), `animations/` (requires manim)
- **Focus changes on working modules only.**

## Project Structure

### Key Directories
```
FLITS/
├── flits/                    # Core FRB modeling package
│   ├── models.py            # FRBModel class
│   ├── params.py            # FRBParams dataclass  
│   ├── sampler.py           # FRBFitter MCMC interface
│   └── plotting.py          # Visualization utilities
├── scattering/              # Advanced scattering analysis
│   ├── run_scat_analysis.py # Main analysis script
│   ├── scat_analysis/       # Analysis pipeline modules
│   └── configs/             # YAML configuration files
├── simulation/              # FRB simulation engine
│   ├── engine.py            # Main simulation engine
│   └── *.py                 # Supporting modules
├── tests/                   # Unit tests
├── dispersion/              # Dispersion measure analysis
├── crossmatching/           # TOA crossmatching (broken)
├── scintillation/           # Scintillation analysis (broken)
└── animations/              # Visualization (requires manim)
```

### Critical Files
- `requirements.txt` -- Install with `pip install -r requirements.txt`
- `flits/__init__.py` -- Main package exports
- `tests/test_*.py` -- Basic validation tests
- `scattering/configs/telescopes.yaml` -- Telescope parameters
- `scattering/configs/sampler.yaml` -- MCMC configuration

## Common Pitfalls
- **PYTHONPATH**: Always set before importing. Missing PYTHONPATH causes "No module named 'flits'" errors.
- **Data Files**: Many examples reference .npy files that don't exist. Use synthetic data for testing.
- **MCMC Patience**: Never cancel MCMC operations. They legitimately take minutes to hours.
- **Module Dependencies**: Some modules require external packages not in requirements.txt.

## Timing Reference
| Operation | Time | Notes |
|-----------|------|-------|
| `pip install -r requirements.txt` | 3-5 min | NEVER CANCEL |
| Core tests | < 5 sec | Quick validation |
| MCMC (100 steps) | ~10 sec | Development testing |
| MCMC (1000 steps) | ~100 sec | NEVER CANCEL. Set 200s timeout |
| Full analysis | 10-30 min | NEVER CANCEL. Set 60+ min timeout |

## Quick Start Checklist
- [ ] `pip install -r requirements.txt`
- [ ] `export PYTHONPATH=/home/runner/work/FLITS/FLITS:/home/runner/work/FLITS/FLITS/scattering:/home/runner/work/FLITS/FLITS/simulation`
- [ ] Run core functionality test
- [ ] Run MCMC validation test
- [ ] Test changes with synthetic data before using real data files