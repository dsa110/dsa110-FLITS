# FLITS Lead Developer Onboarding Guide

**Date:** December 13, 2025  
**New Lead Developer Quick Reference**

---

## 🎯 Project Mission

FLITS (**F**itting **L**ikelihoods **I**n **T**ime-Frequency **S**pectra) is a telescope-agnostic toolkit for analyzing Fast Radio Burst (FRB) dynamic spectra. The core mission is to extract physical properties of FRBs and their propagation environments through:

1. **Pulse broadening (scattering)** - Measure scattering timescales and frequency scaling
2. **Scintillation** - Characterize multi-path propagation effects
3. **Dispersion** - Precise DM estimation via phase-coherence methods

**Primary Science Application:** DSA-110 + CHIME co-detection sample (12 bursts)

---

## 📊 Project Status Overview

### ✅ Production-Ready Components

- **Scattering Pipeline** (`scattering/scat_analysis/`) - Mature MCMC-based analysis
- **Scintillation Pipeline** (`scintillation/scint_analysis/`) - ACF fitting and parameter extraction
- **Two-Screen Simulator** (`simulation/`) - Publication-quality validation tools
- **Batch Processing** (`flits/batch/`) - CLI for multi-burst analysis with SQLite database

### 🚧 Active Development

- **Interactive Widgets** - Recently modularized from notebooks to `scint_analysis/widgets.py`
- **Joint Analysis** - Combining τ-scattering with Δν-scintillation constraints
- **Configuration System** - YAML-based telescope/sampler configs with manifest-driven batch runs

### ⚠️ Needs Integration

- **DM Estimation** (`dispersion/dmphasev2.py`) - Functional but not called by main pipelines
- **Cross-matching** (`crossmatching/`) - DSA+CHIME specific, needs generalization
- **Galaxy Queries** (`galaxies/`) - Project-specific scripts, not integrated with burst metadata

---

## 🏗 Architecture Overview

### Directory Structure

```
FLITS/
├── flits/                      # Core package
│   ├── models.py               # Basic FRB signal models (dispersion only)
│   ├── params.py               # Parameter containers
│   ├── sampler.py              # Simple emcee wrapper
│   ├── plotting.py             # Visualization utilities
│   ├── batch/                  # Batch processing system
│   │   ├── cli.py              # Command-line interface (flits-batch)
│   │   ├── batch_runner.py     # Orchestrator
│   │   ├── results_db.py       # SQLite results database
│   │   ├── joint_analysis.py   # τ-Δν joint constraints
│   │   └── summary_plots.py    # Cross-burst visualizations
│   └── common/                 # Shared constants
│
├── scattering/                 # Scattering analysis pipeline
│   ├── scat_analysis/          # Core modules
│   │   ├── burstfit.py         # Physics kernel (631 lines) - CRITICAL
│   │   ├── burstfit_pipeline.py # OO orchestrator
│   │   ├── burstfit_modelselect.py # M0→M1→M2→M3 BIC selection
│   │   ├── burstfit_robust.py  # Diagnostics (sub-band, leave-one-out)
│   │   ├── burstfit_corner.py  # Corner plots
│   │   └── config_utils.py     # YAML config parsing
│   ├── configs/                # Telescope and burst configs (YAML)
│   ├── notebooks/              # Analysis notebooks
│   └── run_scat_analysis.py    # CLI entry (flits-scat)
│
├── scintillation/              # Scintillation analysis pipeline
│   ├── scint_analysis/         # Core modules
│   │   ├── pipeline.py         # ScintillationAnalysis orchestrator
│   │   ├── core.py             # ACF computation
│   │   ├── analysis.py         # Model fitting (Lorentzian, Gaussian)
│   │   ├── widgets.py          # Interactive notebook widgets (NEW)
│   │   ├── plotting.py         # Publication-quality plots
│   │   └── config.py           # Configuration management
│   ├── configs/                # Burst-specific configs
│   ├── notebooks/              # Per-burst analysis notebooks
│   └── chime_acfs/             # Pre-computed ACF data from CHIME
│
├── simulation/                 # Two-screen scintillation simulator
│   ├── engine.py               # FRBScintillator class
│   ├── screen.py               # Screen geometry
│   ├── recreate_figures.py     # Reproduce Pradeep+ 2025 figures
│   └── validate_*.py           # Validation suite
│
├── dispersion/                 # DM estimation (phase-coherence)
│   └── dmphasev2.py            # DMPhaseEstimator class
│
├── crossmatching/              # TOA cross-matching (DSA ↔ CHIME)
│   ├── toa_crossmatch.py       # Barycentric correction, geometric delays
│   └── toa_utilities.py        # FWHM measurement
│
├── galaxies/                   # Catalog queries (NED, SDSS, DESI)
│   └── query_*.py              # Various catalog-specific scripts
│
├── animations/                 # 3D visualizations (Manim)
│   └── frb_anim.py
│
├── batch_configs/              # Batch processing configurations
│   ├── manifest.yaml           # Master burst list (12 bursts × 2 telescopes)
│   ├── chime/                  # CHIME-specific configs
│   └── dsa/                    # DSA-110-specific configs
│
├── bursts.yaml                 # Single source of truth for burst metadata
├── results/                    # Standardized output directory
├── .archive/                   # Legacy code (local only, .gitignore'd)
└── tests/                      # Unit tests (basic coverage)
```

---

## 🔑 Key Technical Components

### 1. Scattering Analysis (`burstfit.py`)

**Model Hierarchy:**

- **M0:** Baseline (Gaussian pulse + dispersion only)
- **M1:** M0 + pulse broadening (intrinsic width ζ)
- **M2:** M0 + frequency-dependent scattering (τ₁GHz, no intrinsic width)
- **M3:** M1 + M2 (full model: ζ + τ₁GHz + α + δDM)

**Physics:**

```python
# Dispersion delay: Δt = K × DM × (ν⁻² - ν_ref⁻²)
# Intra-channel smearing: σ_DM = 8.3×10⁻⁶ × DM × Δν / ν³
# Total width: σ_total = √(σ_DM² + ζ²)
# Scattering kernel: K(t) ∝ exp(-t/τ), τ = τ₁GHz × (ν/1GHz)⁻ᵅ
```

**MCMC Sampler:**

- `emcee` with customizable walkers and steps
- Log-space sampling for positive parameters (`c0`, `zeta`, `tau_1ghz`)
- Optional Jeffreys prior (1/x weighting)
- Gaussian or Student-t likelihood
- Multi-component bursts supported (shared scattering parameters)

**Critical Features:**

- Safe division guards to prevent NaN
- Causal convolution for scattering (not centered)
- Per-frequency noise estimation (MAD-based)
- Gelman-Rubin convergence diagnostics

### 2. Scintillation Analysis

**Workflow:**

1. Load dynamic spectrum (`.npy` or CHIME pickle)
2. Select on/off-pulse windows (interactive widget)
3. Compute frequency ACF
4. Fit multi-component models (Lorentzian + Gaussian)
5. Extract scintillation bandwidth (ν_s) and timescale (t_s)

**Models:**

- Narrow component (Lorentzian): Interstellar scattering
- Broad component (Gaussian): Unresolved scattering screen
- Noise template: Off-pulse ACF subtraction

### 3. Batch Processing System

**CLI Commands:**

```bash
# Generate configs from data directory
flits-batch generate-configs /path/to/data -o batch_configs/

# Run batch analysis
flits-batch run /path/to/data --output results/ --db flits_results.db

# Joint τ-Δν analysis
flits-batch joint-analysis flits_results.db -o joint_plots/

# Generate summary plots
flits-batch summary flits_results.db -o summary_plots/

# Export results
flits-batch export flits_results.db -f csv,latex,json
```

**Database Schema (SQLite):**

- `bursts` - Metadata (DM, RA/Dec, MJD)
- `scattering_results` - MCMC posteriors, BIC, convergence
- `scintillation_results` - ACF fit parameters, ν_s, t_s
- Cross-burst queries and LaTeX table export

---

## 🧪 Burst Sample

**12 Co-detected FRBs (DSA-110 + CHIME):**

| Name       | DM (pc/cm³) | RA (deg) | Dec (deg) | Scattering Status          |
| ---------- | ----------- | -------- | --------- | -------------------------- |
| Casey      | 491.207     | 169.984  | 70.676    | ✅ Fitted: τ=0.23ms, α=3.9 |
| Freya      | 912.4       | 88.188   | 74.200    | ✅ Fitted: τ=3.52ms, α=4.2 |
| Wilhelm    | 602.346     | 315.130  | 72.038    | ✅ Fitted: τ=2.82ms, α=4.1 |
| Chromatica | 272.664     | 312.619  | 73.9      | ⏳ Pending                 |
| Hamilton   | 518.799     | 305.037  | 70.793    | ⏳ Pending                 |
| Isha       | 411.568     | 71.411   | 70.307    | ⏳ Pending                 |
| JohnDoeII  | 696.506     | 335.975  | 73.026    | ⏳ Pending                 |
| Mahi       | 960.128     | 39.767   | 71.018    | ⏳ Pending                 |
| Oran       | 396.882     | 318.045  | 72.827    | ⏳ Pending                 |
| Phineas    | 610.274     | 177.781  | 71.696    | ⏳ Pending                 |
| Whitney    | 462.174     | 134.721  | 73.491    | ⏳ Pending                 |
| Zach       | 262.368     | 310.200  | 72.882    | ⏳ Pending                 |

**Best-Fit Parameters (from legacy notebooks):**

- Stored in `bursts.yaml` under `scattering:` field
- Original fits in `.archive/scattering/` notebooks

---

## 🔧 Development Environment

### Installation (Method 1: Quick Setup)

```bash
# Activate existing Python 3.10+ environment
conda activate dsa_contimg

# Install dependencies
pip install numpy scipy matplotlib pandas emcee lmfit corner pyyaml tqdm \
            ipywidgets ipympl astropy h5py chainconsumer

# Install FLITS in development mode
cd /Users/jakobfaber/Documents/research/caltech/ovro/dsa110/FLITS
pip install -e .
```

### Key Dependencies

- **MCMC:** `emcee >= 3.1`, `corner >= 2.2`
- **Optimization:** `lmfit >= 1.0`, `scipy >= 1.7`
- **Interactive:** `ipywidgets >= 8.0`, `ipympl` (matplotlib widget backend)
- **Analysis:** `chainconsumer >= 0.34` (MCMC visualization)

### Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest scattering/scat_analysis/tests/
pytest scintillation/scint_analysis/tests/
pytest flits/batch/tests/

# Run with coverage
pytest --cov=flits --cov=scattering --cov=scintillation
```

---

## 📝 Recent Refactoring (Last 2 Weeks)

### Completed

1. **Consolidated legacy code** → `.archive/` (excluded from git)
2. **Created `bursts.yaml`** → Single source of truth for burst metadata
3. **Standardized `results/` directory** → Consistent output structure
4. **Extracted notebook widgets** → `scint_analysis/widgets.py` (reduced 2500 lines to ~50)
5. **Implemented batch processing** → CLI with SQLite database
6. **Added joint analysis** → τ-Δν consistency checks

### Pending Changes (Staged)

- `flits/models.py` - Minor edits (check `git diff --staged`)

---

## 🚀 Immediate Priorities (Next Steps)

### High Priority

1. **Integrate DM estimation** into preprocessing
   - Call `dmphasev2.py` before scattering analysis
   - Update `dm_init` in configs automatically
2. **Complete scattering fits for remaining 9 bursts**
   - Use `flits-batch run` with existing configs
   - Validate convergence (Gelman-Rubin R̂ < 1.1)
3. **Fix MCMC walker initialization**

   - Current `walker_width_frac = 0.01` sometimes too tight
   - Tune per-burst or add adaptive scaling

4. **Documentation**
   - Add `docs/` directory with Sphinx
   - Tutorial notebooks in `docs/tutorials/`
   - API reference for public functions

### Medium Priority

5. **Generalize cross-matching**
   - Make telescope-agnostic (currently DSA+CHIME only)
   - Add to batch pipeline as optional step
6. **Galaxy catalog integration**
   - Link with `bursts.yaml` (RA/Dec)
   - Add `host_candidates` field with impact parameters
7. **CI/CD**
   - GitHub Actions for pytest on push
   - Automated notebook execution tests
8. **Publication figures**
   - Standardized plotting functions
   - `recreate_paper_figures.py` script

### Low Priority

9. **Nested sampling**
   - `dynesty` or `ultranest` integration (earmarked in code)
   - More robust evidence estimation
10. **Anisotropic scattering**
    - 2D scattering kernel (earmarked features in `burstfit.py`)

---

## 🐛 Known Issues

1. **MCMC Initialization**

   - Lines 394-433 in `scattering/scat_analysis/burstfit.py`
   - Occasionally fails with "Probability was NaN" if walkers start outside valid region
   - **Workaround:** Increase `walker_width_frac` in sampler config

2. **Notebook Code Duplication**

   - Still ~20 burst-specific notebooks in `scintillation/notebooks/`
   - **Fix in progress:** Migrate to template-based approach

3. **Configuration Inconsistency**

   - Some configs use `dm_init: 0.0`, others use actual DM values
   - **Decision needed:** Should configs reference `bursts.yaml` or be self-contained?

4. **Missing Tests**
   - `tests/` only has 2 files (basic smoke tests)
   - No integration tests for batch pipeline
   - **Action:** Add comprehensive test suite

---

## 📚 Code Navigation Guide

### Where to Start for Common Tasks

**Add a new burst:**

1. Add entry to `bursts.yaml`
2. Place `.npy` files in `data/chime/` and `data/dsa/`
3. Run `flits-batch generate-configs data/` to auto-create configs
4. Add to `batch_configs/manifest.yaml`

**Modify scattering model:**

1. Edit `scattering/scat_analysis/burstfit.py`
2. Update `FRBModel.__call__()` for forward model
3. Update `FRBParams` dataclass if adding parameters
4. Update `_ORDER` dict in `FRBFitter` class

**Add new diagnostic plot:**

1. Edit `scattering/scat_analysis/burstfit_plots.py`
2. Or add to `flits/batch/summary_plots.py` for cross-burst plots

**Modify scintillation fitting:**

1. Edit `scintillation/scint_analysis/analysis.py`
2. Update `fit_acf_model()` or add new model components

**Change batch processing logic:**

1. Edit `flits/batch/batch_runner.py`
2. Database schema: `flits/batch/results_db.py`

---

## 🎓 Learning Resources

### Key Files to Read (in order)

1. `README.md` - High-level overview
2. `bursts.yaml` - Burst sample metadata
3. `REFACTORING_SUMMARY.md` - Recent organizational changes
4. `ANALYSIS_INVENTORY.md` - Complete analysis catalog
5. `scattering/scat_analysis/burstfit.py` - Core physics (631 lines)
6. `flits/batch/cli.py` - User-facing interface

### Example Workflows

**Run single burst (interactive):**

```bash
python scattering/run_scat_analysis.py \
    data/chime/casey_chime.npy \
    --config scattering/configs/bursts/casey_chime.yaml \
    --model-scan --plot
```

**Batch processing:**

```bash
# Generate all configs
flits-batch generate-configs data/ -o batch_configs/

# Run scattering analysis on all bursts
flits-batch run data/ --output results/ --steps 10000 --nproc 8

# Joint analysis
flits-batch joint-analysis flits_results.db -o joint_plots/

# Export LaTeX table for paper
flits-batch export flits_results.db -f latex -o paper_table.tex
```

---

## 🤝 Collaboration Context

### Git Workflow

- **Main branch:** `main` (currently at `c9e9694`)
- **Recent commits:** Focus on refactoring and documentation
- **Staging area:** `flits/models.py` has uncommitted changes

### Communication Channels

- Documentation living in `.md` files (this approach)
- Configuration via YAML (not hardcoded in scripts)
- Results in SQLite database (queryable, version-controlled via exports)

---

## ✅ Quick Health Check

Run this to verify your environment is ready:

```bash
# Test imports
python -c "import flits; from scattering.scat_analysis import burstfit; \
           from scintillation.scint_analysis import pipeline; \
           print('✓ All modules imported successfully')"

# Check CLI tools
flits-batch --help
flits-scat --help
flits-scint --help

# Run basic tests
pytest tests/

# Check database
sqlite3 flits_results.db ".tables"
```

---

## 🎯 Your Mission (Lead Developer)

As the new lead developer, your primary responsibilities are:

1. **Complete the burst sample analysis** - Fit all 12 bursts with validated convergence
2. **Prepare for publication** - Generate reproducible figures and LaTeX tables
3. **Improve code quality** - Add tests, documentation, type hints
4. **Manage technical debt** - Address known issues and refactor where needed
5. **Guide new contributors** - Maintain clear workflows and onboarding materials
6. **Advance the science** - Integrate new models (anisotropy, nested sampling, etc.)

**Remember:** This is a research code that needs to balance scientific flexibility with software engineering best practices. Prioritize reproducibility and clarity over premature optimization.

---

**Good luck! You've inherited a well-organized, scientifically validated toolkit with clear paths forward. The hard physics is done; now it's time to scale the analysis and prepare for publication.**

---

_Last Updated: 2025-12-13 by Gemini AI Assistant_  
_Contact: Jakob Faber (original author)_
