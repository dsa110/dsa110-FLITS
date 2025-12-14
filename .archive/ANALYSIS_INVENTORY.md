# FLITS Repository - Analysis Inventory

This document catalogs all the various analyses performed within the FLITS repository, distinguishing between core pipeline functionality and burst-specific analyses.

---

## Core Pipeline Analyses

### 1. **Scattering Analysis Pipeline** (`scattering/scat_analysis/`)
**Status:** ✅ Well-integrated, production-ready

**Purpose:** Fit pulse broadening models to FRB dynamic spectra using MCMC

**Key Components:**
- `burstfit_pipeline.py` - OO orchestrator for end-to-end analysis
- `burstfit.py` - Physics kernel (dispersion, scattering, smearing) + likelihood
- `burstfit_modelselect.py` - Sequential model comparison (M0→M1→M2→M3) via BIC
- `burstfit_robust.py` - Robustness diagnostics (sub-band consistency, leave-one-out)
- `burstfit_corner.py` - Corner plots and chain visualization
- `config_utils.py` - YAML-based telescope/sampler configuration
- `pool_utils.py` - Multi-processing support

**Entry Point:** `run_scat_analysis.py` (CLI: `flits-scat`)

**Outputs:**
- Model parameter posteriors (pickled samplers)
- 4-panel and 16-panel diagnostic plots
- BIC model selection tables
- Corner plots with parameter constraints

**Models Supported:**
- M0: Baseline (Gaussian pulse + dispersion only)
- M1: M0 + pulse broadening (scattering)
- M2: M1 + spectral index (frequency-dependent scattering)
- M3: M2 + multiple Gaussian components

---

### 2. **Scintillation Analysis Pipeline** (`scintillation/scint_analysis/`)
**Status:** Well-integrated, production-ready

**Purpose:** Measure scintillation parameters from frequency auto-correlation functions (ACFs)

**Key Components:**
- `pipeline.py` - `ScintillationAnalysis` class orchestrator
- `core.py` - ACF computation and fitting
- `analysis.py` - Parameter extraction (scintillation bandwidth, timescale)
- `noise.py` - Noise characterization and subtraction
- `plotting.py` - Diagnostic visualizations
- `config.py` - YAML configuration loading

**Entry Point:** `run_analysis.py` (CLI: `flits-scint`)

**Outputs:**
- Scintillation bandwidth (ν_s) measurements
- Scintillation timescale (t_s) estimates
- ACF plots with model fits
- Noise-subtracted ACFs
- JSON results files

**Input Data:** 
- Dynamic spectra (.npy files)
- ACF data (pickled from CHIME)

---

### 3. **Two-Screen Scintillation Simulator** (`simulation/`)
**Status:** Publication-quality, validated

**Purpose:** Simulate two-screen scintillation (MW + host galaxy) for FRBs

**Key Components:**
- `engine.py` - `FRBScintillator` class with full 2-screen physics
- `screen.py` - `Screen` and `ScreenCfg` classes
- `geometry.py` - Geometric calculations (angular diameter distances, etc.)
- `instrument.py` - Instrumental noise modeling
- `frb_scintillator.py` - Convenience facade

**Special Scripts:**
- `recreate_figures.py` - Reproduce figures from Pradeep et al. (2025)
- `multifreq_analysis.py` - Broadband analysis (Figure 16 replication)
- `validate_sim.ipynb`, `validate_doppler_rate.py`, `validate_unresolved_case.py` - Validation tests

**Features:**
- Numba-accelerated where available
- 1D and 2D screen geometries
- Resolution power (RP) calculations
- ACF fitting with component isolation
- Monte Carlo averaging support

**Outputs:**
- Synthetic dynamic spectra
- ACF curves with narrow/broad components
- Scintillation bandwidth vs. frequency
- Resolution diagnostic plots

---

### 4. **Dispersion Measure Estimation** (`dispersion/`)
**Status:** Functional, needs integration

**Purpose:** Precise DM estimation via phase-coherence method

**Key Components:**
- `dmphasev2.py` - `DMPhaseEstimator` class
  - Coherent FFT-based phase analysis
  - Bootstrap uncertainty estimation
  - Weighted channel support
  - High-frequency cutoff filtering

**Analysis Type:** Vectorized phase-coherence method with quadratic peak fitting

**Outputs:**
- Optimized DM value
- Bootstrap-derived uncertainty (σ_DM)
- DM curve (coherent power vs. trial DM)

**Test Suite:** `test_dm_phase.py`, `test_dm_phase.ipynb`

---

### 5. **Time-of-Arrival (TOA) Cross-matching** (`crossmatching/`)
**Status:** ⚠️ Partially integrated, burst-specific

**Purpose:** Cross-match FRB detections between telescopes (DSA-110 ↔ CHIME)

**Key Components:**
- `toa_crossmatch.py` - TOA calculation with geometric delay corrections
- `toa_utilities.py` - FWHM measurement, downsampling utilities
- `test.py` - Validation script

**Analysis Type:**
- Barycentric correction
- Geometric delay from Earth rotation
- Reference frequency standardization (400 MHz)
- Pulse width (FWHM) measurement

**Outputs:**
- `toa_crossmatch_results.json` - TOA comparison table
- `frb_analysis_with_fwhm.json` - Burst properties with FWHM

**Data:** Co-detection sample (DSA-110 + CHIME)

---

### 6. **Galaxy Catalog Queries** (`galaxies/`)
**Status:** ⚠️ Project-specific, needs generalization

**Purpose:** Query astronomical catalogs for potential FRB host/foreground galaxies

**Key Scripts:**
- `query_cat.py` - Multi-catalog query (NED, Pan-STARRS, SDSS, DESI)
- `query_desi.py` - DESI DR1 specific
- `query_dr10.py` - Legacy Survey DR10
- `crossmatch.ipynb` - Cross-matching notebook

**Sub-Projects:**
- `dr8_dec70_75/` - Legacy Survey sweep file processing
- `tarrio_ps1_photz/` - Tarrio+2021 photo-z catalog matching
- `wise-ps1-strm/` - WISE+PS1+Stromlo cross-match

**Analysis Type:**
- Cone search around FRB positions
- Impact parameter calculation (proper distance)
- Stellar mass estimation
- Redshift matching

**Outputs:**
- Excel workbooks with galaxy candidates
- Impact parameter rankings
- Multi-catalog merged tables

---

### 7. **Animations** (`animations/`)
**Status:** Standalone visualization tools

**Purpose:** 3D visualizations of FRB timing and wavefront propagation

**Scripts:**
- `frb_anim.py` - Manim-based 3D Earth + telescope + FRB wavefront
- `art_frb.py` - Artistic renderings
- `test_plane.py` - Geometry tests

**Outputs:** MP4 animation files

---

## 🔬 Burst-Specific Analyses

### **Scattering Analysis** (`scattering/`)

Each burst has a dedicated notebook/script following the pattern:
- `{burst_name}_{telescope}_new.ipynb` or `.py`

**Bursts Analyzed:**
1. **Casey** (DSA-110) - `casey_dsa_new.ipynb`
2. **Freya** (CHIME & DSA-110) - `freya_chime_new.ipynb`, `freya_chime_new.py`, `freya_dsa_new.ipynb`
3. **Wilhelm** (CHIME & DSA-110) - `wilhelm_chime_new.ipynb`, `wilhelm_dsa_new.ipynb`

**Common Pattern:**
```python
# 1. Load data (.npy file)
# 2. Configure pipeline (telescope, downsampling, MCMC steps)
# 3. Run BurstPipeline
# 4. Model selection (M0→M1→M2→M3)
# 5. Generate diagnostic plots
# 6. Interactive corner plots and parameter extraction
```

**Test/Exploratory Notebooks:**
- `burstscat_test.ipynb` (+ Copy1, Copy2)
- `synthetic_scatter_fit.ipynb` - Test on synthetic data
- `ui_seed.ipynb` - Initial guess UI prototype

---

### **Scintillation Analysis** (`scintillation/notebooks/`)

**Burst-Specific Directories:** Each burst has its own folder with multiple analysis versions

1. **casey/** - 3 notebooks (automated + manual versions)
2. **freya/** - 5 notebooks (multiple manual analysis iterations)
3. **wilhelm/** - Similar structure
4. **zach/** - 2 notebooks
5. **isha/** - 2 notebooks
6. **mahi/** - 2 notebooks
7. **johndoeII/** - 1 manual notebook
8. **hamilton/** - (files TBD)
9. **phineas/** - (files TBD)
10. **whitney/** - (files TBD)
11. **oran/** - (files TBD)
12. **chromatica/** - Special: zero-crossing analysis

**Notebook Naming:**
- `{burst}.ipynb` - Automated pipeline run
- `{burst}_manual.ipynb` - Interactive/manual analysis
- `{burst}_manual_v0.ipynb`, `v1.ipynb` - Iteration versions

**Common Pattern:**
```python
# 1. Load ACF data (from CHIME pickle files)
# 2. Configure ScintillationAnalysis
# 3. Fit ACF models
# 4. Extract ν_s, t_s, DM
# 5. Generate diagnostic plots
# 6. Save results to JSON
```

**General/Multi-Burst Notebooks:**
- `general_manual.ipynb`, `general_manual_2.ipynb`, `general_manual_3.ipynb` - Cross-burst comparisons
- `interveners.ipynb` - Foreground screen analysis
- `spec_hist.ipynb` - Spectral histograms
- `3dmap.ipynb` - 3D scintillation map

**Special Analyses:**
- `twoscreen/` - Two-screen model fitting (`twoscreen.ipynb`, `twoscreen_old.ipynb`)
- `chromatica/zero_crossing.ipynb` - ACF zero-crossing analysis
- `debug/` - Debugging notebooks

**Data Sources:**
- `chime_acfs/pickle.ipynb` - ACF data preprocessing

---

## 📈 Meta-Analyses and Validation

### **NE2001 Integration** (`scintillation/ne2001/`)
- `query_ne2001_scint.py` - Query NE2001 electron density model
- `healpix.ipynb` - HEALPix-based sky map analysis

### **Simulation Validation** (`simulation/`)
- `validate_sim.ipynb` - Comprehensive simulator validation
- `validate_doppler_rate.py` - Doppler broadening validation
- `validate_unresolved_case.py` - Unresolved regime tests
- `monte_carlo.py` - Monte Carlo utilities
- `analysis_utils.py` - Helper functions for multi-frequency analysis

---

## 🗂️ Organization Issues Identified

### ❌ Problems:

1. **Burst-specific notebooks scattered** - Each burst has 2-5 notebooks in different locations
   - Scattering: `scattering/{burst}_*.ipynb`
   - Scintillation: `scintillation/notebooks/{burst}/{burst}_*.ipynb`

2. **No clear entry point** for new bursts - Users must copy/modify existing notebooks

3. **Duplicate analyses** - Same burst analyzed multiple times with slight variations (`_manual`, `_v0`, `_v1`, etc.)

4. **Configuration inconsistency** - Some use YAML, others hardcode parameters in notebooks

5. **Results not centralized** - JSON files scattered across directories

6. **Unclear pipeline flow** - Hard to see which tools to use in what order

7. **Co-detection specific code mixed with general tools** - e.g., TOA cross-matching assumes DSA+CHIME

8. **Incomplete integration** - DM estimation (`dmphasev2.py`) not called by main pipelines

9. **Galaxy query tools** - Hardcoded target lists, not integrated with FRB metadata

10. **No unified burst database** - Each analysis redefines burst properties

---

## Well-Organized Components:

1. **Core scattering pipeline** (`scat_analysis/`) - Modular, well-documented
2. **Core scintillation pipeline** (`scint_analysis/`) - Clean API
3. **Simulator** (`simulation/`) - Publication-ready, validated
4. **Configuration system** - YAML-based telescope/sampler configs

---

## 🎯 Suggested Improvements (for next steps):

1. **Create `analyses/` directory** - Move all burst-specific work here
   ```
   analyses/
   ├── bursts/
   │   ├── casey/
   │   ├── freya/
   │   └── ...
   └── samples/
       └── dsa_chime_codetections/
   ```

2. **Unified burst metadata** - Single YAML/JSON file per burst

3. **Template notebooks** - `template_scattering.ipynb`, `template_scintillation.ipynb`

4. **Integrate DM estimation** - Add to preprocessing step

5. **Generalize cross-matching** - Telescope-agnostic TOA comparison

6. **Centralize results** - `results/{burst_name}/` with standardized output structure

7. **Documentation** - Add `docs/` with analysis workflows

8. **Testing** - Move to `tests/` and add CI

---

## 📝 Summary Statistics

- **Total Notebooks:** ~54
- **Bursts Analyzed (Scintillation):** ~12+
- **Bursts Analyzed (Scattering):** ~3 (Casey, Freya, Wilhelm)
- **Core Pipelines:** 2 (scattering, scintillation)
- **Simulation Tools:** 1 (two-screen)
- **Utility Analyses:** 5 (DM estimation, TOA, galaxy queries, animations, validation)

**Co-detection Sample (DSA-110 + CHIME):** Primary focus of burst-specific analyses
