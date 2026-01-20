# FLITS System Reference Documentation

This document provides a comprehensive technical reference for the FLITS (Fitting Likelihoods In Time-Frequency Spectra) codebase. It details the organizational structure, analysis pipeline components, data architecture, and implementation specifics.

## 1. Codebase Structure

The codebase is organized into modular components handling specific aspects of the FRB analysis workflow.

### Top-Level Directories

| Directory | Purpose |
|-----------|---------|
| `flits/` | **Core Package**. Contains shared utilities, fitting logic, and batch processing modules. |
| `scattering/` | **Scattering Pipeline**. Contains the `scat_analysis` package, notebooks, and scripts for scattering analysis. |
| `scintillation/` | **Scintillation Pipeline**. Contains the `scint_analysis` package and notebooks for scintillation analysis. |
| `configs/` | **Configuration**. YAML files for burst metadata (`bursts.yaml`), batch processing (`batch/`), and telescope specs. |
| `data/` | **Data Storage**. Default location for input data and fit results (though large data is often external). |
| `analyses/` | **Analysis Notebooks**. Jupyter notebooks for specific burst analyses, organized by burst nickname. |
| `crossmatching/` | **Multi-Telescope Tools**. Scripts for cross-matching detections between observatories (e.g., TOA calculation). |
| `docs/` | **Documentation**. Architecture overviews, user guides, and developer notes. |

### Analysis Pipeline Architecture

The analysis pipeline is designed to characterize FRB intensity variations, focusing on scattering and scintillation.

#### 1. Pre-processing (`BurstDataset`)
Located in `scattering/scat_analysis/burstfit_pipeline.py`.
- **Loading**: Reads raw `.npy` files.
- **Bandpass Correction**: Normalizes the spectrum using off-pulse statistics to remove instrumental bandpass shape.
- **Trimming**: Removes edge channels or time bins if specified (`outer_trim`).
- **Downsampling**: Averages frequency channels and time bins to increase S/N or match desired resolution (`f_factor`, `t_factor`).

#### 2. Physics Modeling (`FRBModel`)
Located in `scattering/scat_analysis/burstfit.py`.
- Generates dispersed Gaussian pulses.
- Convolves pulses with a scattering kernel (exponential decay).
- **Scattering Modes** (as implemented in `FRBFitter`):
    - `M0`: **Unresolved Pulse**. Delta-function source.
      - *Params*: `c0` (amp), `t0` (time), `gamma` (spectral index).
      - *Note*: DM is fixed to `dm_init`; $\beta=2$ fixed.
    - `M1`: **Resolved Pulse**. Adds intrinsic width.
      - *Params*: `M0` + `zeta` (intrinsic width).
    - `M2`: **Scattered Unresolved Pulse**. Adds scattering tail.
      - *Params*: `M0` + `tau_1ghz`.
      - *Note*: Scattering index $\alpha$ is fixed (default 4.4).
    - `M3`: **Scattered Resolved Pulse**. Full model.
      - *Params*: `M1` + `tau_1ghz`, `alpha`, `delta_dm`.
      - *Note*: Only M3 refines `delta_dm` and fits `alpha`.
    - `mixed`: **Mixed Multi-Component**. Fits multiple components with different models.
      - *Params*: Global `delta_dm` (if any component is M3), plus per-component parameters indexed by component number (e.g., `c0_1`, `tau_1ghz_2`).
      - *Example*: `["M0", "M3"]` fits an unresolved first component (no scattering) and a scattered resolved second component.

#### 3. Fitting & Sampling (`FRBFitter`)
Located in `scattering/scat_analysis/burstfit.py`.
- Wraps `emcee` for MCMC sampling.
- **Priors**: Supports uniform, log-uniform, and Gaussian priors defined in `build_priors`.
- **Likelihood**: Gaussian likelihood (default) or Student-t likelihood for robustness against outliers.

#### 4. Model Selection (`fit_models_bic`)
Located in `scattering/scat_analysis/burstfit_modelselect.py`.
- Fits multiple models (M0, M2, M3) sequentially.
- Selects the best model using the Bayesian Information Criterion (BIC).

#### 5. Diagnostics (`burstfit_robust`)
Located in `scattering/scat_analysis/burstfit_robust.py`.
- **Sub-band Consistency**: Checks if fitted parameters are consistent across different frequency sub-bands.
- **Influence Maps**: Analyzes the influence of individual data points on the fit ("leave-one-out" analysis).

## 2. Data Architecture

### Raw Data Structure
- **Format**: `.npy` (NumPy binary format).
- **Structure**: 2D array with shape `(n_freq, n_time)`.
- **Orientation**: Frequency axis is standardized to **ascending order** (index 0 = lowest frequency) during loading in `BurstDataset`.
- **Units**: Arbitrary intensity units (usually uncalibrated filterbank data).

### Metadata Schema
Metadata is managed centrally to ensure consistency across the pipeline.

#### Source of Truth
- **`configs/bursts.yaml`**: The primary registry for burst properties.
- **`chimedsa_burst_specs.csv`**: A CSV catalog mapping nicknames to TNS names and coordinates.

#### Key Fields
| Field | Description | Example |
|-------|-------------|---------|
| `name` | Informal nickname used internally. | `casey`, `freya` |
| `chime_id` | CHIME/FRB Event ID. | `362593221` |
| `TNS` | Official TNS Name. | `FRB 20240229A` |
| `dm` | Dispersion Measure (pc/cm³). | `491.207` |
| `mjd` | Modified Julian Date of arrival. | `60369.371` |
| `ra_deg` / `dec_deg` | Right Ascension and Declination (J2000). | `169.98`, `70.67` |

### Results Storage
- **Format**: JSON (`.json`).
- **Content**:
  - `best_params`: Dictionary of best-fit parameter values.
  - `best_key`: The selected model (e.g., "M3").
  - `goodness_of_fit`: $\chi^2$, $R^2$, and other metrics.
  - `chain_stats`: Summary statistics of the MCMC chain.

## 3. Special Cases

### Simultaneous Bursts (DSA-110 + CHIME)
Code for handling multi-telescope detections is in `crossmatching/`.
- **TOA Calculation**: `compute_toa` in `toa_crossmatch.py` calculates Time of Arrival referenced to infinite frequency or a specific band.
- **Geometric Delay**: `compute_geometric_delay` accounts for the light-travel time difference between observatories (e.g., OVRO vs. DRAO).

### Naming Conventions
- **Internal Nicknames**: Short, human-readable names (e.g., `casey`) are used for filenames, config keys, and variable names.
- **TNS Names**: Official identifiers (e.g., `FRB 20240229A`) are used for publication and external cross-matching.
- **Mapping**: The module `scattering.scat_analysis.burst_metadata` provides functions like `load_tns_name` to convert between them.

## 4. Implementation Details

### File Formats
| Type | Extension | Usage |
|------|-----------|-------|
| **Data** | `.npy` | Raw dynamic spectra. |
| **Config** | `.yaml` | Pipeline configurations, telescope parameters. |
| **Metadata** | `.csv` | Burst catalog (TNS mapping). |
| **Results** | `.json` | Fit parameters and statistics. |
| **Plots** | `.png` / `.pdf` | Diagnostic plots (waterfalls, corner plots). |

### Version Control
- **Git**: The codebase is version-controlled via Git.
- **Practices**:
  - Configuration files (`.yaml`) are committed to ensure reproducibility.
  - Large data files (`.npy`) are **excluded** (via `.gitignore`) and stored externally.
  - Results (`.json`) are often committed or archived to track analysis history.

### Preprocessing Requirements
Before fitting, data undergoes:
1.  **Bandpass Correction**: Essential for removing instrumental response. Done automatically in `BurstDataset`.
2.  **RFI Mitigation**: Heavily RFI-contaminated channels should be masked (set to NaN or 0 weight) in the input `.npy` or handled via the `noise_std` estimation.
