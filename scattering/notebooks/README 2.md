# FRB Scattering Analysis Notebooks

This directory contains the unified notebook interface for FLITS scattering analysis.

## Quick Start

1. Open `scattering_analysis.ipynb`
2. Set `config_path` to your burst's YAML configuration file
3. Run all cells

## Configuration Files

All burst configurations are stored in `../configs/bursts/`:

### DSA-110 Observations
```
configs/bursts/dsa/
├── casey_dsa.yaml
├── chromatica_dsa.yaml
├── freya_dsa.yaml
├── hamilton_dsa.yaml
├── isha_dsa.yaml
├── johndoeII_dsa.yaml
├── mahi_dsa.yaml
├── oran_dsa.yaml
├── phineas_dsa.yaml
├── whitney_dsa.yaml
├── wilhelm_dsa.yaml
└── zach_dsa.yaml
```

### CHIME Observations
```
configs/bursts/chime/
├── casey_chime.yaml
├── chromatica_chime.yaml
├── freya_chime.yaml
├── hamilton_chime.yaml
├── isha_chime.yaml
├── johndoeII_chime.yaml
├── mahi_chime.yaml
├── oran_chime.yaml
├── phineas_chime.yaml
├── whitney_chime.yaml
├── wilhelm_chime.yaml
└── zach_chime.yaml
```

## YAML Configuration Format

Each burst configuration file specifies:

```yaml
# Data location
path: "/path/to/burst_data.npy"

# Telescope and initial conditions
telescope: "dsa"  # or "chime"
dm_init: 0.0      # Initial dispersion measure

# MCMC sampling parameters
steps: 10000
nproc: 16
extend_chain: true
chunk_size: 2000
max_chunks: 5

# Processing parameters
f_factor: 384     # Frequency downsampling
t_factor: 2       # Time downsampling

# Analysis options
model_scan: true       # Run BIC model selection
diagnostics: true      # Generate diagnostic plots
plot: true            # Save publication plots
```

## Workflow

The unified notebook provides a complete pipeline:

### 1. Configuration Loading
```python
config_path = Path("../configs/bursts/dsa/casey_dsa.yaml")
```

### 2. Pipeline Execution
- Data preprocessing (dedispersion, downsampling, centering)
- Initial parameter guess optimization
- Model selection (M0: thin screen, M1: extended medium, M2: hybrid, M3: thick screen)
- MCMC sampling with emcee
- Convergence diagnostics

### 3. Automated Outputs
- **16-panel diagnostic plot**: `{burst_name}_scat_fit.pdf`
  - Dynamic spectrum (observed vs. model)
  - Pulse profile evolution
  - Scattering timescale vs. frequency
  - MCMC chain diagnostics
  - Sub-band analyses
  - Goodness-of-fit metrics

- **Corner plot**: `{burst_name}_scat_corner.pdf`
  - Full posterior distributions
  - Parameter correlations
  - Best-fit markers

### 4. Interactive Analysis (Optional)
- Extended convergence checking
- Custom diagnostic plots
- Parameter exploration

## Command-Line Alternative

For batch processing, use the CLI script:

```bash
# Basic usage
python ../run_scat_analysis.py configs/bursts/dsa/casey_dsa.yaml

# Override parameters
python ../run_scat_analysis.py configs/bursts/dsa/casey_dsa.yaml \
    --steps 5000 \
    --nproc 32 \
    --no-extend-chain

# Custom telescope/sampler configs
python ../run_scat_analysis.py configs/bursts/dsa/casey_dsa.yaml \
    --telcfg custom_telescopes.yaml \
    --sampcfg custom_sampler.yaml
```

## Output Directory Structure

Results are saved to the directory specified in the config file (or `../plots/{telescope}` by default):

```
plots/
├── dsa/
│   ├── casey_scat_fit.pdf
│   ├── casey_scat_corner.pdf
│   └── ...
└── chime/
    ├── freya_scat_fit.pdf
    ├── freya_scat_corner.pdf
    └── ...
```

## Models

The pipeline supports four scattering models:

- **M0**: Thin screen (single scattering layer)
- **M1**: Extended medium (distributed scattering)
- **M2**: Hybrid (thin screen + extended medium)
- **M3**: Thick screen (multiple scattering layers)

Model selection uses Bayesian Information Criterion (BIC) to balance fit quality and complexity.

## Legacy Notebooks

Previous burst-specific notebooks have been archived to `../legacy/`:
- `*_new.ipynb` - Individual burst analysis notebooks (replaced by unified notebook)
- `burstscat_test*.ipynb` - Early test notebooks
- `ui_seed.ipynb` - Interactive parameter seed generator
- `synthetic_scatter_fit.ipynb` - Synthetic data tests

These are preserved for reference but no longer maintained.

## Troubleshooting

**Import errors**: Ensure you're in the scattering directory and the package is installed:
```bash
cd /path/to/FLITS/scattering
pip install -e .
```

**Missing dependencies**:
```bash
pip install numpy matplotlib scipy emcee chainconsumer arviz seaborn pyyaml
```

**Multiprocessing issues**: Reduce `nproc` in your config file or add `yes: true` to auto-confirm pool creation.

**Convergence failures**: Increase `steps`, `max_chunks`, or adjust prior ranges in `configs/sampler.yaml`.

## Support

For issues or questions:
1. Check the main FLITS repository README
2. Review example configurations in `configs/bursts/`
3. Examine pipeline documentation in `scat_analysis/burstfit_pipeline.py`
