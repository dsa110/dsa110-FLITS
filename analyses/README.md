# FLITS Analyses

This directory contains all burst-specific analyses and multi-burst studies using the FLITS pipelines.

## Structure

### `bursts/{burst_name}/`

Individual burst analyses for the DSA-110 + CHIME co-detection sample (12 bursts total):

- **casey**, **chromatica**, **freya**, **hamilton**, **isha**, **johndoeii**, **mahi**, **oran**, **phineas**, **whitney**, **wilhelm**, **zach**

Each burst directory contains:

- `README.md` - Burst properties, analysis summary, notes
- `scattering_*.ipynb` - Scattering analysis notebooks (DSA/CHIME)
- `scintillation.ipynb` - Scintillation analysis (when available)

### `samples/`

Multi-burst analyses and population studies:

- **dsa_chime_codetections/** - Cross-telescope comparisons (TOA, scattering, scintillation)

### `templates/`

Starting points for new burst analyses:

- `scattering_template.ipynb` - Template for scattering pipeline
- `scintillation_template.ipynb` - Template for scintillation pipeline

## Quick Start

### Analyze a New Burst

1. Create directory: `mkdir -p bursts/{new_burst_name}`
2. Copy template: `cp templates/scattering_template.ipynb bursts/{new_burst_name}/scattering.ipynb`
3. Update burst metadata in `configs/bursts.yaml`
4. Run analysis (see template for details)

### Run an Existing Analysis

```bash
# Using Jupyter
jupyter notebook analyses/bursts/casey/scattering_dsa.ipynb

# Using the CLI
flits-scat scattering/configs/bursts/casey_dsa.yaml
```

## See Also

- Burst metadata: `configs/bursts.yaml`
- Pipeline documentation: `docs/workflows/`
- Results: `results/bursts/{burst_name}/`
