# FLITS Analyses

This directory contains all burst-specific analyses and multi-burst studies using the FLITS pipelines.

## Structure

### `bursts/{burst_name}/`

Individual burst analyses for the DSA-110 + CHIME co-detection sample (12 bursts total):

- **casey**, **chromatica**, **freya**, **hamilton**, **isha**, **johndoeii**, **mahi**, **oran**, **phineas**, **whitney**, **wilhelm**, **zach**

Each burst directory contains:

- `README.md` - Burst properties, analysis summary, notes

Historical notebooks were removed from git during the cleanup of generated
analysis artifacts. Re-run analyses through the package CLIs and keep notebooks
or rendered products outside the repository unless they are intentionally
promoted as small fixtures.

### `samples/`

Multi-burst analyses and population studies:

- **dsa_chime_codetections/** - Cross-telescope comparisons (TOA, scattering, scintillation)

### `templates/`

Starting points for new burst analyses now live in the package CLIs and config
files rather than checked-in notebooks.

## Quick Start

### Analyze a New Burst

1. Create directory: `mkdir -p bursts/{new_burst_name}`
2. Update burst metadata in `configs/bursts.yaml`
3. Add or update a YAML config under `scattering/configs/bursts/`
4. Run analysis with `flits-scat` or `python -m scattering.run_scat_analysis`

### Run an Existing Analysis

```bash
flits-scat scattering/configs/bursts/casey_dsa.yaml
```

## See Also

- Burst metadata: `configs/bursts.yaml`
- Pipeline documentation: `docs/workflows/`
- Results: `results/bursts/{burst_name}/`
