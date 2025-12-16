# FLITS Configuration Files

This directory contains all configuration files for FLITS analyses, organized by purpose.

## Structure

```
configs/
├── README.md           # This file
├── bursts.yaml         # Burst metadata (DM, positions, dates)
├── telescopes.yaml     # Telescope parameters (symlink → scattering/configs/)
├── sampler.yaml        # MCMC sampler settings (symlink → scattering/configs/)
└── batch/              # Batch processing configs
    ├── manifest.yaml
    ├── chime/          # CHIME-specific batch configs (12 bursts)
    └── dsa/            # DSA-110-specific batch configs (12 bursts)
```

## Configuration Files

### `bursts.yaml`

**Purpose:** Single source of truth for burst scientific metadata

**Contents:**

- Burst identifiers (CHIME IDs)
- Dispersion measures (DM ± error)
- Sky coordinates (RA, Dec)
- Observation times (MJD, UTC)
- Legacy scattering parameters (τ, α, width)

**Usage:**

```python
import yaml
with open('configs/bursts.yaml') as f:
    bursts = yaml.safe_load(f)

casey_dm = bursts['bursts']['casey']['dm']  # 491.207
```

**Note:** This file contains **metadata only**. Pipeline-specific settings (downsampling, MCMC steps) are in pipeline configs or batch configs.

### `telescopes.yaml` (symlink)

**Purpose:** Telescope-specific instrumental parameters

**Source:** `scattering/configs/telescopes.yaml`

**Contents:**

- Frequency ranges
- Channel widths
- Time resolutions
- Instrumental smearing parameters

**Telescopes:** DSA-110, CHIME

### `sampler.yaml` (symlink)

**Purpose:** MCMC sampler default settings

**Source:** `scattering/configs/sampler.yaml`

**Contents:**

- Number of walkers
- Burn-in steps
- Production steps
- Convergence criteria

### `batch/`

**Purpose:** Batch processing configurations for running multiple bursts

**Structure:**

- `manifest.yaml` - Batch job definitions
- `chime/*.yaml` - Per-burst configs for CHIME data (12 files)
- `dsa/*.yaml` - Per-burst configs for DSA-110 data (12 files)

**Usage:**

```bash
# Run batch analysis for all CHIME bursts
flits-batch --config configs/batch/manifest.yaml --telescope chime

# Run single burst from batch config
flits-scat configs/batch/dsa/casey_dsa.yaml
```

## Pipeline-Specific Configs

Some configs remain in pipeline directories for modularity:

- `scattering/configs/bursts/` - Individual burst run configurations (paths, downsampling)
- `scintillation/configs/` - Scintillation-specific analysis configs

These contain **run parameters** (input paths, processing options), not scientific metadata.

## Migration Notes

**Previously:**

- `bursts.yaml` was in repository root
- `batch_configs/` was in repository root
- Telescope/sampler configs only in `scattering/configs/`

**Now:**

- All configs centralized in `configs/`
- Symlinks provide backward compatibility
- Clear separation: metadata (`bursts.yaml`) vs. run params (pipeline configs)

## See Also

- Burst metadata: `configs/bursts.yaml`
- Analysis workflows: `docs/workflows/`
- Analysis outputs: `results/bursts/`
