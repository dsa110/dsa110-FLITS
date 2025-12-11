# Results Directory

This directory stores all analysis outputs from FLITS pipelines.

## Structure Convention

```
results/
├── {burst_name}/
│   ├── scat/           # Scattering analysis outputs
│   │   ├── sampler.pkl     # MCMC sampler state
│   │   ├── bic_table.json  # Model comparison
│   │   └── corner.png      # Parameter posteriors
│   ├── scint/          # Scintillation analysis outputs
│   │   ├── acf_fit.json    # ACF parameters
│   │   └── acf_plot.png    # ACF diagnostic
│   └── plots/          # Combined diagnostic figures
│       ├── 4panel.png
│       └── 16panel.png
├── joint/              # Cross-burst and joint analyses
└── summary/            # Batch processing summary plots
```

## Usage

When running the pipeline, specify output directory:

```bash
flits-scat config.yaml --output results/freya/scat/
flits-scint config.yaml --output results/freya/scint/
```

Or use batch processing:

```bash
flits-batch run ./data --output results/
```
