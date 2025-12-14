# FLITS Quick Start

## One-liner Quick Start

Run the pipeline using the installed `flits-scat` command (or `python scattering/run_scat_analysis.py`). You need a YAML configuration file, but you can override specific parameters (like the input file) from the command line.

```bash
# fit a burst using a template config, overriding the data path
flits-scat scattering/configs/dsa/template_dsa.yaml \
    --path data/my_burst.npy \
    --steps 500 \
    --plot
```

**Flags of interest** (see `flits-scat -h` for full list):

| Flag           | Meaning                                               |
| -------------- | ----------------------------------------------------- |
| `--model-scan` | Enable/Disable model selection (default: from config) |
| `--plot`       | Show diagnostic plots                                 |
| `--steps`      | Override number of MCMC steps                         |
| `--dm_init`    | Override initial Dispersion Measure                   |
