# FLITS Architecture Overview

## High-level Data Flow

```mermaid
raw .npy       ┌──────────────────┐   best sampler & params
file  ───────▶ │ pre‑processing   │──┐
               │ (analysis script)│  │   ┌───────────────────┐     influence map
               └──────────────────┘  └──▶│  diagnostics      │──▶ (optional plots)
                         │   ds,f,t      │  (robust helper)  │
                         ▼               └───────────────────┘
               ┌──────────────────┐
               │  model scan      │  BIC table
               │  (modelselect)   │──▶ best model key
               └──────────────────┘
```

- **pre‑processing** — band‑pass correct, trim, down‑sample, normalise.
- **model scan** — runs MCMC for M0…M3, picks the winner by BIC.
- **diagnostics** — optional robustness checks before publication.

---

## Directory Layout

```text
FLITS
├── scattering/
│   ├── scat_analysis/
│   │   ├── burstfit_pipeline.py    # OO pipeline orchestrator
│   │   ├── burstfit.py             # Core physics + MCMC wrappers
│   │   ├── burstfit_modelselect.py # Model comparison via BIC
│   │   └── burstfit_robust.py      # Robustness diagnostics
│   └── run_scat_analysis.py        # CLI entry point
├── scintillation/
│   └── scint_analysis/             # Scintillation pipeline components
├── docs/                           # Documentation hub
└── flits/                          # Shared utilities and batch processing
```

---

## Module Cheat-Sheet

| Module                                  | Responsibility           | Public API                                             |
| --------------------------------------- | ------------------------ | ------------------------------------------------------ |
| `scat_analysis/burstfit.py`             | Physics kernel & sampler | `FRBModel`, `FRBFitter`, `FRBParams`, `build_priors()` |
| `scattering/run_scat_analysis.py`       | Command‑line driver      | CLI main, `BurstPipeline` wrapper                      |
| `scat_analysis/burstfit_modelselect.py` | Sequential fits + BIC    | `fit_models_bic()`                                     |
| `scat_analysis/burstfit_robust.py`      | Robustness diagnostics   | `subband_consistency()`, `leave_one_out_influence()`   |

---

## Diagnostics at a Glance

- **Sub‑band consistency** — fit τ₁ GHz in N frequency chunks; large spread ⇒ per‑band systematics.
- **Leave‑one‑out influence** — χ²‑based heat‑map of how each channel pulls the global fit.
- **(Optional) SBC** — simulation‑based calibration helper planned for v2.1.
