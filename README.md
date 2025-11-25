# FLITS | FRB Intensity Analysis Pipeline

**F**itting **L**ikelihoods **I**n **T**ime-Frequency **S**pectra: A lightweight, modular, telescope-agnostic toolkit for fitting pulse-broadening and scintillation in Fast Radio Burst (FRB) dynamic spectra, and instrumental effects. 

---

## Directory layout

```
FLITS
├── burstfit_pipeline.py        # OO pipeline orchestrator
├── burstfit.py                 # core physics + MCMC wrappers
├── burstfit_utils.py           # helper functions MCMC fit
├── burstfit_modelselect.py     # sequential M0 → M1 → M2 → M3 model comparison + selection
├── burstfit_robust.py          # sub‑band & per-channel leave‑one‑out diagnostics + others
├── config_utils.py             # reads telescope-specific raw-data parameters (in .yaml)
└── pool_utils.py               # handles multi-processing (if applicable)
```

---

## High‑level data flow

```
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
* **pre‑processing** — band‑pass correct, trim, down‑sample, normalise.
* **model scan** — runs MCMC for M0…M3, picks the winner by BIC.
* **diagnostics** — optional robustness checks before publication.

---

## One‑liner quick start

```
bash
# fit a burst with down‑sampling (e.g., 384×2) and run model selection
python burstfit_casey_analysis.py casey.npy \
       --downsample-f 384 --downsample-t 2 \
       --model-scan --plot
```

**Flags of interest** (see `-h` for full list):

| Flag           | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| `--model-scan` | run `burstfit_modelselect.fit_models_bic()` internally |
| `--plot`       | show data / model / residual heat‑maps                 |
| `--save`       | pickle the sampler to disk for later inspection        |

---

## Module cheat‑sheet

| Module                       | Responsibility                      | Public API                                             |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------ |
| `burstfit.py`                | Physics kernel & sampler            | `FRBModel`, `FRBFitter`, `FRBParams`, `build_priors()` |
| `burstfit_casey_analysis.py` | Command‑line “notebook replacement” | `prep_dynamic()`, CLI main                             |
| `burstfit_modelselect.py`    | Sequential fits + BIC               | `fit_models_bic()`                                     |
| `burstfit_robust.py`         | Robustness diagnostics              | `subband_consistency()`, `leave_one_out_influence()`   |

---

## Diagnostics at a glance

* **Sub‑band consistency** — fit τ₁ GHz in N frequency chunks; large spread ⇒ per‑band systematics.
* **Leave‑one‑out influence** — χ²‑based heat‑map of how each channel pulls the global fit.
* **(Optional) SBC** — simulation‑based calibration helper planned for v2.1.

---

## Scintillation Pipeline

To be added.

---

## Citing & license

Please cite **Faber et al., *in prep.* (2025)** if you use this code.

---



These scripts contain helper functions that facilitate the analysis of burst properties within the PARSEC dashboard (see dsa110-pol repository).

---

## Scattering: new options (v0.1)

CLI flags in `scattering/run_scat_analysis.py` and `scat_analysis/burstfit_pipeline.py`:

- Alpha controls
  - `--alpha-fixed <val>`: fix frequency scaling exponent α (τ ∝ ν^-α)
  - `--alpha-mu <4.4>` / `--alpha-sigma <0.6>`: Gaussian prior for α when free

- DM offset
  - `--delta-dm-sigma <0.1>`: top-hat prior half-width for δDM around dm_init

- Likelihood
  - `--likelihood {gaussian,studentt}`: residual likelihood
  - `--studentt-nu <5.0>`: degrees of freedom for Student‑t

- Sampling
  - `--no-logspace`: disable log-space sampling for positive params

- Multi-component bursts (shared PBF)
  - `--ncomp K`: fit K Gaussian components sharing (τ_1GHz, α, δDM)
  - `--auto-components`: placeholder for greedy BIC selection (earmarked)

- Earmarks / future
  - `--sampler nested`: nested sampling route (earmarked)
  - `--anisotropy-enabled`, `--anisotropy-axial-ratio`: anisotropic kernels (earmarked)
  - `--baseline-order`, `--correlated-resid`: baseline marginalization & correlated residuals (earmarked)

Diagnostics:
- Residual ACF panel now includes PPC envelope (median and 5–95%).
- Δν_d diagnostic reported from τ via 2π τ Δν_d ≈ 1.16 (report only).
