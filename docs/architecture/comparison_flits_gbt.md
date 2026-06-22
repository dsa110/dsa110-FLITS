# `dsa110-FLITS` vs. `FLITS_GBT` — Lineage & Comparison

This document compares the current `dsa110-FLITS` repository against its
predecessor, [`jakobtfaber/FLITS_GBT`](https://github.com/jakobtfaber/FLITS_GBT),
the original Breakthrough Listen / GBT-era FLITS pipeline.

**Bottom line:** `dsa110-FLITS` is **not** a reorganization of `FLITS_GBT`. It is a
near-total rewrite with a different scope. `FLITS_GBT` is a *voltage-domain*
extraction / coherent-dedispersion / polarimetric-calibration pipeline driven by
SPANDAK, DSPSR, and PSRCHIVE. `dsa110-FLITS` is an *intensity-domain* fitting
toolkit (scattering + scintillation MCMC, two-screen simulation, DM estimation)
built around the CHIME–DSA-110 co-detection sample. **No source files are shared
verbatim**, and the two repositories have independent git histories (i.e. this is
not a GitHub fork of `FLITS_GBT`).

---

## 1. Top-level structure

| `FLITS_GBT` (~39 commits, ~83 `.py`) | `dsa110-FLITS` (~202 commits, ~206 `.py`) |
|---|---|
| `extractor/` — raw voltage extraction & splicing | `flits/` — installable package (`batch`, `fitting`, `scattering`, `common`, `utils`) |
| `dedispersion/` — DSPSR coherent dedispersion, `DM_phase.py`, `OnPulseRMS.py` | `scattering/` — `burstfit` MCMC pulse-broadening pipeline |
| `calibration/` — pol/flux/RM calibration (`polfluxrm_auto.py`) | `scintillation/` — ACF scintillation-bandwidth pipeline + NE2001 |
| `intensity/` — `burst_drift.py`, `sburst_classifier.py`, `scintarc_frb.py` | `dispersion/` — `dmphasev2.py` (`DMPhaseEstimator`) |
| `playground/` — FRB121102, FRB180916 (GBT/GMRT) analyses | `simulation/` — two-screen scintillation simulator |
| `utilities/` — `run_spandak.py`, `gen_database.py` | `crossmatching/`, `galaxies/`, `animations/` |
| flat scripts, no packaging | `pyproject.toml`, `environment.yml`, `requirements.txt`, `docs/`, `tests/`, CI configs |

`dsa110-FLITS` additionally carries top-level docs/manifests that have no
counterpart in `FLITS_GBT`: `SCINTILLATION_PIPELINE_TECHNICAL_REPORT.md`,
`FLITS-Complete-Agent-Configuration-Guide.md`, `ONBOARDING_STATUS_REPORT.md`,
`codetections_manifest.yaml`, `DATA_LOCATIONS.md`, plus `.archive/` and
`.deprecated/` trees holding legacy code.

---

## 2. Shared files / modules

**None as files.** `dsa110-FLITS` keeps no verbatim copy of any `FLITS_GBT`
script. In particular, the modules called out for comparison are absent:

| `FLITS_GBT` module | Present in `dsa110-FLITS`? | Notes |
|---|---|---|
| `intensity/DM_phase.py` (also `dedispersion/DM_phase.py`) | No (file) / **algorithm reimplemented** | See `dispersion/dmphasev2.py` below |
| `intensity/burst_drift.py` (2D-ACF / ellipse drift-rate fit, `photutils`) | No | No drift-rate equivalent in active code |
| `intensity/sburst_classifier.py` (sub-burst ID; depends on `iautils`, `frb_common`) | No | No sub-burst classifier in active code |
| `intensity/scintarc_frb.py` (secondary-spectrum / scintillation-arc) | No | Scintillation handled via ACF pipeline, a different method |
| `extract2cdd_auto.py`, `polfluxrm_auto.py`, `splicer_raw.py`, `RMfit_curve.py` | No | Entire voltage/calibration half is absent |

The only carry-over is **algorithmic, not textual**: `dispersion/dmphasev2.py`
reimplements the same phase-coherence DM method as `FLITS_GBT/intensity/DM_phase.py`
— the coherent-power metric

$$P'_{\rm Co}(\omega, {\rm DM}) = \left| \sum_{\nu} w_\nu \frac{S(\omega, \nu)}{|S(\omega, \nu)|} \right|^2 \omega^2$$

but as a clean, vectorized `DMPhaseEstimator` class with bootstrap-derived
`σ_DM`, MAD-based channel weighting, and optional frequency-window selection —
rather than the original interactive CLI script built on PSRCHIVE.

---

## 3. New / expanded in `dsa110-FLITS`

- **`burstfit` scattering pipeline** (`scattering/scat_analysis/`): MCMC fitting
  of pulse-broadening models with sequential model selection (M0 → M1 → M2 → M3)
  via BIC, robustness diagnostics (sub-band consistency, leave-one-out), corner
  plots, and nested-sampling support. Entirely new.
- **Scintillation pipeline** (`scintillation/scint_analysis/`): frequency-ACF
  computation, scintillation-bandwidth / timescale fitting, noise subtraction,
  2D fitting, and NE2001 queries (`scintillation/ne2001/`).
- **Two-screen scintillation simulator** (`simulation/`): Numba-accelerated
  MW + host-galaxy two-screen physics; reproduces figures from Pradeep et al. (2025).
- **DSA-110-specific functionality**:
  - `crossmatching/` — DSA-110 ↔ CHIME TOA cross-matching with barycentric and
    geometric-delay corrections.
  - `galaxies/` — host / foreground galaxy catalog queries (DESI, Pan-STARRS,
    Legacy Survey, WISE), impact-parameter and stellar-mass estimation.
  - `codetections_manifest.yaml`, co-detection sample handling, `animations/`.
- **Packaging & infrastructure**: pip-installable `flits` package with console
  scripts (`flits-scat`, `flits-scint`, `flits-batch`, `flits-configs`), conda
  `environment.yml`, `docs/` site, `pytest` suite, Codacy/CI, and agent configs.
- **Dependency shift**: away from `PSRCHIVE` / `DSPSR` / `photutils` / `iautils` /
  `frb_common` (voltage + CHIME-pipeline tooling) → toward `emcee`, `lmfit`,
  `dynesty`, `chainconsumer`, `corner`, `numba`, and `mwprop` (NE2001).

---

## 4. Absorption of `SPANDAK_extension`, `DM_phase`, `DM-power`

| Upstream repo | Absorbed into `dsa110-FLITS`? | Detail |
|---|---|---|
| [`SPANDAK_extension`](https://github.com/jakobtfaber/SPANDAK_extension) | **No** | Zero references anywhere in the repo. The whole voltage-extraction / CDD / calibration stage is simply not present. |
| [`DM_phase`](https://github.com/jakobtfaber/DM_phase) | **Algorithm reimplemented** | `dispersion/dmphasev2.py` is a rewrite of the phase-coherence method, not a copy of the original file/package. |
| [`DM-power`](https://github.com/jakobtfaber/DM-power) | **Not as code** | Referenced only in docs/inventory notes. Its coherent-power concept overlaps with `dmphasev2.py`, but none of the `DM_power.py` / MPI code is present. |

What *was* folded in (per the README's "2026-04 monorepo fold"): the Mac-local
`chime_dsa_codetections/` tree, now under `.archive/`, `notebooks/codetections/`,
and a personal `burstfit` fork retained in `.archive/`.

---

## 5. Summary

`dsa110-FLITS` shares the **FLITS name and scientific lineage** with `FLITS_GBT`
but is effectively a new codebase: it drops the voltage-domain
extraction/dedispersion/calibration pipeline entirely, reimplements only the
DM phase-coherence idea, and adds substantial new intensity-domain analysis
(scattering MCMC, scintillation ACF, two-screen simulation) plus DSA-110-specific
co-detection, cross-matching, and host-galaxy tooling — all wrapped in a proper
installable package with docs, tests, and CI.
