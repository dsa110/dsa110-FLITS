# Implementation Plan: defensible burst E_iso (flux calibration gate)

**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Phase 1 Complete · Phase 2 Blocked-on-external-data
**Related:** [Calibration review](../analysis/burst_energies/CALIBRATION_REVIEW.md) · `../analysis/burst_energies/references.bib`

## Goal

Stop the energetics code from publishing fake isotropic-equivalent energies, and
leave a one-line seam so a real `E_iso` drops out the moment per-band flux
calibration arrives.

## Current state (before)

`analysis/calculate_burst_energies.py` summed two per-band fluence integrals
(`e_C + e_D`) whose amplitudes `c0_C`, `c0_D` are in **independent, uncalibrated,
per-telescope native units** (see the review), applied **no `(1+z)`**
k-correction, and wrote a publishable LaTeX erg table from those numbers. The
result is not an energy and its cross-burst ranking is unreliable.

## What changed — Phase 1 (buildable now, done)

- `scattering/configs/telescopes.yaml`: added `flux_jy_per_unit: null` per band
  (CHIME, DSA) — the absolute flux scale (Jy per native `.npy` fluence unit) from
  each telescope's SEFD + beam response at the burst position. `null` = uncalibrated.
- `analysis/calculate_burst_energies.py`:
  - New `band_energy_erg(I, flux_scale, d_l_m, z, kcorr=True)` — pure, applies the
    `(1+z)` bandwidth k-correction (Zhang 2018).
  - `compute()` gates on `flux_jy_per_unit` for **both** bands. Calibrated → per-band
    Jy energies + a legitimate sum + `(1+z)`. Uncalibrated → band fluence integrals
    in native units, flagged, **no energy**.
  - `main()` writes a calibration-pending LaTeX **stub** (no table) when uncalibrated,
    overwriting the stale fake-energy `burst_energies.tex`; loud stdout refusal.
  - `_check()` oracle: closed-form integral vs quadrature (kept) + flat-spectrum
    analytic energy + `(1+z)` halving + linearity in flux scale.

**Verification (run):** `--check` passes; full run takes the uncalibrated path
(refuses, native-unit fluences, stub `.tex`); a dummy-scale smoke confirms the
calibrated branch emits energies with the `(1+z)` ratio exact (chromatica
6.12e37 → 5.70e37 = /1.074).

## Phase 2 — Blocked on external data

Obtain, for each of the 8 spectroscopic-z sightlines:
- **CHIME** `flux_jy_per_unit`: SEFD + primary-beam response at the burst position
  (Andersen+2023 method; baseband Catalog 2024 for the position-corrected gain).
- **DSA-110** `flux_jy_per_unit`: SEFD + beam response (Law+2024 method; the
  standalone DSA-110 system paper is unpublished — use Law+2024 / Connor+2023 / thesis).

Then set the two `flux_jy_per_unit` values in `telescopes.yaml` and re-run; the
energy table regenerates automatically. (If the scale is per-burst rather than
per-band — beam position differs by sightline — extend `flux_scales()` to read a
per-burst override; not built until the data shape is known. YAGNI.)

## What we're NOT doing

- Not reprocessing raw voltages or touching telescope backends — calibration is a
  scalar SEFD×beam per band, supplied as config.
- Not extrapolating across the 0.8–1.3 GHz gap (kept band-restricted).
- Not building per-burst flux-scale plumbing until the data dictates it.

## Success criteria

- [x] `python analysis/calculate_burst_energies.py --check` passes.
- [x] Uncalibrated run emits NO erg and overwrites the stale energy `.tex` with a stub.
- [x] Calibrated branch applies `(1+z)` and sums only Jy-scaled bands (smoke-verified).
- [ ] Phase 2: real `flux_jy_per_unit` supplied for both bands → real `E_iso` table.

## Review gate

Run `/boris` over the Phase 1 diff (`analysis/calculate_burst_energies.py`,
`scattering/configs/telescopes.yaml`) before this feeds the manuscript.

## Open questions

(none)
