# Implementation Summary: Radiometer flux calibration (S/N → Jy) for FLITS burst energetics

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Status:** Complete
**Plan Reference:** [plan-radiometer-flux-cal.md](plan-radiometer-flux-cal.md)

---

## Overview

Converted uncalibrated FLITS dynamic-spectrum amplitudes into physical Jy for both CHIME
(0.4–0.8 GHz) and DSA-110 (1.31–1.50 GHz), so the `flux_jy_per_unit` config seam carries a real
data-driven scale and the isotropic-equivalent energy (`E_iso`) table in
`analysis/calculate_burst_energies.py` unblocks. The band fluence is computed model-based from the
joint scattering fit: `F = ∫ σ_S(ν) · [c₀(ν/ν_ref)^γ / noise_std(ν)] dν` over the fit's valid
channels, with `σ_S` the per-channel radiometer noise (DSA measured beam + coherent-beam SEFD;
CHIME documented cylinder beam G≈1 + system SEFD). Both bands calibrated → the both-bands-or-nothing
gate opens → a real `E_iso` table with the (1+z) k-correction and propagated error bars.

**Implementation Duration:** 2026-06-22 → 2026-06-23

**Final Status:** ✅ Complete

## Plan Adherence

**Plan Followed:** [plan-radiometer-flux-cal.md](plan-radiometer-flux-cal.md)

**Deviations from Plan:**

- **Deviation 1:** Estimator switched from a per-channel **on-pulse S/N sum** to the **model-based**
  fluence `∫ σ_S · c₀(ν/ν_ref)^γ / noise_std dν`.
  - **Reason:** The Gaussian⊗exp scattering kernel is unit-area (`∫model dt = c₀`, verified
    numerically), so the model-based fluence is tail-complete and needs only `c₀/γ` — no on-pulse
    window to tune. A physics justification was written before the refit (CALIBRATION_REVIEW.md,
    "Estimator choice").
  - **Impact:** Robust, window-free; the linear-integral and model-based estimators agree to within
    the catalog cross-check tolerance.

- **Deviation 2:** `c₀` required a `/noise_std` correction and a **valid-channel mask**.
  - **Reason:** The joint fit's data is bandpass-normalized (`noise_std ≈ 0.04`, not z-scored to 1),
    so `c₀` is in bandpass units; dividing by `noise_std` converts it to S/N. RFI-masked channels
    have `noise_std ≈ 0` and must be excluded or the integral explodes (wilhelm 5.8×10⁴⁹ →
    9.9×10³⁹ erg once restricted to `m.valid`).
  - **Impact:** All 8 energies land physical; the correction is asserted by a catalog cross-check test.

- **Deviation 3:** CHIME beam handled as a **documented cylinder approximation** (G≈1 for
  baseband-localized bursts) rather than a measured beam cube.
  - **Reason:** CHIME baseband beams are formed at the source position, so the primary-beam
    attenuation at the burst is ≈1 — opposite to DSA's fixed-pointing transit offset. Documented in
    research-chime-singlebeam-flux-units.md (Phase 6 resolution).
  - **Impact:** CHIME `σ_S` reduces to SEFD/√(n_pol·Δν·Δt); SEFD derived analytically (34.5 Jy).

## Phases Completed

### Phase 5 (+ 5c-B): DSA per-channel calibration, config seam, rigorous γ_D refit
- ✅ **Status:** Complete
- **Completion Date:** 2026-06-22
- **Summary:** Added `analysis/flux_cal.py` radiometer kernel + DSA band-fluence integral; wired the
  `"fluxcal"` sentinel into `calculate_burst_energies.py` behind the both-bands gate. Rigorous
  calibrated refit (`refit_calibrated.py`) established the DSA γ_D rail is a **real** steep
  single-band spectrum (partly beam-correlated, not a joint-coupling artifact). Commits `3c13a71`,
  `c7a756a`.

### Phase 6: CHIME primary beam + SEFD
- ✅ **Status:** Complete
- **Completion Date:** 2026-06-23
- **Summary:** Added `analysis/chime_beam.py` (separable-Gaussian cylinder beam, G≈1 at the
  baseband source position) and `chime_sefd.csv` with the analytically-derived SEFD (34.5 Jy =
  2k_B·Tsys/(η·A)). Commit `7ae7c93`.

### Phase 7: Open the E_iso gate — combined table + validation
- ✅ **Status:** Complete
- **Completion Date:** 2026-06-23
- **Summary:** Both bands route through `joint_band_fluence_jy_ms_hz`; gate opens; `E_iso` table
  emitted with the (1+z) k-correction and a `±` error column (c₀ posterior width ⊕ SEFD/beam
  systematic). All 8 spectroscopic-z sightlines land 4.6×10³⁸ – 1.1×10⁴¹ erg. Commit `6e1456c`.

## Files Modified

**Created:**
- `analysis/flux_cal.py` — radiometer kernel + per-channel/model-based band-fluence integrals.
- `analysis/chime_beam.py` — CHIME documented cylinder beam + SEFD helpers.
- `analysis/burst_energies/refit_calibrated.py` — rigorous γ_D refit (Phase 5c-B).
- `analysis/burst_energies/{dsa_sefd,dsa_pointing,chime_sefd}.csv` — instrument inputs w/ provenance.
- `analysis/burst_energies/CALIBRATION_REVIEW.md` — estimator justification + γ_D-rail findings.
- `tests/test_flux_cal.py`, `tests/test_chime_beam.py`, `tests/test_burst_energies_fluxcal.py` — tests.

**Modified:**
- `analysis/calculate_burst_energies.py` — `"fluxcal"` routing, error propagation, `±` LaTeX column,
  energy-table (not pending-stub) output.
- `scattering/configs/telescopes.yaml` — both bands `flux_jy_per_unit: fluxcal`.

**Deleted:** No files deleted.

## Key Changes Summary

1. **Model-based band fluence** — `analysis/flux_cal.py:joint_band_fluence_jy_ms_hz`
   integrates `σ_S(ν)·c₀(ν/ν_ref)^γ/noise_std(ν)` over `m.valid` channels; tail-complete via the
   unit-area scattering kernel.
2. **Energetics gate opened** — `analysis/calculate_burst_energies.py:compute` sums both Jy band
   integrals into `E_iso` only when both bands carry a scale; `_tex_val_err` renders `(a±b)×10^c`.
3. **Instrument models** — DSA measured beam + coherent-beam SEFD; CHIME cylinder beam (G≈1) +
   analytic SEFD (`analysis/chime_beam.py`).

## Verification Results

### Automated Verification

- ✅ `pytest tests/test_flux_cal.py tests/test_burst_energies_fluxcal.py` — 14 passed
- ✅ `pytest tests/ -m "not slow"` — 74 passed
- ✅ `python analysis/flux_cal.py --check` — `self-check OK` (radiometer + flat-band oracles)
- ✅ `python analysis/calculate_burst_energies.py --check` — `self-check OK` (integral vs quadrature,
  energy oracle, k-correction identity, gate logic)
- ✅ `ruff check analysis/flux_cal.py analysis/chime_beam.py` — clean
- ✅ `burst_energies.tex` is an energy table (no "calibration pending"); every row has a `±` error bar

**Command Output:**
```
self-check OK: integral matches quadrature; energy oracle, k-correction, and gate exact
14 passed in 20.17s    |    74 passed, 3 deselected
```

### Manual Verification

- ✅ Energies in 10³⁸–10⁴¹ erg (7/8); wilhelm 1.1×10⁴¹ just above the upper edge, consistent with
  the most distant/luminous sightline (z=0.51) — user-confirmed acceptable.
- ✅ Published-fluence cross-check: model-based DSA fluence vs Law+2024 — oran 0.99×, zach 1.27×,
  whitney 2.16× (within factor ~2).
- ✅ Bandpass diagnostic (`refit_calibrated.review.json`): figure-reviewed `match`.
- ✅ DSA SEFD sanity vs the per-element measurement — user-confirmed.

## Issues Encountered

### Issue 1: wilhelm E_CHIME = 5.8×10⁴⁹ erg (9 orders high)
- **Impact:** One burst's CHIME-band integral blew up.
- **Resolution:** Restricted the integral to `m.valid` channels — RFI-masked channels with
  `noise_std ≈ 0` were dividing `c₀` by near-zero. After the fix: 9.9×10³⁹ erg.
- **Files Affected:** `analysis/flux_cal.py` (`_band_noise_grid`, `joint_band_fluence_jy_ms_hz`).

### Issue 2: model-based DSA fluence 14–25× low vs catalog
- **Impact:** Calibrated fluences far below Law+2024.
- **Resolution:** `c₀` is in bandpass-normalized units (`noise_std ≈ 0.04`), not S/N; added the
  `/noise_std` correction.
- **Files Affected:** `analysis/flux_cal.py`.

### Issue 3: post-edit autoformatter stripped an import
- **Impact:** `NameError` on a lambda's deferred reference.
- **Resolution:** Add imports in the same edit as their first consumer (per repo CLAUDE.md).
- **Files Affected:** `analysis/calculate_burst_energies.py`.

## Testing Summary

**Tests Added:**
- `tests/test_burst_energies_fluxcal.py::test_both_bands_emit` — gate opens; (1+z) k-correction
  identity; finite positive `E_iso_erg_err`.
- `tests/test_flux_cal.py::test_joint_band_fluence_matches_catalog_scale` — model-based DSA fluence
  within factor 3 of Law+2024 (guards the `/noise_std` + valid-channel fixes).
- `tests/test_chime_beam.py` — 5 tests (boresight gain, half-power FWHM, chromatic scaling,
  radiometer σ, SEFD derivation).

**All Tests Passing:** ✅ Yes (14 targeted; 74 in `tests/`)

## Performance Observations

Performance was not a primary concern. Band-fluence I/O dominates (loads each burst `.npy`); the
energetics run completes in seconds for the 8-burst table.

## Documentation Updated

- ✅ `analysis/burst_energies/CALIBRATION_REVIEW.md` — estimator justification + γ_D-rail findings.
- ✅ `docs/rse/specs/research-chime-singlebeam-flux-units.md` — Phase 6 resolution.
- ✅ `scattering/configs/telescopes.yaml` — `flux_jy_per_unit: fluxcal` with explanatory comments.
- ✅ **Faber2026 manuscript** (separate repo `jakobtfaber/Faber2026`, branch `feature/burst-energetics`,
  commit `26b8072`): added the `Isotropic-equivalent energies` Results subsection (deluxetable + bib
  entries). Not pushed.

## Remaining Work

- [ ] Rebase `feature/cluster-catalog-engine` onto `feature/radiometer-flux-cal` to fold in Phase 7
      (the cluster branch diverged before `6e1456c`).
- [ ] Push `feature/radiometer-flux-cal` and `feature/burst-energetics` / open PRs (held for sign-off).
- [ ] Verify the Faber2026 bib metadata (`Law2024` vol/page, `Michilli2021` authors) against ADS.

## Next Steps

1. Validate against the plan with `ai-research-workflows:validating-implementations`.
2. Push branches / open PRs once approved.
3. Swap manuscript nicknames → TNS names at submission (flagged with a `% TODO` in `results.tex`).

**Recommended Actions:**
- Perform systematic validation against the plan
- Push `feature/radiometer-flux-cal` and open a PR for review

## Lessons Learned

**What Went Well:**
- The unit-area scattering kernel made the model-based estimator window-free and tail-complete.
- Catalog cross-checks (Law+2024) caught both unit bugs (`/noise_std`, valid-channel mask).

**What Could Be Improved:**
- The bandpass-normalization unit of `c₀` was non-obvious and cost two debugging passes; worth a
  one-line note at the joint-fit output.

**Technical Insights:**
- DSA γ_D rail is real astrophysics (steep single-band spectrum), not a calibration artifact — it
  partly relaxes under beam correction but stays steep on-axis (phineas).
- CHIME baseband beams are formed at the source ⇒ G≈1, unlike DSA's transit-pointing offset.

## References

**Plan Document:**
- [Plan: Radiometer flux calibration](plan-radiometer-flux-cal.md)

**Research Documents:**
- [Research: CHIME singlebeam flux units](research-chime-singlebeam-flux-units.md)

**Commits:**
- `3c13a71` — Rigorous calibrated re-fit: the DSA gamma_D rail is a real steep spectrum, partly beam
- `c7a756a` — feat(flux): calibrated-energetics outputs + flux-cal config seam
- `7ae7c93` — feat(energetics): Phase 6 — CHIME beam + SEFD (documented cylinder approximation)
- `6e1456c` — feat(energetics): open E_iso gate — calibrated table with k-corr + error bars
- `26b8072` — results: isotropic-equivalent energy table from CHIME-DSA joint fit *(Faber2026 repo)*

---

**Implementation completed by AI Assistant on 2026-06-23**
