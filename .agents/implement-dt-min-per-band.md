# Implementation Summary: Per-band `dt_min` (issue #37 remainder)

---
**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Complete
**Related Documents:**
- [Plan: Per-band dt_min](plan-dt-min-per-band.md)
- [Research: Multi-component joint-fit evidence kernel](research-multicomponent-joint-evidence.md)

---

## Summary

Both plan phases landed. `_JointPriorTransformOrdered` now accepts a scalar (broadcast)
or a per-group `dt_min`, and `fit_joint_scattering` derives `[dt_C, dt_D]` per band by
default. Each band's components are bound by that band's own time resolution; the
comment/code tension from issue #37 is resolved, and the finer band is no longer
over-constrained by the coarser one. All existing kernel tests pass unchanged; new
transform tests and an end-to-end smoke confirm the behavior.

## What Was Built

### Phase 1 — Per-group floor in the transform (complete)
- `scattering/scat_analysis/burstfit_joint.py` `_JointPriorTransformOrdered.__init__`:
  `dt_min` normalized to a `list[float]` of length `len(t0_groups)` — scalar broadcast,
  or a per-group sequence with a `ValueError` on length mismatch. Stored as a plain list
  (pickle-safe for dynesty pools).
- `__call__`: iterates `zip(self.t0_groups, self.dt_min, strict=True)`, using each group's
  own floor `dtm` in the `usable` width and the cumulative offset. `n<2` skip and the
  degenerate-width collapse branch are unchanged.
- Docstring updated to describe scalar-or-per-group.

### Phase 2 — Per-band derivation in the caller (complete)
- `fit_joint_scattering`: when `dt_min is None`, builds
  `[3*median(|diff(time)|) for m in (model_C, model_D)]` → `[dt_C, dt_D]`, aligned with
  `[grp_C, grp_D]`. An explicit scalar `dt_min` is left scalar (the transform broadcasts).
  Comment rewritten to state the per-band intent.

### Tests added
- `tests/test_joint_prior_ordered.py` (5 tests): per-group floor honored over 20k draws;
  scalar broadcast equals a uniform sequence (backward-compat); degenerate-width collapse;
  length-mismatch raises; `n<2` group skipped.

## Deviations From Plan

- **Backward-compat check substitution.** The plan listed running
  `adv_merge_attack_independent.py` (GATE-3 `dt_min` probe) as an automated criterion.
  That script runs nested fits and its GATE-3 only exercises the *scalar* path, which the
  new `test_scalar_broadcast_matches_uniform_sequence` covers directly (and the script's
  transform construction at `:174` passes a scalar, unchanged-compatible). Substituted a
  faster end-to-end `fit_joint_scattering` smoke (both per-band and scalar paths). Net
  coverage is equal-or-better; no behavior left unchecked.
- **Lint scope.** A pre-existing `B905` (`zip` without `strict=`) at
  `burstfit_joint.py:247` inside `_gain_marginal_multi_band` remains. The plan scoped the
  kernel OUT ("not touching `_gain_marginal_multi_band`"), and that file is not in the
  repo's default ruff path, so it was left untouched. The newly introduced `zip` got
  `strict=True`.

## Verification Results

- `conda run -n flits python -m pytest tests/test_joint_prior_ordered.py tests/test_gain_marginal_multi_band.py -q`
  → **14 passed** (5 new + 9 existing kernel regression).
- End-to-end smoke (`fit_joint_scattering`, components_C=2/components_D=1, tiny nlive):
  per-band path `dt_min=None` → floors `[0.24, 0.06]` (bands differ), finite logZ;
  scalar-override `dt_min=0.5` path, finite logZ. **SMOKE OK.**
- `ruff check scattering/scat_analysis/burstfit_joint.py tests/test_joint_prior_ordered.py`
  → clean except the pre-existing `:247` B905 noted above.
- Diff reviewed: band order `[model_C, model_D] → [grp_C, grp_D]` correctly aligned;
  no residual scalar read of `self.dt_min` (only `:568`/`:572`, both moved to the loop var).
- Post-merge science check (N=1 vs N=2 `lnZ` ladder, synthetic single-component truth): **no
  spurious N=2 selection**. nlive=300: per-band dlnZ_21 = **−0.17** (N=2 doesn't win) vs old-max
  **−5.66**. The per-band floor is measurably more permissive — `lnZ(N=2)` ~+5.9 higher than
  old-max — eroding the N=1 margin without flipping the choice. (An earlier nlive=40 probe gave
  −0.62 ± 3.21 but was noise-dominated; superseded — see PR #11 correction.) A converged
  same-data magnitude on real bursts is the HPCC recovery-campaign's job.

## Files Changed
- `scattering/scat_analysis/burstfit_joint.py` — transform + caller (per-band `dt_min`).
- `tests/test_joint_prior_ordered.py` — new (5 tests).
- `.agents/plan-dt-min-per-band.md` — checkmarks / status → Complete.

## Follow-ups (out of scope)
- `analysis/scattering-refit-2026-06/verify_zach_c2.py:124` still hardcodes
  `max(dt_C, dt_D)*3` for a printed diagnostic; will not break, but its printed multiples
  no longer match the per-band production floors. Update/annotate if relied on.
- Pre-existing `B905` at `burstfit_joint.py:247` (kernel) — trivial one-line fix if the
  repo ever widens its ruff path.
- Optional: post the #37 status comment upstream (gated, outward-facing) — deferred.
- Optional: an N-sweep evidence-ladder helper (`force_multi=True`, fixed `gain_s2`) —
  separate feature, not part of this change.
