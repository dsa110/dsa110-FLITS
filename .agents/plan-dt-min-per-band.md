# Implementation Plan: Per-band `dt_min` for the multi-component joint fit (issue #37 remainder)

---
**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Complete
**Related Documents:**
- [Research: Multi-component joint-fit evidence kernel & N=1 commensurability](research-multicomponent-joint-evidence.md)
- GitHub issue #37 (`dt_min` comment/code tension + "consider per-band `dt_min`")

---

## Overview

The research phase established that issue #37 is already implemented except for the
`dt_min` minimum-component-separation floor in the multi-component joint fit. Today a
single scalar `dt_min` is applied to both bands; when auto-derived it is
`max` over the two bands of `3 * median(|diff(time)|)` (the coarser band's floor),
while the adjacent comment states "the binding constraint is the tighter (smaller-dt)
band's resolution." This plan makes the floor **per band** so each band's components
are constrained by that band's own time resolution, which makes the comment true and
stops the finer band (DSA, ~1.4 GHz) from being over-constrained by the coarser band
(CHIME, ~0.6 GHz).

**Goal:** `_JointPriorTransformOrdered` enforces a per-group minimum separation, with
`fit_joint_scattering` deriving `dt_C` and `dt_D` independently. A scalar `dt_min`
still works (broadcast to all groups), so every existing caller is unaffected.

**Motivation:** The multi-component path exists for evidence-based component-count
(N) selection. An over-tight floor in the finer band forbids genuinely resolvable
close components there, biasing N selection in exactly the band with the best time
resolution. Per-band floors remove that bias and close both #37 `dt_min` items.

## Current State Analysis

**Existing Implementation:**
- `scattering/scat_analysis/burstfit_joint.py:534-577` — `_JointPriorTransformOrdered`;
  stores `self.dt_min = float(dt_min)` (`:557`), one scalar used for every group in
  `__call__` (`:568`, `:572`).
- `scattering/scat_analysis/burstfit_joint.py:874-886` — `fit_joint_scattering`
  derives `dt_min = max(dts)` where `dts = [3*median(|diff(time)|)]` per band
  (`:876-881`), then builds `_JointPriorTransformOrdered(spec, [grp_C, grp_D], dt_min=dt_min)`.
- Direct external construction with a scalar: `analysis/scattering-refit-2026-06/adv_merge_attack_independent.py:174`
  (`_JointPriorTransformOrdered(spec, [grp_C, grp_D], dt_min=dt_min)`).

**Current Behavior:** Both bands' `t0` groups share one separation floor equal to the
coarser band's 3-sample resolution.

**Current Limitations:**
- The comment (`:874-875`) contradicts the code (`max`, not the tighter band).
- The finer band cannot place components closer than the coarse band's floor, even
  when its own sampling would resolve them.

## Desired End State

**New Behavior:** Each `t0` group is constrained by its own `dt_min`. When
`fit_joint_scattering(dt_min=None)` (the default), `dt_C = 3*median(|diff(time_C)|)`
and `dt_D = 3*median(|diff(time_D)|)` are passed as `[dt_C, dt_D]`. An explicit scalar
`dt_min=<float>` is broadcast to all groups (unchanged override semantics).

**Success Looks Like:**
- A multi-component fit places the DSA band's components down to the DSA resolution
  while the CHIME band keeps its own (coarser) floor.
- Every realized prior draw still satisfies, per group, gap ≥ that group's `dt_min`.
- All existing kernel tests and external callers run unchanged.

## What We're NOT Doing

- [ ] Touching the evidence kernel `_gain_marginal_multi_band` — it is correct and
      tested; this is purely the prior transform + its caller.
- [ ] Changing the `force_multi` / `gain_s2` API (already implemented; #37 item 1).
- [ ] Adding an N-sweep / evidence-ladder helper (separate follow-up if wanted).
- [ ] Posting to GitHub issue #37 (outward-facing; deferred to a separate gated step).

**Rationale:** Keep the diff to the single open item; do not re-do landed work.

## Implementation Approach

**Technical Strategy:** Generalize `dt_min` in `_JointPriorTransformOrdered` to accept
either a scalar (broadcast) or a per-group sequence aligned with `t0_groups`; iterate
groups with their own floor in `__call__`. Derive per-band floors in
`fit_joint_scattering` only when `dt_min is None`.

**Key Architectural Decision:**
1. **Decision:** `dt_min` becomes scalar-or-sequence, stored internally as a list of
   length `len(t0_groups)`.
   - **Rationale:** Backward-compatible (scalar still valid), minimal surface change,
     and the only structural change the per-band floor requires.
   - **Trade-offs:** Slightly more `__init__` logic; no behavior change for scalar
     callers.
   - **Alternatives considered:** (a) keep `max` and only fix the comment — rejected
     because it leaves the finer band over-constrained; (b) switch to `min` — rejected
     because it lets the coarse band propose sub-resolution pairs that only the
     eigenvalue guard catches, rather than preventing them in the prior.

**Patterns to Follow:** The data-derived floor already uses `3*median(|diff(time)|)`
per band (`burstfit_joint.py:877-880`); reuse it verbatim, just don't collapse with
`max`. Store the normalized `self.dt_min` as a plain Python `list` of floats (not
`np.asarray`) — these transforms are pickled to dynesty pools (`:859`, `:886`), and a
list matches the file's explicit pickle-safety discipline.

## Implementation Phases

### Phase 1: Per-group floor in the transform

**Objective:** `_JointPriorTransformOrdered` honors a per-group `dt_min`.

**Tasks:**
- [ ] Generalize `__init__` to accept scalar or sequence.
  - Files: `scattering/scat_analysis/burstfit_joint.py:552-557`
  - Changes: store `self.dt_min` as a list of floats length `len(self.t0_groups)`
    — `[float(d) for d in dt_min]` when `np.ndim(dt_min)` else
    `[float(dt_min)] * len(self.t0_groups)`; assert lengths match when a sequence.
- [ ] Use the per-group floor in `__call__`.
  - Files: `scattering/scat_analysis/burstfit_joint.py:559-577`
  - Changes: `for grp, dtm in zip(self.t0_groups, self.dt_min):` and replace the two
    `self.dt_min` uses (`:568`, `:572`) with `dtm`.
- [ ] Update the class docstring (`:548-549`) to state the floor is per group.

**Verification:**
- [ ] A unit test pushes random cube points through a `[dtC, dtD]` transform and
      asserts each group's realized gaps ≥ its own floor; a scalar transform still
      yields a uniform floor.

### Phase 2: Per-band derivation in `fit_joint_scattering`

**Objective:** Default `dt_min=None` derives per-band floors.

**Tasks:**
- [ ] Replace the `max(dts)` collapse with a per-band list.
  - Files: `scattering/scat_analysis/burstfit_joint.py:874-886`
  - Changes: when `dt_min is None`, build `dt_min = [dt_C, dt_D]` with
    `dt_b = 3*median(|diff(time_b)|)`; when a float is passed, leave it scalar (the
    transform broadcasts). Pass it straight into `_JointPriorTransformOrdered`.
- [ ] Rewrite the comment (`:874-875`) to say each band is bound by its own resolution.

**Dependencies:** Requires Phase 1.

**Verification:**
- [ ] Construct the loglike+transform via the same path the function uses on two toy
      `FRBModel`s with different `time` grids; assert the CHIME group floor == its
      3-sample value and the DSA group floor == its (smaller) 3-sample value.

## Success Criteria

### Automated Verification
- [x] `pytest tests/test_gain_marginal_multi_band.py` still passes (existing 9). ✔
- [x] New transform tests pass: `tests/test_joint_prior_ordered.py` (5 tests — per-group
      gaps over 20k draws, scalar-broadcast == uniform sequence, degenerate collapse,
      length-mismatch raises, n<2 skipped). 14 passed combined. ✔
- [x] End-to-end smoke: `fit_joint_scattering` runs the per-band path (`dt_min=None` →
      [dtC=0.24, dtD=0.06]) and the scalar-override path (`dt_min=0.5`), both finite logZ. ✔
      *(Replaced the full `adv_merge_attack_independent.py` run — its GATE-3 probe exercises
      the scalar path, already covered by the scalar-broadcast unit test, and the full
      script runs nested fits. The transform-construction line `:174` is unchanged-compatible.)*
- [x] `ruff check` clean on the changed lines. One pre-existing B905 at
      `burstfit_joint.py:247` (inside `_gain_marginal_multi_band`, explicitly out of scope)
      remains; not introduced by this change. ✔

### Manual Verification
- [x] DSA group reaches smaller component separations than the old `max` floor while CHIME
      keeps its own — confirmed on synthetic two-band draws (min realized DSA-gap 0.0606 vs
      its 0.06 floor; the old `max` floor would have been 0.24). ✔
- [x] N=1 vs N=2 `lnZ` ladder, single-component truth, `force_multi=True`, fixed `gain_s2`:
      **no spurious N=2 selection**. nlive=300: per-band dlnZ_21 = **−0.17** (N=2 doesn't win)
      vs old-max **−5.66**; per-band's looser floor is measurably more permissive (`lnZ(N=2)`
      ~+5.9 higher), eroding the N=1 margin without flipping it. (Earlier nlive=40 probe −0.62 ±
      3.21 was noise-dominated; superseded — PR #11 correction.) Converged same-data magnitude
      on real bursts → HPCC campaign `/experiment`. ✔

## Testing Strategy

**Unit Tests** (`tests/test_gain_marginal_multi_band.py`, or a new
`tests/test_joint_prior_ordered.py`):
- [ ] Per-group floor: `[dtC, dtD]`, 20k cube draws, every group gap ≥ its floor.
- [ ] Scalar broadcast: `dt_min=dt` still applies `dt` to all groups (regression).
- [ ] Degenerate width (`hi - lo - (n-1)*dtm <= 0`) still collapses the group
      (existing branch `:573-575`) — guard against an off-by-one in the per-group loop.

**Existing coverage reused:** the adversarial GATE-3 probe in
`adv_merge_attack_independent.py` exercises the scalar path end-to-end.

## Backward Compatibility

A scalar `dt_min` (the only form any current caller uses, incl.
`adv_merge_attack_independent.py:174` and any explicit
`fit_joint_scattering(dt_min=<float>)`) is broadcast to all groups → identical
behavior. The public `dt_min: float | None` signature is unchanged.

## Risk Assessment
1. **Risk:** Off-by-one when iterating `zip(t0_groups, dt_min)` vs the old single scalar.
   - **Likelihood:** Low — **Impact:** Medium — **Mitigation:** the degenerate-width
     and per-group-gap unit tests above.
2. **Risk:** A looser DSA floor lets two near-identical DSA components survive and
   spuriously favor N=2.
   - **Likelihood:** Low — **Impact:** Medium — **Mitigation:** the eigenvalue/rank-1
     guard still Occam-penalizes a true merge; the manual N-ladder sanity check.
3. **Risk:** Breaking picklability (transforms go to dynesty pools).
   - **Likelihood:** Low — **Impact:** High — **Mitigation:** store `self.dt_min` as a
     plain `list[float]`; both list and scalar pickle trivially. Verified: the only
     reads of `self.dt_min` are `:568`/`:572`, both moving into the loop.

## Edge Cases and Error Handling
1. **Case:** Sequence `dt_min` length ≠ number of groups.
   - **Expected:** raise `ValueError` in `__init__` (fail fast, total cube map).
2. **Case:** Single-component group (`n < 2`).
   - **Expected:** unchanged — the `n < 2: continue` branch (`:563-564`) skips it.

## Documentation Updates
- [ ] Update the `_JointPriorTransformOrdered` docstring and the
      `fit_joint_scattering` `dt_min` comment.
- [ ] After implementation, mark the research doc's #37-status `dt_min` row resolved.
- [ ] Known-stale (out of core scope, follow-up): `analysis/scattering-refit-2026-06/verify_zach_c2.py:124`
      independently hardcodes `max(dt_C, dt_D)*3` for its printed "×dt_min" diagnostic.
      It will not break (it never builds the transform), but its printed multiples will
      no longer match the per-band production floors — update or annotate if that
      diagnostic is still relied on.

## Open Questions

*(None — the per-band approach is chosen; scalar broadcast preserves all callers.)*

---

## References

**Research Documents:**
- [Research: Multi-component joint-fit evidence kernel & N=1 commensurability](research-multicomponent-joint-evidence.md)

**Files To Change:**
- `scattering/scat_analysis/burstfit_joint.py` (`_JointPriorTransformOrdered`, `fit_joint_scattering`)
- `tests/test_gain_marginal_multi_band.py` or new `tests/test_joint_prior_ordered.py`

**Files Analyzed (unchanged):**
- `analysis/scattering-refit-2026-06/adv_merge_attack_independent.py` (scalar caller)

---

## Review History

### Version 1.0 — 2026-06-22
- Initial plan; `dt_min` design fork resolved to per-band (scalar broadcast retained).
