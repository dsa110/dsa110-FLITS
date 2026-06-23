# Handoff: Per-band `dt_min` for the multi-component joint fit (issue #37 remainder)

---
**Date:** 2026-06-22 20:30
**Author:** AI Assistant
**Status:** Handoff
**Branch:** main
**Commit:** eed6f04 (uncommitted changes present — see Git State)

---

## Task(s)

| Task | Status | Notes |
|------|--------|-------|
| Research #37 evidence kernel / N=1 commensurability | ✅ Complete | Found #37 ~90% already implemented (`force_multi`, `gain_s2`, kernel tests). |
| Plan per-band `dt_min` (the one open item) | ✅ Complete | Design fork resolved to per-band (scalar broadcast retained). |
| Implement per-band `dt_min` | ✅ Complete | Transform + caller; new tests; verified. |
| Validate | ✅ Complete (PASS) | Full suite 356 passed / 0 regressions. N-ladder resolved (PR #11): no spurious N=2 *selection* — nlive=300 per-band dlnZ_21 = −0.17 (vs old-max −5.66); per-band somewhat more permissive (`lnZ(N=2)` ~+5.9 higher). |
| Commit / PR | ✅ Complete | Committed `df23cce` (pathspec), pushed; PR #11 squash-merged (`3e45712`). |

**Current Workflow Phase:** Complete (merged)

## Workflow Artifacts

**Research:** [research-multicomponent-joint-evidence.md](research-multicomponent-joint-evidence.md) — maps the multi-component evidence kernel, the flat-vs-proper-prior contrast, and the #37-status table (kept current; `dt_min` row now Resolved).
**Plan:** [plan-dt-min-per-band.md](plan-dt-min-per-band.md) — 2 phases, success criteria, risks; Status Complete.
**Implement:** [implement-dt-min-per-band.md](implement-dt-min-per-band.md) — what landed, deviations, verification evidence.

## Critical References (read first)

- `scattering/scat_analysis/burstfit_joint.py` — `_JointPriorTransformOrdered` (~545-588) and `fit_joint_scattering` `dt_min` block (~882-897): the change.
- [implement-dt-min-per-band.md](implement-dt-min-per-band.md) — exact diff intent + verification.
- `tests/test_joint_prior_ordered.py` — the 5 new transform tests.

## Recent Changes (this task's lane only)

- `scattering/scat_analysis/burstfit_joint.py` (+24/-13):
  - `_JointPriorTransformOrdered.__init__` (~559-568): `dt_min` normalized to `list[float]`
    (scalar broadcast or per-group sequence; `ValueError` on length mismatch; plain list for pickling).
  - `__call__` (~570-588): `zip(self.t0_groups, self.dt_min, strict=True)`, per-group floor `dtm`.
  - `fit_joint_scattering` (~885-892): `dt_min=None` → `[3*median(|diff(time)|) for (model_C, model_D)]`
    = `[dt_C, dt_D]`, aligned with `[grp_C, grp_D]`; scalar override broadcasts. Comment rewritten.
- `tests/test_joint_prior_ordered.py` (new, 5 tests).
- `.agents/research-*`, `plan-*`, `implement-*` (workflow docs).

## Learnings

- **#37 was mostly already done** before this work: `force_multi` flag + `gain_s2` param
  (commensurate N=1 evidence) and `tests/test_gain_marginal_multi_band.py` (Woodbury / label-swap /
  rank-1) all pre-existed. Only the `dt_min` comment/code tension + per-band floor were open.
- The old code used `max(dts)` (coarser band's floor) for **both** bands while the comment claimed the
  tighter band binds — that over-constrained the finer (DSA) band. Per-band fixes both.
- The transform is pickled to dynesty pools — keep `self.dt_min` a plain `list`, not `np.asarray`.
- Pre-existing `B905` (`zip` without `strict=`) at `burstfit_joint.py:247` is in
  `_gain_marginal_multi_band` (kernel), scoped OUT; that file isn't in the repo's default ruff path.

## Git State / ⚠️ Separate lane (DO NOT bundle)

The working tree contains a **second, unrelated lane** authored by another active session
(`entire session list` shows a session "active now" in this worktree):
- `galaxies/v2_0/{config,engines,search,test_search_pipeline}.py` (modified, +200/-101)
- `analysis/burst_energies/`, `analysis/calculate_burst_energies.py` (untracked)
- `scratch/codetection/why_missed.py` (untracked)

These are **separate-active** — preserve, do not edit, do not commit. Commit this task with an
explicit pathspec ONLY:

```
git switch -c fix/issue-37-per-band-dt-min
git add scattering/scat_analysis/burstfit_joint.py tests/test_joint_prior_ordered.py .agents/*dt-min* .agents/research-multicomponent-joint-evidence.md
git commit -m "..."   # never `git add -A` / `git commit -a`
```

## Action Items & Next Steps

1. [x] Committed `df23cce` (pathspec), pushed; PR #11 squash-merged (`3e45712`).
2. [x] N=1 vs N=2 `lnZ` ladder done (synthetic single-component): no spurious N=2 *selection*.
       nlive=300: per-band dlnZ_21 = −0.17 (N=2 doesn't win) vs old-max −5.66; per-band's
       looser floor is measurably more permissive (`lnZ(N=2)` ~+5.9 higher), eroding the N=1
       margin but not flipping the choice. Result + nlive=300 correction on PR #11. A converged
       same-data magnitude is deferred to the HPCC recovery-campaign `/experiment` (real bursts).
3. [ ] (Optional, gated/outward) Post a "#37 mostly implemented; remainder = per-band `dt_min` done"
       status comment to the upstream issue (`dsa110/dsa110-FLITS#37`). **Still open.**
4. [ ] (Optional follow-up) `analysis/scattering-refit-2026-06/verify_zach_c2.py:124` hardcodes the old
       `max(dt_C,dt_D)*3` for a printed diagnostic — annotate/update if relied on.
5. [ ] (Optional, trivial) Fix the pre-existing `:247` B905 if the repo ever widens its ruff path.

**Recommended Next Command:** commit (pathspec) → optionally `/experiment` or a manual fit run for item 2.

## Other Notes

- Verification recorded via `verify-gate` for all task paths (methods: test / trivial). Full suite:
  `conda run -n flits python -m pytest -q -m "not slow"` → 356 passed, 1 skipped (needs `bursts.yaml`).
- No fit-validation 3-level contract applies — this is prior-transform infrastructure, not a science fit.

---

**Handoff created by AI Assistant on 2026-06-22**
