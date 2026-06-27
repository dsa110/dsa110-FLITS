# Validation: Figure regen Phase 1 — fig:budget τ overlay

> Validated against [handoff-2026-06-27-02-03-figure-regen-phase1.md](handoff-2026-06-27-02-03-figure-regen-phase1.md) and [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md) Phase 1 @ commit `068f6e79` (PR #77 merge) on 2026-06-27.

## Implementation Status

| Phase 1 step | Verdict | Evidence |
|--------------|---------|----------|
| Wire all-exp joint τ via citable roster + ADR-0004 gate | ✅ Pass | `load_allexp_joint_tau_for_budget()` in `tau_consistency.py`; `_find_best_tau_fit()` prefers `allexp_joint` in `sightline_budget.py` |
| Partial photo-z CSV promotion (isha/phineas/whitney) | ✅ Pass (partial) | `results/isha_galaxies.csv`, `phineas_galaxies.csv`, `whitney_galaxies.csv` on main @ `068f6e79`; full `scratch/photoz-fix/` promotion still @decision |
| Regenerate budget artifacts | ✅ Pass | Fresh `python -m galaxies.foreground.sightline_budget` → CSV/PNG/MD |
| 9 measured τ rows on budget | ✅ Pass | CSV: 9 finite `tau_obs_ms` (Tier A/B + whitney + johndoeII per roster) |
| Tests | ✅ Pass | 25/25 foreground tests |
| Figure-review gate (budget PNG) | ✅ Pass (prior session) | `results/figures.review.json` verdict `match` for 9 diamonds |
| Faber2026 copy + prose | ⏸ Manual | @human — not in PR #77 scope |

**Overall Phase 1 verdict: PASS** — code merged; manuscript sync remains @human.

## Automated Verification Results

- ✅ `pytest galaxies/foreground/test_sightline_budget.py galaxies/foreground/test_tau_consistency.py -q` — **25 passed** (conda `flits`, 2026-06-27, worktree @ `068f6e79`)
- ✅ `python -m galaxies.foreground.sightline_budget` — wrote `results/sightline_dm_scattering_budget.{csv,png,md}`; **9 measured τ** sightlines
- ✅ PR #77 CI — merged @ `068f6e795176e7301ac3c31feff5fb934208bf6a` (review + Python 3.10/3.12 + Socket green pre-merge)

## Code Review Findings

**Matches plan**

- Citable roster authority via `citable_alpha_roster.json` + ADR-0005; MARGINAL ingest only for `source=allexp_joint`.
- Legend uses “measured burst” not false PASS label.
- Budget plotter unchanged; overlay unblocked by data wiring.

**Deviations**

- Photo-z promotion partial (3/12 CSVs) — acceptable per handoff @decision gate; not a Phase 1 blocker.
- johndoeII appears as 9th measured τ (Tier B) though excluded from some manuscript α prose — documented in handoff.

**No regressions observed** in targeted foreground test suite.

## Manual Testing Required

- [ ] Copy `results/sightline_dm_scattering_budget.{png,svg}` → `Faber2026/figures/` (@human)
- [ ] Update `Faber2026/sections/results.tex:52-53` — remove “not yet overlaid” (@human)
- [ ] `make all` on Faber2026 with new figure (@human)

## Recommendations

### Critical

- None for Phase 1 code merge.

### Important

- Promote remaining `scratch/photoz-fix/*_galaxies.csv` before claiming full budget foreground fidelity (@decision).
- Faber2026 lane closeout after figure copy.

### Follow-Up

- Phase 2 all-exp joint figures (`handoff-figure-regen-2026-06-27.md` §Phase 2).

## References

- [handoff-2026-06-27-02-03-figure-regen-phase1.md](handoff-2026-06-27-02-03-figure-regen-phase1.md)
- [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md)
- [PR #77](https://github.com/jakobtfaber/dsa110-FLITS/pull/77)
