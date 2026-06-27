# Handoff: Figure regen Phase 1 — fig:budget τ overlay (PR #77 open)

---
**Date:** 2026-06-27 02:03
**Author:** AI Assistant
**Status:** Implement — Phase 1 code complete; merge + Faber2026 sync pending
**Branch:** `feat/final-figure-regen` @ `72df1290` (worktree)
**Commit:** `72df1290` (pushed); base `main` @ `abe62604` (PR #76 handoff merged)
**Prior handoffs:** [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md), [handoff-2026-06-26-23-43-pass2-closeout.md](handoff-2026-06-26-23-43-pass2-closeout.md)

---

## Task(s)

Pass 2 manuscript closeout is **complete**. This session started the **FINAL figure regen** tranche per ask-matt routing handoff (PR #76).

| Task | Status | Notes |
|------|--------|-------|
| Pass 2 pipeline + Faber2026 | ✅ Complete | PR #74 @ `c0696a6`, PR #13 @ `bad84ce` |
| Figure-regen routing handoff | ✅ Merged | PR #76 @ `abe62604` |
| **Phase 1: fig:budget measured τ overlay** | 🔄 PR open | [PR #77](https://github.com/jakobtfaber/dsa110-FLITS/pull/77) — CI green |
| Phase 2: all-exp joint figures → `dsa_figs/` | 📋 Planned | Generators uncommitted; HPCC pulls partial |
| Faber2026 budget figure + prose sync | 📋 Blocked | @human — copy PNG/SVG; drop “not yet overlaid” in `results.tex:52-53` |
| Full photo-z promotion → `results/` | 📋 @decision | Only isha/phineas/whitney CSVs updated so far |

**Current workflow phase:** Implement (Validate after PR #77 merge)

## Workflow artifacts

- [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md) — campaign routing (superseded for Phase 1 detail by this doc)
- [handoff-2026-06-26-23-43-pass2-closeout.md](handoff-2026-06-26-23-43-pass2-closeout.md) — Pass 2 closed
- [ALLEXP_PBF_RUN.md](../../analysis/scattering-refit-2026-06/joint_ladder/ALLEXP_PBF_RUN.md) — joint figure regen runbook (L131+)
- [0005-citable-alpha-roster.md](../../docs/adr/0005-citable-alpha-roster.md) — τ overlay roster authority
- `.agents/deferred-tasks.md` — #31 figures, #37 photo-z budget

## Critical references

- `galaxies/foreground/tau_consistency.py:158` — `load_allexp_joint_tau_for_budget()` (citable roster + ADR-0004 gate)
- `galaxies/foreground/sightline_budget.py:309` — `_find_best_tau_fit()` prefers all-exp joint
- `analysis/scattering-refit-2026-06/citable_alpha_roster.json` — Tier A/B + whitney exemplar nicknames
- `Faber2026/sections/results.tex:36-53` — prose still says measured τ “not yet overlaid”

## Recent changes (PR #77 branch)

- `galaxies/foreground/tau_consistency.py` — `load_allexp_joint_tau_for_budget`, `load_citable_budget_nicknames`, `find_citable_joint_json`
- `galaxies/foreground/sightline_budget.py:309-435` — ingest MARGINAL all-exp joint τ; fix withheld `x` when measured τ present; legend “measured burst” (not false PASS)
- `galaxies/foreground/test_*.py` — 25 tests (casey joint τ, mahi excluded, budget ingests MARGINAL)
- `results/isha_galaxies.csv`, `phineas_galaxies.csv`, `whitney_galaxies.csv` — photo-z foreground updates
- `results/figures.manifest.json`, `results/figures.review.json` — budget PNG reviewed `match`
- `results/sightline_dm_scattering_budget.md` — regen table (PNG/CSV gitignored)

## Reproducibility & data state

- **Env:** conda `flits` (Python 3.12)
- **Worktree (required):** `~/Developer/scratch/worktrees/flits-final-figure-regen` on `feat/final-figure-regen`
- **Regen command:** `python -m galaxies.foreground.sightline_budget` → writes `results/sightline_dm_scattering_budget.{csv,png,md,svg}` (csv/png gitignored)
- **All-exp fits:** `_a1_fits/` + `citable_alpha_roster.json`; whitney uses `local_runs/.../whitney_fine_joint_fit_C2D2_pbf-exp-exp.json`
- **Photo-z canonical (not fully promoted):** `scratch/photoz-fix/` — Casey PSF star removed per deferred-tasks
- **Faber2026:** `main` @ `bad84ce`; submodule pin `c0696a6` (pipeline submodule dirty from separate lanes)

## Verification state / known-broken

### Passing

- `pytest galaxies/foreground/test_sightline_budget.py galaxies/foreground/test_tau_consistency.py` — **25 passed** (worktree, 2026-06-27)
- PR #77 CI — review + Python 3.10/3.12 + Socket **SUCCESS**
- Figure review — `results/figures.review.json` verdict `match` for budget PNG (9 measured τ diamonds per manifest)
- verify-gate recorded on Phase 1 code paths (`sha256=0318390a92f6`)

### Uncommitted / unpushed (separate lanes — do not sweep)

| Lane | Location |
|------|----------|
| Faber2026 agent memory | `Faber2026/.remember/` untracked |
| Pipeline submodule dirty | `Faber2026/pipeline` — untracked `jointmodel_figs/`, `jointmodel_montage.svg`, `docs/literature/` |
| Entire tracing | canonical `dsa110-FLITS` `docs/handoff-figure-regen-2026-06` branch — `docs/entire-tracing-checkpoints.md` dirty |
| GDrive authority | PR #73 `@separate-lane` |

### Unverified / decision pending

- **Faber2026 `make all`** not rerun with new budget figure (figure not copied yet)
- **Phase 2 joint figures** not regenerated; `_tau_ladder_allexp.py` / `_ppc_montage_allexp.py` still uncommitted
- **Full photo-z promotion** — only 3/12 galaxy CSVs updated; Casey promotion path in `scratch/photoz-fix/casey_galaxies.csv` exists but not diff-checked this session
- **johndoeII** in Tier B roster — may appear as 9th measured τ on budget though excluded from some manuscript α prose

## Learnings

- `make_budget_figure` already plotted measured τ when `tau_obs_ms` populated — blocker was **data wiring**, not missing plotter (`sightline_budget.py:744+`).
- All-exp joint τ must go through `gate_one` + citable roster; ingest **MARGINAL** for `source=allexp_joint` only (`sightline_budget.py:427-430`).
- Budget PNG/CSV are **gitignored** — manuscript copy goes to `Faber2026/figures/` manually or via separate PR.
- One agent per worktree: implement on `flits-final-figure-regen`, not Faber2026 main checkout.

## Action items

1. [ ] **Merge [PR #77](https://github.com/jakobtfaber/dsa110-FLITS/pull/77)** (@human review)
2. [ ] Copy `results/sightline_dm_scattering_budget.{png,svg}` → `Faber2026/figures/`; update `results.tex:52-53` + conclusions overlay language; `make all` (@human / Faber2026 PR)
3. [ ] **Phase 2** — pull HPCC all-exp JSONs; commit generators; regen `dsa_figs/` per `ALLEXP_PBF_RUN.md`; figure-review gate
4. [ ] @decision — promote remaining `scratch/photoz-fix/*_galaxies.csv` → `results/` or authorize scratch-only regen
5. [ ] Optional — reconcile campaign counts (deferred-tasks #29); delete stale mixed-PBF s² JSONs (@human)

**Recommended next skill:** `ai-research-workflows:validating-implementations` after PR #77 merge; then `ai-research-workflows:implementing-plans` for Phase 2 from [handoff-figure-regen-2026-06-27.md](handoff-figure-regen-2026-06-27.md).

## Other notes

- Submodule bump on Faber2026 optional until pipeline code merges beyond `c0696a6`.
- Do **not** touch PR #73 gdrive lane or codetection `feat/codetection-freya`.

---

**Handoff created 2026-06-27**
