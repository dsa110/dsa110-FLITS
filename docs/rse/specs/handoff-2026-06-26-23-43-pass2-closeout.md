# Handoff: Pass 2 manuscript closeout — complete

---
**Date:** 2026-06-27 (final refresh)
**Author:** AI Assistant
**Status:** Pass 2 complete — pipeline PR #74 + Faber2026 PR #13 merged
**Branch:** pipeline `main` @ `c0696a6`; Faber2026 `main` @ `bad84ce` (submodule pin `c0696a6`)

---

## Task(s)

Pass 2 manuscript authority closeout per `/grill-with-docs` (2026-06-27). Fork B merged via Faber2026 PR #11; Pass 2 drove pipeline artifacts then Faber2026 table/prose.

| Task | Status | Notes |
|------|--------|-------|
| Lock Pass 2 grill decisions (7 items) | ✅ Complete | `.agents/pass-2-grill-decisions.md` |
| Phase 1: HPCC all-exp pull + gate regen + energy re-export | ✅ Complete | N=8; gate 11/11 MARGINAL; committed |
| Phase 2: fixed-s² all-exp HPCC campaign (24 jobs) | ✅ Complete | 24/24 COMPLETED; s² adjudication in `joint_ladder/` |
| Pipeline PR → `main` | ✅ Merged | [PR #74](https://github.com/jakobtfaber/dsa110-FLITS/pull/74) @ `c0696a6` |
| Phase 3: Faber2026 `draft/pass-2-closeout` | ✅ Complete | 8-row energies; zach out of `tab:alpha`; fixed-s² prose |
| Faber2026 stacked PR (ask-matt) | ✅ Merged | [PR #13](https://github.com/jakobtfaber/Faber2026/pull/13) @ `bad84ce`; `make all` PASS |

**Current workflow phase:** Ship complete. Next tranche is FINAL figure regen + budget overlay (`fig:budget`).

## Workflow artifacts

- `.agents/pass-2-grill-decisions.md` — locked decisions + Phase 1–3 results
- `docs/rse/specs/decision-map-manuscript-completion.md` — upstream ADR map
- `analysis/scattering-refit-2026-06/hpcc/submit_pass2_s2_allexp.sh` — 24-job s² recipe
- `analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py` — s² adjudication (fail-closed all-exp PBF)

## Critical references

- `.agents/pass-2-grill-decisions.md` — **read first**
- `Faber2026/CONTEXT.md` — manuscript language (Pass 2 roster; on `main` @ `bad84ce`)
- `docs/adr/0005-citable-alpha-roster.md` — Tier A/B α roster

## What landed on `main` (Pass 2)

**Pipeline merge [PR #74](https://github.com/jakobtfaber/dsa110-FLITS/pull/74) @ `c0696a6`:**
- ADR-0004 sub-Kolmogorov floor + gate regen
- N=8 burst energies re-export (`burst_energies.{json,tex}`)
- Pass 2 HPCC pull, s² grid, sbatch infra, grill decisions

**Faber2026 merge [PR #13](https://github.com/jakobtfaber/Faber2026/pull/13) @ `bad84ce`:**
- Submodule pin → `c0696a6`
- `tab:burst-energies`: six→eight rows (oran, whitney)
- `tab:alpha`: zach row removed
- Abstract/conclusions/results: fixed-s² adjudicated; population α qualitative pending FINAL regen

### s² adjudication (`_s2verdict.py`, all-exp canonical)

| Burst | Verdict | Manuscript implication |
|-------|---------|------------------------|
| oran | C2D1 vs C1D1 **REAL** | Prefer C2D1 ladder cell |
| isha | C2D1 vs C1D1 **REAL** | Prefer C2D1 |
| mahi | C2D1 vs C1D1 **weak** | No strong multiplicity upgrade |
| phineas | C3D3 vs C3D2 **NOT robust** | Do not claim extra DSA component |
| zach | C2D3 vs C2D2 **NOT robust** | Drop from `tab:alpha` (grill #6) |

## Reproducibility & data state

- **HPCC:** `ssh hpcc`; joint fits via `/central/scratch/jfaber/flits-runs/run_joint.sbatch`
- **Venv:** symlinks `/central/scratch/jfaber/envs/flits-joint`, `/home/jfaber/flits` → `_quarantine/flits-20260625`
- **Energies:** N=8 roster; c₀,γ from mixed-legacy `joint_json/` (α-independent per ADR)
- **Legacy mixed-PBF s² on HPCC:** must not use (`_s2verdict.py` fail-closed)

## Verification state

### Passing

- Gate regen: 11/11 MARGINAL
- Energy re-export: N=8, `--check` PASS
- Phase 2 HPCC: 24/24 COMPLETED
- `pytest` joint_ladder/test_s2verdict + burst_energies: 11 passed (post-merge)
- Faber2026 `make all` on `main` @ `bad84ce`

### Separate lanes (preserve)

| Lane | Location |
|------|----------|
| Codetection | `feat/codetection-freya` |
| Joint-model figures | `?? jointmodel_figs/`, `jointmodel_montage.svg` |
| Literature | `?? docs/literature/` |
| Faber2026 agent memory | `?? .remember/` |

## History: Phase 2 first attempt (infra only)

2026-06-26 first submit (64618351–64618374) failed immediately: `/home/jfaber/flits/venv` missing (tree quarantined 2026-06-25). Fixed via quarantine symlinks + sbatch `PATH` patch; resubmitted successfully same night.

## Action items

1. [x] **Merge [PR #74](https://github.com/jakobtfaber/dsa110-FLITS/pull/74)** — pipeline → `main`
2. [x] **Phase 3 Faber2026** — submodule pin + tables/prose
3. [x] **Stacked Faber2026 PR #13** — merged; LaTeX compiles

## Next tranche (@human / @decision)

- FINAL figure regeneration (`dsa_figs/` all-exp joint figures → Faber2026)
- Measured-vs-predicted budget overlay (`fig:budget`)
- Reconcile ADR-contradicting campaign counts in science-plan docs (deferred-tasks)
- Optional: delete 48 stale mixed-PBF `*_s2-*.json` under deletion-safety gate

**Recommended next skill:** `/handoff` into a fresh session for figure-regen campaign.

## Other notes

- Abstract pending clause (grill #7): fixed-s² adjudicated in Phase 3 prose; population α + budget overlay remain qualitative pending FINAL regen.
- Packaging order executed: pipeline PR first, Faber2026 stacked on submodule pin.

---

**Handoff closed 2026-06-27**
