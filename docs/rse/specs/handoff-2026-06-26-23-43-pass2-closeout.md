# Handoff: Pass 2 manuscript closeout — pipeline PR open, Phase 3 next

---
**Date:** 2026-06-27 (refreshed)
**Author:** AI Assistant
**Status:** Phase 1–2 complete; pipeline PR #74 mergeable; Phase 3 (Faber2026) next
**Branch:** pipeline `manuscript-pass-2-2026-06` @ `a4bfd48` (pushed); Faber2026 `main` @ `5c87f6c` (submodule pin pending Phase 3)

---

## Task(s)

Pass 2 manuscript authority closeout per `/grill-with-docs` (2026-06-27). Fork B merged via Faber2026 PR #11; Pass 2 drives pipeline artifacts then Faber2026 table/prose.

| Task | Status | Notes |
|------|--------|-------|
| Lock Pass 2 grill decisions (7 items) | ✅ Complete | `.agents/pass-2-grill-decisions.md` |
| Phase 1: HPCC all-exp pull + gate regen + energy re-export | ✅ Complete | N=8; gate 11/11 MARGINAL; committed |
| Phase 2: fixed-s² all-exp HPCC campaign (24 jobs) | ✅ Complete | 24/24 COMPLETED; s² adjudication in `joint_ladder/` |
| Pipeline PR → `main` | ✅ Open | [PR #74](https://github.com/jakobtfaber/dsa110-FLITS/pull/74) — MERGEABLE; CI may still be running |
| Phase 3: Faber2026 `draft/pass-2-closeout` | 🔄 Next | After #74 merges: submodule pin + tables/prose |
| Faber2026 stacked PR (ask-matt) | 📋 Blocked | Gated on pipeline merge + submodule bump |

**Current workflow phase:** Implement (Phase 3 manuscript) — pipeline ship gate is PR review/merge.

## Workflow artifacts

- `.agents/pass-2-grill-decisions.md` — locked decisions + Phase 1/2 results
- `docs/rse/specs/decision-map-manuscript-completion.md` — upstream ADR map
- `analysis/scattering-refit-2026-06/hpcc/submit_pass2_s2_allexp.sh` — 24-job s² recipe
- `analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py` — s² adjudication (fail-closed all-exp PBF)

## Critical references

- `.agents/pass-2-grill-decisions.md` — **read first**
- `Faber2026/CONTEXT.md` — manuscript language (Pass 2 roster; uncommitted on Faber2026 `main`)
- `docs/adr/0005-citable-alpha-roster.md` — Tier A/B α roster (on `main` after merge)

## What landed on `manuscript-pass-2-2026-06`

**Commits (vs `main` base):**
- `31b63c1` — ADR-0004 sub-Kolmogorov floor + gate regen
- `c6f76ce` — N=8 burst energies re-export
- `5a0ba9b` — batch joint-model figure export (manuscript montage lane)
- `4c4dbd2` — Pass 2: HPCC pull, s² grid, sbatch infra, grill decisions
- `a4bfd48` — merge `origin/main` (ADR-0005 citable-α roster)

**Phase 1 artifacts (committed):**
- `_hpcc_pull/joint/` — 33 all-exp JSONs (scattering-only; no c₀,γ on cluster)
- `burst_energies.provenance.json` — n_bursts=8, mixed-legacy c₀,γ authority
- Gate: **11/11 MARGINAL, 0 L1 FAIL**

**Phase 2 artifacts (committed):**
- `joint_ladder/` — 24 new s² JSONs (oran/isha/mahi/phineas) + 6 zach pre-existing
- `hpcc/run_joint.sbatch`, `run_burst.sbatch` — scratch venv via `PATH` (`/central/scratch/jfaber/envs/flits-joint`)
- HPCC jobs **64618456** + **64618479–64618501** — 24/24 COMPLETED

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
- verify-gate recorded on sbatch, grill decisions, merge resolution

### Separate lanes (preserve)

| Lane | Location |
|------|----------|
| Codetection | `feat/codetection-freya` |
| Joint-model figures | `?? jointmodel_figs/`, `jointmodel_montage.svg` |
| Literature | `?? docs/literature/` |
| Faber2026 prose | `CONTEXT.md` dirty on Faber2026 `main` |

## History: Phase 2 first attempt (infra only)

2026-06-26 first submit (64618351–64618374) failed immediately: `/home/jfaber/flits/venv` missing (tree quarantined 2026-06-25). Fixed via quarantine symlinks + sbatch `PATH` patch; resubmitted successfully same night.

## Action items

1. [ ] **Merge [PR #74](https://github.com/jakobtfaber/dsa110-FLITS/pull/74)** — pipeline → `main` (@human review)
2. [ ] **Phase 3 Faber2026** — branch `draft/pass-2-closeout`: bump submodule to merged SHA; 8-row energies; drop zach from `tab:alpha`; update abstract (fixed-s² done — swap pending clause to locked component language); qualitative α stats
3. [ ] **Stacked Faber2026 PR** — ask-matt flow after submodule pin

**Recommended next skill:** `ai-research-workflows:implementing-plans` for Phase 3 manuscript updates.

## Other notes

- Abstract pending clause (grill #7): was "fixed-s² pending for four sightlines" — **now adjudicated**; Phase 3 prose should reflect verdicts above, not HPCC blocker language.
- Packaging order unchanged: pipeline PR first, Faber2026 stacked on submodule pin.

---

**Handoff refreshed 2026-06-27**
