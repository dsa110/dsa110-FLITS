# Handoff: Pass 2 manuscript closeout — Phase 1 done, Phase 2 blocked on HPCC venv

---
**Date:** 2026-06-26 23:43 (PDT)
**Author:** AI Assistant
**Status:** Phase 2 complete — Phase 3 ready
**Branch:** pipeline `manuscript-pass-2-2026-06` @ `5a0ba9b`; Faber2026 `main` @ `5c87f6c` (submodule not pinned)
**Commit:** see branches above — substantial uncommitted work on pipeline branch

---

## Task(s)

Pass 2 manuscript authority closeout per `/grill-with-docs` (2026-06-27 decisions). Fork B (manuscript) merged via PR #11; Pass 2 drives pipeline energy re-export, fixed-s² HPCC campaign, then Faber2026 table/prose updates.

| Task | Status | Notes |
|------|--------|-------|
| Lock Pass 2 grill decisions (7 items) | ✅ Complete | `.agents/pass-2-grill-decisions.md`, `Faber2026/CONTEXT.md` (uncommitted) |
| Phase 1: HPCC all-exp pull + gate regen + energy re-export | ✅ Complete | N=8 energies; gate 11/11 MARGINAL; artifacts uncommitted |
| Phase 2: fixed-s² all-exp HPCC campaign (24 jobs) | ✅ Complete | 24/24 COMPLETED after venv fix; s² adjudication run |
| Phase 3: Faber2026 `draft/pass-2-closeout` manuscript updates | 🔄 Next | s² verdicts locked; ready to implement |
| Pipeline PR → Faber2026 stacked PR (ask-matt) | 🔄 In progress | Pass 2 artifacts committing; PR to `main` next |

**Current Workflow Phase:** Experiment (s² campaign) → blocked; next is fix infra → re-run → Validate → Implement (Phase 3).

## Workflow Artifacts

**Plan / decision docs:**
- `.agents/pass-2-grill-decisions.md` — locked Pass 2 decisions + Phase 1/2 status (uncommitted)
- `docs/rse/specs/decision-map-manuscript-completion.md` — upstream ADR map (ADR-0003/0004)
- `docs/rse/specs/plan-manuscript-completion.md` — manuscript completion plan
- `Faber2026/CONTEXT.md` — manuscript authority + Pass 2 roster (modified, uncommitted)

**Prior handoff (context):**
- `docs/rse/specs/handoff-2026-06-24-07-52-manuscript-figures-landed.md` — figure pipeline landed (separate lane)

**Submit recipe:**
- `analysis/scattering-refit-2026-06/hpcc/submit_pass2_s2_allexp.sh` — 24-job s² grid (uncommitted; also copied to HPCC `flits-runs/`)

## Critical References

- `.agents/pass-2-grill-decisions.md` — **read first** — locked decisions, Phase 1 results, Phase 2 job ids
- `analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py` — s² adjudication (fail-closed to all-exp PBF only)
- `Faber2026/CONTEXT.md` — manuscript language, Pass 2 roster, abstract pending clause swap

## Recent Changes

**Pipeline branch `manuscript-pass-2-2026-06` (mostly uncommitted):**
- `analysis/scattering-refit-2026-06/_hpcc_pull/joint/` — 33 all-exp JSONs rsync'd from HPCC (scattering-only; no c₀,γ)
- `analysis/burst_energies/burst_energies.provenance.json` — git_sha bumped to `5a0ba9b`, n_bursts=8
- `analysis/scattering-refit-2026-06/hpcc/submit_pass2_s2_allexp.sh` — new submit script (sha256 verified this session)
- Gate regen on branch: **11/11 MARGINAL, 0 L1 FAIL** (matches `manuscript-closeout-2026-06` committed state)

**Faber2026 `main` (uncommitted):**
- `CONTEXT.md:45-50` — Pass 2 locked roster (oran/whitney energies, drop zach from tab:alpha, qualitative α stats)
- `pipeline` submodule pointer — checked out to `manuscript-pass-2-2026-06`, not committed on main

**Merged earlier this arc (on main):**
- PR #11 manuscript authority closeout @ `db544ad` (Fork B tables/prose, pipeline pin `5a0ba9b` closeout branch)
- `.gitignore`: `figures/prototypes/` @ `5c87f6c`

## Reproducibility & Data State

- **Compute host:** HPCC (`ssh hpcc`). Joint fits via `/central/scratch/jfaber/flits-runs/run_joint.sbatch` + `run_joint_fit.py`.
- **HPCC data paths:**
  - All-exp joint JSONs: `/central/scratch/jfaber/flits-runs/data/joint/*_pbf-exp-exp.json`
  - Local mirror: `analysis/scattering-refit-2026-06/_hpcc_pull/joint/` (33 files)
  - s² outputs (target): `{burst}_joint_fit_CxDy_s2-{1|10|100}_pbf-exp-exp.json` → rsync to `joint_ladder/`
- **Energy c₀,γ authority:** mixed-legacy `joint_json/` (alpha-independent per ADR); all-exp HPCC fits are scattering-only (0/19 have c₀,γ on cluster)
- **Energy roster (Phase 1 re-export):** N=8 — chromatica, hamilton, isha, **oran**, phineas, **whitney**, wilhelm, zach; `--check` PASS
- **zach s² (pre-existing, local):** C2D3 **NOT robust** (ΔlnZ +1443 / −759 / −0.4 sign flips) — supports grill decision to drop zach from `tab:alpha`
- **Legacy mixed-PBF s² grids on HPCC:** exist for oran/isha/mahi/phineas — **must not use** (`_s2verdict.py` fail-closed)

### In-flight / failed jobs (Phase 2)

- **Submitted:** 2026-06-26 22:40 PDT via `submit_pass2_s2_allexp.sh`
- **Job ids:** 64618351–64618374 (24 jobs); log: `/central/scratch/jfaber/flits-runs/logs/pass2_s2_allexp_submit.20260626T224013.ids`
- **Queue:** empty (all finished immediately)
- **Result:** **24/24 FAILED** — root cause identical across jobs:
  ```
  /var/spool/slurmd/job64618351/slurm_script: line 15: /home/jfaber/flits/venv/bin/activate: No such file or directory
  ```
- **Existing s² outputs on cluster:** 6 files, **zach only** (from Jun 24 run, before venv broke)
- **Fix required before resubmit:** restore or repoint venv in `run_joint.sbatch:15` (and `run_burst.sbatch:24`). Documented path `/home/jfaber/flits/venv` in `JOINT_FIT_STATE.md:45` is stale — `/home/jfaber/flits/` does not exist on cluster today. Last successful joint job: `flits-joint_64416088` (2026-06-19).

## Verification State / Known-Broken

### Known-broken / unverified

- **Phase 2 s² campaign:** 24/24 FAILED — no new oran/isha/mahi/phineas s² outputs. Phase 3 blocked.
- **HPCC Python env:** `/home/jfaber/flits/venv/bin/activate` missing; `/central/scratch/jfaber/envs/wf1` exists but lacks `dynesty`. Must locate or rebuild FLITS venv before any joint re-run.
- **Pipeline branch:** dirty + untracked Phase 1/2 artifacts not committed or pushed.
- **Faber2026:** `CONTEXT.md` + submodule pointer dirty; `.remember/` untracked.

### Passing / verified this session

- Gate regen: 11/11 MARGINAL, 0 FAIL (matches closeout branch committed state)
- Energy re-export: N=8, `--check` PASS
- verify-gate recorded: `pass-2-grill-decisions.md`, `submit_pass2_s2_allexp.sh` (sha256=5308c25d962d), Phase 2 submit log cross-check

### Separate lanes (preserve — do not sweep)

| Lane | Location | Status |
|------|----------|--------|
| Codetection | `feat/codetection-freya` | stash: `lane: manuscript-closeout WIP before codetection-freya` |
| Pipeline extras | `?? jointmodel_figs/`, `jointmodel_montage.svg`, `docs/literature/` | unrelated to Pass 2 |
| Prototypes | `figures/prototypes/` | gitignored on Faber2026 main |

## Learnings

- **All-exp HPCC JSONs are scattering-only** — verified 0/19 have c₀,γ on cluster. Energy table correctly uses mixed-legacy amplitudes until full-amplitude all-exp fits exist.
- **Phase 2 submit script is correct** — failure is infra (venv), not the 24-job recipe. Do not use legacy mixed-PBF s² grids already on HPCC.
- **zach demonstrator:** local s² adjudication already supports dropping from `tab:alpha`; may remain in energies table (grill #6).
- **Sacct retention:** only earliest job id may appear in `sacct` now; use per-job `.err` logs under `flits-runs/logs/` for full failure audit.

## Action Items & Next Steps

1. [ ] **Fix HPCC venv** — locate or rebuild dynesty-capable env; patch `run_joint.sbatch:15` (+ `run_burst.sbatch:24` if needed); smoke-test one joint job before bulk resubmit.
2. [ ] **Resubmit Phase 2** — rerun `submit_pass2_s2_allexp.sh` on HPCC; drain 24/24; confirm outputs in `data/joint/*_s2-*_pbf-exp-exp.json`.
3. [ ] **Rsync + adjudicate** — pull s² JSONs → `joint_ladder/`; run `_s2verdict.py` for oran, isha, mahi, phineas (+ zach/whitney as needed).
4. [ ] **Commit pipeline branch** — `_hpcc_pull/`, provenance, grill decisions, submit script, any regen outputs; open **pipeline PR first**.
5. [ ] **Phase 3 Faber2026** — branch `draft/pass-2-closeout`: bump submodule pin, 8-row energies (oran+whitney), drop zach from `tab:alpha`, swap abstract pending clause, keep α stats qualitative; stacked PR (ask-matt).

**Recommended Next Skill:** `ai-research-workflows:validating-implementations` — after HPCC venv fix, validate Phase 2 drain + s² adjudication; then `ai-research-workflows:implementing-plans` for Phase 3 manuscript updates.

## Other Notes

- Grill locked abstract swap: gate-regen pending → **fixed-s² pending for four sightlines** (oran, isha, mahi, phineas-DSA).
- Pipeline PR packaging order is explicit: pipeline first, Faber2026 stacked on submodule pin.
- SSH alias: `hpcc` (not h17 for this work).

---

**Handoff created by AI Assistant on 2026-06-26**
