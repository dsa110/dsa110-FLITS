# Implementation Plan: Scattering Refit Validation & Reconciliation (Thread 1)

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Status:** Draft
**Related Documents:**
- Memory: `chimedsa-tns-corrections` (TNS designations; johndoeii = FRB 20230814B)

---

## Overview

The `analysis/scattering-refit-2026-06/` campaign produced joint CHIME–DSA
nested-sampling scattering fits for 11 of the 12 co-detected sightlines
(`joint_json/{burst}_joint_fit.json` + `{burst}_joint_ppc.json`), but **none
carries a persisted PASS/MARGINAL/FAIL flag**, and the manuscript
(`Faber2026/sections/results.tex`) still claims "only three sightlines have an
attempted DSA scattering fit, and all three fail" — which is now stale. Several
fits rail at the α prior edge (whitney 1.46, oran 1.44, johndoeii 1.37 below the
1.5 physical floor; freya/chromatica 6.00, hamilton 5.99 at the ceiling).

This plan (1) gates every committed joint fit against the **runtime**
`classify_fit_quality` contract, (2) adversarially re-verifies the verdicts with
a separate judge, (3) confirms the diagnostic figures, (4) re-fits the railed
sightlines with the physical α floor to test whether the rails are a prior
artifact or a genuine non-detection, and (5) reconciles the manuscript narrative
(results + observations + budget + discussion) to the real verdict table.

**Goal:** A reproducible, adversarially-verified PASS/MARGINAL/FAIL verdict for
all 12 sightlines, and a manuscript whose scattering claims match it.

**Motivation:** The paper's core deliverable is the scattering measurement
(shared α via the CHIME–DSA lever arm). Right now there is no gate-validated
fit and the prose contradicts the committed artifacts. This is the blocking
measurement for the paper.

## Current State Analysis

**Existing Implementation:**
- `scattering/scat_analysis/burstfit.py:1362` — `classify_fit_quality(chi2_reduced, r_squared=None, normality_pvalue=None) -> (flag, notes)`. The **authoritative** Level-2 gate: PASS `0.3 ≤ χ²_red ≤ 1.5`, MARGINAL `1.5 < χ²_red ≤ 10` or `< 0.3`, FAIL `> 10` or non-finite. R² and normality are informational only (do not flip the flag).
- `.claude/workflows/fit-verify.js` — adversarial verifier (one judge agent per `*_fit_results.json`, told to REFUTE the PASS claim against the runtime cut points). `GATE_CONTRACT` in it is the verbatim authoritative rubric and states the doc / `VALIDATION_THRESHOLDS.py` "PASS up to 3.0" is **dead** (runtime PASS ceiling is 1.5).
- `analysis/scattering-refit-2026-06/run_joint_fit.py:164` — joint-fit driver; `alpha_bounds=(a.alpha_lo, a.alpha_hi)` (CLI `--alpha-lo/--alpha-hi`); writes `{burst}_joint_fit.json` (`:194`).
- `analysis/scattering-refit-2026-06/joint_ppc.py` — writes `{burst}_joint_ppc.json` (`chi2_chime`, `chi2_dsa`).
- `analysis/scattering-refit-2026-06/gate_summary.py` — prior art, but reads HPC scratch `$FLITS_RUNS` and gropes for goodness fields that live in the *paired* `_ppc.json`, so it returns `None` for χ²/verdict on the committed artifacts.
- `analysis/scattering-refit-2026-06/build_joint_deck.py:26` — `ALO, AHI = 1.0, 6.0`: the committed campaign's α prior floor (1.0) is **below** the 1.5 physical gate → the source of the sub-1.5 rails.
- `analysis/scattering-refit-2026-06/inject_recovery.py:53` — already uses `alpha_bounds=(1.5, 6.0)`; `burstfit_joint.py:46` default `(2.0, 6.0)`.
- `analysis/scattering-refit-2026-06/dsa_figs/{burst}_fitquality.png` (+ `dsa_fitquality_montage.png`) — per-sightline DSA fit-quality figures.

**Current Behavior:** `{burst}_joint_fit.json` stores `alpha` (median ± err),
`tau_1ghz`, `log_evidence`, `alpha_bounds`, `percentiles`, `ncall`. `{burst}_joint_ppc.json`
stores `chi2_chime`, `chi2_dsa`. No flag is computed or persisted; the manuscript
narrative is hand-written and stale.

**Current Limitations:**
- No PASS/MARGINAL/FAIL flag on any committed fit.
- α rails are a prior-edge artifact of the 1.0 floor; their scientific meaning (non-detection vs mis-specified model) is uncharacterized.
- `results.tex` (+ observations/budget/discussion) contradict the 11/12-fit reality.

## Desired End State

**New Behavior:**
- A committed verdict table (`analysis/scattering-refit-2026-06/joint_gate_verdicts.{csv,md}`) with per-sightline α (rail-flagged), τ, χ²_C, χ²_D, Level-1/2/3 outcomes, and a FINAL flag + reason, computed by reusing `classify_fit_quality`.
- A separate-judge adversarial confirmation of each FINAL flag (fit-verify aggregate).
- Figure-review verdicts for the `dsa_figs/*_fitquality.png` set.
- Re-fits of the railed sightlines with the 1.5 α floor, gated, showing whether α relocates (recovered measurement) or stays railed (confirmed non-detection).
- `Faber2026` results/observations/budget/discussion updated to the verdict table.

**Success Looks Like:**
- `pytest analysis/scattering-refit-2026-06/test_gate_joint_committed.py` passes.
- The gate, run over `joint_json/`, classifies all 11 fits; the verdict table exists and matches the adversarial aggregate.
- Re-fit JSONs for every railed sightline exist and are gated.
- The manuscript no longer states "only three sightlines have an attempted fit."

## What We're NOT Doing

- [ ] Re-deriving or modifying the joint fitter (`burstfit_joint.py`) physics.
- [ ] Re-running the full campaign (only the railed sightlines are re-fit).
- [ ] Touching the scintillation census or DM-budget computation.
- [ ] Gating the legacy per-burst `*_fit_results.json` (superseded by `joint_json/`).
- [ ] Adding a τ×Δν Level-3 check (no per-sightline scintillation bandwidth is available; report N/A).

**Rationale:** The fits exist; the work is validation + reconciliation, not
re-analysis. Shortest diff that produces a defensible verdict (ponytail).

## Implementation Approach

Reuse the runtime classifier and existing workflow/agents; add only a thin gate
that joins `_joint_fit.json` + `_joint_ppc.json`. Combine the two bands worst-of
for the FINAL flag. Re-fits reuse `run_joint_fit.py` with `--alpha-lo 1.5`. Each
phase is test-first where it adds logic. All FLITS commits go through a worktree
off `origin/main` (the main checkout is on the `feature/cluster-catalog-engine`
separate lane — do not disturb it); Faber2026 edits likewise via a worktree.

---

## Implementation Phases

### Phase 1 — Deterministic gate over the committed joint fits

**Objective:** Produce a per-sightline verdict table from the committed
artifacts, reusing `classify_fit_quality`.

Tasks (test-first):

1. Write the failing test `analysis/scattering-refit-2026-06/test_gate_joint_committed.py`:
   ```python
   from gate_joint_committed import gate_one  # (burst, fit_dict, ppc_dict) -> dict

   def _fit(alpha, tau=0.3, bounds=(1.0, 6.0)):
       return {"alpha": {"median": alpha}, "tau_1ghz": {"median": tau},
               "alpha_bounds": list(bounds)}

   def test_alpha_below_floor_fails_level1():
       v = gate_one("x", _fit(1.4), {"chi2_chime": 1.0, "chi2_dsa": 1.0})
       assert v["final"] == "FAIL" and "alpha" in v["reason"].lower()

   def test_alpha_at_ceiling_fails_level1():
       assert gate_one("x", _fit(6.0), {"chi2_chime": 1.0, "chi2_dsa": 1.0})["final"] == "FAIL"

   def test_physical_alpha_good_chi2_passes():
       v = gate_one("x", _fit(4.0), {"chi2_chime": 1.0, "chi2_dsa": 1.2})
       assert v["final"] == "PASS"

   def test_catastrophic_chi2_fails():
       assert gate_one("x", _fit(4.0), {"chi2_chime": 1.0, "chi2_dsa": 12.0})["final"] == "FAIL"

   def test_rail_flagged_within_edge_of_bound():
       v = gate_one("x", _fit(1.46), {"chi2_chime": 1.1, "chi2_dsa": 1.1})
       assert v["rail"] is True
   ```
2. Run it, watch it fail (no module yet).
3. Implement `analysis/scattering-refit-2026-06/gate_joint_committed.py`:
   - `from scattering.scat_analysis.burstfit import classify_fit_quality` (reuse — do NOT re-implement χ² cut points).
   - `gate_one(burst, fit, ppc)`:
     - α = `fit["alpha"]["median"]`; τ = `fit["tau_1ghz"]["median"]`; bounds = `fit["alpha_bounds"]`.
     - **Level 1:** FAIL unless `1.5 < α < 6.0` and `1e-4 < τ < 100`.
     - **rail flag:** `min(α - bounds[0], bounds[1] - α) < 0.1` (within EDGE of a prior bound).
     - **Level 2:** `flag_C,_ = classify_fit_quality(ppc["chi2_chime"])`; `flag_D,_ = classify_fit_quality(ppc["chi2_dsa"])`; band combine = worst of the two.
     - **Level 3 (α-physics):** FAIL if `α < 2.0 or α > 6.0`; MARGINAL if not `3.5 ≤ α ≤ 4.5`; else PASS-consistent. (τ×Δν: N/A — record `"tau_dnu": "N/A (no dnu_d)"`.)
     - FINAL = FAIL if any level FAILs; else MARGINAL if any MARGINAL; else PASS. `reason` = first failing/limiting clause.
   - `main()`: glob `joint_json/*_joint_fit.json`, pair each with `*_joint_ppc.json` (None if absent → Level-2 unknown → MARGINAL with note), write `joint_gate_verdicts.csv` + `.md`, and one `joint_json/{burst}_joint_gate.json` per burst in a `*_fit_results.json`-shaped dict (`{"burst","chi2_reduced","tau","alpha","quality_flag","notes"}`) for Phase 2.
4. Run the test, watch it pass.
5. Run `main()`; eyeball the table against the known hand-scan (whitney/oran/johndoeii FAIL L1; freya/chromatica/hamilton FAIL L1; wilhelm/zach/phineas the only L1 survivors).

**Verification:**
- Automated: `pytest analysis/scattering-refit-2026-06/test_gate_joint_committed.py -q` → all pass; `joint_gate_verdicts.csv` exists with 11 rows.
- Manual: the FINAL flags match the documented α/χ² hand-scan.

### Phase 2 — Adversarial re-verification (separate judge)

**Objective:** A judge that did not produce the fits confirms each FINAL flag
(Boris: adversarial verification; CLAUDE.md: use the workflow for a campaign).

Tasks:
1. Phase 1 already emits `joint_json/{burst}_joint_gate.json` in the
   `*_fit_results.json` shape, so `.claude/workflows/fit-verify.js` runs unmodified.
2. Run the workflow over `analysis/scattering-refit-2026-06/joint_json/*_joint_gate.json`
   (set its `TARGET` glob or pass the dir). Each verifier attempts to REFUTE the flag
   against `GATE_CONTRACT`.
3. Capture the aggregate (confirmed-PASS / MARGINAL / FAIL + per-fit reasons) to
   `analysis/scattering-refit-2026-06/joint_gate_adversarial.md`.

**Verification:**
- Automated: workflow completes; aggregate file written for all 11.
- Manual: any verdict the judge flips from Phase 1 is investigated and the gate reconciled.

### Phase 3 — Figure review

**Objective:** Satisfy the runtime figure-review gate — a numeric PASS needs
visually-assessed diagnostics.

Tasks:
1. Dispatch the `figure-reviewer` agent on `analysis/scattering-refit-2026-06/dsa_figs/*_fitquality.png` (+ `dsa_fitquality_montage.png`), comparing each to its sightline's verdict.
2. Write per-figure verdicts (`match` / `anomaly` / `skipped:<why>`) to `analysis/scattering-refit-2026-06/dsa_figs/figures.review.json`.

**Verification:**
- Manual: every L1-surviving sightline (wilhelm/zach/phineas + any re-fit recoveries) has a reviewed figure with no unexplained anomaly.

### Phase 4 — α-rail re-fits (physical floor)

**Objective:** Test whether the rails are a prior artifact or a genuine
non-detection by re-fitting with the 1.5 floor.

Tasks:
1. Confirm inputs exist for the railed sightlines (whitney, oran, johndoeii, freya, chromatica, hamilton): the per-burst dynamic spectra (`.npy`) + joint configs (`gen_dsa_configs.py` / `check_joint_configs.py`). If only on HPC scratch (`$FLITS_RUNS`), run there. **ABORT this phase and report** if inputs are absent — do not fabricate.
2. For each, re-run `python analysis/scattering-refit-2026-06/run_joint_fit.py --alpha-lo 1.5 --alpha-hi 6.0 …` (same other args as the committed run; see its argparse) → `*_joint_fit.json` + `joint_ppc.py` → `*_joint_ppc.json`, written under a `refit_alpha15/` subdir (do not overwrite the [1.0,6.0] artifacts).
3. Gate the re-fits with Phase 1's `gate_joint_committed.py` pointed at `refit_alpha15/`.
4. Compare per sightline: does α relocate off 1.5 with PASS χ² (recovered), or stay pinned at 1.5 with comparable χ² and log_evidence (confirmed non-detection)?

**Verification:**
- Automated: a `*_joint_fit.json` exists under `refit_alpha15/` for every railed sightline and is gated into `joint_gate_verdicts_refit.csv`.
- Manual: the relocate-vs-rail interpretation per sightline is recorded with its log_evidence delta.

### Phase 5 — Reconcile the manuscript (broader narrative pass)

**Objective:** Make the Faber2026 scattering narrative match the verdict table.

Tasks (in a `Faber2026` worktree off `origin/main`):
1. Rewrite `sections/results.tex` scattering paragraph + Fig. caption to the real counts (N PASS / M MARGINAL / K FAIL of 12; the α rails interpreted per Phase 4) — replacing "only three sightlines have an attempted DSA scattering fit, and all three fail."
2. Sweep `sections/observations.tex`, `sections/budget.tex`, `sections/discussion.tex` for sentences that lean on the old scattering claim; update to the verdict table.
3. PR to `Faber2026` main (propagates to Overleaf).

**Verification:**
- Automated: `rg "only three sightlines" Faber2026/sections/` returns nothing.
- Manual: each edited paragraph reads correctly and cites the verdict table's numbers.

---

## Success Criteria

### Automated Verification
- [ ] `pytest analysis/scattering-refit-2026-06/test_gate_joint_committed.py -q` passes.
- [ ] `gate_joint_committed.py` run emits `joint_gate_verdicts.csv` (11 rows) + per-burst `_joint_gate.json`.
- [ ] `fit-verify` workflow completes and writes the adversarial aggregate for all 11.
- [ ] `refit_alpha15/` contains a gated `*_joint_fit.json` for each railed sightline (or the phase is reported ABORTED with the missing-input reason).
- [ ] `rg "only three sightlines" Faber2026/sections/` → no matches.

### Manual Verification
- [ ] Phase-1 FINAL flags match the α/χ² hand-scan; any adversarial flip reconciled.
- [ ] `figure-reviewer` verdicts written; no unexplained anomaly on a PASS sightline.
- [ ] Re-fit relocate-vs-rail interpretation recorded per sightline.
- [ ] Manuscript scattering narrative reads correctly against the verdict table.

## Testing Strategy

- **Unit:** `test_gate_joint_committed.py` — Level-1 bounds, rail flag, Level-2 via `classify_fit_quality`, Level-3 α-physics, worst-of-band combine (Phase 1 task 1).
- **Integration:** run the gate over the real `joint_json/`; cross-check against the `fit-verify` adversarial aggregate (Phase 2).
- **Manual:** figure review (Phase 3), re-fit interpretation (Phase 4), manuscript read (Phase 5).

## References
- `scattering/scat_analysis/burstfit.py:1362` — `classify_fit_quality` (authoritative Level-2 gate).
- `.claude/workflows/fit-verify.js` — adversarial verifier + verbatim `GATE_CONTRACT`.
- `analysis/scattering-refit-2026-06/run_joint_fit.py:164`, `joint_ppc.py`, `build_joint_deck.py:26`, `gate_summary.py`, `inject_recovery.py:53`.
- `.cursor/rules/AGENT_CONFIGURATION_FLITS.md` — the three-level contract (Level-2 cut points superseded by the runtime classifier).
