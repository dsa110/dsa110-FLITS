# Implementation Plan: Window-campaign scientific gate closure

---
**Date:** 2026-07-17  
**Author:** Codex  
**Status:** Complete  
**Related Documents:**
- `research-window-campaign-gate-closure-2026-07-17.md`
---

## Overview

Close PR #192's three remaining scientific gates with reproducible, reviewable evidence.
Use the existing CHIME guard functions, generate all standard and high-resolution campaign
figures from the refreshed branch, inspect every registered image, and promote only real passes.

**Goal:** A committed evidence bundle whose statuses can be recomputed and whose figures have
hash-bound visual verdicts, followed by exact-head PR merge if code and CI remain green.

## Current State Analysis

The campaign fitter and pending manifest exist, but artifact controls and reproducible injection
evidence are absent from its output. Status is therefore correctly fail-closed.

## Desired End State

- Fixed-seed injection recovery asserts known-truth recovery and writes JSON plus a diagnostic.
- Each campaign record contains off-pulse, low-lag, and three-subbands evidence.
- All manifest images have SHA-256-bound visual verdicts.
- A deterministic finalizer promotes only the intersection of all three passes.

## What We're NOT Doing

- Changing scientific thresholds after seeing results.
- Treating an inconclusive control as a pass.
- Rewriting or deleting shared history.

## Implementation Phases

### Phase 1: Control integration

- [x] Add failing tests for fail-closed aggregation and low-lag refits.
- [x] Observe the tests fail because the API does not exist.
- [x] Implement the shared-guard adapter and campaign-fitter excision path.
- [x] Observe the focused tests pass.

### Phase 2: Reproducible injection evidence

- [x] Run 100 trials per truth point with seed 20260717.
- [x] Require median gamma relative error <=10%, isolated recovery >=95%, two-component
  adoption >=95%, false-split rate <=5%, and median modulation-index error <=5%.
- [x] Record Python/package versions, git commit, seed, command, aggregate results, and figure.

These tolerances are predeclared before the repository script is run. They are wider than the
noise-only numerical precision while still rejecting the order-unity envelope bias the 2L model
is intended to fix. The 5% false-split ceiling bounds scientifically material over-selection.

### Phase 3: Real-data controls and visual review

- [x] Regenerate all 24 available products from current code.
- [x] Record the repository off-pulse, low-lag, and minimum-three-subbands verdicts.
- [x] Render and inspect every manifest figure, then write hash-bound verdicts.
- [x] Run the deterministic finalizer and retain diagnostic-only failures.

### Phase 4: Reproduction and merge

- [x] Re-run focused/full tests and the evidence validation in the locked `uv` environment.
- [ ] Run closeout checks, commit, push, wait for exact-head CI, and merge PR #192.

## Success Criteria

- `uv run pytest scintillation/scint_analysis/tests/test_window_campaign.py` passes.
- Injection `gate_status` is `pass` under the predeclared criteria.
- Every required figure has a rendered, hash-matching verdict.
- No record is `measurement` unless all three gates pass.
- PR #192 merges only at the reviewed head SHA.

## Risk Assessment

The main risk is that real-data controls fail or are inconclusive. That is a scientific result,
not a software blocker: those records remain `diagnostic_only`, while the evidence and fail-closed
implementation can still merge.
