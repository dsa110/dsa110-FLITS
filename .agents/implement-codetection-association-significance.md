# Implementation Summary: CHIME–DSA co-detection association significance (pillars 1–4)

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Status:** Complete (automated verification); manual verification pending
**Plan Reference:** [plan-codetection-association-significance.md](plan-codetection-association-significance.md)

---

## Overview

Implemented the four-pillar association-significance apparatus as a new pure-function module
`crossmatching/association.py`, assembled into a new `crossmatching/association_report.json`. The
golden `crossmatching/toa_crossmatch_results.json` is untouched (asserted by test).

**Final Status:** ✅ Complete (all 5 phases, automated-verified). Manual verification awaits the user.

## Plan Adherence

**Deviations from Plan:**
- **Test style:** the plan's snippets used `__import__("pytest").approx(...)` to be self-contained
  per snippet; the real `tests/test_association.py` uses a normal top-level `import pytest`.
  *Reason:* cleaner, identical behaviour. *Impact:* none.
- **Lint touch-ups not in the plan snippets:** added `strict=True` to `zip()` in `residual_pedestal`
  (ruff B905) and let ruff sort imports. *Reason:* repo lint gate. *Impact:* none (semantics identical;
  `strict=True` is a real safety check — residuals/errors are paired same-length).
- **Committed the generated `association_report.json`.** *Reason:* it is the visible deliverable and a
  sibling of the committed golden artifact; output is deterministic (pure math) so it will not churn.
  *Impact:* none.

Otherwise the implementation followed the plan exactly (same function names, signatures, and bodies).

## Phases Completed

### Phase 1: Pillar 1 — analytic chance-coincidence probability — ✅ `7d0550c`
`f_dm`, `chance_mu`, `chance_probability`, `expected_chance_associations` + cited constants. Pinned to
the experiment (`chance_mu(500)=5.023345e-9`) and cross-checked against a seeded background MC.

### Phase 2: Pillar 2 — independent DM agreement — ✅ `e9cc07c`
`dm_agreement` (n_sigma + consistent flag; explicit null+reason when CHIME DM absent).

### Phase 3: Pillar 3 — timing budget + residual pedestal — ✅ `25ae6c9`
`timing_budget_ms` (full quadrature) and `residual_pedestal` (inverse-variance-weighted mean residual
significance — tests the +2.4 ms pedestal).

### Phase 4: Pillar 4 — positional coincidence — ✅ `6bbbe87`
`omega_disk_deg2` and `position_consistent` (DSA arcsec position vs CHIME disk; astropy lazy-imported).

### Phase 5: Assemble report — ✅ `55daff2`
`build_association_report` + `main`; `python -m crossmatching.association` writes
`association_report.json` (golden untouched).

## Files Modified

**Created:**
- `crossmatching/association.py` — the four pillars + assembler (pure functions, documented constants).
- `tests/test_association.py` — 12 tests (regression pin, analytic↔MC, DM/timing/position, report+golden-untouched).
- `crossmatching/association_report.json` — generated report (12 bursts).

**Modified:**
- `docs/codetection-science-plan.md:30` — `crossmatching/` row updated from "Stub / aspirational".
- `.agents/plan-codetection-association-significance.md` — Status → Implemented; fixed stale
  Petroff→Foster citation (consistency with the corrected research doc).

**Deleted:** none.

## Key Changes Summary

1. **Chance-coincidence estimator (pillar 1)** — `crossmatching/association.py:36-62`. The decisive
   statistic; per-burst P ~ few×10⁻⁹, Σμ = 5.46×10⁻⁸.
2. **DM / timing / position pillars (2–4)** — `crossmatching/association.py:65-120`. Pillars 2/4 return
   explicit null+reason (CHIME-side data not yet sourced), per scope.
3. **Assembler + artifact** — `crossmatching/association.py:123-175`; `association_report.json`.

## Verification Results

### Automated Verification
- ✅ `pytest tests/test_association.py -q` → **12 passed**.
- ✅ `pytest tests/test_crossmatching_notebook_reproduction.py -q` → **3 passed** (golden intact).
- ✅ `ruff check crossmatching/association.py tests/test_association.py` → clean; `ruff format --check` clean.
- ✅ `python -m crossmatching.association` → exits 0, prints `sum_mu=5.460e-08`, writes the report.
- ✅ `git status --porcelain crossmatching/toa_crossmatch_results.json` → empty (golden untouched).
- ✅ report invariants: 12 bursts, all `chance_coincidence_P < 1e-3`, `expected_chance_associations < 1e-3`.

```
15 passed in 4.29s        # association (12) + reproduction (3)
wrote .../association_report.json  (sum_mu=5.460e-08)
```

### Manual Verification (pending user)
- [ ] Per-burst P (~1e-9) and Σμ (~5e-8) match the experiment's order of magnitude.
- [ ] `inputs` block records the conservative (chance-maximising) windows/DM model.
- [ ] `dm_agreement`/`position_consistent` are explicit null+reason, not fabricated.

## Issues Encountered
- **Autoformatter import handling:** added `import json`/`Path` in the same change as their consumers
  (and `astropy` lazy inside `position_consistent`) to avoid the post-edit formatter stripping them.
  Resolved; no NameErrors.

## Testing Summary
**Tests added (12, `tests/test_association.py`):** chance regression + linear scaling + analytic↔MC
+ Σμ sum (P1); DM agreement consistent/inconsistent/null (P2); timing quadrature + pedestal
significance (P3); disk area + position in/out (P4); report-for-all-12 + golden-untouched (P5).
**All passing:** ✅ yes.

## Documentation Updated
- ✅ Docstrings on every `association.py` function (constants' provenance inline).
- ✅ `docs/codetection-science-plan.md` §A `crossmatching/` row.
- ✅ Plan Status + citation fix.

## Remaining Work
- [ ] Manual verification (above) by the user.
- [ ] (Scoped-out, future) source real CHIME independent DMs + localization to activate pillars 2/4
      beyond null+reason.

## Next Steps
1. Hand off to `ai-research-workflows:validating-implementations`.
2. Open PR → review → merge.

## References
**Plan:** [plan-codetection-association-significance.md](plan-codetection-association-significance.md)
**Research:** [research-codetection-validation-rigor.md](research-codetection-validation-rigor.md)
**Experiment:** [experiment-chance-coincidence-falsealarm.md](experiment-chance-coincidence-falsealarm.md)
**Commits:** `7d0550c` (P1), `e9cc07c` (P2), `25ae6c9` (P3), `6bbbe87` (P4), `55daff2` (P5).

---

**Implementation completed by AI Assistant on 2026-06-23**
