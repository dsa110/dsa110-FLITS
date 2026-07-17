# Research: Window-campaign scientific gate closure

**Date:** 2026-07-17  
**Scope:** Internal codebase and local CHIME products  
**Related Documents:** `plan-window-campaign-gate-closure-2026-07-17.md`

## Question / Scope

Determine what evidence PR #192 still needs before merge, and use the repository's
existing scientific acceptance rules rather than creating campaign-specific exceptions.

## Codebase Findings

- `scintillation/scripts/run_window_campaign.py` generated fits and figures but hard-coded
  `artifact_validation_status=not_run` and `figure_review_status=pending`.
- `scintillation/scint_analysis/chime_artifact_guards.py` already defines fail-closed
  off-pulse, low-lag, and minimum-three-subbands verdicts.
- `scintillation/scint_analysis/pipeline.py` demonstrates how to run the off-pulse and
  low-lag controls on the same channels and ACF estimator as the on-pulse result.
- `docs/dev/figure-review-protocol.md` requires a manifest expectation, an actual rendered
  inspection, and a per-figure `match`, `anomaly`, or `skipped` verdict.
- The local checkout contains all 12 standard and all 12 high-resolution CHIME products.
- The earlier scratch injection harness used known Lorentzian truth but only printed results;
  it had no assertions, provenance record, committed artifact, or figure-review entry.

## Synthesis

Gate closure means recording honest pass/fail outcomes, not forcing every burst into
`measurement`. The campaign should call the shared guards, remain diagnostic on any false or
inconclusive check, and promote only records for which injection recovery, artifact controls,
and visual review all pass. Synthetic recovery uses known injected parameters as the independent
truth surface. The real-data controls use the exact fixed subband channels and campaign fitter.

## References / Sources

- `scintillation/scripts/run_window_campaign.py`
- `scintillation/scint_analysis/window_refit.py`
- `scintillation/scint_analysis/chime_artifact_guards.py`
- `scintillation/scint_analysis/pipeline.py`
- `docs/dev/figure-review-protocol.md`

