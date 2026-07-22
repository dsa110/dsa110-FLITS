# Implementation Summary: G1(a) July morphology parity

**Date:** 2026-07-22

**Status:** Complete for G1(a) command construction

**Plan:** [Faber2026 manuscript science gates](https://github.com/jakobtfaber/Faber2026/blob/921928ea150b9ae2a441063578ca84ac0ea10611/docs/rse/specs/plan-manuscript-science-gates-2026-07-15.md)

## Scope

Port the recoverable G1(a) work onto FLITS `origin/main` without importing the
stale parent-repository or results-library changes from the abandoned worktree.
This change constructs fixed-alpha refit commands only. It does not run fits,
write production fit products, or promote manuscript claims.

## Implementation

- Load the seven `accepted_physical` rows from the frozen July adjudication CSV.
- Fail closed if the accepted nickname/variant roster changes.
- Parse each `CxDy` variant into explicit CHIME/FRB and DSA-110 component counts.
- Pass the recorded per-band fixed dispersion-measure residuals to the joint fitter.
- Fix alpha to 4 through the current beta-coherent runner interface.
- Name outputs with the accepted component variant and retain legacy-read fallback.
- Add a seven-sightline dry run and contract tests for accepted and rejected inputs.

## Plan deviation

The 2026-07-15 plan described explicit per-band pulse-broadening-function flags.
Those flags no longer exist: accepted ADR-0006 couples kernel shape to sampled
beta and removed `--pbf-C`/`--pbf-D`. The port therefore uses the current runner
contract and tests that the obsolete flags are absent.

## Provenance

- FLITS base: `ab6af1f713496abd2ff2d71bf11edf4100871e94`
- Recovered implementation: `11403e6ca361c990a3937b4910f94b69f97241ff`
- Parent plan snapshot: `921928ea150b9ae2a441063578ca84ac0ea10611`
- Adjudication CSV SHA-256: `7ed91f3eab8f09f9f414254b138ae2baceecb7d92c0f89437760a366eb97877d`
- Runtime: clean clone of Conda environment `flits`, Python 3.12.13, with this
  worktree installed editable and without dependency resolution. No stochastic
  fit was executed.

## Verification

```bash
conda run -n flits pytest \
  analysis/scattering-dm-locked-2026-07-14/test_fit_adjudication.py::test_pbf_roster_is_exactly_the_physically_accepted_subset \
  galaxies/foreground/test_tau_consistency.py -q \
  -k 'pbf_roster or july or parse_cxdy or build_alpha4 or run_burst_command or non_accepted or dry_run'

conda run -n flits python -m galaxies.foreground.run_tau_consistency_refits --dry-run

conda run -n flits ruff check \
  galaxies/foreground/run_tau_consistency_refits.py \
  galaxies/foreground/tau_consistency.py \
  galaxies/foreground/test_tau_consistency.py
```

Results: 31 focused tests passed; lint and format checks passed; dry run emitted
exactly the seven accepted sightlines. The 31 tests and dry run were reproduced
in the clean cloned environment. The full tau-consistency test file has three
missing-fit-artifact failures also present on the base commit.

## Remaining work

Production fit execution, fit-quality review, and any manuscript promotion are
separate science-gate phases and remain unstarted.
