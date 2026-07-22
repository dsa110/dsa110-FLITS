# Validation Complete

> Validated against the linked Faber2026 plan and
> `docs/rse/specs/implement-g1a-july-morphology-parity.md` on 2026-07-22.

## Overall status: Ready for focused review

- Scope: G1(a) command construction fully implemented.
- Automated checks: 31 focused tests passing; lint and format passing.
- Known failures: three pre-existing missing-fit-artifact failures.
- Critical issues: none.
- Production science products: none created or certified.

## Plan adherence

The accepted July roster, component counts, fixed per-band dispersion-measure
residuals, and alpha=4 constraint match the source CSV. The only deviation is
required by the newer accepted beta-coherent model: removed per-band
pulse-broadening-function flags are not reintroduced.

## Verification results

### Passing

- 31 focused roster, parser, command, rejection, and dry-run tests.
- Exact accepted-roster agreement with the adjudication contract.
- Roster-drift negative test raises `ValueError`.
- Focused tests and dry run reproduce in a clean cloned Conda environment with
  this worktree installed.
- `ruff check` passes.
- `ruff format --check` passes.
- Dry run emits seven accepted sightlines and no rejected sightlines.
- `git diff --check` passes.

### Pre-existing failures

`pytest galaxies/foreground/test_tau_consistency.py -q` reports 36 passed and
three failures. All three require fit JSON files absent from this clean checkout:

- `test_load_allexp_joint_tau_for_budget_casey`
- `test_load_allexp_joint_tau_johndoeii_promoted_c2d2`
- `test_load_joint_free_alpha_johndoeii_uses_promoted_c2d2`

The same failures were reproduced on current `origin/main`; this branch does not
change them. The broader adjudication test likewise cannot read its absent
`results/fit_summaries` artifacts.

## Manual review

Review the seven-line dry-run roster before any production submission. Expected
variants: whitney C2D3, oran C2D1, isha C2D1, phineas C3D3, freya C1D1,
johndoeii C2D2, and mahi C1D2.

## Recommendation

Merge this focused command-construction change after review. Keep production fit
execution and scientific acceptance fail closed in later phases.

## References

- Implementation: `docs/rse/specs/implement-g1a-july-morphology-parity.md`
- Decision: `docs/adr/0006-beta-coherent-scattering-comodel.md`
