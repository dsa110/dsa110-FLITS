---
description: Validate that an implementation was correctly executed against its plan (FLITS fit-validation gates + plan criteria)
user-invocable: true
---

Use the `validating-implementations` skill (vendored at `.claude/skills/validating-implementations/SKILL.md`) to handle this request.

For FLITS, "validate" also means running the mandatory PASS/MARGINAL/FAIL fit gates from `.cursor/rules/AGENT_CONFIGURATION_FLITS.md` (Level 1: convergence + physical bounds + Jacobian cond; Level 2: chi2_red / R2 / residuals; Level 3: tau*dnu in [0.1, 2.0], alpha near 4.0 = Kolmogorov). A numeric PASS without figure review is not a pass — clear the `.claude/hooks/figure-review-gate.sh` stop gate too.

Arguments (topic, file references, or instructions): $ARGUMENTS

If no arguments were provided, enter the skill's Collaborative mode and ask what is needed before proceeding.
