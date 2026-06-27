---
description: Make research/scientific code robust with correctness, regression, and stability checks
user-invocable: true
---

Use the `hardening-research-code` skill (vendored at `.claude/skills/hardening-research-code/SKILL.md`) to handle this request.

For FLITS, the canonical physics kernel is `scattering/scat_analysis/burstfit.py` — `flits/` wraps it. Hardening belongs at the kernel boundary; do not fork physics into the wrapper. Prefer analytic limits (thin-screen tau*dnu ≈ 0.159, Kolmogorov alpha ≈ 4.0) over regression-pinning where derivable.

Target code / module: $ARGUMENTS

If no target was provided, ask which code to harden.
