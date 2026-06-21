#!/usr/bin/env bash
# PostCompact hook: re-inject the FLITS fit-validation contract so goal-drift
# can't silently drop "don't rationalize fits" after compaction.
# Reads PostCompact stdin (ignored), emits additionalContext JSON on stdout.
cat >/dev/null
# ponytail: self-proving trace — next real compaction proves the hook fired (*.log gitignored)
printf '%s PostCompact fired\n' "$(date -u +%FT%TZ)" >> "${CLAUDE_PROJECT_DIR:-$PWD}/.claude/hooks/postcompact.log"
exec python3 - <<'PY'
import json, sys

ctx = """FLITS FIT-QUALITY CONTRACT (burstfit.py classify_fit_quality, AUTHORITATIVE):
- PASS: 0.3<=chi2_red<=1.5 AND Level-1 gates (converged; 1e-4<tau<100ms; 1.5<alpha<6.0; cov non-singular).
- MARGINAL: 1.5<chi2_red<=10, OR chi2_red<0.3 (noise overestimated).
- FAIL: chi2_red>10 or non-finite, OR any Level-1 gate fails, OR physics fail (tau*dnu outside [0.1,2.0]; alpha outside [2.0,6.0]).
- R2 and residual-normality p are INFORMATIONAL notes ONLY; they never flip the flag (low weighted R2 expected at low S/N).
- Physics refs: tau*dnu thin-screen 0.159, extended 1.0; Kolmogorov alpha 4.0.
- FIGURE-REVIEW GATE (commit 0f4fa17): every fit MUST emit diagnostic figures (data-vs-model, residuals, hist, Q-Q) and they MUST be visually assessed before declaring success. A numeric PASS is NOT sufficient.
- Do NOT rationalize a failing/marginal fit into a pass. Report the flag the contract gives."""

print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PostCompact",
    "additionalContext": ctx,
}}))
PY
