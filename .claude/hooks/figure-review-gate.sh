#!/usr/bin/env bash
# Stop hook: block end-of-turn while generated figures have not been VISUALLY assessed.
#
# Contract: a figure-producing run writes <dir>/figures.manifest.json listing each PNG
# and the expectation it should satisfy. The review step (figure-reviewer subagent, or
# inline) Reads each PNG and writes <dir>/figures.review.json with per-figure verdicts.
# If any manifest is newer than its review (or has no review), the figures were produced
# but not looked at -> block, so "I made a plot" can never silently become "validated".
#
# Pure mtime check (no transcript parsing). No-op when no manifests exist.
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"

unreviewed=()
while IFS= read -r man; do
  [ -n "$man" ] || continue
  rev="$(dirname "$man")/figures.review.json"
  # -nt is true if man is newer than rev OR rev does not exist
  if [ "$man" -nt "$rev" ]; then
    unreviewed+=("$(dirname "$man")")
  fi
done < <(find "$ROOT" -name figures.manifest.json -not -path '*/.git/*' 2>/dev/null)

# Nothing produced-but-unreviewed -> allow stop.
[ "${#unreviewed[@]}" -eq 0 ] && exit 0

dirs="$(printf '%s\n' "${unreviewed[@]}")"
read -r -d '' REASON <<EOF || true
FIGURE-REVIEW GATE: figures were produced but not visually assessed.

Before finishing you MUST actually LOOK at each PNG (Read the image file so it renders)
and compare it to the stated expectation in that dir's figures.manifest.json, the way a
person reviewing a colleague's plots would. Check: axes/units sane, expected features
present (peaks, tails, scales), no artifacts/empty panels/wrong ranges, and that annotated
numbers match the visual.

Then write figures.review.json in the same dir:
  {"reviewed_by": "...", "verdicts": [
     {"path": "<png>", "verdict": "match" | "anomaly" | "skipped:<why>", "notes": "<what you SAW>"}]}

Fastest path: dispatch the figure-reviewer subagent (Agent tool, subagent_type "figure-reviewer")
on each dir below. Escape hatch: if a figure genuinely cannot be assessed, record
verdict "skipped:<reason>" so the gate can clear.

Unreviewed figure dirs:
${dirs}
EOF

# Emit the block decision as JSON (python stdlib is hook-PATH-safe for json.dumps).
python3 - "$REASON" <<'PY' 2>/dev/null || printf '{"decision":"block","reason":"Figures produced but unreviewed; see figures.manifest.json and write figures.review.json."}\n'
import json, sys
print(json.dumps({"decision": "block", "reason": sys.argv[1]}))
PY
exit 0
