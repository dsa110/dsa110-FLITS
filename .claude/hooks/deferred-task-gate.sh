#!/usr/bin/env bash
# Stop hook: block end-of-turn while the deferred-task ledger has UNCHECKED @agent items.
#
# Policy: a session shall not be completed while deferred tasks remain that THIS agent can
# execute or implement itself. The ledger (.agents/deferred-tasks.md) is a markdown
# checklist; each open `- [ ]` line is tagged with exactly one of:
#   @agent         -> the agent can do it now            => BLOCKS the stop
#   @human         -> needs a person / a one-way door    => does not block
#   @decision      -> a product/science choice is pending => does not block
#   @separate-lane -> another task's lane, not ours       => does not block
# Checked items (`- [x]`) never block. No-op when the ledger is absent.
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$PWD}"
LEDGER="$ROOT/.agents/deferred-tasks.md"
[ -f "$LEDGER" ] || exit 0

open_agent="$(grep -nE '^[[:space:]]*-[[:space:]]\[[[:space:]]\].*@agent' "$LEDGER" 2>/dev/null || true)"
[ -z "$open_agent" ] && exit 0

read -r -d '' REASON <<EOF || true
DEFERRED-TASK GATE: the session has unfinished work this agent can do itself.

Policy: a session shall not be completed while deferred tasks remain that the agent can
execute or implement (ledger: .agents/deferred-tasks.md). For each item below, either
FINISH it and check it off ('- [x]'), or -- only if it genuinely needs a person, a
one-way door (push/publish), or a pending product/science decision -- retag it
@human / @decision / @separate-lane (those do not block).

Open @agent items:
${open_agent}
EOF

# Emit the block decision as JSON (python3 stdlib is hook-PATH-safe).
python3 - "$REASON" <<'PY' 2>/dev/null || printf '{"decision":"block","reason":"Open @agent deferred tasks remain; see .agents/deferred-tasks.md."}\n'
import json, sys
print(json.dumps({"decision": "block", "reason": sys.argv[1]}))
PY
exit 0
