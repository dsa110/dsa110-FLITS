#!/usr/bin/env bash
# PostToolUse hook (Write|Edit): auto-fix + format a just-edited Python file with ruff.
#
# Contract: the agent pipes the hook payload as JSON on stdin. We pull the target
# path(s) with python3 stdlib (jq may be absent) from either shape — Claude Code's
# tool_input.file_path (absolute), or Codex apply_patch's '*** {Add,Update} File:'
# lines in tool_input.command (relative to payload cwd) — then run `ruff check --fix`
# and `ruff format` on each .py file.
#
# Fails OPEN: always exit 0, never block. No-op on non-.py paths or when ruff is
# unavailable. The flits-env binary is a direct path (no conda activation needed in a
# bare hook shell); a PATH ruff is the portable fallback.
set -uo pipefail

# Read the payload off stdin into a var first: the python heredoc below claims
# stdin (program-from-`-`), so the payload must reach it via env, not sys.stdin.
PAYLOAD="$(cat)"
FILES="$(PAYLOAD="$PAYLOAD" python3 - <<'PY' 2>/dev/null
import json, os, re
d = json.loads(os.environ.get("PAYLOAD", "") or "{}")
ti = d.get("tool_input", {}) or {}
cwd = d.get("cwd") or os.getcwd()
out = []
fp = ti.get("file_path")
if fp:  # Claude Code shape
    out.append(fp)
else:   # Codex apply_patch: path(s) embedded in the patch text
    for p in re.findall(r'^\*\*\* (?:Add|Update) File: (.+)$', ti.get("command", "") or "", re.M):
        out.append(p if os.path.isabs(p) else os.path.join(cwd, p))
print("\n".join(out))
PY
)"
[ -n "$FILES" ] || exit 0

RUFF_BIN=""
if [ -x "$HOME/.conda/envs/flits/bin/ruff" ]; then
  RUFF_BIN="$HOME/.conda/envs/flits/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  RUFF_BIN="ruff"
fi
[ -n "$RUFF_BIN" ] || exit 0
while IFS= read -r FILE; do
  case "$FILE" in
    *.py) [ -f "$FILE" ] || continue ;;
    *) continue ;;
  esac
  "$RUFF_BIN" check --fix "$FILE" || true
  "$RUFF_BIN" format "$FILE" || true
done <<EOF
$FILES
EOF
exit 0
