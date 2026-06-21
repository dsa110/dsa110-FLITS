#!/usr/bin/env bash
# PostToolUse hook (Write|Edit): auto-fix + format a just-edited Python file with ruff.
#
# Contract: Claude Code pipes the hook payload as JSON on stdin; tool_input.file_path
# is the absolute path Write/Edit touched. We pull it out with python3 stdlib (jq may
# be absent), and if it ends in .py we run `ruff check --fix` then `ruff format` on it.
#
# Fails OPEN: always exit 0, never block. No-op on non-.py paths or when ruff is
# unavailable. The flits-env binary is a direct path (no conda activation needed in a
# bare hook shell); a PATH ruff is the portable fallback.
set -uo pipefail

FILE="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)"
case "$FILE" in
  *.py) [ -f "$FILE" ] || exit 0 ;;
  *) exit 0 ;;
esac

RUFF_BIN=""
if [ -x "$HOME/.conda/envs/flits/bin/ruff" ]; then
  RUFF_BIN="$HOME/.conda/envs/flits/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  RUFF_BIN="ruff"
fi
if [ -n "$RUFF_BIN" ]; then
  "$RUFF_BIN" check --fix "$FILE" || true
  "$RUFF_BIN" format "$FILE" || true
fi
exit 0
