#!/usr/bin/env bash
# D3: rsync h17 arc_archive_2026-06 into iacobus (runs rsync ON iacobus).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INVENTORY="$REPO_ROOT/machine_inventory.yaml"
IACOBUS="${IACOBUS_HOST:-iacobus}"
LOG_DIR="${LOG_DIR:-$HOME/logs/h17_transfers}"
RSYNC_OPTS='-av --partial --ignore-existing --stats'
H17_SSH='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=60'

usage() {
  echo "Usage: $0 [--dry-run]" >&2
  exit 1
}

DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    *) usage ;;
  esac
done

ID="h17_arc_archive_copy"
mapfile -t FIELDS < <(python3 - "$INVENTORY" "$ID" <<'PY'
import sys, yaml
inv = yaml.safe_load(open(sys.argv[1]))
e = next(x for x in inv["migration_map"] if x["id"] == sys.argv[2])
print(e["source_path"])
print(e["target_path"])
PY
)
SRC="${FIELDS[0]}"
DST="${FIELDS[1]}"
LOG="$LOG_DIR/${ID}.log"
mkdir -p "$LOG_DIR"

H17_SRC="h17:${SRC}/"
IACOBUS_DST="$DST/"

{
  echo "[$(date '+%F %T')] START $ID dry_run=$DRY"
  echo "  src: $H17_SRC"
  echo "  dst: $IACOBUS_DST"
} >>"$LOG"

RSYNC_DRY=()
(( DRY )) && RSYNC_DRY=(--dry-run)

if ssh -A -o BatchMode=yes "$IACOBUS" \
  "mkdir -p '$IACOBUS_DST' && rsync ${RSYNC_DRY[*]:-} $RSYNC_OPTS -e '$H17_SSH' '$H17_SRC' '$IACOBUS_DST'" >>"$LOG" 2>&1; then
  echo "[$(date '+%F %T')] SUCCESS $ID" >>"$LOG"
  echo "OK $ID (log: $LOG)"
else
  echo "[$(date '+%F %T')] FAILED $ID" >>"$LOG"
  echo "FAIL $ID — see $LOG" >&2
  exit 1
fi
