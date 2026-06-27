#!/usr/bin/env bash
# Phase 2: rsync one migration_map id from h23 into iacobus (runs rsync ON iacobus).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INVENTORY="$REPO_ROOT/machine_inventory.yaml"
IACOBUS="${IACOBUS_HOST:-iacobus}"
LOG_DIR="${LOG_DIR:-$HOME/logs/h23_transfers}"
RSYNC_OPTS='-av --partial --ignore-existing --stats'
H23_SSH='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=60'

usage() {
  echo "Usage: $0 --id MIGRATION_MAP_ID [--dry-run]" >&2
  exit 1
}

DRY=0
ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --id) ID="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$ID" ]] || usage

mapfile -t FIELDS < <(python3 - "$INVENTORY" "$ID" <<'PY'
import sys, yaml
inv = yaml.safe_load(open(sys.argv[1]))
e = next(x for x in inv["migration_map"] if x["id"] == sys.argv[2])
src = e["source_path"]
dst = e["target_path"]
if e["id"] == "h23_dm_budget" and not dst.rstrip("/").endswith("h23_dm_budget"):
    dst = dst.rstrip("/") + "/h23_dm_budget"
print(src)
print(dst)
PY
)
SRC="${FIELDS[0]}"
DST="${FIELDS[1]}"
NAME="$ID"
LOG="$LOG_DIR/${NAME}.log"
mkdir -p "$LOG_DIR"

H23_SRC="h23:${SRC}/"
# trailing slash on dst dir
IACOBUS_DST="$DST/"

{
  echo "[$(date '+%F %T')] START $NAME dry_run=$DRY"
  echo "  src: $H23_SRC"
  echo "  dst: $IACOBUS_DST"
} >>"$LOG"

RSYNC_DRY=()
(( DRY )) && RSYNC_DRY=(--dry-run)

# rsync executes on iacobus; jakob-mbp forwards agent for h23 auth chain
if ssh -A -o BatchMode=yes "$IACOBUS" \
  "mkdir -p '$IACOBUS_DST' && rsync ${RSYNC_DRY[*]:-} $RSYNC_OPTS -e '$H23_SSH' '$H23_SRC' '$IACOBUS_DST'" >>"$LOG" 2>&1; then
  echo "[$(date '+%F %T')] SUCCESS $NAME" >>"$LOG"
  echo "OK $NAME (log: $LOG)"
else
  echo "[$(date '+%F %T')] FAILED $NAME" >>"$LOG"
  echo "FAIL $NAME — see $LOG" >&2
  exit 1
fi
