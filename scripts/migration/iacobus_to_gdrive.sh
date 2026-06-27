#!/usr/bin/env bash
# Copy/sync iacobus CHIME_DSA_Codetections → Google Drive (jakobtfaber@gmail.com).
# Runs rclone ON iacobus (direct upload; jakob-mbp is orchestrator only).
#
# Prerequisite — gdrive remote (once per host that runs rclone):
#   rclone config   # name: gdrive-jakob, type: drive, scope: drive, account: jakobtfaber@gmail.com
# OAuth needs a browser. Headless iacobus flow:
#   1. On jakob-mbp:  rclone authorize "drive"
#   2. Copy the {"access_token":...} blob
#   3. On iacobus:    rclone config  → n) New remote → gdrive-jakob → paste token at "config_token"
# Verify: rclone about gdrive-jakob: && rclone lsd gdrive-jakob:Research/
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MANIFEST="${MANIFEST:-$REPO_ROOT/codetections_manifest.yaml}"
IACOBUS="${IACOBUS_HOST:-iacobus}"
REMOTE="${RCLONE_REMOTE:-gdrive-jakob}"
GDRIVE_ROOT="${GDRIVE_ROOT:-Research/CHIME_DSA_Codetections}"
SRC="${IACOBUS_SRC:-/Users/iacobus/Research/CHIME_DSA_Codetections}"
LOG_DIR="${LOG_DIR:-$HOME/logs/gdrive_transfers}"
# Perf defaults: saturate ~3 MiB/s home uplink without hammering Drive API.
# Override: RCLONE_TRANSFERS=2 RCLONE_PARALLEL_JOBS=3 ...
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-4}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-8}"
RCLONE_CHUNK="${RCLONE_CHUNK:-64M}"
RCLONE_BUFFER="${RCLONE_BUFFER:-32M}"
RCLONE_OPTS=(
  --progress --stats-one-line --stats 30s -v
  --ignore-existing
  --transfers "$RCLONE_TRANSFERS"
  --checkers "$RCLONE_CHECKERS"
  --drive-chunk-size "$RCLONE_CHUNK"
  --buffer-size "$RCLONE_BUFFER"
  --fast-list
)
TRANSFER="${TRANSFER:-copy}" # copy | sync

usage() {
  cat <<EOF >&2
Usage: $0 [--dry-run] [--subdir NAME] [--verify-only] [--sync] [--parallel]

  --dry-run       rclone dry-run (no bytes moved)
  --subdir NAME   transfer one top-level subdir only (e.g. metadata)
  --verify-only   sentinel SHA-256 check on remote; no transfer
  --sync          use rclone sync instead of copy (destructive on remote extras)
  --parallel      run disjoint subtree copies in parallel ON iacobus (fast path)

Env: RCLONE_REMOTE, GDRIVE_ROOT, IACOBUS_HOST, LOG_DIR,
     RCLONE_TRANSFERS (4), RCLONE_CHECKERS (8), RCLONE_CHUNK (64M),
     RCLONE_BUFFER (32M)
EOF
  exit 1
}

DRY=0
SUBDIR=""
VERIFY_ONLY=0
PARALLEL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --subdir) SUBDIR="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --sync) TRANSFER=sync; shift ;;
    --parallel) PARALLEL=1; shift ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

NAME="iacobus_to_gdrive"
[[ -n "$SUBDIR" ]] && NAME="${NAME}_${SUBDIR}"
LOG="$LOG_DIR/${NAME}.log"
mkdir -p "$LOG_DIR"

LOCAL_SRC="$SRC/"
REMOTE_DST="${REMOTE}:${GDRIVE_ROOT}/"
[[ -n "$SUBDIR" ]] && LOCAL_SRC="${SRC}/${SUBDIR}/" && REMOTE_DST="${REMOTE}:${GDRIVE_ROOT}/${SUBDIR}/"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

sentinel_verify() {
  python3 - "$MANIFEST" "$REMOTE" "$GDRIVE_ROOT" <<'PY' >>"$LOG" 2>&1
import hashlib, json, subprocess, sys, yaml

manifest_path, remote, gdrive_root = sys.argv[1:4]
manifest = yaml.safe_load(open(manifest_path))
meta = manifest["subdirs"]["metadata"]
rel = meta["sentinel_path"]
expected = meta["sentinel_sha256"]
nbytes = meta["sentinel_sha256_prefix_bytes"]
remote_path = f"{remote}:{gdrive_root}/{rel}"

proc = subprocess.run(
    ["rclone", "cat", remote_path],
    capture_output=True,
)
if proc.returncode != 0:
    print(f"SENTINEL FAIL: rclone cat {remote_path}: {proc.stderr.decode().strip()}")
    sys.exit(1)
data = proc.stdout[:nbytes]
got = hashlib.sha256(data).hexdigest()
if got != expected:
    print(f"SENTINEL FAIL: {rel} sha256 prefix mismatch")
    print(f"  expected: {expected}")
    print(f"  got:      {got}")
    sys.exit(1)
print(f"SENTINEL PASS: {rel} ({nbytes} B prefix sha256 match)")
PY
  return $?
}

if (( VERIFY_ONLY )); then
  log "VERIFY-ONLY sentinel check remote=$REMOTE dst=$GDRIVE_ROOT"
  sentinel_verify || { echo "FAIL sentinel — see $LOG" >&2; exit 1; }
  echo "OK sentinel (log: $LOG)"
  exit 0
fi

RCLONE_DRY=()
(( DRY )) && RCLONE_DRY=(--dry-run)

log "START $NAME transfer=$TRANSFER dry_run=$DRY parallel=$PARALLEL"
log "  src: iacobus:$LOCAL_SRC"
log "  dst: $REMOTE_DST"

# Preflight: remote exists on iacobus
if ! ssh -o BatchMode=yes "$IACOBUS" "rclone listremotes" 2>>"$LOG" | grep -q "^${REMOTE}:$"; then
  log "FAILED preflight: remote ${REMOTE}: not configured on $IACOBUS"
  echo "FAIL — configure gdrive remote on iacobus first (see script header). Log: $LOG" >&2
  exit 1
fi

if (( PARALLEL )) && [[ -z "$SUBDIR" ]]; then
  IACOBUS_LOG_DIR="${IACOBUS_LOG_DIR:-/Users/iacobus/logs/gdrive_transfers}"
  ssh -o BatchMode=yes "$IACOBUS" "mkdir -p '$IACOBUS_LOG_DIR'"
  log "Launching 4 parallel rclone jobs on $IACOBUS (logs under $IACOBUS_LOG_DIR/)"
  ssh -o BatchMode=yes "$IACOBUS" bash -s \
    "$TRANSFER" "$DRY" "$SRC" "$REMOTE" "$GDRIVE_ROOT" "$IACOBUS_LOG_DIR" \
    "$RCLONE_TRANSFERS" "$RCLONE_CHECKERS" "$RCLONE_CHUNK" "$RCLONE_BUFFER" <<'REMOTE'
set -eo pipefail
TRANSFER=$1 DRY=$2 SRC=$3 REMOTE=$4 GDRIVE_ROOT=$5 LOG_DIR=$6
RCLONE_TRANSFERS=$7 RCLONE_CHECKERS=$8 RCLONE_CHUNK=$9 RCLONE_BUFFER=${10}
OPTS=(
  --progress --stats-one-line --stats 30s
  --ignore-existing
  --transfers "$RCLONE_TRANSFERS"
  --checkers "$RCLONE_CHECKERS"
  --drive-chunk-size "$RCLONE_CHUNK"
  --buffer-size "$RCLONE_BUFFER"
  --fast-list
)
launch() {
  local name=$1; shift
  local log="$LOG_DIR/iacobus_to_gdrive_${name}.log"
  echo "[$(date '+%F %T')] START parallel job $name $*" >>"$log"
  if (( DRY )); then
    nohup rclone "$TRANSFER" --dry-run "${OPTS[@]}" "$@" >>"$log" 2>&1 &
  else
    nohup rclone "$TRANSFER" "${OPTS[@]}" "$@" >>"$log" 2>&1 &
  fi
  echo $! >>"$LOG_DIR/iacobus_to_gdrive_parallel.pids"
  echo "[$(date '+%F %T')] PID $! job $name" >>"$log"
}
: >"$LOG_DIR/iacobus_to_gdrive_parallel.pids"
# burstprop_paper: separate project (not CHIME–DSA codetections); see DATA_LOCATIONS out-of-scope.
launch archive --exclude "burstprop_paper/**" "${SRC}/archive/" "${REMOTE}:${GDRIVE_ROOT}/archive/"
launch burst_pickles "${SRC}/burst_pickles/" "${REMOTE}:${GDRIVE_ROOT}/burst_pickles/"
launch burst_npys "${SRC}/burst_npys/" "${REMOTE}:${GDRIVE_ROOT}/burst_npys/"
launch rest \
  --exclude "archive/**" --exclude "burst_pickles/**" --exclude "burst_npys/**" \
  "${SRC}/" "${REMOTE}:${GDRIVE_ROOT}/"
cat "$LOG_DIR/iacobus_to_gdrive_parallel.pids"
REMOTE
  log "Parallel jobs launched on $IACOBUS — monitor: ssh $IACOBUS 'tail -f $IACOBUS_LOG_DIR/iacobus_to_gdrive_*.log'"
  echo "OK ${NAME}_parallel launched (orchestrator log: $LOG)"
  exit 0
fi

RSYNC_CMD=(rclone "$TRANSFER" "${RCLONE_DRY[@]}" "${RCLONE_OPTS[@]}" "$LOCAL_SRC" "$REMOTE_DST")
if ssh -o BatchMode=yes "$IACOBUS" "${RSYNC_CMD[@]}" >>"$LOG" 2>&1; then
  log "SUCCESS rclone $TRANSFER"
else
  log "FAILED rclone $TRANSFER"
  echo "FAIL $NAME — see $LOG" >&2
  exit 1
fi

if (( ! DRY )) && [[ -z "$SUBDIR" || "$SUBDIR" == "metadata" ]]; then
  log "sentinel verify metadata/interveners.ipynb"
  sentinel_verify || { echo "FAIL sentinel — see $LOG" >&2; exit 1; }
fi

echo "OK $NAME (log: $LOG)"
