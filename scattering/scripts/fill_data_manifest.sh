#!/usr/bin/env bash
# fill_data_manifest.sh — populate sha256 + bytes in data-manifest.csv.
#
# Run where the burst .npy are reachable (CANFAR arc / OVRO lxd, or a local
# replica). For each manifest row it looks for the file in $DATA_DIR (by
# filename), and if found computes sha256 + byte size and writes them back,
# flipping status PENDING_CHECKSUM -> OK. Rows it cannot resolve are left as-is
# (their status flags, e.g. CONFIG_BUG_*, are preserved).
#
# Usage:
#   DATA_DIR=/arc/home/jfaber/.../DSA_bursts ./scattering/scripts/fill_data_manifest.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${MANIFEST:-$REPO_ROOT/data-manifest.csv}"
DATA_DIR="${DATA_DIR:-/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts}"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 2; }
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2; exit 2; }

sha_cmd() { command -v sha256sum >/dev/null 2>&1 && sha256sum "$1" | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'; }
bytes_cmd() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null; }

tmp="$MANIFEST.tmp"
filled=0; missing=0
{
  IFS= read -r header
  echo "$header"
  while IFS=, read -r burst tel dm filename arc_path sha bytes status; do
    [ -z "${burst:-}" ] && continue
    f="$DATA_DIR/$filename"
    if [ -f "$f" ]; then
      sha="$(sha_cmd "$f")"; bytes="$(bytes_cmd "$f")"; status="OK"
      filled=$((filled+1))
    else
      missing=$((missing+1))
    fi
    echo "$burst,$tel,$dm,$filename,$arc_path,$sha,$bytes,$status"
  done
} < "$MANIFEST" > "$tmp"

mv "$tmp" "$MANIFEST"
echo "Filled $filled rows; $missing not found in $DATA_DIR. Review CONFIG_* rows by hand."
