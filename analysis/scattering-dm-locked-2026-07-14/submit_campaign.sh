#!/bin/bash
set -euo pipefail

RUNS="${1:?campaign run directory}"
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${FLITS_REPO:?set FLITS_REPO to this campaign checkout}"

tail -n +2 "$HERE/fit_roster.csv" | while IFS=, read -r variant burst adopted dm_c dm_d comp_c comp_d nlive; do
  fix_c="$(awk -v a="$adopted" -v d="$dm_c" 'BEGIN {printf "%.9f", a-d}')"
  fix_d="$(awk -v a="$adopted" -v d="$dm_d" 'BEGIN {printf "%.9f", a-d}')"
  sbatch --job-name="dm-${variant}" --export=ALL,FLITS_REPO="$FLITS_REPO" \
    "$HERE/run_joint.sbatch" "$RUNS" "$variant" "$burst" "$nlive" \
    "$comp_c" "$comp_d" "$fix_c" "$fix_d"
done
