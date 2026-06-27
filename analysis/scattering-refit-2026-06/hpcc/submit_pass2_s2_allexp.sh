#!/usr/bin/env bash
# Pass 2 fixed-s² all-exp component ladders (ADR-0003 canonical PBF).
# Bursts: oran, isha (C1D1 vs C2D1), mahi (C1D1 force-multi vs C2D1), phineas (C3D2 vs C3D3).
# s² ∈ {1, 10, 100} per cell → 24 jobs.
set -euo pipefail
RUNS="${FLITS_RUNS:-/central/scratch/jfaber/flits-runs}"
ACCT="${FLITS_SLURM_ACCT:-radiolab}"
NLIVE="${NLIVE:-600}"
COMMON=(--alpha-lo 1.0 --alpha-hi 6.0 --pbf-C exp --pbf-D exp --marginalize-gain)
S2=(1 10 100)
LOG="${RUNS}/logs/pass2_s2_allexp_submit.$(date +%Y%m%dT%H%M%S).ids"
mkdir -p "${RUNS}/logs"
: >"$LOG"

submit() {
  local burst=$1 c=$2 d=$3 s2=$4
  shift 4
  local extra=("$@")
  local j
  j=$(sbatch --parsable -A "$ACCT" --job-name="${burst}-C${c}D${d}-s2-${s2}" \
    "${RUNS}/run_joint.sbatch" "$burst" "$NLIVE" \
    "${COMMON[@]}" --components-C "$c" --components-D "$d" --gain-s2 "$s2" "${extra[@]}")
  printf '%s %s C%sD%s s2=%s %s\n' "$j" "$burst" "$c" "$d" "$s2" "${extra[*]:-}" | tee -a "$LOG"
}

for burst in oran isha; do
  for s2 in "${S2[@]}"; do
    submit "$burst" 1 1 "$s2" --force-multi
    submit "$burst" 2 1 "$s2"
  done
done

for s2 in "${S2[@]}"; do
  submit mahi 1 1 "$s2" --force-multi
  submit mahi 2 1 "$s2"
done

for s2 in "${S2[@]}"; do
  submit phineas 3 2 "$s2"
  submit phineas 3 3 "$s2"
done

echo "Submitted $(wc -l <"$LOG") jobs; ids -> $LOG"
