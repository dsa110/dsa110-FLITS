#!/bin/bash
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

RUNS="${1:?campaign run directory}"
VARIANT="${2:?variant}"
BURST="${3:?burst}"
NLIVE="${4:?nlive}"
COMP_C="${5:?CHIME component count}"
COMP_D="${6:?DSA component count}"
FIX_C="${7:?fixed CHIME residual DM}"
FIX_D="${8:?fixed DSA residual DM}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
VARIANT_RUNS="$RUNS/variants/$VARIANT"
mkdir -p "$VARIANT_RUNS/configs" "$VARIANT_RUNS/data/joint"
cp "$RUNS/configs/${BURST}_chime_run.yaml" "$VARIANT_RUNS/configs/"
cp "$RUNS/configs/${BURST}_dsa_run.yaml" "$VARIANT_RUNS/configs/"

export FLITS_RUNS="$VARIANT_RUNS"
export FLITS_REPO="${FLITS_REPO:?set FLITS_REPO to this campaign checkout}"
export PATH="${FLITS_VENV:-/central/scratch/jfaber/envs/flits-joint}/bin:${PATH}"

mkdir -p "$RUNS/logs"
exec >"$RUNS/logs/${VARIANT}_${SLURM_JOB_ID}.out" \
     2>"$RUNS/logs/${VARIANT}_${SLURM_JOB_ID}.err"

python -u "$FLITS_REPO/analysis/scattering-refit-2026-06/run_joint_fit.py" \
  "$BURST" "$NLIVE" "${SLURM_CPUS_PER_TASK:-8}" \
  --marginalize-gain \
  --components-C "$COMP_C" \
  --components-D "$COMP_D" \
  --fixed-delta-dm-C "$FIX_C" \
  --fixed-delta-dm-D "$FIX_D"
