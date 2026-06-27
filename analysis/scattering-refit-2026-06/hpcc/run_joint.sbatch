#!/bin/bash
#SBATCH --job-name=flits-joint
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/central/scratch/jfaber/flits-runs/logs/%x_%j.out
#SBATCH --error=/central/scratch/jfaber/flits-runs/logs/%x_%j.err
set -euo pipefail
BURST="${1:?pass a burst name}"
NLIVE="${2:-600}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
FLITS_VENV="${FLITS_VENV:-/central/scratch/jfaber/envs/flits-joint}"
FLITS_REPO="${FLITS_REPO:-/central/scratch/jfaber/flits/dsa110-FLITS}"
export PATH="${FLITS_VENV}/bin:${PATH}"
export FLITS_REPO
NPROC="${SLURM_CPUS_PER_TASK:-8}"
echo "[$(date)] node=$(hostname) joint burst=${BURST} nlive=${NLIVE} nproc=${NPROC} venv=${FLITS_VENV} extra=${*:3}"
python -u /central/scratch/jfaber/flits-runs/run_joint_fit.py "${BURST}" "${NLIVE}" "${NPROC}" "${@:3}"
echo "[$(date)] done rc=$?"
