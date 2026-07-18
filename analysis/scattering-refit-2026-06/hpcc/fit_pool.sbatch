#!/bin/bash
#SBATCH --job-name=jtfp
#SBATCH --partition=main
#SBATCH --mem-per-cpu=1G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/ubuntu/flits-runs/logs/%x_%j.out
#SBATCH --error=/home/ubuntu/flits-runs/logs/%x_%j.err
# cpus-per-task is set at submit time via --cpus-per-task=<nproc>
set -uo pipefail
BURST="$1"; NLIVE="$2"; NPROC="$3"; shift 3; EARGS="$*"
WT=/home/ubuntu/worktrees/joint-tf-fits
export FLITS_REPO="$WT" FLITS_RUNS=/home/ubuntu/flits-runs
export PYTHONPATH="$WT:$WT/scattering"
# BLAS stays 1 per pool worker so the NPROC dynesty-pool procs do not oversubscribe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp MPLBACKEND=Agg
# FLITS_MAX_CHANNELS / FLITS_SNR_TARGET inherited from submit env (--export) when set
set +u  # conda.sh references unbound PS1 under strict mode (esp. with --export)
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate flits-a1-312
set -u
cd "$WT/analysis/scattering-refit-2026-06"
T0=$(date +%s)
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID START=$(date -Iseconds) BURST=$BURST NLIVE=$NLIVE NPROC=$NPROC EARGS=[$EARGS] MAXCH=${FLITS_MAX_CHANNELS:-def} SNRT=${FLITS_SNR_TARGET:-def}"
python -u run_joint_fit.py "$BURST" "$NLIVE" "$NPROC" $EARGS
RC=$?
echo "DONE BURST=$BURST RC=$RC WALL_SEC=$(( $(date +%s)-T0 )) END=$(date -Iseconds)"
exit $RC
