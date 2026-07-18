#!/bin/bash
#SBATCH --job-name=jtfpl
#SBATCH --partition=main
#SBATCH --mem-per-cpu=1G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/ubuntu/flits-runs/logs/%x_%j.out
#SBATCH --error=/home/ubuntu/flits-runs/logs/%x_%j.err
set -uo pipefail
BURST="$1"; NLIVE="$2"; NPROC="$3"
WT=/home/ubuntu/worktrees/joint-tf-fits
export FLITS_REPO="$WT" FLITS_RUNS=/home/ubuntu/flits-runs
export PYTHONPATH="$WT:$WT/scattering:$WT/analysis/scattering-refit-2026-06"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp MPLBACKEND=Agg FLITS_JOINT_AUTO_TF=1
set +u
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate flits-a1-312
set -u
cd "$WT/analysis/scattering-refit-2026-06"
T0=$(date +%s)
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID START=$(date -Iseconds) BURST=$BURST NLIVE=$NLIVE NPROC=$NPROC PLPBF=1"
python -u run_joint_fit_plpbf.py "$BURST" "$NLIVE" "$NPROC"
RC=$?
echo "DONE BURST=$BURST RC=$RC WALL_SEC=$(( $(date +%s)-T0 )) END=$(date -Iseconds)"
exit $RC
