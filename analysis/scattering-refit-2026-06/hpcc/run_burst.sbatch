#!/bin/bash
#SBATCH --job-name=flits-fit
#SBATCH --partition=expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/central/scratch/jfaber/flits-runs/logs/%x_%j.out
#SBATCH --error=/central/scratch/jfaber/flits-runs/logs/%x_%j.err

# Usage: sbatch --job-name=<burst> run_burst.sbatch <config.yaml>
set -euo pipefail
CONFIG="${1:?pass a config yaml as arg 1}"

# Pin BLAS to 1 thread/process: the dynesty pool runs <cpus-per-task> python
# workers; without this each spawns its own BLAS pool and oversubscribes the
# node (256 threads on 16 cores), which silently inflates wall time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/jfaber/flits/venv/bin/activate
cd /home/jfaber/flits/dsa110-FLITS

echo "[$(date)] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} config=${CONFIG}"
python -u scattering/scripts/run_scattering_analysis.py "${CONFIG}"
echo "[$(date)] done rc=$?"
