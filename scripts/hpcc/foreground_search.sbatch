#!/bin/bash
#SBATCH --job-name=fg-search
#SBATCH --account=radiolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/foreground_search_%j.out
#SBATCH --error=logs/foreground_search_%j.err

set -euo pipefail

REPO="${FLITS_REPO:-$HOME/flits/dsa110-FLITS}"
OUT="${FG_SEARCH_OUT:-scratch/repro-foreground-search-hpcc}"
VENV="${FLITS_VENV:-$HOME/flits/venv}"

cd "$REPO"
mkdir -p logs
source "$VENV/bin/activate"

python scripts/hpcc/run_foreground_search.py --output-dir "$OUT"
