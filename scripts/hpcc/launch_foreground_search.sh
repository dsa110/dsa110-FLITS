#!/usr/bin/env bash
# Launch foreground galaxy search on Caltech HPCC (login node submits SLURM job).
#
# Prereqs: ssh hpcc works (ProxyJump campus-pi; may need interactive Duo once).
# Repo on HPCC: ~/flits/dsa110-FLITS (see dotfiles memory project_dsa110-flits-dev-topology).
#
# Usage (from Mac):
#   cd ~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS
#   git push origin HEAD   # sync code first
#   ./scripts/hpcc/launch_foreground_search.sh
#
# Or via hpcc-run after `hpcc-dev` holds a node (interactive dev, not batch):
#   hpcc-run --login 'cd ~/flits/dsa110-FLITS && git pull && sbatch scripts/hpcc/foreground_search.sbatch'
set -euo pipefail

REPO_LOCAL="${REPO_LOCAL:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
BRANCH="${BRANCH:-$(git -C "$REPO_LOCAL" rev-parse --abbrev-ref HEAD)}"
REMOTE_REPO="${REMOTE_REPO:-$HOME/flits/dsa110-FLITS}"

echo "Local branch: $BRANCH"
echo "Submitting foreground search on hpcc (repo $REMOTE_REPO)..."

ssh -o BatchMode=yes hpcc bash -s <<EOF
set -euo pipefail
cd "$REMOTE_REPO"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull --ff-only origin "$BRANCH"
mkdir -p logs
JOB=\$(sbatch -A radiolab scripts/hpcc/foreground_search.sbatch | awk '{print \$4}')
echo "submitted job \$JOB"
squeue -j "\$JOB" -o '%.18i %.9P %.20j %.8T %.10M %.6D %R'
echo "tail -f $REMOTE_REPO/logs/foreground_search_\${JOB}.out"
EOF
