set -euo pipefail
F="${FLITS_RUNS:-/central/scratch/jfaber/flits-runs}"
B=$F/data/joint/_prefix_deltadm_2026-06-19
mkdir -p "$B"
# Preserve pre-fix (2026-06-19) result before the refit overwrites it
for f in wilhelm_joint_fit.json wilhelm_joint_samples.npz; do
  [ -f "$F/data/joint/$f" ] && cp -np "$F/data/joint/$f" "$B/$f" && echo "backed up $f"
done
cd "$F"
JID=$(sbatch --parsable -A "${FLITS_SLURM_ACCT:-radiolab}" --job-name=wilhelm-joint run_joint.sbatch wilhelm 600)
echo "submitted wilhelm-joint job=$JID"
squeue -u "$USER"
echo "logs: $F/logs/wilhelm-joint_${JID}.out"
