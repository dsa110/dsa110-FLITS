FLITS_REPO="${FLITS_REPO:-/home/jfaber/flits/dsa110-FLITS}"
FLITS_RUNS="${FLITS_RUNS:-/central/scratch/jfaber/flits-runs}"
FLITS_VENV="${FLITS_VENV:-/home/jfaber/flits/venv}"
source "$FLITS_VENV/bin/activate"
cd "$FLITS_REPO/scattering" || exit 1
FLITS_REPO="$FLITS_REPO" FLITS_RUNS="$FLITS_RUNS" \
  python -u "$FLITS_RUNS/fullband_aligned.py" "${1:-wilhelm}" 2>&1 \
  | grep -viE 'warn|deprecat|loading' | tail -5
