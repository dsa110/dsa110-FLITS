#!/usr/bin/env bash
# Regenerate manuscript budget figures from the frozen census and sync into Faber2026.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

flits_py() {
  env -i HOME="$HOME" PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin" \
    /opt/anaconda3/bin/conda run -n flits python "$@"
}

flits_py -m galaxies.foreground.build_artifacts
flits_py -m galaxies.foreground.sightline_budget
flits_py -m galaxies.v2_0.systems_figures --out-dir scratch/repro-foreground-figures
flits_py tools/sync_figures.py --apply

echo "Synced -> ~/Developer/overleaf/Faber2026/figures/"
echo "Visually review results/sightline_dm_scattering_budget.png and scratch/repro-foreground-figures/*.png before committing."
