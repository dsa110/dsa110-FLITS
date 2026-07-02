#!/usr/bin/env bash
# Pull HPCC foreground search output, diff vs frozen census, regen figures if unchanged.
#
# Usage:
#   scripts/manuscript/slot_hpcc_foreground.sh              # pull + diff only
#   scripts/manuscript/slot_hpcc_foreground.sh --apply    # also regen + sync figures
#
# After a census change: adjudicate through scratch/codetection/* then rerun with --apply.
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HPCC="${HPCC:-hpcc}"
REMOTE="${REMOTE_DIR:-~/flits/dsa110-FLITS/scratch/repro-foreground-search-hpcc}"
LOCAL="$REPO/scratch/repro-foreground-search-hpcc"

mkdir -p "$LOCAL"
rsync -av --delete "${HPCC}:${REMOTE}/" "$LOCAL/"

flits_py() {
  env -i HOME="$HOME" PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin" \
    /opt/anaconda3/bin/conda run -n flits python "$@"
}

cd "$REPO"
flits_py - <<'PY'
from pathlib import Path
import pandas as pd

local = Path("scratch/repro-foreground-search-hpcc")
old = pd.read_csv("scratch/codetection/foreground_final.csv")
new_rows = []
for path in sorted(local.glob("*_galaxies.csv")):
    df = pd.read_csv(path)
    if df.empty:
        continue
    df["nickname"] = path.name.replace("_galaxies.csv", "")
    new_rows.append(df)
new = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()

print("=== HPCC search vs frozen census ===")
print("frozen rows:", len(old), "| fresh candidate rows:", len(new))
if local.joinpath("done.txt").exists():
    done = local.joinpath("done.txt").read_text().strip().splitlines()
    print("sightlines completed:", len(done), "->", ", ".join(done))
if not new.empty:
    print("\nfresh rows by nickname:")
    print(new.nickname.value_counts().sort_index().to_string())
print("\nNo automatic census merge — diff manually before changing foreground_final.csv.")
PY

if [[ "$APPLY" -eq 1 ]]; then
  echo "Regenerating manuscript figures from frozen census..."
  bash "$REPO/scripts/manuscript/regenerate_budget_figures.sh"
fi
