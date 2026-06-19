#!/usr/bin/env bash
# Batch-run the scattering pipeline over all 12 CHIME codetection bursts.
#
# Designed to run on the host where the burst .npy data live (CANFAR arc /
# OVRO lxd). It does NOT trust the path baked into each config: it takes the
# data filename from the config, re-points it at $DATA_DIR, writes a temp config,
# and runs the pipeline on that -- so you only set DATA_DIR once.
#
# Each fit's *_fit_results.json (now carrying tau +/- sigma and the recalibrated
# quality_flag) is collected into $OUT_DIR. Failures are logged and skipped; the
# run continues. After it finishes, verify + label with:
#     python scattering/scripts/verify_fits.py "$OUT_DIR" --csv "$OUT_DIR/summary.csv"
# then scp $OUT_DIR back for ingestion into the DM/scattering budget.
#
# Usage:
#   DATA_DIR=/path/to/DSA_bursts ./scattering/scripts/run_all_chime_bursts.sh
#   DATA_DIR=... OUT_DIR=... BURSTS="wilhelm freya casey" ./...sh   # subset
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/CHIME_bursts/dmphase}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/scattering_runs/$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="$REPO_ROOT/scattering/configs/bursts/chime"
PYTHON="${PYTHON:-python}"
RUNNER="$REPO_ROOT/scattering/scripts/run_scattering_analysis.py"

ALL_BURSTS="zach whitney oran isha wilhelm phineas freya hamilton mahi chromatica casey johndoeII"
BURSTS="${BURSTS:-$ALL_BURSTS}"

mkdir -p "$OUT_DIR"
echo "Repo:      $REPO_ROOT"
echo "Data dir:  $DATA_DIR"
echo "Out dir:   $OUT_DIR"
echo "Bursts:    $BURSTS"
echo "Python:    $PYTHON"
echo "==============================================================="

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Set DATA_DIR to where the *_cntr_bpc.npy files live." >&2
  exit 2
fi

ok=0; fail=0; missing=0
for burst in $BURSTS; do
  cfg="$CONFIG_DIR/${burst}_chime.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[$burst] SKIP: no config $cfg" >&2; missing=$((missing+1)); continue
  fi

  # Extract the data filename from the config's path: line, re-point at DATA_DIR.
  orig_path="$(grep -E '^[[:space:]]*path:' "$cfg" | head -1 | sed -E 's/^[[:space:]]*path:[[:space:]]*//')"
  fname="$(basename "$orig_path")"
  data_path="$DATA_DIR/$fname"
  if [[ ! -f "$data_path" ]]; then
    # Fall back to a glob on the burst name in case the id tokens differ.
    alt="$(ls "$DATA_DIR/${burst}"*_cntr_bpc.npy 2>/dev/null | head -1 || true)"
    if [[ -n "$alt" ]]; then
      data_path="$alt"; fname="$(basename "$alt")"
    else
      echo "[$burst] SKIP: data not found ($data_path)" >&2; missing=$((missing+1)); continue
    fi
  fi

  tmp_cfg="$OUT_DIR/${burst}_chime.runtime.yaml"
  # The runtime config is written to OUT_DIR, so the source config's relative
  # telcfg_path/sampcfg_path (../../telescopes.yaml) no longer resolve. Rewrite
  # the data path AND pin telcfg/sampcfg to absolute repo paths; append them if
  # the source config omitted them (some configs rely on the loader default).
  telcfg="$CONFIG_DIR/../../telescopes.yaml"; telcfg="$(cd "$(dirname "$telcfg")" && pwd)/$(basename "$telcfg")"
  sampcfg="$CONFIG_DIR/../../sampler.yaml";   sampcfg="$(cd "$(dirname "$sampcfg")" && pwd)/$(basename "$sampcfg")"
  sed -E \
    -e "s#^([[:space:]]*path:[[:space:]]*).*#\1$data_path#" \
    -e "s#^([[:space:]]*telcfg_path:[[:space:]]*).*#\1$telcfg#" \
    -e "s#^([[:space:]]*sampcfg_path:[[:space:]]*).*#\1$sampcfg#" \
    "$cfg" > "$tmp_cfg"
  grep -qE '^[[:space:]]*telcfg_path:'  "$tmp_cfg" || echo "telcfg_path: $telcfg"   >> "$tmp_cfg"
  grep -qE '^[[:space:]]*sampcfg_path:' "$tmp_cfg" || echo "sampcfg_path: $sampcfg" >> "$tmp_cfg"

  echo "[$burst] running -> $data_path"
  log="$OUT_DIR/${burst}.log"
  if "$PYTHON" "$RUNNER" "$tmp_cfg" > "$log" 2>&1; then
    # Collect the fit_results.json the pipeline wrote (data.parent/analysis_*/).
    found="$(ls -t "$(dirname "$data_path")"/analysis_*/*fit_results.json 2>/dev/null | head -1 || true)"
    if [[ -n "$found" ]]; then
      cp "$found" "$OUT_DIR/${burst}_fit_results.json"
      echo "[$burst] OK -> $OUT_DIR/${burst}_fit_results.json"
      ok=$((ok+1))
    else
      echo "[$burst] WARN: ran but no fit_results.json found (see $log)" >&2
      fail=$((fail+1))
    fi
  else
    echo "[$burst] FAIL: pipeline error (see $log)" >&2
    fail=$((fail+1))
  fi
done

echo "==============================================================="
echo "Done. ok=$ok fail=$fail missing=$missing  -> $OUT_DIR"
echo "Next: $PYTHON scattering/scripts/verify_fits.py \"$OUT_DIR\" --csv \"$OUT_DIR/summary.csv\""
