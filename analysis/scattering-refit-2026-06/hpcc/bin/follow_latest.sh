#!/bin/bash
# Runs ON the HPCC login node (inside tmux). Tails the newest job log and
# auto-switches when a newer one appears (tail -F survives truncation).
LOGDIR=${1:-/central/scratch/jfaber/flits-runs/logs}
trap 'kill $TPID 2>/dev/null; printf "\n[log follow stopped]\n"; exit 0' INT
cur=""; TPID=""
while true; do
  L=$(ls -t "$LOGDIR"/*.out 2>/dev/null | head -1)
  if [ -n "$L" ] && [ "$L" != "$cur" ]; then
    [ -n "$TPID" ] && kill "$TPID" 2>/dev/null
    cur="$L"; clear
    printf '\033[1;32m== LIVE LOG: %s ==\033[0m\n' "${L##*/}"
    stdbuf -oL tail -n 40 -F "$L" 2>/dev/null & TPID=$!
  fi
  sleep 4
done
