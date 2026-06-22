#!/bin/bash
# Runs ON the HPCC login node (inside tmux). Flicker-free squeue + recent sacct.
INTERVAL=${1:-8}
trap 'printf "\033[?25h\n[queue watch stopped]\n"; exit 0' INT
printf '\033[2J\033[?25l'
while true; do
  frame=$(
    printf '\033[1;36m== QUEUE  %s ==\033[0m\n' "$(date '+%H:%M:%S')"
    squeue -u "$USER" -o '%.10i %.16j %.9P %.8T %.9M %.6C %R' 2>/dev/null
    printf '\n\033[1;36m== recent jobs (sacct, today) ==\033[0m\n'
    sacct -u "$USER" -X --starttime today \
      --format=JobID,JobName%16,State,Elapsed,NCPUS,MaxRSS -P 2>/dev/null \
      | tail -6 | column -t -s'|'
  )
  printf '\033[H'; printf '%s\n' "$frame" | sed $'s/$/\033[K/'; printf '\033[J'
  sleep "$INTERVAL"
done
