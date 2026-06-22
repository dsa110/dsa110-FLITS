#!/bin/bash
# Live HPCC FLITS run monitor. Polls Slurm queue + tails the newest job log.
# Flicker-free: builds the whole frame (incl. the slow ssh fetch) BEFORE touching
# the screen, then repaints in place (home + per-line clear-to-EOL). Never blanks.
HPCC=hpcc
LOGDIR=/central/scratch/jfaber/flits-runs/logs
INTERVAL=${1:-6}
trap 'printf "\033[?25h\n"; echo "[monitor stopped]"; exit 0' INT
printf '\033[2J\033[?25l'   # one initial clear + hide cursor
while true; do
  frame=$(
    printf '\033[1m=== HPCC FLITS runs — %s ===\033[0m\n' "$(date '+%H:%M:%S')"
    ssh -o BatchMode=yes "$HPCC" "
      echo '--- squeue (jfaber) ---'
      squeue -u jfaber -o '%.10i %.16j %.10P %.8T %.9M %.9l %.5C %R' 2>/dev/null
      echo
      L=\$(ls -t $LOGDIR/*.out 2>/dev/null | head -1)
      echo \"--- newest log: \${L##*/} ---\"
      if [ -n \"\$L\" ]; then
        grep -aE 'node=|\[M[0-9]\] log\(Z\)|BEST|Best model|tau_1ghz|Reduced Chi2|R-squared|Flag:|ANALYSIS COMPLETE|done rc=' \"\$L\" 2>/dev/null | tail -16 || tail -16 \"\$L\"
      fi
    " 2>/dev/null
    printf '\n\033[2m(refresh %ss · Ctrl-C to stop)\033[0m' "$INTERVAL"
  )
  printf '\033[H'                              # cursor home, no clear
  printf '%s\n' "$frame" | sed $'s/$/\033[K/'  # paint, clearing each line to EOL
  printf '\033[J'                              # erase any leftover below a shorter frame
  sleep "$INTERVAL"
done
