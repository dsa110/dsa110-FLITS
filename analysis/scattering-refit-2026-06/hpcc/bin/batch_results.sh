#!/bin/bash
# Runs ON the HPCC login node. Live per-burst results board for the 12-CHIME batch:
# squeue state + each burst's best model / tau_1GHz / reduced-chi2, scanned from
# its job log. Flicker-free (build frame, then repaint in place).
LOGDIR=/central/scratch/jfaber/flits-runs/logs
INTERVAL=${1:-8}
BURSTS="casey chromatica freya hamilton isha johndoeII mahi oran phineas whitney wilhelm zach"
trap 'printf "\033[?25h\n[results board stopped]\n"; exit 0' INT
printf '\033[2J\033[?25l'
while true; do
  frame=$(
    printf '\033[1;36m== 12-CHIME batch  %s ==\033[0m\n' "$(date '+%H:%M:%S')"
    squeue -u "$USER" -o '%.10i %.16j %.8T %.5C %.9M' 2>/dev/null
    printf '\n\033[1m%-12s %-9s %-6s %-12s %-8s\033[0m\n' BURST STATE MODEL "tau1GHz(ms)" "chi2/dof"
    for b in $BURSTS; do
      L=$(ls -t "$LOGDIR/${b}-chime_"*.out 2>/dev/null | head -1)
      st="-"; model="-"; tau="-"; chi="-"
      if [ -n "$L" ]; then
        if grep -aq "ANALYSIS COMPLETE" "$L" 2>/dev/null; then st="DONE"; else st="run"; fi
        model=$(grep -aoE "Best model by evidence: M[0-9]" "$L" 2>/dev/null | tail -1 | grep -oE "M[0-9]")
        tau=$(grep -aE "tau_1ghz *\|" "$L" 2>/dev/null | tail -1 | sed -E 's/.*\| *//; s/ *$//')
        chi=$(grep -aE "Reduced Chi2:" "$L" 2>/dev/null | tail -1 | sed -E 's/.*: *//')
      fi
      printf '%-12s %-9s %-6s %-12s %-8s\n' "$b" "${st:--}" "${model:--}" "${tau:--}" "${chi:--}"
    done
  )
  printf '\033[H'; printf '%s\n' "$frame" | sed $'s/$/\033[K/'; printf '\033[J'
  sleep "$INTERVAL"
done
