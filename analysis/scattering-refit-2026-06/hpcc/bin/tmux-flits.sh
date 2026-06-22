#!/bin/bash
# Build (idempotently) the persistent `flits` tmux session on the HPCC login node:
#   pane 0 (left)  = interactive shell in the run dir
#   pane 1 (top-right) = squeue + sacct watcher
#   pane 2 (bot-right) = live follow of the newest job log
S=flits
RUNDIR=/central/scratch/jfaber/flits-runs
BIN="$HOME/flits/bin"
# Use a stable home-dir socket dir so the build here and the attach from iTerm
# agree. (tmux only runs on compute nodes — login nodes reap its server.)
export TMUX_TMPDIR="$HOME/.tmux-tmp"
mkdir -p "$TMUX_TMPDIR"; chmod 700 "$TMUX_TMPDIR"
tmux has-session -t "$S" 2>/dev/null && exit 0   # already up — leave it

tmux new-session -d -s "$S" -c "$RUNDIR" -x 230 -y 55
tmux rename-window -t "$S:0" hpcc
tmux split-window -h -t "$S:0.0" -c "$RUNDIR"     # pane 1 (right)
tmux split-window -v -t "$S:0.1" -c "$RUNDIR"     # pane 2 (right-bottom)
tmux send-keys -t "$S:0.1" "$BIN/queue_watch.sh 8" C-m
tmux send-keys -t "$S:0.2" "$BIN/follow_latest.sh" C-m
tmux resize-pane -t "$S:0.0" -x 58%               # left shell wider

# titles / borders / colors (rendered in plain attach; iTerm -CC uses its own chrome)
tmux set -t "$S" pane-border-status top
tmux set -t "$S" pane-border-format ' #[bold]#{pane_title} '
tmux select-pane -t "$S:0.0" -T 'SHELL · flits-runs'
tmux select-pane -t "$S:0.1" -T 'QUEUE + ACCT'
tmux select-pane -t "$S:0.2" -T 'LIVE LOG'
tmux set -t "$S" mouse on
tmux set -t "$S" status-style 'bg=colour24,fg=colour231'
tmux set -t "$S" status-left ' #[bold]FLITS@HPCC '
tmux set -t "$S" status-right ' %H:%M '
tmux select-pane -t "$S:0.0"
