#!/usr/bin/env bash
# PreToolUse(Bash) guard: refuse `git commit` while HEAD is a protected branch
# (main/master). Branch hygiene is otherwise prose-only here, and origin/main
# already carries direct non-PR commits (a2333b5, eed6f04, ...). The agent should
# branch first, then commit.
#
# Fails OPEN whenever it cannot prove the branch is protected (not a git repo,
# detached HEAD, git missing) -- a guard must never wedge unrelated Bash calls.
# Scope: only the agent's own `git commit` calls in a Claude session; an external
# auto-committer is a separate path this hook cannot see.
set -uo pipefail

payload="$(cat)"

# Detect a real `git commit` invocation, token-aware: find a `git` token, skip
# git's global options (and the values of -C/-c/--git-dir/...), and require the
# subcommand to be exactly `commit`. This catches `git -C <path> commit` and
# `git -c k=v commit` (forms the agent actually emits) while NOT matching
# `git commit-graph`, `git log --grep=...commit`, etc. Any python/parse failure
# prints nothing -> treated as not-a-commit -> fail open (a guard must never
# wedge unrelated Bash calls).
cmd_is_commit="$(printf '%s' "$payload" | python3 -c '
import json, shlex, sys
try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    toks = shlex.split(cmd, comments=False)
except Exception:
    print("0"); sys.exit(0)
val_opts = {"-C", "-c", "--git-dir", "--work-tree", "--namespace",
            "--super-prefix", "--exec-path"}
i = 0
while i < len(toks):
    if toks[i] == "git":
        j = i + 1
        while j < len(toks):
            t = toks[j]
            if t in val_opts:
                j += 2
            elif t.startswith("-"):
                j += 1
            else:
                break
        if j < len(toks) and toks[j] == "commit":
            print("1"); sys.exit(0)
    i += 1
print("0")
' 2>/dev/null || true)"
[ "$cmd_is_commit" = "1" ] || exit 0

branch="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
[ -z "$branch" ] && exit 0  # detached HEAD / not a repo -> fail open

case "$branch" in
  main | master)
    printf 'refuse: commit on protected branch "%s". Branch first (git switch -c <feature-branch>), then commit.\n' "$branch" >&2
    exit 2
    ;;
esac
exit 0
