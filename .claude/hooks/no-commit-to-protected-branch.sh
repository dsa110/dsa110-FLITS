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
#
# Worktree-aware: the guard must judge the branch the commit ACTUALLY targets,
# not whichever checkout is the process root. The hook's cwd is always the
# project root (the main checkout); a commit in a separate worktree on a feature
# branch must not be blocked just because the main checkout sits on main. So the
# parser also emits the commit's effective working dir -- a leading `cd <path>`
# and/or the matching `git -C <path>` -- and the branch check runs against that.
parse="$(printf '%s' "$payload" | python3 -c '
import json, os, shlex, sys
try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    toks = shlex.split(cmd, comments=False)
except Exception:
    print("0\t"); sys.exit(0)
val_opts = {"-c", "--git-dir", "--work-tree", "--namespace",
            "--super-prefix", "--exec-path"}
# a leading `cd <path> &&` sets the base cwd the rest of the command runs in
base = toks[1] if len(toks) >= 2 and toks[0] == "cd" else ""
i = 0
while i < len(toks):
    if toks[i] == "git":
        j = i + 1
        gitc = ""
        while j < len(toks):
            t = toks[j]
            if t == "-C" and j + 1 < len(toks):
                gitc = toks[j + 1]; j += 2
            elif t in val_opts:
                j += 2
            elif t.startswith("-"):
                j += 1
            else:
                break
        if j < len(toks) and toks[j] == "commit":
            if gitc and os.path.isabs(gitc):
                target = gitc
            elif gitc and base:
                target = os.path.join(base, gitc)
            else:
                target = gitc or base
            print("1\t" + target); sys.exit(0)
    i += 1
print("0\t")
' 2>/dev/null || true)"
cmd_is_commit="${parse%%$'\t'*}"
commit_dir="${parse#*$'\t'}"
[ "$cmd_is_commit" = "1" ] || exit 0

branch="$(git -C "${commit_dir:-.}" symbolic-ref --short HEAD 2>/dev/null || true)"
[ -z "$branch" ] && exit 0  # detached HEAD / not a repo -> fail open

case "$branch" in
  main | master)
    printf 'refuse: commit on protected branch "%s". Branch first (git switch -c <feature-branch>), then commit.\n' "$branch" >&2
    exit 2
    ;;
esac
exit 0
