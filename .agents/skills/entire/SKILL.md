---
name: entire
description: Recall prior session/checkpoint context in this repo via the Entire CLI. Use when the user asks about previous work, past prompts, earlier commits/sessions, "what did we do last time", or "find where we…". Read-only; the commands are allowlisted so they don't prompt.
---

# Entire

Entire tracks agent sessions as checkpoints (git ref `entire/checkpoints/v1`). Use it to answer "what happened before" instead of guessing from git log alone. All commands below are read-only and allowlisted in `.Codex/settings.json`.

- `entire search "<query>" --json` — hybrid semantic+keyword search over checkpoints, commits, sessions (needs `entire login`). Scoped to this repo; `--all-repos` to widen. For nontrivial searches, dispatch the `entire-search` subagent instead of running this inline.
- `entire recap` — summarize recent checkpoint activity.
- `entire activity` — activity overview.
- `entire checkpoint list|explain|search` — inspect specific checkpoints.
- `entire status` — is tracking on, current session.

Skip Entire for normal `git log`/`git blame` questions — reach for it only when the answer lives in *session/prompt* history, not the commit tree.
