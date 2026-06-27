# Entire CLI auth — status note (tooling, not the FLITS pipeline)

`STATUS: tooling note — session-tracing setup on hpcc; unrelated to the science.`

**Installed + enabled.** `entire` v0.7.7 at `~/.local/bin/` (official linux/amd64 release, SHA verified; upgraded from v0.7.5 on 2026-06-21 — see token-persistence note for why). `entire enable` run in `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS`: `.entire/settings.json` + git hooks present; agents Claude Code, Codex, et al.

**Sync policy.** Mirror the durable handoff context in tracked markdown; keep `.entire/` runtime artifacts and per-session `entire/<sha>` branches local to the host.

**Selective checkpoints.** Use `scripts/entire_checkpoint.py` to append a compact host-local snapshot to `docs/entire-tracing-checkpoints.md`, then commit that ledger when the snapshot is worth preserving for future agents.

**Automation.** The repo installs `.githooks/` through `core.hooksPath`. Hooks append the compact checkpoint ledger on relevant commits, merges, and rewrites, but they must not create commits themselves. **Agents:** always commit and push `docs/entire-tracing-checkpoints.md` at session closeout (see CLAUDE.md → Entire tracing ledger); use `--no-verify` on checkpoint-only commits to avoid a hook loop.

**Token persistence — root cause + fix (corrected 2026-06-21; the earlier "SOLVED via env" note was wrong).** Login nodes have no usable Secret Service / D-Bus keyring, so the default keyring token save fails (`failed to unlock collection 'login'`). A file-backed token store is forced via `~/.bashrc`:
```
export ENTIRE_TOKEN_STORE=file
export ENTIRE_TOKEN_STORE_PATH="$HOME/.config/entire/token.json"
```
**That env is necessary but was NOT sufficient on v0.7.5.** v0.7.5's `entire login` saved the token straight to the keyring regardless of the env — only `entire auth status` *reads* honored the file store — so login kept erroring with the keyring failure even though `auth status` looked fine (which is what the old note mistook for "solved"). Confirmed by reading the CLI source:
`internal/entireclient/tokenstore/tokenstore.go` `resolveBackendLocked()` routes BOTH read and write (`Get`/`Set`) through the file store when `ENTIRE_TOKEN_STORE=file`, and `login` uses that `Set`; v0.7.5 predated this and hardcoded the keyring save.

**The real fix is to upgrade to ≥0.7.7** (`curl -fsSL https://entire.io/install.sh | bash` — user-space `~/.local/bin`, checksum-verified), after which `entire login` writes `token.json` directly. Verified 2026-06-21: upgraded 0.7.5→0.7.7, ran `entire login` (headless device-code flow), `token.json` rewritten at login time, `entire auth status` → logged in. Keep the two `~/.bashrc` exports — the git hooks (post-commit, pre-push) read the same file store; without it session sync silently no-ops.

**Routine re-login** (token expiry): in a shell with the env active,
`cd ~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS && entire login`. Headless → device-code: open the printed `us.auth.entire.io/cli/auth?user_code=…` URL in a browser and approve, then `entire auth status`.

**Obsolete fallback (pre-0.7.7 only):** running login inside `dbus-run-session -- bash -lc 'gnome-keyring-daemon --unlock …; entire login'` was the old workaround when login forced the keyring. With ≥0.7.7 + the file-store env it is unnecessary — the keyring is never touched. `dbus-run-session` and `gnome-keyring-daemon` do exist on the login nodes if ever needed.
