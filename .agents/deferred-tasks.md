# Deferred-task ledger

Open follow-ups carried by a session. The deferred-task Stop gate
(`.claude/hooks/deferred-task-gate.sh`) blocks end-of-turn while any **unchecked**
item tagged `@agent` remains — work the agent can do itself. Policy:
[CLAUDE.md → "Deferred tasks gate completion"](../CLAUDE.md).

Tags (exactly one per item):
- `@agent` — the agent can execute/implement it now → **blocks** completion until done.
- `@human` — needs a person or a one-way door (push/publish/PR) → does not block.
- `@decision` — a product/science choice is pending → does not block.
- `@separate-lane` — belongs to another task's git lane → does not block.

To clear an `@agent` item: finish it and change `- [ ]` to `- [x]`. Only retag to a
non-blocking tag if it genuinely cannot be done by the agent now.

## Open

- [ ] @human Regenerate `analysis/burst_energies/burst_energies.{json,tex}` on a host with the CHIME/DSA `.npy` staged (iacobus / `data/{chime,dsa}/`) by running `python analysis/calculate_burst_energies.py`. The energy trust boundary now drops FAIL-gated joint fits (johndoeII, oran, whitney) and stamps `quality_flag`; in the energy table this removes oran + whitney (johndoeII is already skipped as a placeholder z=1.0). The committed artifacts predate the boundary (still list oran/whitney, no `quality_flag`) — the live config is `fluxcal`, so regeneration needs the data, which is not available in CI.

## Done

_(archive completed items here, or prune as the ledger grows)_
