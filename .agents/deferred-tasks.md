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

_(none)_

## Done

- [x] Regenerate `analysis/burst_energies/burst_energies.{json,tex}` with the energy trust boundary (#39). **Done 2026-06-24** on jakob-mbp using the local arc replica `~/Developer/dsa110-local-data/DSA_bursts/` (24 cubes; the "arc mount" reachable from here — `DATA_SOURCES.md` local replica), staged under `data/{dsa,chime}/`. Ran `python analysis/calculate_burst_energies.py` in the `flits` env. `--check` PASS; the quality gate refused the 3 FAIL joint fits (johndoeii, oran, whitney). Result: 8→6-burst energy table (removed oran+whitney; johndoeii also placeholder z=1.0), `quality_flag` now stamped on every row (all MARGINAL), all calibrated, E_iso 4.6e38–1.1e41 erg. Artifacts now show as `M` in the working tree (tracked); commit/push left to the user per the push gate.
