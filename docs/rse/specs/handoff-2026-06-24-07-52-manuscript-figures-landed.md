# Handoff: CHIME–DSA manuscript figures landed + multi-agent worktree hygiene

---
**Date:** 2026-06-24 07:52 (PDT)
**Author:** AI Assistant
**Status:** Handoff
**Branch:** work landed on `main`; this doc authored from worktree `flits-iso` (detached at `origin/main`)
**Commit:** `origin/main` @ `7d62c0c`

---

## Task(s)

Took over the h17 manuscript-figure work (per `outputs/handoff_manuscript-figures_2026-06-24/HANDOFF.md`) and drove it to merged + stable, then resolved a concurrent-agent collision that surfaced along the way.

| Task | Status | Notes |
|------|--------|-------|
| Land the `analysis/` figure pipeline + CHIME-side DM 12-panel grid | ✅ Complete | PR #45 (`bad0ba4`) |
| Land the complementary figure→Overleaf delivery tool | ✅ Complete | PR #46 (`aa1c41d`) — `tools/sync_figures.py` |
| Stop concurrent agents cross-blocking each other's Stop-gates | ✅ Complete | PR #48 (`7d62c0c`) — one-agent-per-worktree rule in `CLAUDE.md` |
| Stand up an isolated worktree for parallel agents | ✅ Complete | `~/Developer/scratch/worktrees/flits-iso` |

**Current Workflow Phase:** Validate → Complete (session closing). No open implementation task remains for *this* lane.

## Workflow Artifacts

No new `research-*/plan-*/experiment-*` docs were produced this session — the work was figure/manuscript infrastructure, not the research-design cycle.

**Prior handoff (read for upstream context):**
- `outputs/handoff_manuscript-figures_2026-06-24/HANDOFF.md` — the h17-over-SSH brief this session executed (now committed via PR #45).

## Critical References

- `analysis/README.md` — the add/update/view convention for the manifest-driven figure system (read first).
- `analysis/build_manuscript.py` + `analysis/chime_dm/plot_dm_grid.py` — the figure pipeline and the CHIME-side DM grid generator.
- `tools/sync_figures.py` — ships manuscript-bound figures into the external Overleaf `Faber2026` (driven by `results/figures.manifest.json`; dry-run by default, `--apply` to copy).
- `CLAUDE.md` → "Concurrent agents: one agent per working tree" — the rule that prevents the Stop-gate cross-block (added this session).

## Recent Changes

All merged to `origin/main` (squash):
- PR #45 `bad0ba4` — `analysis/` manifest-driven figure pipeline + `analysis/chime_dm/` (grid SVG, manifest, review, `chime_dm.tex`), `analysis/build_manuscript.py`, `analysis/README.md`, `burst_energies` `manuscript_order`.
- PR #46 `aa1c41d` — `tools/sync_figures.py`, `results/figures.manifest.json`, `results/figures.review.json`.
- PR #48 `7d62c0c` — `CLAUDE.md` one-agent-per-worktree rule (`docs(agents)`).

## Reproducibility & Data State

- **Compute/data host:** remote `h17` (repo at `/home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS`). SSH via the **`h17`** alias — `lxd110h17` (used in the prior handoff) **times out**. Env on h17 is conda **`casa6`**, not `flits`; prefix non-interactive commands with `source /opt/miniforge/etc/profile.d/conda.sh && conda activate casa6 && cd <repo>`.
- **Figure regen:** `python analysis/chime_dm/plot_dm_grid.py` → `tsmooth=12, 8/12 constrain` (reproduced this session). Reads bridge artifacts on h17: `…/chime-dsa-codetections/results/chime_dm_grid_fits.json` + `chime_dm_grid_waterfalls.npz` (gitignored/external; `CHIME_DM_DATA` default points there).
- **Figure outputs:** PNG/PDF are **gitignored**; only the vector **SVG** is tracked. Re-rendering the SVG produces byte-different (matplotlib metadata) but visually identical output — don't re-commit it over the reviewed one.
- **Result:** 8/12 bursts constrain the CHIME-side DM (`|dDM| ≤ 0.84`, within 1 pc/cm³ floor); 4/12 honest non-detections. This 8/12 state is already on `main` via the #41 squash (its title still says "5 constrain" — stale title, the merged tree is 8/12).

## Verification State / Known-Broken

- **CI:** all three PRs (#45/#46/#48) merged **green** (Python 3.10/3.12, Socket Security, review). `verify-gate` recorded for the `CLAUDE.md` edit (cross-check vs hook source).
- **Figure review:** `analysis/chime_dm/figures.review.json` verdict `match` (8 green constrained, 4 grey non-detections); independently re-read this session.
- **Uncommitted/unpushed:** none of *mine*. `flits-iso` shows a dirty `docs/entire-tracing-checkpoints.md` — that's the Entire checkpoint tool auto-appending, not session content; ignore.
- **Separate active lane (DO NOT TOUCH):** a concurrent agent owns the **main checkout**, rapidly flipping its branch (`feat/figure-sync`→`chore/fork-b-takeover`→`feat/figure-vector`) and has **PR #47 open**. Also a separate worktree `~/Developer/scratch/worktrees/flits-gate` on `feat/joint-fit-gate` (19 ahead, no PR). Both preserved untouched.

## Learnings

- **Never push a pre-squash branch as a new PR.** The h17 `feat/custom-dm-tool` (PR #41 head) was squash-merged + deleted; pushing it would have **reverted ~4100 lines** across merged PRs #42/#43/#44/#40. The fix: cherry-pick only the genuinely-new commits onto a fresh branch off current `origin/main` (the downsampling commit came up **empty** on cherry-pick — already in the #41 squash).
- **PR #41's squash captured the 8/12 downsampling** despite its "5 constrain" title — verify merged *content*, not the squash title.
- **Two figure subsystems coexist, not duplicates:** `analysis/` = in-repo authoring/assembly; `tools/sync_figures.py` = delivery to external Overleaf.
- **Multi-agent shared checkout → Stop-gate cross-blocking.** The figure-review and deferred-task `Stop` hooks scan `CLAUDE_PROJECT_DIR:-$PWD` (verified: `.claude/hooks/figure-review-gate.sh:13,23`, `deferred-task-gate.sh:14,15,18`), so two agents in one checkout block each other on each other's in-progress work. Fix is filesystem isolation (one worktree per agent), now documented in `CLAUDE.md`.

## Action Items & Next Steps

1. [ ] **(Another agent's lane — not yours)** Triage/merge PR #47 on `feat/figure-vector`. Leave it alone unless explicitly handed that lane.
2. [ ] **(Decision pending)** `.agents/deferred-tasks.md` carries a `@decision` item: manuscript energies-table reconciliation (blocked on a TNS-naming/sample call). Does not block; needs a human/science decision.
3. [ ] **Future parallel agents:** launch from a dedicated worktree (`git worktree add ~/Developer/scratch/worktrees/flits-<lane> -b <branch> origin/main`), never the shared main checkout. `flits-iso` is clean and reusable.

**Recommended Next Skill:** `ai-research-workflows:validating-implementations` — if a session resumes the in-flight figure work (PR #47) or reconciles the two figure subsystems. The work *this* session owned is complete and merged; no further action required on it.

## Other Notes

- This handoff was authored from `flits-iso` (off `origin/main`), not the shared main checkout, to avoid colliding with the live concurrent agent — itself an instance of the one-agent-per-worktree rule.
- Memory written: `h17-ssh-alias` (SSH alias + `casa6` env), indexed in the project `MEMORY.md`.

---

**Handoff created by AI Assistant on 2026-06-24**
