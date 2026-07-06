# Handoff: Migrate frb-foreground-halos into dsa110-FLITS (successor of the just-merged los_halos VO layer)

---
**Date:** 2026-07-05 14:38
**Author:** AI Assistant
**Status:** Handoff
**Branch:** `main` (dsa110-FLITS)
**Commit:** `030c159c` — "feat(foreground): integrate los_halos VO-TAP pipeline as galaxies/foreground/vo (#122)"

---

## Task(s)

| Task | Status | Notes |
|------|--------|-------|
| Migrate `los_halos` → `galaxies/foreground/vo/` | ✅ Complete | PR #122 squash-merged to main (`030c159c`); source repo archived (local `archived/los_halos` + GitHub `isArchived: true`) |
| Migrate `frb-foreground-halos` → dsa110-FLITS | 📋 Planned | **This handoff.** User-approved scope: integrate into `galaxies/foreground/`, reconcile with the `vo/` port, then archive |
| Archive `frb-foreground-halos` + `subhalos` | 📋 Planned | After integration verified. Follow the los_halos archive recipe (below) |

**Current Workflow Phase:** Research (survey done) → next is Plan

## Why this migration, and the key tension to resolve

`frb-foreground-halos` (github.com/jakobtfaber/frb-foreground-halos, local clone at
`~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/`) is — per its own README —
"a clean consolidation of the useful ideas from the older `subhalos` and `los_halos`
repositories": subhalos' reusable architecture (Sightline domain type, catalog query
*planning* + run-plan execution, candidate deduplication, UCD role inference) plus
los_halos' provenance/schema/reduction rules. It is **newer (June 2026) and more evolved
than the los_halos code that was just ported** to `galaxies/foreground/vo/` (Aug 2025
lineage). The `vo/` port went first only because the ffh repo wasn't surveyed until after
PR #122 merged.

**The central design question for the Plan phase:** `galaxies/foreground/vo/` and
`frb_foreground_halos` overlap heavily (both do discover → query → normalize → reduce over
VO TAP with the same schema philosophy; ffh's `normalize.py`/`reduce.py`/`provenance.py`
descend from the same los_halos code). Do NOT keep two parallel VO layers. Options, roughly
in order of prior plausibility:

1. **Replace-and-absorb (likely right):** port `frb_foreground_halos` as the new
   `galaxies/foreground/vo/` (or `galaxies/foreground/ffh/` then rename), migrating the
   few things the current `vo/` has that ffh lacks (check: `xmatch.py` CDS helpers,
   `find_columns`, the `flits-halos` CLI surface incl. `run-targets`, the
   `test_frb_recovery.py` zach/whitney/isha validation targets — these validation tests are
   telescope-project-specific and MUST survive whatever the merged shape is).
2. **Extend:** keep `vo/` API, graft ffh's unique modules (catalog_plan, dedupe, domain,
   classify, halos, io, plotting, discovery improvements) alongside. Risk: two normalize/
   reduce lineages drift.
3. Diff first, decide per-module. `vo/` modules were adversarially verified equivalent to
   los_halos originals, and ffh consolidated those same originals — so a module-level diff
   of ffh vs `vo/` is the cheapest way to see what ffh actually improved.

## Critical References (read first)

1. `galaxies/foreground/vo/README.md` — the just-merged VO layer: pipeline, schema, CLI,
   and the migration-notes pattern to replicate (what-was-not-ported inventory).
2. `~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/README.md` + `CLAUDE.md`
   (6.0K, repo-specific agent instructions) + `AGENTS.md` (untracked!) — the source repo's
   own docs.
3. `~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/src/frb_foreground_halos/`
   — 15 modules: `catalog_plan.py`, `classify.py`, `cli.py` (click-based), `dedupe.py`,
   `discovery.py`, `domain.py`, `halos.py`, `io.py`, `normalize.py`, `plotting.py`,
   `provenance.py`, `query.py`, `reduce.py`, `py.typed`, `__init__.py`.
4. dsa110-FLITS `CLAUDE.md` — binding conventions: ponytail (lazy-minimalist), post-edit
   autoformatter caveat, protected-branch commit guard, deferred-task + figure-review Stop
   gates, fit-validation contract.

## Source-repo state (frb-foreground-halos @ `7e2e964`)

- Tracked tree clean; last commit "Catalog query planning, run-plan, and candidate
  deduplication (#1)" 2026-06-05.
- **Untracked (separate-lane, decide during migration):** `AGENTS.md`,
  `docs/reviews/*-2026-06-05.{stdout,stderr}.log` + `codex-scientific-review-2026-06-05.json`
  (agent review artifacts — likely worth preserving in the migration notes or archiving
  as-is), `gurobi.log` (junk), `results` (symlink → `~/Data/frb-foreground-halos/results`,
  correct per the ~/Data convention — keep the physical data in `~/Data`, repoint consumers).
- `_quarantine/` holds `halos_2.py` + README (already-triaged debris; do not port).
- Deps: astropy, click, numpy, pandas, pyvo, PyYAML, scipy, **tenacity** (again NOT in the
  flits conda env — same substitution as last time: local `_with_retries` in
  `galaxies/foreground/vo/query.py:57`), click → decide argparse rewrite (FLITS convention;
  `flits-halos` is argparse) or add click. Ponytail says argparse.
- Packaging: PEP 621 src-layout, `py.typed`, `uv.lock`, pytest+coverage+timeout dev extras.

## Verification State / Known-Broken

- **ffh tests:** 55/55 pass via `uv run --frozen --extra dev python -m pytest tests -q`
  (verified 2026-07-05, took ~14 s). They do NOT run in the flits conda env (package + click/
  tenacity not installed there) — the port must make them pass in the flits env.
- **dsa110-FLITS:** main at `030c159c`; `galaxies/foreground/vo/` 44 offline tests green +
  2 live-network smoke green; 151 pre-existing `galaxies/foreground` tests green (all in
  flits conda env, agent-safe invocation:
  `env -i HOME="$HOME" PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin" /opt/anaconda3/bin/conda run -n flits python -m pytest …`).
- **Uncommitted in FLITS:** this handoff file itself (commit it as the first commit on the
  `feat/ffh-integration` branch — the protected-branch guard blocks committing on main), and
  `docs/entire-tracing-checkpoints.md` (auto-appending
  tracing ledger, re-dirtied by the post-merge checkpoint hook; protected-branch guard
  blocks committing it on main — sweep it into the next feature branch, documented pattern).
- **subhalos:** local clone clean @ `d830c0d0` (Aug 2025); GitHub NOT archived yet
  (`isArchived: false`). Nothing to port from it directly (ffh already consolidated it) —
  verify that claim during the diff pass before archiving.

## Learnings (gotchas that cost time last round — do not rediscover)

1. **`galaxies/` is a namespace dir (no `__init__.py`)** → pytest imports colocated test
   packages as `foreground.vo.*`, so string-based `mock.patch("galaxies.foreground…")`
   patches a *twin* module object and silently doesn't take. Patch by module object
   (`monkeypatch.setattr(mod, …)`) — see `galaxies/foreground/vo/test_query.py:43`.
2. **Agent-safe conda:** bare `conda run -n flits` can resolve base Anaconda python
   (PATH inheritance); always use the `env -i …` form above.
3. **Post-edit autoformatter** in dsa110-FLITS strips imports unused at edit time — land
   imports in the same edit as their first consumer.
4. **Protected-branch guard** blocks `git commit` on main — branch first
   (`git switch -c feat/ffh-integration`). Pathspec-only commits; the entire-tracing ledger
   gets a separate `--no-verify` tail commit (see `48ff0e4` pattern).
5. **tenacity is not in the flits env** even though `pydantic`/`typer`/`click` may appear
   installed elsewhere — check with the env-i probe before assuming any dep.
6. **Verify-gate Stop hook** requires per-path verification records
   (`verify-gate record --paths … --method test|adversarial-review|… --check … --evidence …`).
   The los_halos round used: test-method for suite-covered files + one adversarial-review
   subagent diffing port vs original (it found 2 real minors — worth repeating for ffh).
7. **los_halos archive recipe** (worked cleanly): archive-notice README → local commit →
   `mv` repo to `~/Developer/repos/github.com/jakobtfaber/archived/` → (gated, user-approved)
   `git push` + `gh repo archive <owner>/<repo> --yes` → check off in
   `.agents/deferred-tasks.md`. Pushes/archives are one-way: batch them into a single
   user-approval ask.
8. FLITS mkdocs (`mkdocs.yml`, `docs_dir: docs-analysis`) is a curated analysis narrative —
   don't wire new package docs into it; a colocated README suffices.

## Action Items & Next Steps

1. [ ] Read the three Critical References; then diff `frb_foreground_halos` modules against
   `galaxies/foreground/vo/` equivalents (normalize/reduce/provenance/query descend from the
   same ancestors — catalog_plan/dedupe/domain/classify/halos/io/plotting are net-new).
2. [ ] Produce a plan (`ai-research-workflows:planning-implementations`) deciding
   replace-vs-extend for `vo/` (option 1 above is the prior), the CLI shape (argparse,
   preserve `flits-halos` subcommands incl. `run-targets`), dep substitutions
   (tenacity→local retry; click→argparse), and test-layout (colocated, markers registered
   via conftest `pytest_configure` — `--strict-markers` is on).
3. [ ] Implement on `feat/ffh-integration` branch; port ffh's 55 tests + keep vo's
   validation tests (zach/whitney/isha recovery) green in the flits conda env.
4. [ ] Verify: offline suite + `pytest galaxies/foreground` regression (151+44 baseline) +
   ruff + one live-network smoke + editable reinstall if entry points change + adversarial
   review subagent (port vs ffh originals) + verify-gate records + agent-closeout-check
   (runtime packet if pyproject touched).
5. [ ] Update `DATA_LOCATIONS.md` related-repos lines for frb-foreground-halos + subhalos;
   keep physical data at `~/Data/frb-foreground-halos/` (symlink pattern).
6. [ ] Commit (pathspec) + ledger tail commit; PR; on user approval merge.
7. [ ] Archive `frb-foreground-halos` AND `subhalos` per the recipe in Learnings #7
   (subhalos needs no code migration — ffh consolidated it; confirm via the diff pass, then
   staleness-proof: clean tree, no open PRs, no unmerged unique delta).
8. [ ] Check off the corresponding `.agents/deferred-tasks.md` items (add them at branch
   time, tags: work `@agent`, pushes/archives `@human`).

**Recommended Next Skill:** `ai-research-workflows:planning-implementations` (survey is
done — this document is the research artifact; go straight to the plan).

## Other Notes

- FLITS science context: this is the "Sightline Attribution" long-view goal (CLAUDE.md) —
  DM/scattering budgets across the 12 co-detected sightlines, 49 candidate intervening
  systems. The curated engines (`galaxies/foreground/engines*.py`, census registry) remain
  the primary science surface; ffh/vo is the wide-net VO discovery layer feeding it.
- Remaining archive-not-merge candidates surveyed 2026-07-05 (user informed, no action
  authorized): `dsa110-scat` (superseded, 1.8G), `FLITS_GBT` (dormant 2021 fork),
  `frb_analysis` (empty). `FLITS` (2021 SPANDAK+/BL, 1.0G) stays separate — different
  instrument/era; its scint/scat artifacts were already recovered June 2026.
- los_halos precedent artifacts: PR #122, `galaxies/foreground/vo/README.md` migration
  notes, adversarial-review verdict (PASS; 2 minors fixed in `e2fe029`).

---

**Handoff created by AI Assistant on 2026-07-05**
