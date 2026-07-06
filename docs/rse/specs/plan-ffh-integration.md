# Implementation Plan: Integrate frb-foreground-halos into galaxies/foreground/vo (union-merge)

---
**Date:** 2026-07-05
**Author:** AI Assistant
**Status:** Draft
**Related Documents:**
- [Handoff: frb-foreground-halos migration](handoff-2026-07-05-14-38-frb-foreground-halos-migration.md) *(research artifact for this plan)*

---

## Overview

`frb-foreground-halos` (ffh, github.com/jakobtfaber/frb-foreground-halos @ `7e2e964`, local clone
`~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/`) is the June-2026 consolidation of
`subhalos` + `los_halos`. It post-dates and supersedes the los_halos lineage that PR #122 just
ported to `galaxies/foreground/vo/`. This plan merges ffh into `vo/` so exactly **one** VO layer
remains, then clears the way to archive `frb-foreground-halos` and `subhalos`.

The module-level diff (executed during plan research, 2026-07-05) resolved the handoff's central
replace-vs-extend question with a refinement: the two codebases are **complementary, not parallel**.
Only four modules share the los_halos lineage (`query`, `normalize`, `reduce`, provenance), and in
all four ffh is strictly ahead (distance-ordered ADQL truncation, truncation metadata, per-row
provenance via a dedicated module, keyword-only APIs, default Planck18). Everything else is
disjoint: `vo/` uniquely has RegTAP service discovery, cached table discovery with a z-value sanity
gate, async wall-time budgets, and CDS X-Match; ffh uniquely has halo-mass estimation (SHMR),
catalog query planning (Gurobi/greedy), host/foreground classification, cross-catalog dedupe, a
domain model, and plotting.

**Goal:** `galaxies/foreground/vo/` contains the union — ffh's evolved shared-lineage modules +
ffh's seven net-new modules + vo's unique discovery/xmatch layers — under one argparse `flits-halos`
CLI, with ffh's 55 tests and vo's 44-offline/2-network tests all green in the `flits` conda env, so
both source repos can be archived.

**Motivation:** Two diverged normalize/reduce lineages would rot (handoff risk #2). ffh's science
additions (halo masses → b/R_vir intersection flags → per-sightline summaries) are exactly what the
Sightline-Attribution goal (CLAUDE.md) needs: DM/scattering budgets across the 12 co-detected
sightlines and 49 candidate intervening systems.

## Current State Analysis

### Blocking pre-existing condition: stale drift in the main checkout (Phase 0)

Six `vo/` files in the main checkout are dirty with content that *reverts* PR #122's merged state
(verified 2026-07-05 ~15:15):

- `galaxies/foreground/vo/cli.py` — `run-targets` subparser removed (function `cmd_run_targets` remains)
- `galaxies/foreground/vo/discover.py` — `^redshift$` dropped from `PROBABLE_Z_NAME_PATTERNS`
- `galaxies/foreground/vo/reduce.py` — `strict=True` dropped from a `zip`
- `galaxies/foreground/vo/test_cli.py` — used `Path` import removed (NameError at runtime)
- `galaxies/foreground/vo/test_discover.py`, `test_query.py` — module-object `monkeypatch` reverted
  to string-form `@patch` **without importing `patch`** → collection `NameError`

`pytest galaxies/foreground/vo -m "not network"` on the dirty tree: **2 collection errors** (verified).
`lane-liveness` verdict: **live** (editor lock, unresolved owner) — consistent with an editor holding
pre-merge buffers that overwrote merged files on save. Per the separate-lane protocol this drift is
**preserved, decision pending**: Phase 0 requires user confirmation before restoring to HEAD.

### The just-merged vo/ layer (baseline = HEAD `030c159c`)

- `galaxies/foreground/vo/registry.py:124-222` — `discover_tap_services`: RegTAP ADQL + pyvo-registry
  fallback + anchor merge; offline-degrading. **vo-unique, keep.**
- `galaxies/foreground/vo/discover.py:290-376` — `discover_tables`: grouped TAP_SCHEMA prefilter,
  IN-list chunking (`_IN_LIST_CHUNK = 50`, line 52), sampled z-value sanity gate (`_z_value_stats`,
  lines 176-193), parquet cache keyed by service hash; `find_columns` (lines 211-258). **vo-unique, keep.**
- `galaxies/foreground/vo/query.py` — `quote_table` (semantically identical to ffh's),
  `build_cone_adql` (**behind ffh**: fixed `TOP 10000`, no distance ordering), `_with_retries`
  (lines 57-65, the tenacity substitution), `query_sync` (68-86), `cone_query` returning a bare
  DataFrame with provenance in `df.attrs` (89-121), `safe_search` async wall-time budget (124-189).
- `galaxies/foreground/vo/normalize.py` — `SCHEMA_COLUMNS` (identical 13-column contract to ffh),
  `ColumnMapping`, `normalize` (whole-frame provenance), `to_common_schema` (90-122, positional args).
- `galaxies/foreground/vo/reduce.py` — `merge_and_rank(df, frb_ra_deg, frb_dec_deg, cosmo)`
  positional-cosmo variant; `compute_rdelta_from_mdelta` (ffh name: `rdelta_from_mdelta`).
- `galaxies/foreground/vo/utils.py` — `set_tap_timeout` (10-30, **keep**, pyvo has no default
  timeout), `make_provenance` (33-44, superseded by ffh `provenance.py`).
- `galaxies/foreground/vo/targets.py` — `Target` dataclass + `load_targets` (YAML key `z_host`) +
  `get_cosmology`. Superseded by ffh `domain.Sightline` + `io.read_targets_yaml` (YAML key `redshift`).
- `galaxies/foreground/vo/xmatch.py` — CDS X-Match helpers. **vo-unique, keep.**
- `galaxies/foreground/vo/cli.py` — argparse `flits-halos`: `services tables cone run-targets
  discover query reduce` (cache-based pipeline). Entry point `pyproject.toml:62`.
- `galaxies/foreground/vo/test_frb_recovery.py` — zach/whitney/isha validation (**must survive**);
  uses `Target(z_host=…)` (lines 22-46) and `merge_and_rank(…, frb.ra, frb.dec, get_cosmology())`
  (line 107, positional cosmo).
- Tests: 44 offline + 2 network at HEAD; markers registered in `conftest.py:pytest_configure`
  (unit/integration/network) because root `pyproject.toml` sets `--strict-markers`.

### The ffh package (source of the port)

15 modules under `src/frb_foreground_halos/`; 55 tests under `tests/` (all green via
`uv run --frozen --extra dev python -m pytest tests -q`, verified 2026-07-05):

- `query.py` — `build_cone_adql(…, *, columns, top, order_by_distance=True)` emits
  `DISTANCE(…) AS sep_deg … ORDER BY sep_deg` so `TOP {maxrec}` keeps the *closest* rows
  (completeness guarantee, lines 44-64); `cone_query` returns `(df, metadata)` with a `truncated`
  flag (102-148); `run_tap_sync` uses **tenacity** (line 92, must be substituted).
- `normalize.py` — `normalize(df, mapping, *, service, table, adql=None, extra_provenance=None)`
  with per-mapping provenance via `provenance.make_provenance` and z_prior rewrite for photo rows
  (111-128); `to_common_schema` keyword-only wrapper (131-156); `infer_z_type` also matches `zsp`.
- `reduce.py` — `merge_and_rank(df, *, frb_ra_deg, frb_dec_deg, cosmo=None)` defaulting Planck18;
  `rdelta_from_mdelta`.
- `provenance.py` — `make_provenance(adql, *, service, table, column_mapping=None, extra=None)`,
  `parse_provenance`, `utc_now_iso`.
- `halos.py` — `HaloAdapter` (M* direct / K-band Bell+03 / Moster+13 SHMR inversion via `brentq`),
  per-row `mass_provenance`, `ADAPTERS` for glade2 / glade-plus (f_dL z-provenance fix) /
  hecate-v2 / two-mrs. Needs `scipy` (present in flits env).
- `catalog_plan.py` — `CatalogProfile`, `max_cone_radius_arcmin`, `plan_queries` with Gurobi MIP +
  greedy fallback (`gurobipy` import guarded, lines 419-435; **gurobipy absent from flits env →
  greedy fallback exercises automatically**; no test references gurobi).
- `classify.py` — `add_intersection_flags` (b/R_vir strict/nominal/loose), `summarize_sightlines`
  (foreground-only counts), `empty_sightline_row`.
- `dedupe.py` — spatial-bucketed union-find dedupe across catalogs (30″ + Δz≤0.01), spec-z-preferring
  representative, `catalog_sources` provenance.
- `domain.py` — frozen `Sightline`, `ForegroundCandidate`, `QueryResult`.
- `io.py` — `read_table`/`write_table` (suffix dispatch), `read_targets_yaml` → `list[Sightline]`
  with `normalize_host_redshift` placeholder-z guard (z ≥ 0.999 or `host_z_placeholder: true` → None;
  encodes the AGENTS.md fact that placeholder hosts must not classify every z<1 galaxy as foreground).
- `plotting.py` — Agg-backend sky maps + summary; consumes normalized+reduced tables via
  `TargetColumns` join. Needs matplotlib only (no seaborn import; present in flits env).
- `cli.py` — **click** group (must be rewritten to argparse): `discover tables cone reduce
  plot-candidates plan-queries run-plan dedupe-candidates run-catalog` + helpers
  `_load_target_records`, `_merge_mass_provenance`, `_load_query_plan`, `_process_catalog_query`,
  `_classify_foreground_candidates` (host/foreground flags with spec-z vs photo-z Δz tolerances),
  `_finalize_catalog_run` (summary status priority: no_candidates_in_cone / possibly_truncated /
  z_frb_unknown).
- `discovery.py` — static anchors + `choose_columns`/`is_probable_redshift`. **Subsumed** by vo
  `registry.py` + `discover.py` (vo's Z_EXCLUDE set is a superset incl. the `estsize` fix; anchors
  identical). Drop; fold its two tests into `test_discover.py` where not duplicative.
- Dependency deltas vs flits env (probed 2026-07-05): `tenacity` MISSING, `gurobipy` MISSING
  (fallback OK), `click` present-but-undeclared in FLITS pyproject (do not rely on it);
  scipy/matplotlib/yaml/requests/pyvo/seaborn all present. **No FLITS pyproject dependency changes
  needed.** Entry point `flits-halos` unchanged → no editable reinstall.
- ffh tests already patch by module object (e.g. `tests/test_cli.py:47`
  `monkeypatch.setattr(cli_module, "cone_query", …)`) — ports cleanly past the namespace-package
  patching gotcha.
- Untracked in ffh (separate-lane): `AGENTS.md` (2 durable facts — both already encoded in
  `halos.py` comments and `io.normalize_host_redshift`), `docs/reviews/*` agent-review artifacts,
  `gurobi.log` (junk), `results` symlink → `~/Data/frb-foreground-halos/results` (correct pattern).
  `_quarantine/halos_2.py` — triaged debris, do not port.

### Current Limitations

- Two normalize/reduce/query lineages one merge apart from drifting.
- vo's `TOP 10000` unordered truncation can silently drop the closest source — a correctness bug
  for impact-parameter work that ffh's distance-ordered ADQL fixes.
- vo has no halo-mass path, no host-exclusion, no dedupe, no per-sightline summary — the science
  surface stops at "ranked candidates".

## Desired End State

**New behavior:** one package `galaxies/foreground/vo/` where
`flits-halos` exposes the union CLI: vo's cache pipeline (`services tables cone run-targets
discover query reduce`) plus ffh's science pipeline (`rank plan-queries run-plan run-catalog
dedupe-candidates plot-candidates`). Library API: `cone_query` returns `(df, metadata)` with
`truncated`; `normalize`/`to_common_schema` produce per-mapping provenance with z_prior; halo
masses, intersection flags, dedupe, and plotting are importable from `galaxies.foreground.vo`.

**Success looks like:**
- `pytest galaxies/foreground/vo -m "not network"` in the flits conda env: **~95 tests pass, 0 fail**
  (44 vo + 55 ffh − a handful merged as duplicates), plus 2 network tests on demand.
- `pytest galaxies/foreground` (crossmatch/engines regression): 151 pre-existing tests still green.
- `flits-halos run-catalog <targets.yaml> glade-plus --radius-arcmin 10` works end-to-end offline-degrading.
- zach/whitney/isha recovery tests still pass with `-m network`.
- Both source repos archivable (Phase 10).

## What We're NOT Doing

- [ ] **Not porting ffh `discovery.py`** — subsumed by vo `registry.py`/`discover.py` (record in
      migration notes).
- [ ] **Not porting** `_quarantine/halos_2.py`, `gurobi.log`, `examples/headline_report.py`, or
      ffh docs (`phase5_gladep.md`, `catalog-query-planning.md`, `pre-run-checklist.md`) — the two
      contract docs' rules (`schema.md`, `migration.md`) are folded into the vo README; the rest
      stays retrievable in the archived repo and is listed in the not-ported inventory.
- [ ] **Not adding dependencies to FLITS pyproject** — tenacity→local retry, click→argparse,
      gurobipy stays an optional lazy import with greedy fallback.
- [ ] **Not keeping a `frb-foreground-halos` console script** — `flits-halos` is the single CLI.
- [ ] **Not wiring anything into mkdocs** (`docs_dir: docs-analysis` is a curated narrative) —
      colocated README only.
- [ ] **Not running a new science campaign** — no re-execution of the GLADE+ runs against the 12
      sightlines in this plan; physical data stays at `~/Data/frb-foreground-halos/` (symlink pattern).
- [ ] **Not migrating any `subhalos` code** — ffh already consolidated it; Phase 10 verifies that
      claim before archiving.
- [ ] **Not pushing, merging the PR, or archiving repos without explicit user approval** — one-way
      doors, batched into a single ask (Phase 9/10, `@human`).

**Rationale:** ponytail (shortest working diff, delete-over-add) + the handoff's user-approved scope.

## Implementation Approach

**Technical strategy:** union-merge inside the existing `galaxies/foreground/vo/` package (no
rename, no parallel `ffh/` package). For the four shared-lineage modules, adopt the ffh
implementation and graft the vo-unique capabilities into it. Port ffh's seven net-new modules
nearly verbatim (import-path changes + tenacity substitution only). Rewrite ffh's click CLI as
additional argparse subcommands on the existing `flits-halos` parser. Port ffh's tests file-by-file,
merging where filenames collide.

**Key architectural decisions:**

1. **Decision:** Union-merge into `vo/` (handoff option 1 refined by the option-3 diff).
   - **Rationale:** the diff proves ffh is ahead on the shared lineage but *lacks* RegTAP/caching/
     async-budget/xmatch — a pure replace would regress discovery; a pure extend would keep two
     normalize/reduce lineages.
   - **Trade-offs:** slightly larger one-time merge in `query.py`/`cli.py`; zero long-term duplication.
   - **Alternatives considered:** (a) replace `vo/` wholesale with ffh — loses registry/discover/
     xmatch/safe_search; (b) graft ffh beside vo untouched — leaves two normalize/reduce/query
     lineages, the exact drift risk the handoff forbids.

2. **Decision:** `cone_query` adopts ffh's `(df, metadata)` tuple return with `truncated`.
   - **Rationale:** truncation detection is load-bearing (`possibly_truncated` sightline status in
     `_finalize_catalog_run`); `df.attrs` provenance is fragile (silently dropped by many pandas ops).
   - **Trade-offs:** must update 5 vo callsites (cli ×3, test_query ×2, test_frb_recovery ×1).
   - **Alternatives considered:** keeping attrs-provenance and stuffing `truncated` into attrs —
     rejected for fragility and because ffh's ported tests pin the tuple contract.

3. **Decision:** `Sightline` (ffh `domain.py`) replaces `Target`; targets YAML accepts both
   `redshift` and legacy `z_host` keys; `targets.py` shrinks to a shim (`load_targets = read_targets_yaml`,
   `get_cosmology`).
   - **Rationale:** single domain type; `read_targets_yaml` carries the placeholder-z science guard
     that `Target` lacks; existing vo YAML/fixtures keep working via the alias.
   - **Trade-offs:** mechanical `z_host`→`redshift` attribute renames in conftest, test_targets,
     test_frb_recovery.
   - **Alternatives considered:** keeping both types — two lineages again; renaming YAML keys
     everywhere — churn in committed example files for no gain.

4. **Decision:** argparse for all new subcommands; ffh's `reduce` becomes `rank` (vo `reduce` =
   cache aggregation already exists); ffh's `discover`/`tables`/`cone` are dropped as redundant
   (vo `services`/`tables`/`cone` cover them; vo `cone` gains `--maxrec`/`--output`).
   - **Rationale:** FLITS convention + user-approved handoff scope ("argparse; preserve flits-halos
     subcommands incl. run-targets"); click is not a declared FLITS dependency.
   - **Trade-offs:** ffh's 10 CliRunner tests must be rewritten to the `cli.main([...])` + capsys
     pattern (`galaxies/foreground/vo/test_cli.py:14-28` is the model).
   - **Alternatives considered:** adding click as a dependency — violates convention for zero
     functional gain.

5. **Decision:** tenacity `@retry` on `run_tap_sync` → existing `_with_retries` (same 5 attempts /
   exponential 0.5→8 s semantics), keeping `set_tap_timeout` injection from vo.
   - **Rationale:** tenacity is not in the flits env (probed); precedent from the los_halos port
     (`query.py:57`).

**Patterns to follow:**
- Module-object patching in tests — `galaxies/foreground/vo/test_query.py:29-30` (HEAD version).
- Marker registration via `conftest.py:pytest_configure` (strict-markers on).
- Import lands in the same edit as its first consumer (post-edit autoformatter strips unused imports).
- Pathspec-only commits on a feature branch; ledger tail commit `--no-verify` (pattern `48ff0e4`).

## Implementation Phases

Phases 1-8 are sequential commits on `feat/ffh-integration`. Ports are executed test-first: copy
the ported test file in first, watch it fail (module missing / import error), then port the module,
watch it pass. `PY` below abbreviates the agent-safe runner:

```bash
PY="env -i HOME=$HOME PATH=/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin /opt/anaconda3/bin/conda run -n flits python"
FFH=~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/src/frb_foreground_halos
FFT=~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/tests
VO=galaxies/foreground/vo
```

### Phase 0: Resolve stale drift, branch, ledger entries

**Objective:** clean baseline at `030c159c`, feature branch, tracking in place.

**Tasks:**
- [ ] **[@decision — user gate]** Confirm the 6 dirty `vo/` files are editor-buffer drift, then close
      any editor tabs holding `galaxies/foreground/vo/*` and restore:
  ```bash
  git stash push -m "stale vo/ drift (pre-ffh-integration, provably reverts PR #122)" \
    -- galaxies/foreground/vo/
  ```
  (stash, not `git restore` — recoverable if the owner turns out to be real).
- [ ] **Verify baseline green:**
  ```bash
  $PY -m pytest galaxies/foreground/vo -m "not network" -q   # expect: 44 passed
  ```
- [ ] **Branch (protected-branch guard blocks main):**
  ```bash
  git switch -c feat/ffh-integration
  ```
- [ ] **Commit the handoff + this plan** (first branch commit, pathspec-only):
  ```bash
  git add docs/rse/specs/handoff-2026-07-05-14-38-frb-foreground-halos-migration.md \
          docs/rse/specs/plan-ffh-integration.md
  git commit -m "docs(rse): handoff + plan for frb-foreground-halos integration" \
    -- docs/rse/specs/
  ```
- [ ] **Add deferred-task items** to `.agents/deferred-tasks.md` under `## Open`:
  ```markdown
  - [ ] Integrate frb-foreground-halos into galaxies/foreground/vo per
        docs/rse/specs/plan-ffh-integration.md (Phases 1-8). @agent
  - [ ] Push feat/ffh-integration + open PR; on approval merge. @human
  - [ ] Archive frb-foreground-halos and subhalos per the los_halos recipe
        (after integration verified; batch the pushes/archives into one approval ask). @human
  - [ ] Sweep docs/entire-tracing-checkpoints.md into a --no-verify tail commit. @agent
  ```

**Verification:**
- [ ] `git status --short -- galaxies/foreground/vo/` → empty
- [ ] `git branch --show-current` → `feat/ffh-integration`

### Phase 1: Shared-lineage core — provenance, query, normalize, reduce

**Objective:** adopt ffh's evolved versions of the four common-ancestor modules, keeping vo's
unique query capabilities (`safe_search`, `set_tap_timeout`, local retries).

**Tasks:**
- [ ] **Port ffh test files in first (failing).** Copy `$FFT/test_query.py` content into
      `$VO/test_query.py` *merged after* the existing vo tests (rename ffh's duplicate-named tests
      with an `_ffh` suffix where they collide, e.g. `test_quote_table_variants_ffh`); rewrite
      imports to relative (`from .query import …`) and add `@pytest.mark.unit`. Same for
      `$FFT/test_normalize.py` → `$VO/test_normalize.py`, `$FFT/test_reduce.py` → `$VO/test_reduce.py`.
      ffh tests asserting the new contracts will fail against the old modules, e.g.:
  ```python
  @pytest.mark.unit
  def test_build_cone_adql_orders_by_distance():
      adql = build_cone_adql("t", "ra", "dec", 10.0, 0.0, 0.1, top=500)
      assert "AS sep_deg" in adql and adql.rstrip().endswith("ORDER BY sep_deg")
      assert "SELECT TOP 500" in adql

  @pytest.mark.unit
  def test_cone_query_returns_metadata_with_truncated(monkeypatch):
      from . import query as qmod
      monkeypatch.setattr(qmod, "run_tap_sync", lambda *a, **k: pd.DataFrame({"ra": [1.0]}))
      df, meta = cone_query("https://x.invalid/tap", "t", "ra", "dec", 1.0, 2.0, 0.1, maxrec=1)
      assert meta["truncated"] is True and "adql" in meta
  ```
- [ ] **Run, watch fail:** `$PY -m pytest $VO/test_query.py $VO/test_normalize.py $VO/test_reduce.py -q`
      → expect failures (old signatures).
- [ ] **Create `$VO/provenance.py`** = ffh `provenance.py` verbatim (no import changes needed).
- [ ] **Rewrite `$VO/query.py`** as the merge: ffh's `quote_table`, `build_cone_adql`
      (distance-ordered), `_table_to_dataframe`, `run_tap_sync`, `cone_query` (tuple return) — with
      the tenacity decorator replaced and vo's `safe_search` + `_with_retries` retained:
  ```python
  def run_tap_sync(access_url: str, adql: str, *, maxrec: int = 10000) -> pd.DataFrame:
      if TAPService is None:
          return pd.DataFrame()

      def _run() -> pd.DataFrame:
          service = TAPService(access_url)
          set_tap_timeout(service, timeout_seconds=10.0)
          result = service.run_sync(adql, MAXREC=maxrec)
          return _table_to_dataframe(result)

      return _with_retries(_run)
  ```
      Keep vo's `_with_retries` (lines 57-65 at HEAD) and `safe_search` (124-189) verbatim; import
      `set_tap_timeout` from `.utils`. Delete `query_sync` (replaced by `run_tap_sync`; keep the
      name as `query_sync = run_tap_sync` **only if** grep shows external consumers — none known).
- [ ] **Rewrite `$VO/normalize.py`** = ffh `normalize.py` verbatim (it already includes
      `to_common_schema` and z_prior handling; imports `from .provenance import make_provenance`).
- [ ] **Rewrite `$VO/reduce.py`** = ffh `reduce.py` verbatim. Add a back-compat alias only if the
      regression grep needs it: `compute_rdelta_from_mdelta = rdelta_from_mdelta` (grep first:
      `rtk grep -n "compute_rdelta_from_mdelta" -t py` — expected: only vo-internal).
- [ ] **Shrink `$VO/utils.py`** to `set_tap_timeout` only (delete `make_provenance`; its callers
      now use `.provenance`). Same-edit rule: update `query.py` imports in the same commit.
- [ ] **Update vo callsites of the changed signatures** (same commit):
  - `$VO/cli.py:48-53` `cmd_cone`: `df, meta = cone_query(...)`; print `meta["adql"]` after the head.
  - `$VO/cli.py:56-63` `cmd_run_targets`: `df, _ = cone_query(...)`.
  - `$VO/cli.py:91-112` `cmd_query`: `df, _ = cone_query(...)`.
  - `$VO/cli.py:115-146` `cmd_reduce`: `to_common_schema(df, ra_col=row["ra_col"],
    dec_col=row["dec_col"], z_col=row["z_col"], service=…, table=…)` (keyword-only now).
  - `$VO/test_frb_recovery.py:65-84` `_collect_candidates`: unpack tuple; keyword `to_common_schema`.
  - `$VO/test_frb_recovery.py:107,141,164`: `merge_and_rank(pd.concat(…), frb_ra_deg=frb.ra,
    frb_dec_deg=frb.dec, cosmo=get_cosmology())`.
  - `$VO/test_cli.py` fake `cone_query` lambdas → return `(fake_rows, {"adql": "SELECT …"})`.
- [ ] **Update `$VO/__init__.py`** exports in the same commit: add `make_provenance`,
      `parse_provenance`, drop nothing yet.
- [ ] **Run, watch pass:**
  ```bash
  $PY -m pytest $VO -m "not network" -q    # expect: 44 + ~7 ffh core tests, 0 fail
  ```
- [ ] **Commit:** `git commit -m "feat(vo): adopt ffh query/normalize/reduce/provenance core" -- galaxies/foreground/vo/`

**Verification:**
- [ ] `$PY -m pytest $VO -m "not network" -q` → 0 failures
- [ ] `$PY -c "from galaxies.foreground.vo import cone_query, to_common_schema, make_provenance"` → ok

### Phase 2: Domain + IO unification (Sightline replaces Target)

**Objective:** single domain type with the placeholder-z guard; legacy `z_host` YAML still loads.

**Tasks:**
- [ ] **Port tests first:** `$FFT/test_domain.py` → `$VO/test_domain.py`, `$FFT/test_io.py` →
      `$VO/test_io.py` (relative imports, unit markers). Add the failing alias test to `$VO/test_io.py`:
  ```python
  @pytest.mark.unit
  def test_read_targets_yaml_accepts_legacy_z_host_key(tmp_path):
      p = tmp_path / "t.yaml"
      p.write_text("targets:\n- {name: A, ra: 1.0, dec: 2.0, z_host: 0.25}\n")
      (s,) = read_targets_yaml(p)
      assert s.redshift == 0.25 and s.name == "A"
  ```
- [ ] **Run, watch fail** (modules absent): `$PY -m pytest $VO/test_domain.py $VO/test_io.py -q`
- [ ] **Create `$VO/domain.py`** = ffh `domain.py` verbatim.
- [ ] **Create `$VO/io.py`** = ffh `io.py` with the two-line alias inside `read_targets_yaml`'s loop:
  ```python
      for row in targets:
          if "z_host" in row and "redshift" not in row:
              row = {**row, "redshift": row["z_host"]}
          metadata = {k: v for k, v in row.items()
                      if k not in {"name", "ra", "dec", "redshift", "z_host"}}
  ```
- [ ] **Rewrite `$VO/targets.py`** as the shim (keeps `get_cosmology`, kills the second lineage):
  ```python
  """Target loading (legacy shim) and cosmology selection."""

  from __future__ import annotations

  from astropy.cosmology import Planck18
  from astropy.cosmology.core import Cosmology

  from .domain import Sightline
  from .io import read_targets_yaml as load_targets

  Target = Sightline  # legacy alias; z_host attribute is now .redshift

  def get_cosmology(name: str | None = None) -> Cosmology:
      if name is None or name == "Planck18":
          return Planck18
      raise ValueError(f"Unsupported cosmology: {name}")

  __all__ = ["Sightline", "Target", "get_cosmology", "load_targets"]
  ```
- [ ] **Update the `z_host=`/`.z_host` construction sites** (attribute kwarg is not aliased):
  `$VO/conftest.py:16-19` fixtures (`z_host=` → `redshift=`; keep the YAML fixture emitting
  `z_host` keys — it now exercises the alias), `$VO/test_targets.py`, `$VO/test_frb_recovery.py:24,33,42`.
- [ ] **Run, watch pass:** `$PY -m pytest $VO -m "not network" -q`
- [ ] **Commit:** `git commit -m "feat(vo): Sightline domain + io with placeholder-z guard; Target shim" -- galaxies/foreground/vo/`

**Dependencies:** Phase 1 (io tests use `read_table` round-trips only; domain standalone).

**Verification:**
- [ ] `$PY -c "from galaxies.foreground.vo.targets import Target, load_targets;
      s = load_targets('galaxies/foreground/vo/targets_example.yaml'); print(s[0].redshift)"` → prints, no error

### Phase 3: Net-new science modules — halos, classify, dedupe

**Objective:** the mass→R_vir→intersection science layer.

**Tasks:**
- [ ] **Port tests first:** `$FFT/test_halos.py` (14 tests) → `$VO/test_halos.py`,
      `$FFT/test_classify.py` (3) → `$VO/test_classify.py`, `$FFT/test_dedupe.py` (4) →
      `$VO/test_dedupe.py`. Relative imports + `@pytest.mark.unit`.
- [ ] **Run, watch fail:** `$PY -m pytest $VO/test_halos.py $VO/test_classify.py $VO/test_dedupe.py -q`
      → ModuleNotFoundError-driven failures.
- [ ] **Create `$VO/halos.py`** = ffh verbatim (imports: astropy/numpy/pandas/scipy only — no changes).
- [ ] **Create `$VO/classify.py`** = ffh verbatim.
- [ ] **Create `$VO/dedupe.py`** = ffh verbatim (`from .reduce import angular_separation_arcmin`
      resolves against the Phase-1 reduce).
- [ ] **Run, watch pass**, then **commit:**
      `git commit -m "feat(vo): halo-mass adapters (SHMR), intersection classify, cross-catalog dedupe" -- galaxies/foreground/vo/`

**Dependencies:** Phase 1 (`reduce.angular_separation_arcmin`, schema columns).

**Verification:**
- [ ] `$PY -m pytest $VO -m "not network" -q` → 0 failures
- [ ] Numerical spot-check (Moster+13 SHMR inversion is monotonic and self-consistent):
  ```bash
  $PY -c "from galaxies.foreground.vo.halos import mstar_to_mhalo; \
          m = mstar_to_mhalo(5e10, 0.05); assert 1e12 < m < 1e13, m; print(f'{m:.3e} OK')"
  ```

### Phase 4: Catalog query planning

**Objective:** `plan_queries` (Gurobi-with-greedy-fallback) available; greedy path exercised in flits env.

**Tasks:**
- [ ] **Port tests first:** `$FFT/test_catalog_plan.py` (10 tests) → `$VO/test_catalog_plan.py`
      (relative imports, unit markers; no gurobi references exist in it — verified).
- [ ] **Run, watch fail**, **create `$VO/catalog_plan.py`** = ffh verbatim
      (`from .domain import Sightline`, `from .halos import ADAPTERS`, `from .reduce import
      rdelta_from_mdelta` all resolve), **run, watch pass.**
- [ ] Confirm the gurobi-absent path: `$PY -c "from galaxies.foreground.vo.catalog_plan import
      plan_queries; from galaxies.foreground.vo.domain import Sightline; \
      p = plan_queries([Sightline(ra=1, dec=2, name='a', redshift=0.3)], budget=5); \
      print(p.solver, p.metadata.get('gurobi_fallback'))"` → `greedy import error: …` (fallback engaged).
- [ ] **Commit:** `git commit -m "feat(vo): catalog query planning (greedy + optional gurobi)" -- galaxies/foreground/vo/`

**Dependencies:** Phases 2-3.

### Phase 5: Plotting

**Objective:** sky-map/summary plotting importable, Agg-only, one smoke test.

**Tasks:**
- [ ] **Port test first:** `$FFT/test_plotting.py` (1 test) → `$VO/test_plotting.py`; watch fail.
- [ ] **Create `$VO/plotting.py`** = ffh verbatim; watch pass.
- [ ] **Commit:** `git commit -m "feat(vo): candidate sky-map plotting" -- galaxies/foreground/vo/`

**Dependencies:** none beyond Phase 1 schema (joins on normalized columns).
Note: the test writes PNGs to `tmp_path` only — no `figures.manifest.json` is created, so the
figure-review Stop gate is not triggered by this phase.

### Phase 6: CLI merge (argparse) + examples

**Objective:** `flits-halos` gains `rank`, `plan-queries`, `run-plan`, `run-catalog`,
`dedupe-candidates`, `plot-candidates`; ffh's CLI tests ported off click.

**Tasks:**
- [ ] **Port tests first:** merge `$FFT/test_cli.py` (10 tests, 21K) into `$VO/test_cli.py`,
      rewriting CliRunner invocations to the argparse pattern (model: `$VO/test_cli.py:14-28`):
  ```python
  # click (ffh)                                  # argparse (ported)
  runner = CliRunner()                           cli.main(["run-catalog", str(targets), "glade2",
  result = runner.invoke(main, ["run-catalog",       "--radius-arcmin", "10", "--maxrec", "5",
      str(targets), "glade2", ...])                  "--output", str(out)])
  assert result.exit_code == 0                   out_df = pd.read_csv(out)   # assert on artifacts
  ```
      Error-path tests assert `SystemExit` (argparse `parser.error`) instead of
      `result.exit_code != 0`. The `monkeypatch.setattr(cli_module, "cone_query", …)` style ports
      unchanged. Copy the ffh fixtures (`fake_glade2_rows` etc.) into the merged file.
- [ ] **Run, watch fail:** `$PY -m pytest $VO/test_cli.py -q` → unknown-subcommand SystemExit failures.
- [ ] **Extend `$VO/cli.py`:** port the six ffh helpers (`_load_target_records` — click.ClickException
      → `SystemExit(parser.error(...))` or `raise SystemExit(msg)`; `_merge_mass_provenance`;
      `_load_query_plan`; `_process_catalog_query`; `_classify_foreground_candidates`;
      `_finalize_catalog_run` — `click.echo` → `print`) and register the subparsers inside `main()`:
  ```python
  sp = sub.add_parser("rank", help="Rank an already-normalized candidate table (first target)")
  sp.add_argument("targets_yaml", type=Path)
  sp.add_argument("candidates", type=Path)
  sp.add_argument("--output", type=Path, default=Path("results/ranked.csv"))
  sp.set_defaults(func=cmd_rank)

  sp = sub.add_parser("plan-queries", help="Optimize catalog queries per sightline under a budget")
  sp.add_argument("targets", type=Path)
  sp.add_argument("--budget", type=int, default=50)
  sp.add_argument("--solver", choices=["gurobi", "greedy"], default="gurobi")
  sp.add_argument("--require-mass", action=argparse.BooleanOptionalAction, default=True)
  sp.add_argument("--require-spec-z", action=argparse.BooleanOptionalAction, default=False)
  sp.add_argument("--rvir-multiple", type=float, default=1.2)
  sp.add_argument("--output", type=Path, default=Path("results/query_plan.csv"))
  sp.add_argument("--json-output", type=Path, default=None)
  sp.add_argument("--maxrec", type=int, default=5000)
  sp.set_defaults(func=cmd_plan_queries)

  sp = sub.add_parser("run-plan", help="Execute a plan-queries CSV/JSON")
  sp.add_argument("plan_file", type=Path)
  sp.add_argument("--maxrec", type=int, default=None)
  sp.add_argument("--output", type=Path, default=Path("results/run_plan_candidates.csv"))
  sp.add_argument("--summary-output", type=Path, default=None)
  sp.add_argument("--host-theta-arcmin-max", type=float, default=0.1)
  sp.add_argument("--host-dz-max", type=float, default=0.05)
  sp.add_argument("--host-dz-spec-max", type=float, default=0.005)
  sp.set_defaults(func=cmd_run_plan)

  sp = sub.add_parser("run-catalog", help="End-to-end: cone-query a catalog at each sightline")
  sp.add_argument("targets", type=Path)
  sp.add_argument("catalog_name")
  sp.add_argument("--access-url", default=_SERVICE_ALIASES["vizier"])
  sp.add_argument("--radius-arcmin", type=float, default=10.0)
  sp.add_argument("--maxrec", type=int, default=200)
  sp.add_argument("--output", type=Path, default=Path("results/run_catalog_candidates.csv"))
  sp.add_argument("--summary-output", type=Path, default=None)
  sp.add_argument("--host-theta-arcmin-max", type=float, default=0.1)
  sp.add_argument("--host-dz-max", type=float, default=0.05)
  sp.add_argument("--host-dz-spec-max", type=float, default=0.005)
  sp.set_defaults(func=cmd_run_catalog)

  sp = sub.add_parser("dedupe-candidates", help="Merge duplicate galaxies across catalogs per sightline")
  sp.add_argument("input_path", type=Path)
  sp.add_argument("--output", type=Path, default=Path("results/candidates_deduped.csv"))
  sp.add_argument("--summary-output", type=Path, default=None)
  sp.set_defaults(func=cmd_dedupe_candidates)

  sp = sub.add_parser("plot-candidates", help="Sky maps from a normalized candidate table")
  sp.add_argument("candidates", type=Path)
  sp.add_argument("targets_csv", type=Path)
  sp.add_argument("--output-dir", type=Path, default=Path("figures"))
  sp.add_argument("--search-radius-arcmin", type=float, default=None)
  sp.add_argument("--label-top-n", type=int, default=5)
  sp.set_defaults(func=cmd_plot_candidates)
  ```
      Each `cmd_*` wraps the corresponding ffh command body with `args.<name>` access (bodies port
      near-verbatim; `_process_catalog_query` and `_finalize_catalog_run` were already refactored
      free of click except `echo`). Add `--maxrec` and `--output` to the existing `cone` subparser
      and write via `io.write_table` when `--output` given.
- [ ] **Run, watch pass:** `$PY -m pytest $VO/test_cli.py -q`
- [ ] **Copy ffh example data** (small, science-relevant): `examples/chime_dsa_targets.yaml`,
      `examples/chime_dsa_real_targets.yaml`, `examples/chime_dsa_hostgalaxies.csv` →
      `$VO/examples/`. (`targets_example.yaml` stays as the smoke-test file.)
- [ ] **Commit:** `git commit -m "feat(vo): argparse CLI for rank/plan-queries/run-plan/run-catalog/dedupe/plot + examples" -- galaxies/foreground/vo/`

**Dependencies:** Phases 1-5.

**Verification:**
- [ ] `$PY -m galaxies.foreground.vo.cli --help 2>&1 | grep -c "rank\|run-plan\|run-catalog"` ≥ 3
      *(module invocation; console script resolves after any reinstall — entry point unchanged)*
- [ ] `flits-halos run-targets … --help` still present (must-preserve surface)

### Phase 7: Docs, exports, data pointers

**Objective:** README reflects the union; migration notes replicate the los_halos pattern;
DATA_LOCATIONS updated.

**Tasks:**
- [ ] **Update `$VO/__init__.py`**: export the full public surface (add `ADAPTERS`, `HaloAdapter`,
      `add_halo_masses`, `add_intersection_flags`, `summarize_sightlines`, `deduplicate_candidates`,
      `Sightline`, `ForegroundCandidate`, `QueryResult`, `read_targets_yaml`, `plan_queries`) —
      same-edit rule: exports and their imports land together.
- [ ] **Rewrite `$VO/README.md`**: pipeline (discover → plan → query → normalize → halo-mass →
      reduce → classify → dedupe → plot), schema table (fold ffh `docs/schema.md` rules: photo-z as
      priors, no Δ-definition conversion, Planck18 default, spec-z preference in dedupe), CLI
      examples for all 13 subcommands, and a **Migration notes (frb-foreground-halos → here,
      2026-07-05)** section listing: ported modules; refactors (click→argparse, tenacity→local
      retry, `reduce`→`rank` rename, Target→Sightline with `z_host` YAML alias); **not ported**
      (discovery.py subsumed, headline_report.py, phase5_gladep.md + other docs, _quarantine/,
      gurobi.log); pointer to ffh's untracked `docs/reviews/*` review artifacts (preserved in the
      archived repo); the GLADE+ `f_dL` z-provenance fact and placeholder-z rule now encoded in
      code comments.
- [ ] **Update `DATA_LOCATIONS.md:125-126`**:
  ```markdown
  - subhalos: https://github.com/jakobtfaber/subhalos — archived 2026-07-XX; consolidated into
    frb-foreground-halos (June 2026), itself integrated into `galaxies/foreground/vo/`
  - frb-foreground-halos: https://github.com/jakobtfaber/frb-foreground-halos — archived 2026-07-XX;
    integrated into `galaxies/foreground/vo/` (PR #NNN). Physical results data:
    `~/Data/frb-foreground-halos/results/` (symlink pattern per ~/Data convention)
  ```
  (fill the date/PR at execution time; los_halos line 126 stays).
- [ ] **Commit:** `git commit -m "docs(vo): union README + migration notes; DATA_LOCATIONS pointers" -- galaxies/foreground/vo/ DATA_LOCATIONS.md`

**Dependencies:** Phases 1-6 (documents what landed).

**Verification:**
- [ ] `rtk grep -n "frb-foreground-halos" DATA_LOCATIONS.md` shows the integrated+archived lines
- [ ] README CLI section lists every subcommand emitted by `--help`

### Phase 8: Full verification battery

**Objective:** prove the merge before the one-way doors.

**Tasks:**
- [ ] **Offline suite:** `$PY -m pytest $VO -m "not network" -q` → expect **≈95 passed** (record
      exact number), 0 failed.
- [ ] **Foreground regression:** `$PY -m pytest galaxies/foreground -m "not network" -q --ignore=galaxies/foreground/vo`
      → 151 passed (baseline from PR #122).
- [ ] **Repo default suite unaffected:** `$PY -m pytest -m "not slow" -q` → same pass/fail profile
      as `030c159c` (vo/ is outside `testpaths`; this catches accidental import leakage).
- [ ] **Lint:** `$PY -m ruff check galaxies/foreground/vo && $PY -m ruff format --check galaxies/foreground/vo`
- [ ] **Live smoke (network, one each):**
  ```bash
  $PY -m pytest $VO/test_live_services.py -m network -q          # RegTAP/VizieR reachable
  $PY -m pytest $VO/test_frb_recovery.py::test_frb_isha_control -m network -q
  ```
- [ ] **End-to-end CLI smoke (network):**
  ```bash
  cd /tmp && flits-halos run-catalog \
    $OLDPWD/galaxies/foreground/vo/examples/chime_dsa_targets.yaml glade2 \
    --radius-arcmin 5 --maxrec 50 --output /tmp/ffh_smoke/cands.csv
  # expect: candidates CSV + cands.summary.csv with status column
  ```
- [ ] **Adversarial review subagent** (repeat of the los_halos-round check that caught 2 real
      minors): dispatch a review agent to diff every ported module against its ffh original
      (`$FFH/*.py`) and every merged test against `$FFT/*.py`, hunting behavior drift introduced by
      the argparse/tenacity/import rewrites. Findings fixed before Phase 9.
- [ ] **Verify-gate records** for every touched path:
  ```bash
  verify-gate record --paths "galaxies/foreground/vo/*.py" --method test \
    --check "pytest galaxies/foreground/vo -m 'not network' (flits env)" \
    --evidence "<N> passed, 0 failed"
  verify-gate record --paths galaxies/foreground/vo/cli.py --method adversarial-review \
    --check "port-vs-ffh-original diff review" --evidence "<verdict>"
  ```
- [ ] **Closeout check:** `agent-closeout-check --repo . --touched galaxies/foreground/vo --touched DATA_LOCATIONS.md`
      (pyproject untouched → no runtime packet expected; if the checker demands one, produce it).
- [ ] **Ledger tail commit:** `git add docs/entire-tracing-checkpoints.md && git commit --no-verify -m "chore: entire-tracing checkpoint" -- docs/entire-tracing-checkpoints.md`

**Dependencies:** Phases 1-7.

### Phase 9: Ship (one-way doors — @human)

**Objective:** PR up, merged on approval.

**Tasks:**
- [ ] Push `feat/ffh-integration`; open PR titled
      `feat(foreground): integrate frb-foreground-halos into galaxies/foreground/vo`, body linking
      this plan + handoff, listing the union decision, test counts, and the not-ported inventory.
      **Gated: batch the push+PR into the same user-approval ask as Phase 10's archives where practical.**
- [ ] On merge: check off the `@human` PR item in `.agents/deferred-tasks.md`.

### Phase 10: Archive frb-foreground-halos + subhalos (@human gated)

**Objective:** retire both source repos per the los_halos recipe.

**Tasks:**
- [ ] **subhalos consolidation proof** (handoff requires verifying "nothing to port" before archive):
  ```bash
  cd ~/Developer/repos/github.com/jakobtfaber/subhalos && git status --short  # expect clean @ d830c0d0
  gh pr list --repo jakobtfaber/subhalos --state open                          # expect none
  ```
      plus a read-only sweep confirming every subhalos public symbol has an ffh/vo descendant
      (domain objects, adapter concept, column-role inference — per ffh `docs/migration.md` the
      architecture was the ported item; record the sweep verdict in the archive-notice README).
- [ ] **ffh archive prep:** commit the currently-untracked review artifacts so they survive on
      GitHub (`AGENTS.md`, `docs/reviews/*`; exclude `gurobi.log`), then add the archive-notice
      README pointing at `dsa110-FLITS/galaxies/foreground/vo/` (+ PR #NNN) and noting physical data
      stays at `~/Data/frb-foreground-halos/`.
- [ ] **Recipe per repo** (los_halos precedent, Learnings #7): archive-notice README → local commit
      → `mv` to `~/Developer/repos/github.com/jakobtfaber/archived/` → **(user-approved, batched)**
      `git push` + `gh repo archive jakobtfaber/frb-foreground-halos --yes` +
      `gh repo archive jakobtfaber/subhalos --yes` → check off `.agents/deferred-tasks.md`.
- [ ] Update `DATA_LOCATIONS.md` archive dates (placeholder from Phase 7).

## Success Criteria

### Automated Verification

- [ ] `$PY -m pytest galaxies/foreground/vo -m "not network" -q` → ≈95 passed / 0 failed in flits env
- [ ] `$PY -m pytest galaxies/foreground -m "not network" -q --ignore=galaxies/foreground/vo` → 151 passed
- [ ] `$PY -m ruff check galaxies/foreground/vo` → clean
- [ ] `$PY -c "import galaxies.foreground.vo as v; assert all(hasattr(v, n) for n in
      ['ADAPTERS','plan_queries','deduplicate_candidates','summarize_sightlines','Sightline'])"`
- [ ] `flits-halos --help` lists all 13 subcommands incl. `run-targets`, `rank`, `run-catalog`
- [ ] Files exist: `$VO/{provenance,domain,io,halos,classify,dedupe,catalog_plan,plotting}.py` and
      `$VO/test_{domain,io,halos,classify,dedupe,catalog_plan,plotting}.py`
- [ ] `rtk grep -rn "import click\|import tenacity\|from click\|from tenacity" galaxies/foreground/vo/` → no hits
- [ ] verify-gate records exist for all touched paths (Stop hook passes)

### Manual Verification

- [ ] Phase 0 drift decision: user confirms the 6 dirty vo/ files are discardable editor drift
- [ ] Live smokes green (network): RegTAP/VizieR reachable; isha control behaves; `run-catalog glade2`
      end-to-end produces candidates + summary CSVs with sensible statuses
- [ ] Adversarial-review verdict: no unresolved behavior drift vs ffh originals
- [ ] PR review + merge; archive approvals (one batched ask for push/PR/archives)
- [ ] zach recovery test (`-m network -m slow`) run once before archiving the source repos

### Reproducibility & Correctness (research code)

- [ ] Deterministic ranking preserved: `merge_and_rank` mergesort + tiebreakers unchanged from ffh
      (byte-identical reruns; pinned by the ported reduce/dedupe tests)
- [ ] SHMR inversion numerically pinned by ported `test_halos.py` (14 tests: unit conversions,
      Bell+03 K-band path, Moster+13 round-trips, quality-flag skips, per-row mass provenance)
- [ ] Environment: flits conda env (Python 3.12), agent-safe `env -i` invocation recorded above;
      no new dependencies introduced

## Testing Strategy

**Unit (written in-phase):** every ported module lands behind its ported test file (fail→pass);
signature-change regressions covered by the updated vo tests (tuple `cone_query`, keyword-only
`to_common_schema`/`merge_and_rank`, `z_host` alias).

**Integration:** `run-catalog`/`run-plan` CLI tests with monkeypatched `cone_query` (offline,
ported from ffh — cover zero-row targets, truncation status, host-exclusion Δz split by z_type,
unknown-catalog errors); live network smokes (Phase 8); zach/whitney/isha recovery (must-survive).

**Manual:** Phase 8 end-to-end CLI smoke; figure eyeball of one `plot-candidates` output against
the archived ffh `figures/` reference before archiving.

**Test data:** ffh example YAML/CSV files copied into `$VO/examples/` (Phase 6); all other test
data is synthetic in-test frames.

## Migration Strategy

**Migration steps:** Phases 0-10 above.
**Rollback plan:** every phase is one pathspec commit on `feat/ffh-integration`; revert the branch
or drop unmerged commits. Phase 0's stash preserves the drift. Archives happen only after the PR
merges and are themselves reversible (`gh repo unarchive`), though treated as one-way for approval.
**Backward compatibility:** `flits-halos` keeps all seven existing subcommands (incl. `run-targets`);
`Target`/`load_targets`/`get_cosmology` remain importable via the shim; `z_host` YAML keys still load;
`to_common_schema`/`merge_and_rank`/`cone_query` signatures change **intra-repo only** (all callers
updated in Phase 1; no external consumers — grep verified within FLITS).

## Risk Assessment

1. **Risk:** the dirty-lane owner (editor buffer) re-saves stale content mid-implementation.
   - **Likelihood:** Medium · **Impact:** High (silently reverts merged/ported code)
   - **Mitigation:** Phase 0 user gate closes the editor tabs first; `git status` check in every
     phase verification; work happens on a fresh branch so drift is visible as unexpected dirt.
2. **Risk:** post-edit autoformatter strips imports added ahead of consumers.
   - **Likelihood:** Medium · **Impact:** Medium (NameError later)
   - **Mitigation:** same-edit rule enforced in task ordering; per-phase pytest run is the backstop.
3. **Risk:** distance-ordered ADQL (`sep_deg`) rejected by some TAP services.
   - **Likelihood:** Low (VizieR/vollt compatibility already handled via alias form) · **Impact:** Medium
   - **Mitigation:** `order_by_distance=False` escape hatch preserved; live smoke in Phase 8;
     `safe_search`-based discovery paths don't use cone ADQL.
4. **Risk:** click→argparse rewrite changes CLI error semantics the ported tests assumed.
   - **Likelihood:** Medium · **Impact:** Low (test-only churn)
   - **Mitigation:** assert on artifacts/stdout rather than exit codes where possible; SystemExit
     for error paths.
5. **Risk:** merged test files exceed pytest name-collision tolerance (duplicate function names).
   - **Likelihood:** Low · **Impact:** Low
   - **Mitigation:** `_ffh` suffix rule on collision (Phase 1); pytest errors loudly if missed.

## Edge Cases and Error Handling

1. **Case:** targets YAML row has both `z_host` and `redshift`.
   - **Expected:** `redshift` wins (alias only fills when `redshift` absent) — pinned by the Phase 2 alias test.
2. **Case:** placeholder host redshift (z ≥ 0.999 or `host_z_placeholder: true`).
   - **Expected:** `Sightline.redshift = None`; run-catalog marks the sightline `z_frb_unknown`
     rather than classifying every z<1 galaxy as foreground (`io.normalize_host_redshift`, ported tests).
3. **Case:** TAP returns exactly `maxrec` rows.
   - **Expected:** `metadata["truncated"] = True`; summary status `possibly_truncated`
     (`_finalize_catalog_run`, ported run-catalog tests).
4. **Case:** pyvo absent / offline environment.
   - **Expected:** `TAPService = None` guard → empty DataFrames, no raises (pattern preserved from
     both parents; offline suite runs without network).
5. **Case:** gurobipy absent (flits env).
   - **Expected:** `plan_queries(solver="gurobi")` returns a greedy plan with
     `metadata["gurobi_fallback"]` set (Phase 4 check).
6. **Error:** unknown catalog name in `run-catalog`/`run-plan`.
   - **Handling:** `SystemExit` with the sorted known-adapter list (ported from click.ClickException).

## Documentation Updates

- [ ] `$VO/README.md` — union pipeline, schema rules, 13-subcommand CLI, migration notes + not-ported inventory (Phase 7)
- [ ] `DATA_LOCATIONS.md:125-126` — subhalos/ffh archive + integration pointers (Phases 7/10)
- [ ] `.agents/deferred-tasks.md` — items added Phase 0, checked off Phases 8-10
- [ ] Archive-notice READMEs in both retired repos (Phase 10)

## Timeline Estimate

- Phase 0: minutes (gated on one user decision)
- Phases 1-2: the careful half (signature merges + callsite updates)
- Phases 3-5: fast near-verbatim ports
- Phase 6: the large mechanical rewrite (click→argparse, 21K of CLI tests)
- Phases 7-8: docs + verification battery
- Phases 9-10: gated on user approval

## Open Questions

*(none — the one pending item, Phase 0 drift disposition, is a user gate inside the plan, not an
unresolved design question)*

---

## References

**Research documents:**
- [Handoff (research artifact): frb-foreground-halos migration](handoff-2026-07-05-14-38-frb-foreground-halos-migration.md)

**Files analyzed (plan research, 2026-07-05):**
- `galaxies/foreground/vo/` — all 14 modules + README + tests (HEAD `030c159c` + dirty-tree diff)
- `~/Developer/repos/github.com/jakobtfaber/frb-foreground-halos/` — all 15 `src/` modules, all 12
  test files (55 tests), `README.md`, `CLAUDE.md`, `AGENTS.md`, `docs/{migration,schema}.md`,
  `pyproject.toml`, `examples/`
- `pyproject.toml` (FLITS) — entry points, deps, pytest config
- `DATA_LOCATIONS.md:125-126`, `.agents/deferred-tasks.md`
- flits conda env dependency probe (click ✓ / tenacity ✗ / gurobipy ✗ / matplotlib ✓ / scipy ✓ /
  yaml ✓ / requests ✓ / pyvo ✓ / seaborn ✓)

**Precedent:**
- PR #122 (`030c159c`) — los_halos → `vo/` port; migration-notes pattern; adversarial-review verdict
  (PASS, 2 minors fixed in `e2fe029`); archive recipe.

---

## Review History

### Version 1.0 — 2026-07-05
- Initial plan. Union-merge decision resolved from the module-level diff (handoff option 1 refined
  by option 3's evidence). Phase 0 added for the stale vo/ drift discovered during plan research
  (lane-liveness: live → preserved, decision pending).
