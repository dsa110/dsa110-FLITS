# galaxies.foreground.vo — VO-TAP wide-net foreground discovery

Automatically discover, query, and consolidate foreground galaxy halos and clusters
along precisely-localized FRB sightlines, across arbitrary Virtual Observatory TAP
services. Complements the curated engines in `galaxies.foreground` (Vizier/NED/DESI
specific catalogs): this layer casts a wide net over everything RegTAP knows about,
and adds a curated-catalog pipeline with halo masses and host/foreground
classification.

Consolidates two archived repos:
[`los_halos`](https://github.com/jakobtfaber/los_halos) (migrated 2026-07-05, PR #122)
and [`frb-foreground-halos`](https://github.com/jakobtfaber/frb-foreground-halos)
(itself the June-2026 consolidation of `los_halos` + `subhalos`; migrated 2026-07-05).
See "Migration notes" below.

## Pipelines

**Wide-net cache pipeline** (los_halos lineage — arbitrary TAP services):

1. **discover** — RegTAP finds TAP services (`registry.discover_tap_services`);
   `discover.discover_tables` enumerates candidate tables with RA/Dec/redshift
   columns via UCDs first, then name/description heuristics, with a sampled
   z-value sanity gate (numeric fraction ≥ 0.1, −0.01 ≤ z ≤ 10). Results cached
   as parquet under the cache dir.
2. **query** — ADQL cone searches per candidate table per FRB (`query.cone_query`),
   sync with retries, or async under a wall-time budget (`query.safe_search`).
   ADQL is **distance-ordered** (`DISTANCE(...) AS sep_deg ... ORDER BY sep_deg`)
   so a `TOP maxrec` truncation keeps the *closest* sources; `cone_query` returns
   `(rows, metadata)` with a `truncated` flag.
3. **normalize** — map heterogeneous columns to the common schema
   (`normalize.to_common_schema`), classify `z_type` ∈ {spec, photo, unknown, none},
   keep photo-z rows as priors (`z_prior=true` in provenance).
4. **reduce** — merge across services, compute impact parameter b (kpc) from angular
   separation and D_A(z), derive R_Δ from M_Δ when present (no Δ-definition
   conversions), rank by b/R_Δ else b (`reduce.merge_and_rank`).

**Curated-catalog science pipeline** (frb-foreground-halos lineage — GLADE2/GLADE+/
HECATEv2/2MRS adapters):

1. **plan** — `catalog_plan.plan_queries` optimizes which catalogs to query per
   sightline under a TAP budget (Gurobi MIP when `gurobipy` is importable, greedy
   fallback otherwise), sizing cone radii for b ≤ 1.2 R_Δ of a 1e13 M☉ group halo.
2. **query + mass** — `halos.HaloAdapter.prepare` derives M_200c per row: direct
   stellar mass (linear/1e10/log10) or K-band (Bell+03 M/L) → Moster+13 SHMR
   inversion; per-row `mass_provenance` records the method.
3. **classify** — `classify.add_intersection_flags` adds b/R_vir intersection
   brackets (strict 0.8 / nominal 1.0 / loose 1.2); the CLI's host-exclusion flags
   `likely_host` candidates (θ ≤ 0.1′ and Δz within a z_type-dependent tolerance:
   0.005 spec / 0.05 photo-unknown) and restricts headline counts to
   `is_foreground` rows. Per-sightline summaries carry explicit statuses:
   `ok / no_candidates_in_cone / z_frb_unknown / possibly_truncated`.
4. **dedupe** — `dedupe.deduplicate_candidates` merges the same galaxy seen by
   multiple catalogs (30″ + Δz ≤ 0.01, spec-z-preferring representative,
   `catalog_sources` provenance).
5. **plot** — `plotting.plot_all_candidate_sky_maps` renders per-FRB tangent-plane
   sky maps + a multi-FRB summary (Agg backend, no display needed).

## Normalized row schema

`name, id?, ra (deg), dec (deg), z, z_type, richness?, m_delta? (M☉), r_delta? (kpc),
delta_def? (200/500/…), service, table, provenance_json (sorted keys)`

Units: RA/Dec degrees; separations arcmin; distances kpc; masses M☉.
Cosmology: Planck18 by default. Rules (contractual):

- Column order is the wire format; reduction *appends* (`theta_arcmin`, `b_kpc`,
  `r_delta_computed`, `rank_key`), never reorders.
- Photo-z is a prior, not a measurement (`z_prior=true`; prefer spec rows in
  dedupe and interpretation).
- Never convert between overdensity definitions — `rdelta_from_mdelta` computes
  R_Δ *at the row's existing Δ* only.
- Ranking is deterministic: mergesort on `[rank_key, service, table, id, name]`;
  `rank_key = b/R_Δ` when both exist, else `b`, else inf (smaller = better).
- Host redshifts ≥ 0.999 (or `host_z_placeholder: true`) are placeholders →
  `redshift=None` → status `z_frb_unknown`, never "everything is foreground".

## CLI

Installed as `flits-halos` (see `pyproject.toml`). Targets YAML accepts
`redshift` or the legacy `z_host` key.

```bash
# wide-net cache pipeline
flits-halos services --keywords galaxy cluster        # RegTAP service discovery
flits-halos tables <tap-url>                          # candidate tables at one endpoint
flits-halos cone <tap-url> <table> <ra_col> <dec_col> <ra> <dec> [--maxrec N --output f.csv]
flits-halos run-targets <tap-url> <table> <ra_col> <dec_col> --targets <targets.yaml>
flits-halos discover --services vizier,mast,datalab   # cache services + tables
flits-halos query  --targets galaxies/foreground/vo/targets_example.yaml
flits-halos reduce --targets galaxies/foreground/vo/targets_example.yaml --out results/

# curated-catalog science pipeline
flits-halos plan-queries <targets.yaml|.csv> --budget 50 --solver greedy
flits-halos run-plan results/query_plan.csv --output results/candidates.csv
flits-halos run-catalog <targets.yaml> glade-plus --radius-arcmin 10 --maxrec 5000
flits-halos dedupe-candidates results/candidates.csv --summary-output results/summary.csv
flits-halos rank <targets.yaml> results/candidates.csv --output results/ranked.csv
flits-halos plot-candidates results/candidates.csv examples/chime_dsa_hostgalaxies.csv \
  --output-dir figures/
```

Example target files: `targets_example.yaml` (smoke),
`examples/chime_dsa_targets.yaml`, `examples/chime_dsa_real_targets.yaml` (the
CHIME–DSA co-detection sample), `examples/chime_dsa_hostgalaxies.csv`.

## Tests

Colocated (not in the default `pytest` testpaths, same as the rest of `galaxies/`):

```bash
pytest galaxies/foreground/vo -m "not network"   # offline unit tests
pytest galaxies/foreground/vo -m network         # live VizieR/RegTAP integration + validation
```

`test_frb_recovery.py` validates recovery of manually-curated foreground objects for
the co-detection bursts zach/whitney, with isha as a control sightline.

## Migration notes

### los_halos → here (2026-07-05, PR #122)

Ported: the `los_halos` package (registry/discover/query/normalize/reduce/xmatch),
tests, and the smoke targets. Refactors during the port: typer CLI → argparse
(the typer app defined `discover` twice; the shadowed service-listing command is
now `services`), pydantic `Target` → dataclass, tenacity → local retry loop
(tenacity is not in the flits env), dead code removed from `discover.py`,
TAP_SCHEMA IN-lists chunked (50/query — oversized clauses fail on VizieR; from
`fixed_discover.py`), `cache_dir` parameterized.

Not ported (superseded or debris; retrievable from the archived repo):
- `foreground_search_with_fetch.py`, `run_foreground_for_list.py`,
  `smoke_test_catalogs.py` — the astroquery-based (Hussaini-style) foreground
  search; superseded by `galaxies/foreground/` (engines, census registry) and
  `galaxies/wise-ps1-strm/`.
- `debug_discovery.py`, `quick_fix_discovery.py`, `fixed_discover.py` — debug
  iterations (the one durable fix, IN-list chunking, is folded in here).
- `frb_data/DSA110_CHIME_Codetection_BurstProperties_Foreground.csv` — stale
  Aug-2025 sibling of `scratch/codetection/source/` (canonical in FLITS).
- `results/TEST_*` smoke outputs.

### frb-foreground-halos → here (2026-07-05)

ffh was the newer consolidation of `los_halos` + `subhalos`; this merge adopted
its evolved shared-lineage modules and net-new science layer while keeping the
vo-unique wide-net capabilities (RegTAP registry, cached table discovery with
the z-sanity gate, `safe_search` async budgets, CDS X-Match).

Ported (near-verbatim): `halos.py` (incl. the GLADE+ `f_dL` z-provenance fix —
`f_dL` 1=photo/3=spec is the redshift-origin flag; `f_zcmb` is only the
peculiar-velocity correction flag), `catalog_plan.py`, `classify.py`,
`dedupe.py`, `domain.py`, `io.py` (placeholder-z guard), `plotting.py`; the
evolved `query.py` (distance-ordered ADQL, `(df, metadata)` + `truncated`),
`normalize.py` (per-mapping provenance), `reduce.py`, `provenance.py`; 53 of its
55 tests.

Refactors during the port: click CLI → argparse subcommands on `flits-halos`
(ffh `reduce` renamed `rank`; ffh `discover` subsumed by `services`); tenacity →
the local retry loop; `Target` → `Sightline` (`targets.py` is now a shim;
`z_host` YAML keys still load).

Not ported (superseded or debris; retrievable from the archived repo):
- `discovery.py` — static-anchor service list + column-role inference; subsumed
  by `registry.py` + `discover.py` (whose exclude-list already covers the
  `access_estsize` false-positive fix). Its two tests duplicate
  `test_discover.py` coverage.
- `examples/headline_report.py`, `docs/{phase5_gladep,catalog-query-planning,
  pre-run-checklist}.md` — run narratives; the durable rules from
  `docs/{schema,migration}.md` are folded into this README.
- `_quarantine/halos_2.py` (triaged debris), `gurobi.log`.
- Untracked agent-review artifacts (`docs/reviews/*`, `AGENTS.md`) — preserved
  in the archived repo; their two durable facts (GLADE+ `f_dL`, placeholder-z)
  are encoded in `halos.py` comments and `io.normalize_host_redshift`.

Physical results data lives at `~/Data/frb-foreground-halos/results/`
(`~/Data` symlink convention; see `DATA_LOCATIONS.md`).
