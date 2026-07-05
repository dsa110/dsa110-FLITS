# galaxies.foreground.vo — VO-TAP wide-net foreground discovery

Automatically discover, query, and consolidate foreground galaxy halos and clusters
along precisely-localized FRB sightlines, across arbitrary Virtual Observatory TAP
services. Complements the curated engines in `galaxies.foreground` (Vizier/NED/DESI
specific catalogs): this layer casts a wide net over everything RegTAP knows about.

Migrated 2026-07-05 from the standalone [`los_halos`](https://github.com/jakobtfaber/los_halos)
repo (now archived). See "Migration notes" below for what was and wasn't ported.

## Pipeline (discover → query → normalize → reduce)

1. **discover** — RegTAP finds TAP services (`registry.discover_tap_services`);
   `discover.discover_tables` enumerates candidate tables with RA/Dec/redshift
   columns via UCDs first, then name/description heuristics, with a sampled
   z-value sanity gate (numeric fraction ≥ 0.1, −0.01 ≤ z ≤ 10). Results cached
   as parquet under the cache dir.
2. **query** — ADQL cone searches per candidate table per FRB (`query.cone_query`),
   sync with retries, or async under a wall-time budget (`query.safe_search`).
   Raw results cached; ADQL + timestamps recorded per query.
3. **normalize** — map heterogeneous columns to the common schema
   (`normalize.to_common_schema`), classify `z_type` ∈ {spec, photo, unknown, none},
   keep photo-z rows as priors (`z_prior=true` in provenance).
4. **reduce** — merge across services, compute impact parameter b (kpc) from angular
   separation and D_A(z), derive R_Δ from M_Δ when present (no Δ-definition
   conversions), rank by b/R_Δ else b (`reduce.merge_and_rank`).

## Normalized row schema

`name, id?, ra (deg), dec (deg), z, z_type, richness?, m_delta? (M☉), r_delta? (kpc),
delta_def? (200/500/…), service, table, provenance_json (sorted keys)`

Units: RA/Dec degrees; separations arcmin; distances kpc; masses M☉.
Cosmology: Planck18 by default (`targets.get_cosmology`).

## CLI

Installed as `flits-halos` (see `pyproject.toml`):

```bash
flits-halos services --keywords galaxy cluster        # RegTAP service discovery
flits-halos tables <tap-url>                          # candidate tables at one endpoint
flits-halos cone <tap-url> <table> <ra_col> <dec_col> <ra> <dec>
flits-halos run-targets <tap-url> <table> <ra_col> <dec_col> --targets <targets.yaml>
flits-halos discover --services vizier,mast,datalab   # cache services + tables
flits-halos query  --targets galaxies/foreground/vo/targets_example.yaml
flits-halos reduce --targets galaxies/foreground/vo/targets_example.yaml --out results/
```

## Tests

Colocated (not in the default `pytest` testpaths, same as the rest of `galaxies/`):

```bash
pytest galaxies/foreground/vo -m "not network"   # offline unit tests
pytest galaxies/foreground/vo -m network         # live VizieR/RegTAP integration + validation
```

`test_frb_recovery.py` validates recovery of manually-curated foreground objects for
the co-detection bursts zach/whitney, with isha as a control sightline.

## Migration notes (los_halos → here, 2026-07-05)

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
