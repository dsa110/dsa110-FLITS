# Data Locations for CHIME-DSA Co-Detection Project

**4-host model (Phase 5 complete 2026-06-25):** jakob-mbp (code), iacobus (data authority), arc (`.npy`/CANFAR), h17 (compute). Retired hosts h23, hpcc, dsacamera are read-only quarantine references only.

Plan: [`docs/infrastructure/MIGRATION_PLAN_4HOST.md`](docs/infrastructure/MIGRATION_PLAN_4HOST.md) · Inventory: [`machine_inventory.yaml`](machine_inventory.yaml) · Query: [`scripts/query_machine_inventory.py`](scripts/query_machine_inventory.py)

## Code (GitHub canonical)

| Host | Path |
|------|------|
| **jakob-mbp** | `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS` |
| **GitHub** | https://github.com/jakobtfaber/dsa110-FLITS |

Do not develop on hpcc, arc checkout, or h23 trees. h17 may hold an optional clone for docker workflows.

## Processed data (iacobus authority)

**Canonical:** `iacobus:~/Research/CHIME_DSA_Codetections/` (~218 GiB, 1546 files as of 2026-06-17).

CloudDocs mirror: `~/Library/Mobile Documents/com~apple~CloudDocs/Research/CHIME_DSA_Codetections/` — on jakob-mbp this is placeholders only; bytes live on iacobus.

| Subdir | Role |
|--------|------|
| `burst_npys/` | DSA/CHIME burst `.npy` (h23 drained 2026-06-25) |
| `burst_pickles/` | 24 full-Stokes `.pkl` (Dropbox → iacobus) |
| `dsa_fullstokes_waterfalls/` | IQUV `.npy` from h23 |
| `scattering_results/` | Fit PDFs, corners (h23 + hpcc JSON merged) |
| `dm_budget/` | DM budget code + h23 merge |
| `metadata/` | CSVs, localizations |
| `archive/` | `OLD_CHIME_DSA_Codetections/`, `burstprop_paper/`, `dsa110-scat/` |

Sentinels: [`codetections_manifest.yaml`](codetections_manifest.yaml)

**Out of scope:** nihari (`Research/nihari/` on iCloud; h23 `jfaber/nihari/` remains on source).

## Burst `.npy` for fits (arc + local replica)

| Host | Path | Role |
|------|------|------|
| **arc** | `arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts` | CANFAR primary (~2.9G) |
| **arc** | `.../CHIME_bursts` | Separate namespace from iacobus burst_npys |
| **jakob-mbp** | `~/Developer/dsa110-local-data/DSA_bursts` | Offline replica (gap-synced Phase 3) |

## Compute workspace (h17)

| Path | Size | Notes |
|------|------|-------|
| `/data/research/astrophysics/frbs/chime-dsa-codetections` | 29G | CHIME docker / filterbanks / numpy |
| `/data/jfaber/upchan_codetections` | 473M | Upchan products |
| `/data/jfaber/arc_archive_2026-06` | 36G | arc trash copy; optional dedupe → iacobus (Phase 4) |

## Legacy Documents-Area Clone

Drain-only snapshot, not active dev:

- `~/Documents/research/caltech/ovro/dsa110/dsa110-FLITS/` (iCloud Areas mirror)
- See [`docs/migration/chime-dsa-documents-area-migration.md`](docs/migration/chime-dsa-documents-area-migration.md)

## Retired hosts (quarantine / read-only)

Move-only policy; restore one-liners in each host's `_quarantine/README.md`.

| Host | Quarantine path | Status |
|------|-----------------|--------|
| **h23** | `/media/ubuntu/ssd/_quarantine/jfaber-drain-20260625/` | Partial: archive, burstprop_paper, chime_dsa_codetections (~137G). Residual at `jfaber/`: nihari, tools, dsa110-continuum, frb_inventory, scratch |
| **h23** | `/dataz/dsa110/T3/` | **Not quarantined** — 59T raw pipeline; leave on source |
| **hpcc** | `/home/jfaber/_quarantine/flits-20260625` | Full flits tree quarantined 2026-06-25; JSON artifacts on jakob-mbp |
| **dsacamera** | — | Decommissioned; no codetection content |

## CANFAR arc compute

Storage: `vos`/`vls` with `~/.ssl/cadcproxy.pem`. Compute: `canfar` CLI (`canfar create --gpu N`). Notebook sessions without `--gpu` are CPU-only. Live A100 MIG smoke test passed 2026-06-25.

## Dropbox exit

Authoritative staging on iacobus CloudDocs `Dropbox-Migration/`; codetection-scope folders renamed `*_retired_2026-04-24_iCloud_is_authoritative/` on Dropbox source. Burst pickles live only under `Research/CHIME_DSA_Codetections/burst_pickles/`.

## Related repos

- subhalos: https://github.com/jakobtfaber/subhalos
- los_halos: https://github.com/jakobtfaber/los_halos
- dsa110-scat: git + iacobus `archive/dsa110-scat/`
