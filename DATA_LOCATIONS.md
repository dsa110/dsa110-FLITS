# Data Locations for CHIME-DSA Co-Detection Project

**Cloud authority (2026-06-26):** Google Drive **jakobtfaber@gmail.com** — `Research/CHIME_DSA_Codetections/` (~280 GiB target from iacobus staging).  
**4-host model:** jakob-mbp (code), **gdrive** (data authority), iacobus (staging source), arc (`.npy`/CANFAR), h17 (compute). Retired hosts h23, hpcc, dsacamera are read-only quarantine references only.

Plan: [`docs/infrastructure/MIGRATION_PLAN_4HOST.md`](docs/infrastructure/MIGRATION_PLAN_4HOST.md) · Inventory: [`machine_inventory.yaml`](machine_inventory.yaml) · Query: [`scripts/query_machine_inventory.py`](scripts/query_machine_inventory.py) · Upload: [`scripts/migration/iacobus_to_gdrive.sh`](scripts/migration/iacobus_to_gdrive.sh)

## Code (GitHub canonical)

| Host | Path |
|------|------|
| **jakob-mbp** | `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS` |
| **GitHub** | https://github.com/jakobtfaber/dsa110-FLITS |

Do not develop on hpcc, arc checkout, or h23 trees. h17 may hold an optional clone for docker workflows.

## Processed data (Google Drive authority)

**Canonical:** `gdrive-jakob:Research/CHIME_DSA_Codetections/` (rclone remote on iacobus; account **jakobtfaber@gmail.com**).

| Access | Path |
|--------|------|
| **rclone** | `gdrive-jakob:Research/CHIME_DSA_Codetections/` |
| **Drive for Desktop** | mount after adding jakobtfaber@gmail.com — expected `~/Library/CloudStorage/GoogleDrive-jakobtfaber@gmail.com/My Drive/Research/CHIME_DSA_Codetections/` |

**Staging source (iacobus):** `iacobus:~/Research/CHIME_DSA_Codetections/` (~223 GiB as of 2026-06-26; excludes `CHIME_Morphologies/`). Bytes upload via `scripts/migration/iacobus_to_gdrive.sh` (direct iacobus→Drive; jakob-mbp orchestrates only).

**Legacy iCloud mirror:** `~/Library/Mobile Documents/com~apple~CloudDocs/Research/CHIME_DSA_Codetections/` — demoted; jakob-mbp shows placeholders only. iacobus CloudDocs clone retained until gdrive upload verified.

| Subdir | Role |
|--------|------|
| `burst_npys/` | DSA/CHIME burst `.npy` (h23 drained 2026-06-25) |
| `burst_pickles/` | 24 full-Stokes `.pkl` (Dropbox → iacobus) |
| `dsa_fullstokes_waterfalls/` | IQUV `.npy` from h23 |
| `scattering_results/` | Fit PDFs, corners (h23 + hpcc JSON merged) |
| `dm_budget/` | DM budget code + h23 merge |
| `metadata/` | CSVs, localizations |
| `presentations/` | DM phase deck/keynote plus `DSA_DM_phase/` per-burst PDFs |
| `archive/` | `OLD_CHIME_DSA_Codetections/`, `dsa110-scat/` |

Sentinels: [`codetections_manifest.yaml`](codetections_manifest.yaml)

**Out of scope:** nihari (`Research/nihari/` on iCloud; h23 `jfaber/nihari/` remains on source).

## CHIME Morphologies (separate project)

**Not part of CHIME–DSA co-detections or the gdrive upload.**

| Host | Path | Size (2026-06-26) |
|------|------|-------------------|
| **iacobus** | `~/Research/CHIME_Morphologies/burstprop_paper/` | 67G |

Moved from `CHIME_DSA_Codetections/archive/burstprop_paper/` on iacobus 2026-06-26. iCloud CloudDocs may still show the old path until mirror sync catches up; authoritative live path is `~/Research/CHIME_Morphologies/`.

### rclone setup (gdrive-jakob)

No Drive remote existed on jakob-mbp or iacobus as of 2026-06-26. One-time OAuth (browser required):

```bash
# jakob-mbp — interactive config
rclone config
# n) New remote → name: gdrive-jakob → Storage: drive → scope: drive
# client_id/secret: blank → advanced: drive.readonly=false
# auto config: y  (opens browser; sign in jakobtfaber@gmail.com)

rclone about gdrive-jakob:
rclone mkdir gdrive-jakob:Research/CHIME_DSA_Codetections   # create canonical root

# Headless iacobus — copy token from jakob-mbp authorize:
rclone authorize "drive"    # jakob-mbp: copy JSON blob
ssh iacobus rclone config   # paste at config_token prompt
```

Verify: `rclone about gdrive-jakob:` · `rclone lsd gdrive-jakob:Research/`

**Note:** jakob-mbp currently mounts **jakobtfaber.caltech@gmail.com** only (`GoogleDrive-jakobtfaber.caltech@gmail.com`); personal account mount is optional after upload.

## Burst `.npy` for fits (arc + local replica)

| Host | Path | Role |
|------|------|------|
| **arc** | `arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts` | CANFAR primary (~2.9G) |
| **arc** | `.../CHIME_bursts` | Separate namespace from iacobus burst_npys |
| **jakob-mbp** | `~/Data/Faber2026/dsa110/DSA_bursts` | Offline replica (gap-synced Phase 3; moved from `~/Developer/dsa110-local-data/DSA_bursts` 2026-06-30 — see `DATA_SOURCES.md`) |

## Compute workspace (h17)

| Path | Size | Notes |
|------|------|-------|
| `/data/research/astrophysics/frbs/chime-dsa-codetections` | ~65G | CHIME docker / filterbanks / numpy / arc archive |
| `.../upchan_codetections` | 473M | Upchan products (5 targets; moved from `/data/jfaber/` 2026-06-27) |
| `.../archive/arc_trash_2026-06` | 36G | arc trash copy (moved from `/data/jfaber/` 2026-06-27); optional dedupe → iacobus |
| `.../scripts/upchannelize_chime.py` | 12K | Baseband upchan worker (moved from `/data/jfaber/` 2026-06-27) |

`/data/jfaber/` is empty of codetection products as of 2026-06-27.

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
