# Phase 2 design — h23 → iacobus data drain

**Status:** designed → **Phase 2 executed 2026-06-25** (see [`MIGRATION_LOG.md`](MIGRATION_LOG.md))  
**Parent:** [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md)  
**Inventory:** [`machine_inventory.yaml`](../../machine_inventory.yaml) (`migration_map` phase-2 entries)  
**Sentinels:** [`codetections_manifest.yaml`](../../codetections_manifest.yaml)

Phase 2 moves unique codetection bytes off **h23** into **iacobus** (`~/Research/CHIME_DSA_Codetections`). jakob-mbp never transits bulk data. After every subtree passes verification, h23 sources become read-only until Phase 5 quarantine.

---

## Executive summary

**Most Phase 2 targets may already be satisfied.** iacobus holds 218G materialized data (2026-06-17 manifest: 1546 files, sentinel-verified). April 2026 consolidation ran `h23_resilient_transfer.sh` from jakob-mbp. Phase 2 is therefore **audit-first, rsync-second**: measure delta per `migration_map` id, transfer only gaps, then re-run sentinels.

**Blockers before any rsync:**

1. **iacobus → h23 SSH** — `ssh iacobus 'ssh h23 …'` fails today (`Could not resolve hostname h23`). Fix iacobus `~/.ssh/config` (ProxyJump `dsa110maas` or Tailscale) before unattended pulls.
2. **Path corrections** — legacy script paths ≠ current h23 layout (see §Path corrections).
3. **burst_npys reconciliation** — iacobus 47 `.npy` / 32 top-level entries vs h23 `chime_dsa_codetections/bursts` 68 `.npy`; manifest still claims 52 from defunct `h23:…/burst_npys`.

**Estimated net new transfer (upper bound, if audit finds full delta):** ~55G (bursts 7.5G + stokes 983M + scat archive 5.9G + burstprop delta ~16G + archive delta TBD). Likely **≪55G** after dedupe.

---

## Architecture

```
jakob-mbp (orchestrator, logs)
    │  ssh -A iacobus
    ▼
iacobus ──rsync──► h23:/media/ubuntu/ssd/jfaber/…
    │
    └──► ~/Research/CHIME_DSA_Codetections/   (data authority)
              └── CloudDocs mirror (bird upload; not blocking Phase 2)
```

- **Orchestrator:** jakob-mbp runs `scripts/migration/h23_to_iacobus.sh` (wrapper around per-id jobs).
- **Data plane:** rsync executes **on iacobus**, pulling from h23 (`-e 'ssh …'`). Bytes never touch jakob-mbp disk.
- **Auth:** SSH agent forwarding jakob-mbp → iacobus → h23 (same pattern as legacy `h23_resilient_transfer.sh`).
- **Logs:** `~/logs/h23_transfers/` on jakob-mbp (or `$FLITS_ROOT/logs/migration/`).

---

## Path corrections (legacy script vs 2026-06 inventory)

| migration_map id | Legacy `h23_resilient_transfer.sh` | Correct h23 source (2026-06 probe) | iacobus target |
|------------------|-------------------------------------|-------------------------------------|----------------|
| `h23_stokes_i_npys` | `chime_dsa_codetections/data/` | `…/data/stokes_I_npys` (983M) | `dsa_fullstokes_waterfalls/` — **verify**: manifest source was `data/dsa_fullstokes_waterfalls`; may already be merged |
| `h23_chime_bursts` | `burst_npys/` (**path gone**) | `…/chime_dsa_codetections/bursts` (7.5G, 68 `.npy`) | `burst_npys/` |
| `h23_scattering` | `chime_dsa_codetections/scattering/` | same (21 files, 24M) | `scattering_results/` (115 files, 86M on iacobus — likely superset) |
| `h23_dm_budget` | `…/dm_budget/` → `dm_budget/h23_dm_budget/` | same | `dm_budget/h23_dm_budget/` |
| `h23_dm` | *(not in legacy script)* | `…/dm/` (732K) | merge into `dm_budget/` |
| `h23_localizations` | `…/localizations/` | same | `metadata/` |
| `h23_old_chime_archive` | `OLD_CHIME_DSA_Codetections/` (top-level) | `archive/OLD_CHIME_DSA_Codetections` (79G) | `archive/OLD_CHIME_DSA_Codetections/` (105G on iacobus — dedupe before rsync) |
| `h23_burstprop_bursts` | `burstprop_paper/` | `burstprop_paper/bursts` (42G) | `archive/burstprop_paper/` (26G on iacobus — delta ~16G) |
| `h23_dsa110_scat_archive` | *(not in legacy script)* | `archive/dsa110-scat` (5.9G) | `archive/dsa110-scat/` (new on iacobus) |

**Out of scope on h23:** `nihari/`, `/dataz/dsa110/T3`, `dsa110-continuum`, `tools/`, `ada/`, `ay122b` archives.

---

## Phase 2 sub-phases

### 2.0 — Prerequisites (half day)

| Step | Action | Exit |
|------|--------|------|
| 2.0a | Add h23 SSH stanza on **iacobus** (`Host h23`, `ProxyJump dsa110maas`, keys) | `ssh iacobus 'ssh h23 hostname'` → `lxd110h23` |
| 2.0b | Smoke rsync 1 MiB test file h23→iacobus | checksum match |
| 2.0c | Refresh `codetections_manifest.yaml` on iacobus (file counts + sentinels) | baseline for diff |

### 2.1 — Audit (per migration_map id)

Run from jakob-mbp:

```bash
python scripts/migration/audit_h23_delta.py --json reports/phase2_audit.json
```

For each phase-2 `migration_map` entry, record:

| Field | Meaning |
|-------|---------|
| `source_files` / `source_bytes` | h23 side |
| `target_files` / `target_bytes` | iacobus side |
| `overlap_files` | basename intersection |
| `missing_on_target` | need rsync |
| `extra_on_target` | keep; document |
| `sentinel_ok` | SHA-256 prefix vs `codetections_manifest.yaml` |
| `recommendation` | `skip` / `rsync_delta` / `reconcile` / `dedupe_first` |

**Expected audit outcomes (2026-06-25 probes):**

| id | Likely recommendation |
|----|------------------------|
| `h23_stokes_i_npys` | `skip` or `rsync_delta` — iacobus already 58 files / 6.0G waterfalls |
| `h23_scattering` | `skip` — iacobus superset |
| `h23_dm_budget`, `h23_dm`, `h23_localizations` | `skip` or tiny delta |
| `h23_chime_bursts` | **`reconcile`** — merge 68 h23 `.npy` into `burst_npys`; resolve naming (subdir vs flat) |
| `h23_old_chime_archive` | **`dedupe_first`** — iacobus 105G vs h23 79G; sentinel `freya_…pkl` |
| `h23_burstprop_bursts` | `rsync_delta` — ~16G gap |
| `h23_dsa110_scat_archive` | `rsync_delta` — 5.9G new |

### 2.2 — Transfer waves (smallest risk first)

Execute only ids where audit ≠ `skip`. Order:

| Wave | ids | ~size | Notes |
|------|-----|-------|-------|
| **A** | `h23_localizations`, `h23_dm`, `h23_scattering` | <25M | verify-only if audit says skip |
| **B** | `h23_dm_budget` | 329M | rsync into `dm_budget/h23_dm_budget/` |
| **C** | `h23_stokes_i_npys` | 983M | `--ignore-existing` if waterfalls overlap |
| **D** | `h23_chime_bursts` | 7.5G | **after** reconciliation policy (§burst_npys) |
| **E** | `h23_dsa110_scat_archive` | 5.9G | new tree under `archive/` |
| **F** | `h23_burstprop_bursts` | ≤16G delta | `--partial` resume |
| **G** | `h23_old_chime_archive` | 0–79G | **last**; dedupe policy required |

Each wave:

```bash
scripts/migration/h23_to_iacobus.sh --id h23_dm_budget --dry-run
scripts/migration/h23_to_iacobus.sh --id h23_dm_budget
python scripts/migration/audit_h23_delta.py --id h23_dm_budget  # post verify
```

Update `machine_inventory.yaml`: `status: completed`, `completed_utc`, `notes` with file/byte counts.

### 2.3 — burst_npys reconciliation policy

**Problem:** Three namespaces:

- h23 `chime_dsa_codetections/bursts/` — 68 `.npy`, burst-named subdirs
- iacobus `burst_npys/` — 47 `.npy`, 21G (partial vs manifest 52)
- manifest sentinel `burst_npys/I_240224aaad_mayra.npy` — may not exist on either side

**Policy (proposed):**

1. Inventory both trees → `reports/burst_npys_reconcile.csv` (path, size, sha256, burst nickname).
2. **Union** into iacobus `burst_npys/` with flat or `{nickname}/` layout — pick one; default **flat** to match existing iacobus layout.
3. Never delete h23 or iacobus files during reconcile; duplicates → `burst_npys/_dedupe_candidates/` on iacobus with manifest note.
4. Refresh `codetections_manifest.yaml` `burst_npys` sentinels from a file that exists post-merge.
5. Gate: `file_count ≥ 68` and sentinel pass before marking `h23_chime_bursts` completed.

### 2.4 — OLD_CHIME dedupe policy

iacobus `archive/OLD_CHIME_DSA_Codetections` (105G) vs h23 `archive/OLD_CHIME` (79G).

1. Compare sentinel `archive/OLD_CHIME_DSA_Codetections/polcal_fils/freya_230325aaag_fullstokes_interp.pkl` (SHA in manifest).
2. If sentinel passes on iacobus → **`skip` rsync**; mark completed with note "iacobus canonical".
3. If sentinel fails → path-level diff; rsync only `missing_on_target` from h23.
4. Phase 3 (`arc_old_chime_dedupe`) handles arc 77G — do not conflate with Phase 2.

---

## Verification gates (per id)

1. **Post-rsync:** `audit_h23_delta.py --id …` → `missing_on_target: 0`
2. **Sentinel:** first 64 MiB SHA-256 matches `codetections_manifest.yaml` (or updated sentinel after reconcile)
3. **Inventory:** `migration_map[].status: completed`
4. **Global:** `python scripts/query_machine_inventory.py --migration-map --migration-status pending | rg '^h23_'` → empty

---

## Scripts to add (Phase 2 implementation — not yet written)

| Script | Purpose |
|--------|---------|
| `scripts/migration/h23_to_iacobus.sh` | Per-id rsync wrapper; reads paths from `machine_inventory.yaml` via yq |
| `scripts/migration/audit_h23_delta.py` | Remote file counts + optional sha256 prefix; emits JSON report |
| `scripts/migration/reconcile_burst_npys.py` | Implements §2.3 policy |

Legacy reference: `~/Developer/research-holding/caltech/ovro/dsa110/chime_dsa_codetections/h23_resilient_transfer.sh` (paths stale; do not run verbatim).

---

## Disk budget

| Host | Free (2026-06-25) | Phase 2 need |
|------|-------------------|--------------|
| iacobus | 209G | ≤55G upper bound; likely <20G net |
| h23 | 814G | read-only source |
| jakob-mbp | 25G | **0** — orchestration only |

---

## Rollback

- Rsync is additive (`--ignore-existing` default for reconcile waves).
- h23 sources untouched until Phase 5 `h23_jfaber_root` quarantine.
- Restore: re-run rsync from h23; no deletes in Phase 2.

---

## Open decisions

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | iacobus h23 SSH route | ProxyJump `dsa110maas` vs Tailscale IP | ProxyJump (matches jakob-mbp) |
| D2 | burst_npys layout | flat vs per-burst subdirs | flat (match iacobus today) |
| D3 | OLD_CHIME canonical | iacobus vs h23 | iacobus if sentinel passes |
| D4 | dsa110-scat on iacobus | full tree vs git-only | full 5.9G archive copy; code stays GitHub |

---

## Checklist → Phase 3

Phase 2 complete when all `h23_*` migration_map entries (except `h23_nihari`, `h23_raw_t3`, `h23_jfaber_root`) are `completed` or `skipped` with audit JSON on disk, and `codetections_manifest.yaml` refreshed on iacobus.
