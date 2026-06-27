# Phase 4 design — h17 compute / staging

**Status:** executed (2026-06-26) — inventory + decisions complete; optional copy waves deferred  
**Parent:** [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md)  
**Inventory:** [`machine_inventory.yaml`](../../machine_inventory.yaml) (`migration_map` phase-4 entries, `machines.h17`)  
**Sentinels:** [`codetections_manifest.yaml`](../../codetections_manifest.yaml)

Phase 4 keeps **h17** as an active OVRO compute target (CHIME docker, 2×2080 Ti, baseband workspace). It audits staging paths, optional arc-trash promotion to iacobus, and empty stub cleanup. **No deletes and no bulk transfers** in this design pass.

Phase 2 (h23 → iacobus) and Phase 3 (arc ↔ iacobus dedupe) may run in parallel; Phase 4 does **not** touch h23/iacobus rsync paths.

---

## Executive summary

**Phase 4 is audit-first, keep-compute-second, optional-copy-third.**

Live probe (2026-06-25, `reports/phase4_audit.json`):

| Audit id | h17 (2026-06-25) | iacobus compare | Likely outcome |
|----------|------------------|-----------------|----------------|
| `h17_compute_workspace` | 339 f / 29G | — | **keep** on h17 |
| `h17_upchan_products` | 11 f / 473M | — | **keep**; promote finished `.npy` → arc when stable |
| `h17_arc_archive_copy` | 1924 f / 36G | target **missing** | **dedupe_then_copy** after Phase 3 OLD_CHIME policy |
| `h17_ubuntu_stub` | 0 f / empty tree | — | **remove_stub** (empty `chime_singlebeam` subdir only) |
| `h17_chime_singlebeam_empty` | 0 f | — | **remove_stub** |
| `iacobus_chime_canfar_archive` | — | 725 f / 2.7G vs research `archive/` 105G+ | **dedupe_into_research** |

**Blockers before any h17 → iacobus copy:**

1. **Phase 3 OLD_CHIME canonical** — iacobus sentinel passes; do not bulk-copy 36G arc trash until overlap with `archive/OLD_CHIME_DSA_Codetections` is mapped.
2. **Phase 2 rsync in flight** — another agent may be running h23→iacobus; Phase 4 orchestration must not invoke those scripts.
3. **Promotion policy** — upchan products and compute workspace baseband are **working sets**; only finished artifacts move to arc/iacobus.

---

## Architecture

```
jakob-mbp (orchestrator, audit JSON, git push)
    │  ssh h17
    ▼
h17 (/data, 1.8T free)
    ├── compute workspace  …/chime-dsa-codetections  (29G, keep)
    ├── upchan products    /data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections (473M, keep)
    ├── arc trash archive  .../archive/arc_trash_2026-06 (36G; moved from /data/jfaber/ 2026-06-27)
    └── empty stubs        /data/ubuntu/chime-dsa-codetections (remove)

iacobus (data authority) ◄── optional deduped copy of arc trash
arc (CANFAR) ◄── promote stable .npy / fit inputs only
```

- **Orchestrator:** jakob-mbp runs `scripts/migration/audit_h17_delta.py`.
- **Data plane:** read-only `ssh h17 find/du`; optional `ssh iacobus` for compare targets.
- **Auth:** Tailscale SSH alias `h17` → `100.85.172.12`, user `ubuntu`.
- **Reports:** `reports/phase4_audit.json`.

---

## Path table

| migration_map / audit id | h17 path | Compare target | Action |
|--------------------------|----------|----------------|--------|
| `h17_compute_workspace` | `/data/research/astrophysics/frbs/chime-dsa-codetections` | — | **keep**; canonical docker workspace |
| `h17_upchan_products` | `/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections` | arc `CHIME_bursts` (Phase 3) | **keep**; gap-promote when CANFAR needs file |
| `h17_arc_archive_copy` | `.../archive/arc_trash_2026-06` | iacobus `archive/arc_trash_2026-06` | **keep** on h17; optional deduped rsync |
| `h17_ubuntu_stub` | `/data/ubuntu/chime-dsa-codetections` | compute workspace | **remove** empty stub |
| `h17_chime_singlebeam_empty` | `/data/jfaber/chime_singlebeam` | workspace `chime_singlebeam/` (14G) | **remove** empty dir |
| `iacobus_chime_canfar_archive` | — | iacobus `Archives/CHIME_canfar` vs `Research/.../archive` | dedupe into research archive |

**Compute workspace breakdown (2026-06-25):**

| Subdir | Size | Role |
|--------|------|------|
| `chime_singlebeam/` | 14G | staged baseband `.h5` for docker |
| `numpy/` | 8.4G | intermediate arrays |
| `dsa_filterbanks/` | 6.6G | DSA-side filterbanks |
| `diagnostics/` | 25M | logs/plots |
| `bin/`, `scripts/` | <200K | docker wrapper + helpers |

**arc trash archive breakdown (2026-06-25):**

| Subdir | Size | Notes |
|--------|------|-------|
| `fullstokes_pkl/` | 25G | largest; overlap risk with iacobus OLD_CHIME pickles |
| `stokes_cubes_npy/` | 4.3G | |
| `other_data_npy/` | 3.4G | |
| `other_data_pkl/` | 1.9G | |
| `trashed_directories/` | 710M | **new vs inventory** (2026-06 probe) |
| `plots/`, `notebooks/`, … | <1G each | |

---

## Phase 4 sub-phases

### 4.0 — Prerequisites (half day)

| Step | Action | Exit |
|------|--------|------|
| 4.0a | Confirm `ssh h17 hostname` → `lxd110h17` | BatchMode OK |
| 4.0b | Confirm docker image present | `chimefrb/baseband-analysis:latest` (~8.6G) |
| 4.0c | Phase 3 OLD_CHIME policy documented | iacobus sentinel pass cited |
| 4.0d | Confirm Phase 2 rsync **not** invoked from Phase 4 scripts | no `h23_to_iacobus.sh` calls |

### 4.1 — Audit

```bash
python scripts/migration/audit_h17_delta.py --stdout
# single id:
python scripts/migration/audit_h17_delta.py --id h17_arc_archive_copy
```

Per entry: `h17` / `iacobus` file counts + bytes, optional `h17_children`, `recommendation`.

### 4.2 — Keep compute workspace + upchan (no migration)

1. Document canonical paths in `machine_inventory.yaml` (already tagged `migration_target: h17`).
2. FLITS repo on h17 (`~/Developer/repos/.../dsa110-FLITS`, ~909M) — optional docker checkout; **GitHub/jakob-mbp canonical**.
3. Upchan products: five targets complete (casey, whitney, phineas, mahi, isha); pull to jakob-mbp only when needed for analysis — not bulk.

### 4.3 — arc trash optional copy (after Phase 3 dedupe)

Policy for `h17_arc_archive_copy`:

1. Create iacobus target only after gap analysis: `archive/arc_trash_2026-06/`.
2. Hash-map `fullstokes_pkl/` and `other_data_pkl/` against iacobus `archive/OLD_CHIME_DSA_Codetections/` before rsync.
3. rsync from **h17 → iacobus** (not via jakob-mbp); `--ignore-existing` default.
4. Gate: no net iacobus growth without written size budget; cite `phase4_audit.json`.

**Do not execute in Phase 4.1 design pass.**

### 4.4 — Empty stub removal

| Path | Probe | Action |
|------|-------|--------|
| `/data/ubuntu/chime-dsa-codetections` | 0 files; empty `chime_singlebeam/` subdir | `rmdir` stub tree after human sign-off |
| `/data/jfaber/chime_singlebeam` | 0 bytes | remove; real data under compute workspace |

**Do not execute in Phase 4.1 design pass.**

### 4.5 — iacobus CHIME_canfar dedupe

`iacobus_chime_canfar_archive` (725 files, 2.7G under `~/Archives/CHIME_canfar`):

1. Inventory vs `Research/CHIME_DSA_Codetections/archive/`.
2. Merge unique CANFAR session exports; duplicates → `_dedupe_candidates/`.
3. Independent of h17 bytes; can run parallel with 4.3.

---

## Verification gates

1. **Post-audit:** `reports/phase4_audit.json` exists; every Phase 4 audit id present.
2. **Inventory match:** h17 location sizes within ~5% of live `du` (update notes if drift).
3. **No Phase 2 collision:** audit script and docs explicitly exclude h23 paths.
4. **Promotion:** any h17 → arc/iacobus transfer requires Phase 3 quota check if arc-bound.
5. **Inventory status:** update `migration_map[].status` only after executed waves — not in 4.1.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/migration/audit_h17_delta.py` | h17 `find/du` + iacobus compare; emits JSON |
| `scripts/migration/audit_arc_delta.py` | Phase 3 (arc side) |
| `scripts/migration/audit_h23_delta.py` | Phase 2 (h23 side) — **do not run transfers from Phase 4** |

Future (not in scope for 4.1):

| Script | Purpose |
|--------|---------|
| `scripts/migration/h17_to_iacobus.sh` | Optional arc-trash rsync after dedupe |
| `scripts/migration/promote_h17_upchan.sh` | scp/vcp finished upchan → arc or jakob replica |

---

## Disk budget

| Host | Relevant size (2026-06-25) | Phase 4 constraint |
|------|----------------------------|-------------------|
| h17 `/data` | 11T used / 1.8T free | keep 29G workspace + 36G archive; stubs negligible |
| iacobus | 209G free | optional +36G only after dedupe; no blind copy |
| jakob-mbp | 25G free | orchestration + selective scp only |

---

## Rollback

- Phase 4.1 is read-only (`ssh find/du`).
- Stub removal and rsync are reversible (h17 archive unchanged until explicit copy).
- Restore stubs: recreate empty dirs; restore archive from h17 if iacobus copy exists.

---

## Open decisions

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | arc trash canonical | h17 vs iacobus | **h17 until dedupe**; iacobus after gap map |
| D2 | upchan promotion | arc vs iacobus burst_npys | **arc** if CANFAR fit input; else jakob replica |
| D3 | ubuntu stub removal timing | before vs after Phase 5 | after audit sign-off; low risk |
| D4 | CHIME_canfar merge target | flat under `archive/` vs subdir | `archive/chime_canfar/` |
| D5 | trashed_directories (710M) | copy vs skip | include in hash-map with fullstokes_pkl |

---

## Checklist → Phase 5

Phase 4 complete when:

- `h17_compute_workspace`, `h17_upchan_products` marked **keep** with audit JSON
- `h17_arc_archive_copy` either **completed** (deduped copy) or **skipped** with documented overlap
- Empty stubs removed (post-approval) or documented as retained
- `iacobus_chime_canfar_archive` deduped or scoped
- `migration_map` phase-4 entries updated with audit citation
- Retired-host drain (h23/hpcc) unaffected — Phase 5 quarantine follows
