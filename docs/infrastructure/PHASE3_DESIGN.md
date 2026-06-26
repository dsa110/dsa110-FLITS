# Phase 3 design — arc ↔ iacobus dedupe

**Status:** executed (2026-06-25T10:30:22Z)  
**Parent:** [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md)  
**Inventory:** [`machine_inventory.yaml`](../../machine_inventory.yaml) (`migration_map` phase-3 + phase-1 arc dedupe entries)  
**Sentinels:** [`codetections_manifest.yaml`](../../codetections_manifest.yaml)

Phase 3 reconciles **arc VOSpace** burst products and archive trees with **iacobus** (data authority) and **jakob-mbp** (small local replicas). No bulk upload of the 218G iacobus tree to arc — arc quota is ~200G and `baseband_morphologies` already holds ~113G.

Phase 2 (h23 → iacobus) may run in parallel; Phase 3 does not depend on Phase 2 completion except where both touch `OLD_CHIME_DSA_Codetections` policy.

---

## Executive summary

**Phase 3 is audit-first, dedupe-second, upload-never-without-quota-check.**

Live probe (2026-06-25, `reports/phase3_audit.json`):

| Audit id | arc | iacobus / jakob-mbp | Likely outcome |
|----------|-----|---------------------|----------------|
| `arc_dsa_bursts` | 33 f / 2.9G (deep) | jakob local 24 f / 2.8G | **arc authoritative** for CANFAR fits; sync jakob replica gaps only |
| `arc_chime_bursts` | ~5.9G (shallow) | iacobus `burst_npys` 99 f / 21G | **reconcile** — different layout/namespaces |
| `arc_old_chime_dedupe` | ~77.6G | iacobus archive 384 f / 105G | **iacobus canonical** — manifest sentinel passes on iacobus only |
| `arc_flits_checkout` | ~211M top-level | jakob repo 4.9G | **diff vs GitHub**; stop arc-side dev |
| `arc_codetection_flits_tree` | ~5G legacy tree | jakob/GitHub canonical | **diff vs GitHub**; not canonical |

**Blockers before any arc upload:**

1. **Quota math** — 218G iacobus + 77G arc OLD_CHIME overlap ≠ fit in 200G quota; upload only gap files after dedupe.
2. **VOS name constraints** — `vls` cannot traverse nodes with spaces/colons in names (CANFAR session dirs under `CHIME_bursts/dmphase/`); shallow byte totals still valid via dir aggregate sizes.
3. **OLD_CHIME layout drift** — iacobus uses `polcal_fils/`; arc uses `CHIME_pkl/` with different filenames; sentinel path on arc may not exist — compare by hash after path mapping, not path equality.

---

## Architecture

```
jakob-mbp (orchestrator, audit JSON, small DSA replica ≤5G)
    │  vls/vcat (~/.ssl/cadcproxy.pem)
    ▼
arc VOSpace ──dedupe──► iacobus ~/Research/CHIME_DSA_Codetections
    │                      │
    │ CANFAR compute       └── iCloud mirror (Phase 4+)
    └── DSA_bursts / CHIME_bursts (fit inputs)
```

- **Orchestrator:** jakob-mbp runs `scripts/migration/audit_arc_delta.py`.
- **Data plane:** read-only `vls`/`vcat` on arc; `ssh iacobus find/du` for local authority.
- **Auth:** X509 proxy `~/.ssl/cadcproxy.pem` (exp **2026-07-18**).
- **Reports:** `reports/phase3_audit.json`.

---

## Path table

| migration_map / audit id | arc path | Compare target | Action |
|--------------------------|----------|----------------|--------|
| `arc_dsa_bursts` | `arc:…/data/DSA_bursts` | jakob `~/Developer/dsa110-local-data/DSA_bursts` | gap sync to local replica; arc stays primary for CANFAR |
| `arc_chime_bursts` | `arc:…/data/CHIME_bursts` | iacobus `burst_npys/` | basename + size reconcile; no blind rsync |
| `arc_old_chime_dedupe` | `arc:…/OLD_CHIME_DSA_Codetections` | iacobus `archive/OLD_CHIME_DSA_Codetections` | pick canonical via sentinel; **iacobus wins** if sentinel passes |
| `arc_flits_checkout` (phase 1) | `arc:home/jfaber/dsa110-FLITS` | GitHub / jakob-mbp repo | diff; rescue CANFAR-only commits via PR |
| `arc_codetection_flits_tree` (phase 1) | `arc:…/chime_dsa_codetections/FLITS` | GitHub / jakob-mbp repo | dedupe; ~5G not canonical |
| `arc_flits_run` | `arc:home/jfaber/flits_run` | — | **skipped** (ephemeral CANFAR scratch) |

**Tiering reminder (from 4-host plan):**

| Asset | Primary | arc role |
|-------|---------|----------|
| Burst `.npy` for fits | arc `DSA_bursts` / `CHIME_bursts` | keep |
| Pickles, waterfalls, archive | iacobus | do not bulk mirror to arc |
| FLITS code | jakob-mbp + GitHub | arc checkouts → dedupe only |

---

## Phase 3 sub-phases

### 3.0 — Prerequisites (half day)

| Step | Action | Exit |
|------|--------|------|
| 3.0a | Confirm `~/.ssl/cadcproxy.pem` valid | `vls arc:home/jfaber` lists |
| 3.0b | Record arc quota headroom | sum shallow `vls -l` under `baseband_morphologies` + `home/jfaber` |
| 3.0c | Phase 2 OLD_CHIME policy aligned | iacobus sentinel pass documented (Phase 2 §2.4) |

### 3.1 — Audit

```bash
python scripts/migration/audit_arc_delta.py --sentinel --stdout
# optional deep walk on small trees:
python scripts/migration/audit_arc_delta.py --id arc_dsa_bursts --deep
```

Per entry, record: `arc` / `local` / `iacobus` file counts + bytes, `recommendation`, optional `sentinel` block.

**Default arc mode:** shallow (`vls -l` sum — directory lines include subtree bytes). Use `--deep` only for trees &lt;5G without space-in-name dirs.

### 3.2 — DSA_bursts reconcile

1. Deep audit arc vs jakob local (24 vs 33 `.npy`-class files).
2. **Union** missing basenames into jakob replica (≤5G budget).
3. Do **not** delete arc files; FLITS configs keep arc URIs for CANFAR.
4. Gate: jakob local ⊇ arc corrected burst set OR documented subset for offline-only bursts.

### 3.3 — CHIME_bursts reconcile

1. Inventory arc `CHIME_bursts/{dmphase,dmtransform}` vs iacobus `burst_npys/` → CSV (basename, bytes, nickname).
2. arc dirs with spaces (CANFAR session outputs) — manual manifest or Science Portal FS listing.
3. **No upload** from iacobus to arc unless a specific `.npy` is required on CANFAR and missing on arc.

### 3.4 — OLD_CHIME canonical pick

Policy (extends Phase 2 §2.4):

1. Sentinel `freya_230325aaag_fullstokes_interp.pkl` — first 64 MiB SHA-256 vs manifest.
2. **2026-06-25 probe:** iacobus **PASS**; arc path `polcal_fils/…` **absent** (arc has `CHIME_pkl/freya_*` variants).
3. If iacobus sentinel passes → mark `arc_old_chime_dedupe` **iacobus_canonical**; arc tree is subset/different layout — dedupe not bulk copy.
4. Optional: hash-map arc `CHIME_pkl/*.pkl` against iacobus `polcal_fils/` + `burst_pickles/` before any arc deletion (Phase 5+).
5. **Do not** rsync 105G iacobus → arc or 77G arc → iacobus without gap analysis.

### 3.5 — Phase 1 arc FLITS dedupe (can start in Phase 3 audit)

| id | Steps |
|----|-------|
| `arc_flits_checkout` | `git diff` arc export vs `jakob-mbp`; cherry-pick CANFAR-only commits; mark completed |
| `arc_codetection_flits_tree` | same; confirm no unique data under `scintillation/` not in GitHub |

---

## Verification gates

1. **Post-audit:** `reports/phase3_audit.json` exists; every phase-3 `migration_map` id present.
2. **Sentinel:** `arc_old_chime_dedupe.sentinel.iacobus_ok == true` before declaring iacobus canonical.
3. **Quota:** any proposed arc upload has written size budget &lt; remaining quota (200G − current usage).
4. **Inventory:** update `migration_map[].status` / `notes` only after human review — not in automated Phase 3.1.
5. **Global:** no `vcp` bulk transfers executed without explicit Phase 3.2+ approval.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/migration/audit_arc_delta.py` | arc `vls` vs iacobus/jakob; optional `--sentinel`; emits JSON |
| `scripts/migration/audit_h23_delta.py` | Phase 2 parallel (h23 side) |

Legacy: arc uploads via CANFAR container FS — not CLI `vcp` — when paths contain spaces.

---

## Disk / quota budget

| Host | Relevant size (2026-06-25) | Phase 3 constraint |
|------|----------------------------|-------------------|
| arc | ~113G `baseband_morphologies`; quota ~200G | **≤~87G headroom** — no 218G upload |
| iacobus | 218G data authority; 209G free | source for archive canonical |
| jakob-mbp | 25G free | DSA replica ≤5G only |

---

## Rollback

- Phase 3.1 is read-only (vls/vcat/ssh find).
- No deletes on arc or iacobus in Phase 3 design scope.
- Restore: re-run audit; revert any future gap-sync with rsync from arc.

---

## Open decisions

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | OLD_CHIME canonical | arc vs iacobus | **iacobus** (sentinel pass 2026-06-25) |
| D2 | arc CHIME_bursts vs iacobus burst_npys | merge vs separate namespaces | keep separate; map by burst nickname |
| D3 | jakob DSA replica scope | 24 vs 33 files | sync missing 9 from arc |
| D4 | arc FLITS trees | delete vs quarantine | quarantine arc paths after GitHub diff; no rm |
| D5 | Upload direction | arc←iacobus vs arc→iacobus | **neither bulk**; gap-only arc←iacobus if CANFAR needs a file |

---

## Checklist → Phase 4

Phase 3 complete when:

- `arc_old_chime_dedupe`, `arc_dsa_bursts`, `arc_chime_bursts` audited with signed-off canonical policy
- Phase 1 ids `arc_flits_checkout`, `arc_codetection_flits_tree` diffed vs GitHub
- `migration_map` phase-3 entries updated to `completed` or `skipped` with audit JSON citation
- No unplanned arc quota growth
