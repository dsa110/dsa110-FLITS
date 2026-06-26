# Deferred migration work (post Phase 5)

**Status:** Phases 1–5 closed on `main` @ `a8c2b004` (PR #67).  
**Inventory gate:** `python scripts/query_machine_inventory.py --migration-status pending` → empty.  
**Policy:** move-only; no bulk transfers without explicit approval. See [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md).

Optional follow-ups below are **skipped** in `machine_inventory.yaml` — not blockers for the 4-host model.

---

## D1 — `CHIME_bursts` cross-namespace reconcile

| Side | Path | Size (audit 2026-06-25) |
|------|------|-------------------------|
| arc (fits) | `arc:…/data/CHIME_bursts` | 60 f / 6.3 G |
| iacobus (archive) | `~/Research/CHIME_DSA_Codetections/burst_npys` | 218 f / 30.7 G |

**Finding:** arc holds fit-ready `.npy` under `dmphase/` and `dmtransform/` namespaces (24 codetection basenames in [`reports/phase3_chime_basename_inventory.csv`](../../reports/phase3_chime_basename_inventory.csv)); iacobus `burst_npys` uses a mixed nickname/TNS namespace — **zero basename overlap** with arc inventory rows.

**Next step (read-only first):**

```bash
python scripts/migration/audit_arc_delta.py --stdout   # refresh arc vs iacobus counts
# Manual: map nickname ↔ TNS ↔ arc basename per burst via configs/bursts.yaml + data-manifest.csv
```

**Do not:** bulk rsync arc → iacobus without quota check (~200 G arc cap).

---

## D2 — iacobus `CHIME_canfar` archive merge

| Source | Target (planned) | Size |
|--------|------------------|------|
| `~/Archives/CHIME_canfar` (iacobus) | `~/Research/CHIME_DSA_Codetections/archive/chime_canfar/` | 725 f / 2.7 G |

**Finding (2026-06-26):** zero basename overlap vs `Research/…/archive` (937 f / 178 G). Merge is **additive**, not dedupe — unique CANFAR session exports (includes 3 `analysis_*` session dirs with spaces in names).

**Next step (after approval):**

```bash
# on iacobus — move-only, verify counts before/after
mkdir -p ~/Research/CHIME_DSA_Codetections/archive/chime_canfar
rsync -av --remove-source-files ~/Archives/CHIME_canfar/ \
  ~/Research/CHIME_DSA_Codetections/archive/chime_canfar/
# then: python scripts/query_machine_inventory.py --migration-map --id iacobus_chime_canfar_archive
```

**Inventory id:** `iacobus_chime_canfar_archive` (`status: skipped`).

---

## D3 — h17 arc trash → iacobus

| Source | Target (planned) | Size |
|--------|------------------|------|
| h17 `/data/jfaber/arc_archive_2026-06` | iacobus `archive/arc_trash_2026-06/` | 1924 f / 36 G |

**Finding (2026-06-26 sample):** 25 `.pkl` basenames under h17 `fullstokes_pkl/`, `other_data_pkl/`, `processed_spectra_pkl/` vs 24 in iacobus `OLD_CHIME_DSA_Codetections/` — **zero basename overlap**, zero size fingerprint overlap. Numeric vs nickname naming schemes differ.

**Next step (read-only wave):**

```bash
python scripts/migration/audit_h17_delta.py --stdout
# Full hash-map script not yet implemented (h17_to_iacobus.sh planned in PHASE4_DESIGN.md)
```

**Do not:** rsync 36 G until hash-map confirms unique bytes vs iacobus `OLD_CHIME`.

**Inventory id:** `h17_arc_archive_copy` (`status: skipped`).

---

## D4 — Docs: CANFAR GPU access

Local-only note through 2026-06-25; committed separately in PR (see `DATA_SOURCES.md` § CANFAR compute and GPU access).

Smoke test (verified 2026-06-25):

```bash
canfar create headless skaha/astroml-cuda:latest --gpu 1 -n gpu-smoke-test -- nvidia-smi
```

---

## Quick reference

| id | Risk if rushed | Approval needed |
|----|----------------|-----------------|
| D1 CHIME_bursts | wrong namespace / duplicate fits | yes — reconcile map first |
| D2 CHIME_canfar | none (additive, iacobus-local) | yes — move-only merge |
| D3 h17 arc trash | 36 G duplicate storage | yes — hash-map wave |
| D4 GPU docs | none | no (docs only) |

**Audit artifacts:** `reports/phase3_audit.json`, `reports/phase4_audit.json`, `reports/phase3_chime_basename_inventory.csv`.

**Related closeouts:** [`PHASE4_CLOSEOUT.md`](PHASE4_CLOSEOUT.md), [`PHASE5_CLOSEOUT.md`](PHASE5_CLOSEOUT.md).
