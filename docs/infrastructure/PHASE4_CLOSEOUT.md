# Phase 4 closeout — h17 compute / staging inventory

**Date:** 2026-06-26  
**Branch:** `migration/phase1-easy-wins`  
**Audit:** `reports/phase4_audit.json` (regenerated 2026-06-26T18:26:26Z)

## Summary

Phase 4 inventory and skip/copy decisions are complete. h17 is reachable via Tailscale SSH (`lxd110h17`). No bulk transfers executed — move-only policy preserved.

Prior attempt (agent 3c73c828) failed on h17 PING timeout; retry succeeded via SSH BatchMode.

## h17 connectivity

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 h17 hostname
# → lxd110h17
```

## Actions / decisions

| audit id | migration_map id | decision | result |
|----------|------------------|----------|--------|
| `h17_compute_workspace` | `h17_compute_workspace` | **keep** | **completed** — 339f/29G docker workspace on h17 |
| `h17_upchan_products` | `h17_upchan_products` | **keep** | **completed** — 11f/473M; five targets on h17 |
| `h17_ubuntu_stub` | `h17_ubuntu_stub` | remove stub | **completed** — path already absent; no `rmdir` needed |
| `h17_chime_singlebeam_empty` | `h17_chime_singlebeam_empty` | remove stub | **completed** — path already absent |
| `iacobus_chime_canfar_archive` | `iacobus_chime_canfar_archive` | dedupe vs research | **skipped** — 725f/2.7G; zero basename overlap with `Research/.../archive` (937f/178G); unique CANFAR tree retained at `Archives/CHIME_canfar`; move-only merge to `archive/chime_canfar/` deferred (D4) |
| `h17_arc_archive_copy` | `h17_arc_archive_copy` | dedupe_then_copy | **skipped** — 1924f/36G on h17; iacobus target missing; basename+size sample of 25 pkls vs `OLD_CHIME_DSA_Codetections` shows no exact duplicates (numeric vs nickname naming); hash-map wave deferred |

### iacobus CHIME_canfar compare (2026-06-26)

- Source: `/Users/iacobus/Archives/CHIME_canfar` — 725 files, 2.7G
- Compare: `/Users/iacobus/Research/CHIME_DSA_Codetections/archive` — 937 files, 178G
- Basename overlap: **0** (including `OLD_CHIME_DSA_Codetections/` subtree)
- Conclusion: not duplicate bytes; merge is additive, not dedupe — deferred

### h17 arc trash vs iacobus OLD_CHIME (2026-06-26 sample)

- h17: 25 `.pkl` under `fullstokes_pkl/`, `other_data_pkl/`, `processed_spectra_pkl/`
- iacobus OLD_CHIME: 24 distinct `.pkl` basenames (nickname_TNS scheme)
- Basename overlap: **0**; sampled size fingerprint overlap: **0**
- Conclusion: copy deferred until full hash-map; h17 remains canonical for arc trash

## Stub removal (4.4)

Planned `rmdir` targets were already absent:

- `/data/ubuntu/chime-dsa-codetections` — does not exist
- `/data/jfaber/chime_singlebeam` — does not exist

## Docs / inventory updates

- `machine_inventory.yaml` — six phase-4 `migration_map` entries updated
- `reports/phase4_audit.json` — refreshed via `audit_h17_delta.py`
- `MIGRATION_LOG.md` — Phase 4 entry appended
- `PHASE4_DESIGN.md` — status → executed (partial transfers deferred)

## Verification

```bash
python scripts/migration/audit_h17_delta.py --stdout
python scripts/query_machine_inventory.py --migration-map --migration-status pending
python scripts/query_machine_inventory.py --migration-map --phase 4
```

**Expected:** zero `pending` phase-4 entries after closeout.

## Deferred (not Phase 4 scope)

- Move-only merge `Archives/CHIME_canfar` → `archive/chime_canfar/` (when D4 approved)
- h17 → iacobus rsync of `arc_archive_2026-06` after hash-map (`h17_to_iacobus.sh`, not yet implemented)
- Upchan `.npy` promotion to arc when CANFAR fit inputs stabilize
