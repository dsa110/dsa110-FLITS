# Migration execution log

Chronological record of executed `migration_map` actions. Plan: [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md).

## 2026-06-25 — Phase 1 easy wins

| id | action | result |
|----|--------|--------|
| `hpcc_joint_fit_json` | migrate | **done** — 22 JSON already on jakob-mbp; `sha256sum` identical to hpcc pre-quarantine |
| `hpcc_flits_tree` | quarantine | **done** — `mv /home/jfaber/flits → /home/jfaber/_quarantine/flits-20260625` |
| `dsacamera_decommission` | decommission | **done** — no codetection content |
| `arc_flits_run` | skip | ephemeral CANFAR scratch |

## 2026-06-25 — Phase 2 h23 → iacobus

**Prerequisite:** iacobus `~/.ssh/config.d/10-ovro-h23.conf` (ovro → dsa110maas → h23); agent forward from jakob-mbp.

| id | action | result |
|----|--------|--------|
| `h23_stokes_i_npys` | skip | iacobus superset; sentinel PASS |
| `h23_scattering` | skip | iacobus superset; sentinel PASS |
| `h23_dm` | skip | merged on iacobus; sentinel PASS |
| `h23_localizations` | skip | sentinel PASS |
| `h23_old_chime_archive` | skip | iacobus canonical (105G); freya sentinel PASS |
| `h23_dm_budget` | migrate | rsync 847f/329M → `dm_budget/h23_dm_budget/` |
| `h23_dsa110_scat_archive` | migrate | rsync 314f/5.9G → `archive/dsa110-scat/` |
| `h23_burstprop_bursts` | migrate | rsync; iacobus 183f/49G superset of h23 132f/42G |
| `h23_chime_bursts` | migrate | rsync reconcile; iacobus 218f/29G superset of h23 166f/7.5G |

Audit: `reports/phase2_audit.json` · scripts: `scripts/migration/{audit_h23_delta.py,h23_to_iacobus.sh}`

**Phase 2 exit:** all `h23_*` phase-2 entries `completed` or `skipped`. **Next:** Phase 3 arc ↔ iacobus dedupe; Phase 5 `h23_jfaber_root` quarantine after Phase 3.

## 2026-06-25 — Phase 3 arc ↔ iacobus dedupe

| id | action | result |
|----|--------|--------|
| `arc_old_chime_dedupe` | dedupe | **skipped** — iacobus canonical (105G); freya sentinel PASS |
| `arc_dsa_bursts` | migrate | **done** — vcp gap sync to jakob-mbp local replica |
| `arc_chime_bursts` | dedupe | **skipped** — separate namespaces; no bulk copy |
| `arc_flits_checkout` | dedupe | **done** — GitHub/jakob-mbp canonical |
| `arc_codetection_flits_tree` | dedupe | **done** — jakob superset |
| `arc_flits_run` | skip | ephemeral CANFAR scratch |

Audit: `reports/phase3_audit.json`

## 2026-06-25 — Phase 5 quarantine + decommission

| id | action | result |
|----|--------|--------|
| `h23_jfaber_root` | quarantine | **done** — partial mv to `/media/ubuntu/ssd/_quarantine/jfaber-drain-20260625/` (archive, burstprop_paper, chime_dsa_codetections ~137G); nihari + tools + dsa110-continuum remain at jfaber root; T3 untouched |
| `hpcc_flits_tree` | quarantine | **done** (Phase 1) — verified still at `_quarantine/flits-20260625` |
| `dsacamera_decommission` | decommission | **done** (Phase 1) |

Closeout: [`PHASE5_CLOSEOUT.md`](PHASE5_CLOSEOUT.md) · h23 restore: `/media/ubuntu/ssd/_quarantine/README.md`

**Phase 5 exit:** all retired-host `migration_map` entries `completed`, `skipped`, or `out_of_scope`. Phase 4 inventory closed 2026-06-26 (see below).

## 2026-06-26 — Phase 4 h17 compute / staging inventory

**Prerequisite:** `ssh h17 hostname` → `lxd110h17` (prior PING timeout; SSH OK on retry).

| id | action | result |
|----|--------|--------|
| `h17_compute_workspace` | keep | **completed** — 339f/29G docker workspace stays on h17 |
| `h17_upchan_products` | keep | **completed** — 11f/473M five-target upchan on h17 |
| `h17_ubuntu_stub` | skip | **completed** — path already absent; no rmdir |
| `h17_chime_singlebeam_empty` | skip | **completed** — path already absent |
| `iacobus_chime_canfar_archive` | dedupe | **skipped** — 725f/2.7G; 0 basename overlap vs Research/archive; merge deferred |
| `h17_arc_archive_copy` | dedupe | **skipped** — 1924f/36G; sample vs OLD_CHIME no exact dupes; hash-map copy deferred |

Audit: `reports/phase4_audit.json` (2026-06-26) · closeout: [`PHASE4_CLOSEOUT.md`](PHASE4_CLOSEOUT.md)

**Phase 4 exit:** all phase-4 `migration_map` entries `completed` or `skipped`. Optional copy waves deferred.
