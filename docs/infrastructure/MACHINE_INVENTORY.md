# Machine inventory (CHIME–DSA codetections)

Canonical file: [`machine_inventory.yaml`](../../machine_inventory.yaml)

Live-probed 2026-06-25 across jakob-mbp, iacobus, h17, h23, hpcc, dsacamera, and arc (VOSpace).

**4-host migration plan:** [`MIGRATION_PLAN_4HOST.md`](MIGRATION_PLAN_4HOST.md) — Phase 2 design: [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md) — log: [`MIGRATION_LOG.md`](MIGRATION_LOG.md)

## Quick reference

| Machine | migration_status | Role | Key path |
|---------|------------------|------|----------|
| **iacobus** | target | data authority | `/Users/iacobus/Research/CHIME_DSA_Codetections` (218G) |
| **jakob-mbp** | target | dev + placeholders | `~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS` |
| **arc** | target | institutional storage + CANFAR compute | `arc:home/jfaber/baseband_morphologies/chime_dsa_codetections` |
| **h17** | target | OVRO compute + arc archive | `/data/research/.../chime-dsa-codetections` (includes `archive/arc_trash_2026-06`) |
| **h23** | retired | cold upstream (drain) | `/media/ubuntu/ssd/jfaber/chime_dsa_codetections` |
| **hpcc** | retired | Slurm batch (drain) | `/home/jfaber/flits/dsa110-FLITS` |
| **dsacamera** | retired | negligible codetection | — |

## Query with Python

```bash
# all paths on iacobus
python scripts/query_machine_inventory.py --machine iacobus

# every git repo entry
python scripts/query_machine_inventory.py --kind git_repo --json

# find DSA burst paths anywhere
python scripts/query_machine_inventory.py --path-contains DSA_bursts

# pending migration_map entries
python scripts/query_machine_inventory.py --migration-map --migration-status pending

# locations on retired hosts
python scripts/query_machine_inventory.py --migration-status retired

# everything targeting iacobus (locations + migration_map union)
python scripts/query_machine_inventory.py --migration-target iacobus

# retired-host coverage gate (exit 0 = all h23 codetection subtrees mapped)
python scripts/query_machine_inventory.py --check-retired-coverage
```

## Query with yq

```bash
yq '.migration' machine_inventory.yaml
yq '.migration_map[] | select(.status=="pending")' machine_inventory.yaml
yq '.canonical' machine_inventory.yaml
yq '.machines.iacobus.locations[] | select(.id=="research_chime_codetections")' machine_inventory.yaml
yq '.. | select(has("path")) | .path' machine_inventory.yaml | rg burst
```

## Refresh

Re-run SSH/`vls` probes and update `generated_utc` + affected subtrees. Companion manifest for iacobus sentinels: [`codetections_manifest.yaml`](../../codetections_manifest.yaml).
