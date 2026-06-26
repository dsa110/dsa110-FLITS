# Phase 5 closeout — quarantine + decommission

**Date:** 2026-06-25  
**Branch:** `migration/phase1-easy-wins`

## Summary

Retired-host codetection paths are drained and quarantined. The project now runs on four hosts only (jakob-mbp, iacobus, arc, h17).

## Actions executed

### h23 — partial jfaber quarantine

Full root `mv` blocked by **nihari** (out of scope). Move-only partial quarantine:

```bash
# on h23 via ProxyJump dsa110maas
mkdir -p /media/ubuntu/ssd/_quarantine/jfaber-drain-20260625
mv /media/ubuntu/ssd/jfaber/{archive,burstprop_paper,chime_dsa_codetections} \
   /media/ubuntu/ssd/_quarantine/jfaber-drain-20260625/
```

- **Quarantined:** ~137G (86G + 42G + 8.8G)
- **Residual at** `/media/ubuntu/ssd/jfaber/`: nihari, tools, dsa110-continuum, frb_inventory, scratch
- **Untouched:** `/dataz/dsa110/T3` (59T raw pipeline)
- **Restore:** `/media/ubuntu/ssd/_quarantine/README.md`

### hpcc — already quarantined (Phase 1)

`/home/jfaber/flits` → `/home/jfaber/_quarantine/flits-20260625` (verified 2026-06-25)

### dsacamera — decommissioned (Phase 1)

No codetection trees; inventory entry `completed`.

## Docs / inventory updates

- `DATA_LOCATIONS.md` — 4-host model; retired hosts as quarantine references
- `machine_inventory.yaml` — `h23_jfaber_root` → `completed`; h23 locations refreshed
- `MIGRATION_LOG.md` — Phase 5 entry appended

## Verification

```bash
# retired hosts: zero pending (out_of_scope OK)
for h in h23 hpcc dsacamera; do
  python scripts/query_machine_inventory.py --migration-map --migration-status pending --machine "$h"
done

python scripts/query_machine_inventory.py --check-retired-coverage
```

**Expected pending on target hosts only (Phase 4, not Phase 5):** none after 2026-06-26 closeout — see [`PHASE4_CLOSEOUT.md`](PHASE4_CLOSEOUT.md)

## Not in Phase 5 scope

- Phase 4 h17 stub removal (`h17_ubuntu_stub`) — design only
- arc CANFAR ephemeral `flits_run` — skipped, not quarantined on disk
