# `old_stuff/` — archival snapshot

Source: `~/Documents/research/caltech/ovro/dsa110/chime_dsa_codetections/old_stuff/`
Date folded: 2026-04-23
Folded as part of Phase 4 of the consolidation plan (`chime-dsa_consolidation_phases_964d46a0.plan.md`).

## What's here

Historical analysis scripts, notebooks, and PDFs from the earlier iterations of the CHIME–DSA co-detection work. Kept verbatim for provenance; not part of the active code path.

## Filtering applied during the fold

Rsynced with:

```
rsync -a --max-size=5M \
      --exclude='.git/'   --exclude='.venv/' \
      --exclude='__pycache__/' --exclude='.ipynb_checkpoints/' \
      --exclude='.DS_Store' \
      $SRC/old_stuff/ .archive/old_stuff/
```

- **108 of 238 files transferred** (46 MB on disk in `.archive/old_stuff/`).
- **130 files skipped** by `--max-size=5M`. These are filterbank data (`.fil`) and a few large binary traces, all of which are either (a) reproducible outputs or (b) iCloud-dataless placeholders on the source (apparent 500 MB, 0 blocks on disk).
- Nested clone `old_stuff/dsa110-scat/.git` is a broken git reference and was excluded.

## Excluded binary blobs (for reference)

Per the plan ("oversized binary blobs moved to iCloud, fold carries source only"), the >5 MB entries in the source should be retrieved from iCloud / iacobus when needed. Representative paths:

```
old_stuff/polcal_filterbanks/{burst}/{event}_dev_polcal_I.fil
old_stuff/dsa_filterbanks/{burst}_{event}/{event}_dev_polcal_{I,Q,U,V}.fil
```

These live on the source disk and are iCloud-offloaded. To rehydrate a specific `.fil`, `brctl download` the individual file on the source Mac.
