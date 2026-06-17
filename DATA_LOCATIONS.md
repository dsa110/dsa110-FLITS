# Data Locations for CHIME-DSA Co-Detection Project

## Code (this repo)
- Local: ~/Developer/repos/github.com/dsa110/dsa110-FLITS/
- GitHub: https://github.com/dsa110/dsa110-FLITS

## Legacy Documents-Area Clone

The older iCloud/Documents clone is a source snapshot to drain, not an active
development repo:

- Legacy clone: `~/Documents/research/caltech/ovro/dsa110/dsa110-FLITS/`
- Resolved iCloud path: `~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Areas/research-holding/caltech/ovro/dsa110/dsa110-FLITS/`
- Legacy HEAD observed during migration: `569adc7`
- Current canonical HEAD at migration start: `caf2e8f`
- Developer staging attempt: `~/Developer/scratch/2026-06/chime-dsa-documents-area-staging/`

The tracked Git files match the canonical repository lineage. The value left in
the Documents-area clone is local artifact state, chiefly the 5.4 GiB `data/`
tree. Do not import those arrays into Git; keep them external and reference them
through manifests or data-location docs. See
[`docs/migration/chime-dsa-documents-area-migration.md`](docs/migration/chime-dsa-documents-area-migration.md).

## Processed Data

**Canonical storage host/path: `iacobus:~/Research/CHIME_DSA_Codetections/`** (500 GB host on LAN).
The bulky CHIME/DSA FRB products should stay materialized on `iacobus` for
storage. The `iacobus` CloudDocs copy at
`~/Library/Mobile Documents/com~apple~CloudDocs/Research/CHIME_DSA_Codetections/`
is mirrored into this real data root; other iCloud views, when present on other
Macs, may be placeholders and are not sufficient proof of local availability.

The storage layout on `iacobus` is:

  - `dsa_fullstokes_waterfalls/` — 58 files, 6.0 GiB: IQUV .npy arrays (from h23)
  - `burst_pickles/` — 24 full-Stokes interpolated .pkl files, 60.8 GiB (from Dropbox, via rclone direct-API)
  - `burst_npys/` — 52 files, 20.6 GiB: DSA burst waterfalls broader sample (from h23)
  - `scattering_results/` — 112 files, 86 MB: per-burst PDFs, corner plots, model fits (from h23)
  - `dm_budget/` — 799 files, 127 MB: DM budget code + pre-existing local data (combined h23 + Mac)
  - `metadata/` — 7 files, 584 KB: CSVs, localizations, burst properties, interveners notebook
  - `presentations/` — 2 files, 32 MB: DM phase .pptx / .key
  - `archive/` — 491 files, 130.7 GiB: `OLD_CHIME_DSA_Codetections/` plus `burstprop_paper/`

**Live SSH check on 2026-06-17 after APFS clone mirror:
234,392,304,408 logical bytes (~218.3 GiB), 1546 files across 8 subdirs.** See
[`codetections_manifest.yaml`](codetections_manifest.yaml) for per-subdir
source machine provenance and SHA-256 sentinels from the original April
consolidation manifest.

Note: the separate `nihari` project was explicitly removed from scope for
this consolidation; it lives at iCloud `Research/nihari/`.

## Raw Data (h23 — keep on server, do not copy)
- h23:/media/ubuntu/ssd/jfaber/chime_dsa_codetections/ — organized analysis dir
- h23:/media/ubuntu/ssd/jfaber/OLD_CHIME_DSA_Codetections/ — historical archive
- h23:/dataz/dsa110/T3/ — raw T3 pipeline output (59 TB)

## Dropbox Exit (codetection-scope source materials)

Dropbox is being retired as a storage repository. The authoritative staging
tree lives on `iacobus` under iCloud Drive and surfaces as online-only
placeholders on any Mac signed into the same iCloud account:

- Staging (iacobus, authoritative): `iacobus:~/Library/Mobile Documents/com~apple~CloudDocs/Dropbox-Migration/`
- iCloud mirror (this Mac, placeholders): `~/Library/Mobile Documents/com~apple~CloudDocs/Dropbox-Migration/`

Codetection-scope coverage (source -> target):

| Dropbox source (`~/Library/CloudStorage/Dropbox/`) | Target in `Dropbox-Migration/` | Status |
|---|---|---|
| `121102_bursts/` (3 files, 599.62 MiB, AO FRB 121102 `.tar` archives) | same | complete |
| `Codetections_DSA_Filterbanks/` (15 burst bundles, 60 `.fil`, 28.13 GiB) | same | complete; `wilhelm_221203aaaa_253635173/` and `zach_240203aacl_210456524/` backfilled 2026-04-24 via `rclone copy dropbox: -> iacobus` |
| `Faber2025/` (21 files, 4.05 MiB, manuscript + Overleaf) | same | complete; `overleaf_texdocs_faber2025/references/` backfilled 2026-04-24 |
| `Apps/CANFAR_backup/` (85 files, 96.94 MiB) | same | complete |
| `Apps/CHIME_DSA_Codetections/` (24 burst pickles, 60.79 GiB) | not mirrored to `Dropbox-Migration/` | the same 24 pickles are authoritatively staged at `Research/CHIME_DSA_Codetections/burst_pickles/`; no duplicate maintained |
| `archive/dsa110-contimg.bkp/` (~7 MiB code backup) | partial (41 of 330 files) | source of truth is `https://github.com/dsa110/dsa110-contimg`; further backfill skipped |

Dropbox retirement gates:

1. Close codetection gaps above: done 2026-04-24.
2. Confirm iCloud materialization from `iacobus`: observability only; bytes are preserved on `iacobus` local disk regardless. In flight as of 2026-04-24 for the freshly backfilled filterbank and Faber2025 files.
3. Rename Dropbox source folders in place with a retirement suffix so the Dropbox cloud copy persists as insurance without breaking sync semantics: done 2026-04-24. The six codetection-scope folders were renamed locally to `*_retired_2026-04-24_iCloud_is_authoritative/`, except `dsa110-contimg.bkp`, which used `*_retired_2026-04-24_github-is-authoritative/`. Renames were done while Dropbox.app was not running; they will reconcile to Dropbox cloud by content hash when Dropbox.app next launches.
4. Downgrade or cancel Dropbox after the insurance window passes.

## Related Repos
- subhalos: https://github.com/jakobtfaber/subhalos
- los_halos: https://github.com/jakobtfaber/los_halos
- dsa110-scat: ~/Documents/research/caltech/ovro/dsa110/dsa110-scat/
- DM_phase: h23:/media/ubuntu/ssd/jfaber/DM_phase/
