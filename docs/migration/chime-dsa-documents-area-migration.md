# CHIME-DSA Documents-Area Migration

Date: 2026-06-17

## Purpose

Drain the older iCloud/Documents clone of `dsa110-FLITS` into the active
Developer clone without turning the Documents clone into a second active
development surface.

## Repositories

- Canonical repo: `~/Developer/repos/github.com/dsa110/dsa110-FLITS/`
- Legacy Documents clone: `~/Documents/research/caltech/ovro/dsa110/dsa110-FLITS/`
- Legacy resolved path: `~/Library/Mobile Documents/com~apple~CloudDocs/Documents/Areas/research-holding/caltech/ovro/dsa110/dsa110-FLITS/`
- Legacy moved path: `~/Library/Mobile Documents/com~apple~CloudDocs/_trash-git-repos-2026-06-17/Documents/Areas/research-holding/caltech/ovro/dsa110/dsa110-FLITS/`
- Developer staging copy: `~/Developer/scratch/2026-06/chime-dsa-documents-area-staging/`

## Git State

The legacy clone is a real Git repository with `origin` pointing at
`https://github.com/dsa110/dsa110-FLITS.git`.

- Legacy clone HEAD: `569adc7` (`Merge pull request #29 from dsa110/consolidation/fold-satellites`)
- Canonical clone HEAD at migration start: `caf2e8f` (`Merge pull request #30 from dsa110/fix/parameterize-flits-root`)
- Both clones report 506 tracked files.
- The canonical clone is five tracked notebook/script edits ahead of `569adc7`.
- `DATA_LOCATIONS.md` in the Documents clone contained an uncommitted Dropbox retirement section; that section was migrated into the canonical repo.

## Staging Snapshot

A direct whole-tree copy from iCloud/Documents to `~/Developer/scratch/2026-06/`
was attempted with `rsync -aH`. The copy produced a Git-valid staging worktree at
`~/Developer/scratch/2026-06/chime-dsa-documents-area-staging/`, but repeated
full-tree scans timed out because the source path is backed by iCloud/File
Provider.

Observed staging state:

- Staging size copied before timeout: about 414 MiB.
- Source size: about 5.8 GiB.
- Staging `git fsck --connectivity-only --no-reflogs` reported only dangling
  objects; the reachable HEAD was valid enough for Git inventory.

Treat the staging directory as a provenance snapshot, not a complete mirror of
the Documents data tree.

## Content Classification

Tracked code/docs:

- Do not wholesale import. The canonical repo is already ahead of the legacy
  HEAD.
- Preserve the migrated `DATA_LOCATIONS.md` Dropbox retirement section.

Local data products:

- `data/`: about 5.4 GiB in the Documents clone.
- `data/chime/`: about 2.6 GiB, including 12 current CHIME `.npy` files, 9
  `backup_before_standardize` files, and 12 `backup_original_freq_axis` files.
- `data/dsa/`: about 1.4 GiB, including 12 current DSA `.npy` files and 12
  `backup_before_standardize` files.
- `data/chime_backup_standardization/`: about 1.5 GiB, including 12 standardized
  `.npy` files and 12 original-frequency-axis backups.
- `legacy_documents_data_checksum_manifest.yaml` records this data tree. It
  covers 104 paths and 10,425,999,308 logical bytes. On 2026-06-17, 61 files
  were hashable locally and 43 files were iCloud `dataless` placeholders, so
  those 43 entries are listed with `status: dataless_placeholder` and
  `sha256: null`.
- The broader CHIME/DSA co-detection product tree belongs on
  `iacobus:~/Research/CHIME_DSA_Codetections/` for storage, materialized on the
  `iacobus` host. The `iacobus` CloudDocs tree at
  `~/Library/Mobile Documents/com~apple~CloudDocs/Research/CHIME_DSA_Codetections/`
  is a source/mirror; other iCloud views, if any, should be treated as
  placeholders rather than proof of local availability on this Mac.
- On 2026-06-17, the CloudDocs tree was mirrored into the real data root with
  `cp -cRp`, using APFS clonefile semantics. The mirror completed in 5.74 s,
  created separate inodes for checked sentinel files, and did not materially
  reduce free disk space.
- A live SSH metadata check on 2026-06-17 found matching source and mirror
  trees on `iacobus` with 1546 files and 234,392,304,408 logical bytes
  (~218.3 GiB), including the manifest sentinel files under `burst_pickles/`,
  `dsa_fullstokes_waterfalls/`, `burst_npys/`, and `metadata/`.
- The legacy `_trash-git-repos-2026-06-17/.../dsa110-FLITS/data` path also
  exists on `iacobus`, but its 48 files occupy only about 1 MiB on disk while
  reporting ~4.6 GB logical size, consistent with placeholder-style files rather
  than a complete materialized local copy.

Generated/cache state:

- `.mypy_cache/`: about 76 MiB. Exclude from migration.
- `__pycache__/`, `.pytest_cache/`, local agent config/cache directories, and
  notebook checkpoints are generated or host-local. Exclude unless a specific
  file is intentionally promoted.

Archival/paper materials:

- `.archive/old_stuff/Twelve Fast Radio Bursts Co-Detected by DSA-110 and CHIME-FRB/`
  contains the old TeX draft and figures.
- Keep as archival provenance unless the paper is revived as an active manuscript
  under a dedicated `papers/` subtree.

## Migration Rule

Only small text metadata, manifests, and provenance documentation should be
committed to Git. Large arrays, pickles, filterbanks, plots, and generated
analysis products should remain outside Git on `iacobus` storage and be
referenced through `DATA_LOCATIONS.md`, `codetections_manifest.yaml`, or a
future checksum manifest.

## Untracked `flits/scintillation/` Classification

Parallel read-only classification of the untracked `flits/scintillation/`
directory found that it is an experimental legacy migration, not a ready
production replacement for the tracked `scintillation/scint_analysis/` pipeline.

Observed package files:

- `flits/scintillation/__init__.py`
- `flits/scintillation/acf.py`
- `flits/scintillation/analyser.py`
- `flits/scintillation/fitting.py`
- `flits/scintillation/physics.py`
- `flits/scintillation/preprocessing.py`
- `flits/scintillation/secondary.py`

Classification:

- Likely origin: migrated/refactored legacy scripts from
  `.archive/old_stuff/pipeline/scint_pipeline.py` and
  `.archive/old_stuff/pipeline/scint_pipeline_funcs.py`.
- Current active implementation remains `scintillation.scint_analysis`; the
  `flits-scint` entry point still resolves to
  `scintillation.scint_analysis.run_analysis:main`.
- `analyser.py` duplicates broad pipeline behavior but lacks the tracked
  pipeline's configuration, data containers, masking/noise handling, model
  selection, 2D fitting, tests, and CLI integration.
- `acf.py`, `fitting.py`, `physics.py`, `preprocessing.py`, and `secondary.py`
  contain potentially reusable low-level utilities, but they duplicate or
  diverge from tracked science logic and have weaker contracts.
- `secondary.py` is the most clearly standalone helper, but secondary arc
  fitting remains a placeholder in the neighboring analyser.

Recommendation:

Do not stage this directory as-is. If a public `flits.scintillation` namespace
is desired, design it as a thin adapter over the tracked
`scintillation.scint_analysis` pipeline rather than preserving a parallel
pipeline. Otherwise classify the untracked directory as stale experimental work
and remove it only after owner confirmation.

## Remaining Decisions

- Whether to reconcile the 43 legacy iCloud dataless placeholders against the
  `iacobus` storage tree and complete their SHA-256 entries in
  `legacy_documents_data_checksum_manifest.yaml`.
- Whether to keep the partial Developer staging directory after the drain is
  complete.
- Whether to revive the old `Twelve Fast Radio Bursts...` draft as an active
  paper directory or leave it in `.archive/old_stuff/`.
- Whether to preserve, redesign, or delete the untracked `flits/scintillation/`
  experimental package.
