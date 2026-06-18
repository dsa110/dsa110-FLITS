# Data sources

The FRB burst dynamic spectra (`*.npy`, ~250 MB each, ~6–18 GB for the 24
CHIME+DSA files) are **not committed to git** — `.gitignore` excludes `*.npy`
and `/data/{raw,interim,processed}/`. This file documents where the data lives
and how the repo references it without bloating the tree.

## Canonical store

CANFAR arc (institutional, durable, shared; the pipeline typically runs on
CANFAR / OVRO lxd where this is mounted):

```
/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts/
```

The 24 expected files (12 CHIME + 12 DSA) and their DMs are listed in
[`data-manifest.csv`](data-manifest.csv). Fill `sha256`/`bytes` once the data is
reachable with `scattering/scripts/fill_data_manifest.sh` (see below); commit the
filled manifest for reproducibility + corruption detection.

## Running the pipeline against the data

Configs under `scattering/configs/bursts/{chime,dsa}/` bake in the arc path, but
the batch runner does **not** trust it — it re-points each config's `path:` at
`$DATA_DIR` at runtime:

```bash
# on a host where the .npy live (arc / lxd), in the repo root:
DATA_DIR=/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts \
  ./scattering/scripts/run_all_chime_bursts.sh
#   subset/smoke:  BURSTS="wilhelm freya casey" DATA_DIR=... ./...sh
python scattering/scripts/verify_fits.py <OUT_DIR> --csv <OUT_DIR>/summary.csv
```

## Local replica (only if remote latency bites)

```bash
rsync -av <user>@<arc-or-lxd>:/arc/home/jfaber/.../DSA_bursts/ ~/Developer/dsa110-local-data/DSA_bursts/
export DATA_DIR=~/Developer/dsa110-local-data/DSA_bursts
```

Keep any local replica **out of git** (`*.npy` is already ignored; a top-level
`/dsa110-local-data/` or `/data/` replica stays untracked). Do **not** use
OneDrive/iCloud for the raw `.npy` — arc is the durable source; pull a local
copy only for intensive local dev.

## Known config issues (resolve before any DSA run)

Surfaced while building the manifest. CHIME configs are clean; DSA needs care:

- **`oran_dsa.yaml` referenced hamilton's file** (`hamilton_dsa_l_518_799_...`) — a
  real copy-paste bug that would have silently fit hamilton's burst as oran.
  **Neutralized** to a non-resolving `oran_dsa_FIXME_VERIFY_ON_ARC_*.npy` placeholder
  so a DSA run fails loudly; the real `oran_dsa` filename (DM 397) must be located
  on arc and filled in.
- **`johndoeII` DSA file is named `johndoell`** (lowercase L, no `II`) — likely the
  actual arc filename (the data was generated with that spelling), not necessarily
  a defect; CHIME uses `johndoeII`. Verify on arc and reconcile the manifest.
- Note: DSA `path:` values are double-quoted YAML (`path: "..."`) while CHIME are
  unquoted — both valid; a DSA batch runner must strip the quotes when extracting
  the path (the current `run_all_chime_bursts.sh` is CHIME-only).

CHIME path bugs already fixed: `johndoeII_chime.yaml` had `johndoeII_dsa.yaml_chime_...`
spliced into the filename (commit `88747e7`); `casey_chime.yaml` repointed from a
nonexistent relative `data/chime/` path to arc (commit `8e5f8df`).
