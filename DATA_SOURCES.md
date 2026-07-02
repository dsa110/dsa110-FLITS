# Data sources

The FRB burst dynamic spectra (`*.npy`, ~250 MB each, ~6–18 GB for the 24
CHIME+DSA files) are **not committed to git** — `.gitignore` excludes `*.npy`
and `/data/{raw,interim,processed}/`. This file documents where the data lives
and how the repo references it without bloating the tree.

> **Scintillation (up-channelized CHIME + DSA Δν(ν)) data** has its own band-by-band
> provenance ledger — which spectra/ACF products exist, what band/resolution each is,
> what the pipeline consumes, and the CANFAR/arc paths — in
> [`scintillation/DATA_PROVENANCE.md`](scintillation/DATA_PROVENANCE.md).

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

## CANFAR compute and GPU access

Important: `vos`/`vls` access to `arc:` is storage-only. That limitation was
previously mistaken for "CANFAR has no GPU access for us." After installing the
`canfar` CLI and authenticating with `~/.ssl/cadcproxy.pem`, GPU sessions were
verified live on 2026-06-25:

```bash
canfar create headless skaha/astroml-cuda:latest --gpu 1 -n gpu-smoke-test -- nvidia-smi
```

The session completed and `canfar logs` showed `NVIDIA A100-PCIE-40GB` with CUDA
12.8, exposed as a MIG slice (~20 GiB). Existing notebook sessions that were
created without `--gpu` still have `requestedGPUCores: "0"`; recreate the
session with `--gpu N` when accelerator access is needed.

## Local replica (only if remote latency bites)

```bash
rsync -av <user>@<arc-or-lxd>:/arc/home/jfaber/.../DSA_bursts/ ~/Data/Faber2026/dsa110/DSA_bursts/
export DATA_DIR=~/Data/Faber2026/dsa110/DSA_bursts
```

**Moved 2026-06-30** from `~/Developer/dsa110-local-data/DSA_bursts/` to
`~/Data/Faber2026/dsa110/DSA_bursts/` (the machine-wide canonical location for
Faber2026 project data). `data/dsa/` and `data/chime/` in this repo, and
`overleaf/Faber2026/pipeline/data/{dsa,chime}/`, are now symlinks into that
location — no code changes needed, relative paths still resolve.

Keep any local replica **out of git** (`*.npy` is already ignored; a top-level
`/Data/Faber2026/dsa110/` or `/data/` replica stays untracked). Do **not** use
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
- **OPEN: stored scintillation does not reproduce from the current arc files +
  committed joint fits** (investigated 2026-06-22; root cause NOT yet isolated —
  do not trust a one-line explanation). All 24 arc spectra were fetched to
  `~/Data/Faber2026/dsa110/DSA_bursts/` (formerly `~/Developer/dsa110-local-data/DSA_bursts/`, moved 2026-06-30) and load with correct shapes
  (DSA `(6144, 2500)`, CHIME `(1024, 32000)`). Re-running the scint chain
  (`gain_ladder.py` → `multiscale_fit.py`) on those files using the committed
  `joint_json/*_joint_fit.json` reproduces the stored Δν for **freya CHIME**
  (verified: dnu_1L ladder ≈ stored) but gives matched-filter **gain S/N ≈ 0**
  (model pulse off the burst) for most band-instances — and the pattern is
  burst/band-specific, NOT a clean "CHIME good / DSA bad" split (alive: freya/
  chromatica/isha/oran CHIME, whitney DSA; dead: most others incl. several CHIME).
  Symptoms seen, cause unconfirmed: (a) many joint-fit `t0_C`/`t0_D` cluster at
  ~28.5 ms (plausibly just onpulse-crop centering in a ~57 ms window, not
  necessarily placeholder); (b) some `t0` are wild (mahi `t0_D`=183 ms); (c) DSA
  onpulse crops sometimes collapse to <1.5 ms. Candidate causes to check before
  any conclusion: the joint fits were produced with different `BurstDataset`
  framing (`f_factor`/`t_factor`/`outer_trim`/crop) than `configs/batch/*`; arc
  files were re-generated/re-centered vs the joint-fit inputs; or a DSA-loader/
  crop bug. Until reconciled, treat stored Δν/τ as not-yet-regenerable from the
  arc data. (NOTE: an earlier version of this bullet claimed a clean "DSA
  re-centered ~82 ms vs t0_D ~30 ms, 47–54 ms offset" — that was a raw-frame `dt`
  artifact + n=1 overgeneralization and is RETRACTED.)

CHIME path bugs already fixed: `johndoeII_chime.yaml` had `johndoeII_dsa.yaml_chime_...`
spliced into the filename (commit `88747e7`); `casey_chime.yaml` repointed from a
nonexistent relative `data/chime/` path to arc (commit `8e5f8df`).
