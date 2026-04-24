# Data Locations for CHIME-DSA Co-Detection Project

## Code (this repo)
- Local: ~/Documents/research/caltech/ovro/dsa110/dsa110-FLITS/
- GitHub: https://github.com/dsa110/dsa110-FLITS

## Processed Data

**Current canonical staging location: `iacobus:~/Research/CHIME_DSA_Codetections/`** (500 GB host on LAN).
**Long-term canonical target: iCloud `Research/CHIME_DSA_Codetections/`** (migration deferred pending local disk availability).

The subdirectory layout is stable across both locations:

  - `dsa_fullstokes_waterfalls/` — 58 files, 6.1 GiB: IQUV .npy arrays (from h23)
  - `burst_pickles/` — 24 full-Stokes interpolated .pkl files, 60.8 GiB (from Dropbox, via rclone direct-API)
  - `burst_npys/` — 52 files, 20.6 GiB: DSA burst waterfalls broader sample (from h23)
  - `scattering_results/` — 115 files, 86 MB: per-burst PDFs, corner plots, model fits (from h23)
  - `dm_budget/` — 1654 files, 456 MB: DM budget code + pre-existing local data (combined h23 + Mac)
  - `metadata/` — 7 files, 584 KB: CSVs, localizations, burst properties, interveners notebook
  - `presentations/` — 2 files, 32 MB: DM phase .pptx / .key
  - `archive/` — 401 files, 100.6 GiB: `OLD_CHIME_DSA_Codetections/` (~78 GB) + `burstprop_paper/` (~23 GB)

**Total: ~188.6 GiB, 2313 files across 8 subdirs.** See
[`codetections_manifest.yaml`](codetections_manifest.yaml) for per-subdir
file counts, byte totals, source machine provenance, and SHA-256 sentinels.

Note: the separate `nihari` project was explicitly removed from scope for
this consolidation; it lives at iCloud `Research/nihari/`.

## Raw Data (h23 — keep on server, do not copy)
- h23:/media/ubuntu/ssd/jfaber/chime_dsa_codetections/ — organized analysis dir
- h23:/media/ubuntu/ssd/jfaber/OLD_CHIME_DSA_Codetections/ — historical archive
- h23:/dataz/dsa110/T3/ — raw T3 pipeline output (59 TB)

## Related Repos
- subhalos: https://github.com/jakobtfaber/subhalos
- los_halos: https://github.com/jakobtfaber/los_halos
- dsa110-scat: ~/Documents/research/caltech/ovro/dsa110/dsa110-scat/
- DM_phase: h23:/media/ubuntu/ssd/jfaber/DM_phase/
