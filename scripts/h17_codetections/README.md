# h17 co-detection compute workers

Scripts promoted from the h17 compute workspace
(`/data/research/astrophysics/frbs/chime-dsa-codetections/scripts`) so they live
in the canonical analysis repo instead of only on the compute host.

| Path | Role |
|------|------|
| `extract_chime_singlebeam_toas.py` | Re-extract CHIME 400 MHz TOAs from singlebeam HDF5 |
| `compare_chime_toas.py` | Diff re-extracted TOAs vs notebook fixture |
| `extract_final_parallel.py` | Authoritative CHIME-side DM (arrival regression) |
| `dump_grid_data.py` | Manuscript 12-panel DM grid inputs |
| `extract_chime_dm_v*.py` | DM campaign iterations (v2→v4b) |
| `extract_time0_metadata.py` | FPGA time0 metadata dump |
| `diag/` | One-off docker/DM diagnostics (not production) |

**Shared DM library:** import `dispersion.chime_dm` (identical to the former
`scripts/chime_dm.py` on h17).

**Upchannelization worker:** lives at
`analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py`
(12-target table; defaults to the h17 `chime_singlebeam/` and `upchan_codetections/` paths).

**Runtime:** most baseband scripts expect the `chimefrb/baseband-analysis` Docker
image via `bin/baseband_analysis_python.sh` on the h17 workspace (not in this
repo). See `docs/infrastructure/H17_WORKSPACE.md`.
