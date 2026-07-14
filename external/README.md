# Vendored DM-estimation packages (dm_campaign Phase 1)

Published packages run **as released** — no algorithm edits — so their verdicts
on the co-detection sample are attributable to the authors' code, not our
plumbing (Faber2026 `docs/rse/specs/plan-dm-measurement-methods.md`).

| Package | Upstream | Pinned commit | License | Files vendored |
|---|---|---|---|---|
| DM_phase | github.com/danielemichilli/DM_phase @ master | `b7cf5fd61436` | GPL-3.0 (LICENSE vendored) | `DM_phase.py`, `LICENSE`, `__version__.py` |
| DM-power | github.com/hsiuhsil/DM-power @ main | `f7787355ca28` | none published — vendored for internal reproduction only, do not redistribute | `DM_power.py`, `README.md` |

Fetched 2026-07-09 via `opensrc fetch danielemichilli/DM_phase hsiuhsil/DM-power`.

Access is through `dispersion/dm_campaign/adapters.py` only. Two
import-environment shims there (algorithms untouched):

- `DM_power.py` imports `mpi4py` at module level (used only to distribute
  bootstrap jobs on a cluster, not in the algorithm); the adapter stubs the
  module when mpi4py is absent so `get_power`/`fit_log_dm_width`/`plot_dm_err`
  run single-process.
- `DM_power.py` reads instrument geometry from module globals
  (`dt`, `nchan`, `chan_bw`, `f_arr`, `dm_series`) — the released interface is
  a CLI script; the adapter sets those globals per call, mirroring the
  script's own `__main__` setup.

One source patch is carried (marked `VENDOR PATCH` in the file):

- `DM_phase.py` `_poly_max`: `np.arange(Start, Stop)` →
  `np.arange(int(Start), int(Stop))`. numpy ≥2 no longer auto-converts the
  1-element arrays that argwhere-indexing produces; the released code predates
  numpy 2. No algorithm change.

A third import shim: `DM_power.py` runs a `required=True` argparse at module
level, so the adapter feeds a placeholder argv during import (every global the
adapter consumes is overwritten per call).

Note for interpretation: `DM_power.spec_shiftDM` uses Fourier (circular)
shifting — wrap-around is part of the released algorithm. The in-tree variant
(`dispersion/dm_power_analysis.py`) deliberately replaced this with zero-fill;
that difference is one of the things the injection matrix measures.
