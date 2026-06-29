# `notebooks/codetections/`

Analysis notebooks for the CHIME–DSA co-detected FRB sample.

## Contents

| notebook | purpose | source (pre-fold) |
|---|---|---|
| `interveners.ipynb` | FRB–halo / cluster association and intervener analysis | `chime_dsa_codetections/interveners.ipynb` |

## Local data

| file | purpose | source (pre-fold) |
|---|---|---|
| `data/frb_cluster_associations.csv` | Cluster associations consumed by `interveners.ipynb` | `chime_dsa_codetections/frb_cluster_associations.csv` |
| `data/frb_halo_associations.csv` | Halo associations consumed by `interveners.ipynb` | `chime_dsa_codetections/frb_halo_associations.csv` |

## Excluded from the fold

Two additional files lived at the top of `chime_dsa_codetections/` but were
**not** folded into FLITS:

- `synfit.ipynb` — 198 B stub created by the Jupyter MCP server; never used.
- `synthetic_scatter_fit.ipynb` — same. Both files contain a single markdown
  cell reading "New Notebook Created by Jupyter MCP Server".

Folding these would pollute the repo with server-side detritus. They were
retired into the CHIME-DSA quarantine during the April 2026 fold; the remaining
sparse holding tree was drained on 2026-06-29.
