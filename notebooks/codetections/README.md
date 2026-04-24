# `notebooks/codetections/`

Analysis notebooks for the CHIME–DSA co-detected FRB sample.

## Contents

| notebook | purpose | source (pre-fold) |
|---|---|---|
| `interveners.ipynb` | FRB–halo / cluster association and intervener analysis | `chime_dsa_codetections/interveners.ipynb` |

## Excluded from the fold

Two additional files lived at the top of `chime_dsa_codetections/` but were
**not** folded:

- `synfit.ipynb` — 198 B stub created by the Jupyter MCP server; never used.
- `synthetic_scatter_fit.ipynb` — same. Both files contain a single markdown
  cell reading "New Notebook Created by Jupyter MCP Server".

Folding these would pollute the repo with server-side detritus. They remain
at the source path; the quarantine pass (Phase 6) can reclaim them.
