---
description: Capture environment, data, seeds, and config so a result can be reproduced
user-invocable: true
---

Use the `ensuring-reproducibility` skill (vendored at `.claude/skills/ensuring-reproducibility/SKILL.md`) to handle this request.

For FLITS, provenance anchors already in the repo: `codetections_manifest.yaml`, `data-manifest.csv`, `DATA_LOCATIONS.md`, `DATA_SOURCES.md`, `configs/bursts.yaml` (burst registry), `environment.yml` / `uv.lock` (env pin). Burst nicknames (`casey`, `freya`) key filenames; TNS names (`FRB 20240229A`) are for publication — convert via `scattering.scat_analysis.burst_metadata`.

Target result / experiment: $ARGUMENTS

If no target was provided, ask which result or experiment to make reproducible.
