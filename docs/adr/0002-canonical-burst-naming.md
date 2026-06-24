# Canonical nickname↔TNS designation map and its single source of truth

**Status:** accepted

## Context

The 12 co-detected bursts carry two name systems: internal **nicknames**
(`casey`, `zach`) that key filenames, configs, and results, and **TNS**
designations (`FRB 20240229A`) for publication. The mapping had appeared to
disagree across artifacts, and `johndoeii`'s designation was corrected once
already (origin #8: → FRB 20230814B). A close-out review (2026-06-24, 3-expert
panel) traced every committed artifact and found they are in fact **unanimous**;
the apparent conflict was a transcription error in a scratch decision-map, not a
real artifact disagreement. The preferred source `chimedsa_burst_specs.csv` is
**gitignored and absent** from clean checkouts, so it cannot be the operative
registry.

## Decision

The canonical nickname↔TNS map is the committed
`scattering/scat_analysis/burst_metadata.py::_FALLBACK_TNS`, with
`configs/bursts.yaml` the source of truth for burst *properties* (chime_id, DM,
MJD, coordinates). Conversion is via `scattering.scat_analysis.burst_metadata`;
no hand-maintained second map.

| nickname | TNS | nickname | TNS |
|---|---|---|---|
| zach | FRB 20220207C | freya | FRB 20230325A |
| whitney | FRB 20220310F | johndoeii | FRB 20230814B |
| oran | FRB 20220506D | hamilton | FRB 20230913A |
| isha | FRB 20221113A | mahi | FRB 20240122A |
| wilhelm | FRB 20221203A | chromatica | FRB 20240203A |
| phineas | FRB 20230307A | casey | FRB 20240229A |

`FRB 20240203A` (chromatica) and `FRB 20230814B` (johndoeii) are **distinct
bursts** (different chime_id, MJD, DM), not aliases. johndoeii was double-reported
to TNS (also as …0814A); the DSA-110 archive files it under **B**, which is
canonical here. Nicknames are the internal key; TNS designations are the **only**
identifiers in the manuscript and published figures.

## Consequences

- Any artifact disagreeing with `_FALLBACK_TNS` / `bursts.yaml` is stale and
  reconciles *to* them. Agent memory (`chimedsa-tns-corrections`) is a cache that
  happens to be correct, not the source of truth.
- `chimedsa_burst_specs.csv` must not be cited as the registry while it is
  gitignored/absent (`CLAUDE.md` corrected accordingly).
- Figures with embedded designations (`alpha_pbf_systematic`,
  `whitney_multiplicity`) should derive labels from `burst_metadata`, not
  hard-code them.
- Future TNS corrections edit `bursts.yaml`/`_FALLBACK_TNS` + this table in one
  commit; downstream regenerates.
