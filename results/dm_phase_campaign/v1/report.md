# Controlled DM-phase campaign v1 report

## Status

- Measurement and validation artifacts were generated at FLITS commit
  `2a08f697e1398bec88cc8d5554cca6425790b540`.
- The measurement implementation is byte-identical at rebased commit `ee0c25d`; commit `782bcb1`
  only repairs preflight provenance accounting.
- Preflight was regenerated from clean commit `782bcb1`; see `run_manifest.json`.
- Published oracle: passed on deterministic CHIME and DSA bright fixtures.
- Known-truth validation: passed on the untouched final seed block.
- Measurements: 24/24 terminal; 2 PASS and 22 UNCONSTRAINED.
- Visual review: 24/24 `match`; no numeric pass was visually rejected after the Mahi regression repair.
- Adoption: no event has two-band support; no DM is adopted into manuscript inputs.
- Downstream revalidation: not triggered because the adoption dry run has an empty change set.

## Validated injection domain

CHIME propagation-DM accuracy is supported at profile S/N >= 12 for the tested unscattered,
single-component accuracy domain. DSA precision recovery is supported only at profile S/N >=
50. Scattered and multi-component injections are retained as measurements of the offset between
structure-optimizing and propagation DM, not folded invisibly into statistical error.

Final held-out performance:

| Band | Median bias | 68% coverage | 95% coverage | Catastrophic rate |
|---|---:|---:|---:|---:|
| CHIME | +0.000661 | 0.750 | 0.953 | 0.000 |
| DSA | -0.009589 | 0.708 | 0.958 | 0.000 |

## Accepted per-band measurements

| Event | Band | DM (pc cm^-3) | Total uncertainty | Resolution |
|---|---|---:|---:|---|
| casey | CHIME | 491.007579 | 0.010163 | native frequency, native time |
| hamilton | CHIME | 517.986572 | 0.010163 | native frequency, native time |

Both events have `single-band` support because their DSA products are unconstrained. They are
retained as evidence but are not automatically promoted to event `DM_obs`.

## Visual evidence

Every product has a 12-panel diagnostic under `products/<burst>/<band>/diagnostic.png`.
The two complete overview sheets are `contact_sheets/chime_contact_sheet.png` and
`contact_sheets/dsa_contact_sheet.png`. Review verdicts and notes are in
`figures.review.json`.

## Adoption dry run

`event_summary.csv` contains no adopted DM. Therefore the proposed change set for association
timing, DM budgets, host-DM posteriors, scattering fits, manuscript figures, tables, and prose
is empty. Existing manuscript DM values must not be relabeled as measurements from this suite.
