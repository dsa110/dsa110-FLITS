# H2 rank-2 calibration

Status: **documented fail**; `science_status=diagnostic_only`.

Both subbands passed the off-pulse null and low-lag checks, with diagnostic
on-pulse widths of 38.616 kHz at 523.268 MHz and 64.989 kHz at 723.076 MHz.
The complete battery nevertheless failed injection recovery, fit-window
stability, split-time stability, the comb-residual check, the held-out kernel
cross-check, and manual review. Those widths are therefore not qualified
measurements.

Canonical evidence:

- [`validation.json`](../../../chime-recovery-2026-07-12/results/h2/validation.json)
- [`figures.manifest.json`](../../../chime-recovery-2026-07-12/results/h2/figures.manifest.json)
- [`figures.review.json`](../../../chime-recovery-2026-07-12/results/h2/figures.review.json)
- [`figures/`](../../../chime-recovery-2026-07-12/results/h2/figures/)

All four reviewed PNG diagnostics were regenerated on 2026-07-14 from the
retained rank-2 paired products with historical FLITS commit `8f0479d`, then
visually checked against the existing review. Their relative paths and SHA256
digests are recorded in both the manifest and review.
