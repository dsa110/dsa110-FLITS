# A1 covariance qualification

Lane ID: `a1-covariance` (distinct from the later A1 trigger-calibration campaign).

Status: **documented fail**; `science_status=diagnostic_only`. The additive
off-pulse covariance model did not qualify for an on-pulse CHIME measurement.

The held-out test passed 3/6 low-band slices and 6/6 high-band slices. All 48
real-background injection fits were finite, but width recovery, interval
coverage, and modulation-index recovery failed the declared gates. No Freya
on-pulse fit was performed.

Canonical evidence:

- [`validation.json`](../../../chime-recovery-2026-07-12/results/a1/validation.json)
- [`figures.manifest.json`](../../../chime-recovery-2026-07-12/results/a1/figures.manifest.json)
- [`figures.review.json`](../../../chime-recovery-2026-07-12/results/a1/figures.review.json)
- [`figures/`](../../../chime-recovery-2026-07-12/results/a1/figures/)

The manifest uses paths relative to its own directory and binds each retained
SVG with SHA256. The figures are diagnostics of a failed qualification, not
qualified scintillation measurements.
