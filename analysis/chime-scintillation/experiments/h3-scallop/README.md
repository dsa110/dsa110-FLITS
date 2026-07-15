# H3 stationary-kernel whitening

Status: **documented fail**; `science_status=diagnostic_only`.

The independent held-out kernel cross-check and width/amplitude injection
recovery failed, and manual review recorded corresponding anomalies. Whitening
was not applied to the Freya on-pulse product and no measurement was promoted.

Canonical evidence:

- [`validation.json`](../../../chime-recovery-2026-07-12/results/h3/validation.json)
- [`figures.manifest.json`](../../../chime-recovery-2026-07-12/results/h3/figures.manifest.json)
- [`figures.review.json`](../../../chime-recovery-2026-07-12/results/h3/figures.review.json)
- [`figures/`](../../../chime-recovery-2026-07-12/results/h3/figures/)

The manifest uses paths relative to its own directory and binds each retained
SVG with SHA256. These are failure diagnostics, not qualified scintillation
measurements.
