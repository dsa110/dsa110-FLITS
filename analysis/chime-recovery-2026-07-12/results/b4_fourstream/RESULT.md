# Freya CHIME B4 four-stream result

The B4 estimator is a four-stream, time-disjoint ACF built from the two
polarizations and the early/late halves of the burst window. It uses
leave-one-window-out off-pulse templates to remove the repeatable instrumental
ACF. The frozen replay is a **documented failure**, not a scintillation
measurement.

## What passed

- Producer parity, input provenance, and burst alignment pass.
- The independent off-pulse null passes after template subtraction: the
  corrected control ACFs scatter around zero rather than retaining the
  scallop/gain-curve correlation.
- At modulation index 1.0, injected widths of 3, 6, 10, and 16 native channels
  pass the fixed recovery criterion.

## What failed

- Injection recovery fails for modulation indices 0.15, 0.20, and 0.30.
- The diagnostic burst modulation is only about 0.15--0.17, placing it in the
  unvalidated regime.
- Diagnostic Lorentzian widths move with the fit boundary and have formal
  uncertainties larger than the fitted widths. They cannot be reported as a
  decorrelation bandwidth.

Thus the known instrumental structure can be removed well enough to pass an
independent noise null, but the remaining Freya burst contrast is too weak for
this estimator to demonstrate unbiased bandwidth recovery. The complete gates,
injection cells, trial records, provenance hashes, and diagnostic fits are in
`validation.json`.

## Scientific consequence

This negative result rules out treating the visually plausible Freya CHIME ACF
scale as a measurement. It does not invalidate the separately qualified Oran
DSA-110 measurement. A future CHIME claim needs either a higher-modulation
burst/product or an estimator that passes real-background injection recovery at
modulation index approximately 0.15.

