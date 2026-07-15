# Reference-parity fixture provenance

## `all_pairs_reference_20260717.npz`

- Origin: **RECOMPUTATION** — output of the pre-vectorization per-pair
  `all_pairs_cross_acf` on the seeded C1 synthetic
  (`test_cross_acf._c1_low_modulation_inputs`, seed `20260717`; 2 pols ×
  12 times, m=0.15, width 6 native bins, 256 blocks × 64 channels,
  `max_lag_bins=40`). Generated 2026-07-14 at the last per-pair commit on
  `scint/c1-allpairs-crossgp` before the closed-form vectorization landed.
- Purpose: characterization pin for
  `test_all_pairs_cross_acf_matches_pinned_reference` — every `CrossACF`
  field of the vectorized implementation must match to rtol 1e-10
  (covariance 1e-9); only float summation order may differ.
- SHA-256:
  `5e09f575fdab68ffc94399ac91c45a9e65b9a1ef28db76db59a29d84534902e8`.

## `zach_acf_codetections_fftsize64_downfreq1.npz`

- Origin: byte-for-byte copy of
  `scint_analysis/reference_arc/arc_live/old_scattering_scintillation/`
  `zach_acf_codetections_fftsize64_downfreq1.npz`; `reference_arc/` remains
  unmodified.
- SHA-256: **RECOMPUTATION** of the copied bytes,
  `6965c50d40d1ba67671f4537fab983a8f4af30f609feaf253b38a22aef29c667`.
  It matches `reference_arc/SHA256SUMS.arc_live:18`. The test assertion is
  `test_reference_arc_parity.py:13,257-259`.
- Production parameters `fftsize=64`, `downfreq=1`, six equal-frequency
  sub-bands, and the `0.00610 MHz` printed channel resolution are
  **EXECUTED-NOTEBOOK OUTPUT** from `test_upchan_spec.ipynb`, cell 11
  (`execution_count=82`; JSON lines 746-771 for the call and 692/708 for the
  printed resolution). The artifact-writing source is `utilities/upchan_spec.py:274-290`.
- Keys and shapes `sub_acfs (6, 3274)`, `sub_lags (6, 3274)`, and
  `sub_fcents (6,)` are **EXECUTED-NOTEBOOK OUTPUT** from that same cell 11 and
  saved artifact. They are asserted at `test_reference_arc_parity.py:261-264`.
- Sub-band centers
  `[766.8635050688563, 700.1998902065691, 633.5362753442816,
  566.8726604819944, 500.209045619707, 433.5454307574197] MHz` are
  **EXECUTED-NOTEBOOK OUTPUT** (`test_upchan_spec.ipynb`, cell 11,
  `execution_count=82`; output arrays saved by `upchan_spec.py:285-286`). They
  are transcribed at `test_reference_arc_parity.py:15-24`.
- First retained positive lag `0.012207217517470781 MHz` and central
  ACF values `[0.08595040, 0.06366222, 0.07017246, 0.13167876, 0.09638528,
  0.06162758]` are **EXECUTED-NOTEBOOK OUTPUT** from the same saved cell-11
  arrays. This is the saved product's own grid (within 16 ppm of nominal
  `2 * 0.390625 / 64`), not the stale `0.39101` fit-grid constant at
  `upchan_spec.py:368`. The `2e-13 MHz` absolute tolerance covers the
  sub-band-dependent endpoint rounding visible in the saved float64 lag arrays;
  assertions are `test_reference_arc_parity.py:266-285`.

## Zach fitted HWHM values

- `[0.21456356, 0.08414576, 0.05479846, 0.09827691, 0.02422339,
  0.02155347] MHz` and the derived positive log-log slope
  `alpha = 3.6627513866` are **RECOMPUTATION**, not stored notebook outputs.
  They use the staged executed ACF arrays, the rescued default `+0 < lag <= 1
  MHz` window (`upchan_spec.py:365-402`), and the rescued HWHM Lorentzian
  (`utilities/scint_funcs.py:269-270`). The executable recomputation and 20%
  fit-stability tolerance are `test_reference_arc_parity.py:288-328`.

## Freya legacy cross-check

- `35.19 kHz` HWHM is **NEITHER an executed rescued-notebook output NOR a
  parity-pipeline recomputation**: it is the canonical value from the
  Faber2026 freya instrumental-origin experiment, documented as an
  **instrumental artifact** at `scint_analysis/chime_artifact_guards.py:7-19`
  (the live parity pipeline yields 58.98 kHz). The rescued notebook's own fit
  is unconstrained (`scint_freya.ipynb` cell 13, execution 60:
  `3836.00587 +/- 2132.12621 kHz`, HWHM outside its own fit window), so freya
  CHIME carries no trustworthy legacy gamma; the cross-check test is
  strict-xfail with parity not expected. Test constant:
  `test_reference_arc_parity.py:31`.
- `15.0 kHz` is the unchanged plan tolerance, a **RECOMPUTATION tolerance**
  specified at `logs/plan-chime-scint-recipe-parity.md:552-553` and transcribed
  at `test_reference_arc_parity.py:32`.
- The rescued Freya notebook itself reports the distinct
  `gamma1 = 3836.00587 +/- 2132.12621 kHz` as an
  **EXECUTED-NOTEBOOK OUTPUT** in `scint_freya.ipynb`, cell 13
  (`execution_count=60`; JSON output lines 895-916). Its ACF and fit source is
  JSON lines 929-956. This value is recorded here to correct the plan's
  executed-notebook attribution; it is not substituted for the legacy
  recomputation target.
- The parity-path cross-check is strict-xfail with reason
  `scallop model not ported (P2-F3)` at
  `test_reference_arc_parity.py:331-368`. The current Phase-2 cleaning path
  does not implement the notebook's `make_scallop_model` step, so the 15 kHz
  tolerance is not widened.
