# Plan — Freya high-band independent-polarization cross-ACF

**Status:** DOCUMENTED-FAIL; no real-data measurement permitted

**Branch:** `scint/chime-highband-crossacf` from `scint/chime-additive-likelihood`
**Qualification target:** Freya (`FRB 20230325A`) only

## Question

Can the 627--800 MHz portion of Freya's CHIME baseband yield an unbiased
scintillation bandwidth when the frequency correlation is formed between the
two independently detected polarization streams, rather than by
autocorrelating one Stokes-I spectrum?

The two receiver-noise realizations are retained immediately after coherent
dedispersion and upchannelization, before the historical Stokes-I sum. The
primary cross-ACF is additionally time-disjoint:
`0.5 * (X_even x Y_odd + X_odd x Y_even)`. This removes equal-time polarized
burst self-noise from the expectation while keeping frequency structure common
to the burst. This is a new observable; it does not reopen or relabel the
failed H0/H2/H3/A1 correction experiments.

## Fixed analysis choices

- Analyze 627--800 MHz only. A1's held-out additive-background test passed 6/6
  slices in this band and failed in the low band.
- Coherently dedisperse the complex single-beam voltages at DM 912.4 with
  `time_shift=False`; align later on a padded canvas from `time0` metadata.
- Upchannelize with U=64 (6.1035 kHz fine channels) using the same
  `baseband_analysis` PFB inversion as the current Stokes-I product.
- Preserve `|X|^2` and `|Y|^2` separately and verify their sum reproduces
  Stokes I exactly before serialization.
- Estimate the symmetric cross-ACF only from channel pairs belonging to the
  same parent 0.390625 MHz coarse channel, after removing each polarization's
  mean within that block. Begin the fit at fine-channel lag 2.
- Fit a positive Lorentzian common component plus a constant over at most
  0.25 MHz. No phase cycling, whitening, kernel subtraction, or tuning on the
  on-pulse fit is allowed.
- The full-band width is a 627--800 MHz band-effective quantity. Subband
  comparisons scale their widths to 713.5 MHz with a fixed `nu^4.4` law before
  testing agreement; the raw upper-subband width must exceed the lower one.

## Predeclared gates

The Freya on-pulse fit is forbidden unless all prerequisite gates pass.

1. **Producer parity:** both polarization products share the same frequency and
   padded time grids; their elementwise sum reproduces the regenerated Stokes-I
   product within floating-point precision; hashes and source HDF5 are recorded.
2. **Independent-noise null:** twelve paired burst-duration off-pulse windows
   are tested. Their polarization cross-ACFs must be statistically consistent
   with zero; no individual window may exceed 3 standard errors at retained
   lags, and the aggregate reduced chi-square must be in [0.5, 2.0].
3. **Real-background injection recovery:** inject the same stationary
   Lorentzian scintillation realization into the two real polarization dynamic
   spectra with a fixed burst envelope and source-noise inflation. Test HWHM
   values 3, 6, 10, and 16 fine channels, modulation indices 0.3 and 1.0,
   and at least 50 deterministic trials per cell.
   Every cell must satisfy median width bias below
   `max(10%, 0.25 channel)`, nominal-68% coverage in [0.53, 0.83], and median
   modulation-index bias below `max(10%, 0.05 absolute)`.
   The on-pulse width must lie inside this validated 3--16-channel envelope.
4. **Polarization/common-signal compatibility:** the inferred narrow feature
   must remain under independent early/late on-pulse integrations. The two
   disjoint high-frequency subbands must increase with frequency and agree
   after fixed `nu^4.4` scaling to 713.5 MHz. Missing support or a scaled
   disagreement larger than `max(25%, 2 combined sigma)` is a failure, not an
   upper-level measurement.
5. **Fit-window stability:** accepted widths across fixed 0.15, 0.20, and
   0.25 MHz fit maxima must form a stable plateau, with maximum fractional
   movement below 20% and no bound contact.
6. **Manual figure review:** off-pulse nulls, injection recovery, polarization
   spectra, cross-ACF fit, and split checks must agree with the serialized
   machine verdict before `science_status` can become `measurement`.

Any failed or inconclusive gate leaves CHIME `diagnostic_only`. No other burst
enters this experiment until Freya passes all six gates.

## Stop condition

- **PASS:** all gates pass and the manual review accepts the figures; the result
  may then enter the existing CHIME artifact-control battery.
- **DOCUMENTED-FAIL:** the cross-polarization off-pulse null or injection
  battery fails, or the real signal is polarization/time/subband incompatible.
- **DATA BLOCKED:** the archived HDF5 cannot produce two provenance-bound
  polarization waterfalls. This is distinct from a scientific failure.

If this deliberately favorable high-band test fails, the current archived
CHIME products support limits and diagnostics, not a citable scintillation
bandwidth. Further full-band correction variants are out of scope.

## 2026-07-14 outcome

The authoritative regeneration passed provenance, Stokes-I parity, and burst
alignment (the band-summed peak is at aligned bin 254). The qualification then
failed before the on-pulse fit:

- four of twelve off-pulse windows exceeded the predeclared `|z| <= 3` rule,
  and the curves share a visible positive-low-lag/negative-high-lag structure.
  An adversarial review found the same low-lag sign in 11/12 windows, so the
  verdict does not depend on treating 39 retained lags as independent trials;
- all four `m=0.3` injection cells exceeded their fixed median width-bias
  limits, while all four `m=1.0` cells passed; and
- the validator therefore did not execute or render an on-pulse fit.

The machine-readable verdict and provenance are in
`analysis/chime-recovery-2026-07-12/results/b3_crossacf/validation.json`.
`figures.review.json` records the independent visual review against the hashed
figure manifest. This is a failure of qualification, not a measured CHIME
scintillation bandwidth.

The coherent residual is consistent with short-timescale polarization-common
noise surviving the even/odd construction: the estimator excludes equal-time
products, but its paired samples are still adjacent in time. A future
guard-separated or contiguous-half cross-estimator would be a genuinely new
experiment, with higher variance and a fresh predeclared calibration battery;
it is not a post-hoc threshold change to B3.
