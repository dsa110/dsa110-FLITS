# Research: Two-band scintillation bandwidth and modulation-index validation

**Date:** 2026-07-09
**Scope:** Internal codebase and live local data inventory
**Codebase state:** `dsa110-FLITS` commit `863b8726` (`origin/main`, inspected 2026-07-09)
**Related Documents:** [Freya CHIME instrumental-origin experiment](https://github.com/jakobtfaber/Faber2026/blob/main/docs/rse/specs/experiment-freya-chime-instrumental-origin.md), [`../../../analysis/scintillation-dsa-lorentzian-2026-07-07/README.md`](../../../analysis/scintillation-dsa-lorentzian-2026-07-07/README.md), [`../../../scintillation/DATA_PROVENANCE.md`](../../../scintillation/DATA_PROVENANCE.md)

## Question / Scope

What is required to validate the scintillation decorrelation bandwidths
(`Delta nu_d`) and modulation indices (`m`) in both the CHIME and DSA bands for
the 12 CHIME--DSA co-detections, and what existing code and products can support
a combined sample-level figure?

The pass covers the current local product inventory, the ACF estimators and fit
contracts, artifact and quality gates, stored versus freshly derived results,
and the existing figure/report producer. It does not adopt any measurement or
change a burst config. The follow-up upstream pass also compares the tracked
producer and ACF conventions with the live h17 generation code and the public
Nimmo reference implementation.

## Codebase Findings

### 1. The raw product inventory is sufficient for a uniform attempt

The provenance ledger records pipeline-ready CHIME full-band products for all
12 bursts and pipeline-ready DSA products for all 12 bursts
(`scintillation/DATA_PROVENANCE.md:225-231`,
`scintillation/DATA_PROVENANCE.md:312-314`). A live local inventory confirmed
36 resolved NPZ products: 12 DSA, 12 CHIME full-band, and 12 CHIME high-band
diagnostics. The retired CHIME fit pickles must not be reused because their
preprocessing provenance is insufficient
(`scintillation/DATA_PROVENANCE.md:155-166`).

Three products remain intrinsically lower-confidence: `isha`, `hamilton`, and
`johndoeII` have upper-bound or single-block caveats that must remain visible in
the verdicts (`scintillation/DATA_PROVENANCE.md:316-321`). Product availability
therefore permits a uniform measurement *attempt*, not a promise of 24 accepted
measurements.

### 2. The pipeline measures both bandwidth and modulation, but its formal
errors are not an independent validation

The main ACF estimator mean-normalizes the burst spectrum, omits lag zero from
the fitted data, and adds statistical and finite-scintle terms to the ACF error
(`scintillation/scint_analysis/analysis.py:234-240`,
`scintillation/scint_analysis/analysis.py:287-316`). The model layer reads the
Lorentzian HWHM as `Delta nu_d` and the fitted Lorentzian amplitude parameter as
`m`; their uncertainties are the `lmfit` parameter standard errors
(`scintillation/scint_analysis/analysis.py:1489-1518`,
`scintillation/scint_analysis/analysis.py:1571-1612`). These are useful fit
errors, but they do not by themselves test sensitivity to the frequency mask,
sub-band definition, fit range, low-lag excision, or time-window choice.

The independent revalidation estimator uses a separately implemented
mean-normalized, mask-aware ACF and a Lorentzian plus constant with lag zero
excluded (`scintillation/scint_analysis/revalidation.py:1-19`,
`scintillation/scint_analysis/revalidation.py:94-133`). Its public
`revalidate_dnu` path fits both `gamma` and `m` but returns only `gamma`, with no
fit uncertainty (`scintillation/scint_analysis/revalidation.py:147-179`). The
two-screen path returns `m_wide`, `m_narrow`, and their quadrature total, but
again no uncertainties (`scintillation/scint_analysis/revalidation.py:182-250`).
The independent harness therefore needs a result-returning single-screen API
before it can validate modulation index on equal terms.

### 3. The fresh DSA campaign is a strong starting point, not a final census

The existing driver recomputes from NPZ products rather than reading legacy
`stored_fits`, evaluates 2/3/4 equal-S/N sub-band candidates, and runs the
1/2/3-Lorentzian selector
(`analysis/scintillation-dsa-lorentzian-2026-07-07/README.md:1-26`). It already
stores both `dnu_mhz` and `m`, and flags widths beyond the fit window, fractional
bandwidth errors above unity, `m > 3`, and fractional modulation errors above
unity (`analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:304-322`).

However, candidate viability only requires at least one unflagged component per
sub-band (`analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:119-153`).
The committed table consequently contains large within-burst scale changes and
some accepted-looking rows whose physical stability has not yet been tested.
The existing per-burst medians should be treated as campaign diagnostics until
they survive estimator, window, and masking checks.

### 4. CHIME is correctly fail-closed, and no current full-band config is yet
eligible as a measurement

The latest upstream guard layer requires grid regularization, bandpass
normalization, and harmonic masking, then combines that provenance gate with an
off-pulse ACF null and low-lag stability check. Its output is explicitly
`measurement` or `diagnostic_only`
(`analysis/scintillation-dsa-lorentzian-2026-07-07/README.md:51-82`,
`analysis/scintillation-dsa-lorentzian-2026-07-07/CHANGES-artifact-controls.md:38-49`).
The Freya experiment demonstrates why: off-pulse slices reproduce the apparent
CHIME scale and low-lag excision removes the wing, so the old tens-of-kHz value
is instrumental rather than scintillation.

A live config inventory found that every full-band CHIME config has harmonic
masking and all except Casey have grid regularization, but none enables bandpass
normalization. This matches the documented non-uniformity and means every
current full-band CHIME run must remain `diagnostic_only` until a controlled
variant test establishes and records the appropriate mitigation stack
(`analysis/scintillation-dsa-lorentzian-2026-07-07/CHANGES-artifact-controls.md:67-76`).
The correct next step is not to weaken the gate; it is to run paired variants
and adjudicate whether the mitigation changes the burst signal or removes an
instrumental pattern.

### 5. The existing figure producer can be generalized to the requested
combined result

The driver already produces per-burst ACF diagnostics and a sample-level DSA
bandwidth summary. Its rows retain both bandwidth and modulation plus quality
flags, and its plotting layer already distinguishes usable from flagged points
(`analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:426-469`,
`analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:634-670`).
The missing layer is a canonical two-band verdict table and a figure that shows
`Delta nu_d` and `m` together while visually separating accepted measurements,
upper limits, and diagnostic-only fits.

### 6. The current CHIME products are transferred correctly but are not
reproducible from the tracked producer

All 36 local CHIME source artifacts (12 upchannelized arrays, 12 frequency
arrays, and 12 time-zero metadata files) are byte-identical to their h17
counterparts. The tracked producer is nevertheless stale: it contains only five
targets, discards the array returned by `coherent_dedisp`, and enables
`time_shift=True`. The live h17 producer fixes the return-value bug, supports all
12 targets, and was used with no time shift. The generic metadata-alignment
builder is likewise present only beside the local products. Both exact scripts,
their container/package identity, and array hashes must be promoted before the
products can be called reproducible.

The package's `_upchannel` operation is a non-overlapping block FFT followed by
coherent averaging of adjacent fine-frequency bins. It is not a PFB inverse and
does not itself guarantee a flat passband. This matches the public reference
code accompanying the FRB 20221022A scintillation analysis:
<https://github.com/KenzieNimmo/FRB20221022A_scintillation>.

### 7. The ACF formula is recognizable, but the full fitting methodology needs
numerical calibration

The mask-aware pair-count normalization and optional
`(mean_on - mean_off)^2` denominator match the public reference convention.
However, `calculate_acf` inserts an artificial unit zero-lag point, the main fit
path duplicates positive and negative lags, finite-scintle uncertainty is added
as independent diagonal weight, and component/model BIC totals can use different
successful subband sets. The main optimizer also uses Nelder-Mead before reading
formal standard errors. Existing tests mainly fit preconstructed ACF curves and
therefore do not validate the complete spectrum-to-ACF-to-fit path.

The public CHIME reference drops lag zero and the first positive fine-channel
lag. The controlled validation should retain the symmetric ACF only for plotting,
fit one copy of each independent positive lag, begin CHIME fits at lag 2, and
calibrate bandwidth/modulation bias, interval coverage, and null false-positive
rate with seeded end-to-end spectrum injections.

## Synthesis

The data gap has been closed, but the producer and measurement gaps have not.
DSA has a fresh full-sample diagnostic campaign; CHIME has a full-sample product
inventory and the right fail-closed guards. The remaining work begins with a
blocking producer/ACF/fitting validation gate, not a simple rerender:

1. Promote and pin the exact CHIME producer/builder, reproduce array products,
   and validate upchannelization and metadata alignment with synthetic inputs.
2. Establish direct masked-pair ACF oracles and seeded end-to-end fit-recovery,
   interval-coverage, and null false-positive tests.
3. Extend the independent single-screen revalidation API to return bandwidth,
   modulation, covariance-derived errors, fit diagnostics, and the fitted
   baseline.
4. Run both estimators for both bands with the same product-level provenance,
   off-pulse-null, low-lag, mask, fit-range, and sub-band-stability records.
5. For CHIME, compare the checked-in configuration with a full-mitigation
   generated variant; retain `diagnostic_only` unless the full stack and both
   physical guards pass.
6. For DSA, apply the same physical stability diagnostics even though the CHIME
   provenance gate is inapplicable.
7. Emit one machine-readable verdict row per burst and band, including explicit
   non-measurement reasons, then render a combined bandwidth/modulation figure
   from that table only.

The acceptance unit should be a burst-band result, not an individual attractive
sub-band fit. The combined figure must never silently turn a failed guard or an
upper-bound product into a point measurement.

## References / Sources

- Code and local provenance: `scintillation/DATA_PROVENANCE.md`,
  `scintillation/scint_analysis/analysis.py`,
  `scintillation/scint_analysis/revalidation.py`, and
  `analysis/scintillation-dsa-lorentzian-2026-07-07/` at commit `863b8726`.
- Method provenance already recorded in code: Nimmo et al. (2025) and Pleunis
  et al. (2025), cited in `scintillation/scint_analysis/revalidation.py:1-19`.
- Public method implementation:
  <https://github.com/KenzieNimmo/FRB20221022A_scintillation> and associated
  paper <https://arxiv.org/abs/2406.11053>.
