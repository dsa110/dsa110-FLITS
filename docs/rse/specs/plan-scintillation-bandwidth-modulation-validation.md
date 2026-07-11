# Implementation Plan: Two-band scintillation bandwidth and modulation validation

---
**Date:** 2026-07-09
**Author:** Codex
**Status:** In Progress
**Related Documents:**
- [Research: Two-band scintillation bandwidth and modulation-index validation](research-scintillation-bandwidth-modulation-validation.md)
- [Experiment: Freya CHIME instrumental origin](https://github.com/jakobtfaber/Faber2026/blob/main/docs/rse/specs/experiment-freya-chime-instrumental-origin.md)
---

## Overview

This plan first validates the CHIME product-generation chain and the numerical
ACF/fitting methodology, then turns the existing fresh DSA Lorentzian campaign
and CHIME artifact-control work into one reproducible two-band validation
campaign. It retains every fit as a diagnostic, but only promotes a
burst-band-component to a measurement after the upstream producer,
normalization, fit-recovery, provenance, fit-quality, off-pulse, low-lag, and
independent-estimator checks pass.

The implementation extends the independent Nimmo/Pleunis estimator so it
returns both `Delta nu_d` and `m`, generalizes the existing driver without
changing checked-in burst configs, emits canonical component and burst-band
verdict tables, and renders a combined two-metric figure strictly from those
tables.

**Goal:** Produce validated, machine-readable CHIME and DSA bandwidth and
modulation-index results for every attempted co-detection, with explicit
non-measurement reasons and a combined publication-quality figure.

**Motivation:** A converged Lorentzian fit is not enough. Freya proves that an
instrumental CHIME correlation can mimic a plausible bandwidth, while the
fresh DSA table contains unstable and weak components. The manuscript needs a
single acceptance contract before it can cite or compare either band.

## Current State Analysis

**Existing Implementation:**

- `scintillation/scint_analysis/revalidation.py:94-179` implements an
  independent single-Lorentzian ACF fit but returns only bandwidth.
- `scintillation/scint_analysis/revalidation.py:182-250` returns two-screen
  bandwidths and modulation amplitudes without errors.
- `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:1031-1249`
  runs fresh ACF extraction, component selection, CHIME guards, and burst-level
  status aggregation.
- `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:1469-1597`
  writes component CSV/JSON/Markdown and sample figures, but only one band per
  invocation and no cross-estimator verdict.
- `scintillation/scint_analysis/chime_artifact_guards.py:1-417` provides the
  fail-closed CHIME provenance, off-pulse, low-lag, and harmonic-systematic
  verdicts.

**Current Behavior:** The DSA campaign has committed diagnostics for all 12
bursts. The CHIME driver can run all 12 products but all checked-in full-band
configs remain `diagnostic_only` because no config enables bandpass
normalization. Fit errors come from one model fit and the independent estimator
cannot compare modulation.

**Current Limitations:**

- The tracked CHIME producer discards the array returned by
  `coherent_dedisp`, uses `time_shift=True`, and contains only five targets; it
  therefore cannot reproduce the current 12-target no-time-shift products.
- The live h17 producer and generic metadata-alignment builder that created the
  current products are not tracked in Git.
- The upchannelizer is a block FFT plus coherent adjacent-bin averaging, not a
  polyphase-filterbank inverse; existing prose that claims a flat passband from
  PFB inversion is incorrect.
- Existing tests do not provide a voltage-to-spectrum-to-ACF-to-fit numerical
  oracle. They can pass while relying on an artificial zero-lag point,
  duplicated positive/negative lags, or direct synthetic ACF curves.
- The main fit path uses duplicated non-independent lags, Nelder-Mead formal
  errors, diagonalized finite-scintle uncertainty, and potentially
  non-comparable BIC totals.
- Independent validation cannot compare `m` or uncertainties.
- DSA is exempt from CHIME provenance checks, but its physical stability checks
  are not part of final measurement status.
- The driver has no generated full-mitigation CHIME variant and no variant
  provenance in its run record.
- There is no canonical two-band verdict table or combined bandwidth/modulation
  figure.

## Desired End State

**New Behavior:** A blocking upstream validation stage first promotes and pins
the exact CHIME producer/builder, proves the generated arrays and metadata
mapping, and calibrates ACF normalization and fit recovery with seeded
known-truth ensembles. Only after that stage passes do three campaign
invocations produce complete diagnostic JSON:
DSA `checked_in`, CHIME `checked_in`, and CHIME `full_mitigation`. The latter
enables grid regularization and bandpass normalization and disables polynomial
baseline subtraction in-memory without editing YAML. Each selected component
contains an independent-estimator comparison. A separate reducer emits one row
per burst-band-component plus one burst-band status row. The figure producer
reads only those CSVs.

**Success Looks Like:**

- Every one of the 24 primary burst-band products has either validated component
  rows or a named failure/status reason.
- The 12 current CHIME source products have a tracked producer, builder,
  container digest, package version, input/output hashes, and confidence class.
- Known-truth tests demonstrate unbiased `Delta nu_d` and `m` recovery and
  calibrated false-positive/coverage behavior before any real product is fit.
- A component is `measurement` only if all applicable automated gates pass;
  missing or inconclusive gates remain `diagnostic_only`.
- The combined figure visibly distinguishes measurements, diagnostic fits, and
  upper-bound/single-block products and shows both `Delta nu_d` and `m`.
- Re-running from the recorded command reproduces the CSVs and figures.

## What We're NOT Doing

- [ ] Reusing retired CHIME fit pickles or YAML `stored_fits`.
- [ ] Editing the 24 checked-in burst configurations during the variant run.
- [ ] Weakening the CHIME fail-closed gate to manufacture accepted points.
- [ ] Treating a successful file transfer, metadata-alignment assertion, or
      optimizer convergence as proof of scientific correctness.
- [ ] Claiming that CHIME's PFB response was inverted or flattened by the block
      FFT upchannelizer.
- [ ] Feeding new values into the two-screen attribution matrix or manuscript
      prose in this plan; that is a downstream adoption step after visual review.
- [ ] Treating a missing/failed fit as evidence for zero scintillation.

**Rationale:** This slice establishes trustworthy measurements and figures. It
does not yet reinterpret sightlines or rewrite manuscript claims.

## Implementation Approach

**Technical Strategy:** Establish a tracked provenance spine and executable
numerical oracles before extending result contracts. Preserve compatibility at
public boundaries, but fit only independent positive lags and separate formal
fit covariance, resampling uncertainty, and finite-scintle systematic error.
All generated variants are explicit data in the run manifest. A reducer, not
plotting code, owns measurement eligibility.

**Key Architectural Decisions:**

1. **Decision:** Preserve `revalidate_dnu` as a scalar compatibility wrapper and
   add `fit_single_screen_acf` as the structured API.
   - **Rationale:** Existing callers keep working while the campaign gets `m`,
     errors, baseline, redchi, and success state.
   - **Trade-offs:** One extra API surface, but no hidden return-type break.
   - **Alternative considered:** Change `revalidate_dnu` to return a dict;
     rejected because it would silently break scalar consumers.
2. **Decision:** Generate CHIME mitigation variants in memory.
   - **Rationale:** The experiment must not silently rewrite science configs.
   - **Trade-offs:** The run manifest must carry the exact overlay.
   - **Alternative considered:** Commit full-stack changes to all YAML files;
     rejected until each product is visually adjudicated.
3. **Decision:** Use a reducer-owned component verdict.
   - **Rationale:** Plotters cannot promote or suppress measurements ad hoc.
   - **Trade-offs:** Diagnostic JSON is richer than the canonical CSV.
   - **Alternative considered:** Plot directly from per-band JSON; rejected
     because status logic would be duplicated.
4. **Decision:** Cross-estimator agreement uses a 3-sigma test when both errors
   are finite; otherwise it requires a positive-value ratio no larger than 2.
   - **Rationale:** This uses reported uncertainty when available and the
     repository's existing factor-of-two conservative scale when it is not.
   - **Trade-offs:** The fallback is intentionally coarse and will demote
     borderline cases for review.
5. **Decision:** Phase 0 is a hard gate: no real bandwidth/modulation campaign
   or combined figure starts until every upstream acceptance criterion passes,
   or the failure is recorded as an explicit blocker.
   - **Rationale:** Current products were made by untracked code, and existing
     fit tests do not validate the complete numerical path.
   - **Trade-offs:** Scientific output is delayed if producer or fit coverage
     fails, but invalid measurements cannot flow downstream.
6. **Decision:** Preserve the symmetric ACF representation for plotting and
   compatibility, but mark inserted lag zero as synthetic and fit only one copy
   of each positive lag. CHIME upchannelized fits begin at channel lag 2; DSA
   begins at lag 1.
   - **Rationale:** Negative lags duplicate the same channel pairs, lag zero is
     dominated by self-noise, and the first CHIME fine-channel lag is an
     instrumental-resolution control.
   - **Trade-offs:** Some historical fit values may shift and must remain
     diagnostics rather than anchors.
7. **Decision:** Use least-squares covariance only as a formal diagnostic;
   report seeded resampling intervals as the primary fit uncertainty and keep
   finite-scintle uncertainty as a separate systematic term.
   - **Rationale:** ACF lags and finite-scintle effects are correlated, so adding
     them as independent per-lag errors overstates information.
   - **Trade-offs:** Seeded ensembles and real-spectrum resampling cost more
     runtime than one Nelder-Mead fit.

**Patterns to Follow:**

- Pure JSON-able guard verdicts — `scintillation/scint_analysis/chime_artifact_guards.py:1-417`.
- Testable synthetic Lorentzian fixtures — `scintillation/scint_analysis/tests/test_revalidation.py:96-129`.
- Fresh-run, no-cache driver configuration —
  `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:57-91`.

## Implementation Phases

### Phase 0: CHIME producer, ACF, and fitting validation (blocking gate)

**Objective:** Prove that the current CHIME inputs are reproducible and that the
exact spectrum-to-ACF-to-fit path recovers known bandwidths and modulation
indices with calibrated uncertainty before running any real two-band campaign.

#### Phase 0A: Promote and pin the CHIME provenance spine

**Tasks:**

- [ ] Replace the stale five-target implementation in
  `analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py:68-223`
  with the exact live h17 12-target producer, preserving the fixed
  `dedisp = coherent_dedisp(...)` return path and explicit
  `--no-time-shift` behavior.
- [ ] Promote
  `~/Data/Faber2026/dsa110/upchan_codetections/build_npz_aligned_generic_20260706.py`
  as
  `analysis/scattering-refit-2026-06/baseband_recovery/build_npz_aligned.py` and
  remove target-specific validation exceptions in favor of declarative
  per-product confidence thresholds.
- [ ] Add
  `analysis/scattering-refit-2026-06/baseband_recovery/products/chime_upchannelization_manifest.json`
  recording target parameters, source metadata, input/output SHA-256 hashes,
  the image digest
  `chimefrb/baseband-analysis@sha256:f510909d892d0d5224c982c590cbe80967a49a59b79c396ab72bb710105c4c41`,
  image ID, `baseband_analysis==1.9.0`, command, and UTC generation time.
- [ ] Recompute and compare the 36 local/remote hashes and require exact
  equality; classify products as `full_measurement_candidate`, `marginal`, or
  `single_block_upper_bound`. Hamilton and johndoeII begin as upper bounds;
  Oran begins as marginal until stronger alignment evidence passes.
- [ ] Correct PFB-inverse/flat-passband claims in
  `analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py:13-28`,
  the baseband-recovery README, and product provenance.

**Verification:** A clean checkout contains the exact code needed to regenerate
all 12 products; the manifest validates all 36 hashes; no tracked documentation
claims a PFB inversion; confidence classes are explicit and fail closed.

#### Phase 0B: Validate upchannelization and metadata alignment

**Tasks:**

- [ ] Add
  `analysis/scattering-refit-2026-06/baseband_recovery/test_upchannelize_chime.py`
  with deterministic complex-voltage injections that prove the returned
  coherent-dedispersion array is consumed and that the discarded-return path
  fails the oracle.
- [ ] In the pinned container, test no-time-shift boundary behavior, exact
  output shape/frequency ordering/channel width, FFT power scaling, a swept
  narrowband tone, and white-noise adjacent-bin covariance. Require all array
  invariants exactly and every null correlation to remain within its seeded
  5-sigma Monte Carlo envelope.
- [ ] Add
  `analysis/scattering-refit-2026-06/baseband_recovery/test_build_npz_aligned.py`
  covering the `time0` delay reconstruction, nearest-parent coarse-channel
  mapping, ascending-frequency flip, `2.56e-6 * 2 * U` time step, padded NaN
  regions, and absence of circular overlap.
- [ ] Regenerate each packaged NPZ from the byte-verified source artifacts and
  compare frequency, time, power, and mask arrays exactly or with a documented
  floating-point tolerance. Store array-level hashes in the manifest.
- [ ] Tighten real-product alignment gates: report eligible high-frequency
  slice count, peak scatter in native time bins, tolerance source, rejected
  slices, and known RFI bands. A wide burst FWHM alone cannot create a passing
  tolerance; fewer than three independent eligible slices is inconclusive.

**Verification:** Synthetic injections distinguish the fixed producer from both
known defective generations; current NPZs are reproduced from source artifacts;
weak Oran/johndoeII checks cannot pass as full validation.

#### Phase 0C: Establish ACF computation and normalization oracles

**Tasks:**

- [ ] Add `scintillation/scint_analysis/tests/test_acf_numerical_oracles.py`
  and compare `calculate_acf` at
  `scintillation/scint_analysis/analysis.py:234-340` against a direct pair-loop
  oracle for masks, gaps, nonzero off-burst means, and every retained lag.
- [ ] Verify pair-count normalization with analytically tractable constant,
  alternating, impulse, and white-noise spectra. Assert the returned measured
  positive-lag values rather than the inserted zero-lag plotting sentinel.
- [ ] Add explicit ACF metadata distinguishing measured lags from the synthetic
  zero-lag point; ensure no fit or modulation estimate consumes the sentinel.
- [ ] Test additive correlated off-pulse structure with and without the
  off-burst correction and require the off-pulse/null path to reject the
  instrumental correlation.
- [ ] Make the noise-template path at
  `scintillation/scint_analysis/analysis.py:399-453` use the actual subband
  slice/mask and a recorded seed; add a regression test that different subband
  masks cannot reuse the same first-row template.

**Verification:** The implementation matches the independent direct oracle at
machine precision; masked-pair counts are exact; white-noise positive lags are
consistent with zero; artificial lag zero never enters fitting or `m`.

#### Phase 0D: Validate fitting, model selection, and uncertainties

**Tasks:**

- [ ] Add end-to-end seeded spectrum-to-ACF-to-fit ensembles for one-screen,
  two-screen, white-noise, and correlated-instrumental cases; vary S/N, masks,
  scintle count, channel resolution, and bandwidth-to-subband ratio.
- [ ] Change `_fit_acf_models` at
  `scintillation/scint_analysis/analysis.py:685-779` to fit one copy of each
  positive lag, using first lag 2 for CHIME and 1 for DSA, and use an optimizer
  with an estimable least-squares covariance.
- [ ] Keep finite-scintle uncertainty out of the per-lag diagonal weights;
  report formal covariance, seeded/resampling interval, and finite-scintle
  systematic separately in the structured result.
- [ ] Change `_select_overall_best_model` at
  `scintillation/scint_analysis/analysis.py:1337-1382` so BIC comparisons use
  identical successful subbands and identical lag samples. Change
  `_determine_n_components` at
  `scintillation/scint_analysis/analysis.py:1385-1420` to pass the available ACF
  uncertainties and record the common comparison set.
- [ ] Across at least 200 fixed seeds per nominal condition, require median
  `Delta nu_d` and `m` bias below 10%, empirical coverage of the nominal 95%
  resampling interval between 90% and 98%, and a false measurement rate no
  larger than 5% for white-noise and correlated-instrumental nulls. Conditions
  that cannot meet these thresholds define an explicit non-measurement region.

**Verification:** Known truth is recovered from spectra rather than prebuilt ACF
curves; uncertainties have calibrated coverage; model BICs are comparable; null
ensembles do not manufacture scintillation measurements.

#### Phase 0E: Real-data controls and gate decision

**Tasks:**

- [ ] Run Freya CHIME as the required negative control and reproduce the
  on/off-pulse narrow feature and its low-lag collapse documented in the parent
  manuscript experiment. It must remain `diagnostic_only`.
- [ ] Run the Freya DSA stable wing as the positive control and require recovery
  consistent across lag excision, subbanding, and the independent estimator.
- [ ] Write `PHASE0_VALIDATION.md` with every acceptance criterion, exact
  command, seed, hash, result, plot, and `PASS`/`FAIL`/`INCONCLUSIVE` verdict.
- [ ] Proceed to Phase 1 only if every required Phase 0 gate is `PASS`. A failed
  or inconclusive gate halts the real campaign and is reported as a blocker;
  downstream fitting and figures are not produced.
- [ ] After an overall `PASS`, commit only the promoted producer/builder,
  manifests, Phase 0 tests, numerical-method changes, corrected provenance, and
  `PHASE0_VALIDATION.md` with commit message
  `fix(scint): validate CHIME products and ACF methodology`.

**Dependencies:** Pinned h17 container and local byte-verified source products.

**Verification:** The negative control remains rejected, the positive control
recovers a stable resolved wing, and `PHASE0_VALIDATION.md` records an overall
`PASS` before Phase 1 starts.

### Phase 1: Structured independent one- and two-screen estimators

**Objective:** Return independently fitted bandwidths, modulation amplitudes,
uncertainties, constant, and fit quality for the one- and two-screen cases while
preserving the scalar wrapper.

**Tasks:**

- [ ] Add a failing known-ACF test in
  `scintillation/scint_analysis/tests/test_revalidation.py:96-129`:

  ```python
  def test_single_screen_result_recovers_gamma_and_modulation():
      lags, acf = _synthetic_acf([(0.12, 0.75)], noise=1e-3, seed=11)
      out = fit_single_screen_acf_from_acf(lags, acf)
      assert out["success"] is True
      assert out["dnu_mhz"] == pytest.approx(0.12, rel=0.10)
      assert out["m"] == pytest.approx(0.75, rel=0.10)
      assert np.isfinite(out["dnu_err_mhz"])
      assert np.isfinite(out["m_err"])
  ```

- [ ] Run and observe the missing-symbol failure:
  `NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python -m pytest scintillation/scint_analysis/tests/test_revalidation.py::test_single_screen_result_recovers_gamma_and_modulation -q`.
- [ ] Add `fit_single_screen_acf_from_acf(lags, acf, acf_err=None)` at
  `scintillation/scint_analysis/revalidation.py:136` using the existing
  `_lorentz_w_c` model and return:

  ```python
  {
      "success": bool(result.success),
      "dnu_mhz": abs(float(result.params["gamma"].value)),
      "dnu_err_mhz": _param_stderr(result.params["gamma"]),
      "m": abs(float(result.params["m"].value)),
      "m_err": _param_stderr(result.params["m"]),
      "constant": float(result.params["c"].value),
      "constant_err": _param_stderr(result.params["c"]),
      "redchi": float(result.redchi),
  }
  ```

- [ ] Add
  `fit_single_screen_acf(spec, channel_width_mhz, max_lag_mhz=None, rfi_n_sigma=5.0, first_lag=1, offspec_mean=None)`
  to build the independent ACF and call the result function. Change
  `revalidate_dnu` to return the `dnu_mhz` field from this function using the
  same six arguments.
- [ ] Extend the existing `fit_two_screen_acf` return at
  `scintillation/scint_analysis/revalidation.py:182-250` with
  `dnu_wide_err_mhz`, `dnu_narrow_err_mhz`, `m_wide_err`, `m_narrow_err`,
  `constant`, `constant_err`, and `redchi` from the same independent fit. Add
  assertions to `test_two_screen_wide_and_narrow_recovered` that all four
  parameter errors are finite.
- [ ] Run the complete revalidation tests and expect all pass:
  `NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python -m pytest scintillation/scint_analysis/tests/test_revalidation.py -q`.
- [ ] Commit Phase 1 paths with
  `git add scintillation/scint_analysis/revalidation.py scintillation/scint_analysis/tests/test_revalidation.py && git commit -m "feat(scint): return independent bandwidth and modulation fits"`.

**Dependencies:** Current `lmfit`, NumPy, and synthetic fixtures.

**Verification:** The known `(gamma, m)=(0.12, 0.75)` ACF is recovered within
10%, the seeded two-screen fixture returns two separated components, both paths
have finite formal errors, and all previous scalar tests remain green.

### Phase 2: Driver-level variants and independent component checks

**Objective:** Run the same campaign in both bands with explicit config overlay
and component-level cross-estimator verdicts.

**Tasks:**

- [ ] Add failing tests to
  `analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py`:

  ```python
  def test_full_mitigation_overlay_is_chime_only():
      chime = driver._apply_mitigation_variant(_chime_cfg(), "full_mitigation")
      assert chime["analysis"]["grid_regularization"]["enable"] is True
      assert chime["analysis"]["bandpass_normalization"]["enable"] is True
      assert chime["analysis"]["baseline_subtraction"]["enable"] is False
      dsa = {"telescope": "dsa", "analysis": {}}
      assert driver._apply_mitigation_variant(dsa, "full_mitigation") == dsa

  def test_cross_estimator_agreement_prefers_sigma_then_ratio():
      assert driver._cross_estimator_verdict(1.0, 0.1, 1.2, 0.1)["pass"] is True
      assert driver._cross_estimator_verdict(1.0, np.nan, 2.1, np.nan)["pass"] is False
  ```

- [ ] Run and observe both failures:
  `NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python -m pytest analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py -q`.
- [ ] Add `_apply_mitigation_variant`, `_cross_estimator_verdict`, and CLI
  `--mitigation-variant {checked_in,full_mitigation}` in
  `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py:57-91,1494-1523`.
- [ ] In `_fit_prepared_config` at lines 1031-1179, extract each identical
  channel slice from `pipe.masked_spectrum.get_spectrum(pipe.burst_lims)`. Run
  `fit_single_screen_acf` when `n_preferred == 1`, `fit_two_screen_acf` when
  `n_preferred == 2`, and record `unsupported_component_count` when
  `n_preferred > 2`; use `first_lag=2` for CHIME and `1` for DSA. Match two
  components by ascending bandwidth and store `independent_revalidation` plus
  `cross_estimator` beside each selected component.
- [ ] Make final status fail-closed for both bands when off-pulse, low-lag, or
  cross-estimator checks fail or are inconclusive. Keep the CHIME provenance
  requirement conditional on telescope.
- [ ] Record `band`, `mitigation_variant`, and the exact overlay in the top-level
  `run` block at `run_dsa_lorentzian_fits.py:1562-1581`.
- [ ] Run driver tests and the scintillation suite:
  `NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python -m pytest analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py scintillation/scint_analysis/tests -q`.
- [ ] Commit Phase 2 paths with
  `git add analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py && git commit -m "feat(scint): validate both-band fits across estimators"`.

**Dependencies:** Phase 1.

**Verification:** Tests prove that full mitigation never changes DSA config,
that CHIME overlays are explicit, and that estimator disagreement demotes rather
than disappears.

### Phase 3: Canonical reducer and controlled campaign

**Objective:** Produce canonical burst-band-component and burst-band tables from
three recorded campaign invocations.

**Tasks:**

- [ ] Add `analysis/scintillation-two-band-validation-2026-07-09/test_reduce_results.py`
  with a minimal fixture containing one accepted component, one failed null,
  and one upper-bound product. Assert exact statuses and reason tokens:

  ```python
  assert rows[0]["status"] == "measurement"
  assert rows[1]["status"] == "diagnostic_only"
  assert "off_pulse_null" in rows[1]["status_reasons"]
  assert rows[2]["product_caveat"] == "upper_bound"
  ```

- [ ] Run and observe the missing-module failure:
  `~/.conda/envs/py312/bin/python -m pytest analysis/scintillation-two-band-validation-2026-07-09/test_reduce_results.py -q`.
- [ ] Implement `reduce_results.py` to read per-band JSON, preserve every
  component, and write `two_band_components.csv` and
  `two_band_burst_status.csv`. Status reasons are a semicolon-separated sorted
  vocabulary: `provenance`, `off_pulse_null`, `low_lag_stability`,
  `cross_estimator`, `fit_quality`, `product_caveat`, or `fit_failure`.
- [ ] Run the three campaigns with the clean `py312` interpreter:

  ```bash
  NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python \
    analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py \
    --band dsa --mitigation-variant checked_in --keep-going \
    --output-dir analysis/scintillation-two-band-validation-2026-07-09/results/dsa
  NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python \
    analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py \
    --band chime --mitigation-variant checked_in --keep-going \
    --output-dir analysis/scintillation-two-band-validation-2026-07-09/results/chime-checked-in
  NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python \
    analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py \
    --band chime --mitigation-variant full_mitigation --keep-going \
    --output-dir analysis/scintillation-two-band-validation-2026-07-09/results/chime-full-mitigation
  ```

- [ ] Reduce the DSA checked-in and CHIME full-mitigation primary runs while
  attaching checked-in CHIME as the systematic comparator:
  `~/.conda/envs/py312/bin/python analysis/scintillation-two-band-validation-2026-07-09/reduce_results.py --dsa analysis/scintillation-two-band-validation-2026-07-09/results/dsa/dsa_lorentzian_fits.json --chime analysis/scintillation-two-band-validation-2026-07-09/results/chime-full-mitigation/chime_lorentzian_fits.json --chime-comparator analysis/scintillation-two-band-validation-2026-07-09/results/chime-checked-in/chime_lorentzian_fits.json --output-dir analysis/scintillation-two-band-validation-2026-07-09/results/combined`.
- [ ] Commit reducer, tests, JSON/CSV/Markdown summaries, and small diagnostic
  figures; omit caches and transient logs.

**Dependencies:** Phases 0-2 and all 24 primary NPZ products.

**Verification:** Both canonical CSVs cover all 24 burst-band attempts; no row
with a failed or inconclusive required gate is labeled `measurement`.

### Phase 4: Combined bandwidth and modulation figure

**Objective:** Render both quantities from the canonical table without
recomputing or reclassifying fits.

**Tasks:**

- [ ] Add `test_plot_two_band_summary.py` with a three-row CSV fixture and assert
  that measurement, diagnostic-only, and upper-bound marker categories are all
  present in the plot payload returned by `build_plot_payload`.
- [ ] Run and observe the missing-function failure:
  `~/.conda/envs/py312/bin/python -m pytest analysis/scintillation-two-band-validation-2026-07-09/test_plot_two_band_summary.py -q`.
- [ ] Implement `plot_two_band_summary.py` with a 12-row by 2-column layout:
  left = `Delta nu_d` on a log axis, right = `m`; x-axis = observing frequency;
  CHIME and DSA use distinct colors; accepted measurements are filled;
  diagnostics are hollow; upper bounds use downward arrows; multi-scale
  components use circle/diamond/square shapes. The plotting function consumes
  only `two_band_components.csv`.
- [ ] Write PNG, SVG, and PDF to
  `analysis/scintillation-two-band-validation-2026-07-09/results/combined/figures/`.
- [ ] Add a second `two_band_validation_matrix.pdf` with one burst per row and
  the gate outcomes for both bands, so excluded points are auditable beside the
  science figure.
- [ ] Run plot tests and verify PDF text/font integrity with
  `pdffonts analysis/scintillation-two-band-validation-2026-07-09/results/combined/figures/two_band_scintillation_summary.pdf` and
  `pdftotext analysis/scintillation-two-band-validation-2026-07-09/results/combined/figures/two_band_scintillation_summary.pdf - | rg "CHIME|DSA|Modulation"`.
- [ ] Commit Phase 4 artifacts with
  `git add analysis/scintillation-two-band-validation-2026-07-09 && git commit -m "feat(scint): render validated two-band summary figures"`.

**Dependencies:** Phase 3 canonical CSV.

**Verification:** Reclassifying a fixture row changes its marker category, and
the plotting code contains no status thresholds or fit calls.

### Phase 5: Reproducibility, review, and publish

**Objective:** Prove the campaign is rerunnable and land only the isolated
scintillation lane.

**Tasks:**

- [ ] Add `analysis/scintillation-two-band-validation-2026-07-09/README.md` with
  commit SHA, environment executable, NumPy/SciPy/lmfit versions, input paths,
  SHA-256 hashes for 24 primary NPZs, commands, output inventory, and accepted
  measurement count.
- [ ] Add a data-free fixture replay command and expected hashes for the two
  canonical CSVs.
- [ ] Run the complete targeted suite and reducer/plot replay twice; compare
  CSV hashes and require exact equality.
- [ ] Visually review every per-burst ACF diagnostic plus the combined figure;
  record `accepted`, `diagnostic_only`, or `upper_bound` with a reason for all
  24 burst-band attempts in `VALIDATION_REVIEW.md`.
- [ ] Run
  `mskill tool agent-closeout-check --repo "$PWD" --touched scintillation/scint_analysis/revalidation.py --touched analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py --touched analysis/scintillation-two-band-validation-2026-07-09 --json`.
- [ ] Commit any review/reproducibility corrections, read
  `~/.codex/publish-policy.toml`, run the non-destructive push gate, push branch
  `scint/revalidate-bandwidth-modulation-2026-07`, and open a focused draft PR.

**Dependencies:** Phases 0-4.

**Verification:** Two clean replays have identical canonical CSV hashes;
targeted tests and closeout pass; the PR contains no DM campaign, parent
manuscript, journal, readiness-board, or submodule-pointer changes.

## Success Criteria

### Automated Verification

- [ ] Phase 0 producer/builder tests pass inside the pinned CHIME image, and the
  manifest verifies all 36 source-artifact hashes plus regenerated NPZ array
  hashes.
- [ ] Direct pair-loop ACF oracles match `calculate_acf` at machine precision
  for masks, gaps, and nonzero off-burst means; lag zero is never fit.
- [ ] At least 200 fixed-seed end-to-end realizations per nominal condition meet
  the 10% median-bias, 90-98% coverage, and at-most-5% null false-measurement
  thresholds defined in Phase 0D.
- [ ] `PHASE0_VALIDATION.md` records an overall `PASS` before any Phase 1-5
  campaign artifact is generated.
- [ ] `NUMBA_DISABLE_JIT=1 ~/.conda/envs/py312/bin/python -m pytest scintillation/scint_analysis/tests analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py analysis/scintillation-two-band-validation-2026-07-09 -q` passes.
- [ ] Three requested campaign invocations finish with explicit success/failure
  records; no burst silently disappears.
- [ ] Canonical tables contain 24 burst-band status rows and at least one
  component or a named fit failure per attempt.
- [ ] No failed/inconclusive required gate appears with `status=measurement`.
- [ ] Two consecutive reducer/plot replays produce byte-identical CSVs.
- [ ] PNG, SVG, and PDF combined figures exist and PDF text extraction succeeds.
- [ ] `agent-closeout-check` passes for the isolated FLITS worktree.

### Manual Verification

- [ ] The promoted h17 producer and builder match the product-generation code
  actually used, and the tracked prose describes block-FFT upchannelization
  without claiming PFB inversion.
- [ ] Synthetic tone, white-noise, boundary, and metadata-alignment diagnostics
  show no unexplained channel-order, normalization, covariance, or wraparound
  behavior.
- [ ] All 24 on-pulse windows and frequency masks contain the intended burst
  emission and do not promote obvious RFI/bandpass structure.
- [ ] CHIME checked-in versus full-mitigation diagnostics show whether the
  overlay removes instrumental structure without erasing the burst spectrum.
- [ ] Every accepted component has a visible Lorentzian wing, a passing
  off-pulse null, stable low-lag refits, and reasonable residuals.
- [ ] The combined figure is legible at manuscript column/page width and its
  legend unambiguously distinguishes status, band, component, and upper limit.

### Reproducibility & Correctness

- [ ] Seeds, input hashes, code SHA, dependency versions, and exact commands are
  recorded per `ai-research-workflows:ensuring-reproducibility`.
- [ ] The independent estimator recovers synthetic `gamma` and `m` from input
  spectra—not only prebuilt ACF curves—within the Phase 0D calibration bounds.
- [ ] The known Freya CHIME artifact remains `diagnostic_only`.
- [ ] The Freya DSA positive control retains a stable resolved wing under lag
  excision, subbanding, and independent estimation.
- [ ] Results reproduce from the isolated branch without YAML `stored_fits` or
  cached ACF products.

## Testing Strategy

**Unit Test Coverage:** Coherent-dedispersion return handling, no-wrap
upchannelization, FFT/frequency/power invariants, metadata alignment, exact
masked-pair ACF normalization, synthetic-lag metadata, structured single-screen
results, backward-compatible scalar return, config overlay scope, sigma/ratio
cross-estimator verdicts, status reduction, and marker-category payloads.

**Integration Tests:** Seeded voltage/spectrum-to-ACF-to-fit ensembles, exact
source-to-NPZ regeneration, real Freya CHIME negative control, Freya DSA
positive-control run, all-product keep-going campaign, reducer, and figure
replay.

**Manual Testing:** Inspect on/off windows, masks, ACF wings, residuals, variant
differences, and final figure readability.

**Test Data Requirements:** Synthetic seeded complex voltages, spectra, and
Lorentzian ACFs; the 36 byte-verified source artifacts at
`~/Data/Faber2026/dsa110/upchan_codetections/`; and local NPZ products at
`~/Data/Faber2026/dsa110/scintillation/data/`. Tests do not download data.

## Migration Strategy

No existing result is overwritten. New outputs live under
`analysis/scintillation-two-band-validation-2026-07-09/`. Downstream manuscript
adoption will copy selected figure artifacts and pin the FLITS commit in a
separate PR.

**Rollback Plan:** Revert the focused commits or close the branch; existing DSA
campaign outputs and checked-in configs remain unchanged.

**Backward Compatibility:** `revalidate_dnu` remains scalar and the existing
driver defaults stay `--band dsa --mitigation-variant checked_in`.

## Risk Assessment

1. **Risk:** Full bandpass normalization suppresses real broad spectral
   structure.
   - **Likelihood:** Medium
   - **Impact:** High
   - **Mitigation:** Preserve checked-in and full-stack variants, require visual
     review, and never edit YAML during the campaign.
2. **Risk:** Formal lmfit errors understate correlated ACF uncertainty.
   - **Likelihood:** High
   - **Impact:** Medium
   - **Mitigation:** Require independent-estimator, off-pulse, low-lag, and
     sub-band checks; report systematics separately from formal errors.
3. **Risk:** Runtime or memory pressure for high-U CHIME products.
   - **Likelihood:** Medium
   - **Impact:** Medium
   - **Mitigation:** Run one burst at a time with `--keep-going`, no MC template,
     no cache, and detached logging if the full campaign exceeds the session.
4. **Risk:** The promoted live producer differs from the exact historical code
   that created one or more products.
   - **Likelihood:** Medium
   - **Impact:** High
   - **Mitigation:** Pin script and image hashes, regenerate array products, and
     require array-level equality before accepting provenance.
5. **Risk:** Nominal optimizer errors pass while empirical coverage fails.
   - **Likelihood:** High
   - **Impact:** High
   - **Mitigation:** Gate on fixed-seed coverage and false-positive ensembles;
     define failed conditions as non-measurement regions rather than relaxing
     thresholds after seeing real data.

## Edge Cases and Error Handling

1. **Case:** A required Phase 0 gate is inconclusive.
   - **Expected Behavior:** Stop before the real campaign and record the blocker
     in `PHASE0_VALIDATION.md`.
   - **Implementation:** Only an explicit Phase 0 `PASS` unlocks Phase 1.
2. **Case:** A required per-product downstream gate is inconclusive.
   - **Expected Behavior:** `diagnostic_only`, reason token names the gate.
   - **Implementation:** Reducer requires explicit `True` for every applicable
     gate; `None` never passes.
3. **Case:** Fit succeeds but width exceeds fit window or error exceeds value.
   - **Expected Behavior:** Retain the row, add `fit_quality`, do not promote.
4. **Case:** Multiple statistically supported components.
   - **Expected Behavior:** Preserve each component row and shape separately;
     do not collapse them into one median.
5. **Case:** Burst fails completely.
   - **Expected Behavior:** Burst-band status row with `fit_failure`; no fabricated
     zero/limit.
6. **Case:** Fewer than three independent alignment slices survive.
   - **Expected Behavior:** Product is `marginal` or `single_block_upper_bound`,
     never a full measurement candidate.

## Performance Considerations

- **Expected Load:** 36 real runs (12 DSA checked-in, 12 CHIME checked-in, 12
  CHIME full mitigation), each evaluating 2/3/4 sub-band candidates, after
  producer/builder tests and at least 200 fixed-seed realizations per nominal
  Phase 0D condition.
- **Performance Target:** Complete serially without exceeding available memory;
  correctness outranks wall time.
- **Optimization Strategy:** Disable numba JIT cache reuse, MC noise templates,
  intermediate caches, diagnostic plotting during fit, and 2D fits; render from
  stored results after fitting.

## Documentation Updates

- [ ] Research and plan documents in `docs/rse/specs/`.
- [ ] Tracked CHIME producer, builder, manifest, and corrected baseband-recovery
      provenance/README.
- [ ] `PHASE0_VALIDATION.md` with upstream acceptance evidence and gate verdict.
- [ ] Campaign README and validation-review ledger.
- [ ] Existing driver README updated for mitigation variant and two-band reducer.
- [ ] Generated Markdown summary linking every per-burst diagnostic.

## References

**Research Documents:**
- [Research: Two-band scintillation bandwidth and modulation-index validation](research-scintillation-bandwidth-modulation-validation.md)

**Experiment Reports:**
- [Freya CHIME instrumental-origin experiment](https://github.com/jakobtfaber/Faber2026/blob/main/docs/rse/specs/experiment-freya-chime-instrumental-origin.md)

**Files Analyzed:**
- `scintillation/DATA_PROVENANCE.md`
- `scintillation/scint_analysis/analysis.py`
- `scintillation/scint_analysis/revalidation.py`
- `scintillation/scint_analysis/chime_artifact_guards.py`
- `analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py`
- `analysis/scattering-refit-2026-06/baseband_recovery/npy_to_npz.py`
- `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py`

## Review History

### Version 1.0 — 2026-07-09

- Initial direct-mode plan grounded in commit `863b8726`.
- Locked fail-closed status semantics, generated CHIME variants, independent
  `m` validation, component-preserving reducer, and combined two-metric figure.

### Version 1.1 — 2026-07-09

- Added blocking Phase 0 for CHIME producer/builder provenance,
  upchannelization, metadata alignment, ACF normalization, fitting calibration,
  and real-data controls.
- Required independent positive-lag fits, comparable model-selection samples,
  resampling coverage, and separate finite-scintle systematics.
- Corrected the campaign count, primary-product dependency, and external Freya
  experiment reference.
