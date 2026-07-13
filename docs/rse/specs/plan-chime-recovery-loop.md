# Plan — bounded Freya CHIME recovery loop

**Status:** DOCUMENTED-FAIL, bounded loop terminated
**Branch:** `scint/chime-recovery-loop` from `origin/pin/faber2026`  
**Qualification target:** Freya (`FRB 20230325A`) only

## Purpose and stop condition

Recover a CHIME scintillation product whose instrumental frequency correlation
does not masquerade as a decorrelation bandwidth. Freya is the qualification
target; no other co-detection enters the fleet until Freya passes every required
correction check and its diagnostic figure is manually reviewed.

The loop stops at the first of:

1. **PASS:** all checks in `chime_correction_validation.REQUIRED_CORRECTION_CHECKS`
   are true and the science status is `measurement`;
2. **DOCUMENTED-FAIL:** three materially distinct correction hypotheses have
   failed, with products, manifests, metrics, and reasons retained; or
3. **SCIENCE DECISION:** a result can pass only by changing the definition of
   the observable or by applying a correction whose transfer function cannot be
   validated with injections.

Missing checks remain inconclusive. A smaller off-pulse statistic is not a pass.

## Fixed controls

- Apply instrumental correction before frequency-dependent alignment.
- Mask the dispersed burst plus guard bins while estimating the correction.
- Align with padded placement; never circularly wrap samples.
- Fit unique positive CHIME lags beginning at lag 2.
- Preserve corrected and uncorrected products under different names and bind
  the corrected product to a target-specific SHA-256 manifest.
- Run the same off-pulse slices, fit windows, harmonic mask, and low-lag
  excisions used for the on-pulse result.
- Require injection recovery before accepting any whitening-like operation.

## Iterations

| Iteration | Material hypothesis | Evidence and acceptance test | State |
|---|---|---|---|
| H0 | One robust rank-1 additive time mode per parent CHIME coarse block removes alignment-sheared common mode. | Paired product `freya_chime_coarse_rank1_v1_*`; require both subbands to pass the off-pulse null and low-lag stability. | **FAIL, reproduced 2026-07-12** — builder statistic improved 0.579 to 0.335 but remained correlated. Fresh two-subband adjudication passed provenance and low-lag stability but failed the aggregate off-pulse null. |
| H1 | The remaining structure is the multiplicative intra-coarse PFB scallop documented by the rescued CANFAR recipe; estimate it only from protected off-pulse data, divide it out, then apply H0. | Seeded multiplicative injection recovery; off-pulse null; comparison of pre/post coarse-phase ACF; unchanged injected Lorentzian width within `max(10%, 0.25 channel)`. | **NO-GO before product generation** — the current builder already estimates a per-channel off-pulse gain, and standardized temporal covariance remains positive at low-band lags 1–3. Static channel scaling cannot remove standardized cross-channel covariance, so this mechanism cannot explain the failed null. This no-go does not count toward the three executed correction hypotheses. |
| H2 | The residual is a second independent additive block mode rather than a multiplicative scallop; a robust rank-2 block model is required. | Rank-2 known-truth injection with astrophysical burst masking; held-out off-pulse improvement; no loss or bias of injected widths; all fail-closed gates. | **FAIL, 2026-07-12** — the initial real-data gate cleared provenance, both off-pulse nulls, and low-lag stability, but the complete battery failed injection recovery, low-band fit-window stability, both split-time checks, high-band comb residual, held-out kernel prediction, and manual figure review. Science remains `diagnostic_only`; H2 does not unlock the fleet. |
| H3 | The residual is stationary fine-channel covariance from the upchannelization kernel and requires an off-pulse-derived linear whitening transfer function. | Independent kernel cross-check plus width/amplitude injection recovery over resolved and near-resolution scales. If the transfer function biases or erases a permitted signal, terminate as DOCUMENTED-FAIL. | **DOCUMENTED-FAIL, 2026-07-12** — both held-out half-band checks exceed 3 standard errors; 48/48 finite injections show catastrophic width bias, failed coverage, and systematic amplitude suppression. Manual review confirms the transfer erases permitted signal. It was not applied to Freya's burst. |

No parameter sweep counts as a new hypothesis. Within an iteration, parameters
are fixed from off-pulse or known instrument structure before examining the
on-pulse width.

### H3 predeclared transfer-function gate

H3 uses the rank-1 product as its additive-common-mode baseline; rank-2 output
is not reused. A single zero-phase inverse-square-root covariance transfer is
estimated per 200 MHz half-band from bins 10:105 and applied independently
inside each native 64-channel upchannelization block. The Toeplitz covariance
is projected positive definite with an eigenvalue floor fixed at 10% of its
median eigenvalue. No filter parameter is chosen from the burst spectrum.

Before any corrected Freya burst product is written, all of the following must
pass:

- held-out bins 105:200 have maximum residual fine-channel covariance no more
  than 3 standard errors from zero over lags 1--12;
- 48 seeded Lorentzian injections span HWHM values of 2, 4, 8, and 16 native
  channels, modulation indices 0.3 and 1.0, and both 200 MHz transfer functions;
- every recovered HWHM differs from known truth by less than
  `max(10%, 0.25 channel)`, with nominal-68% coverage within 0.15 of 0.68;
- every recovered modulation index differs from known truth by less than
  `max(10%, 0.05 absolute)`; and
- manual review confirms that the transfer, held-out covariance, and injection
  plots agree with the machine verdicts.

A failed injection gate forbids applying the transfer to the on-pulse Freya
spectrum and closes H3 as DOCUMENTED-FAIL rather than tuning the filter.

### H3 final evidence and loop disposition

The predeclared transfer was estimated from the rank-1 product only. Both
held-out kernel checks failed: maximum residual discrepancies were 3.90
standard errors at 400--600 MHz and 4.44 at 600--800 MHz, above the fixed 3.0
limit. All 48 injection fits were finite, but maximum fractional HWHM bias was
37.28 and nominal-68% coverage was 0.021. Direct modulation recovery also
failed, with maximum absolute bias 0.496; visual review shows systematic
suppression from injected `m=0.3` to roughly 0.15--0.25 and from `m=1.0` to
roughly 0.50--0.84.

The transfer was therefore never applied to the on-pulse Freya spectrum and no
H3 corrected burst product was written. The serialized transfer, metrics,
figure manifest, and bound review live under
`analysis/chime-recovery-2026-07-12/results/h3/`. H0, H2, and H3 are three
materially distinct executed correction failures, so the bounded recovery loop
terminates as DOCUMENTED-FAIL. Fleet propagation remains locked; changing the
observable or accepting an unvalidated transfer would require a separate
science decision.

## Post-loop bounded inference experiment A1

The owner authorized a separate science decision on 2026-07-13: test whether
the unmodified on-pulse ACF can be fit with an additive off-pulse covariance
term in the likelihood. A1 is not a fourth correction hypothesis and does not
reopen the terminated whitening loop.

The rank-1 product is the fixed input. In each of two CHIME subbands, twelve
non-overlapping off-pulse spectra use the same duration and channel boundaries
as the burst. The first six estimate the additive frequency-covariance kernel
and its full sample covariance; the final six are held out. The likelihood is
`ACF_on = Lorentzian(dnu, m) + kernel_off + constant`, with the off-kernel mean
covariance added to the on-ACF statistical covariance. The spectrum itself is
never whitened or kernel-subtracted.

Before fitting Freya's on-pulse ACF, both gates must pass:

- every held-out off-pulse ACF has maximum marginal residual no more than 3
  standard deviations and reduced predictive chi-square between 0.5 and 2.0;
- 48 deterministic injections span HWHM values of 2, 4, 8, and 16 native
  channels, modulation indices 0.3 and 1.0, both subbands, and real held-out
  off-pulse backgrounds. Every width must satisfy
  `abs(recovered-injected) < max(10%, 0.25 channel)`, nominal-68% coverage must
  be within 0.15 of 0.68, and every modulation index must satisfy
  `abs(recovered-injected) < max(10%, 0.05 absolute)`.

Failure forbids the Freya on-pulse fit and closes A1 as DOCUMENTED-FAIL without
changing the CHIME science status. Passing both gates permits the on-pulse fit,
after which the existing fit-window, split-time, split-band, comb-residual, and
manual-figure gates still apply before any measurement can be promoted.

### A1 final evidence

A1 closed as **DOCUMENTED-FAIL**. The held-out covariance test is
frequency-dependent: only 3/6 low-band slices pass (maximum absolute z 3.020;
reduced predictive chi-square range 0.842--3.358), while all 6/6 high-band
slices pass (maximum absolute z 2.160; reduced predictive chi-square range
0.790--1.416). The off-pulse region therefore supplies a usable additive-noise
model in the high band, but not a stationary model that transfers across all
of Freya's CHIME band.

All 48 real-background injection fits were finite, but the predeclared
recovery gates failed: maximum fractional width bias was 31.77, nominal-68%
coverage was 0.25, and maximum absolute modulation-index bias was 0.956. The
worst low-band, two-channel-width injections reached both width bounds,
including a 0.400 MHz recovery for an injected 0.0122 MHz signal. This is an
identifiability failure between near-resolution scintillation and the measured
low-band kernel, not merely a numerical crash.

The corrected likelihood passed a clean no-background synthetic recovery
test, and the figure review agreed with the machine-readable failures. The
Freya on-pulse fit was not performed. CHIME remains `diagnostic_only`; no
measurement or fleet propagation is permitted from A1.

### H0 reproduction evidence

Command: the runner below with `--subbands 2`; full machine-readable result was
written to `/tmp/freya-h0-gate.json`.

| CHIME subband center | On-pulse width | Off-pulse median | Ratio | Null | Low-lag stability |
|---:|---:|---:|---:|---|---|
| 513.631 MHz | 40.919 kHz | 24.673 kHz (10 fits) | 1.658 | **FAIL** | PASS, minimum ratio 0.920 |
| 713.439 MHz | 65.998 kHz | 29.756 kHz (10 fits) | 2.218 | PASS | PASS, minimum ratio 1.000 |

Aggregate status: `product_correction_status=fail`,
`science_status=diagnostic_only`. This is more specific than the earlier draft
PR summary: under the recovered pin-based runner, one of two judged subbands
fails rather than both. The scientific conclusion is unchanged, but future
reports should use this reproduced result and name the frequency dependence.

Additional mechanism diagnostic (off-pulse bins 10:200, channels standardized
individually): the corrected 400–600 MHz product retains correlations
0.0217, 0.0136, and 0.00434 at fine-channel lags 1–3; the 600–800 MHz values
are approximately -0.012. This frequency dependence motivates H2 and rules out
a static scallop-only explanation.

### H2 gate evidence

External paired products and manifest (non-overwriting):

- `~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank2_v1_uncorrected.npz`
- `~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank2_v1_corrected.npz`
- `~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank2_v1_manifest.json`

Full two-subband result: `/tmp/freya-h2-gate.json`.

| CHIME subband center | On-pulse width | Off-pulse median | Ratio | Null | Low-lag stability |
|---:|---:|---:|---:|---|---|
| 523.268 MHz | 38.616 kHz | 14.708 kHz (10 fits) | 2.626 | PASS | PASS, minimum ratio 0.803 |
| 723.076 MHz | 64.989 kHz | 21.582 kHz (10 fits) | 3.011 | PASS | PASS, minimum ratio 1.000 |

The initial artifact-control status was `measurement`, but the complete
correction battery is `fail` and the science status remains `diagnostic_only`.

Final fail-closed battery:

| Check | Verdict | Evidence |
|---|---|---|
| Manifest and provenance | PASS | Target/hash verified; required mitigations recorded. |
| Off-pulse null | PASS | Ratios 2.626 and 3.011. |
| Low-lag stability | PASS | Minimum retained-width ratios 0.803 and 1.000. |
| Injection recovery | **FAIL** | 24/24 finite fits, maximum fractional bias 34.86, nominal-68% coverage 0.125. |
| Fit-window stability | **FAIL** | Low band shifts 32.0% across 0.5-1.0 MHz, exceeding the predeclared 30% limit; high band shifts 6.9%. |
| Split-time stability | **FAIL** | Early-to-late widths collapse 47.1 to 10.4 kHz and 53.4 to 12.1 kHz (3.47 and 4.14 sigma). Late-time modulation fits are also unphysical (`m=2.93`, `1.79`). |
| Split-band stability | PASS | Both bands independently pass artifact gates and bandwidth increases with frequency; two-point scaling index 1.61. |
| Comb residual | **FAIL** | High-band harmonic/background residual ratio 2.208 exceeds the fixed 2.0 threshold. |
| Held-out kernel cross-check | **FAIL** | Maximum discrepancies 3.25 sigma (400-600 MHz) and 4.78 sigma (600-800 MHz), above 3 sigma. |
| Manual figure review | **FAIL** | All four figures reviewed; structured residuals, recovery failure, retained low-band correlation, and stability/kernel failures recorded as anomalies. |

Machine-readable evidence and the bound visual review live in
`analysis/chime-recovery-2026-07-12/results/h2/{validation.json,figures.manifest.json,figures.review.json}`.
This completes H2 as a documented failed hypothesis. H3 remains the next
bounded hypothesis; it must not reuse rank-2 outputs as measurements.

Software verification at this checkpoint:

- focused correction/product suite: 15 passed;
- full non-slow FLITS suite: 612 passed, 3 skipped, 20 deselected;
- Ruff check and format check: passed on all touched Python files.

## Reproduction command

The runner uses the checked-in Freya analysis choices, overrides only the input
product and its manifest, forces fresh ACFs, and exits 2 unless the science
status is `measurement`:

```bash
conda run -n flits env NUMBA_DISABLE_JIT=1 python \
  analysis/chime-recovery-2026-07-12/run_freya_gate.py \
  --product ~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank1_v1_corrected.npz \
  --manifest ~/Data/Faber2026/dsa110/scintillation-data/freya_chime_coarse_rank1_v1_manifest.json \
  --output /tmp/freya-h0-gate.json
```

## Promotion rule

Only a Freya PASS unlocks the 12-target campaign. Fleet results must retain
per-burst PASS/MARGINAL/DOCUMENTED-FAIL status; `isha`, `hamilton`, and
`johndoeII` remain upper-limit candidates unless their own gates justify a
different classification. No manuscript number changes in this loop before
the evidence ledger and sightline attribution matrix consume a validated
product.
