# CHIME artifact-control guards — change note

Date: 2026-07-09. Scope: promote the freya CHIME instrumental-origin
experiment (`docs/rse/specs/experiment-freya-chime-instrumental-origin.md`,
arms A/B1/C; "Next Steps" #1) from one-off scripts into standing guards in the
Lorentzian scintillation driver, and close the confirmed `--band chime`
harmonic-mask wiring trap.

## Files changed

- **new** `scintillation/scint_analysis/chime_artifact_guards.py` — pure,
  I/O-free verdict functions: `apply_harmonic_mask_to_fit`,
  `chime_provenance_status`, `off_pulse_null_verdict`,
  `low_lag_stability_verdict`, `harmonic_mask_systematic`,
  `finalize_measurement_status`.
- `analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py`
  — wire the guards into `_fit_prepared_config`; add helpers
  `_representative_width_mhz`, `_fit_width`, `_low_lag_excision_widths`,
  `_off_pulse_null_widths`, `_artifact_control_summary`; add the
  artifact-control column + prose to the Markdown report.
- `scintillation/scint_analysis/analysis.py` — additive: record
  `subband_channel_slices` in `calculate_acfs_for_subbands` so the off-pulse
  null can re-slice noise on the identical sub-band channel boundaries.
- `scintillation/scint_analysis/pipeline.py` — additive: expose the resolved
  `burst_lims` / `off_pulse_lims` on the pipeline object.
- **new tests** `scintillation/scint_analysis/tests/test_chime_artifact_guards.py`
  (16 unit tests) and
  `analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py`
  (5 driver-level integration tests).

## New JSON fields

Per sub-band (`result["subbands"][i]`):

| field | meaning |
|---|---|
| `harmonic_mask` | `{enabled, spacing_mhz, halfwidth_mhz, n_bins_removed, n_bins_kept}` — comb lag bins excluded from the fit |
| `harmonic_mask_systematic` | `{dnu_unmasked_mhz, dnu_masked_mhz, abs_diff_mhz, systematic_frac}` — mask sensitivity as a systematic band |
| `off_pulse_null` | `{null_pass, on_dnu_mhz, off_median_dnu_mhz, off_n_fits, ratio, reason}` — arm A |
| `low_lag_stability` | `{stable, dnu_full_mhz, dnu_by_excision, min_ratio, failed_ks, reason}` — arm B1 |

Per burst (`result`):

| field | meaning |
|---|---|
| `measurement_status` | `"measurement"` or `"diagnostic_only"` |
| `artifact_control.provenance` | `{is_chime, records, missing, status, reason}` — mitigation-stack gate |
| `artifact_control.off_pulse_null` | burst-aggregated null (`null_pass` = all judged sub-bands pass) |
| `artifact_control.low_lag_stability` | burst-aggregated stability |
| `artifact_control.failed_checks` | named reasons a CHIME burst was demoted |

## Behaviour

- **DSA unchanged.** No DSA config enables `harmonic_mask`, so the mask is a
  no-op there and the DSA fit is numerically identical (verified: the committed
  driver run in the current environment produces the same widths, bit-for-bit,
  as the hardened driver; the only delta vs the Jul-8 committed JSON is
  pre-existing library-level drift, independent of these edits).
- **CHIME fail-closed.** Verified end-to-end on real `freya_chime.npz`: the
  burst is `diagnostic_only` — provenance missing `bandpass_normalization`
  (freya_chime.yaml lacks it; only freya_chime_hi has it), and the off-pulse
  null FAILS in both sub-bands (on-pulse 50.2/67.3 kHz vs off-pulse median
  45.4/61.4 kHz, ratios 1.11/1.10), reproducing the arm-A instrumental
  signature. The harmonic mask removed 68 comb bins per sub-band; the
  harmonic-mask systematic was ~8–10%.

## Config non-uniformity flagged (not silently changed)

The CHIME configs do not carry a uniform mitigation stack: only
`freya_chime_hi.yaml` has `bandpass_normalization`; `casey_chime.yaml` and
`casey_chime_hi.yaml` carry the harmonic mask but neither `grid_regularization`
nor `bandpass_normalization` (they use polynomial baseline subtraction). The
guards make this consequential rather than invisible: any such CHIME product is
demoted to `diagnostic_only` until its config records the full stack. Deciding
whether each product actually needs bandpass normalization / grid
regularization (e.g. already gap-free, not scallop-dominated) is a
per-product science call left to the owner, per the review.

## Tests

Full scint suite: **144 passed, 2 skipped** (was 128/2 — +16 guard unit
tests). Driver integration: **5 passed**. Run with the direct interpreter
(the py312 conda env panics under `conda run`):

```bash
NUMBA_DISABLE_JIT=1 /Users/jakobfaber/.conda/envs/py312/bin/python -m pytest \
  scintillation/scint_analysis/tests/ \
  analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py -q
```
