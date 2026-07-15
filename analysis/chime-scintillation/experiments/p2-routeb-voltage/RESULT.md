# P2 Route B — voltage-derived common-mode-immune Δν_d statistic (`p2-routeb-voltage`)

**Status: `DOCUMENTED-FAIL`. Blinded gate G1 (injection recovery) fails for
both S1 and S2; no on-pulse statistic was computed. This is not a CHIME
scintillation-bandwidth measurement.** The gate G2 (off-pulse null) passes —
the ratio construction cancels the instrumental common mode — so the failure
is an S/N ceiling, not the common-mode confounder that defeated every prior
route.

P2 Route B is the owner-sanctioned successor to the P1 refutation
([`experiment-chime-scint-routeb-voltage.md`](../../../../docs/rse/specs/experiment-chime-scint-routeb-voltage.md),
Faber2026; Gate 0 GO with an admissibility window;
[`research-chime-scint-successor-routes.md`](../../../../docs/rse/specs/research-chime-scint-successor-routes.md)).
The statistic forms the on/off ratio `R_p(ν) = ⟨I_p⟩_on/⟨I_p⟩_off − 1` first,
so the common instrumental response `g(ν)` divides out **algebraically**
(⟨I_p⟩_on and ⟨I_p⟩_off both carry the same `g(ν)`), then takes the pol0×pol1
cross-ACF of `R_p` and fits a Lorentzian. Frozen config sha256
`cb7b21b7…`; 23 064 high-band fine channels (627–800 MHz), 19 465 good after
the RFI/LTE mask.

## What the two gates found

**G2 — off-pulse null: PASS (the key positive result).** 24 off-pulse-only
pseudo-on null realizations at the operating point (100-sample pseudo-on ⟂
reference, no injection), Šidák two-sided FWER 5% (threshold z = 3.071):

| statistic | max \|z\| | detections | verdict |
|---|---:|---:|---|
| S1 | 2.59 | 0 / 24 | PASS |
| S2 | 2.00 | 0 / 24 | PASS |

The off-pulse ratio cross-ACF is consistent with zero — **the ratio cancels
the ~35 kHz common mode that every prior route's cross-ACF retained**
(B3/B4/C1 nulls failed at max \|z\| = 4.8 because `g` survived their intensity
cross-ACF). Route B removes that confounder as designed.

**G1 — injection recovery: FAIL (both statistics).** Synthetic scintillation
injected multiplicatively into off-pulse frames relabeled pseudo-on, at the
real burst contrast `f_b = 0.05`; grid `m ∈ {0.15, 0.17}` ×
`Δν_d ∈ {35, 77, 127, 213, 352} kHz` × 50 realizations
(seed = 1000·cell + realization). A cell **certifies** when
median\|Δν_d bias\| ≤ 30%, \|median amplitude pull\| ≤ 2, and convergence ≥ 90%.
G1 PASS needs every `Δν_d ≥ 127 kHz` cell to certify and the 35 kHz control to
**not** certify.

| statistic | detectable (≥127 kHz) certify | 35 kHz control | G1 |
|---|---|---|---|
| S1 | 0 / 6 | does not certify | **FAIL** |
| S2 | 0 / 6 | m=0.15 spuriously certifies | **FAIL** |

Recovered widths pin at **~20–45 kHz for every injected width from 35 to
352 kHz** (see `figures/injection_recovery_scatter.png`): the fit tracks the
noise floor, not the injection. Representative S1 cells (median recovered Δν_d,
fractional bias):

| Δν_d inj | m=0.15 | m=0.17 |
|---:|---|---|
| 35 kHz (control) | 20.7 kHz (−0.41) | 17.5 kHz (−0.50) |
| 127 kHz | 33.0 kHz (−0.74) | 39.1 kHz (−0.69) |
| 213 kHz | 30.4 kHz (−0.86) | 34.3 kHz (−0.84) |
| 352 kHz | 27.4 kHz (−0.92) | 21.4 kHz (−0.94) |

## Why it fails — S/N ceiling, not a broken estimator

At `f_b = 0.05` the scintillation signature on the ratio is
`f_b·m ≈ 0.0075–0.0085` (cross-ACF amplitude `(f_b·m)² ≈ 5–7 × 10⁻⁵`), which
sits at the per-lag noise floor (`~9 × 10⁻⁵`). The estimator is sound: the same
S1 statistic recovers 213 kHz within ±17% once the **effective** modulation
`s_eff = f_b·m` reaches ~0.05 — about 6× the real 0.0085 (see
`figures/estimator_sanity_snr.png`). The recovery collapses only at the real
signal strength.

This confirms the research record's prediction verbatim — *"Even a perfect
instrumental fix leaves a marginal S/N problem"*
([`research-chime-scint-instrumental-common-mode.md`](../../../../docs/rse/specs/research-chime-scint-instrumental-common-mode.md)).
Route B **provides** the perfect instrumental fix (G2 clean) and then meets the
S/N wall directly. It cleanly separates the two failure modes the campaign had
conflated: common-mode confounder (now solved) vs radiometer-limited S/N (not).

**Gate 0 relation.** Gate 0 was GO — the *optimal quadratic* (Fisher) estimator
reaches SNR ≈ 5 at 213 kHz (m=0.17). The Route-B **lag-space cross-ACF +
Lorentzian fit** falls ~1 order of magnitude short of that ceiling. A future
Route-B attempt would need a matched/optimal estimator (not the lag-space
cross-ACF) or a higher-S/N input.

## S1, S2, S3

- **S1** on/off ratio cross-ACF, **S2** time-split ratio cross-ACF: both
  implemented, run, G2-clean, G1-failed. Selection verdict: no statistic
  qualifies → `DOCUMENTED-FAIL` (`selection.json`).
- **S3** voltage variant: implemented and unit-tested (equals S1 on `|V_p|²`),
  **not run**. h17 is reachable but stages only detected per-pol power; the
  complex fine-channel voltages S3 needs (for the P1 worker's grouped-bin noise
  normalization) are not present and regenerating them from baseband is out of
  P2 scope. S1 already operates on the detected per-pol `|V_p|²`, so S3 adds
  only the noise-normalization refinement.

## Blinding

No statistic was computed on the on-pulse window (samples 250–350). The
`routeb_voltage` statistics refuse any window overlapping [250, 350) unless
`allow_unblind=True`; the harness `unblind` subcommand refuses without both a
passing selection and the explicit `--unblind-i-know-what-i-am-doing` flag.
Because G1 fails, there is no selected statistic and no authorized unblinding —
the one-shot on-pulse computation is not performed.

## Deliverables & reproducibility

- Statistics + blinding guard: `scintillation/scint_analysis/routeb_voltage.py`
- Harness: `routeb_calibration.py` (`freeze` → `g1 --statistic {S1,S2}` →
  `g2 --statistic {S1,S2}` → `select`; `unblind` is orchestrator-only).
- Figures: `plot_routeb.py` → `figures/` (injection-recovery scatter, pull
  distributions, null histogram, example off-pulse ratio + cross-ACF, estimator
  sanity vs S/N).
- Tests: `scintillation/scint_analysis/tests/test_routeb_voltage.py`
  (bandpass cancellation to machine precision, blinding guard, HWHM convention).
- Records: `frozen_config.json` (hashed), `g1_S1/S2.json`, `g2_S1/S2.json`,
  `selection.json`, `validation.json`.
- Env: conda `py312` (numpy 2.4, scipy 1.17, matplotlib 3.10). Data:
  `~/Data/Faber2026/dsa110/upchan_codetections/crossacf-2026-07-14`
  (SHA-256s in `frozen_config.json.inputs_sha256`; verified against the pipeline
  `DATA_MANIFEST.yaml`).
- Reproduce: `python routeb_calibration.py freeze` then `g1`/`g2` per statistic,
  `select`, `python plot_routeb.py`.
