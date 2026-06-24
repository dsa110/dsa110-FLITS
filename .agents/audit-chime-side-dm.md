# Audit: CHIME-side structure-DM extraction (retraction + rebuild)

**Date:** 2026-06-24
**Trigger:** expert visual review of the DM-phase diagnostics — bursts are not vertical, the
sweep is large, the "DM-phase" curves are flat/oscillatory, and it is universal across all 12.
**Status:** Pillar-2 DMs RETRACTED on main (nulled, `dm_confidence=under-audit`); rebuild pending.
Pillar-4 (positions) UNAFFECTED.

## What was wrong (confirmed by `scripts/diag_dedisp.py` on zach, in docker)

singlebeam: `dt=2.56 µs`, `ntime=55949` (~143 ms; pre-dedispersed at record time, burst at ~77 ms),
871 ch over 400.4–799.2 MHz.

1. **Inter-channel unit bug (1000×).** The extraction rolled channels by
   `1e-3 * K_DM * (1/f² − 1/f_ref²) * dm_c` with `K_DM=4148.81`, f in MHz. The physical full-band
   delay @DM=262 is **5.086 s = 1,986,647 samples**; the recipe rolled by **5.086 ms = 1987 samples**
   — **ratio exactly 1000.0**. The `1e-3` factor is wrong for f in MHz. Damage was partly masked
   because the data is already pre-dedispersed into a 143 ms window, but the band is still mis-aligned.
2. **DM-phase recovered nothing.** Every curve is near-flat/oscillatory (flat_ratio 1.2–1.5); the
   robust-argmax just picks the global max of noise wiggles, and the bootstrap σ is small only because
   the noise pattern is stable across resamples. A real structure-DM peak is sharp (ratio ≫ 1). This
   holds even on zach, where proper dedispersion shows a clear **SNR~15** burst.
3. **Downstream signal-loss.** The RFI-mask → coarse-peak (savgol argmax) → ±512 window → time-flip →
   z-score-display chain throws away a burst that `coherent_dedisp(bb, dm_c)` (proper, `time_shift=True`)
   renders clearly. Diagnostic: proper-dedisp collapse peak SNR~14.8 vs recipe-path SNR~12.1, but the
   extraction figures show noise.
4. **The recovery test was blind to all of this.** `tests/test_dmphase_recovery.py` dispersed *and*
   de-dispersed with the same `1e-3*K_DM` constant, so it recovered the injected DM without ever
   checking absolute physical scaling. It must be rebuilt to disperse with the *physical* constant.

## Rebuild plan (full rebuild on library dedispersion — user-approved scope)

Re-derive on `baseband_analysis` proper dedispersion; do not hand-roll inter-channel alignment.

- **P1 — physical-scale recovery test (test-first).** Rewrite `test_dmphase_recovery.py` to disperse a
  synthetic burst with the PHYSICAL constant (`t[s]=4.148808e3*DM*(f_MHz^-2 − f_ref^-2)`) and assert the
  estimator recovers it. Make the same convention the single source of truth for the estimator.
- **P2 — dedisperse correctly.** Use `coherent_dedisp(bb, dm_c)` (default `time_shift=True`), validated
  against zach (must reproduce the SNR~15 vertical burst). Search a small *physical* residual-DM grid for
  the structure-max; no `1e-3` factor, no manual roll.
- **P3 — make the curve peak.** On a known-signal burst (zach), confirm the DM-phase curve has a sharp
  interior peak (flat_ratio ≫ the ~1.3 noise floor) before trusting any other burst. Gate every burst on
  curve sharpness, not just "interior".
- **P4 — re-extract all 12 + figure-review from scratch + re-validate**; only then un-null the DMs that
  genuinely peak, and re-assess which co-detections yield a real CHIME structure-DM.

## Provenance
Prior (retracted) numbers live in git (PR #29, commit cc64b7b) and off-repo at
`/data/.../results/chime_side_inputs.json`. Diagnostic: `scripts/diag_dedisp.py`,
`diagnostics/diag_dedisp_zach.png`.
