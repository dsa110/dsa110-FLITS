# Chromatica (FRB 20240203A) — End-to-End Re-Analysis

Self-contained reproducibility record for the single-sightline end-to-end re-run of the
Chromatica propagation chain (DM/ToA -> scattering PBF -> scintillation) against the
twelve-FRB CHIME<->DSA-110 co-detection pipeline. Generated 2026-07-20.

**Convention:** read existing pipeline products, re-run only what is needed, report
gates/failures honestly. Fits were run remotely (h17 SLURM); nothing was fit locally.

## Contents

`chromatica_e2e_status.md` — running status report + end-to-end verdict (start here).
`stage0_frozen_inputs.json` — frozen canonical baselines (DM, ToA, scattering, scint).

### Stage 1 — DM-phase <-> geometric ToA alignment
- `chromatica_stage1_dm_toa.{json,png}` — DM-phase optima vs geometric ToA target.
- `chromatica_stage1b_crop_sensitivity.{json,png}` — chromatica DM-phase crop-window
  sensitivity (+/-4x FWHM guideline check).
- `catalog_crop_sensitivity.{json,png}` — all-12-burst crop sweep (CHIME robust; DSA
  window-dependent; tension has no directional trend).

### Stage 2 — Scattering, square-law vs power-law PBF (joint CHIME+DSA)
- `chromatica_squarelaw_fit.json`, `chromatica_freealpha_fit.json`,
  `chromatica_plpbf_fit.json` — h17 nested-sampling joint fits (AUTO-TF S/N target 8,
  good mode).
- `chromatica_stage2_pbf_comparison.json` — model-selection verdict.
- `chromatica_stage2_ppc.png` — posterior predictive check + overlay.

### Stage 3 — Scintillation
- `chromatica_stage3_acf_ladder.png` — per-subband ACF ladder (CHIME-hi + DSA).
- `chromatica_stage3_modindex.{json,png}` — m(t) and m(nu) triptych.
- `chromatica_stage3_gamma_alpha.json`, `chromatica_stage3_gamma_nu.png` — gamma(nu)
  and the two-screen decomposition.

## End-to-end verdict (summary)

1. **DM/ToA:** the -4.97 ms CHIME-DSA offset is fully explained by the +0.077 pc/cm^3
   CHIME-DSA DM difference; no DM outside the measured band is needed. CHIME-primary
   DM policy stands. CHIME DM is crop-robust; DSA is broad/shallow and crop-sensitive.
2. **Scattering:** a single thin screen (square-law, alpha=4, beta=4, tau_1GHz=0.126 ms)
   is decisively preferred (dlnZ +2.9 vs free-alpha, +11.7 vs power-law PBF). Both
   free-alpha and PL-PBF rail to beta=4. Honest caveat: DSA S/N is marginal (qualifies
   only at S/N target <=8).
3. **Scintillation:** CHIME-band alpha_scint = 3.09 +/- 0.99 (measurement); DSA
   within-band alpha unphysical (diagnostic only). The tau*Delta_nu consistency test
   (21.9 >> 2) shows the diffractive scintillation screen is distinct from and nearer
   than the pulse-broadening screen -> a two-screen sightline. A single joint alpha
   across 0.6-1.4 GHz is physically rejected (reported only as the tested null).

The single largest science-readiness limiter is DSA S/N.
