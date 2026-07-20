# Chromatica (FRB 20240203A) — Propagation Analysis

Self-contained record of the single-sightline analysis of the Chromatica propagation
chain (DM/ToA -> scattering -> scintillation) against the twelve-FRB CHIME<->DSA-110
co-detection pipeline, sorted by observable / technique.

**Provenance:** end-to-end re-analysis run 2026-07-20; fits executed remotely on h17
(SLURM), nothing fit locally. Delivered via dsa110-FLITS PR #209.

## Layout

```
chromatica/
  chromatica_status.md      # running status report + end-to-end verdict (start here)
  stage0_frozen_inputs.json     # frozen canonical baselines
  dm-phase/                     # DM-phase <-> geometric ToA alignment
    chromatica_stage1_dm_toa.{json,png}
    chromatica_stage1b_crop_sensitivity.{json,png}
    catalog_crop_sensitivity.{json,png}      # all-12-burst crop sweep
  scattering/                   # joint CHIME+DSA PBF model selection (h17)
    chromatica_{squarelaw,freealpha,plpbf}_fit.json
    chromatica_stage2_pbf_comparison.json
    chromatica_stage2_ppc.png
  scintillation/
    acf-analysis/               chromatica_stage3_acf_ladder.png
    modulation-index/           chromatica_stage3_modindex.{json,png}
    gamma-nu/                   chromatica_stage3_gamma_nu.png
                                chromatica_stage3_gamma_alpha.json
```

## Verdict

1. **DM/ToA** — the -4.97 ms CHIME-DSA offset is fully explained by the +0.077 pc/cm3
   CHIME-DSA DM difference; no DM outside the measured band. CHIME-primary DM stands
   (CHIME crop-robust; DSA broad/shallow, crop-sensitive).
2. **Scattering** — single thin screen (square-law, alpha=4, beta=4, tau_1GHz=0.126 ms)
   decisively preferred (dlnZ +2.9 vs free-alpha, +11.7 vs power-law PBF; both rail to
   beta=4). Caveat: DSA S/N marginal (qualifies only at S/N target <=8).
3. **Scintillation** — CHIME-band alpha_scint = 3.09 +/- 0.99 (measurement); DSA
   within-band unphysical (diagnostic). tau*Delta_nu = 21.9 >> 2 -> two-screen sightline
   (diffractive screen distinct from and nearer than the pulse-broadening screen); a
   single joint alpha across 0.6-1.4 GHz is physically rejected (reported as the null).

The single largest science-readiness limiter is DSA S/N.
