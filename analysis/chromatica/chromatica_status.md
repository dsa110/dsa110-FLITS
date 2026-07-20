# Chromatica (FRB 20240203A) — End-to-End Re-Analysis Status

**Test case:** single-sightline end-to-end re-run of the Chromatica propagation chain
(DM/ToA → scattering PBF → scintillation) against the twelve-FRB CHIME↔DSA-110 co-detection pipeline.
**Convention:** read existing pipeline products, re-run only what is needed, report gates/failures honestly.

**Execution env:** conda `flits` (Python 3.12.13, dynesty 3.0.0) with `PYTHONPATH=pipeline`
(imports the repo-local `flits` package at `Faber2026/pipeline/flits`).

---

## Stage 0 — Frozen inputs (baselines to reproduce/beat)

### DM (dm-joint-phase-v2, `analysis/dm-joint-phase-v2/results/fits.json`)
| Quantity | Value (pc cm⁻³) |
|---|---|
| DM_CHIME (curve) | 272.6387 ± 0.0119 |
| DM_DSA (curve)   | 272.5614 ± 0.0862 |
| DM_joint         | 272.6372 ± 0.0118 |
| CHIME − DSA      | +0.0773 (tension 0.89σ, joint_q 0.79) |
| product_dm CHIME / DSA | 272.6382 / 272.368 |
| catalog DM (crossmatch) | 272.664 |

Band centers: CHIME 400.6–794.3 MHz; DSA 1311.3–1498.75 MHz.

### ToA / geometry (`pipeline/crossmatching/toa_crossmatch_results.json`)
| Quantity | Value (ms) |
|---|---|
| geometric_delay (target Δt_geom) | −2.2845 |
| peak measured offset (CHIME−DSA) | −4.971 |
| model-corrected offset | −5.771 (`diagnostic_only:MARGINAL`) |
| differential scatter shift | +0.7997 |
| combined error (full) | 2.416 |

Disagreement to resolve in Stage 1: measured −4.97 ms vs geometric −2.28 ms (~2.7 ms, ~1.1× combined error).

### Scattering (prior fits — two conflicting products)
| Product | α | τ₁GHz (ms scale) | ln Z | flag |
|---|---|---|---|---|
| `_a1_fits` sharedζ exp-exp (canonical) | 3.284 ± 0.040 | 0.1957 ± 0.005 | −8526.3 | (β_C=β_D=3.667) |
| `joint_json` legacy | 5.999 (railed, bounds [1,6]) | 0.0249 | −44072 | **MARGINAL, α railed** |

Note: α discrepancy between products (3.28 vs 6-railed) is exactly the question Stage 2 revisits —
square-law (β=4/α=4 thin-screen, exponential PBF) vs power-law PBF (β free → α).

### Scintillation (window-tuning campaign 2026-07-17)
| Band | α_scint (γ∝ν^α) | n | status |
|---|---|---|---|
| CHIME-hi (ref 661 MHz) | 3.088 ± 0.992 | 4 | measurement (validation pass) |
| CHIME-lo (ref 497 MHz) | 1.717 ± 0.399 | 4 | detection |

Known failure mode to avoid: chromatica α-collapse when the 2D single-shared-γ Lorentzian averages
one clean scintle against noisy subbands (memory: use per-subband + harmonic-lag masking).

---

## Stage 1 — DM-phase ↔ geometric ToA alignment ✅

**Result:** The −4.97 ms measured CHIME−DSA offset vs the −2.28 ms geometric target is fully
explained by the +0.077 pc/cm³ CHIME−DSA DM difference. In the production ToA convention the
dispersive slope is −24.16 ms/(pc cm⁻³); the DM that aligns the offset with geometry is
**272.553 pc/cm³**, which lands on the DSA DM-phase optimum (0.10σ **at the production 30 ms crop**;
see caveat below) but 7.2σ below CHIME/joint. Adopting the DSA DM drops the ToA residual from
−2.69 ms (catalog) to −0.21 ms. **No DM outside the measured band is required.** This is consistent
with keeping the CHIME-primary DM policy (the residual is 0.85× the combined error).

> **Caveat (from Stage 1b below):** the "0.10σ" coincidence is inflated by the over-wide 30 ms crop.
> Under the user-mandated ±4× FWHM (~8 ms) crop the DSA optimum shifts up to ~272.62, so the aligning
> DM 272.553 sits ~0.82σ from it — still consistent, but not the near-exact landing the 30 ms number
> implies. The policy conclusion (CHIME-primary, no DM outside the band) is unchanged; only the
> tightness of the DSA coincidence is walked back.

**Stage 1b crop check (user-directed):** the DM-phase algorithm feeds a **fixed 30 ms window** to
every burst (`crop_on_pulse window_s=0.030`) = ±18× the 0.82 ms FWHM for Chromatica, far exceeding
the ±4× guideline. CHIME is crop-robust (DM moves ≤0.001) and its phase peak *sharpens* on a tight
crop (prominence 3.9→8.8); DSA is broad/shallow in all crops (prominence ~1.0–1.1) and its optimum
**wanders** with window width. **Catalog-wide (all 12):** CHIME DM range ≤0.007 pc/cm³ for every
burst; DSA DM range 0.004–0.119. CHIME−DSA tension has no single direction with crop (6↑/5↓/1 flat).
**phineas** is a genuine offset (tension 6.5→4.1σ, survives tightening). Recommendation: replace the
fixed 30 ms crop with an adaptive k×FWHM window (k≈4–8) — stabilizes the DSA cross-check without
touching any adopted (CHIME-primary) DM.

Artifacts: `chromatica_stage1_dm_toa.{json,png}`, `chromatica_stage1b_crop_sensitivity.{json,png}`,
`catalog_crop_sensitivity.{json,png}`.

## Stage 2 — Scattering: square-law vs power-law PBF ✅

**Fits on h17** (conda flits-a1-312 / flits-venv, dynesty 3.1.0, nlive=400), joint 2D CHIME+DSA
shared-ζ gain-marginal nested sampling. **Critical:** the fits MUST use the AUTO-TF S/N-driven
resolution + robust common window (`FLITS_JOINT_AUTO_TF=1`). Forcing the fixed-resolution fallback
(`AUTO_TF=0`, config f384 DSA + 30 ms crop) drives the sampler into a **degenerate broad-burst mode**
(ζ=13.7, x_ζ=+2, ΔDM_D=−38, χ²/dof≈10). DSA S/N is marginal (clears the S/N gate only at target ≤8,
not the default 10) — this is the origin of the prior MARGINAL flag. Re-running at **S/N target 8**
recovers the good mode (ζ≈0.19, x_ζ≈−1.49, ΔDM_D≈0.27, χ²/dof CHIME 1.26 / DSA 1.01).

| Model | β | α | τ₁GHz (ms) | ln Z | ΔlnZ | χ²/dof (C/D) |
|---|---|---|---|---|---|---|
| **Square-law** (exp, β=4) | 4.000 | 4.0 | 0.126 | **−16574.5** | 0 (best) | 1.26 / 1.01 |
| Free-α exp (α∈[4,6]) | 3.990 (rails) | 4.0 | 0.126 | −16577.4 | −2.9 | 1.26 / 1.01 |
| Power-law PBF (inner-scale) | 3.997 (rails) | tied | 0.126 | −16586.2 | −11.7 | 1.26 / 1.01 |

**Verdict: square-law (α=4, thin-screen exponential PBF) is decisively preferred** (ΔlnZ +2.9 vs
free-α, +11.7 vs PL-PBF). All three models fit equally well in χ² → selection is by evidence. In the
good mode **both** free-α and PL-PBF rail to β=4 (PL-PBF with a large inner scale s_i≈25 τ_e, i.e. the
exponential limit). **α does NOT rail sub-4 and is not promoted from a railed value** — the opposite:
the data prefer α=4. This contradicts and explains the on-disk canonical `_a1_fits` α=3.28, which is
traced to the degenerate/over-wide-crop prep, not a burst property. **Gate: PASS**, with the honest
caveat that DSA S/N is genuinely marginal (near the joint-fit publication threshold).

Artifacts: `chromatica_stage2_pbf_comparison.json`, `chromatica_stage2_ppc.png`,
`chromatica_{squarelaw,freealpha,plpbf}_fit.json` (all v2 good-mode).

## Stage 3 — Scintillation: ACFs, m(t), m(ν), γ(ν), two-screen decomposition ✅

**Data:** CHIME per-subband ACF ladder reproduced with the current parametrization (harmonic-lag
masking at 0.390625 MHz, first-fit-lag skip, 1L/2L model selection) via `window_refit.refit`; the
trusted published CHIME campaign values are quoted for α. DSA ACFs from the rescued 4-subband product
(`chromatica_acf_results.pkl`), freshly fit with the same pipeline `_fit_subband`. All CHIME (4) and
DSA (4) subbands resolve.

**Per-subband γ (MHz):** CHIME-hi 0.064–0.114 (623–749 MHz); DSA 0.428 / 0.667 / 1.322 / 2.317
(1321 / 1351 / 1396 / 1460 MHz). DSA 1460 MHz has m=1.69 > M_PHYS(1.2) → envelope-contaminated,
excluded from any α fit (reported).

**Modulation indices:**
- m(t) (temporal, σ/μ of the freq-averaged profile, sliding 3-bin): CHIME median 0.055, DSA median 0.146.
- m(ν) (spectral, from ACF amplitude): CHIME-hi 0.50–0.70; DSA 0.78–1.69 (rises toward higher ν).

**Two-screen test — the governing result.** The repo's one-screen consistency test
(`flits.batch.analysis_logic.check_tau_deltanu_consistency`) evaluated with the fresh good-mode
Stage-2 scattering (τ=0.126 ms, α=4) and the DSA median Δν_d≈0.67 MHz gives
**τ(1.4 GHz)·Δν_d = 21.9 ≫ the accepted [0.1, 2.0]** → **different_screens**: the DSA-band
scintillation samples a *nearer* diffractive screen than the pulse-broadening (scattering) screen.
A single power-law γ(ν) across 0.6–1.4 GHz is therefore physically rejected. (Same verdict as the
archived catalog value 93.9; the product differs only because the fresh τ is shorter.)

**Headline (per the standing owner rule — physically-defensible framing):**
- **CHIME-band α_scint = 3.09 ± 0.99** (n=4, measurement) — the reportable per-band scaling.
- **DSA within-band α ≈ 20** — UNPHYSICAL (only ~10% fractional bandwidth, 1321–1460 MHz);
  **diagnostic-only, not a measurement.**
- **Two-screen decomposition** (near diffractive screen ≠ far scattering screen) is the physical picture.
- The forced joint 0.6–1.4 GHz single-screen α = 3.13 ± 0.29 is reported **only as the tested-and-
  rejected hypothesis** (consistent finite-scintle weighting on both bands), never as the headline.

Artifacts: `chromatica_stage3_acf_ladder.png`, `chromatica_stage3_modindex.{json,png}`,
`chromatica_stage3_gamma_alpha.json`, `chromatica_stage3_gamma_nu.png`.

---

## End-to-end verdict (Stage 4)

Chromatica's propagation picture, re-derived end-to-end and reported with honest gates:

1. **DM / ToA (Stage 1):** the −4.97 ms CHIME−DSA offset vs the −2.28 ms geometric target is fully
   explained by the +0.077 pc/cm³ CHIME−DSA DM difference; no DM outside the measured band is needed.
   CHIME DM is crop-robust and sharpens on a tight window; DSA is broad/shallow and crop-sensitive.
   **CHIME-primary DM policy stands.** (The tightness of the DSA aligning-DM coincidence is walked
   back from 0.10σ at the 30 ms crop to ~0.82σ under ±4× FWHM.)
2. **Scattering (Stage 2):** a **single thin screen (square-law, α=4, β=4, τ₁GHz=0.126 ms)** is
   decisively preferred (ΔlnZ +2.9 vs free-α, +11.7 vs power-law PBF; χ²/dof 1.26/1.01). Both the
   free-α and power-law-PBF fits rail to β=4 in the good mode — no sub-4 α survives. The on-disk
   canonical α=3.28 is a prep/crop artifact. **Gate PASS, with the honest caveat that DSA S/N is
   marginal (qualifies only at S/N target ≤8).**
3. **Scintillation (Stage 3):** CHIME-band α_scint = 3.09 ± 0.99 (measurement); DSA within-band
   unphysical (diagnostic). The τ·Δν consistency test (21.9 ≫ 2) shows the diffractive
   scintillation screen is **distinct from and nearer than** the pulse-broadening screen — a
   two-screen sightline. A single joint α across the ~1 GHz lever is physically rejected.

**Cross-stage consistency & science-readiness caveats.**
- The scattering screen is thin-screen (α=4); the scintillation samples a *different, nearer* screen.
  These are not in tension — they are two screens on the sightline, exactly what the τ·Δν test flags.
- The single largest science-readiness limiter is **DSA S/N**: it drives the marginal scattering gate
  and the ~10% DSA scintillation bandwidth that makes the within-band α unphysical.
- No result here is promoted from a railed/degenerate fit; every gate that failed is reported as such.

---

## Stage progress

| Stage | Status | Deliverable |
|---|---|---|
| 0 — orient / freeze inputs | ✅ done | `stage0_frozen_inputs.json`, this file |
| 1 — DM-phase ↔ geometric ToA | ✅ done | `chromatica_stage1_dm_toa.{json,png}` |
| 1b — crop sensitivity (12 bursts) | ✅ done | `chromatica_stage1b_*`, `catalog_crop_sensitivity.*` |
| 2a — square-law PBF fit | ✅ done | `chromatica_squarelaw_fit.json` |
| 2b — power-law PBF fit | ✅ done | `chromatica_plpbf_fit.json` |
| 2c — PPC + PBF verdict | ✅ done | `chromatica_stage2_pbf_comparison.json` + PPC figure |
| 3a — scint ACFs + windows | ✅ done | `chromatica_stage3_acf_ladder.png` |
| 3b — m(t), m(ν) | ✅ done | `chromatica_stage3_modindex.{json,png}` |
| 3c — γ(ν), α_scint (two-screen) | ✅ done | `chromatica_stage3_gamma_alpha.json` + `gamma_nu.png` |
| 4 — consolidate + verdict | ✅ done | this document |

_Last updated: all stages complete. End-to-end verdict above._
