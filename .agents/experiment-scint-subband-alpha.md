# Scintillation Δν_d per subband across 400–1500 MHz, and the Δν_d ∝ ν^x_scint scaling

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Branch:** `feat/powerlaw-pbf` (worktree off origin/main)
**Status:** Complete — DSA Δν_d measured (flat); CHIME unresolved; two-screen geometry inferred;
cross-codetection census done (the ~8× MW-floor excess reproduces wherever cleanly resolved)
**Related Documents:**
- [Power-law PBF + per-band PBF](experiment-powerlaw-pbf.md) — the scattering side of the same sightline

---

## Goal

Measure the scintillation decorrelation bandwidth Δν_d in frequency subbands across the
full CHIME (400–800 MHz) + DSA (1200–1500 MHz) co-detection lever arm, and fit the scaling
Δν_d ∝ ν^x_scint. For a single diffractive screen the Kolmogorov expectation is x_scint ≈
+4.0…+4.4, which would independently corroborate the pulse-broadening scattering index α
over the ~1 GHz lever arm. The thin-screen relation 2π·τ·Δν_d = C1 (≈1) links the two.

## Method (and its validation)

Scintillation needs **native** frequency resolution; the scattering joint-fit prep decimates
freq by f_factor 64 (CHIME) / 384 (DSA), which destroys the scintillation structure. Here
each band's raw `.npy` is loaded at native resolution via the scattering `BurstDataset`
(f_factor=1): CHIME 1024 ch (0.39 MHz/ch), DSA 6144 ch (0.031 MHz/ch). Per subband: on-pulse
spectrum (off-pulse baseline subtracted), frequency ACF via the scint pipeline's
`calculate_acf`, then a **two-Lorentzian** fit (narrow diffractive + broad component +
const) to lag>0 (the zero-lag noise spike excluded). Δν_d = narrow HWHM. A subband counts as
resolved only if Δν_d > 3 channels, > 5 scintles span the subband, the error is finite and
< Δν_d/2, the fit is not bound-railed, and the narrow scale is clearly separated from the
broad (Δν_d < broad/2).

**Validation (cross-check, sha256=b0a678dd):** the narrow Δν_d reproduces the scint
pipeline's own stored Lorentzian-component fits (`scintillation/configs/bursts/wilhelm_dsa.yaml`):
my 0.136 / 0.136 / 0.133 MHz at 1335 / 1428 / 1475 MHz vs the stored l_1_gamma ≈ 0.13 MHz.
(The 1381 MHz subband is an ambiguous outlier in both — multiple competing models in the
stored config too.)

## Results

### DSA (1311–1499 MHz, native 0.031 MHz/ch) — resolved

| ν (MHz) | Δν_d (MHz) | resolved | note |
|---|---|---|---|
| 1335 | 0.136 ± 0.031 | yes | 4.4 ch |
| 1382 | 0.55 ± 0.30 | no | ambiguous narrow/broad split (excluded) |
| 1428 | 0.136 ± 0.068 | yes | 4.5 ch |
| 1475 | 0.133 ± 0.058 | yes | 4.4 ch |

**Δν_d ≈ 0.13 MHz, essentially FLAT across DSA: x_scint = −0.23 ± 0.19** over a 10 % lever
arm. This is inconsistent with the steep Kolmogorov diffractive scaling (+4.0…+4.4) and
with any positive index — though the scale is only marginally resolved (4–5 channels) and
the 10 % lever arm is short, so the constraint is weak and possibly resolution-floor-limited.

### CHIME (400–800 MHz, native 0.39 MHz/ch) — unresolved

All CHIME subbands give sub-channel or broad-component fits (0.5–5 ch, large/degenerate
errors). The DSA-anchored diffractive scale (0.13 MHz at 1428) Kolmogorov-scaled to 600 MHz
is **≈ 3 kHz** — ~130× below CHIME's 0.39 MHz channel. **CHIME diffractive scintillation is
unresolvable at native channelization;** only upper limits (≈ 1 channel) are reported.
→ The full 400–1500 MHz x_scint for the diffractive screen is **not measurable** with
current CHIME resolution. Measuring it would require finer CHIME channelization
(upchannelized baseband).

### Two-screen geometry (the main physical result)

Same-screen test, 2π·τ_scatt(ν)·Δν_d with τ from the scattering fit (τ_1GHz = 0.255 ms,
α = 2.62):

| ν (MHz) | τ_scatt | C1 |
|---|---|---|
| 1335 | 120 µs | 102 |
| 1428 | 100 µs | 85 |
| 1475 | 92 µs | 77 |

**median C1 ≈ 85 ≫ 1.** The strong pulse-broadening screen would imply Δν_d ≈ C1/(2πτ) ≈
**1.5 kHz** (unresolved everywhere). The resolved Δν_d ≈ 0.13 MHz is ~85× larger ⇒ it comes
from a **separate, weaker (foreground / Milky Way) screen, not the scattering screen.**

### NE2025 Milky-Way-floor comparison (done)

`query_ne2025_scint.galactic_floor` for wilhelm (l=107.13°, b=16.69°) predicts the smooth
Galactic scattering floor:

| band | NE2025 MW-floor Δν_d | NE2025 MW-floor τ |
|---|---|---|
| CHIME (600 MHz) | 25.9 kHz | 7.1×10⁻³ ms |
| DSA (1405 MHz) | **1095 kHz (1.1 MHz)** | 1.7×10⁻⁴ ms |

Measured DSA Δν_d (136 kHz) is **0.12× the MW floor** — i.e. ~8× *more* scattering than the
smooth NE2025 Milky Way predicts for this (mid-latitude) sightline. So **three distinct
scattering scales** on the wilhelm sightline:

| screen | τ (DSA) | Δν_d (DSA) |
|---|---|---|
| pulse-broadening (joint fit) | ~100 µs | ~1.5 kHz (unresolved) |
| resolved scintillation | ~1.2 µs | 136 kHz (measured) |
| NE2025 MW floor | ~0.17 µs | 1095 kHz (predicted) |

## Cross-codetection census — is the wilhelm 8× excess sightline-specific or systematic?

Extended the resolved-Δν_d-vs-MW-floor comparison to all 12 co-detections. **Δν_d source
= the scint pipeline's per-subband Lorentzian fits** in
`scintillation/configs/bursts/{burst}_dsa.yaml` (`analysis.stored_fits`), not a fresh
raw-`.npy` re-derivation — a naive single-Lorentzian ACF fit returned the **broad**
component for ~half the bursts (verified: it gave chromatica 0.81 MHz where the
pipeline's narrow scale is ~0.22 MHz). The diffractive Δν_d per burst is the **narrowest
consistent cluster** (within a factor 3) of well-constrained Lorentzian/Gen-Lorentz
widths across **all** stored fits (`nladder/scint_candidates.py` →
`scint_mw_final.py`), scanning beyond the BIC-selected model and rejecting the
fitter's **0.060-MHz rails** (widths pinned at the lower bound, err ≫ value). MW floor
from `query_ne2025_scint.galactic_floor` at the catalog coordinate. Recovery of the 7
initially-unresolved sightlines was cross-checked by **per-burst adversarial judge
agents** (`scint_recover_verdicts.json`); the deterministic selector and the judges
agree on every sightline's excess/no-excess classification — they differ only on whether
to report a weak lower limit vs a flat non-detection for one sightline (phineas: selector
≥1.3×, judge "non-detection"), which does not move the headline (phineas is excluded
from the excess set either way).

### NE2025-floor uncertainty → per-sightline significance

NE2025 is accurate only to ~a factor of 2–3, so the excess carries a log-normal floor
uncertainty (σ ≈ 0.4 dex ≈ ×2.5) combined in quadrature with the measurement error:
σ_log10(excess), and **z = log10(excess)/σ** = sigma above the MW floor (excess=1).

| sightline | \|b\| | Δν_d (kHz) | floor (kHz) | excess | z | note |
|---|---|---|---|---|---|---|
| zach | 18.4 | 143 | 1520 | **10.7×** | **2.4** | clean 2-comp |
| wilhelm | 16.7 | 136 | 1095 | **8.0×** | **2.2** | clean, 4 subbands |
| hamilton | 18.7 | 225 | 1629 | **7.2×** | **2.1** | clean L+L (broad 15 MHz) |
| casey | 44.6 | 908 | 8717 | 9.6× | 2.2 | low conf (1 subband, rc 0.38) |
| chromatica | 18.4 | 252 | 1492 | 5.9× | 1.9 | clean 2-comp |
| oran | 16.5 | 383 | 1009 | 2.6× | 1.0 | clean |
| freya | 22.2 | ≤1007 | 1642 | ≥1.6× | 0.5 | broad is a flat baseline → single-Lorentzian, lower limit |
| phineas | 44.7 | 6829 | 8704 | ≥1.3× | 0.3 | only broad (3.8–9.9 MHz) |
| johndoeii | 13.2 | 534 | 245 | 0.5× | −0.8 | low-\|b\|, NE2025 void → floor unreliable |
| isha | 15.8 | 595 | 229 | 0.4× | −0.9 | low-\|b\| void |
| whitney | 34.8 | — | 5906 | ≥0.3× | — | rails only (diffractive unresolved) |
| mahi | 10.0 | — | 152 | ≥0.1× | — | rail-only, b=+10° void |

**Verdict: a COMMON (likely host/intervening) excess — not wilhelm-specific.** The three
clean, high-confidence mid-\|b\| sightlines (zach, wilhelm, hamilton; \|b\|≈17–18°) where
the NE2025 floor is reliable and the diffractive scale cleanly resolved all show **7–11×
excess at z>2**, with chromatica (5.9×, z=1.9) just under; casey (\|b\|=44.6°) adds a
fourth z>2 point at 9.6× but is **low-confidence** (single subband). The excess clusters
tightly at ~7–11× — exactly the wilhelm value — and **never the reverse** where the floor
is trustworthy: no reliable sightline has the measurement *above* the floor. The
sub-unity cases (johndoeii, isha, mahi) all sit at low \|b\| in NE2025 **voids** where the
floor is untrustworthy; the unresolved cases (whitney, mahi) yield only 0.060-MHz rails.

**On combining the significances.** Per-sightline z is ~2σ, and that error budget is
**~94% the NE2025 floor uncertainty (σ=0.4 dex), only ~6% the measurement.** The floor
error is a *common systematic* (one model, one calibration) — correlated across
sightlines, **not independent** — so stacking does **not** buy a √N improvement: if
NE2025 under-predicts the mid-\|b\| floor by ~×2.5, all four points move together and the
joint significance stays ~2σ. The real argument is therefore not a stacked z but the
**asymmetry**: wherever the floor is reliable and the scale resolved, the measurement is
*always* in excess by ~7–11× and *never* below — which a random ±0.4 dex floor scatter
would not produce. Confirming it as a calibrated systematic offset vs a true population
excess needs an independent floor (e.g. pulsar/H I calibration) or finer channelization.

**Caveats.** (1) Significance is NE2025-limited (~2σ), and the floor error is shared, so
do not over-read a stacked significance. (2) casey rests on a single subband (redchi
0.38, possible over-fit) — the judge's defensible alternative is non-detection; treated
as low-confidence and excluded from the robust count of 3. (3) freya's "broad" component
(259 MHz) is a flat baseline wider than the DSA band, so its narrow is effectively
single-Lorentzian → tier B, excess a lower limit (the selector now caps "broad" at the
band width, `DSA_BAND_MHZ`). (4) The recovery scans **all** stored fits, not just
BIC-selected; this is deliberate (BIC picks the best *descriptive* model, which can miss
a real narrow diffractive component) and is why hamilton/chromatica were recovered as
clean, but it is paired with rail/over-fit rejection to avoid spurious narrows.

### Second electron-density model — does the excess survive? (YMW16 / NE2001)

`scint_mw_models.py` recomputes the floor under two more Galactic models (YMW16, NE2001
via pygedm; with a scipy `simps`→`simpson` shim), putting all three on a common footing —
each model's scattering time τ@1 GHz is scaled to the DSA band and converted to a floor
Δν_d via 2π·τ·Δν_d = C1 (C1 = 1.16). Only the electron model differs across columns, so
the τ-derived NE2025 floor reproduces the published SBW-based excess exactly (ratio 1.00,
asserted in-script), and model-to-model excess ratios are C1-/2π-/measurement-independent.

| model | mid-latitude sightlines with excess > 2× (excess value) |
|---|---|
| NE2025 | 6/6 — zach 10.7, casey 9.6, wilhelm 8.0, hamilton 7.2, chromatica 5.9, oran 2.6 |
| NE2001 | 6/6 — zach 7.3, casey 6.0, wilhelm 6.2, hamilton 4.8, chromatica 4.2, oran 2.2 |
| YMW16  | 1/6 — only casey 4.8 |

**NE2025 is the authority here, and the disagreement breaks along a trustworthiness axis,
not at random.** NE2025 and NE2001 both *forward-model* scattering from the
electron-density-fluctuation (C_n²) field; NE2025 is the newest and was refit to 568
pulsar/AGN/maser scattering measurements (scattering rms 0.65 dex vs NE2001's 0.98), with a
thick-disk repartition motivated by NE2001's known high-|b| underestimation — exactly this
mid-to-high-|b|, integrated-to-disk-edge regime. YMW16 is last *for scattering* **by
construction**: its authors state they "do not make use of observations of interstellar
scattering in building the model" (Yao, Manchester & Wang 2017 §3.11) and substitute a
single empirical τ–DM relation with ~1 dex (×10) rms scatter that "cannot be readily
extrapolated to predict the Galactic scattering of extragalactic sources" — precisely this
use case. Its thin-disk density (~0.4 cm⁻³) is ~5× NE2001/NE2025's, so it over-assigns
smooth-disk column and is documented to mispredict per-sightline τ by 1–3 dex (Gum/Vela;
the factor-1200 underscattering of Mall et al. 2022). So the *most-trusted* model gives the
largest excess, its same-lineage parent confirms it 6/6, and only the *least-reliable-for-
scattering* model erases it — via YMW16's expected failure mode (floor *below* measured =
the smooth Galaxy supposedly scattering more than we observe), which is not a contradicting
measurement. The cross-model floors are genuinely apples-to-apples: the in-script assert
shows the τ-derived NE2025 floor reproduces the model's native-SBW excess exactly (ratio
1.00), so all three columns share one conversion. (NB the ranking is *scattering*-specific —
YMW16 is actually better than NE2001 for DM/distance; it is last only for the τ/Δν_d
prediction we use here.)

**What this does and does not establish.** The excess is *robust to model choice among the
scattering-calibrated Cordes-lineage models, and erased only by the one with documented
scattering unreliability* — not "proven," and not a stacked detection. The NE2025 floor
itself carries factor ~2–3 (~0.65 dex) per-sightline uncertainty that is a *shared*
systematic (common thick-disk normalization extrapolated to the disk edge where high-|b|
pulsar anchors are sparse), so it does **not** beat down as √N — no stacked-significance
claim. The low-amplitude sightlines (~2.6–3×, oran/chromatica) sit inside that uncertainty
and are **marginal**; only the high-amplitude ones (~7–11×, zach/wilhelm/hamilton/casey)
survive as robust evidence for an extragalactic screen. Residual checks: a
finite-channel-resolution bias could mechanically inflate the floor/measured ratio (verify
against the mid-|b| ACFs); the fixed C1 = 1.16 / α = 4.4 conventions propagate a correlated
~20–50% into the *absolute* excess; and NE2025 is itself Galactic-calibrated, so the
extragalactic floor is a best-available extrapolation, not an exact prediction. (A truly
independent pulsar/H I floor is still not available locally.)

### Sightline attribution — is the excess an intervening galaxy/CGM? (no)

`sightline_attribution.py` cross-matches the excess sightlines against the project's vetted
intervening-systems catalog (`docs-analysis/foreground.md`, 49 objects classified
confirmed/refuted/inconclusive with impact parameter b). **0/6 excess sightlines pierce the
inner CGM (b < 100 kpc) of a CONFIRMED foreground galaxy.** The closest confirmed-foreground
halos sit in the outer halo where CGM scattering measure is low (casey 171 kpc z=0.20;
chromatica 228 kpc z=0.05); the rest are unreliable PS1-STRM photo-z (extrapolated /
UNSURE), refuted background, or inconclusive (zach's only catalogued nearby object is a
single inconclusive photo-z-extrapolated galaxy at b=76 kpc).
**⇒ the excess favors a host / circumsource screen, not a specific intervening system.**
Caveat: sparse spec-z on these fields cannot exclude a faint undetected intervening dwarf
inside the inner CGM.

## Interpretation / consequences

1. **Scattering and scintillation probe different screens on this sightline.** The α from
   pulse-broadening (the steep-scattering screen) and any x_scint from the resolved
   scintillation are **not** measuring the same medium — they must not be cross-identified
   when attributing the DM/scattering budget.
2. **The resolved scintillation is NOT the smooth NE2025 MW floor** (8× excess scattering) —
   either an enhanced/clumpy Galactic feature or a foreground contribution beyond the smooth
   model. Either way it is ~85× weaker than the pulse-broadening screen.
3. **The dominant pulse-broadening screen is neither the Galactic floor nor the resolved
   scintillation** — it is ~600× stronger than the NE2025 floor and ~85× stronger than the
   resolved screen ⇒ a strong **host / intervening** screen. This is the screen the α /
   scattering science targets, and it is cleanly extragalactic on this sightline.
4. The ~1 GHz lever-arm scintillation cross-check of α is **blocked by CHIME resolution**,
   not by the data in principle — a concrete instrumentation finding (needs upchannelized
   CHIME baseband).

## Artifacts
Promoted into the repo at `analysis/scattering-refit-2026-06/scint_census/`
(scint_candidates.py + scint_mw_final.py are fully reproducible from in-repo configs):
- Scripts: `scint_subband_alpha.py` (wilhelm subband Δν_d), `scint_mw_census.py`
  (BIC-only census, superseded), `scint_candidates.py` (all-stored-fits candidate menu),
  `scint_mw_final.py` (recovery + NE2025-significance census), `scint_mw_models.py`
  (YMW16/NE2001 second-model cross-check), `sightline_attribution.py` (intervening-vs-host
  attribution from `docs-analysis/foreground.md`), `figbank.py` (paper figures)
- Results: `data/scint/{scint_candidates,scint_mw_final,scint_recover_verdicts,scint_mw_census,
  scint_mw_models,sightline_attribution,wilhelm_scint_subband,wilhelm_ne2025_floor}.json`
  (`scint_recover_verdicts.json` = the 7 adversarial-judge verdicts; `scint_mw_final.json` =
  the final 12-burst census; `scint_mw_models.json` = 3-model floor comparison;
  `sightline_attribution.json` = per-sightline foreground attribution)
- Figures (reviewed, verdict match): paper PDFs `pbf_shapes`, `wilhelm_pbf_evidence`,
  `wilhelm_scint_dnud_ne2025`, `codetection_scint_excess` folded into `Faber2026/figures/` +
  `Faber2026/figbank.tex`
