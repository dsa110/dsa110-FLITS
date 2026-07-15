# P4 — exploratory intrinsic-envelope modeling + residual scintillation search: RESULT

**Status: DOCUMENTED-FAIL (envelope not separable) — the predeclared E2 fail
branch. The real on-pulse residual was never scanned (0 of the 3 permitted
looks spent).**

- Record: `docs/rse/specs/experiment-chime-scint-p4-envelope-model.md`
  (Faber2026), frozen before any residual statistic was computed.
- Evidential class: exploratory by construction (the single blind on-pulse
  computation was spent in P3′, owner-authorized 2026-07-15). Every on-pulse
  read in this experiment is owner-sanctioned and flagged in code.
- Frozen config: `p4_frozen_config.json`
  (sha256 `e8e5e01dccf7f890c2f7261599747ea9b4a4a160b464df1b1805a70a2bfbecf9`);
  assets `p4_assets.npz`. Environment: conda `py312`
  (numpy 2.4, scipy 1.17, matplotlib 3.10).

## Question

P3′ found freya's CHIME on-pulse ratio spectrum dominated by broad intrinsic
spectral structure (â ≈ 10⁻³, ~11× the scintillation ceiling
(f_b·m)² ≈ 5.1–7.2×10⁻⁵). Can that envelope be modeled and divided out well
enough that the residual regains sensitivity to a scintle at the ceiling?

## Answer

**No.** The intrinsic envelope carries structure at every scale down to at
least ~0.5–1 MHz, so the frozen envelope/scintle scale separation does not
exist in this data:

- **Stiff models leave un-attributable false structure.** Every operating
  point except one fails the surrogate model-mismatch control: envelope
  fits by a *different* plausible family, analyzed through the chain,
  produce spurious max-z of **5.4–71.9** against a noise-arm p95 of
  ~1.8–2.4 (bound: 1.5×noise). Choosing any of these scales would make
  "residual structure" unattributable between scintillation and
  envelope-model error.
- **The one control-clean model absorbs the signal.** M2 GP at ℓ = 0.5 MHz
  passes control (surrogate p95 = 1.17) but eats the injected scintles:
  recovered-width bias −0.45…−0.93 across the grid, **zero certified
  cells**.
- **Certification only ever happens below the admissible window.** The only
  certifying cells anywhere are Δν_d = 77 kHz (M1 Λ=0.5: pulls 1.2–1.45;
  M2 ℓ=1.0: pulls 0.37–0.54) — below the 127 kHz Gate-0 window for
  m = 0.15 — and only at scales that fail control. At every admissible
  width (≥127 kHz) the residual background biases the recovery by
  3–43σ pulls with widths railed to the 400 kHz scan edge.

Frozen drop rules: M1 and M3 — no control-clean scale; M2 — zero certified
cells at Δν_d ≥ 127 kHz on its only control-clean scale. All three families
dropped → fail branch → the real residual is never scanned.

## E2 summary (full tables in `e2_calibration.json`)

| family | scale [MHz or k_env] | certified cells (of 8) | noise p95 max-z | surrogate p95 max-z | control-clean |
|---|---|---|---|---|---|
| M1 spline | 0.5 | 2 (both 77 kHz) | 1.85 | 12.43 | no |
| M1 spline | 1 / 2 / 5 / 10 | 0 | 1.9–2.4 | 25.8 / 35.1 / 57.1 / 56.5 | no |
| M2 GP | 0.5 | 0 | 2.01 | **1.17** | **yes** |
| M2 GP | 1.0 | 2 (both 77 kHz) | 2.00 | 5.38 | no |
| M2 GP | 2 / 5 / 10 | 0 | 2.0–2.1 | 12.2 / 32.5 / 44.5 | no |
| M3 delay-cut | 25 / 50 / 100 / 200 | 0 | 1.8–2.1 | 71.9 / 52.0 / 32.7 / 37.4 | no |

Injection convention: scintle gains ride the burst component only
(`1 + m·δ·(E_ref−1)/E_ref`), templates rebuilt through the identical
model+subtract chain per operating point and normalized so â = (f_b·m)²
(P3′ units; m = √â/f_b). Seeds: injections 600000+, nulls 650000+,
surrogates 680000+, templates 760000+ — all disjoint from prior spaces.

## E0 (descriptive; `e0_envelope.json`, figures)

- Envelope contrast ⟨R−1⟩ = 0.054 (matches the frozen f_b = 0.05).
- The intrinsic structure appears as a delay-power bump at k ≈ 20–40
  (~3.5–7 MHz scales) — inside the k ≥ 11 scan window, confirming why P3′
  detected it at 40σ.
- Per-channel half-field noise σ ≈ 0.13; half-to-half per-channel
  correlation r = 0.059 (envelope is aggregate-visible, not
  per-channel-visible).
- Profile: **single** component (smoothed S/N 11 at sample ~252) →
  **E3b unavailable** under the frozen component rule.
- Off-pool tail rise at samples ~430–437 (likely dedispersion wrap);
  inherited from the P2/P3 frozen windows, calibrated around by every null
  campaign; noted for completeness.

## Discriminant availability (evaluation-time facts)

- **E3b (component correlation): unavailable** — one profile component.
- **E3c (DSA cross-band anchor): unavailable** — freya has no *trusted*
  DSA-band Δν_d (the DSA-band fits are revoked pending the trust reset;
  the only certified DSA measurement in the sample is FRB 20220506D,
  γ = 0.446 MHz). Even had E2 passed, no candidate could have been promoted
  under the frozen taxonomy.

## Consequence for the manuscript

P4 closes the sanctioned exploratory route with a quantitative statement:

> The intrinsic spectral envelope of the burst is not separable from a
> putative scintillation signal by smooth-envelope modeling: every model
> family stiff enough to preserve a 77–352 kHz scintle leaves
> envelope-model-mismatch false structure at 5–72σ, and the one model
> flexible enough to pass the mismatch control absorbs the injected
> scintles themselves. The CHIME-band scintillation constraint for this
> burst is therefore envelope-confusion-limited at every stage: no
> admissible Δν_d measurement, and no quantifiable post-subtraction upper
> limit either.

This supersedes nothing in P3′ — it strengthens the P3′ closure by showing
the foreground is not removable in-house. Owner review of the closure
wording (deferred since P3′) can now proceed with both results in hand.

## Figures

- `figures/e0_envelope.png` — spectrum + reference envelope, full-k delay
  power, half-to-half scatter, amplitude distribution.
- `figures/e0_profile.png` — profile, smoothed S/N, component rule.
- `figures/e2_certification.png` — certification heatmap per family×scale.
- `figures/e2_control.png` — surrogate vs noise p95 max-z per operating
  point (the control chart that drives the fail branch).
