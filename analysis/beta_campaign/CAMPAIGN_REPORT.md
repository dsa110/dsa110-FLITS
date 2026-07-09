# β-coherent thin-screen campaign — pass 1 report

Locked 2026-07-06 · branch `campaign/beta-thin-screen-pass1` · plan
`docs/rse/specs/plan-beta-coherent-thin-screen-campaign.md` · decision record
ADR-0006 (β co-model), ADR-0004 (rail semantics); ADR-0007 (thin vs extended)
deliberately deferred behind this pass.

## What was run

**Supersession note (2026-07-07):** this pass-1 report originally recorded the
historical JohnDoeII C2D1 run. The jointmodel-pair multiplicity audit superseded
that choice, and the promoted beta-coherent product now uses JohnDoeII C2D2.

All 12 CHIME–DSA co-detections were re-fit with the β-native joint model
(thin-screen PBF family: power-law-tail members for β < 4, exponential member
at the β = 4 square-law limit). β sampled with bounds (3.0, 4.0); α = 2β/(β−2)
derived, never sampled. Per-burst component multiplicity follows the locked
suffix map (`grade_beta_campaign.SUFFIX`):

| suffix | bursts |
|---|---|
| `_sharedzeta` | freya, casey, chromatica, wilhelm, hamilton |
| `_C1D1` | mahi, zach |
| `_C2D1` | oran, isha; johndoeII in historical pass 1 only |
| `_C2D2` | whitney_fine; johndoeII promoted |
| `_C3D3` | phineas |

Every fit was verified against its own posterior-predictive check before
grading (`fleet_status.json`); freya additionally passed the regression gate
against its reference run (|Δβ_ref| = 0.038 < 0.05). Runs root:
`$FLITS_RUNS/data/joint` (fit JSONs + `*_joint_samples*.npz` posteriors);
citable fit JSONs are copied into `analysis/beta_campaign/fits/`.

## Census (grader output, `beta_campaign_verdicts.{json,md}`)

Rail rule (ADR-0004): railed = within 3σ of a β bound OR ≥30% posterior mass
within 0.05 of the bound, from the sampled posterior (not the summary).

| burst | β (−/+) | α | τ₁ᴳᴴᶻ [ms] | χ²ᵣ C/D | rail | gate |
|---|---|---|---|---|---|---|
| freya | 3.722 (−0.015/+0.014) | 4.32 | 0.119 | 1.29/1.03 | **interior** | MARGINAL |
| phineas | 3.228 (−0.018/+0.021) | 5.26 | 0.469 | 1.06/1.24 | **interior** | MARGINAL |
| casey | 3.990 | (4.0 limit) | 0.0186 | 1.57/1.02 | railed-hi | MARGINAL |
| mahi | 3.785 (−0.155/+0.138) | (4.0 limit) | 0.219 | 1.04/0.90 | railed-hi | MARGINAL |
| oran | 3.987 | (4.0 limit) | 0.843 | 1.02/1.22 | railed-hi | MARGINAL |
| isha | 3.841 (−0.105/+0.085) | (4.0 limit) | 0.314 | 1.05/0.91 | railed-hi | MARGINAL |
| johndoeII | 3.936 | (4.0 limit) | 2.219 | 1.09/1.23 | railed-hi | MARGINAL |
| whitney_fine | 3.968 | (4.0 limit) | 1.182 | 1.09/1.42 | railed-hi | MARGINAL |
| wilhelm | 3.979 | (4.0 limit) | 0.269 | 1.57/6.73 | railed-hi | MARGINAL |
| hamilton | 3.978 | (4.0 limit) | 0.0245 | 3.96/1.00 | railed-hi | MARGINAL |
| zach | 3.990 | (4.0 limit) | 0.294 | 2.51/1.31 | railed-hi | MARGINAL |
| chromatica | 3.990 | — | — | 11.59/9.25 | railed-hi | **FAIL** |

Headline: **2 interior members (freya, phineas) are genuine power-law-tail
detections**; the interior β for phineas (3.23) implies α = 5.26 — the
steepest in the sample. **10 posteriors rail at β = 4**: exponential-consistent,
α = 4 quotable only as a geometry-conditioned limit, and each is an ADR-0007
re-open candidate (an extended-medium PBF may fit these without railing).
No burst railed low or came back unconstrained. All gates cap at MARGINAL
because L3 τ×Δν was "not evaluable (no dnu_d)" at gate time — the two-screen
table below closes that observation gap out-of-band. chromatica FAILs L2
outright (χ²ᵣ ≈ 11.6/9.3; the montage shows its model panel flat against
strongly structured data) and is excluded from all citable products.

## Citable roster re-lock (`citable_alpha_roster.json`)

- **Tier A (7, fully adjudicated):** freya, phineas (interior β + derived α);
  casey, mahi, oran, isha, johndoeII (railed-hi, α = 4 limits).
- **Tier B (3, provisional):** wilhelm, hamilton, zach — railed-hi with L2
  χ² caveats (6.73 DSA / 3.96 CHIME / 2.51 CHIME respectively).
- **Multiplicity exemplar:** whitney (C2D2 after local re-prep) — quoted as
  exemplar, not a roster row.
- **Excluded:** chromatica (gate FAIL).

## Two-screen consistency (`two_screen_consistency.{json,md}`, DSA band)

Statistic: τ(1.4 GHz)[s] × Δν_d(1.4 GHz)[Hz] = C₁/(2π); one screen gives
0.159 (thin) … 1.0 (extended), accepted range [0.1, 2]; ≫2 ⇒ the resolved
Δν_d samples a **nearer** screen than the scattering one. Δν_d from stored
subband ACF fits, lowest-BIC model, detection-cut γ > σ(γ) (whitney's
subband_1 BIC winner rails a γ at its 0.06 MHz bound with σ > value — a
non-detection that would otherwise dominate the inverse-variance mean).

| burst | product | verdict |
|---|---|---|
| hamilton | 1.64 | **same_screen** |
| freya | 2.60 | different_screens (marginal) |
| wilhelm | 8.98 | different_screens |
| casey | 60.5 | different_screens |
| oran | 74.6 | different_screens |
| mahi | 87.2 | different_screens |
| chromatica | 93.9 | different_screens (τ not citable — gate FAIL) |
| isha | 39.0 | different_screens |
| zach | 37.6 | different_screens |
| johndoeII | 281 | different_screens |
| phineas | 624 | different_screens |
| whitney_fine | 6259 | different_screens |

Only hamilton is consistent with a single screen doing both the scattering
and the scintillation; everywhere else the resolved DSA scintillation comes
from a nearer (plausibly Galactic) screen, wilhelm's known pattern. CHIME-band
Δν_d is not in stored_fits (needs a fresh ACF pass) — recorded as `@decision`
in `.agents/deferred-tasks.md`.

## Sightline budget integration (`results/sightline_dm_scattering_budget.*`)

Campaign τ posteriors flow into the budget through the roster's `fit_json`
overrides (`galaxies/foreground/tau_consistency.py`). Coverage semantics
(survey_coverage.csv, exact-MOC footprints): a burst outside a survey's
footprint is **unconstrained by that survey, never "no foreground"**; deep
constraint requires a deep survey (DESI DR8-N / SDSS DR12) row that is
footprint_empty or has foreground/with-z hits. Consequences visible in the
regenerated budget: oran and mahi read "intervening screen unconstrained
(not excluded)"; hamilton's predicted intervening τ plausibly dominates its
measurement (pred/obs = 0.68 — consistent with its same_screen verdict
above); chromatica carries no measured τ.

## Artifacts

- `beta_campaign_verdicts.{json,md}` — grader census (this table)
- `citable_alpha_roster.json` — re-locked roster (7A / 3B / exemplar / 1 excluded)
- `beta_table_rows.tex` — manuscript tab:beta rows (11 rows; chromatica in the
  excluded comment)
- `two_screen_consistency.{json,md}` — τ×Δν table + Δν_d provenance
- `fits/` — citable fit JSONs + PPC summaries
- `fleet_status.json` — per-burst fit/PPC completion ledger
- `results/sightline_dm_scattering_budget.{csv,md,svg,png,pdf}` — regenerated
- montage: `plot_jointmodel_montage.py` → figures export for the manuscript

## Open items

- ADR-0007 (thin vs extended medium): re-open with the 10-member railed-hi
  list as the candidate set.
- CHIME-band Δν_d gap (@decision in the ledger).
- Tier B → A promotion pending S2 (second-screen / systematics pass) on
  wilhelm, hamilton, zach.
