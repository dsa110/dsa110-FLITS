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
| `_C1D1` | mahi |
| `_C2D1` | oran, isha |
| `_C2D2` | whitney_fine; johndoeII promoted |
| `_C2D4_cwin` | zach promoted (bespoke morphology-audit refit) |
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
| zach | 3.990 | (4.0 limit) | 0.186 | 1.35/1.02 | railed-hi | MARGINAL |
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
- **Tier B (3, provisional):** wilhelm, hamilton, zach — railed-hi; wilhelm/hamilton
  retain L2 χ² caveats (6.73 DSA / 3.96 CHIME respectively). zach's C2D4 promotion
  clears its earlier CHIME χ² caveat (now 1.35/1.02); tier-B pending_s2 designation
  retained on the second-screen multiplicity robustness question.
- **Multiplicity exemplar:** whitney (C2D2 after local re-prep) — quoted as
  exemplar, not a roster row.
- **Excluded:** chromatica (gate FAIL).

## Two-screen consistency (`two_screen_consistency.{json,md}`, DSA band)

**SUPERSEDED / SCIENTIFICALLY INVALID.** This calculation paired free-alpha
joint-fit tau with DSA bandwidths. The accepted policy requires fixed-index
`tau_consistency` refits, so none of the former same-screen/different-screen
verdicts is current. Exact JSON/Markdown outputs are preserved under
`.archive/outdated-science/2026-07-17/analysis/beta_campaign/` for audit.

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
- `.archive/outdated-science/2026-07-17/analysis/beta_campaign/two_screen_consistency.{json,md}` — invalid historical τ×Δν calculation retained for provenance
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
