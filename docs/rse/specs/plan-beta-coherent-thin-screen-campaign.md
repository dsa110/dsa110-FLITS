# Plan: β-coherent thin-screen campaign (pass 1) → manuscript integration

---
**Date:** 2026-07-06
**Author:** AI Assistant (owner-directed)
**Status:** Approved (owner, 2026-07-06) — "record the decision, draft the plan,
and then go ahead and implement the plan"; PI approvals overridden for the
session (owner directive), report at completion.
**Branch:** `campaign/beta-thin-screen-pass1`
**Depends on:** ADR-0006 (β co-model), ADR-0004 (rail-MARGINAL), ADR-0005
(roster, to be re-locked), ADR-0007 (deferred by this plan)
---

## Decision being implemented

Assume **thin-screen geometry throughout** and run the β-coherent thin-screen
family — power-law-tail PBF members for `beta < 4`, analytic exponential limit
at `beta = 4` — on the **full 12-burst co-detection sample**, which has never
completed successfully under ADR-0006 (only freya has a committed β fit).
Thin-vs-extended (ADR-0007) is **deferred** and re-opens per burst on β-rail
evidence from this pass. This discharges ADR-0006's obligation ("re-run
joint-fit campaigns before quoting population β/α statistics") and replaces
the formally-uncitable free-α+exp roster values.

Key semantics: the power-law-tail vs exponential distinction is a **posterior
question, not model selection** — β is continuous, the exponential is the
`beta = 4` boundary member (`BETA_EXP_EPS = 0.02` switch). Per burst the β
posterior lands either **interior** (genuine β measurement, α = 2β/(β−2)) or
**railed at β = 4** (exponential-consistent; α = 4 quoted as a limit;
ADR-0007 candidate). Rail-MARGINAL per ADR-0004: median within ~3σ of either
prior bound.

## Phases

### 0. Record decision (docs) — this commit
ADR-0007 status → deferred; ledger item retargeted; this plan.

### 1. Kernel boundary verification
`gaussian_powerlaw_convolution` (burstfit.py:197) across β ∈ (2.01, 4]:
- switch continuity at β = 3.98 (power-law vs closed-form exp) after
  cross-correlation t0 alignment (the FFT path's ~0.5-sample registration
  offset is t0-degenerate — docstring; test aligns before comparing),
- area normalization ≈ 1 across a β grid,
- tail-mass monotonicity (heavier tail at lower β; → 0 as β → 4⁻),
- `alpha_from_beta` plateau consistency at the ε-window.
New: `scattering/scat_analysis/tests/test_beta_kernel_boundary.py`.
**Gate:** tests green.

### 2. Runner β-nativization
`analysis/scattering-refit-2026-06/local_runs/run_joint_fit.py` (the β-aware
local variant; the top-level twin still crashes on `FRBParams(alpha=…)` —
fixed the same way):
- add `--beta-lo/--beta-hi` (explicit `beta_bounds`; legacy `--alpha-lo/hi`
  retained as deprecated alias),
- delete the dead `--pbf-C/--pbf-D/--beta-C/--beta-D` knobs (zero kernel
  consumers since ADR-0006 removed `FLITS_PBF`; they set dead attributes and
  mis-tag output files),
- write top-level `beta` + `beta_bounds` into the summary JSON (gate_one's
  β-native path keys off `"beta" in fit`),
- console rail flag computed on the **β** posterior vs β bounds.
**Gate:** full local suite green; freya config smoke-parses.

### 3. Sim-injection gate (before any real-data fleet)
`analysis/beta_campaign/sim_gate.py`: synthetic two-band (CHIME-like 0.4–0.8 +
DSA-like 1.2–1.5 GHz) injections routed through the **production**
`fit_joint_scattering` shared-ζ path (not a private likelihood):
- β_true ∈ {3.3, 3.7}: recovered median within 3σ, un-railed;
- β_true = 4.0 (exponential injection): posterior rails at the β = 4 bound
  (rail classifier fires).
**Gate:** all three verdicts correct, or the fleet does not launch.

### 4. Config completion (local-runs pattern, issue #99)
Runtime configs exist for casey, chromatica, freya, oran, phineas,
whitney_fine, wilhelm; generate the missing five (hamilton, isha, johndoeII,
mahi, zach) with pinned-checkout data paths, per-band f/t factors from the
repo burst templates, `dm_init` = 0.0 CHIME / catalog DM DSA.

### 5. Fleet: 12 β-coherent joint fits (local)
Runner: fixed `local_runs/run_joint_fit.py`, `FLITS_RUNS` = local runtime
(`scratch/flits-local-runs`, data symlinked to `~/Data`). Per-burst model =
the previously adjudicated multiplicity (ADR-0005 / grade_allexp CANON):

| model | bursts |
|---|---|
| shared-ζ C1D1 | casey, chromatica, freya (regression vs committed verdict β=3.684±0.013), wilhelm, hamilton (excluded-class, flagged) |
| C1D1 (multi path) | mahi, zach |
| C2D1 | oran, isha |
| C2D2 | johndoeII, whitney (whitney_fine data) |
| C3D3 | phineas |

β prior: default `(3.0, 4.0)` (α ∈ [4, 6] — matches the L1 gate ceiling; the
β = 3 lower rail maps to the α = 6 L1 bound and is flagged the same way).
nlive 600 (multi ≥ 800 per the ndim≥12 advisory), dlogz 0.5, nproc 6–8,
2–3 bursts concurrent, per-burst logs. PPC (chi2_chime/chi2_dsa) after each
fit, same machinery as the all-exp campaign. Multiplicity re-adjudication
under β is **out of pass-1 scope** unless a fit FAILs its gate.

### 6. Adjudication + campaign report
β-native grading via `gate_one` (already β-aware) + rail-fraction statistics
from the posterior npz (mass within 3σ / within 0.1 of each β bound).
Output: `analysis/beta_campaign/` per-burst verdict JSONs + campaign report
md (committed; npz stay in scratch). Classifications: **interior** /
**β=4-railed** (ADR-0007 re-open list) / **β=3-railed** / gate-FAIL.
Draft the roster re-lock (ADR-0005 amendment): quoted values become β with
derived α; railed bursts quote α = 4 as a geometry-conditioned limit.

### 7. Two-screen scintillation analysis
On the campaign posteriors: Δν_d per band where the GP/ACF surface resolves
it, τ×Δν consistency (C₁ geometry constant), two-screen localization for
sightlines with both scales measured.

### 8. Foreground / sightline analysis — survey-coverage-aware
Refresh foreground galaxy + cluster identification for all 12 sightlines.
**Coverage semantic (owner directive):** survey footprint is a per-burst
applicability mask — a burst outside a survey's coverage is *unconstrained by
that survey*, never "no foreground"; only footprint-covering surveys
constrain a sightline, and the budget/tables must carry
covered-vs-unconstrained per burst explicitly. Audit
`galaxies/foreground/survey_coverage.py` consumers for this semantic before
regenerating `sightline_dm_scattering_budget`.

### 9. Combined analysis + manuscript integration
Join β-campaign scattering results with the coverage-aware foreground set
(τ/DM attribution per sightline); regenerate budget outputs; then Faber2026:
`tab:alpha` → β-campaign values with rail semantics, montage/ladder figures
from the new fits, prose per CONTEXT.md contract, `make` exits 0, pin bump,
push, Overleaf pull.

## Verification map
Phase 1/2 → pytest; Phase 3 → sim-gate verdicts; Phase 5 → per-fit
PASS/MARGINAL/FAIL gates + figure-review of diagnostics; Phase 6 →
adversarial re-check of verdicts (fit-verify pattern); Phase 9 → `make`
clean + figure review. Failures reported, not rationalized.

## Risks / known constraints
- **At plan time, much of the old roster sat at α < 4** (casey 2.40, oran 2.66,
  the now-superseded johndoeII 1.53 under free-α+exp): expect several β = 4 rails. That outcome is the
  *deliverable* (the ADR-0007 target list), not a campaign failure.
- Local wall-time: HPCC history suggests ~30 min × 8 cores per C1D1 burst,
  phineas C3D3 ~1 h × 16 — expect several hours total locally; run
  concurrently and adjudicate as results land.
- freya regression: if the re-run disagrees with the committed verdict
  (β = 3.684 ± 0.013) beyond errors, STOP the fleet and diagnose.
- dynesty runs are not seeded (historical convention); reproducibility is by
  recorded knobs + posterior artifacts, not bitwise identity.
