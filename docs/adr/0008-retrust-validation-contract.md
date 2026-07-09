# Re-trust validation contract for burst-data fits

**Status:** proposed (2026-07-09) — V1 keystone of the trust-reset
re-validation program
([plan-trust-reset-revalidation.md](../../Faber2026/docs/rse/specs/plan-trust-reset-revalidation.md)).
Governs every citable scattering, scintillation, and energy product after the
2026-07-06 three-wave trust reset. Not yet implemented; this ADR freezes the
contract so the machinery (unified rail classifier, replicated-data PPC,
fit-verify coverage) can be built against it.

**Supersedes (for quoting):** the ADR-0005 citable-α roster and every
campaign-era PASS/MARGINAL/FAIL verdict produced before this contract is
implemented and the fits re-run under it. ADR-0004 (sub-Kolmogorov floor) and
ADR-0006 (β co-model) remain in force as physics decisions; this ADR governs
*when a fit is trusted*, not *what the parameters mean*.

**Depends on:** [ADR-0004](0004-l1-sub-kolmogorov-alpha-floor.md),
[ADR-0006](0006-beta-coherent-scattering-comodel.md),
[ADR-0007](0007-extended-medium-pbf-for-shallow-alpha.md)

## Context

The 2026-07-06 trust reset revoked trust in every burst-data fit performed to
date — the β-coherent thin-screen campaign, the sub-band EMG fits, the
scintillation ACF fits, and the spectral amplitudes with every derived energy.
The reset was driven by a systemic provenance hole (unpinned inputs, off-repo
builders, removed scripts, hand-transcribed tables, two fit generations feeding
different manuscript tables) plus three fragmentation defects in the validation
machinery itself:

1. **Three rail definitions disagree.** `grade_beta_campaign.classify_rail`
   uses 3σ-proximity **and** posterior-mass (≥30% within 0.05 of a bound) from
   the sampled posterior — the most rigorous. `sim_gate._railed` uses 3σ
   proximity only. `gate_joint_committed.gate_one` uses a fixed edge distance
   `min(β_med−lo, hi−β_med) < RAIL_EDGE` (0.1) with no σ and no mass — a
   median-only test that misses a tight posterior pinned at the bound and
   false-flags a broad interior posterior whose median happens to sit near an
   edge. A fit's rail status must not depend on which script graded it.

2. **"PPC" is a goodness-of-fit χ², not a posterior-predictive check.**
   `joint_ppc_multi.ols_chi2` is a per-channel OLS-gain reduced-χ² of the data
   against the fitted component kernels — a *frequentist point-estimate*
   statistic. A posterior-predictive check replicates datasets from the
   posterior and compares a summary statistic's distribution to the observed
   value. The current statistic cannot detect the failure mode a PPC exists to
   catch: a fit whose best-fit point matches the data but whose posterior
   *implied data distribution* does not (e.g. an over-confident posterior on a
   misspecified model). Mislabeled as "PPC" in the gate and the campaign
   report, it has been treated as the L2/L3 cross-check it is not.

3. **The adversarial verifier does not see joint fits.**
   `.claude/workflows/fit-verify.js` globs `**/*_fit_results.json` only; the
   joint gate writes `*_joint_gate.json` (`gate_joint_committed.py:8-10`), so
   no independent agent re-checks a joint fit's PASS claim. A joint fit can
   pass its own gate unrefuted.

The consequence: the committed gate caps every fit at MARGINAL
(`gate_joint_committed.py:100`, the trailing `"MARGINAL"` in `_worst(...)`)
because L3 τ×Δν is "not evaluable (no dnu_d)" and the "PPC" is incomplete — so
no fit has ever been certified PASS under the real contract, and the
campaign-era PASS/MARGINAL/FAIL tallies rest on a partial, mislabeled
evaluation.

## Decision

A burst-data fit (scattering, scintillation, or energy) is **citable** only
when all five gates below pass, each produced under this contract — not
inherited from the revoked campaign. The gates are ordered: an earlier failure
stops the evaluation.

### Gate 1 — verified input-data lineage

Every input cube consumed by the fit has a pinned SHA-256 in a tracked manifest
(`data-manifest.csv`, no `PENDING` rows) and is verified gen-2+ (no gen-1
de-chirp defect lineage — V2 forensics). A fit on an unpinned or gen-1 input is
not citable regardless of its statistics.

### Gate 2 — synthetic-injection recovery of known truth

The fitter recovers a known-truth injection under each candidate geometry the
sightline will be adjudicated against (thin-screen and, where ADR-0007 fires,
extended-medium). This is the **correctness anchor** — the independent
reference that distinguishes a verified fit from a regression baseline. The
recovery test is pre-registered: the acceptance criterion (recovered β within
Δβ of truth, τ within a stated factor, α = 2β/(β−2) closure satisfied) is
stated *before* the injection runs, and the tolerance is derived from the
injection's S/N and the fitter's posterior width, not tuned to the result. A
fit whose injection recovery fails is not citable even if its χ² is excellent —
the fitter has not been shown to measure the quantity it reports.

### Gate 3 — prior-rail behavior (unified classifier)

A posterior is classified by a **single** rail classifier,
`flits/fitting/rails.py`, imported nowhere else. The classifier uses the
sampled posterior (not the summary): railed = 3σ-proximity to a bound **or**
≥30% posterior mass within 0.05 of a bound, on whichever parameter is sampled
(β for the co-model; α for legacy free-α fits). A railed posterior is
**model-family rejection** (the thin-screen closure fails for that sightline —
ADR-0007's re-open trigger), never a quotable limit. No α is quoted in any form
for an ex-railed row until per-sightline geometry model selection adjudicates.
This retires the `RAIL_EDGE` median-only test and the 3σ-only test; the
posterior-mass test is the canonical one because it catches a tight posterior
pinned at the bound that the median-only test misses.

### Gate 4 — posterior-predictive check (replicated data)

A **true** PPC: replicate N datasets from the posterior, compute a summary
statistic on each, and compare the observed statistic to the replicated
distribution. The summary statistic is one that the fit's point-estimate χ²
cannot see — e.g. the secondary-pulse residual power, the ACF Lorentzian width,
or the per-band τ ratio — chosen to probe the model family under test. The fit
passes when the observed statistic falls within the central 95% of the
replicated distribution (a two-tailed posterior-predictive p-value in
[0.025, 0.975]). The per-band OLS-gain χ² (`ols_chi2`) is retained as an L2
goodness-of-fit diagnostic but is **not** the PPC and is not labeled as one.

### Gate 5 — independent cross-check (re-run, not inherited)

An independent cross-check — sub-band slope consistency, or τ·Δν_d screen
consistency (L3, now evaluable because the scintillation campaign supplies
Δν_d) — is produced **under this contract**, not inherited from the revoked
campaign. The cross-check must itself rest on Gate 1 inputs and Gate 4 PPC
where it consumes a fit product. A cross-check that passes under the old
campaign is not evidence of trust.

### Figure review

No fit is certified PASS until its diagnostic figures have been visually
reviewed (the existing figure-review Stop gate). A numeric PASS without figure
review is not a pass — this is already repo policy and is restated here so the
contract is self-contained.

## Consequences

- **`flits/fitting/rails.py`** is the single rail definition; the three
  existing ones (`grade_beta_campaign.classify_rail`, `sim_gate._railed`,
  `gate_joint_committed`'s `RAIL_EDGE` test) are replaced by imports of it.
- **A replicated-data PPC** is built and required by the gate; `ols_chi2` is
  relabeled as `chi2_red_band` (a goodness-of-fit diagnostic), not "PPC".
- **`fit-verify.js`** globs both `*_fit_results.json` and `*_joint_gate.json`
  so joint fits are adversarially re-checked.
- **The MARGINAL cap** in `gate_joint_committed.py:100` is removed once Gate 4
  and Gate 5 are evaluable; until then the cap stays as an honest "incomplete
  contract" signal, but the reason string must say *which* gate is unevaluated,
  not the generic "no dnu_d".
- **No citable re-fit** is produced until all five gates are implemented and
  the fits re-run on verified inputs. The ADR-0005 roster and the campaign
  verdicts remain revoked. This ADR does not itself produce a fit — it is the
  contract the C-lane re-fit (plan §C) will be judged under.
- **Manuscript impact:** `tab:beta` stays withheld until C3 (plan §C3); the
  abstract/Discussion/Conclusions β-language purge (F1) is unblocked by this
  ADR's acceptance but the prose is not filled until C3 lands.

## Alternatives considered

- **Patch the existing gate in place** (fix the `RAIL_EDGE` test, relabel the
  PPC, widen the glob): rejected — the trust reset is not a bug fix, it is a
  re-derivation. Patching the gate on the revoked campaign's outputs would
  re-certify fits on unverified inputs, which is the hole the reset opened.
- **Defer the contract until after the geometry-selection campaign (A2/A3):**
  rejected — A2/A3 *produce* fits, so they must be judged under the contract,
  not the other way around. The contract is the gate the re-fit waits on
  (plan dependency spine: V1 → C1).
- **Accept regression-pinning of the revoked campaign as the baseline:**
  rejected per the hardening Iron Law — a regression baseline is not a
  correctness check. The revoked outputs are labeled `UNVERIFIED` and are not
  a reference for the re-fit; Gate 2 (injection recovery) is the correctness
  anchor.
