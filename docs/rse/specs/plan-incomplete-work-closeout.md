# Implementation Plan: Incomplete-work closeout (docs → joint lane → tooling)

---
**Date:** 2026-06-24
**Author:** AI Assistant
**Status:** Draft
**Related Documents:**
- [Research: Survey of apparently-incomplete work in FLITS](research-incomplete-work-survey.md)
- [CHIME/DSA Co-detection Science Plan](../../codetection-science-plan.md)
- [Plan: Manuscript completion](plan-manuscript-completion.md)

---

## Overview

The incomplete-work survey (`research-incomplete-work-survey.md`, commit `3d27970`)
found four buckets of unfinished work: stale documentation, an uncommitted joint-fit
campaign, a genuine open code issue (#4), and unbuilt science tooling. This plan
closes the **agent-doable** subset end-to-end and **explicitly surfaces** the
decision-gated science items so they are not silently dropped.

**Goal:** Every confirmed-stale doc reference corrected; the `joint_ladder/`
gain-marginal campaign landed as reviewable artifacts; issue #4 (N=1 evidence
commensurability) fixed with a regression test; the two-screen consistency layer and
the NE2025 Galactic floor wired into the scintillation pipeline; an ACF re-validation
harness built and unit-tested. Each lands on its own branch as a separate PR.

**Motivation:** The survey showed the codebase reads as *less* finished than it is
(resolved stubs still cited as open) while real work (the joint ladder, the unwired
two-screen funcs) sits one step from done. Closing this gap unblocks the headline
science (empirical α + screen localization) and removes the "fix code that is already
done" trap.

## Current State Analysis

**Existing Implementation:**
- `flits/batch/batch_runner.py` (456 lines) — scint analysis is wired:
  `discover_scint_configs` (`:58`), `_run_scintillation_analysis` (`:128`). No
  `# TODO: Add scintillation config generation` exists. Test:
  `flits/batch/tests/test_scint_config_discovery.py`.
- `flits/batch/analysis_logic.py:109` — `check_tau_deltanu_consistency` is fully
  implemented (computes τ×Δν products + errors + thin-screen implied τ).
- `scattering/scat_analysis/burstfit_joint.py` — tracked. `_gain_marginal_multi_band`
  (`:190`), `_joint_prior_spec_gain_multi` (`:483`). **The issue-#4 routing fix already
  exists:** `fit_joint_scattering` takes `force_multi: bool = False` (`:850`) and the
  gate `multi = bool(force_multi) or int(components_C) > 1 or int(components_D) > 1`
  (`:878`) already routes N=1 through the proper-prior multi path when opted in. The
  default single-component path still uses the flat improper prior in
  `scattering/scat_analysis/burstfit.py:log_likelihood_gain_marginal`. So #4's **code
  half is done**; the missing acceptance item is the regression test.
- `analysis/scattering-refit-2026-06/joint_json/` — **committed** canonical c0/γ joint
  fits for 11 bursts (no casey) + `_gate`/`_ppc` companions. This is the legitimate
  "11/12".
- `analysis/scattering-refit-2026-06/joint_ladder/` — **untracked**: a newer
  gain-marginal ladder (12/12 incl. casey), 99 JSONs (base + `_sharedzeta` + `_CxDy` +
  `_s2-{1,10,100}`) + scripts `_ladder.py` (ranked table), `_s2verdict.py` (cross-N
  Bayes-factor diagnostic, 73 lines, no `__main__`), `_figs.py` (synthesis PNGs).
- `scintillation/scint_analysis/analysis.py` — defined-but-uncalled two-screen funcs:
  `scattering_scintillation_consistency` (`:1098`), `interpret_modulation_index`
  (`:776`), `estimate_emission_region_size` (`:875`), `two_screen_coherence_constraint`
  (`:1011`). Exported in `__init__.py:39-42`; zero pipeline callers.
- `scintillation/scint_analysis/pipeline.py:258-269` — `ScintillationAnalysis.run`
  ends by populating `self.final_results` from `analyze_scintillation_from_acfs`; the
  natural wiring point is after the 2D fit (`~:268`).
- `scintillation/scint_analysis/consistency.py` — `band_consistency` /
  `consistency_table` ARE wired + tested (`tests/test_consistency_wiring.py`); this is
  the band-level relation, distinct from the per-measurement `analysis.py` funcs.
- `scintillation/ne2025/query_ne2025_scint.py` — `galactic_floor(coord, bands, alpha,
  model)` (`:109`) → `{band: {tau_ms, bw_kHz}}`; `query_single` (`:93`). Test:
  `tests/test_ne2025_floor.py`. Callers only in `analysis/.../scint_census/`, not the
  pipeline.
- `crossmatching/toa_crossmatch.py:128` — `compute_geometric_delay()` already exists
  (baseline·source delay → ms; result carries `geometric_delay_ms`).
  `crossmatching/association.py` — pillars 1/3/4 implemented, pillar 2 wired-awaits-data.
- `scintillation/scint_analysis/noise.py` — `NoiseDescriptor`, `_acf_1d` (`:32`),
  `_robust_std` (`:51`); no RFI/off-pulse module exists. Test: `tests/test_noise.py`.
- `docs/architecture/inventory.md:258-264` — scintillation notebook dirs hamilton /
  phineas / whitney / oran listed "(files TBD)".

**Current Behavior:** The scattering + scintillation pipelines run and produce τ, α,
Δν per burst, but the two-screen interpretation layer and the Galactic floor are
computed only in ad-hoc analysis scripts, never attached to pipeline output. Joint
model selection cannot compare N=1 vs N≥2 because the two paths use different
evidence normalizations.

**Current Limitations:**
- Four doc references describe resolved code as unfinished (survey §6).
- The joint ladder + its diagnostic live only in the working tree (data loss risk).
- Issue #4: 1- and N-component lnZ are on different additive scales → no Occam-correct
  component-count selection through `fit_joint_scattering`.
- Two-screen + Galactic-floor results are not in the pipeline's output objects.
- No harness certifies the 3 measured Δν (casey/freya/wilhelm) as diffractive vs RFI.

## Desired End State

**New Behavior:** Docs match code. The joint ladder is committed with a runnable
diagnostic. `fit_joint_scattering(..., force_multi=True)` routes N=1 through the
gain-marginal multi path, giving commensurable lnZ across the N ladder, with a
regression test pinning it to a brute-force oracle. `ScintillationAnalysis.run`
attaches consistency, emission-size, and Galactic-floor results to `final_results`.
A new `scintillation/scint_analysis/revalidation.py` flags RFI/off-pulse
contamination and re-measures Δν on cleaned data.

**Success Looks Like:**
- `rg` finds zero stale references (Phase 1 verification commands all return empty).
- `git ls-files analysis/scattering-refit-2026-06/joint_ladder/` lists the scripts +
  JSONs; `python -m analysis...joint_ladder._s2verdict` runs and prints verdicts.
- `pytest tests/test_issue4_commensurable.py` passes (N=1-via-multi == brute-force).
- `pytest scintillation/scint_analysis/tests/test_pipeline_wiring.py` passes
  (`final_results` carries `consistency`, `emission_size`, `galactic_floor`).
- `pytest scintillation/scint_analysis/tests/test_revalidation.py` passes (RFI spike
  flagged; clean Δν recovered within tolerance).

## What We're NOT Doing

- [ ] **Implementing the `pipeline/core.py` earmark physics** (anisotropy, polynomial
      baseline marginalization, AR(1)/GP residual model — `core.py:1004-1025`).
- [ ] **Probabilistic host-DM treatment** (subtract `p(DM_cosmic|z)` instead of the
      mean) — see Decision Gate D2.
- [ ] **The geometric-delay *localization solver*** (χ² over two-screen effective
      distances). `compute_geometric_delay` exists; the forward model does not — see
      Decision Gate D3.
- [ ] **Running the scint campaign over the 9 unmeasured bursts** — needs hand-tuned
      RFI/window configs (a human+data task; the *code* is done).
- [ ] **Running the ACF re-validation over the 3 real bursts' raw data** — Phase 6
      builds + unit-tests the harness; the real-data run is a data-gated manual step.
- [ ] **The full two-screen forward model from arXiv:2505.04576** (Resolution Power,
      inverting two Δν → screen distances/positions, scintillation-quenching regime) —
      Phase 6 implements only the **empirical** two-component ACF *measurement* (wide +
      narrow Δν, m); turning those into screen distances is the localization solver
      gated by D3.
- [ ] **The manuscript energies 6-vs-8 reconciliation + nickname↔TNS swap** — the
      existing `@decision` ledger item (`.agents/deferred-tasks.md:19`).
- [ ] **Resolving the PR #47 / #49 / #50 figures-docs overlap** — a separate-active
      git lane; branch-hygiene decision for the user.
- [ ] **Pushing branches / opening PRs** — one-way doors; left to the user per the
      push gate.

**Rationale:** Each excluded item is either decision-gated (needs a science modelling
choice this plan cannot make), data-gated (needs external data not in-repo), or a
one-way door. They are surfaced in Decision Gates so the user can unblock them, not
buried.

## Implementation Approach

**Technical Strategy:** Five independent branches, one per concern, each a
test-first unit landing as its own PR. Phase order is by leverage-per-risk: docs
(zero-risk, removes the misleading state) → land the joint lane (artifacts, no logic
risk) → issue #4 (small kernel fix, high value) → pipeline wiring (two-screen +
floor) → ACF harness. Phases are independent; none depends on another's code, only on
its own branch.

**Key Architectural Decisions:**

1. **Decision:** Branch-per-phase off `origin/main`, not off `feat/figure-vector`.
   - **Rationale:** The current branch `feat/figure-vector` is PR #47 (a figures PR).
     Committing closeout work there would pollute it. Untracked files (the joint
     ladder) survive a `git switch -c … origin/main`.
   - **Trade-offs:** Five small PRs vs one big one; cleaner review, more branch admin.
   - **Alternatives considered:** One mega-branch — rejected (couples unrelated review
     surfaces and risks the figures lane).

2. **Decision:** Issue #4 already uses **Option B (`force_multi=True` flag)** in code
   (`burstfit_joint.py:850,878`) — this plan adds the missing acceptance test, it does
   not re-decide the API.
   - **Rationale:** Option B (flag) was the conservative choice and is already
     implemented; default behavior is unchanged. The remaining gap is the regression
     test the issue's Acceptance section requires.
   - **Trade-offs:** Callers must opt in to commensurable N=1 evidence (already so).
   - **Alternatives considered:** Option A (drop the gate) — not taken; would be
     default-changing. (See D1.)

3. **Decision:** Wired pipeline results **attach to `final_results`**, gated behind a
   config flag, defaulting on but no-op when inputs are absent.
   - **Rationale:** Matches the existing `final_results` population pattern
     (`pipeline.py:258`); keeps wiring additive and reversible.
   - **Trade-offs:** None material; functions already return plain dicts.

4. **Decision:** The ACF / scintillation-bandwidth analysis (Phases 4 & 6) follows the
   **Nimmo et al. 2025** (arXiv:2406.11053, Nature; FRB 20221022A two coherent
   scintillation scales) and **two-screen scintillometry** (arXiv:2505.04576
   "Scintillometry of FRBs: Resolution effects in two-screen models", §5.1; the user's
   "Pleunis et al. 2025") recipe, reusing the repo's mature core rather than
   re-implementing.
   - **The recipe (what "follow them closely" means here):**
     1. **Mean-normalized full-spectrum ACF:** `ACF(δν)=⟨(I(ν)−⟨I⟩)/⟨I⟩ · (I(ν+δν)−⟨I⟩)/⟨I⟩⟩`
        at highest frequency resolution (2505.04576 Eq 4.10). Implemented in
        `scintillation/scint_analysis/analysis.py:calculate_acf` (`:209`).
     2. **Fit model = Lorentzian Eq 5.1:** `f(δν)=m²/(1+(δν/HWHM)²)+C`, amplitude `m²`,
        additive constant `C` left free. The repo's `lorentzian_component`
        (`analysis.py:33`) is exactly this; the generalized form `lor_gen`
        `m²/(1+|δν/γ|^(α+2))` (`:41`) is the Kolmogorov extension.
     3. **Decorrelation bandwidth Δν_dc = HWHM** of that fit (`calculate_acf:268-275`).
     4. **Modulation index m = √(peak correlation)** (the fitted `m`; 2505.04576 §5.1b,
        Eq 4.26 N-screen) — valid in the **absence of self-noise**, which the harness
        must subtract/flag (their explicit caveat).
     5. **Two scintillation scales (MW wide + host narrow):** isolate each by
        correlation-thresholding and fit a **wide** and a **narrow** Lorentzian,
        **omitting the lag-0 center** for the wide component (the center is contaminated
        by frequency-uncorrelated noise *and* by the finite intrinsic burst width —
        2505.04576 §4.2, Eqs 4.22-4.23). This is the **new capability** to add.
     6. **Finite-scintle uncertainty:** `N_scint=B/Δν`, fractional error `1/√N_scint`,
        combined in quadrature with the statistical error — already in
        `calculate_acf:279-296`.
     7. **Same-vs-two-screen test:** `ν_s = 1/(2π τ_s)` (Eq 4.15) — already in
        `scattering_scintillation_consistency` (`analysis.py:1098`).
   - **Rationale:** Most of this is already implemented and Nimmo-faithful; the only
     missing piece is the explicit two-component (wide+narrow, center-omitted) fit and
     consolidating it out of the disorganized drafted notebooks into a tested module.
   - **Trade-offs:** Reusing `calculate_acf`/`_fit_acf_models` over a fresh ACF keeps
     one ACF estimator in the codebase (ponytail: don't fork the method).
   - **Alternatives considered:** Re-deriving the ACF in `revalidation.py` — rejected;
     would duplicate the finite-scintle machinery and drift from the pipeline.

**Patterns to Follow:**
- Pathspec-scoped commits (never bare `git commit`) — `CLAUDE.md` S-009 + separate-lane
  rule. The working tree holds unrelated modified files (`galaxies/v2_0/sightline_budget.py`,
  `docs/entire-tracing-checkpoints.md`, etc.); every commit below names exact paths.
- Test oracle style — `tests/test_ne2025_floor.py:30` (analytic scaling) and
  `tests/test_association.py` (regression pinned to a computed value).
- Scintillation test fixtures — `scintillation/scint_analysis/tests/test_noise.py`
  (seeded synthetic spectra).

**Decision Gates (surfaced for the user; per the "decisions surfaced" scope choice):**
- **D1 — Issue #4 API:** already resolved **in code** to Option B (the `force_multi`
  flag at `burstfit_joint.py:850,878`). No decision needed; Phase 3 only adds the
  acceptance test. (Switching to Option A — dropping the gate to make multi the
  default — would be a separate, default-changing call; not proposed.)
- **D2 — Probabilistic host-DM model:** which `p(DM_cosmic|z)` (Macquart) form +
  report as posterior or upper limit. Blocks the negative-host-DM deliverable. Out of
  scope until chosen.
- **D3 — Geometric-delay localization forward model:** how Δt_geo + two-screen D_eff
  map to a position/screen constraint, and the χ² parameterization. Blocks the
  localization solver. Out of scope until chosen.
- **D4 — Manuscript sample (6 vs 8 energies) + nickname↔TNS:** the `@decision` ledger
  item. Out of scope.

## Implementation Phases

Each phase is test-first with real commands. All commits are pathspec-scoped. The
no-commit-to-protected-branch guard requires a feature branch; the push gate keeps
`git push`/PR with the user.

### Phase 1: Documentation reconciliation

**Objective:** Correct the four confirmed-stale references so docs match code. The
"test" for a doc fix is a grep that matches the stale text *before* and is empty
*after*.

**Tasks:**
- [ ] **Branch:** `git switch -c docs/reconcile-stale-refs origin/main`
- [ ] **Write the failing check** (current stale state): confirm each stale string is
      present.
  ```bash
  rg -n 'batch_runner\.py:262' docs/codetection-science-plan.md docs/rse/specs/plan-manuscript-completion.md
  rg -n 'analysis_logic\.py:110' docs/codetection-science-plan.md
  rg -n 'NOT yet committed' analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md
  rg -n 'files TBD' docs/architecture/inventory.md
  ```
  → expect each to PRINT matches (proves the stale refs exist).
- [ ] **Fix `docs/codetection-science-plan.md:51`** — replace the scint-stub item:
  - From: ``3. **Scintillation campaign tooling** — fix `flits/batch` scint config-gen stub (`batch_runner.py:262,275`, `# TODO: Add scintillation config generation`) so the mature scint pipeline runs over all 12.``
  - To: ``3. **Scintillation campaign over the remaining 9** — the code is done (`flits/batch/batch_runner.py` `discover_scint_configs`/`_run_scintillation_analysis`, test `test_scint_config_discovery.py`); what remains is hand-tuned RFI/window configs for the 9 unmeasured bursts.``
- [ ] **Fix `docs/codetection-science-plan.md:55`** — drop the resolved placeholder:
  - From: ``... `flits/` wrapper consolidation; τ(ν) batch placeholder (`analysis_logic.py:110`); `crossmatching/` geometric-delay localization remains unbuilt ...``
  - To: ``... `flits/` wrapper consolidation; `crossmatching/` geometric-delay localization solver remains unbuilt (`compute_geometric_delay` exists; the D_eff χ² forward model does not) ...``
- [ ] **Fix `docs/rse/specs/plan-manuscript-completion.md:71`** — reword the stub cell:
  - From: ``... other 9: **scint config-generation not yet run (stub at `flits/batch/batch_runner.py:262`)** — deferred, NOT unsuitable``
  - To: ``... other 9: **scint pipeline code is wired (`batch_runner.py` `discover_scint_configs`); the 9 await hand-tuned RFI/window configs** — deferred, NOT unsuitable``
- [ ] **Fix `analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md:60`** — the module is
      committed; change ``Module: scattering/scat_analysis/burstfit_joint.py (NOT yet committed).`` →
      ``Module: scattering/scat_analysis/burstfit_joint.py (committed).``
- [ ] **Fix `docs/architecture/inventory.md:261-264`** — fill the TBD counts. First
      derive them:
  ```bash
  for b in hamilton phineas whitney oran; do
    printf '%s: ' "$b"; ls scintillation/notebooks/"$b"/*.ipynb 2>/dev/null | wc -l
  done
  ```
  Then replace each ``- (files TBD)`` with ``- N notebook(s)`` using the counts (if a
  dir is absent, write ``- (no notebook dir in repo)``).
- [ ] **Run it, watch it pass:** re-run the Phase-1 grep block → expect each command to
      print NOTHING (zero matches), confirming the stale refs are gone.
- [ ] **Commit:** `git commit -m "docs: reconcile stale refs to resolved code (survey §6)" -- docs/codetection-science-plan.md docs/rse/specs/plan-manuscript-completion.md analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md docs/architecture/inventory.md`

**Dependencies:** none.

**Verification:**
- [ ] `rg -n 'batch_runner\.py:262|analysis_logic\.py:110' docs/` prints nothing.
- [ ] `rg -n 'NOT yet committed' analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md`
      prints nothing.
- [ ] `rg -n 'files TBD' docs/architecture/inventory.md` prints nothing.

### Phase 2: Land the joint-ladder campaign as artifacts

**Objective:** Commit the untracked gain-marginal ladder (scripts + JSONs) with a
runnable, tested diagnostic. This does **not** close issue #4 (that is Phase 3).

**Tasks:**
- [ ] **Branch:** `git switch -c analysis/land-joint-ladder origin/main` (untracked
      `joint_ladder/` files carry over).
- [ ] **Make `_s2verdict.py` importable + runnable** — wrap its top-level body in
      functions and a guard so it can be tested and CLI-run.
  - File: `analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py`
  - Add at top: `def load_fits(dirpath): ...` (the current `:18-24` glob/parse) and
    `def verdicts(grid): ...` (the current `:26-73` comparison), then:
  ```python
  if __name__ == "__main__":
      import os
      grid = load_fits(os.path.dirname(__file__))
      for line in verdicts(grid):
          print(line)
  ```
- [ ] **Write the failing test** for the cross-N verdict on a fixture.
  - File: `analysis/scattering-refit-2026-06/joint_ladder/tests/test_s2verdict.py` (new)
  ```python
  from analysis_joint_ladder import _s2verdict as v   # adjust import to package path

  def test_consistent_positive_delta_is_real():
      # two configs differing by one D component; ΔlnZ > 5 at every s2 → REAL
      grid = {"freya": {("C1","D1"): {1:-100.0,10:-100.0,100:-100.0},
                        ("C1","D2"): {1:-90.0,10:-90.0,100:-90.0}}}
      out = "\n".join(v.verdicts(grid))
      assert "REAL" in out

  def test_sign_flip_is_not_robust():
      grid = {"oran": {("C1","D1"): {1:-100.0,10:-100.0,100:-100.0},
                       ("C1","D2"): {1:-90.0,10:-105.0,100:-95.0}}}
      out = "\n".join(v.verdicts(grid))
      assert "NOT robust" in out
  ```
- [ ] **Run it, watch it fail:** `pytest analysis/scattering-refit-2026-06/joint_ladder/tests/test_s2verdict.py -v`
      → FAIL (`verdicts`/`load_fits` not yet factored out).
- [ ] **Implement** the refactor above (move the script body into `load_fits`/`verdicts`).
- [ ] **Run it, watch it pass:** same `pytest` command → PASS.
- [ ] **Add a `README.md`** in `joint_ladder/` (≤15 lines): what the ladder is
      (gain-marginal, 12/12 incl. casey), how it differs from the committed
      `joint_json/` (canonical c0/γ, 11/12), and that `_s2verdict.py` is a cross-N
      robustness *diagnostic*, not the issue-#4 fix.
- [ ] **Commit (pathspec-scoped to the ladder only):**
  ```bash
  git add analysis/scattering-refit-2026-06/joint_ladder/
  git status --short -- analysis/scattering-refit-2026-06/joint_ladder/   # confirm ONLY ladder files staged
  git commit -m "analysis: land gain-marginal joint ladder + tested s2verdict diagnostic" -- analysis/scattering-refit-2026-06/joint_ladder/
  ```

**Dependencies:** none (independent of Phase 1).

**Verification:**
- [ ] `git ls-files analysis/scattering-refit-2026-06/joint_ladder/ | wc -l` ≥ 100.
- [ ] `python analysis/scattering-refit-2026-06/joint_ladder/_s2verdict.py` prints
      verdict lines without error.
- [ ] `git status --short` shows the unrelated separate-lane files (e.g.
      `galaxies/v2_0/sightline_budget.py`) still unstaged/untouched.

### Phase 3: Close issue #4 — N=1 evidence commensurability

**Objective:** Add the missing acceptance test. The `force_multi` routing already
exists (`burstfit_joint.py:850,878`); this phase pins N=1-via-multi to a brute-force
Gaussian-evidence oracle and confirms `gain_s2` is fixed (not `None`) on cross-N
comparisons. If the oracle reveals a residual normalization offset, fix it in
`_gain_marginal_multi_band`; if it already matches, the test simply codifies #4's
acceptance and the issue closes.

**Tasks:**
- [ ] **Branch:** `git switch -c fix/issue-4-n1-commensurable origin/main`
- [ ] **Read** `scattering/scat_analysis/burstfit_joint.py:190` (`_gain_marginal_multi_band`
      signature) and `tests/test_gain_marginal_multi_band.py` (oracle style) before
      writing — confirm the exact kwargs (`n_comp`, `gain_s2`, per-band data/model).
- [ ] **Write the failing test** asserting N=1-via-multi equals a brute-force 1D
      Gaussian-marginal evidence at small T, fixed `gain_s2`.
  - File: `tests/test_issue4_commensurable.py` (new)
  ```python
  import numpy as np
  from scattering.scat_analysis import burstfit_joint as bj

  def _brute_force_logZ(d, m, sigma, s2, n=4001):
      # marginal evidence of d = g*m + noise, g ~ N(0, s2): integrate over g
      g = np.linspace(-6*np.sqrt(s2), 6*np.sqrt(s2), n)
      ll = -0.5*((d[:,None]-g[None,:]*m[:,None])**2).sum(0)/sigma**2 \
           - 0.5*len(d)*np.log(2*np.pi*sigma**2)
      lp = -0.5*g**2/s2 - 0.5*np.log(2*np.pi*s2)
      from scipy.special import logsumexp
      return logsumexp(ll+lp) + np.log(g[1]-g[0])

  def test_n1_multi_matches_brute_force_small_T():
      rng = np.random.default_rng(0)
      m = np.array([1.0, 0.7, 0.4]); sigma = 0.5; s2 = 2.0
      d = 1.3*m + rng.normal(0, sigma, size=m.size)
      ref = _brute_force_logZ(d, m, sigma, s2)
      got = bj._gain_marginal_multi_band(d, m, sigma, n_comp=1, gain_s2=s2)  # confirm kwargs in read step
      assert got == np.testing.approx(ref, abs=1e-3)
  ```
- [ ] **Run it, watch it fail-or-pass:** `pytest tests/test_issue4_commensurable.py -v`.
      If it FAILS only on the assumed kwarg names, fix the test call to the real
      signature from the read step (the flag/gate already exist — do **not** re-add
      them). If it FAILS on the *value* (a scale offset of order `T·ln(2πσ²)`), that is
      the real normalization bug → fix it in `_gain_marginal_multi_band` (`:190`) so
      the N=1 multi-path evidence includes the proper `−0.5·T·ln(2πσ²)` term, matching
      the oracle.
- [ ] **Confirm `gain_s2` is fixed on cross-N comparisons:** add an assert that calling
      with `gain_s2=None` (profile) and a fixed `gain_s2` give different lnZ, and that
      the ladder uses the fixed value (the issue's "pair with a fixed `gain_s2`" note).
- [ ] **Run it, watch it pass:** `pytest tests/test_issue4_commensurable.py -v` → PASS.
- [ ] **Run the validation contract** on a joint fit to confirm no level dropped
      (per `.cursor/rules/AGENT_CONFIGURATION_FLITS.md`): dispatch the `fit-validation`
      subagent on a `force_multi=True` N=1 result, or run `pytest tests/test_recovery_campaign.py -m slow` if data present.
- [ ] **Commit:** `git commit -m "fix(joint): route N=1 through gain-marginal multi path for commensurable lnZ (#4)" -- scattering/scat_analysis/burstfit_joint.py tests/test_issue4_commensurable.py`

**Dependencies:** none.

**Verification:**
- [ ] `pytest tests/test_issue4_commensurable.py -v` passes.
- [ ] `pytest tests/test_gain_marginal_multi_band.py -v` still passes (no regression).
- [ ] Fit-validation: a `force_multi=True` N=1 fit returns PASS/MARGINAL (not FAIL on a
      newly-introduced Level-1 gate).

### Phase 4: Wire two-screen consistency + emission size into the pipeline

**Objective:** Attach `scattering_scintillation_consistency`,
`interpret_modulation_index`, and `estimate_emission_region_size` outputs to
`ScintillationAnalysis.final_results`.

**Tasks:**
- [ ] **Branch:** `git switch -c feat/scint-pipeline-wiring origin/main`
- [ ] **Write the failing test** that a run attaches the new keys.
  - File: `scintillation/scint_analysis/tests/test_pipeline_wiring.py` (new)
  ```python
  from scint_analysis.analysis import interpret_modulation_index, scattering_scintillation_consistency

  def test_modulation_interpretation_keys():
      r = interpret_modulation_index(0.9, 0.05)
      assert {"interpretation", "emission_resolved", "resolution_regime"} <= set(r)

  def test_consistency_single_screen_flag():
      # τ·Δν tuned to the thin-screen relation → consistent
      r = scattering_scintillation_consistency(0.5, 0.318, C=1.0)  # 2π·τ·Δν ≈ 1
      assert "C_implied" in r and r["consistent"] in (True, False)
  ```
  (These pin the function contracts the wiring depends on; they fail if the funcs move
  or change shape.)
- [ ] **Run it, watch it fail/pass:** `pytest scintillation/scint_analysis/tests/test_pipeline_wiring.py -v`
      — the two unit tests pass against existing funcs; they guard the wiring contract.
- [ ] **Write the wiring test** (the actual new behavior):
  ```python
  def test_run_attaches_twoscreen(monkeypatch, tiny_scint_config):
      from scint_analysis import pipeline
      a = pipeline.ScintillationAnalysis(tiny_scint_config); a.run()
      comp = next(iter(a.final_results["components"].values()))
      assert "emission_size" in comp and "consistency" in comp
  ```
  → FAIL (keys absent).
- [ ] **Implement** at `scintillation/scint_analysis/pipeline.py:~268` (after the 2D
      fit, before the closing log at `:269`): for each component in `final_results`,
      pull `mod`/`bw`/`freq_mhz`/`scaling_index` from its `subband_measurements`, call
      `interpret_modulation_index`, `scattering_scintillation_consistency`, and
      `estimate_emission_region_size`, and store under `comp["emission_size"]`,
      `comp["consistency"]`, `comp["modulation"]`. Guard with
      `if comp.get("subband_measurements"):` so it is a no-op when inputs are absent.
      Per Decision 4, the `m` fed to `interpret_modulation_index` is √(ACF peak) and the
      `delta_nu_dc` is the HWHM, as the core already reports.
- [ ] **Wire the two-screen coherence constraint when two scales are present:** if the
      component carries both a wide (MW) and narrow (host) Δν (the Nimmo two-coherent-
      scales case, e.g. from Phase 6's `fit_two_screen_acf`, or two stored
      `subband` scales), call `two_screen_coherence_constraint(dnu_wide, dnu_narrow,
      freq_mhz, d_source_mpc)` (`analysis.py:1011`) and store under
      `comp["two_screen"]`. Otherwise omit the key (single-scale bursts).
- [ ] **Run it, watch it pass:** same `pytest` file → PASS.
- [ ] **Commit:** `git commit -m "feat(scint): wire two-screen consistency + emission size + coherence into pipeline output" -- scintillation/scint_analysis/pipeline.py scintillation/scint_analysis/tests/test_pipeline_wiring.py`

**Dependencies:** none.

**Verification:**
- [ ] `pytest scintillation/scint_analysis/tests/test_pipeline_wiring.py -v` passes.
- [ ] `rg -n 'estimate_emission_region_size|scattering_scintillation_consistency' scintillation/scint_analysis/pipeline.py`
      now shows callers (previously zero).

### Phase 5: Wire the NE2025 Galactic floor into the pipeline

**Objective:** Attach `galactic_floor` + a measured/floor extragalactic-excess ratio
to each burst's `final_results`.

**Tasks:**
- [ ] **Branch:** continue on `feat/scint-pipeline-wiring` (same wiring concern) or
      `git switch -c feat/scint-ne2025-floor origin/main` if landing separately.
- [ ] **Write the failing test** (analytic scaling oracle, mirroring
      `tests/test_ne2025_floor.py:30`, plus the attach contract).
  - File: `scintillation/scint_analysis/tests/test_floor_wiring.py` (new)
  ```python
  from astropy.coordinates import SkyCoord; import astropy.units as u
  from scint_analysis.floor_wiring import attach_galactic_floor  # new thin wrapper

  def test_excess_ratio_flags_extragalactic():
      coord = SkyCoord(ra=170*u.deg, dec=70*u.deg, frame="icrs")
      comp = {"scaling_index": 4.4, "subband_measurements":[{"freq_mhz":1405,"bw":2.7}]}
      attach_galactic_floor(comp, coord)
      assert comp["galactic_floor"]["DSA"]["bw_kHz"] > 0
      assert comp["extragalactic_excess"] is True   # measured Δν >> MW floor
  ```
- [ ] **Run it, watch it fail:** `pytest scintillation/scint_analysis/tests/test_floor_wiring.py -v`
      → FAIL (`floor_wiring` missing).
- [ ] **Implement** `scintillation/scint_analysis/floor_wiring.py` — a ≤25-line wrapper
      calling `scintillation.ne2025.query_ne2025_scint.galactic_floor(coord, ...)` and
      computing `extragalactic_excess = measured_bw_kHz < floor_bw_kHz` (smaller measured
      decorrelation bandwidth than the MW floor ⇒ excess scattering ⇒ extragalactic).
      Call it from `pipeline.py` next to the Phase-4 wiring, guarded by availability of
      burst sky coords + the optional `mwprop` dep (skip cleanly if absent, like the
      existing floor test).
- [ ] **Run it, watch it pass:** same `pytest` file → PASS.
- [ ] **Commit:** `git commit -m "feat(scint): wire NE2025 Galactic floor + extragalactic-excess flag" -- scintillation/scint_analysis/floor_wiring.py scintillation/scint_analysis/pipeline.py scintillation/scint_analysis/tests/test_floor_wiring.py`

**Dependencies:** Phase 4 if landed on the same branch (shared pipeline edit region).

**Verification:**
- [ ] `pytest scintillation/scint_analysis/tests/test_floor_wiring.py -v` passes.
- [ ] `pytest tests/test_ne2025_floor.py -v` still passes (no regression to the floor
      module).

### Phase 6: ACF re-validation harness — Nimmo & Pleunis 2025 bandwidth method

**Objective:** Build + unit-test an RFI/off-pulse/self-noise harness that re-measures
Δν on cleaned data **following the Nimmo & Pleunis 2025 recipe** (Decision 4): reuse
`calculate_acf` (mean-normalized full-spectrum ACF + finite-scintle errors), add the
**two-component wide+narrow Lorentzian fit with the lag-0 center omitted**, and report
`m=√(peak)`. Consolidate the method out of the disorganized drafted notebooks into one
tested module. (The real-data run over casey/freya/wilhelm is data-gated — manual.)

**Tasks:**
- [ ] **Branch:** `git switch -c feat/acf-revalidation-harness origin/main`
- [ ] **Survey the drafted notebooks first** (the method to consolidate): read the ACF
      cells in `scintillation/notebooks/scintillation_analysis.ipynb`,
      `scintillation/notebooks/debug/wilhelm_manual.ipynb`, and
      `scintillation/chime_acfs/pickle.ipynb`. Port their *method* (not their
      organization); where a notebook step already lives in `analysis.py`
      (`calculate_acf`, `_fit_acf_models`, `lorentzian_component`), call it instead of
      copying. (Some equivalents live only on arc/CANFAR — note any method detail not
      reproducible in-repo as a manual follow-up rather than guessing.)
- [ ] **Write the failing test** with seeded synthetic spectra (mirror
      `tests/test_noise.py` fixtures) — covering RFI flagging, off-pulse masking, the
      single-screen HWHM Δν via `calculate_acf`, and the **two-screen** wide+narrow
      recovery (the Nimmo/Pleunis fidelity oracle).
  - File: `scintillation/scint_analysis/tests/test_revalidation.py` (new)
  ```python
  import numpy as np
  from scint_analysis.revalidation import (
      off_pulse_mask, rfi_flag, revalidate_dnu, fit_two_screen_acf,
  )

  def test_rfi_spike_flagged():
      rng = np.random.default_rng(0)
      spec = rng.normal(10, 1, 256); spec[128] = 80.0   # one RFI channel
      flags = rfi_flag(spec, n_sigma=5)
      assert flags[128] and flags.sum() <= 3

  def test_offpulse_mask_excludes_burst():
      prof = np.r_[np.ones(40), 50*np.ones(8), np.ones(40)]  # burst in the middle
      m = off_pulse_mask(prof, k=3.0)
      assert not m[44] and m[0] and m[-1]

  def test_clean_dnu_is_hwhm_via_calculate_acf():
      # single-screen: Δν recovered as the Lorentzian HWHM (Pleunis Eq 5.1)
      rng = np.random.default_rng(1)
      white = rng.normal(0, 1, 266)
      corr = np.convolve(white, np.ones(10)/10, mode="valid")[:256]  # Δν ~ 10 chan
      spec = 100 + 20*corr
      dnu = revalidate_dnu(spec, channel_width_mhz=0.39)
      assert 2.0 < dnu < 6.0   # ~10 chan * 0.39 MHz, generous band

  def test_two_screen_wide_and_narrow_recovered():
      # inject two decorrelation scales (wide MW + narrow host); the center-omitted
      # wide fit + thresholded narrow fit recover both within tolerance, and the
      # combined modulation index is sqrt(peak) (Pleunis 2505.04576 §5.1, Eq 4.26)
      rng = np.random.default_rng(2)
      n = 4096
      wide = np.convolve(rng.normal(0,1,n+80), np.ones(80)/80, "valid")[:n]   # broad
      narrow = np.convolve(rng.normal(0,1,n+6), np.ones(6)/6, "valid")[:n]    # fine
      spec = 100*(1 + 0.6*wide)*(1 + 0.6*narrow)                              # 2-screen product
      res = fit_two_screen_acf(spec, channel_width_mhz=0.0305)  # DSA-like fine res
      assert res["dnu_wide_mhz"] > 5 * res["dnu_narrow_mhz"]     # two distinct scales
      assert 0.0 < res["m_total"] <= 2.0 and res["center_omitted"] is True
  ```
- [ ] **Run it, watch it fail:** `pytest scintillation/scint_analysis/tests/test_revalidation.py -v`
      → FAIL (`revalidation` module missing).
- [ ] **Implement** `scintillation/scint_analysis/revalidation.py`:
  - `rfi_flag` (robust-σ channel outliers via `noise._robust_std`), `off_pulse_mask`
    (robust-σ threshold on the time profile).
  - `revalidate_dnu` — mask RFI channels, off-pulse-subtract, then call
    `analysis.calculate_acf` and read the **HWHM** as Δν (do **not** roll a new ACF).
  - `fit_two_screen_acf` — the new Nimmo/Pleunis capability: compute the mean-normalized
    ACF (via `calculate_acf`), fit a **wide** Lorentzian `m²/(1+(δν/γ)²)+C` **with the
    lag-0 bin excluded** (and the channels below a correlation threshold), then fit a
    **narrow** Lorentzian to the residual central component; return
    `{dnu_wide_mhz, dnu_narrow_mhz, m_wide, m_narrow, m_total=√peak, center_omitted:True}`.
    Reuse `analysis.lorentzian_component` and the existing lmfit fit pattern
    (`_fit_acf_models`, `analysis.py:612`).
- [ ] **Run it, watch it pass:** same `pytest` file → PASS.
- [ ] **Commit:** `git commit -m "feat(scint): ACF re-validation harness + two-screen Nimmo/Pleunis bandwidth fit" -- scintillation/scint_analysis/revalidation.py scintillation/scint_analysis/tests/test_revalidation.py`

**Dependencies:** none (reuses `analysis.py`, already present).

**Verification:**
- [ ] `pytest scintillation/scint_analysis/tests/test_revalidation.py -v` passes,
      including `test_two_screen_wide_and_narrow_recovered` (the method-fidelity oracle).
- [ ] `rg -n 'calculate_acf|lorentzian_component' scintillation/scint_analysis/revalidation.py`
      confirms it reuses the core (no forked ACF estimator).
- [ ] Manual (data-gated): run `revalidate_dnu`/`fit_two_screen_acf` over
      casey/freya/wilhelm raw spectra and compare to the stored `delta_nu_dc` —
      certifies diffractive vs artifact, and looks for a second (host) scale.

## Success Criteria

### Automated Verification

- [ ] `rg -n 'batch_runner\.py:262|analysis_logic\.py:110' docs/` → empty.
- [ ] `rg -n 'NOT yet committed' analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md`
      → empty; `rg -n 'files TBD' docs/architecture/inventory.md` → empty.
- [ ] `git ls-files analysis/scattering-refit-2026-06/joint_ladder/ | wc -l` ≥ 100.
- [ ] `pytest analysis/scattering-refit-2026-06/joint_ladder/tests/test_s2verdict.py` passes.
- [ ] `pytest tests/test_issue4_commensurable.py tests/test_gain_marginal_multi_band.py` passes.
- [ ] `pytest scintillation/scint_analysis/tests/test_pipeline_wiring.py scintillation/scint_analysis/tests/test_floor_wiring.py scintillation/scint_analysis/tests/test_revalidation.py` passes.
- [ ] Method-fidelity oracle passes:
      `pytest scintillation/scint_analysis/tests/test_revalidation.py::test_two_screen_wide_and_narrow_recovered`
      (injected wide+narrow scales recovered, center omitted — Nimmo/Pleunis 2025).
- [ ] `pytest tests/test_ne2025_floor.py` still passes (no regression).
- [ ] `ruff check .` clean on all touched files.

### Manual Verification

- [ ] Phase 3: a `force_multi=True` N=1 joint fit and an N=2 fit have lnZ values whose
      difference is a sane Occam factor (not an offset of order T·ln(2πσ²)).
- [ ] Phase 4/5: spot-check one burst's `final_results` — `emission_size`,
      `consistency`, `galactic_floor`, `extragalactic_excess` are physically sane.
- [ ] Phase 6: the harness, run on casey/freya/wilhelm raw data, agrees with or
      overturns the stored Δν — and the verdict (diffractive vs RFI) is defensible.
- [ ] The four new branches are clean, single-concern, and ready for the user to push.

### Reproducibility & Correctness (research code)

- [ ] Phase 3 oracle uses a fixed seed (`default_rng(0)`) and an analytic 1D-Gaussian
      marginal as the reference; tolerance `abs=1e-3` justified by the trapezoid grid
      (n=4001 over ±6σ).
- [ ] Phase 5 floor wiring asserts the analytic τ∝ν^−α / Δν∝ν^+α scaling
      (`test_ne2025_floor.py:30` pattern).
- [ ] Phase 6 fidelity oracle uses a fixed seed (`default_rng(2)`) and injected
      decorrelation scales as the known truth; Δν = HWHM and m = √peak per Pleunis
      Eq 5.1 / Nimmo Eq 4.26; tolerance is the "two distinct scales" ratio (wide > 5×
      narrow), justified by the order-of-magnitude separation the method assumes.
- [ ] All new tests run in the `flits` conda env (`pip install -e ".[nested,perf]"`).

## Testing Strategy

Unit tests are written test-first in each phase. Additional coverage:

**Integration Tests:**
- [ ] Phase 4+5: one end-to-end `ScintillationAnalysis.run` on a tiny config asserting
      all wired keys appear together in `final_results`.
- [ ] Phase 3: confirm `fit_joint_scattering(force_multi=True)` round-trips through the
      existing joint driver without breaking the `gain_s2=None` default path.

**Manual Testing:**
- [ ] Phase 6 real-data run (casey/freya/wilhelm) — needs the multiscale result JSONs /
      raw spectra (external; `DATA_SOURCES.md`).

**Test Data Requirements:**
- Synthetic, seeded fixtures for all unit tests (no external data).
- A `tiny_scint_config` fixture for the pipeline-wiring tests — reuse the smallest
  existing config under `configs/` or construct a 16-channel synthetic ACF input.

## Migration Strategy

No data migration. `force_multi` defaults `False`, so existing single-component
callers are unchanged. Pipeline wiring is additive (new keys; guarded no-op when
inputs absent). **Rollback:** each phase is a single pathspec-scoped commit on its own
branch — `git revert` the commit or drop the branch. **Backward compatibility:**
existing `final_results` consumers see only new keys, never changed ones.

## Risk Assessment

1. **Risk:** Phase 3 oracle disagrees because `_gain_marginal_multi_band`'s kwargs
   differ from the assumed signature.
   - **Likelihood:** Medium — **Impact:** Low (test fails loudly).
   - **Mitigation:** The mandatory read step before the test confirms the signature;
     the oracle math (1D Gaussian marginal) is independent of internal layout.
2. **Risk:** A pathspec-scoped commit accidentally sweeps a separate-lane file.
   - **Likelihood:** Low — **Impact:** Medium.
   - **Mitigation:** Every commit names exact paths; Phase 2 adds a `git status --short`
     check that the separate-lane files stay unstaged.
3. **Risk:** NE2025 floor needs the optional `mwprop` dep, absent in CI.
   - **Likelihood:** Medium — **Impact:** Low.
   - **Mitigation:** Guard the wiring + test with `importorskip` like
     `test_ne2025_floor.py`; the pipeline path is a clean no-op without it.

## Edge Cases and Error Handling

1. **Case:** A burst has no `subband_measurements` (non-detection).
   - **Expected:** Phase 4/5 wiring is a no-op; no key added; no exception.
   - **Implementation:** `if comp.get("subband_measurements"):` guard.
2. **Case:** `inventory.md` TBD dir does not exist on disk.
   - **Expected:** write ``- (no notebook dir in repo)`` rather than a count.
   - **Implementation:** the Phase-1 `ls … 2>/dev/null | wc -l` returns 0 → branch on it.
3. **Error:** `galactic_floor` raises for an out-of-Galaxy sightline.
   - **Handling:** catch in `floor_wiring`, set `comp["galactic_floor"] = None` + log.

## Documentation Updates

- [ ] Phase 2 `joint_ladder/README.md` (ladder vs `joint_json/`; diagnostic role).
- [ ] After Phase 3, append a one-line resolution note to issue #4 and to
      `JOINT_FIT_STATE.md` (the `force_multi` entry point).
- [ ] Docstrings on the new `floor_wiring` and `revalidation` modules.

## Open Questions

*(none — Decision Gates D1 resolved to Option B; D2/D3/D4 are explicitly out of scope
under "What We're NOT Doing" and surfaced for the user, not blocking this plan.)*

---

## References

**Research Documents:**
- [Research: Survey of apparently-incomplete work in FLITS](research-incomplete-work-survey.md)

**Related Plans / Science:**
- [CHIME/DSA Co-detection Science Plan](../../codetection-science-plan.md) §C
- [Plan: Manuscript completion](plan-manuscript-completion.md)
- [Joint fit build state](../../../analysis/scattering-refit-2026-06/JOINT_FIT_STATE.md)

**Files Analyzed:**
- `flits/batch/batch_runner.py`, `flits/batch/analysis_logic.py`
- `scattering/scat_analysis/burstfit_joint.py`, `.../burstfit.py`
- `scintillation/scint_analysis/{analysis,pipeline,consistency,noise}.py`
- `scintillation/ne2025/query_ne2025_scint.py`
- `crossmatching/{association,toa_crossmatch}.py`
- `analysis/scattering-refit-2026-06/joint_ladder/{_ladder,_s2verdict,_figs}.py`
- `docs/{codetection-science-plan.md,architecture/inventory.md}`,
  `docs/rse/specs/plan-manuscript-completion.md`

**External Documentation:**
- Issue #4: `gh issue view 4` (gain-marginal commensurability)
- `.cursor/rules/AGENT_CONFIGURATION_FLITS.md` (validation contract)
- **Nimmo et al. 2025**, "Magnetospheric origin of a fast radio burst constrained
  using scintillation," Nature (FRB 20221022A; two mutually-coherent scintillation
  scales → emission-region size) — arXiv:2406.11053. Already referenced in
  `scintillation/scint_analysis/analysis.py` (Eqs 22-23, 26, 27).
- **arXiv:2505.04576**, "Scintillometry of Fast Radio Bursts: Resolution effects in
  two-screen models" (the user's "Pleunis et al. 2025") — the ACF / scintillation-
  bandwidth recipe followed in Phases 4 & 6: mean-normalized full-spectrum ACF
  (Eq 4.10), Lorentzian fit `m²/(1+(δν/HWHM)²)+C` (Eq 5.1), Δν = HWHM, m = √peak,
  two-component wide+narrow fit with the center omitted (Eqs 4.22-4.23), ν_s=1/(2πτ_s)
  (Eq 4.15).

---

## Review History

### Version 1.0 — 2026-06-24
- Initial plan created from `research-incomplete-work-survey.md`.
