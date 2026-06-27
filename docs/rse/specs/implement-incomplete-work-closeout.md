# Implementation Summary: Incomplete-work closeout (issue #4 + scintillation wiring + Nimmo/Pleunis ACF harness)

---
**Date:** 2026-06-24
**Author:** AI Assistant
**Status:** Complete — manual verification done. #54, #55 **merged**; #53 closed (superseded by merged #56, identical content); follow-up #58 (multi-component auto-selection) **open**.
**Plan Reference:** [plan-incomplete-work-closeout.md](plan-incomplete-work-closeout.md)

---

## Overview

Implemented the agent-doable subset of the incomplete-work closeout plan: the
issue-#4 N=1 evidence-commensurability acceptance tests (Phase 3), the two-screen /
emission-size / consistency interpretation wiring into the scintillation pipeline
(Phase 4), the NE2025 Galactic-floor + extragalactic-excess wiring (Phase 5), and a
new ACF re-validation harness following Nimmo et al. 2025 and Pleunis 2025 (Phase 6).
Phases 1 (doc reconciliation) and 2 (land the joint ladder) were **skipped** because a
concurrent session had already done equivalent work (see Deviations).

**Implementation Duration:** 2026-06-24 (single session).

**Final Status:** ✅ Complete for the four implemented phases. Each landed as a
pathspec-scoped commit on its own feature branch, every test passes under the real
`flits` conda env, and each phase was reviewed by Codex (gpt-5.5 high).

## Plan Adherence

**Plan Followed:** [plan-incomplete-work-closeout.md](plan-incomplete-work-closeout.md)

**Deviations from Plan:**

- **Deviation 1 — Phases 1 & 2 skipped (done by a concurrent lane).**
  - **Reason:** A concurrent session committed the Phase-1 doc reconciliation and the
    ADR-0003/0004 work (`f03fab9` docs reconcile, `1c87ef5` s² PBF guard, `279f4c6`
    ADR-0004 floor, `5080f8a` decision map) and refactored `_s2verdict.py`. Re-doing
    them would duplicate or conflict with that lane.
  - **Impact:** None on the implemented phases; the doc/ladder concerns are covered by
    the concurrent lane's commits, which now sit on the shared base.

- **Deviation 2 — Branched off `feat/figure-vector`, not `origin/main`.**
  - **Reason:** The plan's Decision 1 wanted branch-per-phase off `origin/main`, but a
    `git switch -c … origin/main` aborts: the separate-lane file
    `galaxies/v2_0/sightline_budget.py` has uncommitted changes that conflict with
    `origin/main`. Branching off the current base preserves the working tree.
  - **Impact:** The feature branches descend from the (concurrent-lane-advanced)
    figures base rather than `origin/main`. See "Branch state" below.

- **Deviation 3 — Phase 3 dropped the flaky fixed-vs-profiled-s² test; added a routing
    regression test instead.**
  - **Reason:** The planned `gain_s2`-profile-vs-fixed assertion was flat for random
    data (gilding beyond the issue's two acceptance bullets). Codex's review of the
    brute-force tests flagged that they prove the N=1 *algebra* but not that
    `fit_joint_scattering(force_multi=True)` *routes* N=1 through the multi path.
  - **Impact:** Stronger coverage: the new test stubs `dynesty.NestedSampler` and
    asserts the router (`burstfit_joint.py` gate) hands `_JointLogLikelihoodGainMulti`
    + `JOINT_PARAM_NAMES_GAIN_MULTI(1,1)` to the sampler at N=1, with a
    `force_multi=False` contrast.

- **Deviation 4 — Phase 4 sources external science inputs from `config['source']`, and
    is tested as a pure function rather than via a full `ScintillationAnalysis.run()`.**
  - **Reason:** The plan said to call `scattering_scintillation_consistency` /
    `estimate_emission_region_size` but did not say where their inputs (τ, screen
    distance, source distance) come from — they are *not* in `final_results`, and
    `bursts.yaml` has no distance/redshift. A full pipeline run needs gitignored raw
    spectra. So the wiring reads τ/distances from an optional `config['source']` block
    (each call gated on a finite-positive value → clean no-op without it), and the
    wiring is a pure `attach_scintillation_interpretation(final_results, config)`
    function tested directly on a synthetic `final_results`.
  - **Impact:** `modulation` always attaches (m is intrinsic); `consistency`,
    `emission_size`, and `two_screen` attach only when their science input is present.
    Real emission-size/two-screen output remains decision-gated (no in-repo distances).

- **Deviation 5 — Phase 6 ports Nimmo's ACF estimator instead of reusing
    `analysis.calculate_acf`.**
  - **Reason:** The user explicitly asked to "follow the methods of Nimmo and Pleunis
    et al. 2025 as closely as you can" and provided the Nimmo et al. 2025 release
    (`scint_funcs.py`, the Nature PDF, supp, extended figures, arXiv TeX, and the
    FRB 20221022A data). A *re-validation* harness should be independent of the
    pipeline's own estimator — reusing `calculate_acf` would defeat the cross-check.
  - **Impact:** `revalidation.py` ports Nimmo's `autocorr` (as `_acf_masked` /
    `_mean_normalized_acf`), `lorentz_w_c`, `doublelorentz_w_c`, `res`, and
    `emission_size` with attribution. The plan's "reuse the core (no forked ACF
    estimator)" verification is intentionally not met. **Sub-deviation:** `first_lag`
    defaults to dropping only the universal lag-0 self-noise spike; Nimmo additionally
    drops lag 1 (an upchannelization artifact specific to CHIME's fine channels), which
    is opt-in via `first_lag=2` so the harness stays telescope-agnostic for DSA native
    resolution. Per Codex's review.

## Phases Completed

### Phase 3: Close issue #4 — N=1 evidence commensurability
- ✅ **Status:** Complete — **Completion Date:** 2026-06-24
- **Summary:** Added `tests/test_issue4_commensurable.py` (3 tests): two brute-force
  Gaussian-evidence oracles pinning `_gain_marginal_multi_band` at N=1 (full
  `−0.5·T·ln(2πσ²)` norm + Occam term) and proving N=1/N=2 share an additive scale,
  plus a routing test stubbing `dynesty.NestedSampler` to prove
  `fit_joint_scattering(force_multi=True)` routes N=1 through the multi-component gain
  path. The `force_multi` flag already existed; this codifies the issue's acceptance.

### Phase 4: Wire two-screen consistency + emission size into the pipeline
- ✅ **Status:** Complete — **Completion Date:** 2026-06-24
- **Summary:** Added `analysis.attach_scintillation_interpretation`, called once in
  `pipeline.run` after `analyze_scintillation_from_acfs`. Attaches `modulation`
  (`interpret_modulation_index`), and, gated on `config['source']`, `consistency`
  (`scattering_scintillation_consistency`), `emission_size`
  (`estimate_emission_region_size`), and `two_screen` (`two_screen_coherence_constraint`).

### Phase 5: Wire the NE2025 Galactic floor into the pipeline
- ✅ **Status:** Complete — **Completion Date:** 2026-06-24
- **Summary:** Added `floor_wiring.py` (`attach_galactic_floor` / `_all`,
  `extragalactic_excess`), called from `pipeline.run` when `config['source']` carries
  `ra_deg`/`dec_deg`. Lazily imports `query_ne2025_scint` (optional `mwprop`/`pygedm`)
  so every failure path is a clean no-op. Flags a measured Δν below the MW floor as an
  extragalactic (host/intervening) screen.

### Phase 6: ACF re-validation harness — Nimmo & Pleunis 2025 bandwidth method
- ✅ **Status:** Complete (harness + unit tests) — **Completion Date:** 2026-06-24
- **Summary:** Added `revalidation.py`: `rfi_flag`, `off_pulse_mask`, `revalidate_dnu`
  (single-screen Δν = Lorentzian HWHM), and the new `fit_two_screen_acf` (wide MW +
  narrow host double-Lorentzian, center omitted), porting Nimmo's `autocorr` /
  `lorentz_w_c` / `doublelorentz_w_c` / `res` / `emission_size`. The real-data run over
  casey/freya/wilhelm is data-gated (manual; see Remaining Work).

## Files Modified

**Created:**
- `tests/test_issue4_commensurable.py` — Phase 3 issue-#4 acceptance + routing tests.
- `scintillation/scint_analysis/tests/test_pipeline_wiring.py` — Phase 4 wiring tests.
- `scintillation/scint_analysis/floor_wiring.py` — Phase 5 NE2025 floor wrapper.
- `scintillation/scint_analysis/tests/test_floor_wiring.py` — Phase 5 floor tests.
- `scintillation/scint_analysis/revalidation.py` — Phase 6 Nimmo/Pleunis ACF harness.
- `scintillation/scint_analysis/tests/test_revalidation.py` — Phase 6 harness tests.
- `docs/rse/specs/implement-incomplete-work-closeout.md` — this summary.

**Modified:**
- `scintillation/scint_analysis/analysis.py` — added `attach_scintillation_interpretation`
  (the post-edit formatter also reflowed the previously-unformatted file: single→double
  quotes + line wraps; no logic change, 74 scint tests still pass).
- `scintillation/scint_analysis/pipeline.py` — two wiring calls after the ACF fit.

**Deleted:** No files deleted.

## Key Changes Summary

1. **Issue #4 acceptance (Phase 3)** — `tests/test_issue4_commensurable.py`.
2. **Two-screen interpretation wiring (Phase 4)** —
   `analysis.attach_scintillation_interpretation`, `pipeline.run`.
3. **NE2025 floor wiring (Phase 5)** — `floor_wiring.py`, `pipeline.run`.
4. **Nimmo/Pleunis ACF harness (Phase 6)** — `revalidation.py`.

## Verification Results

### Automated Verification

Run under the **real** `flits` env (`/Users/jakobfaber/.conda/envs/flits/bin/python` —
see Issue 2):

- ✅ `pytest tests/test_issue4_commensurable.py tests/test_gain_marginal_multi_band.py` — 12 passed.
- ✅ `pytest scintillation/scint_analysis/tests/test_pipeline_wiring.py` — 5 passed.
- ✅ `pytest scintillation/scint_analysis/tests/test_floor_wiring.py` — 6 passed (incl. the
  real-floor `importorskip` test, which runs because `mwprop` is present in `flits`).
- ✅ `pytest scintillation/scint_analysis/tests/test_revalidation.py` — 6 passed (incl.
  the two-screen fidelity oracle and the `first_lag=2` Nimmo-CHIME option).
- ✅ `pytest scintillation/scint_analysis/tests/` — 74 passed (no regression from the
  `analysis.py` reflow / wiring).
- ✅ `ruff check` clean on every file I authored (`floor_wiring.py`, `revalidation.py`,
  and the new test files modulo the deliberate `sys.path` E402, matching `test_noise.py`).
- ✅ Each phase reviewed by Codex (gpt-5.5 high); all blocking findings addressed
  (routing test, finite-positive guards, non-finite band-selection guard, lag
  convention + fit-success checks + honest `m_total` docstring).

### Manual Verification (pending — human required)

- [ ] Phase 3: confirm a `force_multi=True` N=1 joint fit and an N=2 fit have lnZ whose
      difference is a sane Occam factor (not an offset of order T·ln(2πσ²)).
- [ ] Phase 4/5: spot-check one burst's `final_results` with a populated
      `config['source']` — `emission_size`, `consistency`, `galactic_floor`,
      `extragalactic_excess` are physically sane.
- [ ] Phase 6: run `revalidate_dnu`/`fit_two_screen_acf` over casey/freya/wilhelm raw
      spectra (and the provided FRB 20221022A data) and compare to the stored Δν.
- [ ] Confirm the feature branches are acceptable to push given the concurrent-lane
      commit interleaving (Branch state below).

## Issues Encountered

### Issue 1: Pre-existing bug in `analyze_scintillation_from_acfs` (surfaced, not fixed)
- **Impact:** Codex's Phase-4 review found the success branch (~`analysis.py:1389-1423`)
  builds `component_params` (tuples) but never appends to `params_per_comp`, and the
  consumer (~`:1430`) expects dicts — so real `subband_measurements` are always empty.
  The Phase-4 wiring is correct given the documented `final_results` contract but would
  be a no-op on real pipeline output until this upstream bug is fixed.
- **Resolution:** Surfaced in `.agents/deferred-tasks.md` (tagged `@decision`: needs the
  intended multi-component dict contract + real ACF data to validate — out of Phase-4
  scope, which only wires the interpretation funcs). Not fixed here.
- **Files Affected:** `scintillation/scint_analysis/analysis.py` (pre-existing).

### Issue 2: `conda run -n flits` silently uses base Anaconda
- **Impact:** On this shell `conda run -n flits python` resolves to base Anaconda
  (py3.13), not the `flits` env (py3.12) — the PATH-leak hazard in `~/CLAUDE.md`. Early
  test runs (and the spurious `mwprop`-absent skip) were under base.
- **Resolution:** Re-ran every test suite under `/Users/jakobfaber/.conda/envs/flits/bin/python`
  (py3.12.13, numpy 2.4.6, lmfit 1.3.4, mwprop present); all pass, and the real-floor
  test now runs instead of skipping.

### Issue 3: Concurrent session shares the working copy / HEAD
- **Impact:** A concurrent session committed to shared HEAD while I was on my branches,
  interleaving its commits with mine (e.g. ADR-0004 `279f4c6` landed between Phase 4 and
  Phase 5 on `feat/scint-pipeline-wiring`). My four code commits are intact, reachable,
  and pathspec-scoped (no concurrent-lane files swept in).
- **Resolution:** Not rewritten — the concurrent lane is active, and rebasing shared
  history could destroy its work. Reported here and to the user for disentangling
  before push (one-way door, the user's call).

## Testing Summary

**Tests Added:** `tests/test_issue4_commensurable.py` (3),
`scintillation/scint_analysis/tests/test_pipeline_wiring.py` (5),
`.../test_floor_wiring.py` (6), `.../test_revalidation.py` (6).

**All Tests Passing:** ✅ Yes (under the real `flits` env).

## Branch state

All four phases are pathspec-scoped commits descending from the figures base
(`feat/figure-vector`), which the concurrent lane has advanced:

- `fix/issue-4-n1-commensurable` — `a76ca5e` (Phase 3).
- `feat/scint-pipeline-wiring` — `30662b1` (Phase 4) and `c50d5b3` (Phase 5), with the
  concurrent lane's `279f4c6` (ADR-0004) interleaved between them.
- `feat/acf-revalidation-harness` — `bf991a9` (Phase 6).

**Caveat:** these branches are not strictly single-concern — they inherit concurrent-lane
commits from the shared base and (on `feat/scint-pipeline-wiring`) one interleaved
concurrent commit. Before opening PRs, cherry-pick the four code commits onto a clean
base if single-concern PRs are desired.

## Remaining Work

- [ ] Manual verification items above (Phase 3 Occam factor; Phase 4/5 spot-check; Phase 6
      real-data run over casey/freya/wilhelm + FRB 20221022A).
- [ ] Fix the pre-existing `analyze_scintillation_from_acfs` component-extraction bug
      (ledger `@decision`).
- [ ] Push branches / open PRs — one-way door, left to the user (push gate).
- [ ] Disentangle concurrent-lane commits from the feature branches if single-concern
      PRs are wanted.

## Next Steps

1. Human manual verification of the four phases.
2. Decide branch disposition (cherry-pick onto a clean base vs accept the mixed branches).
3. Push + open PRs (user; one-way door).
4. Validate with `ai-research-workflows:validating-implementations`.

## References

**Plan Document:** [plan-incomplete-work-closeout.md](plan-incomplete-work-closeout.md)
**Research Document:** [research-incomplete-work-survey.md](research-incomplete-work-survey.md)

**Commits:**
- `a76ca5e` — test(joint): N=1 commensurability + multi-path routing acceptance (#4)
- `30662b1` — feat(scint): wire two-screen consistency + emission size into pipeline output
- `c50d5b3` — feat(scint): wire NE2025 Galactic floor + extragalactic-excess flag
- `bf991a9` — feat(scint): ACF re-validation harness + two-screen Nimmo/Pleunis bandwidth fit

**External:** Nimmo et al. 2025 (arXiv:2406.11053, Nature; FRB 20221022A); Pleunis et al.
2025 / arXiv:2505.04576 (two-screen scintillometry §5.1). Source `scint_funcs.py` and the
FRB 20221022A data were provided by the user in `~/Downloads/`.

---

## Manual verification + branch disposition + push (2026-06-24)

**Manual verification — PASSED** (real `flits` env, py3.12):
- **P4 physics oracle:** `two_screen_coherence_constraint(0.006, 0.124, 600, 65.189)` → `d_product = 8.78 kpc²` (reproduces Nimmo 2025 published **8.8 kpc²**); `d_gal=0.64 → d_s2 ≤ 13.7 kpc` (Nimmo ~14).
- **P6 port oracle:** `res`/`emission_size` give positive, monotone-in-m magnetosphere-scale sizes (m=0.5→7.30e4, 0.78→3.38e4, 0.95→1.39e4 km).
- **P3:** `lnZ(N=1)=-54.812`, `lnZ(N=2)=-58.274`, `ΔlnZ=-3.46` ≪ `T·ln(2π)=18.4` → commensurable, N=2 Occam-penalized.
- **P4/P5 runtime spot-check** (casey coords, real NE2025 floor): all wired keys (`modulation`, `consistency`, `emission_size`, `galactic_floor`, `extragalactic_excess`) finite and physically sane.

**Branch disposition** (Codex-adjudicated, gpt-5.5 high): the 3 feature branches shared a base contaminated with the separate-active scattering-refit lane (`a25bce0`, `5080f8a`, `f03fab9`, `1c87ef5`, `279f4c6`), with `279f4c6` interleaved between Phase 4/5. Verdict: cherry-pick the 4 commits onto clean `origin/main` as 3 single-concern branches; keep the contaminated originals (they are the **only** carriers of `279f4c6` and other separate-active commits). Each clean branch verified `git log origin/main..HEAD` / `git diff --name-only origin/main...HEAD` shows only the intended commit(s) and only `scintillation/`+`tests/`.

**PRs opened** (each rebased-clean onto `origin/main`):
- **#53** `pr/issue-4-commensurable` ← `a76ca5e` — 3 tests pass. **CLOSED**: a concurrent PR **#56** merged the identical content to `main` first (`git cherry` showed `-`, patch already on main); Codex-adjudicated → closed #53 + deleted branch.
- **#54** `pr/scint-pipeline-wiring` ← `30662b1`, `c50d5b3` — 80 tests pass (5 new + full scint suite). **MERGED**.
- **#55** `pr/acf-revalidation` ← `bf991a9` — 6 tests pass. **MERGED**.

---

## Follow-up: multi-component auto-selection (PR #58, 2026-06-24)

User-requested extension beyond the closeout plan ("ensure we can tell if a 2/3-Lorentzian fit is statistically preferred over a single … wire that in").

- **#58** `feat/scint-multicomponent-select` ← `4da08e19` (single commit on current `origin/main`, after #54/#55 merged). **OPEN**.
- Wires `revalidation.compare_lorentzian_components` (BIC ΔBIC>6 **AND** nested F-test) into `analyze_scintillation_from_acfs`: per-sub-band component count → plurality (ties → fewer) → per-component power-law (components identified by ascending Δν). Replaces the dead `2c`/`3c`-in-name heuristic. Gauss/power/lor_gen stay single-component.
- **Codex-reviewed guards** (two passes): per-sub-band justification (a sub-band contributes a forced N-split only if its own selector justified ≥ N), Δν-ambiguity floor (`_MIN_DNU_RATIO`=2× resolvability, applied unconditionally, + 1σ overlap when errors finite), and a ≥2-measurement guard in the shared power-law consumer (degenerate ODR otherwise).
- **Verification:** full scint suite **96 pass** (6 new in `test_multicomponent_select.py` + 2 in `test_acf_extraction.py`); **mutation-checked** (forcing the count wrong fails the new tests → teeth confirmed); real CHIME data (hamilton 637.7 MHz → n=2 with per-component errors); verify-gate recorded (`test` + `adversarial-review`).

---

**Implementation completed by AI Assistant on 2026-06-24**
