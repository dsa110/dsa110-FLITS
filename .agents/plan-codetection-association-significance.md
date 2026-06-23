# Implementation Plan: CHIME–DSA co-detection association significance (pillars 1–4)

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Status:** Implemented — all 5 phases done & automated-verified (see
[implement-codetection-association-significance.md](implement-codetection-association-significance.md)); manual verification pending
**Codebase state:** `ab9d7f1` (2026-06-23)
**Related Documents:**
- [Research: co-detection validation rigor](research-codetection-validation-rigor.md)
- [Experiment: chance-coincidence false-alarm](experiment-chance-coincidence-falsealarm.md)

---

## Overview

The repo currently asserts each CHIME–DSA pair is a real co-detection on **one** weak test:
temporal consistency (`residual = measured_offset − geometric_delay` within `√(DM_unc² + fwhm²)`;
`crossmatching/plotting.py:84`). All 12 pass, but σ reaches 74 ms and the residuals carry an
unexplained +2.4 ms pedestal — the test is necessary but cannot *exclude* chance.

This plan adds the missing rigorous apparatus as a new `crossmatching/association.py` module with four
independent pillars, assembled into an `association_report.json` that sits **alongside** (never
overwrites) the golden `toa_crossmatch_results.json`:

1. **Chance-coincidence probability** (analytic Poisson; experiment-validated) — the decisive statistic.
2. **Independent DM agreement** (CHIME vs DSA DM, each with its own error).
3. **Honest timing error budget** + a test of whether the +2.4 ms residual pedestal is significant.
4. **Positional coincidence** (DSA arcsec position vs CHIME localization disk), feeding a tighter Ω into pillar 1.

**Goal:** every burst carries a quantified, defensible association significance — a chance-coincidence
probability plus independent DM/position/timing consistency — produced by tested code, with the golden
reproduction artifact untouched.

**Motivation:** "consistent within 3σ" is not "the same astrophysical event." The experiment showed the
chance probability is ~5×10⁻⁸ under conservative inputs; this plan turns that prototype into the
production estimator and adds the orthogonal DM/position/timing checks that tighten it.

## Current State Analysis

**Existing Implementation:**
- `crossmatching/toa_crossmatch.py:99` `compute_toa` (cold-plasma shift to 400 MHz, `K_DM=4148.808`).
- `crossmatching/toa_crossmatch.py:128` `compute_geometric_delay` (OVRO−DRAO baseline projection).
- `crossmatching/toa_crossmatch.py:151` `reproduce_notebook_result` → `CrossmatchResult` (fields at
  `:91-93`: `measured_offset_ms`, `combined_dm_uncertainty_ms`, `geometric_delay_ms`).
- `crossmatching/plotting.py:84,90,150` form `residual`, its error, and a DM-slope probe.
- Inputs: `crossmatching/notebook_reproduction_fixture.json` (per burst: `name`, `chime_id`, `dm`,
  `source_coord`, `dm_uncertainty`; `chime.toa_unix_400`; `dsa.dsa_mjd`).
- Tests: `tests/test_crossmatching_notebook_reproduction.py` (golden reproduction, must keep passing).
- Validated prototype to promote: `.experiments/chance-coincidence/{inputs,estimator_analytic}.py`.

**Current Behavior:** the fixture → `reproduce_notebook_result` → `toa_crossmatch_results.json`
(residual + geometric delay + DM-uncertainty error). No chance probability, no independent DM test, no
position test, no timing-systematics budget.

**Current Limitations:**
- No false-alarm probability anywhere.
- DM carried as one shared value at a flat 0.1 pc cm⁻³; no CHIME-vs-DSA agreement test.
- Error model = DM-uncertainty ⊕ pulse width only; omits clock/baseline/intra-channel terms.
- +2.4 ms residual pedestal unexplained and untested for significance.
- No positional-coincidence test.

## Desired End State

**New Behavior:** `python -m crossmatching.association` reads the fixture and writes
`crossmatching/association_report.json` — one row per burst with `chance_coincidence_P`,
`dm_agreement` (n_sigma, consistent), `timing_budget_ms`, `position_consistent`, and a sample-level
`expected_chance_associations` (Σμ) and `residual_pedestal` significance. The golden
`toa_crossmatch_results.json` is unchanged.

**Success Looks Like:**
- All 12 bursts have `chance_coincidence_P < 1e-3` (in fact ~1e-9) under documented conservative inputs.
- `expected_chance_associations` (Σμ) for the sample is reported and ≪1.
- DM-agreement, position, and timing-budget fields are populated (real values where inputs exist;
  explicit `null` + reason where CHIME-side data is not yet sourced — see *What We're NOT Doing*).
- The residual-pedestal significance (weighted mean residual / its error) is reported as a number.
- `pytest tests/test_association.py` and the existing reproduction test both pass.

## What We're NOT Doing

- [ ] **Sourcing real per-event CHIME independent DMs and CHIME localization regions** from the
      CHIME/FRB catalogue. Our singlebeam files carry no DM/localization (attrs are `event_id`,
      `event_date`, `delta_time`, … only). Pillars 2 and 4 implement and unit-test the *machinery* with
      explicit inputs; populating real CHIME DM/position values is a separate data task.
- [ ] **Modifying `crossmatching/toa_crossmatch_results.json`** (the golden artifact) or
      `reproduce_notebook_result`. The new layer is additive.
- [ ] **A joint multi-observable likelihood-ratio null** or empirical-DM-catalogue MC (the experiment
      recommended analytic Poisson; the MC is validation-only). Deferred unless a future need arises.
- [ ] **Re-deriving TOAs or re-running baseband extraction** — provenance is already verified this session.

**Rationale:** the decisive, fully-sourced pillar (chance probability) ships complete; the pillars
gated on CHIME-side catalogue data ship as tested machinery so they are ready the moment that data lands.

## Implementation Approach

**Technical Strategy:** one focused module `crossmatching/association.py` (~120 LoC) housing four pure
functions (one per pillar) + an assembler, promoting the experiment's validated analytic estimator.
Pure functions, explicit inputs, no hidden state — mirrors `toa_crossmatch.py`'s style. Reuse
`compute_geometric_delay`/`compute_toa` and the fixture loader already present.

**Key Architectural Decisions:**
1. **Decision:** new `crossmatching/association.py`, not edits to `toa_crossmatch.py`.
   - **Rationale:** keeps the golden reproduction path frozen; single-purpose surface (ponytail).
   - **Trade-offs:** one extra file vs entangling significance with reproduction.
   - **Alternatives considered:** adding fields to `CrossmatchResult` — rejected (would change the
     golden test and conflate reproduction with significance).
2. **Decision:** analytic Poisson as the production chance estimator.
   - **Rationale:** experiment showed analytic≡MC to 0.3% and the MC is blind at μ~10⁻⁹.
   - **Trade-offs:** depends on a closed-form DM density (documented assumption).
   - **Alternatives considered:** Monte-Carlo — kept only as an in-test cross-check.
3. **Decision:** assemble into a *new* `association_report.json`, golden untouched.
   - **Rationale:** research doc's explicit constraint; verification stays separable from reproduction.

**Patterns to Follow:**
- Pure-function + dataclass style — `crossmatching/toa_crossmatch.py:99,128,151`.
- Fixture-driven test — `tests/test_crossmatching_notebook_reproduction.py:21`.
- Lazy/optional inputs returning explicit `null+reason` — mirrors `chime_singlebeam.py` loud-failure style.

## Implementation Phases

### Phase 1 — Pillar 1: analytic chance-coincidence probability (the core)

**Objective:** promote the experiment's analytic estimator into `crossmatching/association.py` with a
regression pin and an in-test analytic↔MC cross-check.

**Tasks:**
- [ ] **Write the failing test** — `tests/test_association.py` (new):
  ```python
  import math
  import numpy as np
  from crossmatching.association import (
      f_dm, chance_mu, chance_probability, expected_chance_associations,
      OMEGA_WIN_BASELINE_DEG2,
  )

  BASE = dict(rate_per_day=1000.0, omega_win_deg2=OMEGA_WIN_BASELINE_DEG2, dt_s=1.0, ddm=5.0)

  def test_chance_mu_regression_dm500():
      # pinned from the validated experiment (.agents/experiment-chance-coincidence-falsealarm.md)
      assert chance_mu(500.0, **BASE) == __import__("pytest").approx(5.023345e-09, rel=1e-4)

  def test_chance_mu_scales_linearly_in_small_window():
      base = chance_mu(500.0, **BASE)
      assert chance_mu(500.0, **{**BASE, "dt_s": 2.0}) == __import__("pytest").approx(2 * base, rel=1e-9)
      assert chance_mu(500.0, **{**BASE, "ddm": 10.0}) == __import__("pytest").approx(2 * base, rel=1e-9)

  def test_chance_matches_monte_carlo_in_measurable_regime():
      # analytic must equal a direct background MC where the MC has enough hits (mu ~ 0.046)
      infl = dict(rate_per_day=1000.0, omega_win_deg2=200.0, dt_s=3600.0, ddm=50.0)
      p_an = chance_probability(500.0, **infl)
      rng = np.random.default_rng(7)
      lam = chance_mu(500.0, **infl) / f_dm(500.0, 50.0)  # mean events in pos+time box
      N = 2_000_000
      counts = rng.poisson(lam, size=N)
      tot = int(counts.sum())
      dms = np.exp(rng.normal(math.log(500.0), 0.7, size=tot))
      hit = np.zeros(N, bool); hit[np.repeat(np.arange(N), counts)[np.abs(dms-500.0) <= 50.0]] = True
      p_mc = hit.mean()
      assert p_mc == __import__("pytest").approx(p_an, rel=0.05)

  def test_expected_chance_associations_sums_mu():
      dms = [262.4, 500.0, 960.1]
      assert expected_chance_associations(dms, **BASE) == __import__("pytest").approx(
          sum(chance_mu(d, **BASE) for d in dms), rel=1e-12)
  ```
- [ ] **Run it, watch it fail:** `pytest tests/test_association.py -q` → FAIL (module missing).
- [ ] **Implement** `crossmatching/association.py` (new) — promote `inputs.py` + `estimator_analytic.py`:
  ```python
  from __future__ import annotations
  import math

  # CHIME/FRB Catalogue 1 (Amiri et al. 2021, ApJS 257, 59): ~525 FRBs/sky/day above 5 Jy ms.
  R_SKY_PER_DAY_CENTRAL = 525.0
  FULL_SKY_SR = 4.0 * math.pi
  SECONDS_PER_DAY = 86400.0
  DEG2_PER_SR = (180.0 / math.pi) ** 2
  DM_MEDIAN, DM_SIGMA_LN = 500.0, 0.7              # log-normal CHIME DM model (assumption; documented)
  OMEGA_WIN_BASELINE_DEG2 = math.pi * 0.5 ** 2     # 0.5 deg radius disk
  DT_BASELINE_S, DDM_BASELINE = 1.0, 5.0

  def _r_sr_s(rate_per_day): return rate_per_day / FULL_SKY_SR / SECONDS_PER_DAY

  def f_dm(dm, half_width, *, dm_median=DM_MEDIAN, dm_sigma_ln=DM_SIGMA_LN):
      z = (math.log(dm) - math.log(dm_median)) / dm_sigma_ln
      pdf = math.exp(-0.5 * z * z) / (dm * dm_sigma_ln * math.sqrt(2.0 * math.pi))
      return min(1.0, pdf * 2.0 * half_width)

  def chance_mu(dm, *, rate_per_day, omega_win_deg2, dt_s, ddm):
      return _r_sr_s(rate_per_day) * (omega_win_deg2 / DEG2_PER_SR) * (2.0 * dt_s) * f_dm(dm, ddm)

  def chance_probability(dm, **kw):
      return 1.0 - math.exp(-chance_mu(dm, **kw))

  def expected_chance_associations(dms, **kw):
      return sum(chance_mu(d, **kw) for d in dms)
  ```
- [ ] **Run it, watch it pass:** `pytest tests/test_association.py -q` → PASS (4 tests).
- [ ] **Lint:** `ruff check crossmatching/association.py tests/test_association.py` → clean.
- [ ] **Commit:** `git commit -m "feat(crossmatch): analytic chance-coincidence estimator (pillar 1)"`

**Dependencies:** none (promotes validated experiment code).
**Verification:**
- [ ] `pytest tests/test_association.py -q` → 4 passed.
- [ ] `python -c "from crossmatching.association import chance_probability, OMEGA_WIN_BASELINE_DEG2 as O; print(chance_probability(262.4, rate_per_day=1000, omega_win_deg2=O, dt_s=1, ddm=5))"`
      → prints ≈6.3e-9.

### Phase 2 — Pillar 2: independent DM agreement

**Objective:** a CHIME-vs-DSA DM consistency statistic with explicit per-side errors.

**Tasks:**
- [ ] **Write the failing test** — append to `tests/test_association.py`:
  ```python
  from crossmatching.association import dm_agreement

  def test_dm_agreement_consistent():
      r = dm_agreement(dm_chime=500.0, dm_chime_err=2.0, dm_dsa=502.0, dm_dsa_err=1.0)
      assert r["delta"] == __import__("pytest").approx(2.0)
      assert r["sigma"] == __import__("pytest").approx(math.sqrt(5.0))
      assert r["n_sigma"] == __import__("pytest").approx(2.0 / math.sqrt(5.0))
      assert r["consistent"] is True

  def test_dm_agreement_inconsistent_beyond_3sigma():
      r = dm_agreement(dm_chime=500.0, dm_chime_err=1.0, dm_dsa=510.0, dm_dsa_err=1.0)
      assert r["consistent"] is False

  def test_dm_agreement_missing_chime_dm_returns_null_reason():
      r = dm_agreement(dm_chime=None, dm_chime_err=None, dm_dsa=502.0, dm_dsa_err=1.0)
      assert r["consistent"] is None and "no CHIME DM" in r["reason"]
  ```
- [ ] **Run it, watch it fail:** `pytest tests/test_association.py -k dm_agreement -q` → FAIL.
- [ ] **Implement** in `crossmatching/association.py`:
  ```python
  def dm_agreement(*, dm_chime, dm_chime_err, dm_dsa, dm_dsa_err, n_sigma_thresh=3.0):
      if dm_chime is None or dm_dsa is None:
          return {"delta": None, "sigma": None, "n_sigma": None,
                  "consistent": None, "reason": "no CHIME DM available"}
      delta = abs(dm_chime - dm_dsa)
      sigma = math.hypot(dm_chime_err or 0.0, dm_dsa_err or 0.0)
      n = delta / sigma if sigma > 0 else float("inf")
      return {"delta": delta, "sigma": sigma, "n_sigma": n,
              "consistent": bool(n <= n_sigma_thresh), "reason": None}
  ```
- [ ] **Run it, watch it pass:** `pytest tests/test_association.py -k dm_agreement -q` → PASS (3).
- [ ] **Commit:** `git commit -m "feat(crossmatch): independent CHIME-DSA DM agreement (pillar 2)"`

**Dependencies:** Phase 1 (same module/test file).
**Verification:** `pytest tests/test_association.py -k dm_agreement -q` → 3 passed.

### Phase 3 — Pillar 3: timing error budget + residual-pedestal significance

**Objective:** a complete quadrature timing-error budget, and a test of whether the +2.4 ms residual
pedestal (research finding) is statistically significant across the 12.

**Tasks:**
- [ ] **Write the failing test** — append to `tests/test_association.py`:
  ```python
  from crossmatching.association import timing_budget_ms, residual_pedestal

  def test_timing_budget_quadrature():
      got = timing_budget_ms(dm_unc_ms=2.4, fwhm_ms=0.96, clock_ms=0.1,
                             baseline_ms=0.05, intrachannel_ms=0.2)
      assert got == __import__("pytest").approx(math.sqrt(2.4**2 + 0.96**2 + 0.1**2 + 0.05**2 + 0.2**2))

  def test_residual_pedestal_significance():
      # equal residuals of +2.4 with errors 2.4 -> weighted mean 2.4, error 2.4/sqrt(12)
      res = [2.4] * 12; err = [2.4] * 12
      r = residual_pedestal(res, err)
      assert r["weighted_mean_ms"] == __import__("pytest").approx(2.4)
      assert r["error_ms"] == __import__("pytest").approx(2.4 / math.sqrt(12))
      assert r["n_sigma"] == __import__("pytest").approx(math.sqrt(12))
  ```
- [ ] **Run it, watch it fail:** `pytest tests/test_association.py -k "timing or pedestal" -q` → FAIL.
- [ ] **Implement** in `crossmatching/association.py`:
  ```python
  def timing_budget_ms(*, dm_unc_ms, fwhm_ms, clock_ms=0.0, baseline_ms=0.0, intrachannel_ms=0.0):
      return math.sqrt(dm_unc_ms**2 + fwhm_ms**2 + clock_ms**2 + baseline_ms**2 + intrachannel_ms**2)

  def residual_pedestal(residuals_ms, errors_ms):
      w = [1.0 / e**2 for e in errors_ms]
      wm = sum(wi * r for wi, r in zip(w, residuals_ms)) / sum(w)
      err = math.sqrt(1.0 / sum(w))
      return {"weighted_mean_ms": wm, "error_ms": err, "n_sigma": abs(wm) / err}
  ```
- [ ] **Run it, watch it pass:** `pytest tests/test_association.py -k "timing or pedestal" -q` → PASS (2).
- [ ] **Commit:** `git commit -m "feat(crossmatch): timing budget + residual-pedestal significance (pillar 3)"`

**Dependencies:** Phase 1.
**Verification:** `pytest tests/test_association.py -k "timing or pedestal" -q` → 2 passed.

### Phase 4 — Pillar 4: positional coincidence → tighter Ω

**Objective:** test DSA arcsec position against a CHIME localization disk, and expose the disk's Ω so
pillar 1 can use the *actual* positional window instead of the generous baseline.

**Tasks:**
- [ ] **Write the failing test** — append to `tests/test_association.py`:
  ```python
  from crossmatching.association import omega_disk_deg2, position_consistent

  def test_omega_disk_area():
      assert omega_disk_deg2(0.5) == __import__("pytest").approx(math.pi * 0.25)

  def test_position_inside_and_outside_chime_disk():
      dsa = "20h40m47.886s +72d52m56.378s"
      assert position_consistent(dsa, "20h40m50s +72d53m00s", radius_deg=0.2) is True
      assert position_consistent(dsa, "20h00m00s +60d00m00s", radius_deg=0.2) is False
  ```
- [ ] **Run it, watch it fail:** `pytest tests/test_association.py -k "disk or position" -q` → FAIL.
- [ ] **Implement** in `crossmatching/association.py` (add `import astropy.units as u`, `from
      astropy.coordinates import SkyCoord` in the same edit as these functions):
  ```python
  def omega_disk_deg2(radius_deg):
      return math.pi * radius_deg**2

  def position_consistent(dsa_coord, chime_center, radius_deg):
      import astropy.units as u
      from astropy.coordinates import SkyCoord
      a = SkyCoord(dsa_coord, unit=(u.hourangle, u.deg), frame="icrs")
      b = SkyCoord(chime_center, unit=(u.hourangle, u.deg), frame="icrs")
      return bool(a.separation(b).deg <= radius_deg)
  ```
- [ ] **Run it, watch it pass:** `pytest tests/test_association.py -k "disk or position" -q` → PASS (2).
- [ ] **Commit:** `git commit -m "feat(crossmatch): positional coincidence + Omega (pillar 4)"`

**Dependencies:** Phase 1. **Verification:** `pytest tests/test_association.py -k "disk or position" -q` → 2 passed.

### Phase 5 — Assemble the association report (golden untouched)

**Objective:** one assembler that reads the fixture, runs all four pillars, and writes
`crossmatching/association_report.json` without touching the golden file.

**Tasks:**
- [ ] **Write the failing test** — append to `tests/test_association.py`:
  ```python
  import json
  from pathlib import Path
  from crossmatching.association import build_association_report

  ROOT = Path(__file__).resolve().parents[1]

  def test_report_has_chance_P_for_all_12_and_golden_untouched():
      golden_before = (ROOT / "crossmatching/toa_crossmatch_results.json").read_text()
      report = build_association_report(ROOT / "crossmatching/notebook_reproduction_fixture.json")
      assert len(report["bursts"]) == 12
      assert all(b["chance_coincidence_P"] < 1e-3 for b in report["bursts"])
      assert report["expected_chance_associations"] < 1e-3
      # golden file content is unchanged by building the report
      assert (ROOT / "crossmatching/toa_crossmatch_results.json").read_text() == golden_before
  ```
- [ ] **Run it, watch it fail:** `pytest tests/test_association.py -k report -q` → FAIL.
- [ ] **Implement** `build_association_report` + `main()` in `crossmatching/association.py`:
  ```python
  import json

  def build_association_report(fixture_path, *, rate_per_day=1000.0,
                               omega_win_deg2=OMEGA_WIN_BASELINE_DEG2, dt_s=DT_BASELINE_S, ddm=DDM_BASELINE):
      fx = json.loads(open(fixture_path).read())
      bursts, dms = [], []
      for row in fx["bursts"]:
          dm = row["dm"]; dms.append(dm)
          bursts.append({
              "name": row["name"], "chime_id": row["chime_id"], "dm": dm,
              "chance_coincidence_P": chance_probability(
                  dm, rate_per_day=rate_per_day, omega_win_deg2=omega_win_deg2, dt_s=dt_s, ddm=ddm),
              "dm_agreement": dm_agreement(            # CHIME DM not yet sourced -> null+reason
                  dm_chime=None, dm_chime_err=None, dm_dsa=dm, dm_dsa_err=row.get("dm_uncertainty")),
              "position_consistent": None,             # CHIME localization not yet sourced
          })
      return {
          "inputs": {"rate_per_day": rate_per_day, "omega_win_deg2": omega_win_deg2,
                     "dt_s": dt_s, "ddm": ddm, "dm_model": "lognormal(500,0.7) [assumption]"},
          "expected_chance_associations": expected_chance_associations(
              dms, rate_per_day=rate_per_day, omega_win_deg2=omega_win_deg2, dt_s=dt_s, ddm=ddm),
          "bursts": bursts,
      }

  def main():
      import pathlib
      here = pathlib.Path(__file__).resolve().parent
      rep = build_association_report(here / "notebook_reproduction_fixture.json")
      out = here / "association_report.json"
      out.write_text(json.dumps(rep, indent=2))
      print(f"wrote {out}  (sum_mu={rep['expected_chance_associations']:.3e})")

  if __name__ == "__main__":
      main()
  ```
- [ ] **Run it, watch it pass:** `pytest tests/test_association.py -k report -q` → PASS.
- [ ] **Generate the artifact:** `python -m crossmatching.association` → writes `association_report.json`.
- [ ] **Confirm golden untouched:** `git status --porcelain crossmatching/toa_crossmatch_results.json`
      → empty.
- [ ] **Commit:** `git commit -m "feat(crossmatch): assemble association significance report (pillars 1-4)"`

**Dependencies:** Phases 1–4.
**Verification:**
- [ ] `pytest tests/test_association.py -q` → all pass (≥12 tests).
- [ ] `python -m crossmatching.association` prints `sum_mu=…e-08` and creates
      `crossmatching/association_report.json`.
- [ ] `git status --porcelain crossmatching/toa_crossmatch_results.json` → empty.

## Success Criteria

### Automated Verification
- [ ] `pytest tests/test_association.py -q` → all pass.
- [ ] `pytest tests/test_crossmatching_notebook_reproduction.py -q` → still passes (golden intact).
- [ ] `ruff check crossmatching/association.py tests/test_association.py` → clean.
- [ ] `python -m crossmatching.association` exits 0 and writes `crossmatching/association_report.json`.
- [ ] `python -c "import json;r=json.load(open('crossmatching/association_report.json'));assert all(b['chance_coincidence_P']<1e-3 for b in r['bursts']);assert r['expected_chance_associations']<1e-3"`.
- [ ] `git status --porcelain crossmatching/toa_crossmatch_results.json` → empty.

### Manual Verification
- [ ] The per-burst `chance_coincidence_P` values (~1e-9) and `expected_chance_associations` (~5e-8)
      match the experiment's order of magnitude.
- [ ] The reported `dm_model`/window assumptions in `association_report.json["inputs"]` are the
      conservative ones from the experiment, and reviewers agree they are chance-maximising.
- [ ] `dm_agreement`/`position_consistent` are explicit `null` with a reason (CHIME data not yet
      sourced) rather than silently fabricated values.

### Reproducibility & Correctness (research code)
- [ ] Chance estimator pinned by `test_chance_mu_regression_dm500` (5.023345e-9) and cross-checked
      against a seeded MC (`test_chance_matches_monte_carlo_in_measurable_regime`, rel 5%).
- [ ] All inputs (rate, DM model, windows) are constants in `association.py` with cited/labelled
      provenance; report echoes them under `inputs`.
- [ ] Clean-env reproduction: `pytest tests/test_association.py -q` from a fresh `casa6` env passes.

## Testing Strategy

**Unit (test-first, in-phase):** chance μ regression + scaling + analytic↔MC (P1); DM agreement
consistent/inconsistent/null (P2); timing quadrature + pedestal significance (P3); disk area +
position in/out (P4); report assembly + golden-untouched (P5).

**Integration:** `test_report_has_chance_P_for_all_12_and_golden_untouched` exercises the full
fixture→pillars→report path and asserts the golden file is byte-identical before/after.

**Manual:** eyeball `association_report.json` against the experiment numbers; confirm null-reason
fields where CHIME data is absent.

**Test Data:** existing `crossmatching/notebook_reproduction_fixture.json` (12 bursts) — no new data.

## Risk Assessment
1. **Risk:** chance-P absolute value depends on assumed windows/DM model.
   - **Likelihood:** Medium · **Impact:** Low — conclusion robust over 5 orders of magnitude (experiment);
     report echoes inputs; baseline is deliberately conservative.
2. **Risk:** pillars 2/4 ship without real CHIME DM/position data.
   - **Likelihood:** High · **Impact:** Low — machinery is tested; fields return explicit null+reason;
     sourcing is a scoped-out data task.
3. **Risk:** accidental golden-file mutation.
   - **Likelihood:** Low · **Impact:** High — assembler only writes `association_report.json`; a test
     asserts the golden file is unchanged.

## Edge Cases and Error Handling
1. **Missing CHIME DM/position:** `dm_agreement`/`position_consistent` return `null` + reason
   (`test_dm_agreement_missing_chime_dm_returns_null_reason`); never fabricate.
2. **Zero combined DM error:** `dm_agreement` yields `n_sigma = inf` (guarded `sigma>0`).
3. **Zero residual errors in pedestal:** out of scope — all 12 have non-zero `combined_dm_uncertainty_ms`;
   inputs come from the golden JSON which guarantees positive errors.

## Documentation Updates
- [ ] Docstrings on every `association.py` function (provenance of constants inline).
- [ ] One paragraph in `docs/codetection-science-plan.md` §A updating `crossmatching/` from
      "Stub / aspirational" to "association significance: pillars 1–4 implemented; CHIME-side DM/position
      data pending."

## Timeline Estimate
- Phase 1: ~0.5 day · Phases 2–4: ~0.5 day total · Phase 5: ~0.5 day. Total ~1.5 days.

## Open Questions

*(none — CHIME-side DM/position sourcing is explicitly scoped out under "What We're NOT Doing", not an
open question)*

---

## References
**Research:** [research-codetection-validation-rigor.md](research-codetection-validation-rigor.md)
**Experiment:** [experiment-chance-coincidence-falsealarm.md](experiment-chance-coincidence-falsealarm.md)
**Files analyzed:** `crossmatching/toa_crossmatch.py:99,128,151,91-93`, `crossmatching/plotting.py:84,90,150`,
`crossmatching/notebook_reproduction_fixture.json`, `tests/test_crossmatching_notebook_reproduction.py`,
`.experiments/chance-coincidence/{inputs,estimator_analytic,estimator_mc,run}.py`.
**External:** CHIME/FRB Catalogue 1 — Amiri et al. 2021, ApJS 257, 59; Foster et al. 2018 (arXiv:1808.07809);
Law et al. 2017 (arXiv:1705.07553).

---

## Review History
### Version 1.0 — 2026-06-23
- Initial plan created from the research + experiment docs.
