# Plan: activate association pillars 2 & 4 with CHIME-side DM + localization

---
**Date:** 2026-06-23
**Status:** Phases 1-3 done (3 real / 8 marginal / 1 noise DM; 12/12 positions); Phase 4 next
**Research:** [research-chime-side-dm-localization.md](research-chime-side-dm-localization.md)
**Branch:** `feat/chime-side-dm-localization` (NOT main)
---

## Overview
Move association pillars 2 (DM agreement) and 4 (positional coincidence) from `null+reason` to real per-burst values, sourced from the local CHIME singlebeam voltages:
- **Pillar 2:** independent CHIME DM via **structure-maximizing DM-phase** (`dispersion/dmphasev2.DMPhaseEstimator`) + bootstrap σ, extracted in the `chimefrb/baseband-analysis` docker image.
- **Pillar 4:** **point cross-check** of the CHIME tied-beam position (`tiedbeam_locations` ra/dec) vs the DSA arcsec position, consistent within a stated CHIME localization radius (Michilli et al. 2021-justified assumption).

The golden `crossmatching/toa_crossmatch_results.json` is never touched.

## Current state (file:line)
- `crossmatching/association.py:68 dm_agreement(...)` returns real n_sigma when `dm_chime` non-null; fed `None` at `:156-161`.
- `crossmatching/association.py:119 position_consistent(dsa_coord, chime_center, radius_deg) -> bool` (real astropy separation); bypassed at `:162` (`position_consistent=None`).
- `crossmatching/association.py:130 build_association_report(fixture_path, ...)` — single assembler; reads `notebook_reproduction_fixture.json`-shaped fixture.
- `dispersion/dmphasev2.py:25 DMPhaseEstimator(waterfall,(n_t,n_ch) complex, freqs, dt, dm_grid, ref, n_boot, random_state).get_dm()->(dm_best,dm_sigma)`.
- `scripts/extract_chime_singlebeam_toas.py:83-85` — the validated coherent+incoherent dedisp entry path (docker).
- `bin/baseband_analysis_python.sh` — runs `python <script>` in the docker image, `/data` mounted (on h17, under the data root, NOT in this repo).

## Desired end state
- `python -m crossmatching.association` emits a report where all 12 bursts have a non-null `dm_agreement` (real CHIME DM + n_sigma) and a non-null `position` block (CHIME–DSA separation + consistent flag). `inputs` records the CHIME DM method and localization-radius assumption.
- A new permanent host test proves `DMPhaseEstimator` recovers a known injected DM.
- Per-burst DM-phase diagnostic figures produced and passed through the figure-review gate.

## What we're NOT doing
- No S/N-maximizing DM (user chose structure-max only).
- No CHIME error *ellipse* / multi-beam localization (not on disk) — pillar 4 stays a point cross-check with a stated radius.
- No change to pillar 1 or 3, the TOA golden, or the TOA extraction.
- No new dependency: reuse in-repo `DMPhaseEstimator` and the existing docker image.

## Phase 1 — DM-phase recovery harness (host, test-first, no docker)
Settles the estimator mechanics (grid/window) against a known answer before any real extraction.

- [x] **1.1** Write failing test `tests/test_dmphase_recovery.py`:
  ```python
  import numpy as np
  from dispersion.dmphasev2 import DMPhaseEstimator

  def _disperse(n_t, freqs, dt, dm, comps):
      from flits.common.constants import K_DM
      ref = freqs.max()
      wf = np.zeros((n_t, freqs.size))
      for t0, amp, width in comps:  # gaussian sub-pulses, sharp structure
          delay = 1e-3 * K_DM * (1/freqs**2 - 1/ref**2) * dm  # s, per channel
          for j, fj in enumerate(freqs):
              t = (np.arange(n_t)*dt) - t0 - delay[j]
              wf[:, j] += amp*np.exp(-0.5*(t/width)**2)
      return wf

  def test_dmphase_recovers_known_dm():
      rng = np.random.default_rng(0)
      freqs = np.linspace(400.0, 800.0, 256)
      dt, n_t, dm_true = 2.56e-4, 2048, 500.0
      comps = [(0.06, 1.0, 5e-4), (0.063, 0.7, 5e-4)]   # two sharp sub-pulses
      wf = _disperse(n_t, freqs, dt, dm_true, comps) + 0.05*rng.standard_normal((n_t, freqs.size))
      grid = np.arange(dm_true-3.0, dm_true+3.0, 0.05)
      est = DMPhaseEstimator(wf, freqs, dt, grid, ref="top", n_boot=100, random_state=1)
      dm_best, dm_sigma = est.get_dm()
      assert abs(dm_best - dm_true) < max(5*dm_sigma, 0.5)
      assert dm_sigma > 0
  ```
  Run `pytest tests/test_dmphase_recovery.py -q` → watch it fail/error (import path, grid, or recovery).
- [x] **1.2** Settle settings so it passes: adjust grid span/step, `ref`, and (if needed) the `_window_mask` `f_cut` via the estimator's `f_cut` arg. Do **not** change `dmphasev2.py` physics; only the test's call parameters. If a real estimator bug surfaces, stop and file a mismatch.
- [x] **1.3** Run `pytest tests/test_dmphase_recovery.py -q` → **pass**; `ruff check tests/test_dmphase_recovery.py`.

**Automated verification:** `pytest tests/test_dmphase_recovery.py -q` passes; ruff clean.

## Phase 2 — CHIME-side extraction (docker, h17, under the data root)
Produces `crossmatching/chime_side_inputs.json` (committed to the repo) + diagnostic figures.

- [x] **2.1** Write `/data/research/astrophysics/frbs/chime-dsa-codetections/scripts/extract_chime_side_inputs.py` (NOT in the repo tree; mirrors `extract_chime_singlebeam_toas.py`). Per burst keyed by `burst_inputs.json`:
  - read `tiedbeam_locations` ra/dec (h5py, first row) → `chime_ra_deg, chime_dec_deg`;
  - `BBData.from_file`; `coherent_dedisp(bb, DM_c, time_shift=False)`; build `I = |X|²+|Y|²` `(nfreq,ntime)`; transpose to `(ntime,nfreq)`; ascending-freq order;
  - `grid = arange(DM_c-3, DM_c+3, 0.05)`; `DMPhaseEstimator(I, freqs_asc, dt=delta_time, grid, ref="top", n_boot=200, random_state=0)` → `dm_chime, dm_chime_err`;
  - emit per-burst PNG: DM-structure curve (`result()['dm_curve']` vs `dm_grid`, peak marked) + dedispered waterfall, to `diagnostics/chime_side_dm/<name>_dmphase.png`;
  - collect `{name, chime_id, dm_dsa, dm_chime, dm_chime_err, chime_ra_deg, chime_dec_deg, method, dm_grid_span, n_boot}`.
  - Make `dispersion/dmphasev2.py` importable in the container (PYTHONPATH to a copy of the module, or vendor the single file next to the script — it only needs numpy/scipy + `flits.common.constants.K_DM`; if `flits` isn't importable in-image, inline `K_DM = 4.148808e3`).
  - Write `figures.manifest.json` next to the PNGs (per-figure stated expectation: "DM-structure curve peaks within grid near DM_c; dedispersed burst vertical").
- [x] **2.2** Run `bin/baseband_analysis_python.sh scripts/extract_chime_side_inputs.py`; copy the resulting `chime_side_inputs.json` into the repo at `crossmatching/chime_side_inputs.json`.
- [x] **2.3** **Figure-review gate:** Read every `diagnostics/chime_side_dm/*.png` (dispatch `figure-reviewer`), write `figures.review.json` with per-figure verdicts. A DM is not validated until its curve is looked at; flag any burst whose structure-max rails to a grid edge or has a flat/multi-modal curve (→ widen grid or mark low-confidence).

**Automated verification:** `crossmatching/chime_side_inputs.json` exists with 12 rows, each with finite `dm_chime>0`, `dm_chime_err>0`, and `chime_ra_deg/chime_dec_deg`. **Manual:** figure-review verdicts written; no un-reviewed manifest.

## Phase 3 — wire pillars 2 & 4 (repo, test-first)
- [x] **3.1** Replace `position_consistent` (bare bool) with `position_agreement(dsa_coord, chime_ra_deg, chime_dec_deg, radius_deg) -> dict` mirroring `dm_agreement`:
  ```python
  def position_agreement(dsa_coord, chime_ra_deg, chime_dec_deg, radius_deg):
      """CHIME tied-beam point vs DSA position; consistent within a stated CHIME radius."""
      import astropy.units as u
      from astropy.coordinates import SkyCoord
      if chime_ra_deg is None or chime_dec_deg is None:
          return {"separation_deg": None, "radius_deg": radius_deg, "consistent": None,
                  "reason": "no CHIME position available"}
      a = SkyCoord(dsa_coord, unit=(u.hourangle, u.deg), frame="icrs")
      b = SkyCoord(chime_ra_deg, chime_dec_deg, unit=u.deg, frame="icrs")
      sep = float(a.separation(b).deg)
      return {"separation_deg": sep, "radius_deg": radius_deg,
              "consistent": bool(sep <= radius_deg), "reason": None}
  ```
  Update the one position test (`tests/test_association.py:104`) to assert the dict shape (inside → consistent True + sep small; far → False).
- [x] **3.2** Write failing tests in `tests/test_association.py`:
  - `test_position_agreement_inside_and_outside` (sep ~0 consistent True; 13° apart False; null coord → null+reason).
  - `test_report_activates_pillars_2_and_4`: build report from the real fixture + a tiny inline CHIME-side stub (monkeypatch / temp json), assert every burst `dm_agreement.consistent is not None` and `position.consistent is not None`, golden untouched.
- [x] **3.3** Extend `build_association_report(fixture_path, *, ..., chime_inputs_path=None, chime_radius_deg=0.1)`:
  - if `chime_inputs_path`, load it → `{chime_id: {dm_chime, dm_chime_err, chime_ra_deg, chime_dec_deg}}`;
  - feed `dm_agreement(dm_chime=ci.dm_chime, dm_chime_err=ci.dm_chime_err, dm_dsa=dm, dm_dsa_err=row.get("dm_uncertainty"))`;
  - add `"position": position_agreement(row["source_coord"], ci.chime_ra_deg, ci.chime_dec_deg, chime_radius_deg)`;
  - extend `inputs` with `chime_dm_method="DM-phase structure-max (dmphasev2)"`, `chime_localization_radius_deg`, `chime_localization_note="tiedbeam pointing; no multi-beam error ellipse (Michilli+2021 sub-arcmin assumed)"`.
  - `main()` passes `chime_inputs_path = here/"chime_side_inputs.json"` when present.
- [x] **3.4** `pytest tests/test_association.py tests/test_dmphase_recovery.py -q` → all pass; `ruff check/format`. Regenerate `python -m crossmatching.association`; confirm 12 bursts now carry real pillar-2/4 values and `git status` shows golden clean.

**Automated verification:** full association + dmphase tests pass; ruff clean; report has non-null pillar-2/4 for all 12; `git status --porcelain crossmatching/toa_crossmatch_results.json` empty.

## Phase 4 — docs, adversarial verify, PR
- [ ] **4.1** `.agents/implement-chime-side-dm-localization.md`; update `docs/codetection-science-plan.md:30` §A row (pillars 2/4 now active, with the point-localization caveat).
- [ ] **4.2** Adversarial-verification Workflow (mirrors the pillars-1–4 thread): dimensions = DM-phase recovery + extraction integrity; pillar-2/4 numerical correctness (independent recompute of n_sigma + separation); golden-untouched + determinism; lint/cross-doc. Skeptic re-check of any blocking/major finding.
- [ ] **4.3** Fix verified findings; push branch; PR → main with review summary; merge; confirm main CI green.

## Success criteria
**Automated:** (a) `pytest tests/test_dmphase_recovery.py tests/test_association.py -q` all pass; (b) `ruff check`/`format --check` clean on touched files; (c) `crossmatching/chime_side_inputs.json` has 12 finite rows; (d) report: all 12 `dm_agreement.consistent` and `position.consistent` non-null; (e) golden blob byte-identical to origin/main; (f) `python -m crossmatching.association` deterministic (byte-identical report on rerun).
**Manual:** (a) figure-review verdicts for every DM-phase PNG; (b) DM-phase curves peak inside the grid (no edge-railing) for the bursts used as consistent; (c) per-burst CHIME vs DSA DM offsets and n_sigma are physically sane (most within a few σ); (d) adopted CHIME localization radius is defensible vs Michilli 2021.

## Testing strategy
- **Unit (host):** DM-phase known-DM recovery (Phase 1); position_agreement in/out/null; dm_agreement already covered.
- **Integration (host):** report activates both pillars over the 12-burst fixture; golden-untouched invariant.
- **Pipeline (docker, h17):** the extraction script over 12 real bursts; validated by the figure-review gate.

## References
- Research: [research-chime-side-dm-localization.md](research-chime-side-dm-localization.md)
- DM-phase: Hessels et al. 2019 (ApJL 876, L23); Gajjar et al. 2018 (ApJ 863, 2); DM_phase (Seymour/Michilli/Lin).
- CHIME baseband localization: Michilli et al. 2021 (ApJ 910, 147).
