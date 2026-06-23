# Validation: E_iso energetics follow-ups

> Validated against `plan-energetics-followups.md` / `research-energetics-followups.md`
> at base commit `2d91773` (worktree `feat/energetics-followups`), 2026-06-23.
> Every automated command below was re-run in env `~/.conda/envs/flits` by the
> validator (Iron Law: no verdict without fresh self-produced output).

## Verdict: PASS — ship.

All four follow-ups are implemented at the smallest correct size; the gate is
green and an independent adversarial review found no bugs.

## Implementation status (per phase)

- **Phase 1 — code hygiene.** ✅ `dsa_beam.py` `@lru_cache(maxsize=2)` on
  `load_power_beam`; ✅ `flux_cal.py:261` `if valid.any():` guard. C3 (gate
  tightening) correctly **not** done — the mixed float/fluxcal path is intentional
  and tested.
- **Phase 2 — redshift provenance.** ✅ `Z_PROVENANCE` dict + `row["z_src"]`;
  ✅ `test_z_provenance_flags_unpublished_hosts`. No photometric z to demote; the 8
  hosts are spectroscopic, hamilton/chromatica flagged provisional.
- **Phase 3 — catalog-anchor negative result.** ✅ recorded in
  `CALIBRATION_REVIEW.md` (Law+2024 Table 1 is Feb–Oct 2022 only; no later DSA-110
  paper republishes Jy·ms fluences; do not promote model values into `CATALOG`).
- **Phase 4 — manuscript.** ✅ `Faber2026/sections/results.tex`: TNS names in the
  table + prose, `% TODO` removed, provisional footnotes on FRB 20230913A /
  FRB 20240203A; builds clean. Deferred (documented): `observations.tex` /
  `budget.tex` stubs (need sightline budget numbers that do not exist yet) and the
  stale `wilhelm_twoscreen_fig.py:26` comment (active scattering-refit lane subtree
  — reported, not edited).

## Automated verification results (re-run by validator)

- ✅ `pytest tests/test_flux_cal.py tests/test_burst_energies_fluxcal.py -q` —
  **13 passed, 2 skipped** (the 2 skips are the catalog cross-check + SEFD test
  that require the unstaged DSA `.npy`; expected). New provenance test passes.
- ✅ `python analysis/calculate_burst_energies.py --check` — `self-check OK`
  (integral=quadrature, energy oracle, k-correction, gate). The printed
  `matplotlibrc` parse warning is pre-existing and unrelated.
- ✅ `python analysis/flux_cal.py` — `self-check OK` (radiometer + flat-band oracle).
- ✅ `python analysis/dsa_beam.py` — `self-check OK` (cube staged here, so the
  `lru_cache` path is exercised: boresight=1, gain(1.8°,1.4GHz)=0.477).
- ✅ `ruff check` (4 touched modules + test) — `All checks passed!`
- ✅ `Faber2026`: `latexmk -pdf` exit 0, `main.pdf` produced; second `pdflatex`
  pass shows **no undefined references** (`tab:burst-energies` resolves).

## Adversarial code review (separate agent, FLITS diff)

Verdict **clean, no correctness bugs**. Notable cleared checks:
- `lru_cache` returns shared arrays but the only consumer (`beam_gain` →
  `RegularGridInterpolator`) never mutates them (scipy 1.16.3 path inspected); the
  live route always uses the default `DEFAULT_BEAM` key → ~6160 reloads collapse to
  1. `maxsize=2` is harmless headroom.
- The `valid.any()` guard is bit-identical to the old code on 2000 randomized
  non-empty inputs; in the all-masked case it yields the same all-False mask and
  `np.trapezoid(empty, empty) == 0.0` downstream — the correct contribution.
- The 8 `Z_PROVENANCE` keys are exactly the bursts `compute()` emits (casey lacks a
  joint fit; freya/mahi/johndoeii are placeholder z) — nothing silently falls to
  `"unknown"`. No row-shape break in `markdown_table`/`latex_section`/tests.
- No stray changes in the diff.

## Manual verification (human)

- [ ] Glance at the Faber2026 PDF energy table: TNS names render, footnote `a`
  attaches to FRB 20230913A and FRB 20240203A. (Build + ref resolution are
  machine-confirmed; only the visual styling needs an eye.)

## Recommendations

- **Critical / Important:** none.
- **Follow-up:** when the hamilton/chromatica host papers publish, replace the
  provisional footnote with the spectroscopic citation and update `Z_PROVENANCE`.
  The `observations.tex`/`budget.tex` energetics prose remains blocked on the
  sightline DM/scattering budget.

## References
- [research-energetics-followups.md](research-energetics-followups.md)
- [plan-energetics-followups.md](plan-energetics-followups.md)
- [validation-radiometer-flux-cal.md](validation-radiometer-flux-cal.md)
