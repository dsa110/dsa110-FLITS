# Plan: E_iso energetics follow-ups

> Grounded in `research-energetics-followups.md` @ `2d91773`, 2026-06-23.
> Branch: `feat/energetics-followups` (worktree). Manuscript: `Faber2026` `main`.

## Overview
Close out the four post-merge follow-ups to the radiometer-flux-cal epic with the
smallest correct change each: record the verified catalog-anchoring negative
result, encode host-redshift provenance as data, cache the live-route DSA beam
cube, guard one empty-slice warning, and swap nicknames→TNS (with a provisional-z
footnote) in the manuscript energy table.

## What we're NOT doing
- **Not** adding any burst to the `CATALOG` cross-check — research proved no
  *published* fluence exists beyond zach/whitney/oran (Law+2024 is Feb–Oct 2022
  only). Adding a model value would fabricate a "published" anchor.
- **Not** tightening the float/fluxcal gate (C3) — the mixed path is intentional
  and covered by `test_mixed_scalar_and_fluxcal_opens_gate`.
- **Not** editing `wilhelm_twoscreen_fig.py` (stale z comment) — it is in the
  active scattering-refit lane's subtree; reported in research, not touched here.
- **Not** writing `observations.tex`/`budget.tex` prose — needs sightline budget
  numbers that do not exist yet.

## Phase 1 — Code hygiene (FLITS)
- **C1** `analysis/dsa_beam.py`: import `functools.lru_cache`; decorate
  `load_power_beam` with `@lru_cache(maxsize=2)` so the 345 MB cube + arrays are
  read once per path, not once per channel. Behaviour identical (read-only
  consumers). Verify: existing `dsa_beam.py:_check()` and the beam tests still
  pass; `test_dsa_sigma_jy_constant_at_boresight` unaffected (uses a lambda).
- **C2** `analysis/flux_cal.py:261`: guard the nanmedian —
  ```python
  if valid.any():
      valid = valid & np.isfinite(ns) & (ns > 1e-6 * np.nanmedian(ns[valid]))
  ```
  all-False `valid` stays all-False, no warning. Behaviour identical otherwise.

## Phase 2 — Redshift provenance as data (FLITS)
- Add to `analysis/calculate_burst_energies.py` (near `PLACEHOLDER_Z`):
  ```python
  # Host-redshift provenance (nick -> (quality, source)). All 8 E_iso hosts are
  # spectroscopic; hamilton/chromatica have no published host paper yet (value
  # internal, provenance TBD). Sharma+2024 (2409.16964) Gold; Connor+2024
  # (2409.16952) Keck/MOSFIRE for wilhelm. See research-energetics-followups.md.
  Z_PROVENANCE = {
      "zach": ("spec", "Sharma+2024 Keck/LRIS"),
      "whitney": ("spec", "Sharma+2024 Keck/LRIS"),
      "oran": ("spec", "Sharma+2024 Keck/LRIS"),
      "isha": ("spec", "Sharma+2024 P200/DBSP"),
      "phineas": ("spec", "Sharma+2024 Keck/DEIMOS"),
      "wilhelm": ("spec", "Connor+2024 Keck/MOSFIRE"),
      "hamilton": ("spec-provisional", "unpublished host; provenance TBD"),
      "chromatica": ("spec-provisional", "unpublished host; provenance TBD"),
  }
  ```
- In `compute()`, after `row = {...}` add `row["z_src"] = Z_PROVENANCE.get(nick, ("unknown", ""))[0]`
  so `burst_energies.json` carries the provenance.
- **Test-first** in `tests/test_burst_energies_fluxcal.py`:
  ```python
  def test_z_provenance_flags_unpublished_hosts():
      from analysis.calculate_burst_energies import Z_PROVENANCE
      energy = {"zach","whitney","oran","isha","phineas","wilhelm","hamilton","chromatica"}
      assert energy <= set(Z_PROVENANCE)
      provisional = {n for n, (q, _) in Z_PROVENANCE.items() if q.endswith("provisional")}
      assert provisional == {"hamilton", "chromatica"}
  ```

## Phase 3 — Record the catalog-anchoring negative result (FLITS)
- Append to `analysis/burst_energies/CALIBRATION_REVIEW.md` a short subsection:
  Law+2024 Table 1 is the only published Jy·ms DSA-110 fluence catalog and covers
  only Feb–Oct 2022 (zach/whitney/oran are the only co-detections in it); no
  later DSA-110 paper republishes fluences, so the cross-check cannot be widened
  with published values — and the redshift provenance table from Phase 2.

## Phase 4 — Manuscript (Faber2026 `main`)
- `sections/results.tex` `deluxetable*`: replace the nickname Burst column with
  TNS names (mapping in research doc); delete the `% TODO swap … TNS` comment;
  add `\tablenotetext` flags on FRB 20230913A (hamilton) and FRB 20240203A
  (chromatica) — "host redshift not yet published; value provisional." Build with
  `latexmk -pdf` (or `pdflatex`) and confirm no new errors.

## Success criteria
### Automated
- `pytest tests/test_flux_cal.py tests/test_burst_energies_fluxcal.py -q` green,
  including the new `test_z_provenance_flags_unpublished_hosts`.
- `python analysis/calculate_burst_energies.py --check` prints `self-check OK`.
- `python analysis/flux_cal.py` and `python analysis/dsa_beam.py` self-checks pass
  (beam self-check skips if the cube is unstaged — acceptable).
- `ruff check analysis/ tests/` clean on touched files.
- `burst_energies.json` rows carry a `z_src` field (when data is staged).
### Manual
- Faber2026 builds; the energy table shows TNS names and the two provisional
  footnotes render.

## References
- `research-energetics-followups.md`, `validation-radiometer-flux-cal.md`,
  `analysis/burst_energies/CALIBRATION_REVIEW.md`
