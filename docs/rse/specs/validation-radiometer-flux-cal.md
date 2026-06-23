# Validation Report: Radiometer flux calibration (S/N → Jy) for FLITS burst energetics

> Validated against [`plan-radiometer-flux-cal.md`](plan-radiometer-flux-cal.md) and
> [`implement-radiometer-flux-cal.md`](implement-radiometer-flux-cal.md) at
> `feature/radiometer-flux-cal` @ `e81244f` on 2026-06-23. Every automated command below
> was **re-run fresh** in a clean worktree (gitignored `.npy`/beam data symlinked in) by
> independent validator agents; numbers are observed output, not the closeout's claims. Two
> stale docstrings found during review were fixed in the same commit that adds this report.

**Verdict: ✅ PASS with one important caveat.** The implementation is correct, reproducible, and
matches the plan. All 19 targeted tests pass, both analytic self-checks pass, the gate logic and
(1+z) k-correction are exact to machine precision, and the model-based fluence is derivable from the
loader/model code (not a fudge). The single substantive finding is that the **absolute-scale
cross-check against the published catalog is narrower than the closeout implies** — it constrains 3
of the sample's bursts, not all of them.

## Implementation Status

| Phase | Claimed | Verified |
|---|---|---|
| Phase 5 (+5c-B): DSA per-channel cal, config seam, γ_D refit | ✅ | ✅ code present, tests pass, refit figure-reviewed |
| Phase 6: CHIME beam + SEFD | ✅ | ✅ `chime_beam.py` + `chime_sefd.csv`, 5 beam tests pass |
| Phase 7: open the E_iso gate | ✅ | ✅ gate opens, table emitted with ± errors, 19 tests pass |

## Automated Verification Results

All commands run via the agent-safe conda env `flits` on `feature/radiometer-flux-cal` @ `e81244f`.

- ✅ `pytest tests/test_flux_cal.py tests/test_burst_energies_fluxcal.py tests/test_chime_beam.py` —
  **19 passed, 0 failed, 0 skipped** (10 + 4 + 5).
- ✅ `test_both_bands_emit` exists and **passes** (gate opens; k-correction identity; finite +ve error).
- ✅ `test_joint_band_fluence_matches_catalog_scale` exists, **passes, NOT skipped** (DSA `.npy` staged).
  Observed model-based ratios vs Law+2024: **oran 0.99×, zach 1.27×, whitney 2.15×** (all in (⅓, 3)).
- ✅ `python analysis/flux_cal.py --check` → `self-check OK: radiometer noise + flat-band integral …`.
- ✅ `python analysis/calculate_burst_energies.py --check` → `self-check OK: integral matches
  quadrature; energy oracle, k-correction, and gate exact`.
- ✅ `ruff check analysis/flux_cal.py analysis/chime_beam.py` → `All checks passed!`.
- ✅ Files exist: `flux_cal.py`, `chime_beam.py`, `dsa_sefd.csv`, `dsa_pointing.csv`, `chime_sefd.csv`.
- ✅ `burst_energies.tex` contains **0** occurrences of "calibration pending"; it is a populated
  8-row energy table.
- ✅ Full run emits all 8 bursts; every row has a **positive, finite** `E_iso_erg_err`.
- ⚠️ **Energy-range criterion 7/8:** `wilhelm = 1.1146×10⁴¹ erg` exceeds the nominal `10⁴¹` upper
  bound (others all in range). Known, user-accepted: wilhelm is the most distant/luminous sightline
  (z=0.51); 1.1×10⁴¹ erg is physically plausible at the luminous end of the FRB distribution.
- ✅ **Reproducible & deterministic:** two consecutive runs produce **bit-identical** `burst_energies.json`
  (e.g. `zach 4.60101731699249e+38`, `wilhelm 1.1145622829259924e+41`); no RNG/network/time
  dependence. Inputs are exactly the committed `joint_json` + the 3 CSVs + `configs` + local `.npy`;
  no hidden state. Both SEFD CSVs carry per-row provenance columns and physically-sane values.

**Reported numbers reproduce:** the closeout's anchor values (chromatica 1.7×10³⁹, wilhelm 1.1×10⁴¹)
match the fresh run.

## Code Review Findings

All four reviewed areas **match the plan's design** (`matches_plan = true`); no critical or important
code defects. Confirmed:

- **`flux_cal.py`** — `joint_band_fluence_jy_ms_hz` applies the `/noise_std` bandpass-unit correction,
  restricts the integral to the fit's valid (non-RFI) channels, references `c0` to
  `median` of the **full** channel grid, and uses `n_pol=2` with the DSA measured beam / CHIME G≈1.
- **`calculate_burst_energies.py`** — both-bands-or-nothing gate (partial cal leaks no energy,
  verified both directions), error = quadrature of `c0` width and `BAND_SYS_DEX`, `±` rendered by
  `_tex_val_err`, k-correction identity asserted in `_check()`.
- **`chime_beam.py` + `telescopes.yaml`** — SEFD = 2k_B·Tsys/(η·A) ≈ 34.5 Jy, boresight gain = 1,
  both bands routed to `fluxcal`.
- **Tests** — every Testing-Strategy item has a corresponding test.

**Nice-to-have issues (none blocking):**
- *(fixed in this commit)* `calculate_burst_energies.py` module docstring described only the legacy
  scalar `flux_jy_per_unit` path; the `_band_jy` NotImplementedError still said "CHIME is Phase 6".
  Both updated to reflect the live `fluxcal` model-based path.
- `flux_cal.py:261` `np.nanmedian(ns[valid])` edge-fragility if a burst's valid mask were empty
  (not reachable for the current sample).
- The model-based fluence assumes the area-conserving M2/M3 scattering kernel (`∫model dt = c0`) —
  correct for the fits in use, documented in the code.
- `chime_beam.sefd_zenith_jy()` returns 34.516 Jy; CSV/docstring round to 34.5 Jy (cosmetic).
- The live-config half-set guard (`calculate_burst_energies.py`) compares `is None` equality, so a
  one-band-`fluxcal`/one-band-`None` mix would not be flagged by that specific guard (the gate itself
  still prevents an energy leak).
- `test_sn_spectrum_synthetic` asserts only sanity bounds; the catalog cross-check tests `pytest.skip`
  silently when `data/dsa` `.npy` is unstaged (acceptable, but invisible in a data-less CI).

## Adversarial Verification

Four skeptic agents each tried to **refute** a key claim. Results: **3 HOLD, 1 REFUTED.**

- ✅ **HOLD — "all 8 energies physical; the valid-channel mask is a correct fix, not a bug-hider."**
  Confirmed the mask is load-bearing: wilhelm-CHIME **without** the mask = literal `inf` (5 of 16
  channels have `noise_std == 0`); **with** it, a stable 1.40×10⁹ Jy·ms·Hz → E_CHIME 9.9×10³⁹ erg.
  The mask drops channels in one band only and the energies are robust.
- ✅ **HOLD — "the `/noise_std` correction is physically correct, not a unit fudge."** Read
  `io.py:_bandpass_correct` (z-scores by full-window off-pulse std) and `burstfit.py`: the joint
  fit's data is dimensionless S/N at the per-sample level, downsampling drops the noise to
  `noise_std ≈ 0.04–0.06` (empirically median 0.057 for zach DSA), so dividing `c0` by `noise_std`
  to recover S/N is derivable from the loader, not invented.
- ✅ **HOLD — "k-correction identity and the gate are correct."** `--check` exits 0; identity holds
  to `max rel. err 1.6×10⁻¹⁶` across all 8 rows; per-band CHIME+DSA sum equals `E_iso_erg` exactly;
  the gate leaks no energy in either `{C:None,D:1.0}` or `{C:1.0,D:None}`.
- ❌ **REFUTED — "agreement within ~2× of Law+2024 validates the absolute flux scale."** The
  cross-check is real but **narrow**: only **3 of the 7** Law+2024-published bursts (oran/zach/whitney)
  carry catalog fluences in the repo, so 4 published co-detections are untested; the factor-3 gate is
  loose; and several untested model fluences are large (mahi ≈ 468, freya ≈ 156 Jy·ms·band-avg).
  *Mitigant:* mahi/freya/isha-without-z are excluded from the **E_iso table** (placeholder redshifts),
  so the high untested values do not enter the published energies — but the absolute scale is
  **empirically confirmed for 3 sightlines, not the whole sample.**

## Manual Testing Required

These were **confirmed by the user earlier this session** (re-listed for the record):
- ✅ Energies in 10³⁸–10⁴¹ erg (wilhelm 1.1×10⁴¹ at the upper edge — accepted).
- ✅ Bandpass diagnostic (`refit_calibrated.review.json`) — figure-reviewed `match`.
- ✅ DSA SEFD sanity vs the per-element measurement.
- ✅ Published-fluence cross-check (for the 3 catalog bursts) within ~2×.

## Recommendations

**Critical:** none.

**Important:**
1. **Scope or strengthen the absolute-scale claim.** Either (a) add the Law+2024 catalog fluences for
   the other published co-detections and extend `test_joint_band_fluence_matches_catalog_scale`, or
   (b) state explicitly in `CALIBRATION_REVIEW.md` / the manuscript that the absolute scale is
   catalog-validated for oran/zach/whitney only. The E_iso table does not depend on the untested
   high fluences (placeholder-z bursts are excluded), so this is a claim-precision issue, not a number
   bug — but the manuscript table inherits a scale validated on 3 sightlines.

**Nice-to-have:**
2. Tighten the half-set config guard to also catch a `fluxcal`/`None` mix.
3. Strengthen `test_sn_spectrum_synthetic` beyond sanity bounds; make the catalog tests `xfail`/warn
   (not silent skip) when data is absent, so a data-less CI surfaces the gap.

**Follow-up (already tracked in the closeout):**
4. Rebase `feature/cluster-catalog-engine` onto `feature/radiometer-flux-cal` to fold in Phase 7.
5. Push branches / open PRs (held for sign-off).
6. Verify the Faber2026 bib metadata (`Law2024` vol/page, `Michilli2021` authors) against ADS.

## References

- Plan: [`plan-radiometer-flux-cal.md`](plan-radiometer-flux-cal.md)
- Implementation: [`implement-radiometer-flux-cal.md`](implement-radiometer-flux-cal.md)
- Research: [`research-chime-singlebeam-flux-units.md`](research-chime-singlebeam-flux-units.md)
- Calibration review: [`../../../analysis/burst_energies/CALIBRATION_REVIEW.md`](../../../analysis/burst_energies/CALIBRATION_REVIEW.md)

---

**Validation completed by AI Assistant on 2026-06-23** (13-agent workflow: 5 re-verify, 4 code-review,
4 adversarial; commit `e81244f` + this report).
