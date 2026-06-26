# Research: E_iso energetics follow-ups (catalog anchoring, redshift provenance, code hygiene)

> Internal + external research. Codebase state: `feat/energetics-followups` @ `2d91773`
> (= `origin/main`), 2026-06-23. Feeds `plan-energetics-followups.md`.

## Scope

Four follow-ups surfaced by the radiometer-flux-cal validation
(`validation-radiometer-flux-cal.md`) and the post-merge review:

1. **Catalog anchoring** — can the absolute-flux cross-check be widened beyond
   the 3 bursts (zach/whitney/oran) it covers today?
2. **Redshift provenance** — is each of the 8 E_iso host redshifts spectroscopic
   or photometric, and is that quality recorded anywhere?
3. **Code hygiene** — three candidate cleanups in the flux-cal modules.
4. **Manuscript** — TNS-name swap and provisional-z flagging in the energy table.

Dimension: **both**. Internal for the code/config state and the repo's own
burst metadata; external (published DSA-110 / CHIME host catalogs) for the
fluence and redshift provenance.

## Codebase findings

### Catalog cross-check, as built
- The only absolute-scale anchor is `tests/test_flux_cal.py:111`
  (`test_dsa_fluence_matches_catalog_scale`) with
  `CATALOG = {"zach": 16.2, "whitney": 26.2, "oran": 13.2}` Jy·ms (Law+2024
  Table 1), asserted only to a **factor-of-3 sanity band** (`:128`) because the
  catalog boxcar-matched-filter fluence and the FLITS linear band integral are
  different estimators (`:114-116`).
- `analysis/burst_energies/CALIBRATION_REVIEW.md:339-343` already states the
  scope limit: redshift quality and the catalog anchor are inherited, not
  independently flagged.

### Redshift storage — no quality flag
- All 8 E_iso host redshifts come from one place:
  `galaxies/foreground/config.py:16-29`, `TARGETS = [(name, RA, Dec, z), …]`. The
  tuple has **no spec/phot field**.
- `analysis/calculate_burst_energies.py:107-109` `load_redshifts()` returns a
  bare `{nick: z}`; the only quality mechanism is `PLACEHOLDER_Z = 1.0`
  (`:63`, `:156`) that excludes freya/mahi/johndoeii (z unknown).
- The repo's own nick↔TNS↔host_z map is
  `scratch/codetection/foreground_catalog.csv` (`burst,tns,…,host_z,…`). Its
  `redshift_source` column is for the **foreground intervening galaxies**, not
  the host — so it does not record host spec/phot either.

### Code-hygiene sites (verified against source)
- **C1 — beam cube reloaded per channel (live route).**
  `analysis/dsa_beam.py:71-79` `beam_gain()` calls `load_power_beam(path)`
  (`:77`), which opens the ~345 MB `DSA110_beam_1.h5` (`:60-66`) on **every
  call**, then rebuilds a `RegularGridInterpolator` (`:78`).
  `analysis/flux_cal.py:83` `dsa_sigma_jy()` calls it once per channel in a list
  comprehension (~6160 DSA channels). It is on the **live E_iso route**:
  `joint_band_fluence_jy_ms_hz(nick, "D")` → `dsa_sigma_jy(…, beam_gain)`
  (`flux_cal.py:288-290`). So a single DSA-band fluence reloads 345 MB ~6160×.
- **C2 — empty-`valid` nanmedian.** `analysis/flux_cal.py:261`:
  `valid = valid & np.isfinite(ns) & (ns > 1e-6 * np.nanmedian(ns[valid]))`. If
  the incoming `valid` is all-False (all channels RFI-masked), `ns[valid]` is
  empty → `np.nanmedian` raises an all-NaN-slice `RuntimeWarning` and returns
  `nan` (the result is still correct: all-False). Warning noise only.
- **C3 — float/fluxcal gate.** `analysis/calculate_burst_energies.py:127-140`
  `_band_jy` accepts `None` / `"fluxcal"` / a float scalar; the gate
  `calibrated = cal_C and cal_D` (`:166`) opens when **both** bands have any Jy
  scale. `tests/test_burst_energies_fluxcal.py:39-47`
  (`test_mixed_scalar_and_fluxcal_opens_gate`) asserts a float-C + fluxcal-D
  combination opens the gate **on purpose**. Tightening to require both==fluxcal
  would break that test for no live-config benefit (`configs/telescopes.yaml`
  sets both bands `fluxcal`).

### nick ↔ TNS (authoritative, for the manuscript swap)
From `foreground_catalog.csv` `host_z` rows (+ Sharma+2024 for isha, absent
from that file):

| nick | TNS | host z |
|---|---|---|
| zach | FRB 20220207C | 0.0430 |
| whitney | FRB 20220310F | 0.4790 |
| oran | FRB 20220506D | 0.3005 |
| isha | FRB 20221113A | 0.2505 |
| phineas | FRB 20230307A | 0.2710 |
| wilhelm | FRB 20221203A | 0.5100 |
| hamilton | FRB 20230913A | 0.3024 |
| chromatica | FRB 20240203A | 0.0740 |

## Prior-art findings

### Q1 — Law+2024 fluence catalog (arXiv:2307.03344, ApJ 967, 29)
- Table 1 reports per-burst **Fluence [Jy·ms]**, defined as the boxcar /
  S/N-maximizing matched-filter fluence "measured from high-resolution data to
  contain 90% of the flux density" (caption) — i.e. exactly the estimator the
  FLITS docstring already flags as biased low for scattered bursts.
- **Decisive scope limit:** Table 1 is **11 FRBs, all from the Feb–Oct 2022
  science-commissioning window.** The full list: Zach 20220207C (16.2), Alex
  20220307B (3.2), Whitney 20220310F (26.2), Mark 20220319D (8.0), Quincy
  20220418A (4.2), Oran 20220506D (13.2), Jackie 20220509G (5.8), Ansel
  20220825A (5.8), Elektra 20220914A (2.6), Etienne 20220920A (3.9), Juan
  20221012A (5.1). Only **zach/whitney/oran** are FLITS co-detections.
- **Disconfirming check (ran):** every FLITS co-detection outside Feb–Oct 2022
  is structurally excluded — isha (Nov 2022), wilhelm (Dec 2022), phineas/freya
  (2023), hamilton (Sep 2023), chromatica/mahi/casey (2024). Later DSA-110
  papers do **not** republish Jy·ms fluences: Sherman+2024 (2308.06813,
  polarimetry, same 2022 sample, no fluence); Sharma+2024 (2409.16964, 30 hosts
  Feb 2022–Nov 2023, host astrometry/z/P_host only — **no fluence column**). So
  no later catalog can supply an anchor either.
- **Verdict:** the cross-check **cannot be widened** with any *published* value.
  The CALIBRATION_REVIEW fluences for the other bands are FLITS radiometer-model
  outputs, not catalog values; promoting them into `CATALOG` would fabricate a
  "published" anchor. This refutes the earlier adversarial claim that
  isha/phineas/wilhelm were "published but missing."

### Q2 — host-redshift spectroscopic vs photometric (per burst)
All 8 E_iso hosts are **spectroscopic**; none photometric. Six are published,
two are not:

| nick | TNS | z | quality | source |
|---|---|---|---|---|
| zach | FRB 20220207C | 0.0430 | spec (published) | Sharma+2024 Gold, Keck/LRIS |
| whitney | FRB 20220310F | 0.4790 | spec (published) | Sharma+2024 Gold, Keck/LRIS |
| oran | FRB 20220506D | 0.3005 | spec (published) | Sharma+2024 Gold, Keck/LRIS |
| isha | FRB 20221113A | 0.2505 | spec (published) | Sharma+2024 Gold, P200/DBSP (coord match) |
| phineas | FRB 20230307A | 0.2710 | spec (published) | Sharma+2024 Gold, Keck/DEIMOS |
| wilhelm | FRB 20221203A | 0.5100 | spec (published) | Connor+2024 (2409.16952) Keck/MOSFIRE; Hussaini+2025 DM–z table |
| hamilton | FRB 20230913A | 0.3024 | **spec, unpublished** | no host paper; repo-internal value, provenance TBD |
| chromatica | FRB 20240203A | 0.0740 | **spec, unpublished** | no host paper; repo-internal value, provenance TBD |

Sharma+2024 measures every host z spectroscopically (pPXF on Keck-LRIS/DEIMOS,
P200/DBSP); none of its 30 hosts is photometric. hamilton (Sep 2023) and
chromatica (Feb 2024) post-date that sample and have no published host paper —
their z is repo-internal and should be flagged provisional, not presented as
catalog-validated.

- **Light observation (deferred to a separate lane, not edited):**
  `analysis/scattering-refit-2026-06/wilhelm_twoscreen_fig.py:26` still comments
  "z~0.47 from a Macquart-relation DM estimate" — stale, since `TARGETS` already
  uses the real spec z=0.5100 (Connor+2024). That file is under the active
  scattering-refit lane's subtree, so it is reported here rather than edited.

## Synthesis & gaps

- **#1 is a non-gap.** Action is to *record the verified negative result*
  (Law+2024 window; no widening possible) so it is not re-litigated; the test and
  the manuscript caveat are already correct. No code/test change.
- **#2 is real but small.** No photometric z to demote; encode the
  spec-vs-unpublished provenance as data (so it stops being prose-only) and flag
  hamilton/chromatica as provisional in the manuscript table.
- **#3 reduces to two cleanups.** C1 (cache the beam cube — a live-route
  ~345 MB×6160 reload) and C2 (guard the empty-`valid` nanmedian). C3 is a
  no-op: the mixed gate is intentional and tested.
- **#4** is the TNS swap + provisional footnote in `Faber2026/sections/results.tex`;
  the `observations.tex`/`budget.tex` stubs need the sightline DM/scattering
  budget numbers that do not exist yet, so writing them now would be fabrication
  — deferred.

## References
- `docs/rse/specs/validation-radiometer-flux-cal.md`, `…/implement-radiometer-flux-cal.md`
- `analysis/burst_energies/CALIBRATION_REVIEW.md`
- Law et al. 2024, ApJ 967, 29 (arXiv:2307.03344) — DSA-110 first FRB+host catalog
- Sharma et al. 2024 (arXiv:2409.16964) — DSA-110 host galaxies
- Connor et al. 2024 (arXiv:2409.16952); Hussaini et al. 2025 — wilhelm host z
