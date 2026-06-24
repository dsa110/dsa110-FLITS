# Implementation Summary: activate association pillars 2 & 4 with CHIME-side DM + localization

> **⚠️ PILLAR 2 RETRACTED (2026-06-24).** The CHIME structure-DMs below are INVALID — the extraction
> had a 1e-3*K_DM inter-channel unit bug (1000×) and the DM-phase curves never actually peaked (see
> [audit-chime-side-dm.md](audit-chime-side-dm.md)). DMs nulled on main; rebuild on library
> dedispersion pending. **Pillar 4 (positions) is independent and stands.** The 2-real/7-marginal/
> 3-noise headline below is the retracted state, kept for provenance.

---
**Date:** 2026-06-23
**Status:** Phases 1-3 complete (automated-verified + figure-reviewed); Phase 4 (verify + PR) pending a scientific decision
**Plan:** [plan-chime-side-dm-localization.md](plan-chime-side-dm-localization.md)
**Research:** [research-chime-side-dm-localization.md](research-chime-side-dm-localization.md)
**Branch:** `feat/chime-side-dm-localization`
---

## Outcome (headline) — RFI-mask recipe, re-extraction bi27l1ft8
- **Pillar 4 (position) — STRONG:** all **12/12** CHIME tied-beam positions (`tiedbeam_locations`) are
  consistent with the DSA arcsec positions, separations **0.35–2.26 arcmin** (all < the 0.1° stated
  CHIME radius). An independent positional confirmation of every co-detection.
- **Pillar 2 (DM) — HONEST BUT WEAK:** **9/12** CHIME structure-DMs measured, **all consistent** with the
  DSA DM within **≤1.4σ**; 3 nulled as noise (isha, phineas, mahi — multi-peaked/ambiguous DM-phase
  curves, no trustworthy peak). The figure-review gate confirms only **2/12 are confidently real**
  (zach, freya — visible burst + dominant interior DM-phase peak); **7 are marginal** (valid interior
  curve, no eye-visible burst). σ ~2.5–4.3 pc cm⁻³; every CHIME singlebeam is low-S/N (~3–4) — the
  structure-DM *confirms* the DSA DM for 9 sightlines, visibly for 2.

## Plan adherence / deviations
- **Method mechanics (settled in Phase 1, not pre-known):** `DMPhaseEstimator` de-disperses with a sign
  such that real (physically dispersed) data is recovered in the **time-flipped** orientation; its
  `get_dm()` quadratic vertex-fit is **unreliable** on the shallow curves of these scattered bursts
  (returns the grid edge, σ=0), so the extraction uses a **robust peak** (argmax of the mean curve;
  σ = std of bootstrap-curve argmaxes via `est._bs_curves`).
- **Host→docker course-correction:** an initial host-only attempt (`tiedbeam_power` + numpy incoherent
  dedispersion) produced noise — **intra-channel smearing is ~13 ms** at CHIME's bottom band for these
  DMs, removable only by *coherent* dedispersion. Reverted to the plan's docker path (which the prior
  TOA work already validated); confirmed on zach (SNR~44 clean burst).
- **Tractable recipe:** coherent_dedisp(time_shift=False) → numpy roll-align (windowing) → tight
  **DM-independent** window → DM-phase on a **residual** grid around DM_c (so high-DM bursts with ~18 ms
  sweeps don't need huge windows) → flip orientation → robust peak.
- **Real circular import fixed:** `dispersion/dmphasev2` imports `flits.common.constants` while
  `scattering/dm_preprocessing` imported `DMPhaseEstimator` at module top → importing `dispersion` first
  deadlocked. Made that import lazy (Phase 1).
- **Confidence encoding (not in the original plan):** the figure-review classification (real/marginal/
  noise) is carried as `dm_confidence` per burst; noise (isha) → `dm_chime=None` → `dm_agreement`
  null+reason (no fabrication).

## Phases
- **Phase 1 ✅** `ff66c0a` — `tests/test_dmphase_recovery.py` (interior-peak recovery; railed result fails)
  + circular-import fix.
- **Phase 2 ✅** (docker, h17, off-repo) — `scripts/extract_chime_side_inputs.py` +
  `scripts/dmphase_standalone.py` (vendored, K_DM inlined) → `crossmatching/chime_side_inputs.json`
  (12 rows) + 12 diagnostic PNGs + `figures.review.json` (figure-reviewer gate, RFI-mask recipe:
  **2 real / 7 marginal / 3 noise**).
- **Phase 3 ✅** `f45690a`, `4d78f57` — `position_consistent`→`position_agreement` (dict); assembler
  consumes `chime_side_inputs.json`; `dm_confidence` propagated; report regenerated.

## Files
**Created (repo):** `crossmatching/chime_side_inputs.json`, `tests/test_dmphase_recovery.py`.
**Modified (repo):** `crossmatching/association.py` (position_agreement, chime inputs, dm_confidence),
`tests/test_association.py` (+2 tests), `crossmatching/association_report.json` (regenerated),
`scattering/scat_analysis/dm_preprocessing.py` (lazy import).
**Off-repo (h17, `/data/.../scripts`):** `extract_chime_side_inputs.py`, `dmphase_standalone.py`
(extraction provenance; mirrors the existing TOA-script convention of living under the data root).

## Verification
- ✅ `pytest tests/test_association.py tests/test_dmphase_recovery.py` → 15 pass; ruff clean.
- ✅ figure-review gate satisfied (`figures.review.json`, all 12 `match`, fresh on RFI-mask recipe).
- ✅ `python -m crossmatching.association` → `dm_active=9/12`, sum_mu=5.460e-08 (pillar 1 unchanged).
- ✅ golden `toa_crossmatch_results.json` untouched (git clean).
- ✅ positions deterministic (new ra/dec == old to 0 deg); all 12 chime_ids match the fixture.

## Decision (Phase 4): Finalize as-is (user)
PR the honest **2-real / 7-marginal / 3-noise** DM + **12/12-position** result (noise nulled, not
fabricated). The alternative — chasing the faint bursts with TOA-seeded windows — has uncertain payoff;
the RFI-mask re-extraction already un-railed isha and dropped the normalization artifact, and the CHIME
singlebeam is simply low-S/N for these sightlines.

## Finalization guardrails (re-extraction bi27l1ft8, RFI-mask + raw-input recipe)
Decision: **Finalize as-is** (user). Re-extraction supersedes the first-pass numbers. Enforce on completion:
1. **Figure-review from scratch** — overwrite old `figures.review.json`; reviewer judges new-recipe figures with
   NO inherited labels. Self-enforce: figures are off-repo (`/data/...`), so the repo Stop hook may not catch a
   stale review.
2. **`noise` ⇒ `dm_chime=None`** even when `interior=True` (interior only means the peak isn't grid-railed). A
   no-visible-burst peak is on noise and must not feed pillar 2 — null it, don't fabricate a DM agreement.
3. **Pre-merge integrity** — all 12 rows present, none `status: error`, every `chime_id` matches the fixture,
   new ra/dec == old to ~1e-6 (positions are deterministic pointing metadata; drift = bug).
4. **Tests track reality** — set `dm_active`, the isha tag, and the `real` set to what the figures actually yield;
   never force-fit to preserve the old 11 / `{zach,freya,casey}` headline.
5. **Honest headline** — re-classification result stated plainly here + in the PR body (incl. if freya/others move).
6. **Hard pre-PR gate** — `pytest` green, `ruff` clean, golden `toa_crossmatch_results.json` git-clean, figure-review
   written. Commit only repo files + co-author line; off-repo scripts stay off-repo.

## Remaining
- Adversarial verification (Workflow) of the report assembly + determinism + no-fabrication.
- PR → main per the established pattern.
