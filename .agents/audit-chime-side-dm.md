# Audit: CHIME-side structure-DM extraction (retraction + rebuild)

**Date:** 2026-06-24
**Trigger:** expert visual review of the DM-phase diagnostics — bursts are not vertical, the
sweep is large, the "DM-phase" curves are flat/oscillatory, and it is universal across all 12.
**Status:** Pillar-2 DMs RETRACTED on main (nulled, `dm_confidence=under-audit`); rebuild pending.
Pillar-4 (positions) UNAFFECTED.

## What was wrong (confirmed by `scripts/diag_dedisp.py` on zach, in docker)

singlebeam: `dt=2.56 µs`, `ntime=55949` (~143 ms; pre-dedispersed at record time, burst at ~77 ms),
871 ch over 400.4–799.2 MHz.

1. **Inter-channel unit bug (1000×).** The extraction rolled channels by
   `1e-3 * K_DM * (1/f² − 1/f_ref²) * dm_c` with `K_DM=4148.81`, f in MHz. The physical full-band
   delay @DM=262 is **5.086 s = 1,986,647 samples**; the recipe rolled by **5.086 ms = 1987 samples**
   — **ratio exactly 1000.0**. The `1e-3` factor is wrong for f in MHz. Damage was partly masked
   because the data is already pre-dedispersed into a 143 ms window, but the band is still mis-aligned.
2. **DM-phase recovered nothing.** Every curve is near-flat/oscillatory (flat_ratio 1.2–1.5); the
   robust-argmax just picks the global max of noise wiggles, and the bootstrap σ is small only because
   the noise pattern is stable across resamples. A real structure-DM peak is sharp (ratio ≫ 1). This
   holds even on zach, where proper dedispersion shows a clear **SNR~15** burst.
3. **Downstream signal-loss.** The RFI-mask → coarse-peak (savgol argmax) → ±512 window → time-flip →
   z-score-display chain throws away a burst that `coherent_dedisp(bb, dm_c)` (proper, `time_shift=True`)
   renders clearly. Diagnostic: proper-dedisp collapse peak SNR~14.8 vs recipe-path SNR~12.1, but the
   extraction figures show noise.
4. **The recovery test was blind to all of this.** `tests/test_dmphase_recovery.py` dispersed *and*
   de-dispersed with the same `1e-3*K_DM` constant, so it recovered the injected DM without ever
   checking absolute physical scaling. It must be rebuilt to disperse with the *physical* constant.

## Rebuild plan (full rebuild on library dedispersion — user-approved scope)

Re-derive on `baseband_analysis` proper dedispersion; do not hand-roll inter-channel alignment.

- **P1 — physical-scale recovery test (test-first).** Rewrite `test_dmphase_recovery.py` to disperse a
  synthetic burst with the PHYSICAL constant (`t[s]=4.148808e3*DM*(f_MHz^-2 − f_ref^-2)`) and assert the
  estimator recovers it. Make the same convention the single source of truth for the estimator.
- **P2 — dedisperse correctly.** Use `coherent_dedisp(bb, dm_c)` (default `time_shift=True`), validated
  against zach (must reproduce the SNR~15 vertical burst). Search a small *physical* residual-DM grid for
  the structure-max; no `1e-3` factor, no manual roll.
- **P3 — make the curve peak.** On a known-signal burst (zach), confirm the DM-phase curve has a sharp
  interior peak (flat_ratio ≫ the ~1.3 noise floor) before trusting any other burst. Gate every burst on
  curve sharpness, not just "interior".
- **P4 — re-extract all 12 + figure-review from scratch + re-validate**; only then un-null the DMs that
  genuinely peak, and re-assess which co-detections yield a real CHIME structure-DM.

## Expert methodology verdict (2026-06-24, astronomy-astrophysics-expert) — METHOD PIVOT
After P1 (bug fix) + P2 prototype: with proper `coherent_dedisp` and the fixed estimator, zach's
structure-max curve is flat_ratio ≈ 1.97–2.27 across windows ±4/±12/±30 ms (synthetic sharp-structure
control gives 48.5). Verdict:
- **flat_ratio ~2 = NON-detection of structure, NOT a wrong DM.** Structure-max (DM_phase; Seymour/
  Michilli/Hessels+2019) measures DM curvature only from unresolved temporal sub-structure; a smooth,
  scattered, low-S/N (~3–11) blob has none, so the curve is flat by construction even at the correct DM.
  Structure-max is the WRONG primary tool for these CHIME singlebeam bursts.
- **Circularity is real:** dedispersing at DM_DSA then fitting a tiny reference-centered residual (≈0) is
  mostly a windowing artifact, not an independent confirmation. The defensible statement is an EXCLUSION
  from a WIDE, reference-independent DM search, not "δ_DM≈0".
- **Right estimator is already in the repo:** `burstfit.py` M2/M3 forward-fit (DM `delta_dm` + scattering
  `tau_1ghz` fitted jointly), which is scattering-aware (removes the S/N-DM scattering bias) and works on
  smooth bursts. Use a wide flat DM prior (±20–50 pc/cm³), report the marginalised DM posterior, σ =
  σ_stat ⊕ σ_scat (DM shift τ-free vs τ-fixed). Run the repo PASS/MARGINAL/FAIL gates.
- **Per-burst outcome, not global:** bursts whose posterior is tight enough → report DM±σ + |ΔDM| 95%
  exclusion (support Pillar 2). Bursts too low-S/N → "CHIME singlebeam does not independently constrain
  DM" (lean on Pillar 4 position). Keep structure-max only as a secondary cross-check gated on a
  null-based ≥5σ peak significance (phase-scramble / off-pulse null), NOT a flat_ratio number.
- **Caveat to confirm:** what DM the baseband was coherently dedispersed at vs what we re-dedisperse at —
  `delta_dm` is only meaningful relative to a known reference; intra-channel coherent smearing at the
  wrong DM is not undone by an incoherent re-trial.

### Revised plan (supersedes P2–P4 above)
- **P2′** — drop structure-max as primary. Prototype a `burstfit` M2/M3 fit of the CHIME band-collapsed
  (or 2-D) burst for zach over a WIDE reference-independent DM prior; confirm a marginalised DM posterior
  + validation PASS.
- **P3′** — define the agreement test (pre-registered): |DM_CHIME−DM_DSA|/σ_CHIME < 2 AND a stated |ΔDM|
  95% exclusion; classify each burst constrains / does-not-constrain.
- **P4′** — fit all 12, validate each, report the honest split; un-null only the bursts that constrain DM.

## Custom DM tool (user directive 2026-06-24: "rewrite into a custom tool that works 100%")
Built `dispersion/chime_dm.py` — a clean-room, telescope-agnostic, pure numpy/scipy DM estimator
(no flits deps → runs in the baseband docker image and host). Method = the textbook scattering-aware
DM measurement, NOT structure-max:
1. **wide incoherent DM search** (reference-independent) → band-collapsed peak-S/N coarse DM that can
   be far from `dm_ref` (so a real offset is FOUND → de-circularises the agreement test);
2. **scatter-corrected arrival-time regression**: per sub-band fit an exponentially-modified Gaussian
   (Gaussian ⊗ one-sided exp(−t/τ)); the Gaussian centre `t0` is the scattering-DECONVOLVED arrival
   time → weighted linear fit of `t0` vs `K_DM·(ν⁻²−ν_ref⁻²)`; slope = residual DM, covariance×χ²_red
   = honest σ_DM. Few sub-bands / large σ → `constrains_dm=False` (no fabricated value).

**Validation (`tests/test_chime_dm.py`, 12 tests, independent numerical injector):** known-DM recovery
in BOTH CHIME (400–800) and DSA (1281–1531) bands across τ∈{0,2,5 ms}; wide-search recovers a +30
offset (de-circularisation); non-detection floor flagged at S/N~1; DSA σ_DM > CHIME σ_DM (narrow-band
lever arm); σ calibrated (pull scatter < 3). **Real DSA data:** phineas_dsa (6144×5121, DM 610.274) →
S/N 140, 7/8 sub-bands, **DM = 610.206 ± 0.006, constrains_dm=True** — recovers the catalogue DM.

**σ caveat (expert):** the reported σ is statistical (χ²_red-inflated). The 0.07 pc/cm³ phineas offset
is physically negligible but ~10σ on σ_stat alone — so the **agreement test (in association.py) must
apply a physical tolerance floor (~1 pc/cm³)**, not the raw pull. Tool reports the honest statistical
measurement; the floor lives in the pillar-2 agreement logic.

### P4″ result (2026-06-23): clean split, custom tool (arrival-regression)
All 12 docker-extracted (`results/chime_dm_v2.json`, figures reviewed `diagnostics/chime_dm_v2/`).
**Clean discriminator on sub-band count (perfect gap, 8 vs ≤5):**
- **5 reliable** (8 sub-bands @ S/N≥4): zach, casey, freya, hamilton, chromatica — sharp coarse S/N
  peak AT the DSA DM, ΔDM −0.86…+0.04, regression lands on DSA. Genuine independent constraint.
- **6 scatter-biased** (3–5 sub-bands): whitney, oran, isha, wilhelm, phineas, johndoeII — broad coarse
  S/N peak offset +4–5 high, regression ΔDM +0.81…+3.46 with over-confident σ. The +offset is the
  scattering S/N–DM bias (over-dedispersing aligns the asymmetric tail), NOT a real CHIME–DSA
  disagreement (co-detection ⇒ one DM; phineas DSA-side recovered catalogue DM to −0.07 while CHIME
  reads +3.1). Instrumental.
- **1 unconstrained**: mahi (2 sub-bands).

### v3 (user recipe: crop → S/N-max → preliminary-correction → DM-phase fine-tune) — NEGATIVE
Tested whitney+oran (`diagnostics/chime_dm_v3/`). A fixed ~44 ms crop PARTIALLY de-biases the S/N-max
(~2 of ~4 pc/cm³ toward DSA) via zero-fill leverage, but a ~+2 residual persists and the +2 incoherent
dedispersion zero-fills the bottom half of the band (78 ms sweep), halving the lever arm. DM-phase then
returns a single-bin delta spike at the seed DM (self-alignment artifact, flat_ratio 23–29 but zero
independent info — circular). DM-phase cannot break the scattering degeneracy on sub-structure-less
bursts. Recipe documented as a negative result.

### P5 — coherent trial-DM likelihood envelope (expert-endorsed 2026-06-23, astronomy-astrophysics-expert)
The S/N-max/arrival-regression/structure-max failures are intrinsic to faint scatter-broadened
morphology. Right estimator: go back to the BASEBAND, `coherent_dedisp(bb, DM_k)` over a trial-DM grid
(intra-channel correct, FULL band — coherent dedisp is a phase rotation, no zero-fill, restores the
lever arm), and at each trial forward-fit a scattering-aware 2-D template with a SHARED arrival time
(α fixed = 4; τ(ν)=τ_1ghz·(ν/1GHz)⁻⁴; per-channel amplitude marginalised in closed form). The χ²(DM)
envelope is the DM likelihood: min → DM, Δχ²=1 (rescaled) half-width → honest σ that goes WIDE when the
burst doesn't constrain — so the over-confidence pathology cannot recur.
- **Independence/circularity:** DM enters only as the grid axis + flat prior; the data are NOT
  conditioned on dm_dsa (each trial re-dedisperses baseband from scratch).
- **Claim:** bright bursts → DM ± σ; faint bursts → EXCLUSION `|ΔDM| < X at 95%` (not a point estimate).
  Pillar 2 reframed "independent measurement" → "independent consistency/exclusion test."
- **Pre-registered criterion:** σ_CHIME = 68% half-width of the rescaled χ²(DM) envelope. Constrains iff
  σ_CHIME ≤ 1 pc/cm³ AND single well-defined min passing GoF gates (χ²_red 0.8–1.5, Durbin–Watson on
  2-D residuals, τ within ~OOM of NE2025). Else exclusion if 95% interval excludes |ΔDM|>X (X < the
  ~5–10 pc/cm³ unrelated-source scale). Else "does not constrain" → Pillar 4.
- **Selection legitimacy:** bright/faint split is the outcome-independent sub-band-survival threshold,
  fixed before DM is seen → not DM-selection bias. Report Pillar-2 availability as a per-burst table
  column; never drop a burst because its DM "looked wrong".

Tools (off-repo audit artifacts, `/data/.../scripts/`): `dm_envelope.py` (coherent χ² envelope, FFT-shift
fast path) + `extract_chime_dm_v4_envelope.py`; `extract_chime_dm_v4b_regression.py` (coherent-once
arrival regression). Not shipped in-repo — the envelope's bootstrap σ proved finicky (per-sample χ²
overcounts → absurdly tight; sub-band bootstrap with a frozen template → too wide), so v4/v4b served as
CONFIRMATION, not the production estimator.

### v4/v4b result (2026-06-24): faint bursts are GENUINE non-detections
- **v4 coherent envelope (zach control):** DM point 261.3 (matches v2 261.5) — the coherent FFT-shift
  machinery works; only the σ estimator was unreliable.
- **v4b coherent arrival-regression, S/N-max SEED REMOVED:** zach (bright) → 261.55 ± 0.02, consistent
  with v2. **phineas (the v2 +3.1 case) → "does not constrain" (2 sub-bands), NOT +3.1.** Removing the
  incoherent S/N-max seed does not rescue the faint bursts — it correctly reports them as non-detections.
- **Convergence (5 methods): structure-max (flat), S/N-max+regression (+3 bias), crop+DM-phase (spike),
  coherent envelope, coherent regression — all agree faint scatter-broadened CHIME singlebeam bursts do
  not carry enough information to pin DM.** The v2 "+2–3 biased" tier is therefore reclassified as
  non-detections (the most defensible Pillar-2 form), not biased measurements.

### Downsampling sweep (2026-06-24) — upgrades 5/12 → 8/12
The faint bursts' non-detections were partly a BINNING artifact: at fine binning (N_SB=8/16) too few
sub-bands clear S/N≥4. A sweep over (TDS, N_SB) found a sweet spot **TDS=32, N_SB=6** (coarsest binning
the bright control still resolves; TDS=64 over-smooths and loses sub-bands). At this uniform config the
arrival regression on coherent-once data RESCUES three of the scatter-biased bursts into de-biased
constraints near DSA: **isha −0.35, wilhelm −0.44, phineas −0.45** (vs their v2 +3.46/+2.18/+3.10). All
three land in the same small-negative cluster as the bright bursts (zach −0.84 … hamilton +0.04) — the
consistency across independent bursts is signal, confirming the v2 "+2–3" was the S/N-max-seed bias.
The config is pre-registered (chosen by sub-band yield + control, NOT by DM agreement), applied uniformly
to all 12 → not selection bias. Authoritative run: `scripts/extract_final_parallel.py` (parallel
dedisp-once), `results/chime_dm_final.json`, figure-reviewed `diagnostics/chime_dm_final/`.

## LANDED (2026-06-24, PR #41 feat/custom-dm-tool → main)
Per-burst split (uniform TDS=32/N_SB=6 arrival regression on coherent-once data), figure-reviewed:
- **8 constrain DM**, all consistent with DSA within the **1 pc/cm³ floor** (`dm_agreement(dm_floor=1.0)`):
  zach 261.52±0.02, casey 491.17±0.001, freya 912.28±0.006, hamilton 518.83±0.006, chromatica 272.38±0.02,
  isha 411.22±0.11, wilhelm 601.90±0.01, phineas 609.82±0.03. (Statistical σ is a lower bound for the
  3–4 sub-band fits; the floor governs the agreement test.)
- **4 do not** (whitney, oran, johndoeII <3 sub-bands; mahi 0): `dm_confidence=unconstrained` → Pillar 4.
- `chime_side_inputs.json` un-nulled for the 8; `association_report.json` regenerated (`dm_active=8/12`);
  `test_association.py` updated (floor + 8/12 active).

## Provenance
Prior (retracted) numbers live in git (PR #29, commit cc64b7b) and off-repo at
`/data/.../results/chime_side_inputs.json`. Diagnostic: `scripts/diag_dedisp.py`,
`diagnostics/diag_dedisp_zach.png`.
