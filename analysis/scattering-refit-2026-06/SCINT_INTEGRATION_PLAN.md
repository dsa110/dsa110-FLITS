# Scintillation-bandwidth ↔ scattering integration — plan (2026-06-19)

## TL;DR
Do NOT build a scint-bandwidth estimator — a complete, tested one already exists. The
"integration" is **wiring the existing consistency layer to consume the joint-fit
outputs**, and it doubles as the **independent α cross-check** Codex demands for the
contested α=2.41. Lazy path: glue, not new physics.

## What already exists (verified file:line)
- `scintillation/scint_analysis/` — FULL active pipeline (not dead):
  - `analysis.py:209 calculate_acf(spectrum_1d, channel_width_mhz, ...)` — spectral ACF, HWHM→Δν_d, finite-scintle errors.
  - `analysis.py:465 calculate_acfs_for_subbands(...)`, `:1244 analyze_scintillation_from_acfs(...)` — per-subband ACF + model select (Lorentzian/Gaussian/gen-Lorentzian/power-law).
  - `fitting_2d.py` `Scintillation2DResult` (gamma_0, alpha, m_0, nu_ref) — global γ(ν)=γ0(ν/νref)^α fit → **direct α from scintillation**.
  - `physics.py:37 scintillation_bandwidth_to_timescale(dnu_hz, freq_mhz, coefficient=1.0)` — τ=C/(2πΔν_d), C configurable (1.16 thin / 0.72 thick Kolmogorov).
  - `pipeline.py ScintillationAnalysis`, `consistency.py run_consistency_check(...)`.
- `flits/batch/analysis_logic.py` — the cross-pipeline glue ALREADY written:
  - `:97 check_tau_deltanu_consistency(comparison_df)` → product τ(ν_scint)·Δν_d, expects ~C_THIN_SCREEN≈0.16; flags consistent/inconsistent.
  - `:189 analyze_frequency_scaling(comparison_df)` → α_Δν from log(Δν_dsa/Δν_chime)/log(ν_dsa/ν_chime); checks self-consistency vs α_τ.
  - **`:220-231` α_τ is an admitted PLACEHOLDER** ("scientifically questionable") — it just averages the two per-telescope fixed α=4. **The joint fit replaces exactly this with a real cross-band α_τ.**
- Existing OUTPUTS on disk:
  - DSA scint configs+cached fits for ALL 12 bursts: `scintillation/configs/bursts/<b>_dsa.yaml` (`stored_fits/subband_*` hold fitted γ=Δν_d, redchi, stderr).
  - CHIME scint: only 4 bursts `scintillation/chime_acfs/<b>_*_subband_acf_fits.pkl` (chromatica, freya, hamilton, wilhelm). Among clean-M3 joint candidates only **wilhelm** has BOTH bands.

## Why this is the right validator (ties to Codex review)
Pulse-broadening τ and scint Δν_d are two INDEPENDENT probes of the same screen.
α_τ (joint scattering fit) vs α_Δν (scint, frequency-scaled) must agree.
- Agree at ~2.4 → shallow α is REAL.
- α_Δν≈4 but α_τ≈2.4 → joint α is an ARTIFACT (consistent with Codex's DM-smearing/ζ
  contamination suspicion). Scint α doesn't touch the time-domain DM-smearing model at
  all, so it cleanly adjudicates.

## Recommended course of action (phased, lazy-first)
- **P0 (surfaced, needs greenlight — core-pipeline change):**
  - Fix dormant unit bug `burstfit.py:51` `DM_SMEAR_MS 8.3e-6 → 8.3e-3` ms (derive: 2·DM_DELAY_MS/1000 = 2·4.148808/1000 = 8.30e-3). Dormant only because dm_init=0 ⇒ ×0; latent landmine for any dm_init≠0 run.
  - DECISION needed: should intra-channel smearing use `dm_init+delta_dm` (Codex C/D)? Currently `_smearing_sigma(self.dm_init,...)` ignores fitted delta_dm; CHIME delta_dm=4.7 ⇒ unmodelled smearing absorbed by ζ/τ. This is the live α-corruption channel.
- **P1 (the integration — mostly glue):** build a `comparison_df` (cols: burst_name, telescope, tau_1ghz[,_err], delta_nu_dc[,_err], alpha[,_err]) from (a) joint-fit JSON (shared τ_1ghz + the real joint α), (b) scint `stored_fits` Δν_d. Run the EXISTING `check_tau_deltanu_consistency` + `analyze_frequency_scaling`. Patch `analyze_frequency_scaling` to take the joint α_τ instead of the placeholder average.
- **P2 (adjudicate):** for wilhelm (both bands) compare α_τ(joint) vs α_Δν(scint). Generalize once more CHIME ACFs are computed (the chime_acfs pkl path already exists for 4; extend to the rest via the scint pipeline).
- **P3 (only if motivated):** fold scint-derived τ (via `scintillation_bandwidth_to_timescale`) into the scattering fit as a PRIOR or extra likelihood term — a true scatter+scint joint fit. YAGNI until P2 shows it's needed.

## WILHELM CROSS-CHECK RESULT (2026-06-19) — α_τ=2.53 NOT corroborated by scint
Data: DSA narrow Δν_d from `wilhelm_dsa.yaml` stored_fits (4 subbands, two-Lorentzian, narrow l_1_gamma ~0.10-0.21 MHz); CHIME from `chime_acfs/wilhelm_253635173_subband_acf_fits.pkl` (1_lorenz sub_scint_1 ~0.03-0.24 MHz, f_cents 640-728 MHz). Both two-component (narrow + broad screen).
- median resolved Δν_d: CHIME 0.060 MHz @0.684 GHz | DSA 0.154 MHz @1.405 GHz.
- α_Δν (cross-band) ≈ 1.31 BUT per-CHIME-subband spans -0.6..+2.2 -> too noisy to confirm/refute α_τ=2.53.
- τ from resolved Δν_d (=1/2πΔν): CHIME 2.65 µs, DSA 1.03 µs. τ from scattering fit @band ctr (α=4): CHIME 658 µs, DSA 102 µs. RATIO ~250x / ~100x.
- τ·Δν_d product ~3e-5 vs expected ~0.16 (off ~4000x). Δν_d that pairs with scattering τ = 0.24-1.6 kHz — BELOW channel resolution (CHIME native 0.39 MHz, DSA 0.030 MHz) -> UNRESOLVED.
- VERDICT (two-screen): resolved scint = nearby small-τ screen; pulse-broadening τ = FAR screen, scintles unresolvable. Different screens -> scint cannot independently validate α_τ for wilhelm. The 100-250x τ mismatch ALSO flags fitted τ may not be clean diffractive scattering (reinforces Codex ζ/τ-contamination concern).
- Math under independent verification: Codex gpt-5.5 high + wolframscript (brpq16ydt, /tmp/codex_math_verify.out).
- IMPLICATION for integration: the naive `check_tau_deltanu_consistency` (single Δν_d in df) is screen-blind; for these FRBs it will read "inconsistent" because τ probes an unresolved screen. P1 wiring must carry BOTH scint components + label which screen, not feed one Δν_d.

## B — DSA INTRA-CHANNEL SMEARING (decided 2026-06-19: CHIME coherent, DSA NOT)
User confirmed: CHIME data coherently dedispersed (dm_init=0 correct, smearing=0); DSA data
incoherently dedispersed -> real native intra-channel smearing IS in the data but model omits it.
- Code facts: `burstfit.py:51 DM_SMEAR_MS=8.3e-6` (should be 8.3e-3); `:481 sig_dm=DM_SMEAR_MS*dm*self.df_MHz*freq^-3`; `:535 _smearing_sigma(self.dm_init,...)` (dm_init=0 -> term=0). `io.py:96 df_MHz=df_MHz_raw*f_factor` => model df_MHz is DOWNSAMPLED (DSA 11.72 MHz) but intra-channel smearing is set by NATIVE df (0.030518 MHz).
- DM per burst encoded in DSA filename `_<DM_int>_<DM_frac>_` (oran comment confirms "oran DM is 397").
- Computed DSA σ_smear (band ctr, NATIVE df, σ=Δt_DM convention) vs fitted τ_DSA@1.4(α=4):
  freya 83µs/τ38 =2.2x; wilhelm 55/102=0.54; phineas 56/95=0.58; zach 24/114=0.21; whitney 42/304=0.14; oran 36/277=0.13; johndoeII 64/560=0.11; casey 45/0.6=80x. (M1 DSA non-dets: chromatica/hamilton/isha/mahi.)
- DIRECTION: DSA is the high-freq point; removing smearing shrinks τ_DSA -> steeper CHIME->DSA slope -> alpha UP toward Kolmogorov. Credible full explanation of shallow alpha for moderate-τ bursts (esp wilhelm).
- FIX (3 coupled): (1) const 8.3e-6->8.3e-3; (2) smearing use native df not downsampled self.df_MHz; (3) DSA dm_init=catalog DM (filename), CHIME dm_init=0.
- OPEN modeling Q (user): σ=Δt_DM vs σ=Δt_DM/√12 (boxcar variance-matched) -> 3x magnitude swing (wilhelm 55µs vs 16µs). Decides "dominates" vs "15% correction".
- GATED: core burstfit.py + all DSA configs + full refit = science-altering (moves headline alpha + published deck/site). Await user greenlight + convention choice. Verify smearing math w/ Codex+Wolfram before refit ships.

### B IMPLEMENTED 2026-06-19 (greenlit, σ=Δt_DM/√12)
- burstfit.py:51 DM_SMEAR_MS 8.3e-6 -> 8.3e-3. _smearing_sigma: dt_dm=const*dm*df*nu^-3; sig_dm=dt_dm/sqrt(12); hypot(sig_dm,zeta). NOT yet committed.
- io.py:75 FRBModel df_MHz=self.telescope.df_MHz_raw (NATIVE, was downsampled self.df_MHz). df_MHz only used for smearing (verified). core.py:777 copies model.df_MHz -> inherits native. ok.
- DSA run configs (HPCC) patched dm_init=catalog DM from filename (casey 491.2 ... mahi 960.1 ... wilhelm 602.3). CHIME stays 0. gen_dsa_configs.py updated to parse DM (reproducible).
- LOCAL VALIDATION: DM_SMEAR_MS=0.0083; DSA wilhelm sig=15.88us (=dt_DM 55.0/sqrt12); CHIME sig=0; joint demo() passes.
- REFIT LAUNCHED: 12 DSA single-band (64414373-84) + 4 joint floor a=1.0 (oran/johndoeII/wilhelm/phineas 64414385-88). watcher byee18xf0. Codex+Wolfram B-math verify b5lle951y (/tmp/codex_b_verify.out).
- PRE-B joint (floor1.0) for before/after: oran 1.44[edge], johndoeII 1.38, wilhelm 2.53, phineas 3.66. EXPECT alpha UP post-B (smear removed from DSA high-freq point).
- caveat: visualization.py:527 FRBModel(df_MHz=df_MHz) for diag overlay may still get downsampled df (cosmetic, not the fit) — follow-up.

### B run 1 (64414385-88 joint, 64414373-84 DSA): TWO bugs found
1. DSA single-band 64414373-84 FAILED 5-9s: PATH bug — run_burst.sbatch cd's to repo, relative `configs/..` wrong dir. Fix: pass ABSOLUTE config path. (joint uses abs path in driver, unaffected.)
2. oran-joint 64414385 COMPLETED but CORRUPT: alpha=5.68 railing near 6.0, **delta_dm_D=-1777** (degenerate mode [16,84]=[-2585,-1193], NOT flat prior), tau poorly constr [0.20,1.16]. Root: build_priors:983 delta_dm prior = ±DM_MAX = ±3000 (absurd for a RESIDUAL DM around dedispersed catalog DM). The broadened (smeared) DSA model loosened delta_dm -> fit escaped into huge-|delta_dm| degeneracy + alpha overshoot. Pre-B stayed put; B unmasked it.
   FIX: build_priors delta_dm -> ±DM_RESID_MAX=50 (new const, burstfit.py). global (single-band+joint inherit); CHIME unaffected (delta_dm~0). demo passes, prior verified ±50.
- B run 2 (DSA 64414477-88 + joint 64414489-92): abs paths + tightened prior + B smearing. watcher b325yz1gx checks delta_dm stays within ±50 (rail => deeper degeneracy). alpha=5.68 DISCARDED as corrupt; awaiting clean before/after.
- B mechanism CONFIRMED real though: smearing strongly steepens alpha (oran 1.44->~5.7 even if overshot). Magnitude of clean shift TBD from run 2.

### B run 2 CLEAN RESULTS (DSA 64414477-88 + joint 64414489-92, ±50 prior + B smearing)
- delta_dm all sane (joint |ddm|<0.8; single-band freya -23 largest, within ±50). Prior fix HOLDS.
- JOINT alpha pre-B(floor1.0) -> post-B: oran 1.44->1.37(+0.54/-0.28) ; johndoeII 1.38->1.37±0.05 ; wilhelm 2.53->2.70±0.05 ; phineas 3.66->3.58±0.04.
- => B does NOT move joint alpha. DSA smearing (~10-30us at sqrt(12)) too small vs scattering tau except where smear>=tau.
- B's only measurable footprint = freya single-band tau 0.148->0.0071 (smear>tau; "scattering" was mostly smearing) + already-marginal oran/casey. oran single-band M3<->M1 is a 0.2-lnZ TIE (DSA scattering undetectable in-band; joint detects via CHIME). NOT a real flip.
- B's biggest value: surfaced+fixed the ±3000 delta_dm prior landmine.
### POSTERIOR-PREDICTIVE per-band chi2/dof at joint best fit (joint_ppc.py, decisive test)
- johndoeII a=1.37: CHIME 1.14 / DSA 1.03 | oran a=1.37: 1.11/1.06 | wilhelm a=2.70: 1.71/1.30 | phineas a=3.58: 1.20/2.02.
- SHALLOW-alpha bursts fit BOTH bands to chi2~1.0-1.1 (BEST of the 4). Near-Kolmogorov phineas fits DSA WORST (2.02). => shared tau*nu^-1.37 genuinely describes both bands; shallow alpha is NOT a forced compromise. Figs data/joint/<b>_joint_ppc.png.
- TENTATIVE CONCLUSION: per-sightline alpha 1.4->3.6 are real well-fitting measurements; sub-Kolmogorov (johndoeII 1.37±0.05 tight; oran 1.37 loose) looks REAL not artifact.
- UNDER ADVERSARIAL VERIFICATION: workflow wf_417320dd-fd9 (4 lenses: sampler/degeneracy/cross-band-systematics/literature -> synthesize). Verdict pending.
- B follow-up: commit burstfit.py (DM_SMEAR_MS, _smearing_sigma, DM_RESID_MAX) + io.py (native df) + burstfit_joint.py (ncall fix) once alpha verdict in. DSA configs dm_init patch is HPCC-only + gen_dsa_configs.py; repo source bursts/dsa/*.yaml still dm_init:0 (regen covers it).

### ADVERSARIAL VERDICT on shallow alpha (workflow wf_417320dd-fd9, 4 lenses + my re-verification)
- oran: REFUTED as a measurement. CHIME nuisance unconstrained (zeta_C=28.8 rails, alpha err +0.54/-0.28 to floor). DROP oran.
- johndoeII: alpha=1.37±0.05 REAL. Survives sampler (profile -2lnL rejects a=4 by ~2400 w/ all 11 nuisance free; 7.4sig off prior floor; single-band tau-ratio alone gives a~1.0), degeneracy (synthetic a=4 refit@1.37 costs dchi2~40000; flux-invariant; zeta-tau separable), AND morphology.
- Morphology threat (lens 3: claimed on-pulse chi2=2.86 + 4 sub-peaks + within-CHIME FWHM scaling -2.1) DID NOT REPRODUCE on my direct check: on-pulse chi2/pix (±8ms): johndoeII C1.54/D1.10 (CLEANEST of set) vs wilhelm C2.38/D2.00, phineas C1.74/D4.59. CHIME sub-band profiles (data/joint/<b>_chime_subband_profiles.png) show clean scattering tails (sharp rise+exp decay), johndoeII tail (~21ms @0.65GHz) > wilhelm control (~1ms) => MORE scattering, real. within-CHIME width-moment test too noisy (S/N-limited; even wilhelm control gives -3.2) -> inconclusive, not refuting.
- wilhelm 2.70±0.05, phineas 3.58±0.04 tight/real.
- BOTTOM LINE: joint fit = genuine per-sightline alpha 1.37->3.58; johndoeII genuinely sub-Kolmogorov. Caveats: 2-pt cross-band lever (16ch weak within-band); scint says multi-screen possible (alpha = effective dominant-screen index); phineas DSA poorest (on-pulse 4.59); N=3 usable.
- shallow alpha is NOT a DSA-smearing artifact. B's lasting value = fixed ±3000 delta_dm prior landmine.

## CODEX+WOLFRAM MATH VERIFICATION (brpq16ydt, /tmp/codex_math_verify.out)
ALL wilhelm cross-check numbers PASS (kernel-verified; `wolframscript -local` wrapper stalled in Codex sandbox -> used WolframKernel directly). scintτ 2.653/1.033µs; fitτ 657.9/101.6µs; ratios 248/98x; α_Δν 1.309 (span -0.61..2.23); impliedΔν 0.242/1.566 kHz; implied/native 6e-4 / 0.051 (both <<1 unresolved). Reciprocity s·Hz dimensionless confirmed.

## Don'ts
- Don't reimplement ACF/Δν_d (complete + unit-tested already).
- Don't merge the two pipelines into one mega-fit yet (P3, gated on P2).
- Don't trust α=2.41 until P0 modeling question + P2 cross-check resolve.
