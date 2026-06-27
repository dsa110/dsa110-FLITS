# Joint CHIME+DSA scattering fit — build state (2026-06-19)

## Why
Single-band fits with α fixed=4 give **inconsistent τ₁GHz** between CHIME (0.6 GHz) and
DSA (1.4 GHz) for the same sightline → α≠4. Joint fit shares τ₁GHz + frees α; the ~1 GHz
lever arm measures α. DSA τ₁GHz systematically > CHIME → true α likely shallower than 4.

## τ₁GHz per burst (orientation-corrected, ms): CHIME | DSA | model(C/D)
- johndoeII 0.143 | 2.18  (M3/M3)  — 15× off, strongest joint-fit lever
- whitney   0.117 | 1.18  (M2/M3)  — 10×
- oran      0.540 | 1.08  (M3/M3)
- wilhelm   0.144 | 0.396 (M3/M3)
- phineas   0.274 | 0.372 (M3/M3)
- zach      0.262 | 0.44  (M3/M3) — both bands FAIL/marginal (CHIME under-dedispersed)
- freya     0.150 | 0.148 (M2/M3) — AGREES; CHIME χ²=9.6 (under-dedispersed) so trust DSA
- isha      0.290 | M1(non-det) — DSA sees no scattering
- mahi      0.212 | M1 ; hamilton 0.020 | M1 ; chromatica 0.114 | M1 (DSA non-dets)
- casey     CHIME pending | DSA 0.002 (τ≈0)
Best joint-fit candidates (both bands clean M3 detections): oran, johndoeII, wilhelm, phineas.

## Joint fit spec (sketch, ~120 lines, new module burstfit_joint.py)
- Vector ~11 params: [τ₁GHz, α, ζ | c0_C, γ_C, t0_C, ddm_C | c0_D, γ_D, t0_D, ddm_D]
- SHARED: τ₁GHz, α (sightline). PER-TEL: c0, γ, t0, ddm. ζ shared (start) or per-tel.
- `joint_loglike(theta) = model_C.log_likelihood(pC) + model_D.log_likelihood(pD)` —
  two FRBModels sharing (τ,α); independent noise → sum. Reuse the nch²-fixed FRBModel.
- prior_transform: reuse build_priors(absolute_bounds=True) per-tel block; α sampled WIDE
  (e.g. [2,6]) — DROP alpha_fixed. dynesty pool, bump nlive (~600-800 for 11 dim).
- t0 independent per telescope (absorbs inter-tel timing/DM offset). Don't model delay.
- Ponytail: hardcode 2 telescopes, no N-telescope abstraction.
- Validation: joint α tight + away from prior edges → degeneracy broken; joint τ₁GHz
  predicts both bands. Cross-check vs single-band rails.

## Data / infra (HPCC)
- repo /home/jfaber/flits/dsa110-FLITS @ branch fix/scattering-likelihood-nch2-and-sampler-knobs, HEAD 727e3be
- CHIME data /central/scratch/jfaber/flits-runs/data/*.npy ; configs configs/<b>_chime_run.yaml ; jsons data/analysis_*/<b>_fit_results.json
- DSA   data /central/scratch/jfaber/flits-runs/data/dsa/*.npy ; configs configs/<b>_dsa_run.yaml ; jsons data/dsa/analysis_*/<b>_fit_results.json
- telescopes.yaml: chime+dsa freq_descending:true (loader flips to ascending).
- RESOLUTIONS (verified vs raw .npy shapes 2026-06-19, oran):
  - CHIME native (1024 ch, 32000 t): df=0.390625 MHz, dt=2.56 us, band 0.400-0.800 GHz, window 81.92 ms.
    downsample f_factor=64 -> 16 ch (df=25.0 MHz); t_factor=24 -> dt=61.44 us (1333 t).
  - DSA   native (6144 ch, 2500 t):  df=0.030518 MHz, dt=32.768 us, band 1.311-1.499 GHz, window 81.92 ms.
    downsample f_factor=384 -> 16 ch (df=11.719 MHz); t_factor=2 -> dt=65.536 us (1250 t).
  - YES downsampled both axes. 16 ch comes from f_factor ONLY (not t_factor). outer_trim=0.15 then crops 15%/end (res unchanged).
  - 16 ch is coarse for within-band alpha -> the cross-band shared-tau joint fit IS the alpha instrument.
- venv `${FLITS_VENV:-/central/scratch/jfaber/envs/flits-joint}` (symlink → quarantine spack py3.11.6 + dynesty 3.0.0). `run_joint.sbatch` / `run_burst.sbatch` prepend venv to `PATH` (avoid broken `source activate` when `/home/jfaber/flits` is missing). sbatch: run_burst.sbatch (BLAS pinned). submit: sbatch -A radiolab --job-name=<b>-<tel> run_burst.sbatch <cfg>. nproc=8 sweet spot.
- Mac scratch: /Users/jakobfaber/Developer/scratch/2026-06/flits-refit/ {good_fit_diag.py, build_deck.py, build_site.py, gen_dsa_configs.py, check_freq_order*.py}

## Key code facts
- FRBModel.__call__(params, model_key) → model spectrum; .log_likelihood(params, key); .noise_std (per-ch), .valid (mask)
- _PARAM_KEYS["M3"]=(c0,t0,gamma,zeta,tau_1ghz,alpha,delta_dm). fit_single_model_nested(*, model, init, model_key, nlive, dlogz, alpha_prior, alpha_fixed, nproc, ...) keyword-only, builds priors at burstfit_nested.py:368 (absolute_bounds=True).
- plot_fit_quality(data,model,freq,time,noise,valid,params,results,output_path,...) in visualization.py — resid-σ diagnostic; pipeline auto-emits <name>_fitquality.png.
- spawn guard: any script using nproc>1 pool needs `if __name__=="__main__"`.

## Published / pending
- Deck+site live (orientation-corrected, 11 CHIME + casey pending): https://jakobtfaber.github.io/Faber2026/ (gh-pages orphan branch, Faber2026 repo)
- casey LANDED: M3, τ₁GHz=0.01646(5) ms, χ²/dof=3.860 (marginal/poor — a real low-τ M3, NOT the M1 non-det we'd predicted). TODO: pull casey_fq.png, finalize slide, rebuild deck+site.
- 7 DSA config typo fixes (_l_→_I_) still UNCOMMITTED in working tree (separate lane; data globbed by burst name so not blocking).

## JOINT FIT — BUILT 2026-06-19
- Module: scattering/scat_analysis/burstfit_joint.py (NOT yet committed). 12-vec [tau_1ghz,alpha | c0/t0/gamma/zeta/ddm ×(C,D)]. shared (tau,alpha); per-tel zeta. joint ll = ll_C+ll_D. _JointPriorTransform/_JointLogLikelihood picklable (dynesty fork pool). build_priors(absolute_bounds=True) per-tel; alpha~U(2,6). demo() self-check PASSES Mac+HPCC (prefers true alpha & tau).
- Driver: scratch run_joint_fit.py (+ HPCC /central/scratch/jfaber/flits-runs/run_joint_fit.py): reads <b>_chime_run.yaml + <b>_dsa_run.yaml, rebuilds each band's BurstDataset.model + data_driven_initial_guess→refine_initial_guess_mle init, calls fit_joint_scattering, writes data/joint/<b>_joint_fit.json. sbatch: run_joint.sbatch (cpus=8, BLAS pinned). submit: sbatch -A radiolab --job-name=<b>-joint run_joint.sbatch <b> 600.
- SMOKE TEST: oran-joint job 64412933 (nlive=600, nproc=8) — sampler CONVERGED (dlogz=0.001, 37896 it, lnZ=-15784.4) but CRASHED in results-packaging: `int(results.ncall)` — dynesty .ncall is a per-iteration ARRAY not a scalar. Fixed burstfit_joint.py:248 -> `int(np.sum(results.ncall))`. demo() never caught it (tests likelihood, not the run_nested wrapper). Resubmitted 64413201.
- ORAN RESULT (job 64413201, floor alpha=2.0): alpha = 2.41 (+0.70/-0.30), tau_1GHz = 0.668 (+0.17/-0.078) ms, lnZ=-15784.8. SHALLOW (<< Kolmogorov 4). Shared tau bridges CHIME rail 0.54 + DSA rail 1.08. BUT flagged [AT PRIOR EDGE] (median-1.5sig=1.96<2.0); 16th pct=2.11 (some curvature). Nuisance: zeta_C=63 vs zeta_D=0.064 (CHIME intrinsic broad / DSA narrow), delta_dm_C=4.7 (CHIME residual DM).
- EDGE TEST (floor alpha=1.0, jobs 64413375-78, DONE): sbatch `${@:3}` arg-passthrough + `--alpha-lo 1.0`.
  - oran: alpha 2.41->1.44 (+0.68/-0.32) STILL [AT PRIOR EDGE] -> RAILING. oran alpha is PRIOR-SHAPED not measured; alpha=2.41 NOT trustworthy. (Codex predicted this exact failure.)
  - johndoeII: alpha=1.38 (+0.05/-0.06) tight, not flagged, but implausibly shallow -> likely common CHIME-band systematic (confident-but-wrong).
  - wilhelm: alpha=2.53 (+0.06/-0.06) CLEAN, far from both floors -> REAL measurement, sub-Kolmogorov.
  - phineas: alpha=3.66 (+0.03/-0.03) CLEAN, near-Kolmogorov.
  - VERDICT: joint fit constrains alpha for clean-CHIME bursts (wilhelm,phineas), rails for contaminated ones (oran=zeta_C 63, johndoeII). Heterogeneous.

## CODEX gpt-5.5 high REVIEW (2026-06-19, /tmp/codex_joint_review.out)
- MY joint module: NO blocking bug. likelihood sum ll_C+ll_D correct (burstfit_joint.py:129-143); log-uniform inverse-CDF correct (:112-115); 12-vec indexing correct; fork pool picklable+correct. nch^2 bug confirmed FIXED in path used (burstfit.py:578-590).
- BLOCKING (pre-existing burstfit.py, not my code):
  1. DM_SMEAR_MS=8.3e-6 ms is 1000x too small; should be 8.3e-3 (= 2*DM_DELAY_MS/1000 = 2*4.148808/1000). VERIFIED bug. DORMANT here (dm_init=0 -> sig_dm=0) but latent landmine.
  2. lnZ NOT normalized: log_likelihood returns -0.5*sum(resid^2) only (burstfit.py:589-590), omits -sum(log sigma)-N/2 log2pi. Posterior OK up to const (fixed data); absolute lnZ arbitrary -> DON'T compare lnZ across floors/masks/prior-volumes.
  3. delta_dm feeds arrival-time delay (burstfit.py:528-532) but NOT intra-channel smearing (uses fixed dm_init, :535,:477-481). CHIME delta_dm=4.7 => unmodelled smearing absorbed by zeta/tau => LIVE alpha-corruption channel.
- Pre-believe-alpha<4 checklist: fix DM units; per-channel posterior-predictive (no single band dominating); retry shared zeta + fixed delta_dm; channel jackknife/RFI sensitivity (16 ch); synthetic injection through same preprocessing; multi-seed/nlive, tighter dlogz (0.5 coarse).
- See SCINT_INTEGRATION_PLAN.md: scint alpha_Dnu is the independent adjudicator. wilhelm has BOTH-band scint -> immediate cross-check of alpha_tau=2.53.
