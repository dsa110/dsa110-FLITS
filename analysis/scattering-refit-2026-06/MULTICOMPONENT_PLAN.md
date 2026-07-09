The smoking gun is real and reproduces independently: on pure noise, the design likelihood rewards N=2 by +20 to +324 nats as t0_2→t0_1 merge, Occam flips positive, max|g| blows up to ~4700. The min-separation prior is mandatory, not optional.

# Multi-Component Gain-Marginal Joint Likelihood — Integrated Plan

This plan integrates the implemented design, its self-check, the adversarial verdict (independently reproduced below on the merge singularity), and the per-burst diagnosis into one honest build/compute plan. **Status of the core claim:** the N-template likelihood math is correct and verified on synthetic data; it is **NOT production-ready as written** — three required fixes (min-separation prior, proper gain prior for valid evidence, normalization constant) stand between it and a trustworthy sampler run.

Self-check (synthetic, PASS): `/Users/jakobfaber/Developer/scratch/2026-06/flits-refit/multicomp_selfcheck.py`
Kernel read-only: `scattering/scat_analysis/burstfit.py` (`log_likelihood_gain_marginal` :723), `burstfit_joint.py` (`_JointLogLikelihoodGain` :229, `fit_joint_scattering` :304)

---

## 1. CAPABILITY

**Implemented (TEXT only — canonical kernel untouched):**
- `_gain_marginal_multi_band(model, params_list, model_keys)` — exact N-template generalization of `FRBModel.log_likelihood_gain_marginal`. Per channel f solves NxN normal equations `M_f g = b_f` (`M_ij = Σ_t K_i K_j`, `b_i = Σ_t d K_i`), `chi2_f = (S_dd − bᵀM⁻¹b)/var`, `occam_f = −0.5 ln det(M_f)`.
- `_JointLogLikelihoodGainMulti` — picklable joint log-L over n_C CHIME + n_D DSA components sharing (tau_1ghz, alpha).
- `gain_spectra_multi(...)` — per-component per-channel gain spectra g[:, i] (the correct scintillation object).
- Per-channel conditioning guard (PD + scale-free `cond_proxy`) preserving exact N=1 reduction.

**Self-check result (PASS, synthetic only):**
- (c) N=1 exact reduction: |multi − kernel| = 3.5e-11 ✓
- (a) two-component spectrum recovery: median frac err 0.2–1.2% per component per band ✓
- (b) BIC screen (k = N·(F+2)): prefers 2 on 2-pulse (dBIC ~ +1.9M), prefers 1 on 1-pulse (dBIC −190/−375) ✓
- (d) near-degenerate stress at dt=1e-7 ms: all channels culled, finite ✓

**Adversarial verdict: `sound-with-fixes`.** N=1 reduction / gain composition / picklable hold; identifiability / overfit-guard / N≥2 normalization fail. Headline failure reproduced on pure noise:

| dt (ms) | Δ(ll₂−ll₁) | max|g| | median occam |
|---|---|---|---|
| 2.0 | +20.2 | 2.6 | −0.89 |
| 0.5 | +74.9 | 9.4 | +0.47 (flipped) |
| 0.1 | +139.4 | 46.6 | +2.08 |
| 0.01 | +231.6 | 467 | +4.38 |
| 0.001 | +323.7 | 4673 | +6.68 |

On pure noise the likelihood rewards a spurious 2nd component by up to +324 nats as t0_2→t0_1; the Occam term flips positive; the cond_floor=1e-9 guard never fires in the damage band. A continuous-t0 sampler with only sorted-t0 ordering will dive into this and un-rail alpha for the wrong reason. Self-check (b) PASS is an artifact: its 0.025 ms t0 grid never samples dt<0.01 ms; (d) jumps to 1e-7 ms, skipping the dangerous band.

**Required fixes before any production run (priority order):**
1. **Minimum component separation prior** (primary). Enforce t0_{i+1} − t0_i ≥ dt_min in the prior transform (reject below it). dt_min ≈ fraction of kernel width / channel time-resolution. Verify on pure noise that N=2 no longer beats N=1.
2. **Proper (finite-variance) gain prior.** Flat improper prior on g∈ℝ^N makes lnZ arbitrarily normalized → lnZ(N=1) vs lnZ(N=2) is not a valid Bayes factor. Cannot be deferred if lnZ is the overfit guard. (This is the GP-block work the design left at N=1.)
3. **Fix marginal normalization constant** to 0.5·N_eff·ln(2π var) per channel (N_eff = N on ok channels, 0 on fallback), not the N-independent 0.5·ln(2π var). N=1 unchanged.
4. **Tighten conditioning guard** to fire in the damage band (dt ~ 0.01–0.1 ms), e.g. cull on min/max eigenvalue of M_f; add per-burst diagnostics (fraction culled, max|g| per component).
5. Redo self-check (b) with continuous/fine-near-zero t0 search (or short real dynesty run with min-sep prior) + a dedicated noise-only injection asserting N=2 NOT preferred.

---

## 2. THE EXACT DIFF

All marked [VERIFIED] (self-checked numerically) or [OPEN] (specified, not exercised). Nothing written to the read-only kernel.

**`burstfit_joint.py`** — additive, all guarded by component counts defaulting to 1:

```python
def JOINT_PARAM_NAMES_GAIN_MULTI(n_C, n_D) -> Tuple[str,...]:   # [VERIFIED layout]
    names=["tau_1ghz","alpha"]
    for tag,n in (("C",n_C),("D",n_D)):
        for i in range(1,n+1): names += [f"t0_{tag}_{i}", f"zeta_{tag}_{i}"]
        names.append(f"delta_dm_{tag}")
    return tuple(names)
# ndim = 4 + 2*(n_C+n_D); n_C=n_D=1 -> 8 == JOINT_PARAM_NAMES_GAIN (reduces to current code)

def _LOG_NAMES_GAIN_MULTI(n_C,n_D):
    nm=JOINT_PARAM_NAMES_GAIN_MULTI(n_C,n_D)
    return frozenset({"tau_1ghz"} | {n for n in nm if n.startswith("zeta_")})

def _joint_prior_spec_gain_multi(init_C, init_D, alpha_bounds, n_C, n_D): ...   # [VERIFIED reuse]
def _gain_marginal_multi_band(model, params_list, model_keys, cond_floor=1e-9): ...  # [VERIFIED N=1 to 3.5e-11]
class _JointLogLikelihoodGainMulti: ...   # picklable                            # [VERIFIED]
def gain_spectra_multi(model, params_list, model_keys): ...                      # [VERIFIED recovery <1.2%]

# ===== REQUIRED-FIX additions (NOT yet in the self-checked code) =====
class _JointPriorTransformOrdered(_JointPriorTransform):     # [OPEN] sort + ENFORCE dt_min
    def __init__(self, spec, t0_idx_groups, dt_min): ...
    def __call__(self, u):
        x = super().__call__(u)
        for g in self.t0_idx_groups:
            if g.size>1: x[g] = np.sort(x[g])
        # FIX #1: reject (-> -inf) if any x[g][i+1]-x[g][i] < dt_min . MANDATORY.
        return x
# FIX #2: replace flat g-prior with finite-variance (proper) prior so lnZ valid.  [OPEN]
# FIX #3: const -> 0.5*N_eff*ln(2*pi*var) per channel in _gain_marginal_multi_band. [OPEN]
```

`fit_joint_scattering`: add `components_C: int = 1, components_D: int = 1`; new branch BEFORE existing gain branches when `marginalize_gain and (components_C>1 or components_D>1)` → multi names/spec/loglike + `_JointPriorTransformOrdered(spec, t0_groups, dt_min=DT_MIN)`. `nlive>=800-1000` for ndim>=12. `_weighted_percentiles` + return dict key off active names → flow through unchanged [VERIFIED].

**`run_joint_fit.py`**: add `--components-C/--components-D`; post-fit build params_list per band from `t0_{s}_{i}, zeta_{s}_{i}`, shared `delta_dm_{s}`, shared tau/alpha. Multi-peak t0 seeding [OPEN — unimplemented].

**`gain_ladder.py` / `multiscale_fit.py`**: `gain_var_multi(...)` returns g,v shape (F, N): `g=solve(M_f,b_f)`, `v=diag(sig²·M_f⁻¹)`. `multiscale_fit.py` autocorrelates EACH g[:, i] for that component's Δν_d. Rationale [VERIFIED]: each pulse carries its own burst spectrum + its own diffractive screen realization; autocorrelating the total gain mixes two screens and corrupts Δν_d. Total kernel K=ΣK_i used ONLY for the temporal residual-whiteness check.

---

## 3. PER-BURST PREDICTION (9 excluded bursts)

Predictions from the α/χ²/lag-1 signature (HPCC table) — NOT fits (real data HPCC-only; local scratch JSON is stale pre-fix junk). "Recoverable" = N-template clears ΔlnZ>5 with a resolved component, post-refit α unimodal-interior, Δν_d posterior < ~0.3 dex.

**Genuinely recoverable — hidden-pulse, α interior (test A primary):**
| Burst | α | Cause | Expected |
|---|---|---|---|
| wilhelm | 2.63 | hidden pulse, DSA | DSA lag1=−0.15 sharp under-fit → N=2 whitens DSA, α barely moves; narrow Δν_d. Borderline→clean. |
| chromatica | 3.98 | hidden pulse, DSA (strong) | DSA χ²=9.09,lag1=+0.88 biggest fail; big ΔlnZ, α~4. Strong recover. |
| mahi | 3.52 | hidden pulse, DSA | long resolved tail + lag1=+0.67 → real 2nd comp on tail; cleanly recovered. |
| phineas | 3.82 | hidden pulse, DSA (weak) | lag1=+0.81 but χ²=2.14 → low-amp secondary; CHECK ΔlnZ clears 5; if marginal, Δν_d widens. |
| whitney | 3.75 | hidden pulse, CHIME (worst) | χ²=2.87,lag1=+0.89; prime N=3 candidate; α interior; recovered from CHIME dominant-comp gain. |
| zach | 3.32 | hidden pulse, CHIME | CHIME lag1=+0.82; α~3.3; DSA lag1=0.80 borderline — add DSA secondary if persists. |

**Coupled / information-limited (tests B/C):**
| Burst | α | Cause | Expected |
|---|---|---|---|
| isha | 4.68 | hidden pulse + possible degeneracy | tau=0.005ms near-unresolved; multi-peak BOTH bands. Likely recoverable, but watch tau hitting resolution floor → then information-limited on tau (and same-screen cross-check). |
| hamilton | 1.01 (railed low) | hidden pulse (CHIME) → drives α low | Strongest coupled-A: CHIME χ²=3.64,lag1=+0.61, DSA clean. N-template on CHIME should clear ΔlnZ>5, whiten CHIME, AND un-rail α to ~2–4 simultaneously. If α stays low → fall to B then C. |
| johndoeII | 1.06 (railed low) | α↔delta_dm degeneracy OR genuine sub-Kolmogorov | BOTH bands temporally clean → test A will NOT clear threshold. Run B (fix DM): α un-rails → degeneracy (Δν_d carries DM-prior width); α stays ~1 unimodal with DM pinned → genuine sub-Kolmogorov (the verified johndoeII result), kept in via marginalization. Information-limited on α but a legitimate physics point. |
| oran | 5.96 (railed high) | hidden pulse (DSA) + explicit α↔DM multimodality | Richest: DSA χ²=5.21,lag1=+0.80; KNOWN bimodal (separate run α=1.44 w/ large delta_dm_C). Run A (2-comp DSA, expect α off high rail), THEN B (fix-DM, collapse bimodality). Partially recoverable; if two modes survive both → wide Δν_d, flagged information-limited. Strongest test of the coupling. |

**Summary:** 6 cleanly recoverable, 3 coupled/limited. The "2nd pulse drags α to the rail" hypothesis is directly testable only on hamilton and oran; johndoeII railing with clean residuals is the falsifier preventing over-claiming it universally.

---

## 4. METHODOLOGY CHANGE — marginalize, don't gate

Replace the hard cut (exclude if α railed OR χ²>3 OR lag1>0.4) with full posterior propagation of (alpha, delta_dm_C, delta_dm_D) into Δν_d:
1. Run joint fit at the evidence-selected N with `marginalize_gain_gp=True` so Δν_d_C, Δν_d_D are sampled alongside (tau, alpha, delta_dm). Poorly-constrained α → wider K_f(t) ensemble → wider ahat_f → wider GP sigma_g² → wider Δν_d. No manual error inflation.
2. Population result: do NOT collapse each burst to its median first. Draw (alpha_b, delta_dm_b, Δν_d_b) jointly from each burst's weighted nested samples, propagate hierarchically. A railed/multimodal-α burst contributes a broad/bimodal Δν_d cloud and down-weights itself by its own variance.
3. Information-limited bursts stay in with their true (large) credible interval; the GP's 0.3·chan_width lower bound encodes "unresolved → flat limit," so they land as explicit Δν_d limits, labeled, not silent dropouts.

**Component selection (objective):** adopt N+1 over N iff ΔlnZ = lnZ(N+1) − lnZ(N) > 5 (Jeffreys strong) AND new component's t0 resolved (separated by > its own ζ width at 2σ) AND its gain spectrum has positive total power. Cross-check (not decisive): whitened residual lag-1 < 0.2, χ²_red < 2.

**Honest verdict — removes OR masks the bias? BOTH; the protocol separates them:**
- **Genuinely de-biased:** the 6 hidden-pulse bursts. 1-comp was simply wrong; N-template recovers a correct narrow Δν_d at full weight. Removes a real bias because the excluded set was non-random — preferentially the multi-peak/brighter/more-scattered bursts (the tail you care about).
- **Merely widened (bias → uninformative limit):** test-C and surviving-multimodal bursts (johndoeII genuine, oran if two modes persist). Folding them in is honest — no longer bias the sample by absence — but they contribute almost no information.
- **Danger:** tuning N upward until every residual whitens → spuriously-narrow over-fitted Δν_d. The ΔlnZ>5 + resolved-component gate is the only guard and must not be relaxed to chase lag-1 — and ΔlnZ itself is only trustworthy once the proper gain prior (fix #2) is in place.

---

## 5. COMPUTE PLAN (serial HPCC follow-up — subagents cannot ssh)

Against the trustworthy post-fix posteriors (never stale local JSON). Land §1 fixes FIRST — do NOT run the multi-component sampler with the flat prior + sorted-only transform.

**Phase 0 — land fixes + offline validation (before any HPCC run).**
- 0a. Fixes #1 (dt_min), #3 (N_eff const), #4 (eigenvalue guard) — cheap.
- 0b. Fix #2 (proper gain prior) — the GP-block work; required if lnZ is the selection criterion. If deferred, select with the BIC k=N·(F+2) screen and accept it is offline-only.
- 0c. Re-run multicomp_selfcheck.py augmented with: a dt scan 0.3→0.001 ms on pure noise asserting Δll does NOT increase as components merge, and a noise-only N=2-not-preferred test. GATE: do not proceed to HPCC until pure-noise N=2 no longer beats N=1.

**Phase 1 — pull + baseline (HPCC).** Pull current post-fix N=1 gain-marginal posteriors for all 12 bursts. Confirm the 9 exclusions reproduce. Minutes (read-only).

**Phase 2 — 2-component refits, strongest hidden-pulse candidates first.** Order: 1. oran (DSA; richest, A+B). 2. hamilton (CHIME; cleanest α-un-rail test). 3. chromatica (DSA; strongest single fail). 4. mahi, whitney (→ try N=3), zach, phineas, wilhelm. 5. isha (both bands; watch tau-floor). Each: run N=1 and N=2 (N=3 for whitney) under identical priors + dt_min transform. nlive≥800–1000 for ndim≥12. Budget ~1 node-day per burst per N as a planning figure, parallelize across bursts.

**Phase 3 — ΔlnZ decision.** Adopt N+1 iff ΔlnZ>5 AND exceeds quadrature-summed lnZ errors AND new component resolved + positive-power. Repeat borderline (phineas, isha) across ≥2 seeds. For railed bursts, record whether α migrates off the rail into ~2–4.5.

**Phase 4 — test B (fix-DM)** on any still-railed/multimodal burst (johndoeII, oran, hamilton if A failed). Refit with delta_dm fixed at the DM-budget (NE2001/MW + host), α free, DM-budget uncertainty as prior. α un-rails → degeneracy; α stays + unimodal → genuine non-power-law (test C; re-run with alpha_bounds=(1,8) to confirm the mode tracks the bound).

**Phase 5 — marginalized scintillation re-test.** At selected N, `marginalize_gain_gp=True` so Δν_d sampled. Reconstruct per-component gain spectra; autocorrelate the dominant component's gain, marginalizing over component-assignment when powers comparable. Build population Δν_d posterior hierarchically. Flag each burst recoverable vs information-limited.

**Approx total:** Phase 2 dominates — ~9 bursts × (N=1,2[,3]) at nlive~1000. A handful of node-days if parallelized; ~1–2 weeks serial. Phases 4–5 cheaper.

---

## 6. RISKS / OPEN QUESTIONS

**Adversarially-flagged, must-fix (unsound as-written):**
- Merge singularity (reproduced): pure noise rewards N=2 by +20→+324 nats as t0_2→t0_1, Occam flips positive, max|g|→4673. cond_floor=1e-9 never fires in the damage band. A sampler WILL exploit this and un-rail α for the wrong reason. → fix #1 (dt_min) mandatory; verify on noise.
- Improper gain prior → ill-defined evidence: flat prior on g∈ℝ^N has no finite prior volume, so lnZ(N=1) vs lnZ(N=2) is not a valid Bayes factor. Overfit guard does not work until fix #2. Until then select by BIC k=N·(F+2), label offline.
- Mis-normalized marginal for N≥2: const=0.5·ln(2π var) is N-independent; correct is 0.5·N·ln(2π var). Self-check (b) BIC PASS is an artifact of the coarse 0.025 ms t0 grid → redo with continuous/fine search.

**Provenance / data-trust:**
- Stale local posteriors: flits-refit/joint_json/*.json are pre-fix junk (local freya α=6.00 vs real post-fix 4.48). Never ground truth — all §3 numbers are signature-based predictions, not fits.
- Multimodal α: oran explicit (runs at α=1.44 vs 5.96 on same data via the (α,delta_dm) ridge). Run multimodal-aware (dynesty bound='multi'), inspect for both rails.

**Scientific / methodological:**
- Whether 2-comp actually un-rails α on real bursts is the untestable-from-here hypothesis — verified on synthetic only (N=1 reduction 3.5e-11; 2nd-pulse ΔlnL~+23). johndoeII railing with clean residuals is a standing falsifier of the universal form.
- dt_min, cond_floor, conditioning-proxy form are heuristic; need per-burst diagnostic (fraction culled, max|g|) tuned against the measured cond curve.
- Component-assignment ambiguity: comparable gain powers → "which is dominant" for the Δν_d ACF is uncertain; if their diffractive scales differ (different screen path) that burst's population Δν_d is genuinely ill-defined, not merely wide.
- Same-screen cross-check tau = 1/(2π Δν_d) is uninformative where tau hits the resolution floor (isha tau=0.005ms) or Δν_d hits the 0.3·chan_width prior bound — do not report as agreement/tension there.
- fix-DM test (B) is only as good as the DM-budget; a wrong budget misclassifies degeneracy as genuine-non-power-law. Carry DM-budget uncertainty as a prior.
- GP scintillation path NOT generalized to N components (correct GP is block-structured over freq × component); the flat N-template matched filter is the minimal extension to un-rail α + whiten residuals. Generalizing the GP block overlaps with required fix #2 (a proper gain prior IS the GP prior).
