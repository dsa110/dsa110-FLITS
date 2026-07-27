# Two-screen forward-model fitter — provenance

Companion to `PLPBF_FITTER_PROVENANCE.md`. Lane chartered 2026-07-18 (owner Option A):
`Faber2026 docs/rse/specs/charter-two-screen-forward-model-2026-07-18.md`. NON-gating
parallel lane; model-change class (same guardrails as PL-PBF).

## 1. Hypothesis

The casey/wilhelm free-α wedges (α≈2.43 / 2.57, bias ≈ −1.6 / −1.4 from ν⁻⁴; all five
single-screen mechanisms excluded — `report-jointtf-mechanism-closure-2026-07-18.md`)
are produced by TWO scattering screens whose convolved pulse-broadening, sampled across
the CHIME/DSA lever arm, mimics a sub-4 effective index in a single-screen fit.

## 2. Model (rung 1)

Per channel: Gaussian(σ) ⊗ exp(τ₁(ν)) ⊗ exp(τ₂(ν)), both screens tied to the SAME
shared β via the UNCLAMPED thin-screen relation α = 2β/(β−2); one extra parameter
r = τ₂/τ₁ at 1 GHz (achromatic — both screens carry the same α, so r is constant across
ν and the composite width still scales exactly ν⁻⁴). Nested: r→0 recovers the production
single-screen EMG exactly ⇒ ln Z(two-screen) − ln Z(production) is a valid Bayes factor.

**Closed form (no numerical convolution).** Two one-sided exponentials convolve to
[e^(−t/τ₂) − e^(−t/τ₁)]/(τ₂−τ₁); convolving with the Gaussian and using linearity,

    K(t) = [τ₂·EMG(σ,τ₂) − τ₁·EMG(σ,τ₁)] / (τ₂ − τ₁)

a difference of two PRODUCTION EMGs (burstfit.analytic_gaussian_exp_convolution), each
area-normalized, so K is area-normalized (the τ prefactors give (τ₂−τ₁)/(τ₂−τ₁)=1).

Two guarded limits:
- r → 0 (τ₂→0): K → EMG(σ,τ₁). Explicit short-circuit at r < 1e-6.
- r → 1 (τ₁≈τ₂): the divided difference is 0/0 with catastrophic cancellation. The exact
  limit is K = d/dτ[τ·EMG(σ,τ)], which we evaluate cancellation-free using the identity
  exp(−u²/2σ²)·erfcx(b) = 2τ·EMG (u=t−μ, b=σ/√2τ − u/√2σ):

      d/dτ[τ·EMG] = σ/(√(2π) τ²)·exp(−u²/2σ²) − (√2 σ b/τ)·EMG(σ,τ)

  reusing the fully stability-guarded base EMG (only a bare Gaussian is evaluated raw).
  Inside |r−1| < 1e-3 the derivative is evaluated at the MIDPOINT τ=(τ₁+τ₂)/2 (2nd-order
  divided-difference), so the closed and derivative branches agree to O((τ₂−τ₁)²) at the
  switch — seamless likelihood surface.

## 3. Implementation & validation

Files (mirror the PL-PBF set):
- `twoscreen.py` — FRBParamsTwoScreen (+r, UNCLAMPED α), `two_screen_perchan`,
  `_dtau_tau_emg` (r=1 branch), FRBModelTwoScreen (seam swap; upgrade in place via
  `__class__`, mirrors FRBModelPLPBF).
- `twoscreen_loglike.py` — JointLogLikelihoodSharedZetaTwoScreen, 9-vector
  θ=[τ₁, β, log10_r, ζ₁, x, t0_C, ddm_C, t0_D, ddm_D] (one extra param vs production,
  same footing as PL-PBF / free-α).
- `twoscreen_stage0_inject.py` — Stage-0 falsifier.
- `validate_twoscreen.py` — kernel smoke.

Kernel validation (`validate_twoscreen.py`, all PASS 2026-07-18):
- NEST  r→0 vs production EMG(τ₁): relerr 0.00e+00 (exact).
- DERIV f'(τ₁) analytic vs Richardson central difference of τ·EMG: relerr 5.7e-10.
- GT    closed form vs OVERSAMPLED (8×) FFT convolution, r=0.3 / 1.5: 2.1e-5 / 8.5e-6.
- SWITCH closed vs midpoint-derivative at |r−1|=1e-3: relerr 2.0e-7 (seamless).
- AREA  unit area per channel across r∈{0.01…3.0}.

The generic-r branch is EXACT by construction (linear combination of two exact EMGs);
the only new hand-derived math is the r=1 derivative, validated by DERIV above.

## 4. Stage 0 — wedge-reproduction falsifier (pre-registered)

Inject rung-1 two-screen data (α_true=4, both screens exponential) at r∈{0.1,0.3,1.0}
for casey-like and wilhelm-like configs; refit with the FREE-α single-screen EMG
diagnostic (plpbf_inject._LLEMG, the exact fitter that measured the real wedges).
Screen-1 τ₁ = τ_real/(1+r) holds the composite mean delay at each burst's real
production τ_1GHz, so r sweeps SHAPE at fixed total scattering.

Configs (real production fit values; σ DSA-anchored from the least-scattered band):
- casey-like:   τ_real=0.019 ms, σ=0.055 ms (W/τ≈2.9), C 64ch, D 48ch.
- wilhelm-like: τ_real=0.330 ms, σ=0.100 ms (W/τ≈0.30), C 8ch, D 48ch.

Decision rule (pre-registered): PASS (config) = recovered α ≤ α_true − 1 (bias ≤ −1);
Stage-0 PASS = ANY config passes ⇒ proceed to Stage 1. FAIL = max reachable |bias| ≪ 1.6
across the grid (+ a W/τ envelope sweep) ⇒ rung-1 two-screen joins the elimination table,
report to owner with the rung-2 (independent β₂) question, NO real-data fits.

Jobs 153–158 (nlive=400, nproc=4), submitted 2026-07-18 beside the count wave.

## 5. Stage 0 — results

PENDING (grid 153–158 running). Preliminary nlive=60 smoke (casey r=0.3): α_apparent=4.42,
bias +0.42 (no-wedge; single low-nlive point, not dispositive).

## 6. Stages 1–2 (chartered, NOT yet run)

Gated on Stage 0 PASS. Stage 1 = recovery grid + null (r lower-rail collapse on
single-screen injections; r-vs-gain-marginal identifiability is the named stop risk).
Stage 2 = real casey+wilhelm three-way (production / two-screen / free-α), mode-check
before any ΔlnZ, WIN = ΔlnZ>5 AND r interior, COLLAPSE = r railed → τ₂ upper limit; then
owner review. The Stage-2 real-data driver is intentionally NOT built until Stage 0 passes.
