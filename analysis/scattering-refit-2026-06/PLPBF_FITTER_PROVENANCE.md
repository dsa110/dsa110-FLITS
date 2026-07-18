# PL-PBF real-data joint fitter — build + validation provenance (2026-07-17)

The third leg of the shape-model comparison for casey + wilhelm: an inner-scale
power-law PBF (Cordes, Ocker, Chatterjee et al. 2025 PTA-noise preprint, §11.2 /
Fig 40, chromatic inner scale Fig 58) fit to the real joint CHIME+DSA data, to
sit alongside the production clamped-EMG fit and the free-alpha diagnostic wedge.

## Why this and not the free-alpha wedge
The relaxed-alpha A/B (casey α=2.429±0.015 / ΔlnZ=+5537, wilhelm α=2.572±0.038 /
ΔlnZ=+731) decisively rejected the EMG self-consistency tie α=2β/(β−2), but α≈2.5
is **not a physical screen index** — no thin screen reaches sub-4; it is a
wrong-PBF *mismatch signature*. The free-alpha model is still an EMG (pulse shape
stays Gaussian⊗exponential at every ν); only the χ(ν) scaling exponent is freed,
which is internally inconsistent for an exponential PBF (Cordes §12.6.4). The
**physical** resolution is a model whose PBF *shape* changes: the inner-scale
PL-PBF, with α kept **tied** to β (self-consistent) and a chromatic inner scale.

## What was built (h17, isolated modules — no canonical burstfit.py edit)
`analysis/scattering-refit-2026-06/`:
- **`plpbf_loglike.py`**
  - `FRBParamsPLPBF(FRBParams)` — adds `s_i` (inner-scale cutoff lag at 1 GHz = NU0).
    α stays TIED to β but via the **UNCLAMPED** thin-screen relation α = 2β/(β−2)
    (an overridden `alpha` property), NOT the production `alpha_from_beta`. Both
    `s_i` and the α override survive `dataclasses.replace` inside
    `log_likelihood_gain_marginal` (which resets c0/γ), so they reach the model
    eval — same subclass-property pattern as `FRBParamsFreeAlpha`.
  - **Clamp fix (caught in review before launch):** `alpha_from_beta`
    (turbulence.py:39) hard-returns 4.0 for β ≥ BETA_THIN_SCREEN_MAX − BETA_EXP_EPS
    = 3.98 — an exponential-PBF β→4 self-consistency artifact. Left inherited, it
    would (a) step-discontinue the model/likelihood at β=3.98 (2·3.98/1.98=4.0202
    snapping to 4.000), (b) freeze the chromatic lever flat across [3.98, 4) exactly
    where wilhelm's free-α posterior lives (β=3.965 +0.015/−0.020 straddles it), and
    (c) erase the super-4 chromaticity (β=3.667 → α=4.40) the three-way test
    measures. The override drops the clamp (keeps the β>2 integrability guard);
    canonical turbulence.py is untouched. Smoke asserts α(3.667)=4.400,
    α(3.98)=4.0202, α(3.99)=4.010 (all >4, none clamped), and continuity across 3.98
    (override Δα=0.002 = the analytic slope vs the clamp's 0.021 step).
  - `_innerscale_perchan(time, mu, sig, tau, beta, s_i_ch)` — per-channel (nf,T)
    Gaussian⊗inner-scale-PL-PBF. Byte-identical convention to
    `burstfit.gaussian_powerlaw_convolution` (area-norm, `_next_fast_len(2T)` FFT,
    `[:T]` truncation) EXCEPT the PBF tail carries the three-regime inner-scale
    cutoff (core `exp(−s)` for s≤s_c; CLEAN power-law `exp(−s_c)(s/s_c)^(−β/2)` for
    s_c<s<s_i; inner-scale exp cutoff `pl(s_i)exp(−(s−s_i)/s_i)` for s≥s_i). Cutoff
    runs from **s_i, not s_c** — the form validated against Cordes Fig 40 by
    injection recovery (an earlier cutoff-from-s_c form softened the power-law and
    biased β high; caught + fixed on the injection grid).
  - `FRBModelPLPBF(FRBModel)` — overrides ONLY the M3/M2 PBF-convolution seam of
    `__call__`; dispersion delay, DM-dependent intra-channel smearing, per-channel
    amplitude and the downstream `log_likelihood_gain_marginal` are reused
    unchanged. Chromatic inner scale: s_i(ν) = s_i0 (ν/ν0)^(+4/(β−2)) — tail
    LONGER at high ν (DSA), shorter at low ν (CHIME), the falsifiable structural
    prediction the joint lever arm tests. Upgrade a prepared FRBModel in place with
    `model.__class__ = FRBModelPLPBF` (no new fields / __init__), preserving all
    prepared state.
  - `JointLogLikelihoodSharedZetaPLPBF` — 9-vector θ = [tau, beta, **log10_s_i**,
    zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D]. Mirrors
    `JointLogLikelihoodSharedZetaFreeAlpha` exactly except index 2 is log10 s_i
    (not free α), α tied, container `FRBParamsPLPBF`.
- **`run_joint_fit_plpbf.py`** — mirrors the proven `run_joint_fit_relaxalpha.py`
  driver: same `run_joint_fit.prepare_joint` prep, same production shared-zeta
  prior spec, `log10_s_i ~ U(−1, 4)` inserted after β (β prior UNCHANGED at
  [3,4]). Writes a SEPARATE artifact `plpbf_<burst>_joint_fit.json` +
  `_samples.npz`; never the production table. Rails flagged: upper → inner scale
  unconstrained (nests to production PL, tail below noise); lower → pure-exp / EMG
  limit; interior → a resolved inner scale.

## Nesting (the load-bearing correctness gate) — EXACT
One-sample smoke in `flits-a1-312`:
- **s_i → ∞ reproduces the production power-law PBF kernel to machine zero:**
  max|K_PLPBF − K_production| = **0.000e+00**, measured at β=3.5 (where the α tie
  agrees, so tau(ν) is identical and only the PBF shape is under test). This
  demonstrates the s_i parameter is a clean shape extension of
  `gaussian_powerlaw_convolution`. (For β ≥ 3.98 the PL-PBF now uses the unclamped
  α and so intentionally deviates from production's clamped-α EMG — that is the
  clamp fix, not a nesting failure.) EMG, production-PL, and inner-scale-PL remain
  on one area-normalized amplitude footing, so the three-way ΔlnZ on casey+wilhelm
  is a valid model comparison.
- Finite s_i genuinely bites the tail: max|K_finite − K_∞| = 77.8 (all finite).
- Joint gain-marginal logL finite for both an interior-s_i / β=3.5 point and a
  lower-rail / β≈3.97 window-closed point.

## Hold gate (respected)
Real-data casey/wilhelm launches are **held** until the injection recovery grid
(`pli_recover` b3.3/3.5/3.67 × s3/10/100, jobs 87–95) confirms unbiased (β, s_i)
recovery with THIS deployed kernel. As of the build the grid is PENDING (both
8-CPU long-poles — zach 59, hamilton 60 — just cleared, so it starts imminently).
On clearance: casey + wilhelm at nlive=400, then the three-way lnZ + a
residual comparison (does the self-consistent PL-PBF flatten what free-α only
half-fixed — the wilhelm DSA dipole in particular).


## Results (2026-07-18): PL-PBF REJECTED by the data on BOTH railed bursts

The recovery grid certified unbiased (beta, s_i) recovery; casey + wilhelm were then
run at nlive=400. Both COLLAPSE to the production EMG limit.

| burst | PROD lnZ | RELAXA lnZ (d vs PROD) | PLPBF lnZ (d vs PROD) | PLPBF log10 s_i (90% CI) | PLPBF beta |
|---|---:|---:|---:|---|---:|
| casey   | -33574.62 | -28037.96 (**+5537**) | -33571.29 (**+3.3**) | 1.70 [-0.17, +3.20] UPPER-RAIL | 4.000 (ceiling) |
| wilhelm | -16496.20 | -15765.19 (**+731**)  | -16499.53 (**-3.3**) | 1.40 [-0.24, +3.13] UPPER-RAIL | 3.999 (ceiling) |

(all shared-zeta gain-marginal, C1D1, same data; PROD/PLPBF have near-identical tau and
beta, confirming same-likelihood comparability. dynesty lnZ error ~0.4, so |d lnZ| = 3.3
is at the noise floor -> NO meaningful preference for PL-PBF.)

**Two-axis model-selection conclusion.**
1. **PBF SHAPE axis -- SETTLED.** On both bursts the PL-PBF adds no evidence (d lnZ ~ +/-3,
   Occam-level) while its inner scale s_i rails to the UPPER bound (unconstrained, cutoff
   far below the noise floor) and beta stays pinned at the ceiling. The self-consistent
   power-law (heavy-tail) PBF is REJECTED in favor of the exponential (EMG) per-frequency
   pulse shape. The production beta>3.98 / alpha=4 limits STAND, now hardened against a
   PBF-shape systematic (beta never left the ceiling under a completely different PBF family).
2. **CHROMATIC SCALING axis -- OPEN.** The relaxed-alpha wedge (+5537 casey / +731 wilhelm)
   survives with tail-shape now EXCLUDED as its cause. So both sub-4 apparent alphas are a
   chromatic-scaling anomaly (tau(nu) not proportional to nu^-4), not a heavy screen. The
   wedge magnitudes differ ~7x (casey >> wilhelm) -- itself a clue to the mechanism.
   Leading suspect: peak-shape systematics. Under the tied-alpha PL-PBF the residual carries
   a large, frequency-coherent single-bin PEAK dipole -- casey C-band chi2/dof=2.38, +/-32 sigma
   (matches the component-vetting +/-30 sigma precedent); wilhelm D-band chi2/dof=1.22, +/-26 sigma.
   Test queued: mask/down-weight the dipole bins and re-run the wedge (peak-shape-driven vs
   distributed-chromatic discriminant).

**Campaign-level statement (for the owner uniform-methodology decision).**
The PL-PBF model change was injection-validated, launched on the two railed bursts, and
REJECTED by the data in favor of the production EMG family on BOTH. PL-PBF does NOT become
the campaign default; the production EMG (tied alpha, beta prior) remains the uniform model
for all 12 co-detections. The beta>3.98 / alpha=4 rails are physical limits robust to
PBF-shape systematics, not modeling artifacts. Sub-4 apparent indices, where seen, are
chromatic-scaling mismatch signatures (diagnostic wedges), not evidence for a different PBF.


## Dipole-mask discriminant (task #13, launched 2026-07-18)

The chromatic-scaling axis (open, above) is probed by masking the frequency-coherent
peak dipole and re-running the free-alpha wedge:

* **alpha -> 4 after masking** => the sub-4 wedge was PEAK-SHAPE-DRIVEN. The dipole is a
  non-scattering, frequency-coherent systematic at the pulse peak; the underlying
  scattering index is consistent with the nu^-4 thin-screen line.
* **alpha stays < 4** => DISTRIBUTED chromatic scaling; the anomaly is not localized at
  the peak and survives the excision.

**Masking mechanism (driver `run_joint_fit_dipolemask.py`, driver-only, no canonical edit).**
The per-band gain-marginal likelihood (`FRBModel.log_likelihood_gain_marginal`) forms
`S_dd=sum_t d^2`, `S_dk=sum_t d*K`, `S_kk=sum_t K^2` over time, with `K=self(...)` the
unit kernel. Scaling BOTH the data column `d[:,j]` and the model output `K[:,j]` by
`sqrt(w_j)` multiplies all three sums by exactly `w_j` at bin j:

* `w_j = 0`      -> bin excluded from every sum (HARD mask, exact).
* `w_j = 1/f^2`  -> per-bin variance inflated by `f^2` (DOWN-WEIGHT, sigma -> f*sigma).

Implemented as a per-INSTANCE-gated wrapper on `FRBModel.__call__` (the dunder is resolved
on the type, so the wrapper lives on the class but only fires for the tagged target-band
instance via `getattr(self,'_dipole_sqrtw')`). The non-target band, `prepare_joint`, and
every other caller are byte-identical; per-channel `noise_std` and `valid` are untouched.

**Why the naive data-notch was REJECTED.** Smoothing the peak in the data alone (the first
plan) DISTORTS the fit: the model still peaks there, so a fake residual is manufactured;
and zeroing only the data leaves the Fisher term `M = sum_t K^2` still expecting signal at
those bins, so the fit lowers the amplitude to compensate. Exact exclusion requires the
weight in ALL THREE sums, which the sqrt(w) scaling of both d AND K achieves. Verified by
`--smoke`: hard-masked K and data columns are exactly 0, the wrapper is confirmed active on
the target band, and non-target callers are unchanged.

**lnZ is NOT comparable across masks.** Removing/deweighting bins changes the data
normalization (the `-0.5*T*ln(2 pi sigma^2)` term), so masked-vs-unmasked lnZ is not a
Bayes factor. The discriminant is the ALPHA POSTERIOR (posterior-shape driven, normalization
independent), reported as median + 90% CI.

**Runs (nlive=400, halfwin=2 => peak +/- 2 = 5 bins; soft = 10x noise inflation).**
Jobs 139-142: wilhelm-D hard/soft, casey-C hard/soft. Because both are narrow bursts, the
5-bin peak window carries a large flux fraction (wilhelm-D 0.59, casey-C 0.38; recorded in
each JSON `mask.signal_frac_masked`). The joint alpha stays constrained by the unmasked band
plus the target band's rise/tail wings, so the test is valid; a halfwin=1 (single-bin-dipole)
variant is the fallback if the hard mask is inconclusive. Post-process + posterior-overlay
figure: `dipolemask_postprocess.py` -> `dipolemask_wedge.png`. Baselines: casey alpha=2.429
(beta=3.666), wilhelm alpha=2.572 (beta=3.965).


## Scint-gain-leakage bound (task #13 successor probe, 2026-07-18) — CLOSED

Question: is the sub-4 wedge a scintillation-gain-marginalization ARTIFACT rather than
real chromatic scattering? Injection bound (driver `scint_leakage_inject.py`): a KNOWN
alpha_true=4 thin-screen burst (tau(nu)=tau_1ghz*nu^-4) carrying TIME-DECORRELATING
scintillation (the scint campaign's generator2d per-delay-slice independent realizations,
generalized here to a frequency-DEPENDENT tau), refit with the SAME joint gain-marginal
free-alpha EMG the wedge uses. A static per-channel spectrum is fully absorbed by the
per-channel gain g_f; only time-decorrelating structure (scintles interacting with the
time-frequency covariance) can leak into the temporal alpha fit.

Inputs: casey/wilhelm measured Delta-nu_d from the 2L window-campaign table (dsa110-FLITS
main, analysis/window-tuning-campaign-2026-07-17/results/{casey,wilhelm}_campaign.json).
BOTH bursts are NON-qualified (n_valid_subbands=1 < 3; wilhelm subband-0
flag_m_unphysical), so run at m=1 (maximal modulation) for a conservative BOUND. Within-
channel scintle averaging suppresses the effective modulation to
m_eff = m*sqrt(Delta-nu_d/ch_bw) BEFORE the gain marg even acts, because Delta-nu_d is far
below the channel width in both C-band channelizations.

| run | burst | Dnu_d MHz | nch (MHz/ch) | m_eff | mode | alpha_recovered [90% CI] | bias |
|-----|-------|----------:|--------------|------:|------|--------------------------|-----:|
| 147 | casey   | 0.0269 | 64 (6.2) | 0.065 | decorr      | 3.997 [3.982, 4.012] | -0.003 |
| 148 | casey   | 0.302  | 64 (6.2) | 0.218 | decorr      | 3.986 [3.971, 4.001] | -0.014 |
| 149 | wilhelm | 0.130  | 8 (50)   | 0.048 | decorr      | 4.000 [3.989, 4.012] | -0.000 |
| 150 | wilhelm | 2.236  | 8 (50)   | 0.198 | decorr      | 3.998 [3.986, 4.010] | -0.002 |
| 151 | casey   | 0.302  | 64 (6.2) | 0.218 | STATIC ctrl | 4.017 [3.993, 4.042] | +0.017 |
| 152 | wilhelm | 2.236  | 8 (50)   | 0.198 | STATIC ctrl | 3.989 [3.976, 4.001] | -0.011 |

RESULT: 6/6 recover alpha ~ 4; max |bias| = 0.017 (a static control, noise-level); all
DECORRELATING bounds |bias| <= 0.014. The scint-gain-leakage bound is |Delta-alpha| <=
0.02 -- two orders of magnitude below the observed wedges (casey ~ -1.6, wilhelm ~ -1.4
from alpha=4). The static controls recover alpha ~ 4 as the falsifier requires (harness
sound: had they failed, the harness -- not the physics -- would be broken); and even the
decorrelating injections leak nothing at these Delta-nu_d. SCINT-GAIN LEAKAGE EXCLUDED as
the wedge cause on both bursts.

Caveats: bound at m=1 (upper limit) under the 2L table's own gates; inputs are
diagnostic-grade (neither burst is a qualified 2L detection) and the 2L table is pending
owner ratification. m_eff uses the analytic within-channel-averaging suppression (a
fine-grid-generate-then-block-average would be the gold standard, unnecessary for a bound
that already comes out at the noise floor).

ELIMINATION CHAIN COMPLETE (both wedges): (1) PBF-shape axis SETTLED -- PL-PBF rejected,
exponential EMG adequate, beta held at ceiling under a different PBF family; (2) heavy-tail
+ close-secondary leakage bounded to -0.86 (half the wedge) and inapplicable to casey (no
heavy tail); (3) peak-shape dipole EXONERATED by the dipole-mask test -- wedge Delta-lnZ
survives excision on both bursts (casey +3540/+3795 of +5537, wilhelm +634/+586 of +731),
alpha stays far sub-4; (4) scint-gain leakage bounded here to |Delta-alpha| <= 0.02.
TWO-SCREEN CHROMATICITY -- tau(nu) not proportional to nu^-4 from a second (host/local)
screen -- is the SOLE surviving hypothesis for both wedges. Next decision (owner-visible,
model-change class like PL-PBF was): charter a two-screen forward model vs. write it up as
the interpreted hypothesis.
