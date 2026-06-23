# Research: Multi-component joint-fit evidence kernel & N=1 commensurability (issue #37)

---
**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Active
**Related Documents:** GitHub issue #37 (follow-ups from the auto-review of PR #36, merged as `3688be8`)

---

## Research Question

Document, as it exists today, the multi-component gain-marginal evidence path in the
joint CHIME+DSA scattering fit: how `_gain_marginal_multi_band` computes per-band
log-evidence, how component count routes the single vs multi path through
`fit_joint_scattering`, how the single-component `log_likelihood_gain_marginal`
differs, where the gain-prior variance `s2` is set or profiled, the `dt_min`
handling, the `n_supported`/`frac_culled` denominators, and the current test
coverage. Framed by the four code items in issue #37.

## Executive Summary

The multi-component evidence kernel `_gain_marginal_multi_band`
(`scattering/scat_analysis/burstfit_joint.py:190-362`) computes a per-channel
linear-Gaussian gain-marginal evidence for one band: N temporal component kernels
per channel, per-component gains `g ~ N(0, s2 I_N)` integrated analytically, summed
over valid channels. It uses the FULL data normalization
`-0.5*T*ln(2*pi*sigma^2)` and a proper finite-variance Occam term
`-0.5*ln det(I_N + (s2/sigma^2) M)`. The quadratic divisor is `sigma^2` (the
docstring records that the spec's `sigma^4` was a transcription slip, verified
against the brute Gaussian evidence via Woodbury).

The single-component `log_likelihood_gain_marginal`
(`scattering/scat_analysis/burstfit.py:728-765`) is a different estimator: a flat
(improper) prior on the per-channel gain, giving the matched-filter / F-statistic
marginal `-0.5*chi2min - 0.5*ln(S_kk) + 0.5*ln(2*pi*sigma^2)`. Its Occam term is
`-0.5*ln(S_kk)` (improper) rather than the proper `ln det(I + (s2/sigma^2) M)`, so
its additive scale differs from the multi path. This is the commensurability gap
issue #37 item 1 describes.

**Headline finding (state of the code vs. the issue): issue #37 is largely already
implemented.** The N>1 gate the issue asks to remove has been replaced by a
`force_multi` flag plus per-call `gain_s2`, and the regression tests item 2 asks
for already exist (and exceed the requested minimum). See "#37 status against
current code" below. The research phase's value here is precisely this: it shows
that planning/implementing #37 as written would re-do work that already landed.

## Scope

**What This Research Covers:**
- `_gain_marginal_multi_band` evidence math, conditioning guard, and diagnostics.
- The single vs multi vs other-mode routing in `fit_joint_scattering`.
- The single-component `log_likelihood_gain_marginal` flat-prior path.
- `gain_s2` plumbing and the `s2=None` ML-profiling path.
- `dt_min` derivation and the `n_supported`/`frac_culled` denominators.
- Current test coverage of the kernel.

**What This Research Does NOT Cover:**
- Whether the design is correct or optimal (documentarian only — no critique).
- The dynesty sampler internals, the non-gain joint likelihoods beyond routing.
- The scintillation GP path (`log_likelihood_gain_marginal_gp`) beyond noting it
  shares the flat temporal statistics.

## Key Findings

### Finding 1 — Multi-component evidence kernel `_gain_marginal_multi_band`

**Relevant Files:**
- `scattering/scat_analysis/burstfit_joint.py:190-362` — the kernel.
- `scattering/scat_analysis/burstfit_joint.py:197-237` — docstring stating the math.

**How It Works:**
1. Build the per-component unit kernels `Ks` of shape `(N, F, T)` by calling the
   forward model with `c0=1, gamma=0` for each component
   (`burstfit_joint.py:244-250`).
2. Per channel form the sufficient statistics: `S_dd = sum_t d^2` (F,),
   `b = sum_t d*K` (F, N), `M = sum_t K_i*K_j` (F, N, N)
   (`burstfit_joint.py:255-257`).
3. Per-channel marginal evidence (gains integrated analytically):
   `ln Z_f = -0.5*[S_dd/sigma^2 - b^T (M + (sigma^2/s2) I)^-1 b / sigma^2]
   - 0.5*T*ln(2*pi*sigma^2) - 0.5*ln det(I_N + (s2/sigma^2) M)`
   (`burstfit_joint.py:303-307` for the well-conditioned branch). The first line
   is the data fit, the second is the FULL data normalization, the third is the
   proper Occam penalty (grows with N and s2).
4. **Eigenvalue conditioning guard** (`burstfit_joint.py:259-281`): per channel,
   eigendecompose `M`; a channel is `ok` (full-rank-N) when
   `min_eig/max_eig >= eig_rel_floor` (default `1e-6`). A supported-but-ill-
   conditioned channel (`cull`) falls back to a **rank-1 proper evidence on the top
   eigenpair** (`burstfit_joint.py:310-327`), not the gain=0 baseline — the inline
   note explains that at large fixed `s2` the gain=0 baseline would *reward* a
   degenerate merge, so the rank-1 fallback keeps a merge Occam-penalized. A
   genuinely unsupported channel (no signal) gets the gain=0 baseline
   `-0.5*S_dd/var - 0.5*T*ln(2*pi*var)` (`burstfit_joint.py:286`).
5. Returns `(lnZ, diag)`; `lnZ = sum_f ln Z_f` (`burstfit_joint.py:328, 350`).

### Finding 2 — `s2` (gain-prior variance): fixed or ML-profiled

**Relevant Files:**
- `scattering/scat_analysis/burstfit_joint.py:330-348` — the `s2 is None` branch.

**How It Works:**
- If a float `s2` is passed, it is used fixed (`burstfit_joint.py:347-348`).
- If `s2 is None`, `s2` is profiled by 1-D bounded ML over `log s2`
  (`scipy.optimize.minimize_scalar`), anchored on the data amplitude scale
  `var(ahat)` where `ahat = b/diag(M)` (`burstfit_joint.py:330-346`). The issue
  notes this profiled value is a profile/empirical-Bayes Z, not a clean marginal,
  hence its recommendation to fix `s2` for cross-N model selection.
- `s2` flows in from `fit_joint_scattering(gain_s2=...)` →
  `_JointLogLikelihoodGainMulti(s2=gain_s2)` (`burstfit_joint.py:872`) →
  `_gain_marginal_multi_band(..., s2=self.s2)` (`burstfit_joint.py:785-786`).

### Finding 3 — Single-component path `log_likelihood_gain_marginal` (the contrast)

**Relevant Files:**
- `scattering/scat_analysis/burstfit.py:728-765`.

**How It Works:**
- Builds the single unit kernel `K`, forms `S_dd, S_dk, S_kk` per channel
  (`burstfit.py:751-757`).
- Flat-prior matched-filter marginal:
  `ln L_f = -0.5*(S_dd - S_dk^2/S_kk)/sigma^2 - 0.5*ln(S_kk) + 0.5*ln(2*pi*sigma^2)`
  (`burstfit.py:761-764`). The Occam term is `-0.5*ln(S_kk)` (improper, flat), in
  contrast to the multi path's proper `-0.5*ln det(I + (s2/sigma^2) M)`. This is
  the additive-scale difference between a 1-component `lnZ` from this path and a
  ≥2-component `lnZ` from the multi path.
- `log_likelihood_gain_marginal_gp` (`burstfit.py:781-...`) shares the identical
  temporal matched-filter statistics and falls back to this flat path verbatim
  when `delta_nu_d_MHz is None` (`burstfit.py:814-815`).

### Finding 4 — `dt_min` derivation and `n_supported`/`frac_culled` denominators

**Relevant Files:**
- `scattering/scat_analysis/burstfit_joint.py:874-886` — `dt_min` derivation.
- `scattering/scat_analysis/burstfit_joint.py:352-361` — the diagnostics NB.

**How It Works:**
- `dt_min` (min component time separation enforced by the ordered prior transform):
  if `None`, computed as `max` over the two bands of `3 * median(|diff(time)|)`
  (`burstfit_joint.py:876-881`). The inline comment (`:874-875`) says "the binding
  constraint is the tighter (smaller-dt) band's resolution," while the code takes
  `max(dts)` (the coarser band's 3-sample floor). The single scalar `dt_min` is
  applied to both bands' `t0` groups via `_JointPriorTransformOrdered`
  (`burstfit_joint.py:883-886`); there is no per-band `dt_min`.
- `n_supported` vs `frac_culled`: an explicit NB comment (`burstfit_joint.py:352-355`)
  records that they use DIFFERENT denominators. `n_supported = count(ok)` counts only
  well-conditioned (full-rank-N) channels; `frac_culled = mean(~ok)` also counts
  rank-1-fallback channels as culled, so `n_supported != (1 - frac_culled) * F`
  in general (`burstfit_joint.py:356-360`).

### Finding 5 — Test coverage of the kernel

**Relevant Files:**
- `tests/test_gain_marginal_multi_band.py:1-276` — 8 tests, all targeting
  `_gain_marginal_multi_band` via a duck-typed `_FakeModel` with prescribed kernels.

**What It Covers:**
- `test_brute_force_woodbury` (`:103-124`) and `test_brute_force_woodbury_varied`
  (`:127-140`, seeds 0/7/42, N=3) — total `lnZ` equals an independent brute-force
  Gaussian evidence `-0.5 d^T Sigma^-1 d - 0.5 ln det(2 pi Sigma)`,
  `Sigma = sigma^2 I + s2 K^T K`, to `rtol=1e-9`. This is issue #37 item 2(a).
- `test_label_swap_invariance` (`:143-164`) — permuting component order leaves
  `lnZ` unchanged to `rtol=1e-12`. This is issue #37 item 2(b).
- `test_rank1_fallback_on_collinear_channel` (`:167-212`) — a collinear channel
  triggers the rank-1 fallback; checks `frac_culled = 1/F`, `n_supported = F-1`,
  brute-force equality, and that the culled channel sits strictly below its gain=0
  baseline. This is issue #37 item 2(c).
- `test_s2_profiling_finds_interior_optimum` (`:215-246`) — the `s2=None` ML path
  beats fixed `s2` and lands on an interior optimum.
- `test_no_valid_channels_returns_neg_inf` (`:249-258`) and
  `test_valid_mask_subsets_channels` (`:261-275`) — boundary contracts.

## Component Interactions

**Routing flow (`fit_joint_scattering`, `burstfit_joint.py:815-905`):**
1. `multi = bool(force_multi) or components_C > 1 or components_D > 1`
   (`burstfit_joint.py:864`).
2. If `multi`: build `JOINT_PARAM_NAMES_GAIN_MULTI`, the gain-multi prior spec, and
   `loglike = _JointLogLikelihoodGainMulti(..., n_C, n_D, s2=gain_s2)`
   (`burstfit_joint.py:866-873`); derive `dt_min`; use the ordered prior transform
   (`:874-886`).
3. Else, in priority order: `shared_zeta` → `marginalize_gain_gp` →
   `marginalize_gain` → plain joint (`burstfit_joint.py:887-902`).
4. `_JointLogLikelihoodGainMulti.__call__` (`burstfit_joint.py:776-788`) unpacks
   `theta` into per-band component params and calls `_gain_marginal_multi_band` once
   per band, summing `lnZ_C + lnZ_D` (independent noise → additive evidence).

```
fit_joint_scattering(components_C, components_D, force_multi, gain_s2, dt_min)
   │  multi = force_multi OR n_C>1 OR n_D>1
   ├─ multi ─> _JointLogLikelihoodGainMulti(n_C, n_D, s2=gain_s2)
   │              └─ per band: _gain_marginal_multi_band(params, ["M3"]*n, s2)
   │                              └─ proper N(0,s2) prior, full norm, proper Occam
   └─ not multi ─> shared_zeta / gain_gp / gain / plain
                      └─ FRBModel.log_likelihood_gain_marginal  (flat improper prior)
```

## #37 status against current code

| #37 item | Asked for | Current code | Status |
|---|---|---|---|
| 1 — N=1 commensurate | Drop the `>1` gate or add a flag so N=1 routes through the multi path | `force_multi: bool` param (`burstfit_joint.py:836`) → `multi = bool(force_multi) or ...` (`:864`); with `force_multi=True` and `components_C=components_D=1`, N=1 goes through `_gain_marginal_multi_band` (proper prior) | Implemented via flag |
| 1 — fixed s2 for cross-N | Pair with a fixed `gain_s2` | `gain_s2: float | None` param (`:834`) threaded to the kernel (`:872, :785-786`); fixed when a float is passed | Implemented |
| 2 — regression tests | At least (a) brute-force + (b) label-swap | `test_gain_marginal_multi_band.py` has (a)+(b)+(c)+profiling+boundary | Implemented, exceeds ask |
| minor — denominators | Document `n_supported`/`frac_culled` use different denominators | NB comment `burstfit_joint.py:352-355` | Documented |
| minor — `dt_min` comment | Fix `dt_min` comment vs `max(dts)` mismatch; consider per-band `dt_min` | Per-band `dt_min` implemented 2026-06-22 (`_JointPriorTransformOrdered` scalar-or-per-group; `fit_joint_scattering` derives `[dt_C, dt_D]`); comment rewritten. See [plan](plan-dt-min-per-band.md) / [implement](implement-dt-min-per-band.md). | Resolved |

## Edge Cases and Constraints

- All-invalid band → `(-inf, frac_culled=1.0, n_supported=0)`
  (`burstfit_joint.py:241-242`); tested.
- Collinear/near-merged kernels → rank-1 fallback, not gain=0
  (`burstfit_joint.py:281, 310-327`); tested on the exactly-collinear case.
- `s2=None` profiling returns a profile/empirical-Bayes Z, not a clean marginal —
  relevant if `lnZ` is used for cross-N model selection (issue #37 item 1 caveat).

## Open Questions

1. `dt_min` semantics: the comment names the "tighter (smaller-dt) band" as the
   binding constraint but the code applies `max(dts)` (coarser band's floor). Which
   is intended? (Documented as a tension, not resolved here.)
2. Is per-band `dt_min` wanted, or is the single scalar floor deliberate?
3. Given items 1 and 2 are implemented, is issue #37 now reducible to the two
   `dt_min` nits, or is there a model-selection driver (e.g. an evidence-ladder
   helper that sweeps N and calls with `force_multi=True, gain_s2=<fixed>`) still
   missing at the API level?

## References

- Files analyzed: 3 source + 1 test
  - `scattering/scat_analysis/burstfit_joint.py` (kernel, routing, diagnostics)
  - `scattering/scat_analysis/burstfit.py` (single-component flat-prior path)
  - `tests/test_gain_marginal_multi_band.py` (kernel regression tests)
- Related: GitHub issue #37; PR #36 (`3688be8`).
