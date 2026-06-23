# Research: Current state of the joint CHIME+DSA scattering fits

---
**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Active
**Related Documents:**
- [Research: Multi-component joint-fit evidence kernel](research-multicomponent-joint-evidence.md)
- [Implement: Per-band dt_min](implement-dt-min-per-band.md)

---

## Research Question

What is the present state of the joint CHIME+DSA scattering fits stored in the repo:
for each co-detected burst, which `fit_joint_scattering` mode/config produced the
stored result, how many temporal components, what fit quality is recorded, and which
α are constrained vs railed? This is the starting map for any multi-component re-fit
campaign enabled by the #37 work.

## Executive Summary

Eleven of the twelve co-detected bursts have stored joint fits in
`analysis/scattering-refit-2026-06/joint_json/` (casey is absent there; it has
separate multiscale outputs). **Every one of the eleven used the `plain` joint mode
(per-band amplitude `c0` and spectral index `gamma` are sampled — the discriminator;
the gain-marginal modes fix `c0=1, gamma=0`) with a single temporal component**, α
prior bounds `[1.0, 6.0]`. None used
the gain-marginal, gain-marginal-GP (scintillation), shared-ζ(ν), or multi-component
paths. **So the #37 multi-component gain-marginal evidence path — and even the simpler
gain-marginal scintillation-aware paths — have been run on zero real bursts.** The
stored campaign is single-component plain-mode throughout.

α splits three ways: three bursts rail at the upper bound with near-zero τ (scattering
unresolved, α unconstrained); several sit at interior values; three are mid-α with
resolved τ — the scientifically meaty, potentially bias-prone cases. Fit quality is
only partially recorded: 6 of 11 have a PPC file with per-band reduced χ²; the fit
summaries themselves store no goodness metric, so a full PASS/MARGINAL/FAIL flag cannot
be assigned from disk for the other five (notably **zach**, the canonical bias case,
has no PPC).

## Key Findings

### Finding 1 — All stored fits are plain mode, single component

**Relevant Files:**
- `analysis/scattering-refit-2026-06/joint_json/*_joint_fit.json` — 11 fit summaries.
- `analysis/scattering-refit-2026-06/run_joint_fit.py:120,142-186` — the driver; mode
  is chosen by flags (`marginalize_gain`, `marginalize_gain_gp`, `shared_zeta`,
  `force_multi`, `components_C/D`); `multi = components_C>1 or components_D>1 or force_multi`.

**How inferred:** each summary's `percentiles` block contains `c0_C` and `gamma_C`
(and the `_D` analogues), i.e. per-band amplitude and spectral index are *sampled* —
the signature of the plain `_JointLogLikelihood`. The gain-marginal modes fix `c0=1,
gamma=0` (not sampled), and the multi-component mode emits `t0_C1, t0_C2, …`. Neither
appears in any summary; every burst has a single `t0_C`/`t0_D`. Confirmed across all 11.

### Finding 2 — α regime and τ per burst

| burst | mode | nC | α (med) | regime | τ_1GHz (med) | lnZ |
|---|---|---|---|---|---|---|
| chromatica | plain | 1 | 6.00 | **upper-rail** (unconstrained) | 0.025 | −44072 |
| freya | plain | 1 | 6.00 | **upper-rail** | 0.049 | −164303 |
| hamilton | plain | 1 | 5.99 | **upper-rail** | 0.005 | −27010 |
| mahi | plain | 1 | 5.53 | interior (high) | 0.095 | −15431 |
| isha | plain | 1 | 4.96 | interior (high) | 0.347 | −17543 |
| zach | plain | 1 | 3.66 | **interior (mid)** | 0.322 | −173492 |
| phineas | plain | 1 | 3.58 | **interior (mid)** | 0.322 | −23163 |
| wilhelm | plain | 1 | 2.71 | **interior (mid)** | 0.261 | −17951 |
| whitney | plain | 1 | 1.46 | interior (shallow) | 0.486 | −20398 |
| oran | plain | 1 | 1.44 | interior (shallow) | 0.497 | −15776 |
| johndoeII | plain | 1 | 1.37 | interior (shallow) | 0.852 | −15805 |

- **Upper-railed (α→6, τ≈0):** chromatica, freya, hamilton — scattering essentially
  unresolved; α is an upper limit, not a measurement.
- **Mid-interior, resolved τ:** zach (3.66), phineas (3.58), wilhelm (2.71) — α is
  actually measured here; these are the cases where a hidden sub-component could bias α.
- **Shallow-interior (α≈1.4):** johndoeII, oran, whitney — shallow frequency scaling
  with sizable τ.

### Finding 3 — Recorded fit quality (PPC reduced χ²)

Only 6 of 11 have a PPC file (`*_joint_ppc.json`, keys `chi2_chime`, `chi2_dsa`); the
fit summaries store no χ²/R²/Durbin-Watson, so the 3-level contract cannot be applied
to the rest from disk.

| burst | χ²_CHIME | χ²_DSA | note (good ≈0.8–1.5) |
|---|---|---|---|
| johndoeII | 1.14 | 1.03 | both good |
| mahi | 1.05 | 1.08 | both good |
| oran | 1.11 | 1.06 | both good |
| whitney | 1.15 | **1.68** | DSA elevated |
| wilhelm | **1.71** | 1.30 | CHIME elevated |
| phineas | 1.20 | **2.02** | DSA elevated |
| chromatica, freya, hamilton, isha, **zach** | — | — | **no PPC on disk** |

**Elevated per-band χ² = residual structure the single-component plain fit does not
capture** — a direct, data-driven flag for where a hidden temporal component is most
plausible: **wilhelm** (CHIME 1.71), **phineas** (DSA 2.02), **whitney** (DSA 1.68).

## Architecture / How a fit is produced

`run_joint_fit.py` builds per-band `FRBModel`s, refines an MLE init, and calls
`fit_joint_scattering` with mode flags; `multi` routes to the per-band proper-prior
path (`run_joint_fit.py:120`). The stored campaign invoked it with no mode flags →
plain mode. The mode-selection metadata the driver records (`run_joint_fit.py:169-178`)
is not present in these `joint_json` summaries, so mode was inferred from the sampled
parameter set (Finding 1).

## Implications for a multi-component re-fit campaign (factual)

- The multi-component evidence path is **greenfield on real data** — no real burst has
  been fit with it, so the campaign establishes new results, not just re-fits.
- **Mode switch, not just component count:** the stored fits are plain mode; the
  multi-component path is gain-marginal. A clean N-ladder must run **both** N=1 and N≥2
  through the gain-marginal multi path (`force_multi=True`, fixed `gain_s2`) for
  commensurate evidence (per the #37 research) — comparing a stored plain-single lnZ to
  a multi-comp lnZ is **not** apples-to-apples.
- **Data-driven target priority:** the elevated-χ² + mid-α + resolved-τ cases are the
  strongest multi-component candidates — **wilhelm** and **phineas** have *measured*
  single-component misfit (χ² 1.7–2.0). **zach** is the canonical bias case (α 3.66,
  τ 0.32) but has **no PPC on disk**, so its single-component misfit is unquantified.
- The upper-railed trio (chromatica, freya, hamilton; τ≈0) are poor multi-component
  candidates — there is no resolved scattering to re-attribute.

## Edge Cases / Constraints

- `casey` has no `joint_json` fit (separate `adv_casey_results.json` /
  `casey_multiscale_results.json` outputs) — outside this summary set.
- α bounds used were `[1.0, 6.0]` (wider lower bound than the code default `(2.0, 6.0)`).
- "Railed" judged within 0.05 of a bound; the shallow trio (≈1.4) are interior, not
  railed, but sit toward the lower end.

## Open Questions / Gaps

1. Fit quality is unrecorded for chromatica, freya, hamilton, isha, **zach** (no PPC,
   no goodness in the summary). zach's single-component χ² is the key missing number
   for prioritizing it as the canonical bias case.
2. Was casey ever joint-fit? Where do its results live, and in what mode?
3. Do the driver-written mode-metadata records (`run_joint_fit.py:169-178`) survive in
   a `<RUNS>/data/joint/` location distinct from `joint_json/`, i.e. are these summaries
   a re-export that dropped the mode field?

## References

- Files analyzed: 11 `joint_json/*_joint_fit.json`, 6 `*_joint_ppc.json`,
  `analysis/scattering-refit-2026-06/run_joint_fit.py`.
- Related: [research-multicomponent-joint-evidence.md](research-multicomponent-joint-evidence.md),
  [implement-dt-min-per-band.md](implement-dt-min-per-band.md), `CONTEXT.md` (burst nicknames, regimes).
