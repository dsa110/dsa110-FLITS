# Experiment: power-law-spectrum PBF vs the thin-screen exponential

---
**Date:** 2026-06-23
**Author:** AI Assistant
**Branch:** `feat/powerlaw-pbf` (worktree off origin/main)
**Status:** Complete — implementation validated, hypothesis tested and **refuted for DSA**
**Related Documents:**
- [Experiment: wilhelm N-ladder](experiment-wilhelm-nladder.md) — established the DSA misfit + the shaky C1D2 "double"
- [Per-burst residual target list](cropped-residual-target-list.md) — wilhelm DSA χ²=4.70 (the motivating misfit)

---

## Hypothesis

The wilhelm DSA band shows a single-component misfit (χ²=4.70) that the exponential
model could only reduce by adding a second temporal component (C1D2), whose evidence
was **prior-sensitive and multimodal** (+286 with α railed to the floor, only +3.7
with wide α). Cordes review §11.2 shows that fitting a single thin-screen exponential
to data whose true PBF has a heavy power-law tail biases τ̂/α and can fake extra
structure. **Hypothesis:** the DSA misfit / C1D2 double is a PBF-shape artifact — a
single *power-law-spectrum* PBF (heavy `t^{-β/2}` tail) should fit the DSA band better
than the exponential, with one component.

## What was implemented

`scattering/scat_analysis/burstfit.py`:
- `gaussian_powerlaw_convolution(t, mu, sig, tau, beta)` — Gaussian ⊛ power-law-spectrum
  thin-screen PBF. Form from Cordes review §11.2 (after Ostashov & Shishov 1978; Lee &
  Jokipii 1975; Lambert & Rickett 1999): exponential core `e^{-s}` for `s ≤ s_c`, heavy
  tail `(s/s_c)^{-β/2}` for `s > s_c`, with `s = lag/τ` and crossover
  `s_c = 2 ln(2/(4−β))`. Computed by zero-padded FFT linear convolution, area-normalized
  to match `analytic_gaussian_exp_convolution`.
- Env-driven selector in `FRBModel.__init__`: `FLITS_PBF` (`exp` default / `powerlaw`),
  `FLITS_PBF_BETA` (default 11/3 = Kolmogorov). Threads through multiprocessing pools and
  the prepare()/fit stack without signature changes.
- `__call__` dispatches to the power-law branch when `self.pbf == "powerlaw"`.

### Implementation validation (oracle, sha256=30545a93)
- **β→4 reduces to the exponential** (moments match: centroid rel-diff 2.2e-3, rms 2.3e-5).
  The residual ~0.5-sample time offset between the FFT path and the closed form is fully
  degenerate with the free `t0`, so it cancels in any fit/evidence.
- Area-normalized (∫≈1), causal (pre-μ mass ~1e-17), finite over β∈[2.01, 5.0].
- Kolmogorov tail (β=11/3) carries 5.7× the late-time mass of the exponential.

## Result — the DSA band rejects heavy tails (hypothesis refuted)

**Joint CHIME+DSA, wilhelm C1D1** (nlive=800, α∈[2,6], force-multi, gain-marginal),
exp vs power-law(β=11/3), identical data/prep:

| PBF | lnZ | α | τ_1GHz |
|---|---|---|---|
| exponential | 3809.86 ± 0.46 | 2.622 | 0.2551 |
| power-law β=11/3 | 3725.58 ± 0.46 | 2.621 | 0.2522 |

**ΔlnZ(pl − exp) = −84.3** — the global Kolmogorov power-law is decisively *worse*.
(The exp run reproduces the stored baseline 3809.59, confirming commensurability.)

**Per-band decomposition** (`pbf_band_compare.py`, best single-component gain-marginal
lnZ on identical prepared data, PBF toggled in-place; cross-check sha256=965cf7b5):

| band | exp lnZ (χ²) | pl β=2.5 | pl β=3.0 | pl β=11/3 | pl β=3.9 |
|---|---|---|---|---|---|
| CHIME | 4011.13 (1.127) | −42.66 | −9.05 | **+6.15** | −4.64 |
| DSA | −134.51 (4.701) | −842.46 | −601.46 | **−90.36** | −5.26 |

(power-law columns are ΔlnZ vs that band's exponential.)

The per-band ΔlnZ sum (CHIME +6.15, DSA −90.36 = **−84.21**) reproduces the full joint
nested-sampling ΔlnZ (**−84.28**) to 0.07 — the screen and the joint fit agree.

### Interpretation
1. **DSA monotonically rejects a heavier-than-exponential tail.** Every power-law β
   worsens the DSA fit; the heavier the tail (smaller β), the worse (β=3.9 ≈ exp; β=2.5
   catastrophic). The fit tries to compensate by railing α→6 (steeper τ(ν)) and cannot
   recover. **The DSA misfit and the C1D2 "double" are NOT a power-law-PBF tail artifact**
   — a heavy tail is the *opposite* of what the DSA band wants. Combined with the already
   shaky wide-α C1D2 evidence (+3.7), the DSA misfit's cause is most likely genuine
   temporal structure or a *lighter*-than-exponential / asymmetric shape this power-law
   form cannot represent (it only spans exp→heavier).
2. **CHIME mildly prefers a Kolmogorov tail** (β=11/3, +6.15 lnZ, χ² 1.127→1.119). A
   real but modest effect — physically sensible (lower frequency, longer scattering, more
   tail in-band) — but swamped by the DSA penalty in any *global* (single-PBF) joint fit.

## Follow-up result — per-band PBF (implemented + confirmed)

`run_joint_fit.py` now takes `--pbf-C/--pbf-D` (+ `--beta-C/--beta-D`), setting
`model_C.pbf`/`model_D.pbf` after prepare() (the two bands are separate FRBModel
instances, so each carries its own PBF). Joint wilhelm C1D1, CHIME=powerlaw(β=11/3),
DSA=exp, identical settings (nlive=800, α∈[2,6], force-multi, gain-marginal):

| configuration | lnZ | ΔlnZ vs all-exp |
|---|---|---|
| all-exp | 3809.86 ± 0.46 | 0 |
| all-powerlaw (β=11/3) | 3725.58 ± 0.46 | **−84.3** |
| **per-band (CHIME pl, DSA exp)** | **3813.83 ± 0.47** | **+3.97** (~6σ) |

The per-band PBF is the winner: it captures CHIME's tail gain (+3.97, consistent with
the per-band screen's +6.15 reduced by the shared-τ/α coupling) **without** DSA's −90
penalty. Confirms in the full joint fit that the two bands want different PBF shapes —
a single global PBF is the wrong model. **Per-band PBF should be the default for joint
CHIME+DSA fits.**

## Implications / follow-ups
- ✅ **Per-band PBF** — done (above).
- The β↔x_τ physical link (x_τ = 2β/(β−2); β=11/3 ↔ α=4.4) is **not** wired in; α was
  kept free and independent of β. CHIME's preferred β=11/3 ↔ α=4.4 ≈ its fitted α=4.04,
  a mild corroboration worth a dedicated linked-prior test.
- DSA wanting a tail ≤ exponential motivates a *thick-medium* (rounded-rise) or
  Gaussian-image PBF for DSA, not a heavier one.
- See [scintillation subband Δν_d](experiment-scint-subband-alpha.md): the scattering and
  scintillation screens are distinct (C1≈85), so PBF/α from pulse-broadening probes a
  different screen than the resolved scintillation.

## Artifacts
- Code: `scattering/scat_analysis/burstfit.py`, `analysis/scattering-refit-2026-06/run_joint_fit.py` (this branch)
- Joint runs: `…/flits-refit/nladder_{exp,pl,perband}/data/joint/wilhelm_joint_fit_C1D1.json`
- Per-band screen: `…/flits-refit/nladder/pbf_band_compare.py`
