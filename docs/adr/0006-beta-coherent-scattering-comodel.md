# Sample beta as the fundamental scattering parameter

**Status:** accepted (2026-06-30)

## Context

The M3 and joint CHIME+DSA fitters sampled the frequency-scaling index `alpha`
in `tau(nu) = tau_1ghz * (nu/1 GHz)^(-alpha)` while the pulse-broadening
function (PBF) shape was selected independently via `FLITS_PBF` /
`FLITS_PBF_BETA` environment variables. That treats PBF functional form and
frequency scaling as uncorrelated knobs, which is physically inconsistent: both
follow from the same electron-density fluctuation spectrum
`P_n(q) propto q^{-beta}` (Cordes et al. 2025; Faber2026 §3.5).

## Decision

- **`beta` is the sampled parameter** in M3 and all joint-fit vectors
  (`JOINT_PARAM_NAMES*`), replacing `alpha`.
- **`alpha` is derived** at each likelihood evaluation:
  `alpha = 2*beta/(beta-2)` (thin-screen branch; `alpha=4` at `beta=4`).
- **PBF shape is coupled to `beta`**: power-law spectrum PBF for
  `beta < 4 - eps`; analytic exponential limit at `beta -> 4`.
- **`alpha_bounds` remains a deprecated alias** mapped to `beta_bounds` on the
  thin-screen branch for legacy callers and gate reporting.

## Consequences

- Re-run joint-fit campaigns before quoting population beta/alpha statistics.
- `gate_joint_committed.py` accepts either `beta` or legacy `alpha` keys in fit
  JSON; derived `alpha` is reported for literature comparison.
- Sub-Kolmogorov `alpha < 4` on the thin-screen branch is unreachable; such
  sightlines require the extended-medium (`beta > 4`) branch (future work).

## Rationale addendum (2026-07-01): an exponential PBF forces alpha = 4

Recorded post-acceptance because this argument previously lived only in code
docstrings (`turbulence.py` module docstring; `gaussian_powerlaw_convolution`
in `burstfit.py`) and in the manuscript methods (Faber2026 §3.5), with no
pipeline-side statement of the corollary that matters for fit-campaign design.

An exponentially modified Gaussian (EMG) burst model — a Gaussian pulse
convolved with a one-sided exponential PBF — is not scaling-agnostic. Within
thin-screen scattering the exponential kernel is *uniquely* the `beta = 4`
member of the power-law-spectrum PBF family, and `beta = 4` fixes `alpha = 4`:

1. For `P_n(q) propto q^{-beta}` with `2 < beta < 4`, the thin-screen PBF is
   exponential at small lag with a power-law tail `t^{-beta/2}` beyond the
   crossover `s_c = 2 ln(2/(4-beta))` (Cordes et al. 2025 §11.2; implemented in
   `gaussian_powerlaw_convolution`).
2. As `beta -> 4`, `s_c -> infinity`: the tail vanishes and the PBF reduces to
   the pure one-sided exponential. No other `beta` yields an exactly
   exponential PBF. Physically `beta = 4` is the square-law limit:
   `D_phi(rho) propto rho^2`, a Gaussian scattered image, hence an exponential
   PBF, with `theta_d propto nu^{-2}` and `tau propto theta_d^2 propto nu^{-4}`
   (derivation sketch in Faber2026 §3.5).
3. The inertial-range closure `alpha = 2*beta/(beta-2)` at `beta = 4` gives
   `alpha = 4` exactly (`turbulence.alpha_from_beta`).

Corollaries for fit design:

- **"EMG with fixed `alpha = 4`" is not a free modeling choice; it is the
  definition.** It is the `beta = 4` point of this ADR's co-model. Conversely,
  fixing `alpha = 4.4` (Kolmogorov) inside an exponential kernel is
  inconsistent: Kolmogorov `beta = 11/3` implies the power-law-tail PBF, not an
  exponential.
- **"EMG with free `alpha`" combines incompatible turbulence assumptions** (the
  shape asserts `beta = 4` while the scaling asserts otherwise). Free-alpha
  exponential fits — including the all-exp campaign behind ADR-0005 — therefore
  measure departure from the assumed scattering family (burst morphology,
  extended medium, inner scale), not a turbulence spectral index. This is why a
  shallow apparent `alpha < 4` does not map to any `beta` on the thin-screen
  branch; the extended-medium closure `alpha = 8/(6-beta)` for `beta > 4`
  (papers/Bhat_MultiFreqObsPulseBroadening_2004.md) is the branch such
  sightlines would need, and it is not implemented.
