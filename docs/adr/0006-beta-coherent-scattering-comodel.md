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
