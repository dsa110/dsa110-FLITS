# Extended-medium PBF family for shallow-alpha sightlines

**Status:** deferred (2026-07-06) — owner decision: sequenced behind the
β-coherent thin-screen campaign
([plan](../rse/specs/plan-beta-coherent-thin-screen-campaign.md)). The
thin-screen family (power-law-tail members for `beta < 4`, exponential limit at
`beta = 4`) has not yet been run successfully on the full sample; that pass
comes first, assuming thin-screen throughout. This ADR re-opens **per burst, on
evidence**: a β posterior railing at the `beta = 4` boundary (ADR-0004
rail-MARGINAL) and/or structured rise-time residuals in the campaign fits is
the trigger for testing the extended-medium kernel on that sightline. The
proposal below is unchanged; only its sequencing is decided.

**Depends on:** [ADR-0003](0003-single-exponential-pbf.md),
[ADR-0004](0004-l1-sub-kolmogorov-alpha-floor.md),
[ADR-0006](0006-beta-coherent-scattering-comodel.md)

## Context

Several sightlines fit shallow apparent scaling (`alpha < 4`): casey `~3.9`
in the citable roster, legacy freya `4.2` borderline, and the pulsar
literature's population mean (`alpha = 3.86 +/- 0.16`, Bhat et al. 2004).
Under the beta co-model (ADR-0006) these are unrepresentable: **both branches
of the pure power-law-spectrum closure give `alpha >= 4`**
(`alpha = 2*beta/(beta-2)` for `beta < 4`; `alpha = 8/(6-beta)` for
`4 < beta < 6`; both meet `alpha = 4` at the exponential limit `beta = 4` —
Bhat 2004 Eq. 4, verified in
`docs/literature/Bhat_MultiFreqObsPulseBroadening_2004.md`). A shallow alpha
is therefore not a "sub-Kolmogorov beta" on ANY closure branch; ADR-0004
handles it as a MARGINAL flag, and ADR-0006's original addendum misattributed
it to the (unimplemented) `beta > 4` branch — corrected 2026-07-05.

The literature offers four physical mechanisms for `alpha < 4`, in the order
Bhat 2004 weighs them:

1. **PBF misspecification (geometry, not turbulence).** Bhat's PSR J1853+0545:
   the thin-screen PBF yields `alpha = 3.1 +/- 0.2`, while the extended-medium
   PBF (scattering material distributed uniformly along the LOS, Williamson
   1972/1973 kernel — their `PBF_2`) yields alpha consistent with Kolmogorov
   `4.4` **on the same data**. The extended-medium PBF rises more slowly and
   decays with a heavier effective tail than the thin-screen exponential, so
   forcing a thin-screen kernel biases the per-band tau ratios shallow.
2. **Inner-scale cutoff** in the wavenumber spectrum (`~300–800 km` suffices
   for the population-mean departure) — makes the *apparent* alpha frequency-
   dependent rather than a constant.
3. **Refractive bias** near the weak/strong-scattering transition (stronger at
   higher frequency, weakens the apparent index).
4. **Transversely truncated screens** (filaments/sheets).

Mechanism 1 is the only one expressible inside the existing fitter as a PBF
family choice; 2 adds a parameter and breaks the constant-alpha assumption the
tau(nu) model is built on; 3 and 4 are sightline pathologies better handled as
exclusion criteria than model components.

## Decision (proposed)

- **Implement the extended-medium (uniform-LOS) PBF as a selectable family**
  alongside the thin-screen family, per band, in `burstfit.py`'s kernel layer:
  the Williamson uniform-medium kernel with the same beta-coupling contract as
  ADR-0006 (shape and frequency scaling both derived from the sampled beta).
- **Model selection, not reparameterization:** shallow-alpha bursts are
  re-fit under both geometries and the winner is chosen by evidence/BIC —
  exactly the machinery that adjudicated per-band PBF families in the all-exp
  campaign. No change to `alpha_from_beta`; no `beta > 4` branch is added
  (it would move alpha the wrong way — see the ADR-0006 correction).
- **Reporting:** a burst whose evidence prefers extended-medium gets its alpha
  quoted under that geometry with the geometry named; if thin-screen wins and
  alpha stays `< 4`, the MARGINAL flag of ADR-0004 stands and the residual
  shallowness is attributed to mechanisms 2–4 in prose, not fit parameters.

## Consequences

- `scattering/scat_analysis/burstfit.py`: new PBF kernel (uniform-medium
  Williamson form) + beta coupling; `turbulence.py` untouched.
- Priors/model-scan config gain a `pbf_geometry` axis (thin | extended);
  default stays thin-screen so existing campaigns are unaffected.
- Candidate first targets: casey (roster member, `alpha ~ 3.9`), then the
  MARGINAL band of ADR-0004.
- Manuscript §3.5 gains a sentence distinguishing geometry-driven from
  spectrum-driven scaling; the citable roster (ADR-0005) is only touched if a
  re-fit changes a member's alpha class — that re-opening is itself gated on
  the owner accepting this ADR.

## Alternatives considered

- **`beta > 4` closure branch** (the original ledger framing): rejected —
  `alpha = 8/(6-beta)` steepens above 4, so it cannot explain shallow alpha
  (Bhat 2004 Eq. 4).
- **Inner-scale parameter:** physically favored by Bhat for the population
  trend, but it makes alpha(nu) non-constant, which breaks the
  `tau_1ghz * nu^-alpha` contract everywhere downstream (gates, ladders,
  roster). Revisit only if extended-medium fits fail to absorb the shallow
  cases.
- **Status quo (MARGINAL flag only):** keeps shallow-alpha bursts
  uninterpretable in beta; acceptable for the current manuscript but leaves
  casey's roster alpha resting on a geometry assumption its own value
  contradicts.
