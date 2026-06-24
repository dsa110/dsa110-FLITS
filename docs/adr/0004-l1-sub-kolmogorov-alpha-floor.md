# L1 sub-Kolmogorov α-floor policy

**Status:** accepted (panel recommendation 2026-06-24; code/verdict implementation deferred)

## Context

The operative joint-fit L1 gate hard-FAILs α below **1.5** — hardcoded
`ALPHA_MIN, ALPHA_MAX = 1.5, 6.0` in
`analysis/scattering-refit-2026-06/gate_joint_committed.py:26`, which produced the
committed `joint_gate_verdicts.md` (johndoeII 1.373, oran 1.439, whitney 1.458 all
"FAIL | L1 alpha=… outside (1.5,6.0)"). Note the floor is **not** in
`flits/fitting/VALIDATION_THRESHOLDS.py`: its `ALPHA_MARGINAL_MIN = 2.0` has **zero
Python consumers** (a dead constant), and the "1.5 < α < 6.0" bound in
`.cursor/rules/AGENT_CONFIGURATION_FLITS.md:97` is prose that happens to match the
real gate while the same contract's pseudocode (line 249) and L3 text (line 307)
say 2.0 — a pre-existing doc inconsistency. The single operative number is 1.5, in
`gate_joint_committed.py`. johndoeII reads **α ≈ 1.37** — adversarially trusted
(2026-06-19, 4-lens workflow; α=4 rejected at Δ(−2lnL) ≈ 2400) as a genuine
sub-Kolmogorov measurement, and *un-railed* under the prior `alpha_bounds = [1.0,
6.0]` with a tight ±0.05 posterior and clean cross-band χ², so it FAILs only on the
1.5 floor. That floor conflates a *physical-impossibility* bound with a
*Kolmogorov-prior* bound and has no first-principles basis: a Kolmogorov thin
screen gives α = 4.4, a square-law spectrum α = 4.0, but multi-screen, anisotropic,
or inner-scale-truncated media flatten the observed index well below 2, and
sub-Kolmogorov indices are observed in the FRB/pulsar literature. FAILing α = 1.37
rejects a measurement for being *informative* (sub-Kolmogorov is the regime the
two-band lever arm exists to probe), not for being unphysical.

## Decision

**Lower the operative gate floor from α = 1.5 to α = 1.0** — edit
`ALPHA_MIN` in `gate_joint_committed.py:26` (and its test) — and reclassify the band
**1.0 ≤ α < 2.0 as L3 physics-flag MARGINAL** ("sub-Kolmogorov — inspect"), not
FAIL. Keep a hard FAIL only below α = 1.0 (where τ ∝ ν^−α is so flat it is
effectively achromatic and "scattering index" loses meaning). Reconcile the dead
`ALPHA_MARGINAL_MIN` constant and the contradictory contract prose to 1.0 in the
same pass so the SSOT is internally consistent. Separately, a
posterior whose median sits within ~3σ of *either* prior bound is flagged
**rail-MARGINAL regardless of value** — railing, not the number, is the
disqualifier. johndoeII is then citable as a sub-Kolmogorov measurement (overall
MARGINAL).

## Consequences

- Sets johndoeII into the citable-α set as a flagged sub-Kolmogorov result, and
  couples to the energies sample (oran α=1.44, whitney-single-comp α=1.46 also fall
  in the new 1.0–2.0 sub-Kolmogorov MARGINAL band — currently FAIL on the 1.5
  gate per `joint_gate_verdicts.md`) — see the decision map #4/#6.
- **Implementation deferred** (tracked): the gate still hardcodes `ALPHA_MIN = 1.5`
  in `gate_joint_committed.py`; changing it to 1.0 regenerates *every* gated joint
  verdict, so it must be done as a reviewed pass with full regeneration (+ updating
  `test_gate_joint_committed.py` and the contradictory `VALIDATION_THRESHOLDS.py` /
  contract prose), not a silent one-line edit. Until then, code and this ADR are
  knowingly inconsistent.
- The manuscript states the floor and its physical justification in one sentence
  ("α ≥ 1 required for a meaningful τ ∝ ν^−α scaling; 1 ≤ α < 2 admitted as
  sub-Kolmogorov, attributed to multi-screen/anisotropic scattering") so the
  johndoeII MARGINAL flag is defensible to a referee.
- The CONTEXT.md "scattering index" entry records the new lower bound and the
  sub-Kolmogorov convention.
