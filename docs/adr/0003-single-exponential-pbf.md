# Adopt a single exponential pulse-broadening function for both bands

**Status:** accepted

## Context

The default joint fitter used a **mixed PBF** — CHIME power-law, DSA exponential.
That default is physically incoherent: a single sightline samples one scattering
medium, so the PBF *functional form* is fixed by the screen geometry and only its
timescale τ scales with frequency. The per-band preference that motivated the
default (e.g. wilhelm, ΔlnZ ≈ +4 for per-band over all-exponential) is the model
using PBF-shape freedom to absorb band-specific *profile* structure — overfitting,
not medium physics. The all-exp campaign (`ALLEXP_PBF_RUN.md`, 2026-06-24) showed
the PBF choice is a *dominant* α systematic for marginal bursts while immaterial
for clean ones.

## Decision

Fit **every band with a single exponential thin-screen PBF**
(`--pbf-C exp --pbf-D exp`). All citable α come from the all-exp campaign; no
mixed-PBF or all-power-law α is quoted in the manuscript. For the well-constrained
single-screen sightlines the choice is immaterial (|Δα| ≤ 0.1); it matters only
where multiplicity or low S/N already preclude a citable α.

## Consequences

- `results/joint_fit_summary.md` and `joint_gate_verdicts.md` must be regenerated
  from all-exp fits; mixed-PBF α are superseded (tracked follow-up).
- The marquee profile-bias demonstrator **moves off zach onto whitney
  (FRB 20220310F)**: zach's multiplicity correction *reverses sign* with the PBF
  (mixed 3.32→2.41 vs all-exp 3.319→4.59), so it is unreliable as a clean
  demonstrator and is **withheld** (see [0004](0004-l1-sub-kolmogorov-alpha-floor.md)
  context and the decision map). whitney's second DSA component is confirmed real
  (ΔlnZ +2706/+2683/+2671 across s²=1/10/100, no sign-flip), so its α 1.5→5.12
  unrailing is real physics, not a prior artifact.
- The manuscript states the PBF choice explicitly (`budget.tex` PBF argument) and
  cites the |Δα| ≤ 0.1 robustness for the well-constrained set (done, Faber2026
  PR #9).
- Reproducibility hazard to fix: two s²-grid generations coexist — stale
  mixed-PBF `analysis/scattering-refit-2026-06/joint_ladder/*_s2-*.json` (which
  `_s2verdict.py` reads by default) vs the canonical all-exp grids
  (`joint_ladder/allexp_json/*_pbf-exp-exp.json`, already local). The stale set
  must be deleted or `_s2verdict.py` repointed (tracked follow-up). The zach all-exp
  C2D3 grid is already pulled and adjudicated (rescue fails → single-comp α=3.319,
  `ALLEXP_PBF_RUN.md:106`).
