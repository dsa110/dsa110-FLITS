# Plan — Manuscript completion (Faber2026, CHIME–DSA co-detection scattering paper)

Scope/working plan for finishing the manuscript at `~/Developer/overleaf/Faber2026/`.
Authored 2026-06-24. Pursues **both forks in parallel**: (A) finalize the scattering-α
measurement (the headline, HPC-gated), and (B) reconcile manuscript text to the real
analysis state and tighten the already-backed non-α sections (zero new compute).

State verified against the three authoritative scattering-result files in
`analysis/scattering-refit-2026-06/` on 2026-06-24 (see "Scattering state" below).

## Overview

The manuscript is **~80% drafted** — all sections carry real prose except the author
list (stub) and the scattering-result narrative, which is written against a **stale,
over-pessimistic** view of the fits. The science-critical bottleneck is locking the
per-sightline scattering index α; almost everything else is either done or doable now.

Title: *Disentangling the Dispersion and Scattering Budgets of CHIME/FRB–DSA-110
Co-detected Fast Radio Bursts.* Positioning (`docs/adr/0001-two-band-leverage-positioning.md`):
empirical α from the same burst at ~0.6 GHz (CHIME) and ~1.4 GHz (DSA), vs the
CRAFT/Ocker lineage that measures one band and *assumes* α≈4.

## Full sample (N = 12 co-detections)

`casey, chromatica, freya, hamilton, isha, johndoeII, mahi, oran, phineas, whitney,
wilhelm, zach` (`configs/bursts.yaml`, `CONTEXT.md`). **Every subset analysis below must
state, in the manuscript, exactly which bursts it covers and why the rest are excluded.**
See the binding exclusion table.

## Scattering state — three maturity stages (reconciled 2026-06-24)

Not contradictory; three stages of the same fits, generated at different dates:

| Source | Measures | Verdict |
|---|---|---|
| `results/joint_fit_summary.md` (06-19) | human 4-lens adversarial trust, single-comp committed fits | **3 trusted**: johndoeII α=1.37, phineas α=3.58, wilhelm α=2.71 |
| `analysis/scattering-refit-2026-06/joint_gate_verdicts.md` | deterministic gate, same 11 fits | **0 PASS / 8 MARGINAL / 3 FAIL** (johndoeII/oran/whitney FAIL L1 α<1.5) |
| `analysis/scattering-refit-2026-06/joint_ladder/LADDER_SUMMARY.md` (06-23) | multi-component ladder | **8 provisional α** + 2 MARGINAL + 1 FAIL |

Critical caveats baked into `LADDER_SUMMARY.md` itself:
- **Every ladder α was fit under the unphysical mixed-PBF default** (CHIME powerlaw /
  DSA exp). whitney shows this is a *dominant* α systematic: all-exp α=5.12 vs
  all-powerlaw α=1.51, data prefer all-exp by ΔlnZ≈+2708. **All ladder α need an
  all-exp rerun before they are final.**
- Most component counts rest on **profiled-only** lnZ (not clean Bayes factors); the two
  bursts tested with fixed-s2 grids (whitney, johndoeII) had their profiled hint
  *overturned*. Component counts are PROVISIONAL pending s2 grids.
- Tension to resolve: the deterministic L1 gate FAILs johndoeII (α=1.37 < 1.5 floor),
  yet adversarial review *trusts* it as a genuine sub-Kolmogorov measurement. The 1.5
  floor may be too aggressive for a real sub-Kolmogorov α — decide whether to relax L1
  or treat sub-Kolmogorov α as a flagged special case.

Net: **provisional α for ~8 sightlines, 3 adversarially trusted, zero locked under the
strict gate.** The manuscript currently says "only 3 attempted, all 3 fail" — wrong on
the count (11 fits exist) and on the outcome (3 are trusted, not failed).

## Binding requirement — per-section sample & exclusion justification

Reviewers will ask why each subset analysis drops bursts. Every table/figure that covers
fewer than 12 must carry an explicit, defensible reason in the caption or text. Current
coverage and the justification each section must state:

| Analysis | N | Bursts INCLUDED | Excluded + **reason that must appear** |
|---|---|---|---|
| DM & sightline budget | 12/12 | all | none |
| Co-detection association (position+timing) | 12/12 | all | none for position/timing |
| — independent DM agreement | 2/12 | zach, oran | other 10: **CHIME DM extraction suspended pending the inter-channel unit audit** (`crossmatching/association_report.json`); position+timing only |
| Joint scattering α (fits exist) | 11/12 | all but casey | casey: **only single-band multiscale output, no joint `c0/γ` fit in `joint_json/`** |
| — adversarially trusted α | 3/12 | johndoeII, phineas, wilhelm | other 8: **α railed at a prior bound (chromatica/freya/hamilton→6.0), unconstrained (isha σ≈2.6, mahi), or pending multi-component + all-exp finalization (zach, whitney); oran refuted (CHIME nuisance railed)** |
| Isotropic energies E_iso | 6/12 | chromatica, hamilton, isha, phineas, wilhelm, zach | freya/mahi/johndoeII: **placeholder z=1.0, no spectroscopic host redshift → no luminosity distance**; oran/whitney: **FAIL-gated joint fit, dropped by the energy trust boundary**; casey: **no joint c0/γ fit**. (hamilton, chromatica carry **provisional/unpublished z** — flag in table.) |
| Scintillation Δν measured | 3/12 | casey, freya, wilhelm | other 9: **scint config-generation not yet run (stub at `flits/batch/batch_runner.py:262`)** — deferred, NOT unsuitable |
| Profile-bias α case study | 1 | zach (3.32→2.41) | demonstrator; **generalization pending the multi-component campaign** |
| Foreground/intervening census | 49 candidates / 12 sightlines | — | 29 confirmed, 7 background, 13 inconclusive; **MgII for 13 systems pending host-spectra location** (`results/mgii_inventory.csv`) |

Keep this table in sync as fork-A relocks α and fork-B's scint campaign runs.

## Fork A — finalize the scattering-α measurement (HPC-gated; the headline)

The detailed phased plan already exists: **`docs/rse/specs/plan-scattering-refit-validation.md`**.
This fork ADDS the all-exp-PBF requirement discovered after that plan (LADDER_SUMMARY,
06-23/06-24). Steps:

1. **All-exp PBF rerun** of every ladder fit (CHIME and DSA both exponential) — the
   mixed default is unphysical and inflates lnZ. SLURM campaign on
   `hpcc:/central/scratch/jfaber/flits-runs/` (`analysis/scattering-refit-2026-06/hpcc/run_burst.sbatch`,
   partition `expansion`). **GATED: cluster submission is an outward/infra op — confirm before launch.**
2. **Fixed-s2 component grids** (s2 ∈ {1,10,100}) for every burst whose component count
   currently rests on profiled-only lnZ (oran, isha, mahi, zach, phineas-DSA). A
   component is real iff ΔlnZ(N+1 vs N) ≳ 5 and sign-stable across s2.
3. **Re-fit α-railed sightlines** (chromatica, freya, hamilton; whitney, oran) at the
   physical α floor to separate prior artifact from genuine non-detection.
4. **Regenerate the gated verdict table** (`joint_gate_verdicts.md`) + adversarial
   re-verify each PASS with `.claude/workflows/fit-verify.js` (separate judge) +
   figure-review the fit-quality PNGs.
5. **Decide the L1 sub-Kolmogorov policy** (johndoeII α=1.37) — relax floor or flag.

Exit: every sightline carries a final PASS/MARGINAL/FAIL + reason; the trusted-α set is
locked; `results/joint_fit_summary.md` regenerated from the all-exp fits.

## Fork B — reconcile manuscript text + tighten backed sections (no compute)

**Manuscript lane is separate-active/dirty** as of 2026-06-24: `M main.tex,
sections/{intro,discussion,conclusions}.tex` carry uncommitted in-flight edits (abstract +
discussion + conclusions polish, still conservative framing). `sections/observations.tex`
and `sections/results.tex` are CLEAN. Do the scattering-narrative reconciliation as ONE
coordinated pass across observations + results + discussion + conclusions + abstract so
the framing stays internally consistent — **after** the in-flight edits are committed
(else piecemeal edits desync from the conservative in-flight framing).

1. **Commit/settle the in-flight manuscript edits first** (owner action), then:
2. **Reconcile the scattering verdict** everywhere it appears:
   - `sections/observations.tex` §"DSA-110 scattering fits" (lines ~34–48): replace
     "only three sightlines … all three fail" with the real state — 11 joint fits; 3
     adversarially-trusted measurements (johndoeII α=1.37 sub-Kolmogorov, phineas 3.58,
     wilhelm 2.71); the remainder railed/unconstrained or pending the multi-component +
     all-exp finalization. Keep the honest "final verdict pending" caveat.
   - `sections/results.tex` (lines ~27–49): the scattering panel is no longer
     "predicted-only with zero measured τ"; report the trusted measured τ/α and mark the
     rest withheld with per-burst reasons. Update Fig. caption.
   - `sections/discussion.tex`, `sections/conclusions.tex`, abstract (`main.tex`): align
     "principal measurement outstanding / refit underway" with whatever fork A locks.
3. **Add per-section exclusion justifications** (the table above) to every subset table's
   caption/text.
4. **Tighten the already-backed sections** (need no α): DM decomposition results, E_iso
   table (just merged, #42 — 6 bursts, flag provisional z for hamilton/chromatica),
   TOA/association, foreground census, the zach profile-bias case study, wilhelm
   two-screen scint case, bandpass/γ_D diagnostic.
5. **Author list** (`auth.tex` stub) — owner/collaboration action.

## Have now (write-up-ready)
DM budget (12/12); E_iso 6-burst table + calibration audit (on main, #42); TOA+positional
association (12/12); foreground census (49/29-confirmed); profile-bias demonstrator (zach);
wilhelm two-screen scint case; bandpass/γ_D diagnostic.

## Missing / externally blocked
Author list (collaboration); 2 host redshifts (hamilton, chromatica — provisional);
MgII absorber spectra (13 to locate); CHIME flux calibration (to extend energies).

## Success criteria

Automated:
- `analysis/scattering-refit-2026-06/joint_gate_verdicts.md` regenerated from all-exp fits;
  every one of the 12 (11 joint + casey) carries a FINAL flag + reason.
- `results/joint_fit_summary.md` regenerated; trusted-α set locked, no mixed-PBF fits cited.
- Manuscript compiles (`latexmk`/`pdflatex` in `~/Developer/overleaf/Faber2026/`).
- No "only three" / "all three fail" string remains in any `sections/*.tex`.

Manual:
- Each subset table/figure caption states its sample and exclusion reason (the table above).
- Scattering narrative is internally consistent across abstract/observations/results/
  discussion/conclusions.
- Author list populated; provisional-z flags present for hamilton/chromatica.

## References
- `docs/codetection-science-plan.md` — master analysis inventory.
- `docs/rse/specs/plan-scattering-refit-validation.md` — fork-A detailed phases.
- `analysis/scattering-refit-2026-06/{joint_gate_verdicts.md, joint_ladder/LADDER_SUMMARY.md}`,
  `results/joint_fit_summary.md` — scattering state of record.
- `docs/adr/0001-two-band-leverage-positioning.md` — paper framing.
- `analysis/burst_energies/CALIBRATION_REVIEW.md` — energy caveats.
