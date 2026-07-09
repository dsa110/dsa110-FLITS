# Decision map — Faber2026 manuscript completion

Loose idea: finish the CHIME–DSA co-detection scattering manuscript. The text
reconciliation is done (Faber2026 PR #9); what remains are **scientific/naming
decisions** that several analyses depend on. This map sequences them. Bootstrapped
2026-06-24 from the session that landed PR #9 — frontier surfaced from that work,
not a separate grilling; **add/correct tickets freely**.

Resolve via: `/decision-mapping <this file> #<n>`.

## Frontier (resolved 2026-06-24, 3-expert panel)

All seven tickets are answered. **#1, #2, #5** are fully settled (ADR-0002/0004/0003).
**#4 locked 2026-06-26** ([ADR-0005](../../adr/0005-citable-alpha-roster.md)): **5 fully
adjudicated α** + **3 provisional** (pending all-exp fixed-s²) + whitney multiplicity
exemplar. Graded from `_a1_fits/` via `grade_allexp.py`.
**#3** partial s² remains for Tier B (johndoeII/oran/phineas) + isha/mahi.
**#6, #7** unblocked for Tier A; full N=8 after Tier B s² pull.

## Assets / state of record

- [`plan-manuscript-completion.md`](plan-manuscript-completion.md) — full scope, per-section exclusion table.
- [`ALLEXP_PBF_RUN.md`](../../../analysis/scattering-refit-2026-06/joint_ladder/ALLEXP_PBF_RUN.md) — all-exp campaign.
- [`joint_fit_summary.md`](../../../results/joint_fit_summary.md) — 06-19 adversarial trust (mixed-PBF).
- Faber2026 PR #9 — narrative reconciliation + α-PBF/whitney figures (open).
- `.agents/deferred-tasks.md` — energies `@decision` item.

---

## #1: Canonical nickname↔TNS designation map

Blocked by: —
Type: Discuss

### Question

What is the authoritative nickname↔TNS map for the 12 co-detections, and what is
its single source of truth? An earlier draft of this map asserted a three-way
disagreement across agent memory, the manuscript/`alpha_pbf_systematic` figure,
and `configs/bursts.yaml` — **that "conflict" was a transcription error in this
scratch map**, not a real artifact disagreement (see Answer). #8 had already
churned johndoeii once (→ FRB 20230814B), which is what raised the alarm.

### Answer

**Resolved — [ADR-0002](../../adr/0002-canonical-burst-naming.md).** A close-out
review (2026-06-24, 3-expert panel) traced every *committed* artifact
(`burst_metadata.py::_FALLBACK_TNS`, `configs/bursts.yaml`, the manuscript table,
the figures) and found them **unanimous** — the apparent conflict was the
transcription slip above. Canonical map (verified): zach=FRB 20220207C,
whitney=FRB 20220310F, oran=FRB 20220506D, isha=FRB 20221113A,
wilhelm=FRB 20221203A, phineas=FRB 20230307A, freya=FRB 20230325A,
johndoeii=FRB 20230814B, hamilton=FRB 20230913A, mahi=FRB 20240122A,
chromatica=FRB 20240203A, casey=FRB 20240229A. SSOT =
`scattering/scat_analysis/burst_metadata.py::_FALLBACK_TNS` (committed);
`configs/bursts.yaml` owns burst *properties*. `chimedsa_burst_specs.csv` is
gitignored/absent and **must not** be cited as the registry (`CLAUDE.md`
corrected). FRB 20240203A (chromatica) and FRB 20230814B (johndoeii) are distinct
bursts, not aliases.

---

## #2: L1 sub-Kolmogorov α-floor policy

Blocked by: —
Type: Discuss

### Question

The operative joint-fit L1 gate FAILs α below **1.5** (hardcoded
`ALPHA_MIN = 1.5` in `gate_joint_committed.py:26`; this produced
`joint_gate_verdicts.md`. `VALIDATION_THRESHOLDS.py`'s `ALPHA_MARGINAL_MIN=2.0` is
a dead constant with zero consumers — not the operative floor). johndoeII reads
α≈1.37 (adversarially trusted 06-19 as a genuine sub-Kolmogorov measurement),
below that floor. Lower/relax the floor to admit genuine sub-Kolmogorov α, or keep
it and treat sub-Kolmogorov as a flagged special case? Sets whether johndoeII is
citable.

### Answer

**Resolved — [ADR-0004](../../adr/0004-l1-sub-kolmogorov-alpha-floor.md).** Lower
the operative gate floor `ALPHA_MIN` 1.5 → 1.0 in `gate_joint_committed.py:26` and
reclassify 1.0 ≤ α < 2.0 as L3 physics-flag **MARGINAL** ("sub-Kolmogorov —
inspect"), not FAIL — the floor had no first-principles basis (it conflated a
Kolmogorov-prior bound with a physical-impossibility bound), and
multi-screen/anisotropic media legitimately flatten α below 2. A posterior within
~3σ of *either* prior bound is separately flagged rail-MARGINAL regardless of value
(already implemented — chromatica/freya/hamilton rail the 6.0 upper bound in
`joint_gate_verdicts.md`). johndoeII (α≈1.37, un-railed, tight ±0.05) is therefore
citable as a sub-Kolmogorov result (overall MARGINAL). **Code implementation
deferred**: the gate hardcodes `ALPHA_MIN = 1.5`; changing it regenerates every
gated joint verdict, so it is a reviewed full-regeneration pass (+ test + the dead
`ALPHA_MARGINAL_MIN` / contract prose, tracked in `.agents/deferred-tasks.md`), not
a silent one-line edit — code and ADR are knowingly inconsistent until then.

---

## #3: Component counts via fixed-s² grids

Blocked by: —
Type: Research

### Question

Which multi-component counts are *real* (ΔlnZ(N+1 vs N) ≳ 5 and sign-stable across
s²∈{1,10,100}) vs profiled-only artifacts? The grid JSONs already exist
(`analysis/scattering-refit-2026-06/joint_ladder/*_s2-*.json`); `_s2verdict.py`
adjudicates. Settle per burst: oran, isha, mahi, zach (C2D3), phineas-DSA, whitney.

### Answer

**Methodology locked; per-burst adjudication partial.** Criterion stands:
a component is *real* iff ΔlnZ(N+1 vs N) ≳ 5 **and** sign-stable across
s²∈{1,10,100} (`_s2verdict.py`). **whitney C2D2 confirmed** real and stable
(+2706/+2683/+2671 at s²=1/10/100, no sign-flip; PR #9) — the clean marquee case.
**zach C2D3 RESOLVED — rescue fails**: the all-exp fixed-s² grid is already local
and adjudicated (`allexp_json/zach_joint_fit_C2D3_pbf-exp-exp.json`; jobs
64542330–64542345; `ALLEXP_PBF_RUN.md:106`), ΔlnZ +1443/−759/−0.4 — *not*
sign-stable, so C2D3 is rejected as prior-driven and zach falls back to the
single-component all-exp α = 3.319 ± 0.013. oran/isha/mahi/phineas-DSA
**unrun/unread**. **Reproducibility hazard (tracked):** `_s2verdict.py` reads the
*stale mixed-PBF* `*_s2-*.json` grids by default — these must be deleted or
`_s2verdict.py` repointed to the canonical all-exp grids before any new per-burst
verdict, or the adjudication inherits the superseded PBF (see
[ADR-0003](../../adr/0003-single-exponential-pbf.md), `.agents/deferred-tasks.md`).

---

## #4: Canonical citable-α set

Blocked by: #1, #2, #3
Type: Discuss

### Question

Which sightlines get a *quoted* α in the manuscript? `joint_fit_summary` = 3
(johndoeII/phineas/wilhelm, mixed-PBF); `ALLEXP_PBF_RUN` = 7 well-constrained
(|Δα|≤0.1); the figure colors 8 green (incl. zach). Reconcile under all-exp + final
component counts (#3) + L1 policy (#2) + locked naming (#1) into one citable set
with per-sightline FINAL PASS/MARGINAL/FAIL. PR #9 text stays count-free until this
locks.

### Answer

**Locked 2026-06-26 — [ADR-0005](../../adr/0005-citable-alpha-roster.md).** Membership
rule unchanged; roster graded on all-exp `_a1_fits/` with ADR-0004 floor.

**Tier A (5, fully adjudicated):** casey 2.40, wilhelm 2.56 (DSA-shape caveat),
chromatica 3.28, zach 3.32 (C1D1; no profile-bias claim), freya 4.36.

**Tier B (3, provisional — all-exp fixed-s² pending):** johndoeII 1.53, oran 2.66,
phineas 3.32.

**Multiplicity exemplar (prose):** whitney 5.12 (C2D2, fixed-s² confirmed).

**Excluded:** mahi, isha, hamilton; zach C2D3 multiplicity claim.

Mixed-PBF `joint_gate_verdicts.md` superseded for citation. Faber2026: **N = 5** safe
now; **N = 8** target after Tier B s² adjudication (#6 energies follows same split).

---

## #5: zach disposition

Blocked by: #3, #4
Type: Discuss

### Question

zach C2D3 all-exp α=4.59 — the marquee profile-bias number (mixed-PBF 2.41) was
PBF-confounded; single-comp α is PBF-insensitive (3.32→3.319), so the multiplicity
correction *reverses sign* with the PBF. Cite zach for profile bias (which
direction?), or withhold pending C2D3 confirmation under all-exp? Note whitney
already serves as the clean marquee case in PR #9, so zach may simply be dropped
as a demonstrator.

### Answer

**Withhold zach as a profile-bias demonstrator.** The multiplicity correction
*reverses sign* with the PBF (mixed 3.32→2.41 vs all-exp 3.319→4.59) while the
single-component α is PBF-insensitive (3.32→3.319), so the C2D3 number is
PBF-confounded and not a reliable demonstrator in *either* direction. whitney
(FRB 20220310F) already carries the clean marquee case in PR #9 (real 2nd DSA
component, no sign-flip, α 1.5→5.12), so zach is simply dropped as a demonstrator
— not cited for profile bias. The all-exp C2D3 grid (now local + adjudicated, #3)
*confirms* this: the rescue fails (ΔlnZ not sign-stable), so zach is single-component
with no citable multiplicity correction. Its clean single-component all-exp
α = 3.319 (PBF-insensitive, un-railed, GOOD range) still stands as an ordinary
scattering measurement; only the *multiplicity/profile-bias* claim is withheld.
Recorded in [ADR-0003](../../adr/0003-single-exponential-pbf.md).

---

## #6: Energies sample + table reconciliation

Blocked by: #1, #4
Type: Discuss

### Question

Manuscript `tab:burst-energies` = 8 rows (casey/chromatica/hamilton/isha/johndoeII/
mahi/phineas/wilhelm) + abstract "eight … energies"; FLITS #42 regenerated to 6
(chromatica/hamilton/isha/phineas/wilhelm/zach). Which sample is canonical? Then
reconcile the manuscript table + abstract count to it and add the exclusion caption
(no spec-z / FAIL-gated joint fit / no joint c0,γ fit). Needs naming (#1) and the
quality-passing set (#4, energy trust boundary #39).

### Answer

**Selection rule decided; numeric reconciliation stays the tracked `@decision`.**
The energies sample is **not** the 12-burst superset — per the per-section sample
rule it is the subset with (a) a spec-z, (b) a quality-passing DSA-band fluence,
and (c) the data for the energy calc, with every excluded burst named in the table
caption. The 6-row #42 regen (by nickname: chromatica/hamilton/isha/phineas/wilhelm/
zach) and the 8-row manuscript table (casey/chromatica/hamilton/isha/johndoeII/mahi/
phineas/wilhelm) diverge because they were cut on *different* inclusion criteria;
neither is canonical until the citable-α / quality-passing set (#4) finalizes under
the regen pass — and zach's presence in the #42 set is suspect given #5. So the
abstract's "eight … energies" stays **unsettled** and the exact row set is the open
`@decision` in `.agents/deferred-tasks.md`: reconcile once #4 locks, then align
table + abstract count + exclusion caption to the quality-passing-with-spec-z set.

---

## #7: Figure↔text consistency pass

Blocked by: #3, #4
Type: Prototype

### Question

Once the α set locks, align figures/text: freya α=4.48 (per-burst fit,
`budget.tex:260`) vs 4.356 (all-exp, `alpha_pbf_systematic`); zach colored green
"well-constrained" despite blocking the ladder (#5). Regenerate the figures and
reconcile in-text values. (Reviewer nitpicks from PR #9.)

### Answer

**Resolution decided; execution is a Fork-B + figure-regen follow-up.** Once #4
locks: (1) quote the **all-exp** freya α (4.356, `alpha_pbf_systematic`)
everywhere; `budget.tex:260`'s per-burst 4.48 is superseded and edited to match.
(2) **Recolor zach** off "well-constrained green" — it is withheld (#5), so it must
not read as citable in `alpha_pbf_systematic`. (3) Figures derive burst labels from
`burst_metadata` rather than hard-coding (ADR-0002). These are tracked in
`.agents/deferred-tasks.md` (figure regen) and execute on the **separate Faber2026
repo** in the Fork-B consistency pass, gated on #4 — not in this FLITS commit.

---

## Resolved inline (not tickets)

- Stale "only three / all three fail / predicted-only" narrative → reconciled, PR #9.
- Single-exponential PBF adoption (per-band PBF unphysical and immaterial, |Δα|≤0.1
  for clean sightlines) → settled by the all-exp campaign.
- Marquee multiplicity case → whitney (FRB 20220310F); moved off zach (PBF-confounded).
