# Canonical citable-α roster (decision-map #4)

**Status:** accepted (locked 2026-06-26; **3 members provisional** pending all-exp fixed-s² grids)

**Depends on:** [ADR-0002](0002-canonical-burst-naming.md),
[ADR-0003](0003-single-exponential-pbf.md),
[ADR-0004](0004-l1-sub-kolmogorov-alpha-floor.md)

## Context

Decision-map [#4](../rse/specs/decision-map-manuscript-completion.md#4-canonical-citable-α-set)
locked the **membership rule** on 2026-06-24 but withheld the numeric roster until
all-exp fits were graded under ADR-0004 (`ALPHA_MIN` 1.0, sub-Kolmogorov L3 band).
The canonical all-exp JSONs live in
`analysis/scattering-refit-2026-06/_a1_fits/`; grades were reproduced with
`grade_allexp.py` (2026-06-26).

## Membership rule

A sightline gets a **quoted α** in the manuscript iff **all** hold:

1. **All-exp PBF** (`pbf_C=pbf_D=exp`, ADR-0003).
2. **Un-railed** at both prior bounds (α not within 0.1 of 1.0 or 6.0).
3. **PBF-insensitive:** |Δα_mixed→all-exp| ≤ 0.1 for the chosen model tag.
4. **Component count adjudicated** where multiplicity matters (fixed-s² grid via
   `_s2verdict.py` on all-exp JSONs only; zach C2D3 rejected).
5. **Gate:** L1 PASS and L2 not FAIL on the canonical all-exp fit + paired PPC
   (`gate_one`, ADR-0004 floor). FINAL=MARGINAL is expected (L3 τ×Δν unevaluable).

**Profile-bias demonstrator** is separate: whitney (FRB 20220310F) only; zach is
withheld for that claim ([decision-map #5](../rse/specs/decision-map-manuscript-completion.md#5-zach-disposition)).

## Locked roster — quoted α

### Tier A — fully adjudicated (5)

Shared-ζ or single-component with all-exp fixed-s² where needed. Safe to cite now.

| Nickname | TNS | Model | α (all-exp) | χ² C/D | Notes |
|----------|-----|-------|-------------|--------|-------|
| casey | FRB 20240229A | sharedζ | 2.40 ± 0.01 | 1.41 / 0.99 | |
| wilhelm | FRB 20221203A | sharedζ | 2.56 ± 0.04 | 1.14 / 4.55 | **Caveat:** DSA peak-shape misfit |
| chromatica | FRB 20240203A | sharedζ | 3.28 ± 0.04 | 1.14 / 1.16 | |
| zach | FRB 20220207C | C1D1 | 3.32 ± 0.01 | 2.30 / 1.30 | C2D3 rejected (s² sign-unstable); **no** profile-bias claim |
| freya | FRB 20230325A | sharedζ | 4.36 ± 0.04 | 1.30 / 1.03 | Near-Kolmogorov |

### Tier B — provisional (3)

Gate + PBF pass on `_a1_fits/`; **all-exp fixed-s² grid not yet local** for the
quoted multi-component model. Cite only after `_s2verdict.py` adjudication (HPCC pull).

| Nickname | TNS | Model | α (all-exp) | χ² C/D | Blocker |
|----------|-----|-------|-------------|--------|---------|
| johndoeII | FRB 20230814B | C2D1 | 1.53 ± 0.09 | 1.04 / 1.27 | no `*_s2-*_pbf-exp-exp.json` |
| oran | FRB 20220506D | C2D1 | 2.66 ± 0.17 | 1.04 / 1.16 | no all-exp s² grid |
| phineas | FRB 20230307A | C3D3 | 3.32 ± 0.06 | 1.02 / 1.34 | no all-exp s² grid |

**Manuscript:** target **N = 8** tabulated α; **N = 5** fully locked today. Tier B
needs footnote "component count pending all-exp fixed-s²" until grids land.

## Multiplicity exemplar (prose, not extra table row)

| Nickname | TNS | Model | α (all-exp) | Role |
|----------|-----|-------|-------------|------|
| whitney | FRB 20220310F | C2D2 | 5.12 ± 0.17 | Second DSA component confirmed (fixed-s² stable); α 1.5→5.1 unrailing is the marquee multiplicity case |

Fit JSON: `local_runs/data/joint/whitney_fine_joint_fit_C2D2_pbf-exp-exp.json`.

## Excluded from quoted α

| Nickname | Reason |
|----------|--------|
| mahi | PBF-sensitive (Δα −0.63); wide posterior (σ ≈ 1.5) |
| isha | Wide posterior; α rails upper bound |
| hamilton | CHIME↔DSA component-correspondence ambiguity; per-band τ only |
| zach (C2D3) | Fixed-s² grid sign-unstable; multiplicity claim withheld |

## Consequences

- Faber2026 `tab:alpha`: **5 fully locked** now; **8 target** once Tier B s² grids adjudicate.
- Energies sample (#6) and campaign-count reconciliation (#4 deferred items) may
  proceed against this quality-passing set.
- Mixed-PBF `joint_json/` + `joint_gate_verdicts.md` are **superseded** for
  citation; all-exp `_a1_fits/` + this ADR are authoritative.
- Machine-readable roster:
  `analysis/scattering-refit-2026-06/citable_alpha_roster.json`.

## Evidence

```bash
cd analysis/scattering-refit-2026-06
conda run -n flits python grade_allexp.py _a1_fits
cd joint_ladder && conda run -n flits python _s2verdict.py zach  # C2D3 NOT robust
```
