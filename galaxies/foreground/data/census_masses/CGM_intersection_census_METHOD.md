# CGM-intersection census (B7, option 2): method + result

**Goal:** select the foreground halo census by the physical criterion
`impact b ≤ R_vir` (sightline pierces the galaxy's own virial radius), using
real per-object masses — with a mass-quality vetting pass so the census is not
driven by contaminated photometry.

## Pipeline used (the project's own code, not reimplemented)

Per frozen-halo position: query photometry → stellar mass via the pipeline's
priority ladder → `estimate_halo_mass` (Moster+2013 SMHM) → `get_rvir_and_rs`
(Dutton–Macciò c–M) → `b/R_vir`, `intersects_rvir`.

Photometry queried live (Vizier + NOIRLab DataLab, both network-granted):
PS1 g,i (→ Taylor 2011 g−i mass) and WISE W1 (→ Cluver 2014).

## Coverage

All 34 frozen halos returned a measured mass (21 PS1/Taylor, 13 WISE/W1); none
fell to the assumed fallback. But the raw measured census is **mass-limited
where it matters**: 8 halos had logM* > 11.3 (cluster-scale for individual
galaxies), and those drove most of the "added" intersections via inflated R_vir.

## Vetting pass (the 8 suspect masses, logM* > 11.3)

Cross-checked each against a second estimator and physical plausibility
(`suspect_vetting_adjudicated.csv`). Rules:
- **reliable** — PS1-Taylor and WISE agree within 0.3 dex, OR a PS1 mass ≤ 11.6.
- **indeterminate (mass rejected)** — implausible WISE mass (> 11.6 dex) with no
  usable PS1 Taylor color to corroborate it. Two sub-cases among the 5 rejected:
  (i) **no PS1 detection at all** (2 rows) — zach and oran: the galaxy is at/below
  PS1 3π depth (→ faint → genuinely low mass), so a logM* ≈ 12–13 from a 30″ WISE
  cone is source blending / an unresolved neighbour, not the target. (ii) **PS1
  i-band only, g-band missing** (3 rows) — whitney (z=0.5551, both rows) and
  wilhelm: an i-detection exists (i≈20.85/20.88) but no g, so a full Taylor g−i
  mass cannot be formed and the WISE mass has no independent check. In both
  sub-cases the implausible WISE mass is uncorroborated, so the halo is excluded
  from the intersection verdict rather than admitted on a bad mass.

Outcome of the 8: **1 reliable** (chromatica FRB 20240203A — PS1 & WISE agree,
logM*≈11.46), **7 indeterminate** = 5 rejected (implausible WISE mass > 11.6 with
no usable PS1 Taylor color: 2 with no PS1 detection + 3 with i-band but no g-band)
+ 2 weak `wise_only` (whitney z=0.4712, casey — WISE mass ≤ 11.6, still no optical
corroboration). All 7 lack a defensible mass → treated as indeterminate.

NOTE: 7 of the 8 suspects were WISE-W1; the 8th (chromatica) was PS1-Taylor.
The cut is logM*>11.3, not "WISE only" — chromatica was vetted and, uniquely,
survived because its two independent estimators agree.

## Final adjudicated CGM-intersection census (n = 34)

| Outcome | n | meaning |
|---|---|---|
| agree | 11 | intersect R_vir AND in frozen budget |
| **added** | **7** | intersect R_vir, NOT in frozen budget (all on corroborated mass) |
| dropped | 3 | in frozen budget, do NOT intersect (all phineas FRB 20230307A, b/R_vir 1.05–1.73) |
| indeterminate | 7 | suspect mass rejected — no defensible R_vir |

Determinate halos: 27. The 7 added rest on `measured_ok` or `vetted_reliable`
masses (logM* 9.5–11.5, R_vir 107–964 kpc) — physically sensible galaxy halos,
not the earlier cluster-scale artifacts.

## Contrast with the earlier (flawed) numbers

- assumed-mass placeholder: 9 added / 8 dropped — an artifact of a flat logM*=10.
- raw measured (unvetted): 14 added / 3 dropped — 8 of the 14 on suspect masses.
- **vetted (this result): 7 added / 3 dropped, 7 indeterminate** — the defensible
  census.

## Manuscript implication

A CGM-intersection census is viable and now computed. It changes the halo
membership meaningfully vs. the frozen confirmation-gated census (7 in, 3 out),
but 7/34 halos cannot be placed without deeper imaging — that limitation must be
stated. For those 7, either (a) obtain deeper optical photometry to pin the mass,
or (b) report them as "mass-indeterminate, intersection unconstrained".

## Files
- `halo_rvir_ADJUDICATED.csv` — all 34 halos: measured mass, mass_status,
  adjudicated R_vir/b_over_rvir/intersects, vs frozen budget_eligible.
- `suspect_vetting_adjudicated.csv` — the 8 suspects with both estimators, WISE
  W1-W2 AGN flag (Stern+2012), verdict.
- `halo_rvir_MEASURED_diagnostic.csv` — raw measured (pre-vetting).
