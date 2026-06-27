# CHIME/DSA Co-detection Scattering Context

The science domain for FLITS as applied to the CHIME–DSA-110 co-detected FRB sample: measuring pulse-broadening (scattering) and scintillation in the *same* burst at two widely separated frequency bands, to localize the turbulent screen(s) along the line of sight and tie them to the host/CGM/intervening environment.

## Language

**Co-detected FRB**:
A fast radio burst seen by both CHIME (~0.6 GHz) and DSA-110 (~1.4 GHz). The sample is 12 such bursts (nicknames: casey, chromatica, freya, hamilton, isha, johndoeii, mahi, oran, phineas, whitney, wilhelm, zach).
_Avoid_: "joint burst", "dual detection"

**Two-band lever arm**:
The pair of well-separated observing frequencies (CHIME ~0.6 GHz, DSA ~1.4 GHz) for the same burst. The defining asset of this sample — it turns single-band degeneracies into measurable frequency scalings.

**Scattering time** (`tau_1ghz`, τ):
Pulse-broadening timescale referenced to 1 GHz, in ms. Fitted by the scattering kernel (`burstfit.py`, models M0–M3).

**Scattering index** (α):
Exponent in τ ∝ ν^(−α). Kolmogorov thin screen ≈ 4.0–4.4. Prior bounds typically 1.0–6.0. The operative joint-gate hard-FAIL floor is **1.0** (`gate_joint_committed.py:26`; [ADR-0004](docs/adr/0004-l1-sub-kolmogorov-alpha-floor.md)): 1.0 ≤ α < 2.0 is a flagged **sub-Kolmogorov** L3 MARGINAL (physically admissible — multi-screen / anisotropic scattering), not a FAIL. A fit railed at *either* prior bound is **not a measurement** (weak/unresolved scattering), regardless of the value or how tight the posterior looks.
_Avoid_: bare "alpha" — see Flagged ambiguities.

**Scintillation bandwidth** (`dν`, Δν):
Decorrelation bandwidth of the diffractive scintillation pattern, from the ACF. Should *increase* with frequency (≈ ∝ ν⁴ for a thin screen). Measured per band.

**Modulation index** (m, m²):
Fractional intensity modulation of the scintillation; constrains how resolved the source is by the screen and bounds emission-region size (Nimmo et al. 2025 framework).

**Consistency relation**:
The thin-screen identity 2π·τ·Δν ≈ C₁ (C₁ ≈ 1, order-unity). Used here as a **screen counter**: satisfied at a band ⇒ one screen plausibly does both broadening and scintillation; violated ⇒ ≥2 screens / different origins.

**PBF–α coupling**:
The pulse-broadening function *functional form* fixes the scattering index, not merely its prior range. A **single exponential thin-screen PBF** (square-law phase structure on the coherence scale, strong scattering with \(l_d \lesssim l_i\)) **implies α = 4** — fitting α free alongside an exponential PBF conflates functional-form assumptions with a spectral-index measurement. Mixed or power-law PBFs admit different α ranges; see [ADR-0003](docs/adr/0003-single-exponential-pbf.md) for the all-exp campaign choice.
_Avoid_: treating free-α under an exponential PBF as an independent Kolmogorov measurement without stating the PBF constraint.

**Consistency τ** (`tau_consistency`):
Scattering time used for per-band τ–Δν pairing and multi-screen triggers — from **α-fixed** single-exponential PBF refits (α = 4), with τ scaled to each band's reference frequency. Distinct from **joint τ** (`tau_joint`) / **free-α joint α** (`alpha_joint_free`) used for the citable-α roster ([ADR-0003](docs/adr/0003-single-exponential-pbf.md), locked roster [ADR-0005](docs/adr/0005-citable-alpha-roster.md)).
**Dual τ policy (accepted):** attribution matrix carries both tracks; scintillation consistency + multi-screen triggers use `tau_consistency` only; `tab:alpha` stays on free-α joint fits until a future ADR revises citable-α policy. Flag sightlines where free-α joint materially disagrees with α=4 consistency refit (`pbf_alpha_tension`).
_Avoid_: pairing joint free-α τ with per-band Δν for consistency, or collapsing both tracks without an ADR.

**Screen**:
A turbulent scattering layer. Classified by location: **Galactic** (Milky Way ISM, predicted by NE2001/NE2025), **host-CGM** (the FRB host galaxy's circumgalactic/ISM medium), or **intervening** (a foreground galaxy crossing the sightline).

**Localization** (of a screen):
Assigning a screen to Galactic vs host-CGM vs intervening, from the frequency scaling of τ and Δν, the consistency relation, the NE2001 Galactic floor, and the DM budget — *not* sky-position localization of the FRB.

**Constraint ladder**:
The per-burst policy: always attempt the cheapest handle (consistency + ν-scaling); add effective-distance D_eff where a scintillation timescale is measurable; attempt a forward two-screen fit only where data quality supports it. Each burst is constrained as far as its data allows.

**DM budget decomposition**:
Partition of observed DM into MW ISM + MW halo + cosmic/IGM mean + intervening + host (residual). Produced by `galaxies/foreground`.

**Published intervening census**:
The frozen 49-object catalog validated in `scratch/codetection/` (`foreground_final.csv` → `make_catalog_table.py` → `docs-analysis/foreground.md`, `foreground_table.tex`). Paper prose counts (29 confirmed · 7 refuted · 13 inconclusive) anchor here. Regeneration via `galaxies/foreground/search.py` must reconcile to this census, not silently replace it.
_Avoid_: treating live search output as the manuscript table without a diff.

**Hybrid foreground policy**:
Manuscript **table = frozen census** (all 49, all verdicts). **Pipeline τ/DM budgets** (`build_unified`, `sightline_budget`) run only on the **confirmed-foreground subset** from that census — not on refuted background objects, not on a fresh `search.py` candidate list unless reconciled.
_Avoid_: summing mNFW columns over all search hits or over inconclusive systems in budget claims.

**Two-tier foreground bookkeeping**:
1. **Registry tier** — every **confirmed** census object (29) gets a unified record + provenance row, including geometrically irrelevant systems (audit trail).
2. **Budget tier** — `DM_interv` / `τ_interv` sums only **budget-eligible** confirmed objects: halos inside the CGM gate (`b/R_vir` below the interior extrapolation threshold); clusters with `b/R_500 ≤ 1` (sightline pierces ICM). Others stay in registry with `budget_eligible = false`.
Refuted (7) and inconclusive (13) appear in the manuscript table only — excluded from both tiers' sums.
_Avoid_: summing all 29 confirmed regardless of `b/R_500` or impact parameter.

**Intervening census registry**:
Canonical machine-readable SSOT at `galaxies/foreground/data/intervening_census_registry.csv`. Stable object key `(nickname, type, obj)`; columns carry coords, best z + source, `final_verdict`, impact geometry, `budget_eligible`, provenance pointers. Built by `python -m galaxies.foreground.build_artifacts` from the validated `scratch/codetection/` chain; scratch CSVs/scripts are regenerators, not SSOT. `sightline_budget`, `build_unified`, MkDocs, and `foreground_table.tex` all consume this file.
_Avoid_: `sightline_budget` reading `{nickname}_galaxies.csv` without registry join, or `make_catalog_table` merging scratch CSVs ad hoc.

**Sightline attribution matrix**:
Per-burst machine-readable cross-check at `galaxies/foreground/data/sightline_attribution_matrix.csv` — one row per co-detected burst. Joins scattering measurements (`tau_obs`, per-band Δν), two-screen handles (consistency, ν-scaling, optional `D_eff`, Nimmo coherence), foreground registry counts/predictions, and an attribution verdict (Galactic / host / intervening / multi / undetermined). Gap flags use explicit `N/A — <reason>` tokens, not silent blanks.
_Avoid_: prose-only cross-checks that cannot be validated against the matrix.

**Multi-screen triggers** (two-screen analysis entry points):
1. **τ–Δν inconsistency** — per-band consistency relation `2πτΔν ≈ C₁` violated ⇒ that band plausibly samples ≥2 screens / mixed origins; not a fit failure.
2. **Multi-scale Δν** — wide+narrow decorrelation scales from `fit_two_screen_acf` ⇒ Nimmo two-screen coherence constraint eligible when host distance is known.
3. **Inverse Δν scaling** across bands — signal for multi-screen/refractive physics or a non-diffractive ACF artifact; must be adjudicated before citing as measurement.

Wiring target: feed `attach_scintillation_interpretation` with per-burst `tau_d_ms`, `distance_mpc`, and multi-scale Δν columns so consistency + coherence attach automatically; matrix records which triggers fired.
_Avoid_: treating inconsistency as a bug, or single-scale Δν alone as decisive two-screen localization.

**Quality flag**:
Per-fit PASS / MARGINAL / FAIL from the FLITS 3-level validation contract. FAIL fits are withheld from science claims.

**Burst designation**:
Two name systems for the same burst — internal **nickname** (`zach`, keys files/configs/results) and **TNS** designation (`FRB 20220207C`, the only identifier in the manuscript and published figures). Canonical map and SSOT: [ADR-0002](docs/adr/0002-canonical-burst-naming.md) (`burst_metadata.py::_FALLBACK_TNS` + `configs/bursts.yaml`).
_Avoid_: hand-maintaining a second map, or citing the gitignored `chimedsa_burst_specs.csv` as the registry.

**Citable α**:
A scattering index a fit is allowed to *quote* in the manuscript — distinct from a fit that merely ran. Requires: all-exp PBF ([ADR-0003](docs/adr/0003-single-exponential-pbf.md)), un-railed at both prior bounds, final component count confirmed, and a FINAL PASS or flagged MARGINAL verdict (not FAIL).
_Avoid_: quoting a railed or mixed-PBF α.

**PBF-insensitive** (α):
An α whose value barely moves (|Δα| ≤ 0.1) between the mixed and single-exponential PBF — true for the well-constrained single-screen sightlines, where the PBF choice is immaterial. The opposite, **PBF-confounded**, is an α that *moves or reverses sign* with the PBF (e.g. zach's multiplicity correction), making it unreliable to cite. See [ADR-0003](docs/adr/0003-single-exponential-pbf.md).

**Railed / unconstrained**:
A posterior whose median sits within ~3σ of *either* prior bound. The number is **not a measurement** regardless of how tight the posterior looks — railing, not the value, is the disqualifier (flagged rail-MARGINAL). See the scattering-index entry and [ADR-0004](docs/adr/0004-l1-sub-kolmogorov-alpha-floor.md).

**Well-constrained sightline**:
A burst whose joint fit yields a citable α — single dominant screen, un-railed, PBF-insensitive, quality-passing. The set is locked by the all-exp campaign + final component counts, not by figure color.

**Per-section sample rule**:
Every manuscript subset analysis (energies, joint-α, scintillation, …) states *its own* sample and justifies excluded bursts; the 12-burst co-detection set is the superset, not the per-analysis denominator. See `plan-manuscript-completion.md`'s exclusion table.

## Relationships

- A **Co-detected FRB** is observed across the **two-band lever arm**, yielding a **scattering time** and **scintillation bandwidth** at each band.
- The **consistency relation** applied per band counts **screens**; **consistency τ** (α-fixed exp PBF) pairs with each band's Δν — distinct from free-α **joint τ** in the citable-α roster.
- **Localization** is cross-checked against the **DM budget decomposition** and the NE2001/NE2025 Galactic floor.
- **Published intervening census** supplies the manuscript table; **hybrid foreground policy** limits pipeline budgets to the confirmed subset with **two-tier bookkeeping**.
- **Sightline attribution matrix** joins two-screen handles, foreground registry predictions, and per-burst verdicts; **multi-screen triggers** (τ–Δν inconsistency, multi-scale Δν, inverse scaling) gate deep two-screen analysis.
- The **constraint ladder** decides which handles each burst can support, gated by its **quality flags**.

## Example dialogue

> **Dev:** "Freya's Δν is 12.9 MHz at CHIME but 7.0 MHz at DSA — that's inverse, Δν should rise with frequency. Is the two-screen model just wrong here?"
> **Domain expert:** "Inverse scaling isn't a failure — it's a signal. Either the CHIME and DSA ACFs are sampling different screens, or the CHIME 'Δν' isn't diffractive scintillation at all. Don't call it a measurement until you've ruled out self-noise and RFI structure in that ACF."

## Flagged ambiguities

- **"α" / "alpha" is overloaded** — three distinct quantities: (1) **scattering index**, τ ∝ ν^(−α), ~4 Kolmogorov; (2) **scintillation-bandwidth scaling index**, Δν ∝ ν^(+α), ~4; (3) **intrinsic spectral index** of the burst flux. Always qualify which. Resolved convention: unqualified "α" = scattering index.
- **"Scintillation measured"** — only 3/12 bursts have ACF-derived Δν (casey, freya, wilhelm). The other 9 are **not-yet-attempted**, NOT "unsuitable". Do not conflate.
- **Freya τ**: legacy value 3.515 ms vs 2026-06 refit ~0.05–0.12 ms (≈30–50×). The refit supersedes; legacy values in older configs are stale.
