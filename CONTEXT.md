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
Exponent in τ ∝ ν^(−α). Kolmogorov thin screen ≈ 4.0–4.4. Prior bounds typically 1.0–6.0. The operative joint-gate hard-FAIL floor is **1.5** (`gate_joint_committed.py:26`; `VALIDATION_THRESHOLDS.py`'s `ALPHA_MARGINAL_MIN=2.0` is a dead/unused constant), which [ADR-0004](docs/adr/0004-l1-sub-kolmogorov-alpha-floor.md) lowers to 1.0 (implementation deferred): 1.0 ≤ α < 2.0 becomes a flagged **sub-Kolmogorov** MARGINAL (physically admissible — multi-screen / anisotropic scattering), not a FAIL. A fit railed at *either* prior bound is **not a measurement** (weak/unresolved scattering), regardless of the value or how tight the posterior looks.
_Avoid_: bare "alpha" — see Flagged ambiguities.

**Scintillation bandwidth** (`dν`, Δν):
Decorrelation bandwidth of the diffractive scintillation pattern, from the ACF. Should *increase* with frequency (≈ ∝ ν⁴ for a thin screen). Measured per band.

**Modulation index** (m, m²):
Fractional intensity modulation of the scintillation; constrains how resolved the source is by the screen and bounds emission-region size (Nimmo et al. 2025 framework).

**Consistency relation**:
The thin-screen identity 2π·τ·Δν ≈ C₁ (C₁ ≈ 1, order-unity). Used here as a **screen counter**: satisfied at a band ⇒ one screen plausibly does both broadening and scintillation; violated ⇒ ≥2 screens / different origins.

**Screen**:
A turbulent scattering layer. Classified by location: **Galactic** (Milky Way ISM, predicted by NE2001/NE2025), **host-CGM** (the FRB host galaxy's circumgalactic/ISM medium), or **intervening** (a foreground galaxy crossing the sightline).

**Localization** (of a screen):
Assigning a screen to Galactic vs host-CGM vs intervening, from the frequency scaling of τ and Δν, the consistency relation, the NE2001 Galactic floor, and the DM budget — *not* sky-position localization of the FRB.

**Constraint ladder**:
The per-burst policy: always attempt the cheapest handle (consistency + ν-scaling); add effective-distance D_eff where a scintillation timescale is measurable; attempt a forward two-screen fit only where data quality supports it. Each burst is constrained as far as its data allows.

**DM budget decomposition**:
Partition of observed DM into MW ISM + MW halo + cosmic/IGM mean + intervening + host (residual). Produced by `galaxies/foreground`.

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
- The **consistency relation** applied per band counts **screens**; the frequency scaling of τ and Δν **localizes** each screen.
- **Localization** is cross-checked against the **DM budget decomposition** and the NE2001/NE2025 Galactic floor.
- The **constraint ladder** decides which handles each burst can support, gated by its **quality flags**.

## Example dialogue

> **Dev:** "Freya's Δν is 12.9 MHz at CHIME but 7.0 MHz at DSA — that's inverse, Δν should rise with frequency. Is the two-screen model just wrong here?"
> **Domain expert:** "Inverse scaling isn't a failure — it's a signal. Either the CHIME and DSA ACFs are sampling different screens, or the CHIME 'Δν' isn't diffractive scintillation at all. Don't call it a measurement until you've ruled out self-noise and RFI structure in that ACF."

## Flagged ambiguities

- **"α" / "alpha" is overloaded** — three distinct quantities: (1) **scattering index**, τ ∝ ν^(−α), ~4 Kolmogorov; (2) **scintillation-bandwidth scaling index**, Δν ∝ ν^(+α), ~4; (3) **intrinsic spectral index** of the burst flux. Always qualify which. Resolved convention: unqualified "α" = scattering index.
- **"Scintillation measured"** — only 3/12 bursts have ACF-derived Δν (casey, freya, wilhelm). The other 9 are **not-yet-attempted**, NOT "unsuitable". Do not conflate.
- **Freya τ**: legacy value 3.515 ms vs 2026-06 refit ~0.05–0.12 ms (≈30–50×). The refit supersedes; legacy values in older configs are stale.
