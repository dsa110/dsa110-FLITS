# Position the co-detection paper on two-band leverage, not population two-screen

**Status:** accepted

The co-detected CHIME/DSA FRB scattering paper is positioned around the **empirical measurement of the scattering index α (and screen localization) from the same burst seen simultaneously at ~0.6 and ~1.4 GHz** — the one thing prior population two-screen work could not do.

## Context

Population two-screen scattering+scintillation localization already exists: "Two-Screen Scattering in CRAFT FRBs" (arXiv 2305.11477, n=10 ASKAP) applies the Masui et al. 2015 (arXiv 1512.00529) + Ocker et al. 2022b two-screen model to constrain screen distances. The DM-budget-vs-scattering decomposition lineage (Cordes & Ocker, arXiv 2108.01172 and 2101.04784; halos/IGM contribute little scattering, host dominates) is mature. So "we do population two-screen localization + DM budget" is **not** a novel claim.

## Decision

Lead with the **two-band-same-burst leverage**: CRAFT and the Ocker lineage measure scattering in one band and must **assume α≈4** (Galactic scintillation supplies the second screen). CHIME/DSA co-detection measures the **frequency scaling of both τ and Δν directly** across a ~2.3× frequency ratio, pinning α per burst empirically. Secondary framing: first CHIME/DSA co-detected sample (no such scattering/scintillation study published) with per-burst tiered localization + host/CGM context.

## Consequences

- The introduction must explicitly differentiate from CRAFT (arXiv 2305.11477) and Cordes & Ocker — reviewers will raise them.
- The strength of the headline scales with how many bursts deliver a non-railed α; the validation campaign (re-measure anomalous ACFs, run scintillation on the 9 un-measured bursts, finish Monte-Carlo recovery) gates this. If α rails for most of the sample, the paper degrades gracefully to a first-sample census, but that is the fallback, not the pitch.
- The inverse-Δν / refractive anomalies are held for a possible second paper (CGM cloudlet lensing + Pradeep et al. 2025 RP framework), not the lead claim — see the constraint-ladder entry in CONTEXT.md.
