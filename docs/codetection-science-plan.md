# CHIME/DSA Co-detection Science Plan

Scoping the most-interesting science extractable from the 12 CHIME/DSA co-detected FRBs, with a tooling-state inventory and a prioritized to-do list. Vocabulary and decisions: see `CONTEXT.md` and `docs/adr/0001-two-band-leverage-positioning.md`.

## Resolved design decisions (grill outcomes)

| # | Decision | Choice |
|---|---|---|
| 1 | Headline science | Two-screen **screen localization** + tie to **host/CGM/intervening environment** |
| 2 | What "localize" means | **Constraint ladder**: consistency + ν-scaling always; add D_eff (geometric) and forward two-screen fit where data permits |
| 3 | Rigor before science | **Targeted validation** — defend anomalies vs artifacts before claiming them |
| 4 | Sample scope | **Tiered** GOLD/SILVER; report all 12 with per-burst constraint level; deep localization on GOLD |
| 5 | Positioning | **Two-band leverage** — empirical α from same-burst CHIME/DSA, not assumed α=4 (ADR-0001) |

## A. Tooling inventory & dev state

| Surface | Role | State |
|---|---|---|
| `scattering/scat_analysis/burstfit*.py` | Canonical kernel: M0–M3+mixed; emcee BIC, dynesty evidence, robust, **two-band joint fit** | Mature; `joint` untested + not in main flow |
| `scattering/pipeline/` | OO orchestrator, init-guess, MLE+DM refine | Mature |
| `scattering/.../priors_physical.py`, `dm_preprocessing.py` | NE2001/YMW16 physical priors; DM-phase preproc | Partial (soft deps) |
| `scintillation/scint_analysis/` | ACF, 4-model BIC, **2D global ν-scaling (α)**, noise | Mature core; `fitting_2d` untested |
| ↳ two-screen funcs (Nimmo 2025), `ne2025/`, `consistency.py` | Coherence constraint, τ·Δν consistency, modulation→size; Galactic floor | **Present but NOT wired** / brittle |
| `simulation/engine.py` + `sim_fit_bridge.py` | Two-screen forward sim, inject→fit roundtrip | Mature (smoke-tested, not quantitative recovery) |
| ↳ `wave_optics.py`, `multifreq_analysis.py` | Fresnel spike (Gpc-infeasible); broadband analysis | Incomplete |
| ↳ `monte_carlo.py` + `sim_fit_bridge` roundtrip | Recovery campaign; inject→fit | Runs, but **no quantitative τ-recovery validation** (smoke-test only) |
| `flits/batch/` | Multi-burst runner → SQLite, joint τ–Δν, export | Partial (scint config-gen + τ(ν) placeholders) |
| `flits/fitting/`, `orchestration/` | Validation thresholds; Maistro provenance | Mature |
| `galaxies/v2_0/` | Foreground search + **mNFW CGM DM+τ budget** | Mature — most complete science surface |
| `crossmatching/` | CHIME↔DSA TOA + geometric-delay co-detection; **association significance** (`association.py`, pillars 1–4) | TOA + chance-coincidence pillar implemented & tested (`association_report.json`); pillars 2/4 (independent CHIME DM / localization) wired but await CHIME-side data |
| `dispersion/dmphasev2.py` | DM-phase coherence estimator | Partial, standalone |

## B. The sample (ground truth)

12 co-detected FRBs (casey, chromatica, freya, hamilton, isha, johndoeii, mahi, oran, phineas, whitney, wilhelm, zach). CHIME ~0.6 GHz + DSA ~1.4 GHz on the same burst — the two-band lever arm is the defining asset. 234 GB data external (iacobus), none in-repo.

Status: 11/12 joint scattering fits; host search 9/12; DM budget 12/12; TOA cross-match 12/12 (ad-hoc, not via the stub). **Scintillation Δν measured for only 3/12** (casey, freya, wilhelm). 3 fits FAIL (casey, freya, wilhelm). 3 placeholder z=1.0 (freya, mahi, johndoeii).

Data-quality flags surfaced from the result files:
- **α rails at 6.0** (upper bound) for chromatica, freya, hamilton, mahi → scattering weak/unresolved → these become **upper limits**.
- **Inverse Δν scaling** (freya Δν_CHIME 12.9 > Δν_DSA 7.0 MHz; wilhelm 9.95 > 2.72) → either real multi-screen/refractive physics or one band's "Δν" is not diffractive — **must be adjudicated**.
- **Negative host DM** (zach, whitney, wilhelm, phineas) → **diagnosed (2026-06-22): not a code bug.** Budget subtracts the cosmic-DM **mean**; these are exactly the high-z, under-dense sightlines that lie below it (the `verdict_dm` strings already say "sightline below cosmic mean"). Whitney/phineas additionally have core-extrapolated CGM/intervening DM. Correct treatment is a **probabilistic** host DM (subtract `p(DM_cosmic|z)`, report host as posterior/upper limit), not a mechanical patch.
- **Freya τ**: legacy 3.515 ms vs refit ~0.05–0.12 ms (≈30–50×); refit supersedes.

## C. To-be-developed (software)

Ordered by leverage-per-effort:

1. **Wire the two-screen layer** (consistency relation + ν-scaling + modulation→size) into the pipeline. Funcs exist in `scintillation/scint_analysis/analysis.py`, just not called. *Low cost, unblocks the headline.*
2. **NE2025 Galactic-floor integration** — wire `ne2025/query_ne2025_scint.py` so each burst's measured scattering is compared to the predicted MW floor (Galactic vs extragalactic split). *Low cost.*
3. **Scintillation campaign tooling** — fix `flits/batch` scint config-gen stub (`batch_runner.py:262,275`, `# TODO: Add scintillation config generation`) so the mature scint pipeline runs over all 12.
4. **ACF anomaly re-validation harness** — RFI + self-noise + off-pulse checks for the 3 measured Δν, to certify diffractive vs artifact. *New, small.*
5. **Quantitative parameter-recovery validation** — DONE (2026-06-22). `simulation/recovery_campaign.py` ensemble-averages the sim→fit dynamic spectra (suppresses scintillation) and recovers injected τ to **~3% (ratio 1.03, spread 1.4%) across 80× in τ** (0.07–5.84 ms) → `results/recovery_campaign.csv`; slow test `tests/test_recovery_campaign.py`. Side finding: a **narrow-band double-count** in `sim_fit_bridge.roundtrip` — the `nu0^-α` rescale is invalid for a 3% fractional band (it inflated the ratio to ≈2.44=0.8⁻⁴); compare raw τ_1ghz. **Δν recovery added** (same module, `dnu_recovery_curve`): ensemble-averaged frequency ACFs recover injected `nu_s_host` **linearly** (ratio 0.29, spread 2.5%, over ~5× in Δν) → `results/recovery_campaign_dnu.csv`. Both τ and Δν recovery are now validated → defensible error bars and upper limits.
6. **Probabilistic host-DM treatment** — negative-host-DM is DIAGNOSED as expected Macquart-mean scatter on under-dense sightlines (not a `galaxies/v2_0` bug; see §B). Deliverable is to subtract the cosmic-DM *distribution* `p(DM_cosmic|z)` instead of the mean and report host DM as a posterior/upper limit. *Science task, needs a modelling decision — not autopilot.*
7. Debt (not science): `flits/` wrapper consolidation; τ(ν) batch placeholder (`analysis_logic.py:110`); `crossmatching/` geometric-delay localization remains unbuilt (association significance + TOA cross-match are done, §A row).

## D. To-be-explored (science)

| Direction | Handle | Prior art / novelty |
|---|---|---|
| **Empirical α per burst** from two-band τ scaling | scattering joint fit | **Novel angle** — CRAFT/Ocker assume α=4; we measure it (ADR-0001) |
| **Screen localization** (Galactic/host/intervening) | consistency + ν-scaling + NE2025 floor | Method exists (Masui 2015, Ocker 2022b, CRAFT 2305.11477); two-band application is the new part |
| **Population screen census** across 12 | tiered constraint ladder | DM-budget lineage mature (Cordes & Ocker 2108.01172) — differentiator is co-detection + two-band |
| **Inverse-Δν / refractive anomalies** | multi-screen / RP | CGM cloudlet lensing (Sammons 2309.07256), Pradeep 2025 RP — **held for paper 2** |
| **Emission-region size** from modulation index | Nimmo 2025 framework | Nimmo+2024 (Nature 2406.11053) — bonus for brightest bursts |
| **Intervening galaxy / MgII → excess scattering** | `mgii_inventory.csv` + budget | DSA multi-halo precedent (2405.14182) |

## E. Phased plan

**Phase 0 — Reconciliation (done here):** inventory, capability matrix, prior-art recon, positioning (ADR-0001).

**Phase 1 — Targeted validation (gate before science):**
- Re-measure the 3 anomalous ACFs (casey/freya/wilhelm) with RFI/self-noise/off-pulse checks → certify Δν diffractive or flag artifact.
- Run scintillation pipeline over the other 9 bursts (needs B.3).
- Finish Monte-Carlo recovery (B.5) → error bars + upper-limit framework.
- Wire NE2025 floor + consistency relation (B.1, B.2).
- **Exit gate:** every burst has τ (value or upper limit) + Δν (value or upper limit) + PASS/MARGINAL/FAIL, all through the runtime validation contract and figure-review gate.

**Phase 2 — Tiering + localization:**
- Assign GOLD (clean τ+Δν+z) vs SILVER per the matrix.
- Per-burst constraint ladder: consistency screen-count; ν-scaling Galactic-vs-extragalactic; D_eff and forward-fit where data bears it.
- Cross-check localization vs DM budget (B.6 first).

**Phase 3 — Population synthesis + write-up:**
- Empirical α distribution; screen-location census; host/CGM attribution.
- Differentiate explicitly from CRAFT 2305.11477 + Cordes & Ocker.
- Anomalies documented, deferred to paper 2 unless validation promotes them.

**Execution note:** Phases 1–2 are burst-parallel over 12 bursts → use a dynamic workflow paired with `.claude/workflows/fit-verify.js` (adversarial re-check of every `*_fit_results.json`), with a `/goal` completion condition = all 12 bursts at a defined constraint level, not a partial set.
