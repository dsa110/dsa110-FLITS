# Burst energetics — flux-calibration & cross-band validity review

Research note (codebase investigation, 2026-06-22). Scope: does the absence of absolute
flux calibration / per-band normalization invalidate the E_iso table produced by
`analysis/calculate_burst_energies.py`? Reviewer: Claude (session ac2a9913), at user request.

## Verdict (TL;DR)

| Question | Answer | Confidence |
|---|---|---|
| Q1. Are `c0_C` and `c0_D` in commensurable units? | **No** — independent per-telescope arbitrary units | High (code-confirmed) |
| Q2. Is summing `E_CHIME + E_DSA` defensible? | **No** — adds two independently-scaled bands; corrupts even cross-burst *ranking* of the total | High |
| Q3. Is the `γ_D ≈ −5` rail a DSA bandpass-rolloff artifact? | **Likely** — the `c0/γ` fit path applies no bandpass flattening | Medium (plausible mechanism; needs a raw-spectrum check) |

**Bottom line:** the table cannot be published as absolute energies, and the CHIME+DSA *sum*
is not even a reliable relative-ordering statistic. Salvageable only as a single
calibrated-band, order-of-magnitude estimate (see Recommendations).

## Methodology

Traced the amplitude `c0` from the joint-fit output back through the model likelihood to the
data ingest, and inspected the stored posteriors. Files: `burstfit_joint.py`, `burstfit.py`,
`analysis/scattering-refit-2026-06/run_joint_fit.py`, `joint_json/*_joint_fit.json`.

## Evidence

1. **`c0` is fit in per-channel S/N units (corrected 2026-06-22).** Earlier this note said `c0`
   "inherits whatever units the `.npy` carries." Direct read of the loader corrects that: before the
   data reaches `FRBModel`, `BurstDataset._bandpass_correct` (`io.py:131-137`) **z-scores every
   channel** by its own off-pulse mean/std, and `io.py:143-146` downsamples without renormalizing,
   keeping "units as S/N." So the fitted `c0` is the per-channel **signal-to-noise** spectral
   amplitude, with the arbitrary instrumental gain divided out — not raw native units. This does
   **not** weaken the verdict: S/N is still telescope-specific (the S/N→Jy factor σ_S = SEFD/(√(n_pol·Δν·Δt)·G)
   differs per band per channel), so `c0_C` and `c0_D` remain incommensurable and unsummable. It
   sharpens the fix: multiply each band's S/N by its own σ_S(ν) (the radiometer step) — see
   `docs/rse/specs/plan-radiometer-flux-cal.md`.

2. **No flux/units metadata anywhere.** `*_joint_fit.json` keys are
   `[burst, alpha, tau_1ghz, log_evidence, log_evidence_err, alpha_bounds, percentiles, ncall]`
   — no `units`, `calibration`, or `Jy` field. Nothing asserts the `.npy` is in Jy·ms.

3. **`c0_C` and `c0_D` show no common scale** (medians; source:
   `analysis/scattering-refit-2026-06/joint_json/*_joint_fit.json` →
   `percentiles.c0_C/c0_D.median`, verified to match these values 2026-06-23):

   ```
   burst        c0_C    c0_D        burst        c0_C    c0_D
   chromatica  0.719   0.107        oran        0.827   0.082
   freya       0.743   0.294        phineas     0.342   0.361
   hamilton    0.124   0.049        whitney     0.077   0.253
   isha        1.451   0.027        wilhelm     0.127   0.763
   johndoeII   0.131   0.219        zach        1.929   0.185
   mahi        0.059   4.181
   ```
   Each band is ~O(0.01–4) in its own normalization, set by its own telescope's pipeline. CHIME
   (transit cylinder) and DSA-110 (separate array) share no flux standard — there is no
   cross-band tie anywhere in the config or loader.

4. **The pipeline's own design treats the per-channel amplitude/spectrum as a nuisance.** The
   gain-marginalized path (`burstfit_joint.py:95-97`, `burstfit.py:728-742`) integrates the
   per-channel gain out analytically — "the gain absorbs the burst spectrum AND scintillation…
   c0 and gamma drop out of the sampled vector." The energetics calc resurrects `c0/γ` from the
   *other* (sampled) path and treats them as physical flux densities. This is an internal
   contradiction: the pipeline marginalizes the spectrum away precisely because the per-channel
   amplitude is not trusted as an absolute physical quantity.

5. **No bandpass flattening in the `c0/γ` path.** A smooth power law is fit to the channel
   amplitudes as-is. Any uncorrected frequency-dependent gain (band-edge rolloff) is absorbed
   into `γ`. The DSA `γ_D` posteriors rail at the −5 prior floor for several bursts (Phineas
   median −4.998, posterior `[-4.9995, -4.9954]`, err ±0.001-0.003; Chromatica/Zach/Hamilton/Oran
   similar), and Whitney sits at the opposite extreme (`γ_D=+2.91`). A railed/extreme index that
   tracks the band edge is the signature of an uncorrected bandpass, not an astrophysical spectrum.

## Findings per question

**Q1 — commensurability.** `c0_C` and `c0_D` are amplitudes in independent, uncalibrated,
per-telescope units (Evidence 1–3). They are not comparable; their ratio carries no physical
meaning.

**Q2 — the sum.** `E_iso = E_CHIME + E_DSA` adds `c0_C·(CHIME scale)` to `c0_D·(DSA scale)` with
two *different* unknown scales. This is worse than "absolute scale unknown": because the
CHIME:DSA energy mix swings per burst (Isha 0.5% DSA → Wilhelm 74% DSA), the unknown per-band
factors reshuffle the bursts differently, so even the *relative ordering* of the total is not
preserved. The code's caveat ("a common unknown factor leaves ordering intact") assumes ONE
factor; there are two.

**Q3 — the rail.** The `c0/γ` path does not flatten the bandpass, so an uncorrected DSA rolloff
would be absorbed into `γ_D` and drive it to the prior floor — a plausible, possibly dominant,
explanation for the −5 rail. Confirm by overplotting a raw DSA channel-amplitude spectrum against
the fitted power law (or by comparing to the gain-marginal `gain_spectrum`, which is the
calibration-robust per-channel estimate).

## Recommendations (prioritized)

1. **Do not sum the two bands.** Drop `E_iso = E_CHIME + E_DSA` from the table.
2. **Anchor one band and report it alone.** CHIME publishes catalog fluences (Jy·ms, with
   realistic beam-model uncertainties). Tie each CHIME waterfall to its catalog fluence, then
   report **CHIME-band `E_iso`** as an order-of-magnitude value with explicit uncertainty.
3. **Fix the bandpass before trusting `γ`.** Flatten the per-channel gain (or use the
   gain-marginal `gain_spectrum`) and re-fit; the −5 rail may relax. Until then, mark railed
   `γ_D` as prior-limited, not measured.
4. **If neither band is calibratable:** do not report absolute energies. Report only
   calibration-robust quantities (τ, Δν_d, spectral *shape* with caveats) or fluence *ratios*
   within a single band.
5. **Add posterior-propagated error bars** to whatever energy is reported; for railed parameters
   report a one-sided limit.

## Unblock progress (2026-06-22, beam/SEFD hunt)

**DSA beam — found, real cube.** `DSA110_beam_1.h5` (measured Jones E-field, X/Y feeds,
41 freq × 1801 θ × 73 φ; freq axis 1.2–1.6 GHz despite a wrong `freq_Hz`/`MHz` label). Loader +
trilinear `G(θ,φ,ν)` in `analysis/dsa_beam.py` (boresight=1; gain(1.8°,1.4 GHz)=0.477 = half power,
matching the 3.6° phi-avg FWHM). Replaces the analytic Airy fallback in
`dsa110-continuum/.../beam_model.py`. Path seam: `telescopes.yaml beam_model_h5`.

**DSA SEFD — measured per-epoch, not a constant.** Canonical source is the **dsa110-rt** SEFD
dashboard (`github.com/dsa110/dsa110-rt`, served on `lxd110h23:5777`): per-day × per-calibrator ×
baseline-bin (`sefd_0-200m`…`sefd_800-2500m`), median ~few-thousand Jy (warn >5000). Consistent
with the continuum repo's measured `5800 Jy/element` (T_sys=25 K, N_ant=96, η=0.7); the sim
config's "300–500 Jy" is a stale placeholder. **Action:** pull the measured median SEFD nearest
each burst's epoch from the h23 dashboard.

**CHIME — NOT calibrated upstream (verified 2026-06-22).** Direct inspection of a real
`singlebeam_*.h5` via the canonical `baseband_analysis` reader (h17, see
`docs/rse/specs/research-chime-singlebeam-flux-units.md`) **disproves** the earlier assumption
that the singlebeam product is already in Jy. The singlebeam is a `BBData` object
(`tiedbeam_baseband`/`tiedbeam_power`) with **no `units` attribute** and only a complex per-input
*timing/beamforming* gain applied (`calibrator = gain_…taua_ref_cyga_timing.h5`; the whole effect
is `core/calibration.py:66` `baseband *= conj(gain)`, coherent-summation phasing, no SEFD, no
primary-beam, no Jy). `tiedbeam_power` median ≈ 6.5×10⁷ — arbitrary correlator units. The
singlebeam pipeline (`pipelines/form_singlebeam.py`) has **no flux step**. So CHIME needs the
*same* radiometer machinery as DSA (Andersen+2023 / Michilli+2021 baseband-at-known-position:
primary-beam model at the burst position + system SEFD), run as a separate downstream step.
`flux_jy_per_unit_C` stays `null`.

## Remaining to verify

- Per-burst DSA **pointing offsets** (θ,φ from beam centre) to evaluate `G` per sightline.
- ~~CHIME `singlebeam_*.h5` flux units (confirm Jy)~~ — **done**: not Jy, pre-flux-cal
  (`docs/rse/specs/research-chime-singlebeam-flux-units.md`). Now need the CHIME primary-beam model
  + per-event SEFD to *apply* the flux cal.
- ~~Whether the FLITS input `.npy` preserved the power scale or renormalized to S/N~~ — **resolved
  2026-06-22**: `BurstDataset` z-scores each channel to per-channel **S/N** (`io.py:131-146`), for
  both bands. So the eventual calibration is S/N × σ_S(ν) (radiometer), per
  `docs/rse/specs/plan-radiometer-flux-cal.md`.
- Direct bandpass check: raw DSA spectrum vs fitted `γ_D` for Phineas (the worst 30%-DSA case).

## Prior-art recon — how the field calibrates these bursts (verified refs)

Sweep 2026-06-22 (perplexity scholar + ADS/SciX; every reference below resolved via the ADS API,
BibTeX in `references.bib`). The path to a quotable energy is standard radio practice, not a
backend dig.

**The one equation, both bands.** Signal-to-noise → flux density via the radiometer equation:

```
S_nu = (S/N) * SEFD / [ sqrt(n_pol * d_nu * d_t) * G(theta,phi,nu) ]
SEFD = 2 k_B T_sys / A_eff          fluence F = integral S_nu dt  [Jy*ms]
```

The only missing ingredients are each telescope's **SEFD** and the **beam response `G`** at the
burst's sky position. FLITS already whitens by off-pulse MAD, so the data is effectively in S/N
units — it is one SEFD multiply (× beam correction) away from Jy.

**CHIME (400–800 MHz) — templates:**
- Andersen+ 2023 (AJ 166, 138), *Flux Calibration of CHIME/FRB Intensity Data* — the dedicated
  method paper (radiometer + primary-beam model; the equation above).
- CHIME/FRB Collab. 2024 (ApJ 969, 145), *Updating the First CHIME/FRB Catalog with Baseband
  Data* — recomputes Catalog-1 fluences from **lower limits** to true values once the baseband
  position fixes the beam gain. Catalog-1 numbers are lower limits precisely because position
  uncertainty leaves `G` unknown.
- Michilli+ 2021 (ApJ 910, 147), baseband pipeline — flux cal at a **known** position against
  steady calibrators (Crab, Cas A). Co-detections have good positions, so this regime (not the
  real-time intensity one) applies.

**DSA-110 (1.3–1.5 GHz) — templates:**
- Law+ 2024 (ApJ 967, 29), *DSA Science: First FRB and Host Galaxy Catalog* — DSA's catalog/
  flux-cal paper; SEFD, calibrators (3C286/3C48) + noise-diode scale, single-burst fluence.
  **Gap:** the standalone DSA-110 *system* paper (Ravi et al.) is still unpublished, so the
  SEFD/beam description lives in Law+2024 (+ Connor+ 2023, ApJ 949, L26) and the DSA-110 thesis —
  cite those, not an instrument paper.

**Energetics — the field does NOT sum raw bands.** Standard band-limited isotropic energy
(Zhang 2018, ApJ 867, L21):

```
E_iso = 4*pi*D_L(z)^2 / (1+z) * F_nu * d_nu          (flat-spectrum form; the 1/(1+z) is the
                                                       bandwidth k-correction)
```

For a power law F ∝ nu^beta the k-correction generalizes to
`K = integral_rest nu^beta d_nu / integral_obs nu^beta d_nu` with nu_rest = (1+z) nu_obs.
Best practice for two bands is a **joint spectral fit on a common absolute (Jy) scale**, then one
integrated `E_iso` — *not* two independently-scaled per-band energies added together. FLITS
already does a joint fit; it just needs calibrated inputs. (z-from-DM via Macquart+ 2020,
Nature 581, 391, is not needed here — these sightlines have spectroscopic z.)

**Two concrete errors this surfaces in the current `burst_energies.tex`:** (i) it applies **no
`(1+z)`** k-correction; (ii) it **sums** two uncalibrated bands (compounding the Q2 problem).

## Related (separate lane, noted not reviewed)

A concurrent agent reworked the foreground/DM-budget search (`scratch/codetection/why_missed.py`;
`DEFAULT_CLUSTER_IMPACT_KPC = 5000` split from the 100 kpc galaxy gate). Physics is sound
(clusters contribute DM at Mpc impact; galaxies need <100 kpc for scattering) and the user
validated it — out of scope for this energetics review.

## Absolute-scale audit (2026-06-23) — DSA coherent-beam SEFD + estimator caveat

After wiring the per-channel radiometer calibration (Phase 5), the first DSA fluences were
~100x too high (band-avg 218-6860 Jy*ms vs published tens of Jy*ms). Audited against Law et al.
2024 (DSA First FRB & Host Catalog, arXiv:2307.03344, Table 1), cross-matched by detection S/N:

| burst | catalog FRB | Heimdall S/N | catalog fluence | first code | ratio |
|---|---|---|---|---|---|
| zach    | FRB 20220207C | 60.0 | 16.2 Jy*ms | 1680 | 104x |
| whitney | FRB 20220310F | 68.4 | 26.2 Jy*ms | 2728 | 104x |
| oran    | FRB 20220506D | 48.9 | 13.2 Jy*ms |  570 |  43x |

**Two distinct errors found:**

1. **Wrong SEFD (dominant, ~N_ant).** `sigma_S` used the dsa110-rt dashboard SEFD (~8016 Jy),
   which `estimate_sefd.py` computes **per baseline** (`compute_sefd_per_baseline`) — i.e. the
   ~per-**element** SEFD. The `cntr_bpc` arrays are the **coherent detection beam** (Law+2024:
   **48 antennas** combined), whose SEFD = SEFD_element / N_ant. Fix:
   `flux_cal.load_dsa_sefd_beam = load_dsa_sefd / DSA_N_ANT` (N_ant=48 -> ~167 Jy). This collapses
   the ~100x. The empirical matched-filter anchor is ~234 Jy (single value reproduces all three
   catalog fluences exactly via `F=(S/N)*SEFD*sqrt(W)/sqrt(n_pol*dnu)`), implying N_eff~34
   (~70% beamforming efficiency vs the nominal 48).

2. **Estimator difference (residual ~1-3x, burst-dependent).** With the beam SEFD the fluences are
   physical (4.6-143 Jy*ms) but still differ from the catalog by a burst-dependent factor
   (oran 0.90x, whitney/zach ~2.2x). This is **not** a window/crop bug: oran and whitney are both
   compact (>=95% of flux within +-0.5 ms) yet differ 2.4x. The cause is that the per-channel
   **linear band integral** `int S dt dnu` (the true fluence) and the catalog **boxcar
   matched-filter** fluence are *different estimators*; for peaked/structured bursts they diverge
   by the burst structure factor. The shape/slope (gamma_D) result is scale-independent and
   unaffected; the absolute energy is not, so this estimator reconciliation is tracked before any
   E_iso ships. (TNS-suffix note: the local detection log lists ...207A/310A/506A; the catalog/TNS
   list ...207C/310F/506D — same bursts by S/N, local suffix likely wrong.)

## Estimator choice — physics justification (linear integral over boxcar matched-filter)

The residual ~1-3x (Section above) is a choice between two fluence estimators. The physics
selects the linear integral:

1. **Fluence is conserved under scattering.** Scattering convolves the intrinsic profile with a
   unit-area exponential kernel (timescale tau): it redistributes flux in time but conserves
   `int S dt`. The physically meaningful fluence for E_iso (total radiated energy) is therefore the
   integral over the FULL scattered profile, tail included.

2. **The boxcar matched-filter is biased low for scattered bursts.** The catalog fluence
   `F = (S/N) SEFD sqrt(W) / sqrt(n_pol dnu)` assumes a top-hat of the S/N-maximizing width W. For a
   scattered burst the S/N-optimal boxcar is narrower than the full scattered extent (adding tail
   past the e-folding grows noise faster than signal), so it misses tail flux. The bias goes as
   `W_box / W_eff < 1`, with `W_eff = int P dt / P_peak` the equivalent width.

3. **The data show this signature.** The linear-vs-catalog ratio tracks burst structure: compact
   oran (95% of flux within +-0.5 ms) agrees (0.90x); tailed/structured zach and whitney sit ~2.2x
   high — the `W_eff/W_box > 1` pattern of a catalog that under-counts the tail.

**Decision: the linear integral is the correct E_iso estimator** (conserved, tail-complete), made
robust by integrating the burst's significant extent (to the noise floor, off-pulse baseline
subtracted) rather than an arbitrary wide window (zach's far +-3 ms blob is noise/structure, not a
clean tail, and must be excluded). The robust realization is the **model-based fluence from the
calibrated scattering fit**: the fit gives the profile (intrinsic width (x) exp-tau tail) and the
calibrated amplitude c0(nu); integrating the MODEL includes the tail and excludes noise. This
unifies the estimator with the 5c-B re-fit — one calibrated fit yields both the relaxed gamma_D and
the tail-complete fluence. Report the catalog matched-filter value alongside, ratio = tail-fraction
diagnostic.

## Rigorous calibrated re-fit of gamma_D (2026-06-23) — `refit_calibrated.py`

The bandpass proxy (Section "bandpass diagnostic", `plot_bandpass_check.py`) only measured a log-log
slope and ignored noise weighting. This re-runs the **actual M2 MCMC** (c0, t0, gamma, tau_1ghz)
twice per railed burst with an identical setup — on the S/N data d(nu,t), and on the flux-calibrated
data d·sigma_S(nu) with per-channel noise sigma_S(nu) — so the beam-edge channels are down-weighted
self-consistently. The gamma prior floor is opened from the default -5 to **-10** so the calibration
shift is not clipped at the old rail. Figure `refit_calibrated.png` (figure-reviewed: match).

**Analytic backbone (oracle).** For a power-law spectrum the noise-weighted fit satisfies
`gamma_cal = gamma_SN + slope(sigma_S)` exactly: multiplying the data by sigma_S tilts the spectrum
and the matching 1/sigma_S noise weighting leaves the channel weights unchanged, so the recovered
index shifts by precisely the beam slope. `slope(sigma_S) > 0` off-axis (sigma_S rises where G falls
toward the band edges) ⇒ calibrating makes gamma LESS negative; on-axis (flat sigma_S) it does not
move.

**Results** (median, with the analytic prediction `slope_s` as cross-check):

| burst | G(1.4GHz) | gamma_SN | gamma_cal | d_gamma (MCMC) | slope_s (analytic) |
|---|---|---|---|---|---|
| chromatica | 0.25 | -9.91 | -9.90 | +0.01 | +2.15 |
| oran | 0.67 | -8.49 | -7.84 | +0.65 | +0.62 |
| phineas | 1.00 | -9.91 | -9.95 | -0.04 | +0.00 |
| zach | 0.68 | -7.50 | -7.05 | +0.45 | +0.57 |
| freya | 0.20 | -6.04 | -3.81 | +2.23 | +2.56 |

**Three findings:**

1. **The -5 rail is a real single-band DSA preference, not a joint-coupling artifact.** With the
   floor opened, gamma_SN plunges to -6…-9.9 for every burst — the DSA band alone pushes hard against
   whatever floor is set. The joint fit's gamma_D ≈ -5 (blue marks) was literally the old prior floor.

2. **The calibration mechanism is verified.** d_gamma(MCMC) matches slope(sigma_S) for the four
   bursts not pinned at the floor: oran 0.65 vs 0.62, zach 0.45 vs 0.57, freya 2.23 vs 2.56, phineas
   -0.04 vs 0.00. The relaxation is real and beam-correlated — largest for the lowest-G (most
   off-axis) bursts. freya (G=0.20) lifts -6.04 → **-3.81**, a physically plausible index.

3. **Beam vs astrophysics splits by on/off-axis.** On-axis **phineas** (G≈1.0, slope_s≈0) does NOT
   relax (-9.91 → -9.95): its steepness is astrophysical or a non-beam instrumental effect, NOT
   band-edge. **chromatica** (G=0.25) is so steep it pins at the -10 floor even calibrated — a
   beam-dominated near-total edge dropout (its +2.15 shift stays clipped).

**Implication for E_iso.** gamma_D as a free per-band index does not reliably recover an astrophysical
spectral index from the narrow (14%) DSA band — it is dominated by beam-edge sensitivity off-axis and
a residual steep falloff on-axis. So the energy estimate must NOT extrapolate the gamma_D power law;
the **calibrated per-channel fluence integral** (which integrates the measured channels directly) is
the robust estimator and is insensitive to the gamma rail. This reinforces the estimator decision
above.

## Scope of the absolute calibration (2026-06-23) — what the catalog actually validates

The independent validation pass (`docs/rse/specs/validation-radiometer-flux-cal.md`) found that the
absolute Jy scale is **empirically catalog-anchored for only a subset of the sample**, so the energy
table's calibration confidence is not uniform across sightlines. Scope precisely:

1. **Absolute-scale cross-check covers 3 sightlines, not all.** `test_dsa_fluence_matches_catalog_scale`
   compares the model-based DSA fluence to the Law+2024 catalog for **oran (0.99×), zach (1.27×), and
   whitney (2.15×)** — the only co-detections with a published catalog fluence in the repo. The other
   DSA-band fluences (isha, phineas, wilhelm, hamilton, chromatica) rely on the radiometer model
   (measured beam + coherent-beam SEFD) **without an independent catalog anchor**. The factor-of-3
   test band is a sanity gate, not a tight constraint. Treat the absolute scale as validated to ~2×
   for the 3 anchored bursts and model-trusted (no external check) for the rest.

2. **Untested high model fluences exist but do not enter E_iso.** Some bursts carry large model-based
   band-average fluences (mahi ≈ 468, freya ≈ 156 Jy·ms) with no catalog comparison. These are
   **excluded from the energy table** because they lack a well-constrained host redshift (the
   `z = 1.0000` placeholder flag → skipped, `calculate_burst_energies.py:156`), so they cannot bias
   the published energies — but they flag that the per-sightline scale is not uniformly verified.

3. **Redshift scope.** The E_iso table is restricted to sightlines with a real (non-placeholder) host
   redshift; sources whose redshift is unknown or poorly constrained (freya, mahi, johndoeii) are
   omitted. The per-source redshift quality (spectroscopic vs photometric) is inherited from the
   galaxy-search `TARGETS` list and is **not** independently flagged here — any sightline later found
   to have only a photometric/uncertain redshift should be demoted or annotated in the table.

**Net:** the energies are correct given the model, catalog-anchored to ~2× for oran/zach/whitney, and
band-restricted lower limits throughout. The manuscript and any energetics claim should state the
absolute scale is catalog-validated for the 3 anchored bursts rather than implying uniform validation.

## Follow-up audit (2026-06-23) — the 3-burst anchor is structural; redshift provenance recorded

A dedicated literature + codebase pass (`docs/rse/specs/research-energetics-followups.md`) resolved
the two open scope items above:

1. **The catalog cross-check cannot be widened — it is a structural limit, not an oversight.**
   Law+2024 (arXiv:2307.03344) Table 1 is the *only* published DSA-110 catalog tabulating per-burst
   fluence in Jy·ms, and it contains **11 FRBs, all from the Feb–Oct 2022 science-commissioning
   window**; zach/whitney/oran are the only FLITS co-detections in it. Every other co-detection
   (isha Nov 2022, wilhelm Dec 2022, phineas/freya 2023, hamilton Sep 2023, chromatica/mahi/casey
   2024) is outside that window. Later DSA-110 papers do **not** republish fluences — Sharma+2024
   (arXiv:2409.16964) tabulates host astrometry/redshift only (no fluence column); Sherman+2024
   (arXiv:2308.06813) is polarimetry on the same 2022 sample. So no published value exists to add.
   **Do not promote any model-based fluence into the `CATALOG` dict** — those are FLITS radiometer
   outputs, not catalog measurements; doing so would fabricate a "published" anchor.

2. **Redshift provenance: all 8 E_iso hosts are spectroscopic; none photometric.** Six have a
   published spectroscopic redshift — zach/whitney/oran (Sharma+2024 Gold, Keck/LRIS), isha
   (Sharma+2024 Gold, P200/DBSP), phineas (Sharma+2024 Gold, Keck/DEIMOS), wilhelm (Connor+2024
   arXiv:2409.16952, Keck/MOSFIRE). Two — **hamilton (FRB 20230913A) and chromatica (FRB 20240203A)**
   — post-date Sharma+2024 and have **no published host paper**; their z is repo-internal and
   provisional. This is now encoded as data (`Z_PROVENANCE` in `calculate_burst_energies.py`,
   surfaced as `row["z_src"]` in `burst_energies.json`) and flagged in the manuscript table, rather
   than living only in this prose. There is no photometric host z to demote.
