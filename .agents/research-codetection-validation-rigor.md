# Research: Rigor of CHIME–DSA co-detection validation

**Date:** 2026-06-23
**Scope:** both (internal codebase + external prior art)
**Codebase state:** `ab9d7f1` (2026-06-23)
**Related Documents:** `docs/codetection-science-plan.md` (decision #3 "Rigor before science"),
`docs/adr/0001-two-band-leverage-positioning.md`, `crossmatching/` (TOA cross-match),
`.agents/research-joint-fit-state.md`

## Question / Scope

Have we validated that the bursts detected at DSA-110 and CHIME are the **same astrophysical
event** (a genuine co-detection) in the most rigorous way possible — and if not, what does
"most rigorous" require?

In scope: the apparatus that establishes *coincidence* (same burst, two telescopes) — DM
agreement, time-of-arrival coincidence after dispersive + geometric correction, positional
coincidence, and the chance-coincidence/false-alarm probability. **Out of scope:** the
downstream science (scattering index α, screen localization, DM budget) and the TOA-extraction
provenance already verified separately (CHIME singlebeam re-extraction, this session).

## Codebase Findings

**The repo asserts co-detection on a single test: temporal consistency.** There is no
chance-coincidence probability, no independent DM-agreement test, and no positional-coincidence
test anywhere in the validation path.

- **TOA cross-match (the whole apparatus).** `crossmatching/toa_crossmatch.py:reproduce_notebook_result`
  computes `measured_offset_ms = TOA_CHIME(400 MHz) − TOA_DSA(400 MHz)` and a `geometric_delay_ms`
  (OVRO−DRAO baseline projected onto the source direction; `compute_geometric_delay`,
  `toa_crossmatch.py:128`). Both TOAs are referred to 400 MHz via the cold-plasma shift in
  `compute_toa` (`toa_crossmatch.py:99`, shift at `:121`, `K_DM = 4148.808`).
- **The decision statistic** is formed only in the plotting layer:
  `crossmatching/plotting.py:84` → `residual = measured_offset_ms − geometric_delay_ms`, with
  error `√(combined_dm_uncertainty_ms² + fwhm_ms²)` (`plotting.py:90`), then a `linregress(DM,
  residual)` slope probe for residual DM-dependence (`plotting.py:150`). A residual of 0 means the
  400 MHz offset equals the geometric baseline delay.
- **Quantified state of all 12** (from `crossmatching/toa_crossmatch_results.json`): every burst is
  within 3σ of zero residual (max **2.79σ**, wilhelm), so all pass. **But the test is weak:**
  - Error bars are dominated by an **assumed** `dm_uncertainty = 0.1 pc cm⁻³` propagated to
    400 MHz (the 1/400² lever): typical σ ≈ 2.4 ms, **24 ms (mahi), 74 ms (oran)**. Windows that
    wide cannot exclude a chance alignment — oran's constraint is effectively vacuous.
  - Residuals are **not centered on zero**: mean **+2.42 ms**, std 3.06 ms. A true common-origin
    signal after correct geometric + dispersive correction should scatter about 0; a +2.4 ms
    pedestal points to an unmodeled systematic (inter-site clock offset, DM/reference-frequency
    convention, or geometric-delay sign/site definition) currently absorbed by the wide error bars.
  - The DM uncertainty is a single hard-coded 0.1 for every burst (`fixture` and JSON), not the
    instruments' independently measured DM errors.
- **The error model is incomplete.** It includes only DM-uncertainty and pulse width; it omits the
  absolute-timing budget of each instrument (clock/GPS/maser, station-delay calibration), baseline
  position uncertainty, and intra-channel smearing — all of which enter a sub-ms TOA comparison.
- **No DM-agreement test.** The pipeline uses one shared DM per burst (the DSA/best-fit value); it
  never checks that CHIME's independently measured DM agrees with DSA's within their separate errors.
- **No positional-coincidence test.** The notebook (`crossmatching/toa_crossmatch.ipynb`, cell 2)
  computes DSA primary-beam offsets for display, but nothing tests that the DSA arcsec localization
  lies within CHIME's beam/baseband localization region.
- **The surface is explicitly provisional.** `docs/codetection-science-plan.md` lists
  `crossmatching/` as "**Stub / aspirational**", and decision #3 is "Rigor before science —
  targeted validation… defend anomalies vs artifacts before claiming them."
- **Provenance is solid (separately verified).** The CHIME 400 MHz TOAs were re-extracted from the
  singlebeam baseband in the CANFAR image and reproduce the notebook values bit-exactly for bright
  bursts (`/data/.../results/chime_singlebeam_toa_verification_report.md`; this session). So the
  *inputs* to the consistency test are trustworthy; it is the *coincidence inference* that is thin.

## Prior Art

Theme 1 — **The four-pillar standard for a multi-telescope FRB association.** Robust co-detection
claims rest on (i) DM concordance within combined instrument uncertainties; (ii) arrival times that
agree after dispersive correction to a common reference frequency **and** the inter-site geometric
delay; (iii) overlap of a high-precision (interferometric, arcsec) localization with the wide-field
instrument's error region; and (iv) an explicit chance-coincidence/false-alarm probability. The
canonical worked example is the FRB 121102 multi-telescope campaign — the first *simultaneous*
multi-instrument FRB detection — which leans on cross-instrument DM, timing, and localization
concordance (Law et al. 2017, arXiv:1705.07553). The community reporting standard requires full
dynamic spectra, per-beam data, complete timing/pointing metadata, and explicit exclusion of
terrestrial/RFI origins (Petroff, Houben et al. 2018, MNRAS 481, 2612, arXiv:1808.07809).

Theme 2 — **Chance-coincidence probability is the decisive, and for us missing, statistic.**
Occurrence of an unrelated burst in a given (time, DM, sky) window is modelled as a Poisson process;
the joint probability that two independent instruments each see a burst in their overlapping window,
given the all-sky/per-beam FRB rate, yields the false-alarm probability, with trial factors for
multiple DMs/beams/bands. A robust claim reports this number below a pre-set threshold (commonly
≪1%). [Standard Poisson coincidence formalism; methodology summarized in multi-messenger/multi-
telescope FRB searches — *secondary synthesis*, Perplexity research pass, 2026-06-23.] For two
independent wide-field instruments this is a clean, tractable calculation — exactly the pillar we omit.

Theme 3 (disconfirming search) — **Apparent coincidences have been wrong; the systematics that fake
a sub-ms offset are well catalogued.** Reported multi-facility coincidences have been reinterpreted
as chance alignments when temporal/spatial criteria were too loose or false-alarm rates were
underestimated. The sub-ms TOA comparison is vulnerable to: (a) **inter-observatory clock and
station-delay calibration** (GPS/maser absolute time, cable/electronics path delays) — unmodelled
offsets here mimic a common-origin residual; (b) **DM-constant convention** — 4.148808 vs Kulkarni's
4.149377 GHz² cm³ pc⁻¹ ms (~0.01% difference; Kulkarni 2020, arXiv:2007.02886, *widely-cited
secondary*), plus topocentric-vs-barycentric reference and dedispersion/reference-frequency choices;
(c) **intra-channel dispersive smearing**, which broadens the pulse differently per channelization
and biases the peak. Each is a documented way to manufacture an apparent offset. [Theme-3 synthesis:
Perplexity disconfirming pass, 2026-06-23, corroborated by Petroff et al. 2018 on terrestrial
exclusion and metadata.] **Relevance to us:** our non-zero **+2.4 ms** mean residual is precisely the
signature these systematics produce, and our error model includes none of them.

## Synthesis

**Answer: No — not in the most rigorous way possible.** We have exactly one of the four standard
pillars (temporal consistency), and we run it in its weakest form: error bars dominated by an
assumed flat 0.1 pc cm⁻³ DM uncertainty (σ up to 74 ms), a residual distribution with an unexplained
+2.4 ms pedestal, and an incomplete error budget. All 12 "pass," but the test is currently too blunt
to *exclude* chance — passing it is necessary, not sufficient.

What "most rigorous" adds, in priority order (gaps, not yet a design):

1. **Chance-coincidence probability per burst (highest leverage, lowest cost).** The genuinely
   decisive statistic and the one fully missing. For each DSA burst compute the Poisson probability
   that an unrelated CHIME FRB falls within the coincidence window: `P ≈ R_CHIME · Ω_overlap/Ω_sky ·
   Δt · f_DM`, using the CHIME all-sky FRB rate, the DSA-beam ∩ CHIME-FoV solid-angle overlap, the
   arrival-time window, and the DM-match fraction. For two independent wide-field instruments this is
   expected to be astronomically small per event — turning "consistent" into "chance-excluded." This
   is the single highest-value addition.
2. **Independent DM-agreement test.** Compare CHIME's measured DM to DSA's, each with its *own*
   uncertainty, instead of carrying one shared DM at a flat 0.1. DM concordance across independent
   pipelines is a near-orthogonal coincidence axis.
3. **Honest timing error budget + explain the +2.4 ms pedestal.** Add inter-site clock/GPS, station-
   delay, and baseline-position terms; pin the DM constant (we use 4.148808) and the dedispersion
   reference convention on both sides; quantify intra-channel smearing. Decide whether the +2.4 ms is
   a real clock/convention offset to remove or a genuine inconsistency — do not leave it hidden inside
   wide error bars.
4. **Positional-coincidence test.** Check the DSA arcsec localization against CHIME's beam/baseband
   localization region and fold the (small) chance-overlap probability into pillar 1.

Open questions for planning: is CHIME's per-event DM (with its error) available in the singlebeam
metadata or CHIME/FRB catalog for pillar 2? What absolute-timing accuracy do DSA-110 and CHIME
baseband each claim, for the pillar-3 budget? Is the relevant CHIME rate the all-sky rate or a
per-beam/per-exposure rate for the DSA pointing (pillar 1 normalization)? These are inputs to fetch,
not decisions to make here.

Light recommendation (defer detail to planning): implement pillar 1 first — it is the decisive,
low-cost statistic, reuses the existing `crossmatching/` TOA + geometric machinery, and converts the
current "all within 3σ" into a defensible, chance-excluded co-detection claim. It also reframes
pillars 2–4 as *tightening* an already-significant result rather than the sole evidence.

## References / Sources

- Code: `crossmatching/toa_crossmatch.py:99,128` (`compute_toa`, `compute_geometric_delay`),
  `crossmatching/toa_crossmatch.py:reproduce_notebook_result`; `crossmatching/plotting.py:84,90,150`
  (residual statistic, error model, DM-slope probe); `crossmatching/toa_crossmatch_results.json`
  (12-burst residual state); `crossmatching/toa_crossmatch.ipynb` cell 2 (beam offsets);
  `docs/codetection-science-plan.md` (§A "Stub / aspirational", decision #3);
  `docs/adr/0001-two-band-leverage-positioning.md`.
- External:
  - Law et al. 2017, "A Multi-telescope Campaign on FRB 121102", arXiv:1705.07553 —
    https://ar5iv.labs.arxiv.org/html/1705.07553
  - Petroff, Houben et al. 2018, "Verifying and Reporting Fast Radio Bursts", MNRAS 481, 2612,
    arXiv:1808.07809 — https://academic.oup.com/mnras/article/481/2/2612/5090173
  - Kulkarni 2020, "Dispersion measure: confusion, constants & clarity", arXiv:2007.02886
    (DM-constant convention 4.148808 vs 4.149377) — *widely-cited secondary*.
  - Chance-coincidence Poisson formalism and inter-observatory timing/DM-convention systematics —
    *secondary synthesis*, Perplexity research + disconfirming passes, 2026-06-23.
