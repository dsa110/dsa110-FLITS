# Research: sourcing independent CHIME DM + localization to activate association pillars 2 & 4

---
**Date:** 2026-06-23
**Codebase state:** `583eb03` (post pillars 1–4 merge, PR #20)
**Scope:** internal (repo wiring + baseband_analysis backend) **and** external (DM-phase method, CHIME baseband localization precision)
**Question:** Can we source a genuinely *independent* CHIME DM and a CHIME localization for each of the 12 co-detections from the local singlebeam voltage data, to move association pillars 2 (DM agreement) and 4 (positional coincidence) beyond their `null+reason` placeholders? With what method, and with what limitations?
---

## Decisions taken into this research (user, 2026-06-23)
- **Pillar 4** → **point cross-check**: compare the CHIME tied-beam RA/Dec to the DSA arcsec position; flag consistent within a *stated* CHIME localization radius (an assumption, like pillar-1's DM model). Document the no-error-ellipse caveat.
- **Pillar 2 DM method** → **structure-maximized only** (DM-phase), *not* S/N-maximized. Rationale: this sample is selected for scattering + hidden temporal sub-components; the S/N-optimal DM is biased by the scattering tail and can mis-align sub-bursts, whereas a structure metric targets phase coherence across components.

## Data on disk (h17, `/data/research/astrophysics/frbs/chime-dsa-codetections`)
- `chime_singlebeam/singlebeam_<event_id>.h5` ×12 (~1–1.6 GB each). These ARE the CHIME voltage data: `tiedbeam_baseband` complex64 `(nfreq, 2pol, ntime)`, `tiedbeam_power` float32, `time0`, `first_packet_recv_time`. `delta_time = 2.56e-6 s`. Freq native **descending** (top≈800, bottom≈400 MHz), `nfreq≈871`.
- `tiedbeam_locations` dataset carries `ra, dec, x_400MHz, y_400MHz, pol` — for `singlebeam_210456524` (zach): **ra=310.18066, dec=72.89757**. This differs from the DSA `source_coord` (zach `20h40m47.886s +72d52m56.378s` = 310.1995, +72.8823) by ~0.03° → it is a **CHIME-side** position, *not* the DSA one.
- `chimefrb/baseband-analysis:latest` docker image present (8.61 GB, 3 months old). Wrapper `bin/baseband_analysis_python.sh` runs `python <script>` in the image with `/data` mounted, cwd = codetections root. (Per the prior task: use the **docker image**, not the `baseband-analysis-canfar-src/` checkout, for runtime.)
- `scripts/burst_inputs.json` maps `chime_id → {name, dm, reference_frequency_mhz, fixture_toa_unix_400}` for all 12. `metadata/notebook_reproduction_fixture.json` adds per-burst `source_coord` (DSA), `dm_uncertainty` (0.1; this is the DSA/notebook value), `fwhm_ms`, and the CHIME/DSA TOA blocks.
- No multi-beam beamformed file and no localization-pipeline output on disk → **no CHIME error ellipse is available locally**.

## `tiedbeam_locations` provenance (the pillar-4 position)
- `baseband-analysis-canfar-src/baseband_analysis/analysis/beamform.py:639` — `loc = data.create_dataset("tiedbeam_locations", data=ib)`. The dataset is written from the **input beam pointing `ib`** handed to the beamformer, i.e. the RA/Dec at which the tied beam was *formed* — the CHIME-side localization used at beamforming time (baseband or intensity). It is **not** re-fit from the data, and the file carries **no uncertainty** for it.
- Consequence for pillar 4: we have an independent CHIME *point* (`tiedbeam_locations` ra/dec) vs the DSA arcsec point (`source_coord`). The angular separation is a real CHIME–DSA cross-check. The missing piece is the CHIME localization *uncertainty*, which lives in the (absent) multi-beam localization product → adopt a **stated CHIME localization radius** as a conservative assumption.

## Independent CHIME DM: method + backend
- baseband_analysis ships **S/N**-based DM optimization only: `analysis/dm.py:14 DM_SNR(power, freq, DM_c, DM_range, DM_step, ..., return_uncertainty=False)` and `:125 coherent_DM_SNR`. `DM_SNR`'s uncertainty (`:117`) is the ΔS/N=1 half-width of the S/N-vs-DM curve (`DM_range = x[y > y.max()-1]; DM_err = max-min`). **Not chosen** (user picked structure-max).
- **Structure-maximizing DM already exists in-repo**: `dispersion/dmphasev2.py:25 DMPhaseEstimator(waterfall, freqs, dt, dm_grid, ref=..., n_boot=200, random_state=None)` → `get_dm() -> (dm_best, dm_sigma)`. It is the **DM-phase** algorithm: FFT the (complex-cast) waterfall along time per channel (`:55`), apply the dispersion phase ramp `exp(-2πi f_axis · DM·delay)` over the trial grid (`:77 _phase_cube`), coherently sum across channels with MAD weights and take `|Σ|²·f²` (`:84 _coherent_power`), integrate over a fluctuation-frequency window (`:90 _window_mask`), and the trial DM maximizing that coherent power is the structure DM. σ_DM is a **channel-bootstrap** (`:109`) std of the per-resample quadratic-peak DM (`:124 _fit_peak_bootstrap`). Uses `flits.common.constants.K_DM`.
- **Mechanics decision (must be validated, not assumed):** DM-phase's `delay_sec` (`:57`) uses *absolute* `dm_grid` against `nu_ref`, so it expects a waterfall that is **not** already fully dedispersed, and finds the absolute structure-max DM. But CHIME's 0.39 MHz channels have non-negligible intra-channel smearing at 400–600 MHz that channelized DM-phase cannot undo. **Plan:** in docker, `coherent_dedisp(bb, DM_c, time_shift=False)` to remove intra-channel smearing at the catalogue `DM_c`, build the intensity waterfall `I(ν,t)=|X|²+|Y|²`, then run DM-phase over a **narrow grid centered on DM_c**; report `dm_best` (absolute) and `dm_sigma`. This is the same coherent+incoherent entry path already validated for TOAs (`scripts/extract_chime_singlebeam_toas.py:83-85`). The known-DM-recovery assertion (inject/round-trip, or recover DM_c on a coherently-dedispersed-then-re-dispersed copy) is the experiment gate before trusting the 12 numbers.

## Repo wiring points (where pillars 2 & 4 plug in)
- `crossmatching/association.py:68 dm_agreement(dm_chime, dm_chime_err, dm_dsa, dm_dsa_err, ...)` — already returns real `n_sigma`/`consistent` when `dm_chime` is non-null; currently fed `None` at `:156-161`.
- `crossmatching/association.py:119 position_consistent(dsa_coord, chime_center, radius_deg)` — already does the real astropy separation; currently bypassed (`position_consistent=None` at `:162`).
- `build_association_report` (`:130`) is the single assembler to extend: read a new **CHIME-side inputs** JSON (per-burst `dm_chime`, `dm_chime_err`, `chime_ra_deg`, `chime_dec_deg`) and feed pillars 2 & 4. Golden `toa_crossmatch_results.json` stays untouched (assembler is read-only on it — invariant tested at `tests/test_association.py:111`).
- `crossmatching/association_report.json` regenerates via `python -m crossmatching.association` (deterministic).

## Prior art
- **DM-phase / structure-maximizing DM:** the `DM_phase` tool (Seymour, Michilli, Lin; github.com/danielemichilli/DM_phase) and its use on complex/multi-component bursts — Hessels et al. 2019 (ApJL 876, L23, FRB 121102 sub-structure); Gajjar et al. 2018 (ApJ 863, 2). The structure filter yields a per-trial-DM structure measure *with* an uncertainty, letting the structure-max DM and its error be assigned (cf. arXiv:2302.06220 §; "recovers the correct structure parameters to within a few percent"). Preferred over S/N-max when scattering broadens the burst and creates multiple peaks — the S/N-max DM can lock onto one peak and misalign components.
- **CHIME baseband localization precision:** Michilli et al. 2021 (ApJ 910, 147, "An Analysis Pipeline for CHIME/FRB Full-array Baseband Data") — baseband localizations reach **sub-arcminute** precision (tens of arcsec for bright, well-calibrated events), with a systematic floor; real-time *intensity* localizations are coarser (≈arcmin+). Use this to justify the stated pillar-4 CHIME radius. (Retrieval gave the method/reference framing; exact per-event σ to be cited from Michilli 2021 directly if a hard number is needed in the writeup.)

## Synthesis
- **Pillar 2 is fully activatable** from local data: structure-max (DM-phase) CHIME DM + bootstrap σ per burst, via a docker extraction reusing the TOA coherent-dedisp entry path and the in-repo `DMPhaseEstimator`. Feeds `dm_agreement(dm_chime, dm_chime_err, dm_dsa=DSA_dm, dm_dsa_err)`.
- **Pillar 4 is partially activatable** as the chosen **point cross-check**: CHIME `tiedbeam_locations` (ra,dec) vs DSA `source_coord`, consistent within a stated CHIME radius (Michilli-2021-justified). No CHIME error ellipse locally → documented limitation; the radius is an explicit assumption surfaced in the report `inputs`.
- **Two gates before the 12 numbers are trusted:** (a) an *experiment* proving the DM-phase extraction recovers a known DM (round-trip) — the structure-DM mechanics decision above; (b) the **figure-review Stop gate** on every per-burst DM-phase diagnostic (DM–structure curve + dedispersed waterfall) — a numeric DM is not validated until the curve is looked at.

## Handoff
→ `ai-research-workflows:running-experiments` (validate DM-phase recovery on a known DM) → `planning-implementations` → `implementing-plans`. Adversarially verify (Workflow) as in the pillars-1–4 thread; golden untouched; PR → main.
