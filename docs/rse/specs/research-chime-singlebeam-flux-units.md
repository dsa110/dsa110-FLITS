# Research: CHIME singlebeam flux units — is the singlebeam product in Jy?

**Date:** 2026-06-22
**Scope:** internal (data-provenance research: the CHIME singlebeam HDF5 product, traced through the `baseband_analysis` source that produces it)
**Related Documents:** `analysis/burst_energies/CALIBRATION_REVIEW.md`; `scattering/configs/telescopes.yaml`; `analysis/calculate_burst_energies.py`
**Codebase state:** FLITS at `df23cce`; CHIME `baseband_analysis` CANFAR snapshot `git_sha 0ec3f4c3`, `git_version_tag 2021.11` (image `chimefrb/baseband-analysis:latest`, digest `sha256:f510909d…c4c41`, on `lxd110h17`).

## Question / Scope

Is the CHIME `singlebeam_*.h5` product (the input upstream of the FLITS CHIME
dynamic spectra) already absolutely flux-calibrated to Jy? If so, its flux scale
can be read off the file and used directly as `flux_jy_per_unit_C` in
`analysis/calculate_burst_energies.py`. If not, what calibration *has* been
applied, and what is still needed to reach Jy?

In scope: the HDF5 structure/attributes of a real singlebeam file, and the
`baseband_analysis` source path that creates it. Out of scope: re-deriving the
absolute flux scale (that is planning/implementation), and the DSA side (covered
in `CALIBRATION_REVIEW.md`).

## Codebase Findings

Inspected `singlebeam_210456524.h5` two ways: raw `h5py` on a local copy, and the
canonical `baseband_analysis.core.bbdata.BBData` reader inside the
`chimefrb/baseband-analysis:latest` container on h17 (via
`bin/baseband_analysis_python.sh`). Both agree.

**1. The product is a `BBData` object with no flux/units metadata.**
Root attr `__memh5_subclass = baseband_analysis.core.bbdata.BBData`. The two
payload datasets —

- `tiedbeam_baseband` `(871 freq, 2 beam, 55949 time)` `complex64`
- `tiedbeam_power` `(871, 2, 55949)` `float32` (= |baseband|²)

— carry **no `units` attribute** (`<NONE>` on both). The `BBData` object exposes
**no** flux/Jy/scale/SEFD/sensitivity member (`dir(bb)` filtered → `[]`). The
`beam` axis is length 2 = the two polarizations (`tiedbeam_locations['pol']`).
Frequency axis is 400.391–799.219 MHz, 871 channels.

**2. The only "calibration" applied is a complex per-input gain.** Both payload
datasets carry `calibrator = gain_20220208T054825.134277Z_taua_ref_cyga_timing.h5`,
and `tiedbeam_baseband` has `conjugate_beamform = 1`. That gain file is a **timing/
beamforming** solution from an N² point-source calibrator (Tau A reference, Cyg A).
It is applied in
`baseband_analysis/core/calibration.py:22` `apply_calibration`, whose entire effect
on the data is one line:

```
core/calibration.py:66   data.baseband[:] *= np.conj(gain_reordered[:, :, np.newaxis])
```

i.e. each correlator input's complex voltage is multiplied by the conjugate of its
complex gain `(n_freq, n_input)` so the array sums **coherently** when beamformed.
This fixes per-input phase and relative amplitude; it does **not** impose an
absolute Jy scale. The module's own `centroid_position` even carries a
`# TODO: In principle, take sensitivity weighted average` (`calibration.py:79`) —
sensitivity weighting is noted as not-yet-done, let alone an absolute flux tie.

**3. The singlebeam pipeline has no flux step.** `pipelines/form_singlebeam.py`
is the producer: it reads gains (`read_gains`), applies them
(`form_singlebeam.py:90` `apply_calibration(...)`), tied-array beamforms
(`:92` `beamform.tied_array(...)`), concatenates and saves (`:111`). There is **no**
SEFD multiply, **no** primary-beam division, and **no** Jy conversion between gain
application and save. A full-tree scan for `to_jy`/`jansky`/`flux_cal`/`radiometer`/
`sensitivity` finds nothing in `core`/`pipelines`/`analysis`; the only `flux`
symbols live in `dev/` (lensing *relative*-magnitude `convert_corr_to_relmag`,
VLBI source selection) and plotting helpers — all relative/instrumental, never an
absolute Jy conversion.

**4. The numbers confirm arbitrary units.** `tiedbeam_power` median ≈ 6.5×10⁷,
max ≈ 2.7×10¹⁰ (finite values) — correlator/baseband power units, not a
physically-scaled flux density. `|tiedbeam_baseband|` median ≈ 7.9×10³.

**5. Consequence for FLITS.** `scattering/configs/telescopes.yaml:30` already holds
`chime.flux_jy_per_unit: null`, and the energetics gate in
`analysis/calculate_burst_energies.py` refuses to emit energies while either band's
scale is null. This research confirms `null` is the **correct** value for CHIME: the
scale is not present in the singlebeam product and cannot be read off it.

**Update (2026-06-22): the `.npy` provenance gap is resolved.** FLITS's `BurstDataset`
z-scores every channel to per-channel **S/N** (`scattering/scat_analysis/pipeline/io.py:131-146`),
so whatever (arbitrary) scale the singlebeam-derived `.npy` carried is divided out before fitting —
the fitted `c0_C` is an S/N amplitude. Reaching Jy is therefore the radiometer multiply
S/N × σ_S(ν), with σ_S = SEFD/(√(n_pol·Δν·Δt)·G). See
`docs/rse/specs/plan-radiometer-flux-cal.md`.

## Synthesis

**The CHIME singlebeam product is pre-flux-calibration.** It is beamformed baseband
(and derived power) in arbitrary instrumental units, with only a complex per-input
*timing/beamforming* gain applied for coherent summation. No `units` attribute, no
SEFD, no primary-beam correction, no Jy anywhere in the singlebeam-forming package.

This **corrects** the tentative claim in `CALIBRATION_REVIEW.md` (the "CHIME —
already calibrated upstream … flux scale is read from the singlebeam h5" line):
CHIME is **not** already in Jy. Reaching a CHIME flux density requires the *same*
radiometer machinery as DSA — a separate downstream step (Andersen+2023 intensity
method; Michilli+2021 baseband-at-known-position), needing the **primary-beam model
at the burst position** and the **system sensitivity/SEFD**. That step is not in
this `baseband_analysis` image; it is a distinct CHIME/FRB calibration codebase.

So both bands are symmetric: each needs `S/N → S_ν` via
`S_ν = (S/N)·SEFD / [√(n_pol·Δν·Δt)·G(θ,φ,ν)]`, with a beam model and an SEFD. The
co-detections have good baseband positions, so the beam gain `G` is well-determined
(this is precisely why baseband fluences are true values, not the Catalog-1 lower
limits). `flux_jy_per_unit_C` stays `null` until that step is run.

**Open questions / gaps (deferred to planning):**
- FLITS CHIME `.npy` provenance: does the waterfall extraction from singlebeam
  preserve the beamformed-power scale, or renormalize (e.g. to off-pulse noise)?
  This sets the constant that multiplies the eventual SEFD.
- Which CHIME/FRB flux-cal code/products are reachable for these 12 events (the
  primary-beam model + per-event sensitivity), versus re-deriving from SEFD.
- CHIME SEFD / system-sensitivity value to use, and its frequency dependence across
  400–800 MHz.

## References / Sources

- Data: `singlebeam_210456524.h5` (event 210456524, 2022-02-07); 12 events at
  `h17:/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam/`.
- Code (CHIME `baseband_analysis` CANFAR snapshot, `git_sha 0ec3f4c3`, on h17 at
  `…/chime-dsa-codetections/baseband-analysis-canfar-src/baseband_analysis/`):
  - `core/calibration.py:22` `apply_calibration`; `:66` the gain multiply; `:79` the
    sensitivity-weighting TODO.
  - `pipelines/form_singlebeam.py:90` apply_calibration → `:92` `beamform.tied_array`
    → `:111` save (no flux step).
- Code (FLITS, `df23cce`): `scattering/configs/telescopes.yaml:30`
  (`chime.flux_jy_per_unit: null`); `analysis/calculate_burst_energies.py` (energetics
  gate); `analysis/burst_energies/CALIBRATION_REVIEW.md` (the corrected "upstream"
  claim).
- Reader/inspector: `bin/baseband_analysis_python.sh` (container wrapper, mounts host
  `/data:/data`); inspection script at `…/chime-dsa-codetections/inspect_bb.py`.
- External method refs (to apply, not yet applied): Andersen+2023 (AJ 166, 138);
  Michilli+2021 (ApJ 910, 147); CHIME/FRB Collab. 2024 (ApJ 969, 145). Full BibTeX in
  `analysis/burst_energies/references.bib`.

## Phase 6 resolution (2026-06-23) — beam + SEFD source chosen

The two open questions above (which flux-cal code is reachable; what CHIME SEFD/beam to use) are now
resolved. The full `ch_util`/CHIME beam-model package is **unreachable** from this analysis env: it is
CHIME-private (not pip-installable), `import ch_util` fails on h17's default env (container-gated),
and the local `baseband-analysis` clones carry only a partial `ch_util` (`catalogs/`, no beam
module). Per the plan's sanctioned fallback we use a **documented separable-Gaussian cylinder beam**
(`analysis/chime_beam.py`) anchored to **Amiri et al. 2018 (ApJ 863:48) Table 1** (fetched and
verified this session):
- E-W primary-beam FWHM 2.5 deg (400 MHz) -> 1.3 deg (800 MHz); N-S formed-beam FWHM 40' -> 20'; both
  ~1/nu. For a baseband-localized burst (formed at its own position at transit) G_CHIME ~ 1, so the
  beam adds little position dependence — opposite to DSA's fixed-pointing offset (up to ~2.6 deg).
- SEFD = 2 k_B Tsys / A_eff with Tsys=50 K and A_phys=8000 m^2 (Table 1), eta=0.5 -> **34.5 Jy**
  zenith (`analysis/burst_energies/chime_sefd.csv`), carrying a ~0.25 dex systematic that folds in
  eta, real Tsys, and the unmodeled element-envelope / declination / band-edge terms — matching
  CHIME/FRB Catalog 1 treating beam-model fluences as good only to a factor of a few.

The absolute CHIME scale is thus a documented ~0.25 dex approximation, not the full beam model —
adequate given the rigorous DSA refit (CALIBRATION_REVIEW.md) already established that the energetics
rest on the calibrated per-channel fluence integral, with a comparable systematic floor per band.
