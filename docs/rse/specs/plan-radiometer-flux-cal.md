# Implementation Plan: Radiometer flux calibration (S/N → Jy) for FLITS burst energetics

---
**Date:** 2026-06-22
**Author:** AI Assistant
**Status:** Draft
**Related Documents:**
- [Research: CHIME singlebeam flux units](research-chime-singlebeam-flux-units.md)
- [Calibration review](../../../analysis/burst_energies/CALIBRATION_REVIEW.md)
**Codebase state:** FLITS at `df23cce`.

---

## Overview

The E_iso table in `analysis/calculate_burst_energies.py` is gated closed because neither
band carries an absolute flux scale. This plan implements the radiometer step that converts
FLITS dynamic-spectrum amplitudes to physical Jy, per channel, so the gate can open.

The enabling discovery (this session): `BurstDataset._bandpass_correct`
(`scattering/scat_analysis/pipeline/io.py:131-137`) **z-scores every channel** —
`(data − off_pulse_mean)/off_pulse_std` — and `io.py:145` keeps "units as S/N." So the data
the fitter sees is **already per-channel signal-to-noise**, with the arbitrary instrumental
gain divided out. The only thing standing between FLITS and Jy is the per-channel radiometer
noise σ_S(ν): `S_ν(ν,t) [Jy] = (S/N)(ν,t) · σ_S(ν)`, with
`σ_S(ν) = SEFD(ν) / (√(n_pol·Δν·Δt)·G(θ,φ,ν))`.

Because the z-score divides out the bandpass gain, what remains in the S/N spectrum is the
**sensitivity** shape `1/SEFD(ν)` — which is exactly what σ_S(ν) undoes. So per-channel
calibration is also the principled fix for the `γ_D ≈ −5` rail flagged in the calibration
review (it is the DSA sensitivity rolloff imprinted on S/N, not astrophysics).

**Goal:** A real, per-channel-calibrated DSA-band fluence in Jy·ms for every co-detected burst
with local data, feeding a `flux_jy`-calibrated band integral into the energetics script; then
the same machinery for CHIME, opening the E_iso gate for a publishable table with the (1+z)
k-correction and propagated uncertainties.

**Motivation:** Without this, the burst-energy table cannot be published (it is gated to a
"calibration pending" LaTeX stub). The radiometer step is standard radio practice (Andersen+2023
for CHIME; Law+2024 for DSA) and is now a single per-channel multiply on data we already have in
S/N units.

## Current State Analysis

**Existing Implementation:**
- `scattering/scat_analysis/pipeline/io.py:131-137` — `_bandpass_correct` z-scores each channel
  by its own off-pulse mean/std; `io.py:143-146` downsamples without renormalizing, keeping
  "units as S/N." This is the load path every fit uses (`io.py:60-101`).
- `scattering/scat_analysis/burstfit.py:656-704` — the forward model: `amp = c0·(ν/ν_ref)^γ`
  with `ref_freq = median(freq)`, multiplying a **time-sum-normalized** profile
  (`gauss/Σgauss`, `burstfit.py:696-704`), so `amp` is the per-channel **time-integrated S/N**.
- `scattering/scat_analysis/burstfit.py:619-635` — `_estimate_noise`: per-channel `1.4826·MAD`
  off-pulse noise (shape `n_freq`).
- `analysis/calculate_burst_energies.py:78-94` — `band_integral` integrates `c0·(ν/ν_ref)^γ` over
  the band; `band_energy_erg` multiplies by a **single scalar** `flux_scale` to reach erg.
- `analysis/calculate_burst_energies.py:117-165` — `compute()` gates on both bands having a
  non-null `flux_jy_per_unit`; emits energies only when calibrated.
- `analysis/dsa_beam.py:52-60` — `beam_gain(θ,φ,ν)` from the measured DSA Jones cube,
  boresight=1 (already wired).
- `configs/bursts.yaml` (symlink target `configs/bursts.yaml`) — per burst: `mjd`, `utc`,
  `ra_deg`, `dec_deg`, `chime_id` (epoch + position for SEFD/beam lookup).
- `scattering/configs/telescopes.yaml:16-31` — `df_MHz_raw`, `dt_ms_raw`, band edges,
  `flux_jy_per_unit: null`, `beam_model_h5: null` per band.

**Current Behavior:** `python analysis/calculate_burst_energies.py` reports band fluence integrals
in native units and writes the "pending calibration" LaTeX stub; no energy is emitted.

**Current Limitations:**
- `flux_jy_per_unit` is modeled as one scalar per band in config, but the true scale is
  **per-burst per-channel** (depends on that burst's epoch SEFD, sky-position beam gain, and the
  per-channel noise). The config seam cannot express it.
- The energetics integral uses the smooth fitted `c0/γ`, which (for DSA) is the
  sensitivity-weighted S/N spectrum — its `γ_D` is railed and not the sky spectrum.
- No SEFD, beam-position, or radiometer code exists in the energetics path.

## Desired End State

**New Behavior:** A new `analysis/flux_cal.py` computes, per burst per band, the calibrated band
fluence integral in Jy·ms·Hz by multiplying the per-channel S/N (`data/noise_std`, summed over the
on-pulse) by σ_S(ν). `calculate_burst_energies.py` consumes that integral directly (flux scale
already folded in), keeps its both-bands gate, and — once both bands are calibrated — emits the
E_iso table with the (1+z) k-correction and posterior-propagated error bars.

**Success Looks Like:**
- `pytest tests/test_flux_cal.py` passes: σ_S and the calibrated band integral match analytic
  oracles to <1e-6 relative.
- `python analysis/flux_cal.py --check` prints `self-check OK`.
- Running the energetics script with DSA calibrated and CHIME still null keeps the E_iso gate
  closed but reports a **real DSA-band fluence in Jy·ms** per burst (not native units).
- A committed `analysis/burst_energies/dsa_sefd.csv` carries the per-epoch DSA SEFD for the 12
  bursts, sourced from dsa110-rt, with provenance.
- A bandpass diagnostic figure shows the calibrated DSA spectrum vs the fitted `γ_D` power law,
  answering whether `γ_D` relaxes under calibration.
- After the CHIME epic: `python analysis/calculate_burst_energies.py` emits
  `burst_energies.tex` as an **energy table** (not the pending stub), energies in the
  10^38–10^41 erg FRB range, each with an error bar.

## What We're NOT Doing

- [ ] Re-running the scattering fits. τ/α/zeta science is unchanged; this only adds an amplitude
      calibration on top of existing fits.
- [ ] Absolute polarimetric or Stokes calibration — Stokes I fluence only.
- [ ] A new flux standard or cross-telescope tie beyond each band's own SEFD+beam (the bands stay
      independently calibrated, then summed only after both are in Jy).
- [ ] Modeling the SEFD frequency *shape* within a band as more than the dsa110-rt epoch value (a
      band-representative SEFD is used for the energy; the residual SEFD(ν) shape is treated as a
      documented systematic and probed by the diagnostic, not fit).
- [ ] Filling the 0.8–1.3 GHz inter-band gap — energies stay band-restricted lower limits, as the
      existing LaTeX already states (`calculate_burst_energies.py:261-266`).

**Rationale:** Keep the diff minimal and the science claims defensible; the energy is an
order-of-magnitude quantity with an explicit uncertainty budget, per the calibration review.

## Implementation Approach

**Technical Strategy:** Add one small pure-physics module (`analysis/flux_cal.py`) with the
radiometer kernel and the per-channel band-fluence integral, both backed by analytic self-checks.
Resolve per-burst inputs (epoch, position, SEFD, beam gain) from existing sources
(`bursts.yaml`, `dsa110-rt`, `dsa_beam.py`). Wire the calibrated integral into
`calculate_burst_energies.py` behind the existing gate so partial calibration never leaks an
energy. DSA first (beam wired, SEFD queryable); CHIME second (beam model + SEFD sourced in its
own epic). The bands remain independent until both are in Jy.

**Key Architectural Decisions:**
1. **Decision:** Calibrate the per-channel S/N data, not the fitted `c0/γ` spectrum.
   - **Rationale:** The data is already S/N; multiplying by σ_S(ν) recovers Jy and removes the
     sensitivity weighting that rails `γ_D`. Using the fitted smooth power law would bake the rail
     into the energy.
   - **Trade-offs:** Requires reloading each burst's `.npy` (external data on iacobus/h23) instead
     of reusing the small `joint_fit.json`.
   - **Alternatives considered:** Per-burst scalar on the fitted spectrum (rejected by the user;
     propagates the bandpass artifact and is band-flat).
2. **Decision:** The calibrated band integral is returned already in Jy·ms·Hz; the energetics
   script calls `band_energy_erg(I_jy, flux_scale=1.0, …)`.
   - **Rationale:** Reuses the existing, oracle-tested energy formula and (1+z) k-correction
     unchanged; the per-channel scale lives entirely in `flux_cal.py`.
   - **Trade-offs:** `compute()` branches between the native (uncalibrated) and Jy (calibrated)
     integral source.
   - **Alternatives considered:** Folding σ_S into a per-band scalar `flux_jy_per_unit` in config —
     cannot express per-channel/per-burst variation.
3. **Decision:** Beam gain `G` uses `dsa_beam.py` at the burst's real offset from the DSA pointing
   centre, computed from the **transit geometry**: θ ≈ |Dec_src − Dec_pointing| (HA≈0 at transit),
   φ along the meridian. Dec_src is the voltage localization (`bursts.yaml` `dec_deg`); Dec_pointing
   is the array pointing Dec (≈constant for a transit array), extracted from the filterbank headers /
   h23 localizations into a committed `dsa_pointing.csv`. `G=1` is used **only** if a specific
   burst's pointing Dec is genuinely missing, with a ≤0.30 dex systematic recorded for that burst.
   - **Rationale:** DSA-110 is a transit interferometer; at the burst's meridian transit the
     primary-beam offset is dominated by the declination difference, which we have (pointing Dec +
     source localization). The bursts cluster at Dec 70–74°, ~1–3° from a survey pointing Dec — a
     real, computable G variation near half power, not a negligible boresight case.
   - **Trade-offs:** Needs the per-burst pointing Dec (external: filterbank/h23). HA offset is a
     small documented correction (primary beam spans ±~6 min in HA).
   - **Alternatives considered:** Boresight G=1 for all (rejected — the Dec offsets are non-trivial);
     blocking on full per-instant pointing logs (the transit Dec is sufficient).

**Patterns to Follow:**
- Pure function + `assert`-based `_check()`/`--check`, as in `analysis/dsa_beam.py:63-83` and
  `analysis/calculate_burst_energies.py:287-318`.
- Injectable inputs for testing, as `compute(scales=…)` in
  `analysis/calculate_burst_energies.py:117`.

## Implementation Phases

Each phase is test-first: failing test → run (watch fail) → minimal code → run (watch pass) →
commit. Tests run in the `flits` conda env (agent-safe:
`env -i HOME="$HOME" PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin" /opt/anaconda3/bin/conda run -n flits pytest …`).

### Phase 1: Radiometer kernel (`analysis/flux_cal.py`)

**Objective:** Pure σ_S and calibrated-band-integral functions, validated against analytic cases.
No external data.

**Tasks:**
- [x] **Write the failing test** — `tests/test_flux_cal.py` (new):

  ```python
  import numpy as np
  from analysis.flux_cal import radiometer_sigma_jy, calibrated_band_integral_jy_ms_hz

  def test_sigma_jy_analytic():
      # SEFD=2000 Jy, n_pol=2, dnu=1e6 Hz, dt=1e-3 s, G=1 -> 2000/sqrt(2*1e6*1e-3)
      s = radiometer_sigma_jy(2000.0, n_pol=2, dnu_hz=1e6, dt_s=1e-3, g=1.0)
      assert abs(s - 2000.0 / np.sqrt(2000.0)) < 1e-9
      # beam attenuation G=0.5 doubles the noise
      assert abs(radiometer_sigma_jy(2000.0, 2, 1e6, 1e-3, 0.5) - 2.0 * s) < 1e-9

  def test_band_integral_flat_oracle():
      # flat S/N integral A per channel, flat sigma_S=s0, band [nu1,nu2]:
      # integral = A*s0*dt_ms*(nu2-nu1)
      nf = 64
      freq_hz = np.linspace(1.311e9, 1.499e9, nf)
      sn_integrated = np.full(nf, 3.0)            # per-channel sum_onpulse(S/N)
      sigma_jy = np.full(nf, 5.0)                 # per-channel sigma_S [Jy]
      dt_ms = 0.131072
      I = calibrated_band_integral_jy_ms_hz(sn_integrated, sigma_jy, freq_hz, dt_ms)
      oracle = 3.0 * 5.0 * dt_ms * (freq_hz[-1] - freq_hz[0])
      assert abs(I - oracle) / oracle < 1e-9
  ```

- [x] **Run it, watch it fail:** `pytest tests/test_flux_cal.py -v` → FAIL (module missing). ✓ `ModuleNotFoundError`.
- [x] **Implement** — `analysis/flux_cal.py` (new):

  ```python
  #!/usr/bin/env python
  """Radiometer flux calibration: per-channel S/N -> Jy for FLITS dynamic spectra.

  FLITS data is per-channel z-scored S/N (io.py:131-145), so physical flux density is
      S_nu(nu,t) [Jy] = (S/N)(nu,t) * sigma_S(nu),
      sigma_S(nu) = SEFD(nu) / (sqrt(n_pol*dnu*dt) * G(theta,phi,nu))   [Jy, radiometer].
  The band fluence integral (Jy*ms*Hz) feeds analysis/calculate_burst_energies.band_energy_erg
  with flux_scale=1 (the scale is already folded in here).
  """
  from __future__ import annotations
  import numpy as np

  def radiometer_sigma_jy(sefd_jy, n_pol, dnu_hz, dt_s, g):
      """Per-sample radiometer noise [Jy]. SEFD = 2 k_B T_sys / A_eff; G = beam gain (boresight=1)."""
      return sefd_jy / (np.sqrt(n_pol * dnu_hz * dt_s) * g)

  def calibrated_band_integral_jy_ms_hz(sn_integrated, sigma_jy, freq_hz, dt_ms):
      """int_band [ sigma_S(nu) * dt_ms * sum_onpulse(S/N)(nu) ] dnu   [Jy*ms*Hz].

      sn_integrated: per-channel sum over the on-pulse window of (data/noise_std)  [dimensionless]
      sigma_jy:      per-channel sigma_S [Jy]; freq_hz ascending; dt_ms sample width [ms].
      """
      chan_fluence_jy_ms = sigma_jy * dt_ms * sn_integrated     # [Jy*ms] per channel
      return float(np.trapezoid(chan_fluence_jy_ms, freq_hz))   # [Jy*ms*Hz]
  ```

- [x] **Run it, watch it pass:** `pytest tests/test_flux_cal.py -v` → PASS (2 passed).
- [x] **Add `--check`** — implemented as the bare `__main__` self-check (no flag needed; deviation
      from the `--check` flag in the plan). `python analysis/flux_cal.py` → `self-check OK`.
- [x] **Commit:** `725b34e` (pathspec `analysis/flux_cal.py tests/test_flux_cal.py pyproject.toml`;
      pyproject `pythonpath=["."]` added so `analysis/` imports in tests).

**Dependencies:** none.

**Verification:**
- [x] `pytest tests/test_flux_cal.py -v` → 2 passed. ✓
- [x] `python analysis/flux_cal.py` → `self-check OK`. ✓ (ruff clean.)

### Phase 2: Per-burst S/N spectrum from the `.npy` (data-driven)

**Objective:** Given a burst nickname + band, load its dynamic spectrum via `BurstDataset`, and
return `(freq_hz, sn_integrated, dt_ms, dnu_hz)` — the per-channel on-pulse S/N integral.

**Tasks:**
- [x] **Write the failing test** — `tests/test_flux_cal.py::test_sn_spectrum_synthetic` using a
      synthetic `.npy` (no external data): a `(64, 512)` array, unit-noise + a Gaussian pulse in
      the centre on every channel. *(Deviation: asserts finiteness + `median(sn_int)>5` + axis
      ordering rather than strict per-channel `>0` — a single channel's on-pulse sum can dip
      slightly negative under noise; the median is the robust check.)*

  ```python
  def test_sn_spectrum_synthetic(tmp_path):
      from analysis.flux_cal import sn_spectrum_from_npy
      rng = np.random.default_rng(0)
      nf, nt = 64, 512
      t = np.arange(nt); prof = np.exp(-0.5*((t-nt//2)/4.0)**2)
      data = rng.standard_normal((nf, nt)) + 8.0 * prof[None, :]   # S/N peak ~8
      p = tmp_path / "synth.npy"; np.save(p, data)
      freq_hz, sn_int, dt_ms, dnu_hz = sn_spectrum_from_npy(
          p, telescope="dsa", f_factor=1, t_factor=1)
      assert sn_int.shape == (nf,)
      assert np.all(sn_int > 0) and np.median(sn_int) > 5.0
  ```

- [x] **Run it, watch it fail:** `ImportError: cannot import name 'sn_spectrum_from_npy'`. ✓
- [x] **Implement `sn_spectrum_from_npy`** in `analysis/flux_cal.py` — wrap `BurstDataset`
      with `onpulse_crop=True`. *(Deviation: `io.py` uses package-relative imports (`from ..burstfit`),
      so the helper puts `REPO/scattering` on the path and imports `scat_analysis.pipeline.io` /
      `scat_analysis.config_utils` — NOT `scattering/scat_analysis` + `pipeline.io` as the plan
      sketched. `dnu_hz` returned as `ds.df_MHz*1e6` (≡ `df_MHz_raw*f_factor*1e6`).)*

  ```python
  def sn_spectrum_from_npy(inpath, telescope, f_factor=1, t_factor=1, onpulse_thresh=3.0):
      import sys, pathlib
      REPO = pathlib.Path(__file__).resolve().parents[1]
      sys.path.insert(0, str(REPO / "scattering" / "scat_analysis"))
      from pipeline.io import BurstDataset
      from config_utils import load_telescope_block
      tel = load_telescope_block(str(REPO / "scattering" / "configs" / "telescopes.yaml"), telescope)
      ds = BurstDataset(inpath, inpath, telescope=tel, f_factor=f_factor, t_factor=t_factor,
                        onpulse_crop=True, onpulse_thresh=onpulse_thresh)
      m = ds.model
      noise = np.clip(m.noise_std, 1e-9, None)            # per-channel (n_freq,)
      sn = m.data / noise[:, None]                        # (n_freq, n_time) S/N
      sn_integrated = np.nansum(sn, axis=1)               # sum over the cropped on-pulse window
      freq_hz = m.freq * 1e9                              # io.py freq is GHz, ascending
      return freq_hz, sn_integrated, ds.dt_ms, tel.df_MHz_raw * 1e6 * f_factor
  ```

- [x] **Run it, watch it pass.** 3 passed.
- [x] **Commit:** `8d28226`.

**Dependencies:** Phase 1. `BurstDataset` import path (`scattering` → `scat_analysis.pipeline.io`).

**Verification:**
- [x] `pytest tests/test_flux_cal.py::test_sn_spectrum_synthetic -v` → passed. ✓ (ruff clean.)
- [ ] On one real DSA `.npy` (if present locally): `sn_integrated` finite, positive median, length
      = number of channels.

### Phase 3: Per-burst inputs (epoch, position, beam gain) + DSA SEFD seam

**Objective:** Resolve, per burst per band, the σ_S(ν) inputs: epoch (for SEFD), sky position
(for beam offset → G(ν)), and the SEFD value. SEFD comes from a committed CSV (filled in Phase 4).

**Tasks:**
- [x] **Write the failing test** — `tests/test_flux_cal.py::test_dsa_sigma_jy_for_burst` with an
      injected SEFD and a stub beam (`g=1`): assert σ_S(ν) array has the band shape and equals the
      kernel applied channel-by-channel. *(Implemented as `test_dsa_sigma_jy_constant_at_boresight`
      + `test_dsa_beam_offset_is_dec_difference` + `test_burst_epoch_position_from_yaml`.)*

  ```python
  def test_dsa_sigma_jy_for_burst():
      from analysis.flux_cal import dsa_sigma_jy
      freq_hz = np.linspace(1.311e9, 1.499e9, 32)
      sig = dsa_sigma_jy(freq_hz, sefd_jy=4000.0, dt_s=1.31072e-4,
                         theta_deg=0.0, phi_deg=0.0, beam_gain_fn=lambda th,ph,f: 1.0)
      # G=1, n_pol=2 -> constant sigma across the band
      expect = 4000.0 / np.sqrt(2 * (freq_hz[1]-freq_hz[0]) * 1.31072e-4)  # dnu = channel spacing
      assert np.allclose(sig, sig[0])

  def test_dsa_beam_offset_is_dec_difference():
      from analysis.flux_cal import dsa_beam_offset
      # transit geometry: same-RA boresight -> separation is exactly the dec difference
      theta, phi = dsa_beam_offset(dec_src=70.31, dec_pointing=72.0)
      assert abs(theta - 1.69) < 1e-6 and phi == 0.0
  ```

- [x] **Run it, watch it fail.**
- [x] **Implement `dsa_sigma_jy`** and the burst-input resolvers in `analysis/flux_cal.py`:
  - `burst_epoch_position(nick)` → `(mjd, ra_deg, dec_deg)` from `configs/bursts.yaml`.
  - `dsa_pointing_dec(nick)` → the array pointing Dec from `analysis/burst_energies/dsa_pointing.csv`
    (`burst,pointing_dec_deg,source`), filled from the filterbank headers / h23 localizations
    (`codetections_manifest.yaml:75` → `h23:/media/ubuntu/ssd/jfaber/chime_dsa_codetections/localizations`).
  - `dsa_beam_offset(dec_src, dec_pointing)` → `(theta_deg, phi_deg)`: at transit (HA≈0) the
    boresight is at the source RA, so the separation is exactly the declination difference —
    `theta = abs(dec_src - dec_pointing)`, `phi = 0.0` (meridian; the near-boresight beam is ~azimuthally
    symmetric, dsa_beam phi-avg FWHM≈3.6°). **Only** if `dsa_pointing_dec` is missing for a burst,
    return `(0.0, 0.0)` and set `BEAM_FALLBACK[nick]=True` (adds the ≤0.30 dex term for that burst).
    *(HA refinement, optional: add the E-W offset (LST(utc)−RA_src)·cos(dec) when sub-beam accuracy
    is wanted; leading term is Δdec.)*
  - `dsa_sigma_jy(freq_hz, sefd_jy, dt_s, theta_deg, phi_deg, beam_gain_fn)` → per-channel σ_S via
    `radiometer_sigma_jy(sefd_jy, 2, dnu_hz=channel_spacing, dt_s, g=beam_gain_fn(theta,phi,f_ghz))`.
  - `load_dsa_sefd(nick)` → read `analysis/burst_energies/dsa_sefd.csv` (`burst,mjd,sefd_jy,source`),
    return the row's `sefd_jy`; raise if the burst is absent (gates the band uncalibrated).
- [x] **Run it, watch it pass.**
- [x] **Commit:** `git commit -m "feat(flux): per-burst DSA sigma_S inputs (epoch/position/beam/SEFD seam)" -- analysis/flux_cal.py tests/test_flux_cal.py` *(committed 67bbc9e)*

**Dependencies:** Phase 1; `astropy` (already a dep); `analysis/dsa_beam.py:beam_gain`;
`configs/bursts.yaml`.

**Verification:**
- [x] `pytest tests/test_flux_cal.py::test_dsa_sigma_jy_for_burst -v` → passed.
- [x] `dsa_beam_offset` on a burst returns θ within [0, 5] deg or the boresight fallback with the
      flag set. *(All 12 sample bursts: θ ∈ [0.1, 2.6]°, asserted in `test_dsa_pointing_csv_and_offsets`.)*

### Phase 4: Acquire per-epoch DSA SEFD + pointing Dec

**Objective:** Fill `analysis/burst_energies/dsa_sefd.csv` (measured DSA SEFD nearest each burst's
MJD, from the dsa110-rt store on h23) **and** `analysis/burst_energies/dsa_pointing.csv` (per-burst
array pointing Dec, from the filterbank headers / h23 localizations at
`codetections_manifest.yaml:75`).

**DEVIATIONS (executed):** split into **4a** (pointing) + **4b** (SEFD).
- *4a — pointing:* the filterbank headers did **not** carry the array pointing Dec; the user supplied
  the per-event primary-beam pointings from the DSA detection logs as a CSV
  (`dsa_primary_beam_pointings.csv`, "Dec ibeam"). Ingested by
  `analysis/burst_energies/build_dsa_pointing.py` → `dsa_pointing.csv` (committed `2bfd133`).
- *4b — SEFD:* **no contemporaneous SEFD exists** for the 2022–2024 bursts (dsa110-rt's store is a
  2026-02/03 campaign). Per user decision, used a single epoch-representative value: the **robust
  median** over the dashboard epochs (8016 Jy, ±27% from 1.4826·MAD; 129/151 clean epochs, rejecting
  >15000 Jy). Per-burst σ_S variation is carried by the beam gain G, not the SEFD. Committed `4b041e2`.

**Tasks:**
- [x] **Inspect the SEFD store format** (concrete, not an open question — known location):
      `ssh h23 'ls -la /media/ubuntu/ssd/vikram/sefd/ | head; head -2 <one result file>'`. *(Store is
      `state.json`; each epoch's `full_metrics.median_sefd` is the per-epoch median over baselines.)*
- [x] **Write the failing test** — `tests/test_flux_cal.py::test_dsa_sefd_csv_present`:

  ```python
  def test_dsa_sefd_csv_present():
      import csv, pathlib
      p = pathlib.Path("analysis/burst_energies/dsa_sefd.csv")
      assert p.exists(), "run Phase 4 acquisition"
      rows = list(csv.DictReader(p.open()))
      assert rows and {"burst","mjd","sefd_jy","source"} <= set(rows[0])
      for r in rows:
          assert 500.0 < float(r["sefd_jy"]) < 20000.0, r  # sane DSA array SEFD
  ```

- [x] **Run it, watch it fail** (CSV absent).
- [x] **Implement the acquisition script** `analysis/burst_energies/fetch_dsa_sefd.py`: SSH to h23,
      read `state.json`, take the **robust median** over clean epochs (see DEVIATION 4b — no nearest-MJD
      match is possible), write `dsa_sefd.csv` (`burst,mjd,sefd_jy,sefd_frac_err,source`); pointing Dec
      handled separately in 4a via `build_dsa_pointing.py`. (Read-only on h23; writes only local CSVs.)
- [x] **Run the acquisition**, then **run the test, watch it pass.**
- [x] **Sanity-check** the values against the continuum repo's measured array SEFD
      (`5800 Jy/element`, T_sys=25 K, N_ant=96 → array SEFD ≈ few thousand Jy). *(8016 Jy is ~1.4×
      the single-element 5800 Jy floor — physical for a degraded-subarray median; the ±27% scatter is
      recorded as `sefd_frac_err` for the energy error budget. Epoch-window flag N/A: representative,
      not nearest-MJD.)*
- [x] **Commit:** `git commit -m "data(flux): per-epoch DSA SEFD from dsa110-rt for the 12 bursts" -- analysis/burst_energies/dsa_sefd.csv analysis/burst_energies/fetch_dsa_sefd.py tests/test_flux_cal.py` *(committed `4b041e2`; pointing `2bfd133`)*

**Dependencies:** Phase 3; SSH to h23; `configs/bursts.yaml`.

**Verification:**
- [x] `pytest tests/test_flux_cal.py::test_dsa_sefd_csv_present -v` → passed.
- [x] CSV has a row per burst; all SEFD in [500, 20000] Jy; out-of-window epochs flagged. *(All 12
      rows = 8016.2 Jy; representative value, so no per-epoch window flag.)*

### Phase 5: Wire DSA into the energetics script + bandpass diagnostic

**Objective:** `calculate_burst_energies.py` computes the DSA band integral in Jy·ms·Hz from
`flux_cal`; the gate stays closed (CHIME null) but the DSA-band fluence is reported in Jy·ms. Plus
the calibrated-spectrum-vs-`γ_D` diagnostic.

**SPLIT (executed):** **5a** = the code wiring + monkeypatched gate test (no real data; committed
`51791c0`). **5b** = the real-`.npy` run + bandpass diagnostic figure (needs the 12 DSA `.npy` staged
under `data/dsa/` from `iacobus:burst_npys`, + figure-reviewer) — **pending data access**.

**Tasks:**
- [x] **Write the failing test** — `tests/test_burst_energies_fluxcal.py`: monkeypatch
      `flux_cal.dsa_band_fluence_jy_ms_hz` to a known stub and assert `compute()` (a) puts the DSA
      integral in Jy units on the row, (b) still emits **no** `E_iso_erg` while CHIME is null
      (gate intact), (c) the reported DSA fluence equals the stub.

  ```python
  def test_dsa_calibrated_but_gate_closed(monkeypatch):
      import analysis.calculate_burst_energies as E
      monkeypatch.setattr(E, "dsa_band_fluence_jy_ms_hz", lambda nick: 1.234e6)
      rows = E.compute(scales={"C": None, "D": "fluxcal"})   # D uses flux_cal, C uncalibrated
      assert rows and all("E_iso_erg" not in r for r in rows)        # gate closed
      assert any(abs(r.get("I_DSA_jy_ms_hz", 0) - 1.234e6) < 1 for r in rows)
  ```

- [x] **Run it, watch it fail.**
- [x] **Implement** the branch in `analysis/calculate_burst_energies.py:compute()`: when a band's
      scale is the sentinel `"fluxcal"`, set `I_<band>_jy_ms_hz = <band>_band_fluence_jy_ms_hz(nick)`
      (from `flux_cal`) and use `band_energy_erg(I_jy, 1.0, d_l_m, z)` for that band; keep the
      both-bands-or-nothing gate so a single calibrated band never emits a summed energy. Add
      `dsa_band_fluence_jy_ms_hz(nick)` to `flux_cal.py` composing Phases 2–3 (`sn_spectrum_from_npy`
      → `dsa_sigma_jy` → `calibrated_band_integral_jy_ms_hz`); `.npy`+binning from each burst's
      `configs/batch/dsa/<nick>_dsa.yaml`. *(Gate generalized: a band is calibrated if its scale is a
      float OR `"fluxcal"`; `_band_jy` resolves the sentinel.)*
- [x] **Run it, watch it pass.**
- [x] **(5b)** **Add the diagnostic** `analysis/burst_energies/plot_bandpass_check.py`: per burst,
      overplot the calibrated per-channel spectrum (`sigma_jy*sn_int`) vs the fitted
      `c0_D·(ν/ν_ref)^γ_D` power law + the S/N→cal log-log slope change; `figures.manifest.json`
      states the σ_S(ν)-flattening hypothesis. `figure-reviewer` wrote `figures.review.json` →
      **verdict `match`**. *(Staged 12 DSA `.npy` from CANFAR arc into `data/dsa/`; `dsa_beam` beam
      path moved to the stable repo `data/` location.)*
      **RESULT:** the γ_D≈−5 rail **flattens for off-axis bursts** (freya G=0.20 Δslope=2.56,
      chromatica G=0.25 Δ=2.06) but **persists on-axis** (phineas G=1.0 Δ=0.0) — a beam-edge
      sensitivity artifact for off-axis sightlines, **not uniformly instrumental**.
- [x] **(5a) Commit:** `51791c0` — `analysis/calculate_burst_energies.py analysis/flux_cal.py analysis/dsa_beam.py tests/test_burst_energies_fluxcal.py` (+ CALIBRATION_REVIEW.md). **(5b) Commit:** `dffbd51` — `plot_bandpass_check.py`, `dsa_beam.py`, `figures.{manifest,review}.json`, `dsa_fluences.csv`.

**Dependencies:** Phases 1–4.

**Verification:**
- [x] `pytest tests/test_burst_energies_fluxcal.py -v` → passed (3 tests; 11 with `test_flux_cal.py`).
- [x] `python analysis/calculate_burst_energies.py` → table still the **native/pending** form for
      E_iso (gate closed, N=8); LaTeX still the pending stub. *(The fluxcal Jy column appears only
      when D=`"fluxcal"` is passed; live `telescopes.yaml` is still uncalibrated, so the default run
      is unchanged.)*
- [x] `python analysis/calculate_burst_energies.py --check` → existing self-check still
      `self-check OK` (gate logic unbroken).
- [x] **(5b)** `figures.review.json` written by `figure-reviewer` → `match`. γ_D verdict: relaxes
      off-axis (σ_S(ν) flattening), persists on-axis.
- [x] **(5b, Manual — user)** User reviewed: absolute scale flagged for audit; γ_D verdict
      "not convinced" by the naive slope proxy → both folded into Phase 5c below.

### Phase 5c: Absolute-scale audit + rigorous γ_D (from user review of 5b)

**Objective:** put the DSA absolute Jy scale on the published catalog, and replace the naive
γ_D-slope proxy with a real re-fit.

- [x] **(5c-A) Absolute-scale audit + coherent-beam SEFD.** Commit `8ca87f7`. The per-channel
      fluences were ~100× high: `sigma_S` used the dsa110-rt **per-baseline/element** SEFD (~8016 Jy,
      `estimate_sefd.compute_sefd_per_baseline`) but the `cntr_bpc` arrays are the **coherent
      detection beam** (Law+2024, arXiv:2307.03344: **48 antennas**). Fixed with `DSA_N_ANT=48` +
      `load_dsa_sefd_beam = load_dsa_sefd/N_ant` (~167 Jy) → fluences now physical (4.6–143 Jy·ms),
      validated vs the catalog (zach 16.2 / whitney 26.2 / oran 13.2 Jy·ms, matched by Heimdall S/N
      60/68.4/48.9). New factor-3 `test_dsa_fluence_matches_catalog_scale` gate. Audit in
      `CALIBRATION_REVIEW.md`.
- [ ] **(5c-A residual) Estimator reconciliation.** A burst-dependent ~1–3× remains (oran 0.90×,
      whitney/zach ~2.2×) — **not** a window bug (oran/whitney both compact yet differ 2.4×): the
      per-channel **linear integral** `∫S dt dν` and the catalog **boxcar matched-filter** are
      different fluence estimators. Decide the canonical estimator (or report both) before E_iso.
- [ ] **(5c-B) Rigorous γ_D re-fit (user: "re-fit calibrated spectra").** Replace the naive
      log-log slope proxy: re-run the DSA scattering fit on the **flux-calibrated** spectra (channels
      scaled by σ_S(ν)) for the railed bursts (chromatica/oran/phineas/zach/freya) and compare the
      refit γ_D to the original to test whether the rail relaxes. **NEXT.**

### Phase 6 (CHIME epic): source the CHIME primary beam + SEFD, add `chime_beam.py`

**Objective:** Provide σ_S(ν) for CHIME so its band integral can be computed the same way.

**Tasks:**
- [x] **Recon the CHIME beam model** — `ch_util` unreachable (CHIME-private, not pip; `import
      ch_util` fails on h17's default env; local clones partial). Chose the documented cylinder-beam
      fallback anchored to Amiri+2018 (ApJ 863:48) Table 1; recorded in
      `research-chime-singlebeam-flux-units.md` ("Phase 6 resolution").
- [x] **Write the failing test** — `tests/test_chime_beam.py::test_chime_gain_boresight` (+ 4 more).
- [x] **Implement `analysis/chime_beam.py`** `beam_gain(ra,dec,freq_mhz, *, ra0,dec0)` (separable
      Gaussian, chromatic FWHMs from Table 1; error stated) + `chime_sigma_jy(...)` + SEFD derivation.
- [x] **Acquire the CHIME SEFD** → `analysis/burst_energies/chime_sefd.csv` (34.5 Jy zenith,
      2 k_B Tsys/A_eff from Table 1, ~0.25 dex systematic, full provenance row).
- [x] **Run tests, watch pass; commit.**

**Dependencies:** Phase 1; external CHIME beam/SEFD sources (h17 container, literature).

**Verification:**
- [x] `pytest tests/test_chime_beam.py -v` → passed (5/5).
- [x] `chime_sefd.csv` present; `beam_gain` boresight=1, falls off-axis (self-check OK; half-power at
      FWHM/2, chromatic).

### Phase 7: Open the E_iso gate — combined table + validation

**Objective:** Both bands calibrated → gate opens → E_iso table with (1+z) k-correction and
posterior-propagated error bars; validate against an independent fluence.

**Tasks:**
- [x] **Write the failing test** — `tests/test_burst_energies_fluxcal.py::test_both_bands_emit`:
      with both bands set to `"fluxcal"` (stubbed integrals), `compute()` emits `E_iso_erg` and the
      k-correction identity `E_iso_erg == E_iso_erg_no_kcorr/(1+z)` holds (mirrors the existing
      `_check` at `calculate_burst_energies.py:312`).
- [x] **Implement error propagation:** carry the joint-fit amplitude posterior width and the SEFD +
      beam systematic (`BAND_SYS_DEX` = 0.25 dex C, 0.20 dex D) into a per-burst energy uncertainty;
      added `E_iso_erg_err` to the row and a `±` column rendered in-cell by `_tex_val_err`
      in `latex_section` (plus a `+/- E_iso (erg)` column in `markdown_table`).
- [x] **Run the full script** with both `dsa_sefd.csv` and `chime_sefd.csv` present → emits
      `burst_energies.tex` as an energy table.
- [x] **Validate** each energy lands in 10^38–10^41 erg (wilhelm at 1.1×10^41 sits just above the
      nominal upper edge — consistent with it being the most distant/luminous sightline, z=0.51);
      model-based DSA fluence cross-checks Law+2024 within ~2× (oran 0.99×, zach 1.27×, whitney 2.16×),
      asserted by `test_joint_band_fluence_matches_catalog_scale`.
- [ ] **Commit:** `git commit -m "feat(energetics): open E_iso gate — calibrated table with k-corr + error bars"`

**Dependencies:** Phases 5 and 6.

**Verification:**
- [x] `pytest tests/test_burst_energies_fluxcal.py -v` → all passed.
- [x] `python analysis/calculate_burst_energies.py` → `burst_energies.tex` is the **energy table**
      (not the pending stub); every energy has an error bar and lies in 10^38–10^41 erg
      (wilhelm 1.1×10^41 marginally above the upper edge, see Validate note).

## Success Criteria

### Automated Verification
- [x] `pytest tests/test_flux_cal.py tests/test_burst_energies_fluxcal.py tests/test_chime_beam.py` passes.
- [x] `python analysis/flux_cal.py --check` → `self-check OK`.
- [x] `python analysis/calculate_burst_energies.py --check` → `self-check OK` (gate logic intact).
- [x] `ruff check analysis/flux_cal.py analysis/chime_beam.py` clean.
- [x] Files exist: `analysis/flux_cal.py`, `analysis/burst_energies/dsa_sefd.csv`,
      `analysis/burst_energies/dsa_pointing.csv`, `analysis/burst_energies/chime_sefd.csv`,
      `analysis/chime_beam.py`.
- [x] After Phase 7: `burst_energies.tex` does not contain "calibration pending".

### Manual Verification
- [ ] Bandpass diagnostic (`figures.review.json`): is the calibrated DSA spectrum flatter than the
      railed `γ_D` power law? (Confirms/refutes the sensitivity-artifact hypothesis.)
- [ ] DSA SEFD values are physically sane vs the 5800 Jy/element measurement and vary plausibly
      with epoch.
- [ ] Final energies sit in the expected FRB range (10^38–10^41 erg) and the one cross-checked
      burst agrees with its published fluence within ~2×.

### Reproducibility & Correctness (research code)
- [x] σ_S and the band integral are checked against analytic oracles (Phase 1) with <1e-6 tolerance.
- [x] SEFD/beam inputs are captured in committed CSVs with provenance columns; exact acquisition
      commands recorded in the fetch scripts.
- [x] The energetics run reproduces from the committed CSVs + `joint_json` + a local `.npy` set.

## Testing Strategy

**Unit Test Coverage (written in-phase):** the radiometer kernel and band integral (analytic
oracles), the S/N spectrum extraction (synthetic `.npy`), per-burst σ_S inputs (injected SEFD +
stub beam), the energetics gate (both directions), the k-correction identity, CHIME beam boresight.

**Integration Tests:**
- [ ] End-to-end DSA band: synthetic `.npy` + injected SEFD + real `dsa_beam` → finite Jy·ms
      fluence; gate stays closed with CHIME null.
- [ ] End-to-end both-band (stubbed integrals) → E_iso emitted with k-correction identity.

**Manual Testing:**
- [ ] Bandpass diagnostic figure review.
- [ ] Literature cross-check of one burst's fluence/energy.

**Test Data Requirements:** synthetic arrays generated in-test (no external data for unit tests);
one real DSA `.npy` (iacobus/h23) for the integration smoke; committed SEFD CSVs.

## Migration Strategy

**Migration Steps:** additive — new module + new compute branch behind the existing gate. The
native (uncalibrated) path is unchanged until SEFD CSVs exist.

**Rollback Plan:** revert the `compute()` branch; `flux_cal.py` and CSVs are inert without it.

**Backward Compatibility:** with no `dsa_sefd.csv`/`chime_sefd.csv`, the script behaves exactly as
today (pending stub).

## Risk Assessment

1. **Risk:** Per-burst DSA pointing Dec unavailable → G unknown.
   - **Likelihood:** Low (the pointing Dec is in the filterbank headers / h23 localizations and is
     ~constant for the transit array) · **Impact:** Medium
   - **Mitigation:** Extract Dec_pointing into `dsa_pointing.csv` (Phase 4); θ=|Dec_src−Dec_pointing|.
     Per-burst `G=1` fallback with ≤0.30 dex systematic only if a specific burst's pointing Dec is
     truly missing.
2. **Risk:** dsa110-rt SEFD only a band-scalar, not SEFD(ν).
   - **Likelihood:** High · **Impact:** Low (energy) / Medium (spectral index)
   - **Mitigation:** Band-representative SEFD for the energy; the bandpass diagnostic reveals any
     residual SEFD(ν) shape; documented as a systematic.
3. **Risk:** CHIME primary-beam model unreachable.
   - **Likelihood:** Medium · **Impact:** High (blocks Phase 7)
   - **Mitigation:** DSA band ships independently (Phase 5); a documented cylinder-beam
     approximation with stated error if the full model is unavailable.
4. **Risk:** Local `.npy` for some bursts not materialized.
   - **Likelihood:** Medium · **Impact:** Medium
   - **Mitigation:** Calibrate the bursts with local data first; fetch the rest from iacobus/h23 per
     `DATA_LOCATIONS.md`; missing bursts stay uncalibrated (gated), never faked.

## Edge Cases and Error Handling

1. **Case:** Dead/flagged channels (`noise_std≈0`).
   - **Expected:** excluded from the integral. **Implementation:** `np.clip(noise_std,1e-9,None)`
     and `valid` mask (`burstfit.py:590-591`); zero-S/N channels contribute 0 fluence.
2. **Case:** Burst absent from `dsa_sefd.csv`.
   - **Expected:** that band uncalibrated → gate closed for it. **Implementation:** `load_dsa_sefd`
     raises → caller leaves the band native.
3. **Error:** `.npy` path missing → `BurstDataset` raises `FileNotFoundError` (`io.py:104-105`);
   the burst is reported uncalibrated, not faked.

## Performance Considerations

- 12 bursts × 2 bands; each load is seconds. No performance concern. SSH SEFD fetch is one-time
  into a committed CSV.

## Documentation Updates

- [ ] Correct `analysis/burst_energies/CALIBRATION_REVIEW.md` Evidence 1: the FLITS data is
      per-channel z-scored **S/N** (`io.py:131-145`), not "native units c0 inherits" — and resolve
      the "Remaining to verify: .npy preserved scale or renormalized to S/N" item (renormalized to
      S/N).
- [ ] Update `research-chime-singlebeam-flux-units.md` gap note (the `.npy` provenance gap is now
      resolved: z-scored S/N).
- [ ] Set `scattering/configs/telescopes.yaml` `beam_model_h5` for DSA to the cube path once data
      access is standardized; document that `flux_jy_per_unit` is superseded by per-burst flux_cal
      (the config scalar stays `null` and the gate now reads the CSVs).
- [ ] Docstrings on all new functions (numerical "why" comments allowed per repo style).

## Timeline Estimate
- Phases 1–3 (kernel + inputs, no external data): ~half a day.
- Phase 4 (SEFD acquisition): ~1–2 h incl. SSH format check.
- Phase 5 (wire DSA + diagnostic): ~half a day.
- Phases 6–7 (CHIME epic + gate open): ~1–2 days, dominated by CHIME beam/SEFD sourcing.

## Open Questions

*(none — granularity and scope resolved with the user; the DSA pointing-offset and SEFD(ν)-shape
uncertainties have defined fallbacks above, not open questions.)*

---

## References

**Research Documents:**
- [Research: CHIME singlebeam flux units](research-chime-singlebeam-flux-units.md)

**Files Analyzed:**
- `scattering/scat_analysis/pipeline/io.py` (z-score load path)
- `scattering/scat_analysis/burstfit.py` (model + noise)
- `analysis/calculate_burst_energies.py` (energetics + gate)
- `analysis/dsa_beam.py` (DSA beam gain)
- `configs/bursts.yaml` (epoch + position)
- `scattering/configs/telescopes.yaml` (instrument constants)
- `analysis/burst_energies/CALIBRATION_REVIEW.md`
- `DATA_LOCATIONS.md` (`.npy` storage)

**External Documentation:**
- Andersen+2023 (AJ 166, 138) — CHIME/FRB flux calibration.
- Michilli+2021 (ApJ 910, 147) — CHIME baseband at known position.
- Law+2024 (ApJ 967, 29) — DSA-110 fluence/SEFD.
- Zhang 2018 (ApJ 867, L21) — band-limited E_iso + (1+z) k-correction.
- dsa110-rt SEFD dashboard (`github.com/dsa110/dsa110-rt`, `lxd110h23:5777`).

---

## Review History

### Version 1.0 — 2026-06-22
- Initial plan created (per-channel data-driven, DSA-first; both decisions approved by the user).
