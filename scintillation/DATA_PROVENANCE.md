# Scintillation data provenance (up-channelized CHIME + DSA)

Where the scintillation-bandwidth inputs come from, in both bands, so nobody has to
go digging again. This is the **two-screen Δν(ν)** measurement's data ledger: which
dynamic spectra / ACF products exist, what band and resolution each is, how it was
derived from the raw CHIME voltages, and what the FLITS scintillation pipeline can
consume directly vs. what needs conversion.

Companion to the repo-wide [`DATA_SOURCES.md`](../DATA_SOURCES.md) /
[`DATA_LOCATIONS.md`](../DATA_LOCATIONS.md); this file is the scintillation-specific,
band-by-band detail. Assembled and verified 2026-06-24 against the live files on
`h17` (`lxd110h17`) and the local replica.

> **Verified vs. inferred.** Facts below are tagged where it matters: **[verified]** =
> read directly off the file this session (shape, channel width, band, dict keys);
> **[from script]** = stated in `h17:/data/jfaber/upchannelize_chime.py`;
> **[from filename]** = read off a data filename (e.g. DM encoded in `*_I_<dm>_*`),
> not re-derived. Treat **[from filename]** DMs as hints, not authority — the
> authoritative DM registry is [`configs/bursts.yaml`](../configs/bursts.yaml).

---

## 0. The two bands (don't conflate them)

The single biggest footgun here: **CHIME and DSA up-channelized products look alike
(both are fine-channel `.npy`/`.pkl`) but live in different bands.** Always check the
frequency axis before trusting a "chime_acfs" or "acf_results" label.

| Band | Telescope | Frequency span | Native coarse channel | Up-channelized fine channel |
|------|-----------|----------------|-----------------------|-----------------------------|
| **CHIME** | CHIME/FRB baseband | 400–800 MHz | 0.390625 MHz (400 MHz / 1024) | 0.390625 / U (e.g. U=16 → **24.4 kHz**) |
| **DSA** | DSA-110 | ~1.28–1.53 GHz | ~0.030518 MHz | already ~**30.5 kHz** (native fine) |

The limiting scintle at CHIME is **narrower than one 0.390625 MHz coarse channel**
(NE2025/NE2001 predict sub-channel Δν_d for every co-detection), so it is *unresolved*
at native CHIME resolution. Up-channelization from the raw voltages is what exposes it.
DSA's native ~30.5 kHz channels already resolve its (broader, ν⁴·⁴-scaled) scintle, so
DSA needs no voltage up-channelization. **[from script]**

---

## 1. Burst ID ↔ nickname ↔ DM

Internal nicknames key all filenames/configs; CHIME event IDs key the baseband store.

| nickname | CHIME event ID | DM (pc cm⁻³) | FWHM (ms) | DM source |
|----------|----------------|--------------|-----------|-----------|
| casey    | 362593221 | 491.207 | 0.180 | [from script] |
| whitney  | 215063905 | 462.174 | 0.487 | [from script] |
| phineas  | 274819243 | 610.274 | 2.989 | [from script] |
| mahi     | 354049284 | 960.128 | 24.286 | [from script] |
| isha     | 252069198 | 411.568 | 1.805 | [from script] |
| chromatica | 356959136 | ~272 | — | [from filename] |
| freya    | 278720455 | ~912 | — | [from filename] |
| hamilton | 318353610 | ~518 | — | [from filename] |
| wilhelm  | 253635173 | ~602 | — | [from filename] |
| zach     | 210456524 | ~262 | — | [from filename] (`I_210456524_zach.npy`) |
| johndoeII| 230814aaas (date code) | ~696 | — | [from filename] |
| oran     | (unmapped) | ~397 | — | [from filename] (DSA cube only; CHIME ID not located) |

casey also appears under the date code `240229aaad` in older Stokes cubes
(`I_240229aaad_casey`). **[verified]** The DMs for the five up-channelization targets
are exact (script); the rest are read off `~/Developer/dsa110-local-data/DSA_bursts/`
filenames (`<nick>_<tel>_I_<DM>_…`) and should be confirmed against
[`configs/bursts.yaml`](../configs/bursts.yaml) before citing.

---

## 2. CHIME up-channelized products (voltage-derived, 400–800 MHz)

### 2a. New pass — `upchannelize_chime.py` (current method)

**Producer:** `h17:/data/jfaber/upchannelize_chime.py` — runs **inside the
`chimefrb/baseband-analysis:latest` docker image** on `h17` (`baseband_analysis`
1.9.0 + CADC `vos`). Per target it: (1) locates the ~1 GB `singlebeam_<id>.h5`,
(2) **coherently dedisperses** the complex per-channel baseband at the burst DM,
(3) up-channelizes each 0.390625 MHz coarse channel by a per-target factor U via the
package's internal `_upchannel(fftsize=2U, downfreq=2)`, (4) forms Stokes I, writes a
small `<nick>_chime_upchan.npy` + `<nick>_chime_freq.npy`. **[from script]**

Coherent dedispersion (not incoherent) is mandatory: intra-channel dispersive smearing
imprints a chirp that survives up-channelization as a spurious narrow-band ACF feature
counterfeiting a scintle; only de-chirping the raw voltages removes it exactly. The PFB
inverse gives the fine channels a flat passband (no coarse-channel scallop in the
small-lag ACF). **[from script]**

**Per-target up-channelization factors** (skeptic-corrected, sized so the
dominant/narrower scintle spans ≥4 fine channels across its HWHM):

| target | U | regime | recoverable |
|--------|---|--------|-------------|
| casey   | 16  | host-dominated (host 0.187 < MW floor 0.207 MHz) — the clean host case | yes |
| whitney | 16  | MW-floor-dominated (0.140 MHz floor) | yes |
| phineas | 16  | MW-floor-dominated (0.206 MHz floor) | yes |
| mahi    | 512 | MW-floor-dominated (0.0036 MHz floor); slow python path; safe only because FWHM=24 ms | yes |
| isha    | 256 | **NOT cleanly resolvable** — railed DSA input + scale past the time-smearing wall | **upper-bound only** (`--run-unresolvable`) |

**Outputs — `h17:/data/jfaber/upchan_codetections/` (all 5 targets generated 2026-06-24):**
**[verified]** each target as `<nick>_chime_upchan.npy` (Stokes-I waterfall, float32) +
`<nick>_chime_freq.npy`:

| target | upchan .npy | generated (mtime) |
|--------|-------------|-------------------|
| casey   | `casey_chime_upchan.npy` (86 MB) | 2026-06-23 16:47 |
| whitney | `whitney_chime_upchan.npy` (97 MB) | 2026-06-24 12:01 |
| phineas | `phineas_chime_upchan.npy` (132 MB) | 2026-06-24 12:10 |
| mahi    | `mahi_chime_upchan.npy` (84 MB) | 2026-06-24 12:14 |
| isha    | `isha_chime_upchan.npy` (92 MB, upper-bound) | 2026-06-24 12:18 |

These `.npy` carry **no `times_s`** (spectrum + freq only) — package per the caveat below.
- **casey is additionally packaged + local**: `scintillation/data/casey_chime.npz` — keys `power_2d (12336, 1748)`, `frequencies_mhz` (400.6–799.0 MHz, **24.4 kHz**, 12336 ch), `times_s`; re-wrapped into the FLITS `DynamicSpectrum.from_npz` contract, **directly runnable**. `scintillation/data/casey_chime_hi.npz` is a 711–799 MHz subset (2111 ch). **[verified]**
- whitney / phineas / mahi / isha exist as raw `*_upchan.npy` on h17 but are **not yet packaged to npz nor run** — that is the immediate next campaign step (§6).

> ⚠ **`times_s` caveat.** `upchannelize_chime.py` writes only `*_upchan.npy` (spectrum)
> and `*_freq.npy`. The packaged `.npz` needs `times_s`; the post-up-channelization
> time sample is `dt = 2.56e-6 s × 2 × U` (e.g. U=16 → 81.9 µs). Whoever packages a new
> target must synthesize the time axis from this, as was done for casey.

### 2b. Old pass — `chime_acfs/*_subband_acf_fits.pkl` (legacy fit products)

`scintillation/chime_acfs/{chromatica_356959136, freya_278720455, hamilton_318353610,
wilhelm_253635173}_subband_acf_fits.pkl` — CHIME-band (425–775 MHz), an **earlier**
up-channelization+fit pass. **[verified]** Schema is NOT `acf_results`; keys are
`f_cents`, `1_lorenz`, `2_lorenz`, `acfs_offset`, `lm_fitting_objects` — i.e. they store
the already-fit 1- and 2-Lorentzian per-sub-band results (`sub_scint_1`, `sub_scint_2`,
`sub_scint_uncert_*`, `mods*`). Read by the legacy notebook
[`chime_acfs/pickle.ipynb`](chime_acfs/pickle.ipynb), whose `read_pkl_data` extracts
exactly the `subband_measurements` dict contract (`bw`, `bw_err`, `finite_err`, `mod`,
`mod_err`) that the **current pipeline now emits** — so the new pipeline output is a
drop-in for that reader. Sub-band counts: chromatica 8, freya 4, hamilton 8, wilhelm 4.

These four cover a different set than §2a's five targets; together the CHIME-resolvable
union is **{casey, whitney, phineas, mahi, isha} ∪ {chromatica, freya, hamilton, wilhelm}**.
The `1_lorenz`/`2_lorenz` naming is the lineage the dead `2c`/`3c`-in-name heuristic was
built around; the [multi-component selector](scint_analysis/revalidation.py) (PR #58)
is its statistically-gated replacement.

---

## 3. DSA-band ACF products (~1.3–1.5 GHz)

`h17:/data/jfaber/arc_archive_2026-06/acf_results/{chromatica, freya, wilhelm}_acf_results.pkl`
— **DSA band**, 1321–1466 MHz, 30.5 kHz channels. **[verified]** Schema **is** the
FLITS `acf_results` dict (`subband_acfs`, `subband_lags_mhz`,
`subband_center_freqs_mhz`, `subband_channel_widths_mhz`, `subband_num_channels`;
wilhelm additionally has `noise_template` + `sigma_self_mhz`). **Directly runnable**
through `analyze_scintillation_from_acfs` — confirmed this session:

| burst | sub-bands | auto-BIC best model | n_components |
|-------|-----------|---------------------|--------------|
| chromatica | 4 | `fit_lor` | 1 (selector ran, `n_per_subband=[1,1,1,1]`) |
| freya | 4 | `fit_power` | 1 (selector skipped — non-Lorentzian) |
| wilhelm | 2 | `fit_power` | 1 (selector skipped — non-Lorentzian) |

Derived from `arc_archive_2026-06/processed_spectra_pkl/{chromatica,freya,wilhelm}_processed_spectrum.pkl`.
**DSA-band casey is missing *locally*** (`scintillation/configs/bursts/casey_dsa.yaml`
points at `${FLITS_ROOT}/scintillation/data/casey.npz`) — but it, and the DSA npz for
**all 12 bursts**, are on CANFAR at `$ARC/FLITS/scintillation/data/*.npz`; just `vcp`
them down (§7a).

---

## 4. Raw inputs and the rest of the H17 archive

**Raw CHIME voltages (singlebeam baseband `.h5`, ~1 GB each):**
- **CADC vos source:** `arc:projects/chime_frb/data/chime/baseband/processed/<yyyy>/<mm>/<dd>/astro_<id>/singlebeam_<id>.h5` (vcp'd in-container with `~/.ssl/cadcproxy.pem`). **[from script]**
- **Pre-staged on h17:** `/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam/` — singlebeam `.h5` for IDs 210456524, 215063905, 224263996, 252069198, 253635173, 274819243, 278720455, 311723353, 318353610, 354049284 (and casey 362593221). **[verified]** So up-channelizing the remaining targets needs **no arc fetch** — the voltages are local to h17.
- vcp fallback landing: `h17:/data/jfaber/chime_singlebeam/`.

**Raw intensity spectra (native resolution, used by the SCATTERING pipeline, not scint):**
`~/Developer/dsa110-local-data/DSA_bursts/*.npy` — 24 cubes (12 bursts × {chime,dsa}),
shape `(n_freq, n_time)`, from arc
`/arc/home/jfaber/baseband_morphologies/chime_dsa_codetections/data/DSA_bursts/`.
**[verified]** These are *not* up-channelized.

**`h17:/data/jfaber/arc_archive_2026-06/`** — the consolidated arc archive (pulled
2026-06). Notable subdirs **[verified]**:
- `acf_results/` — the DSA `*_acf_results.pkl` (§3); plus **stacked multi-factor CHIME ACFs** `acf_codetections_fftsize{16,32,512}_downfreq1.npz` and `zach_acf_codetections_fftsize{16,32}.npz` (keys `onburstacf`, `peakburstacf`, `sub_acfs`, `sub_acfs_peak`, `sub_fcents`, `sub_lags` — multi-burst stacked, **no per-burst label**; fftsize = 2U).
- `code/` — the original scintillation lineage: `scinttools_{old,new,v3}.py`, `burstfit_subband.py`, `burstfittools.py`, `frb_scintillator_wAnisotropy*.py`, `baseband_analysis_*.py`.
- `singlebeam_h5/`, `processed_spectra_pkl/` (DSA processed spectra), `stokes_cubes_npy/` (native Stokes-I cubes, `I_<id>_<nick>.npy` naming — a useful ID↔nick cross-check), `fullstokes_pkl/`, `other_data_npy/`, `notebooks/`, `configs_json/`, `scattering_dirs/`, `plots/`, `text_tables/`, `trashed_directories/`.

---

## 5. What the current pipeline can consume

| product | band | schema | pipeline-ready? |
|---------|------|--------|-----------------|
| `scintillation/data/casey_chime{,_hi}.npz` | CHIME | DynamicSpectrum npz | ✅ full pipeline (ACF → fit → Δν(ν)) |
| `arc:.../acf_results/{chromatica,freya,wilhelm}_acf_results.pkl` | DSA | `acf_results` | ✅ `analyze_scintillation_from_acfs` direct |
| `chime_acfs/*_subband_acf_fits.pkl` | CHIME | legacy `1_lorenz`/`2_lorenz` | ⚠ legacy notebook only — needs conversion to re-fit |
| `acf_codetections_fftsize*.npz` | CHIME | stacked, unlabeled | ⚠ needs per-burst de-stacking |
| `upchan_codetections/casey_chime_upchan.npy` | CHIME | raw waterfall (no times) | ⚠ package to npz first (synthesize `times_s`, §2a) |

---

## 6. How to extend the campaign (regenerate / fill gaps)

**Up-channelize the remaining CHIME targets** (voltages already on h17, §4):
```bash
ssh h17
docker run --rm \
  -v /data:/data \
  -v /data/research:/data/research:ro \
  -v "$HOME/.ssl:/root/.ssl:ro" \
  chimefrb/baseband-analysis:latest \
  python3 /data/jfaber/upchannelize_chime.py whitney phineas mahi
# isha: add --run-unresolvable for the lower-confidence upper bound.
# outputs -> /data/jfaber/upchan_codetections/<nick>_chime_upchan.npy (+ _freq.npy)
```
Then **package** each `<nick>_chime_upchan.npy` + `<nick>_chime_freq.npy` into a
`power_2d`/`frequencies_mhz`/`times_s` npz (synthesize `times_s` from
`dt = 2.56e-6 × 2 × U`, §2a caveat) and run the FLITS scintillation pipeline
(`scintillation/configs/bursts/<nick>_chime.yaml`, mirroring `casey_chime.yaml`).

**DSA-band npz (any burst)**: don't regenerate — `vcp` it from CANFAR,
`$ARC/FLITS/scintillation/data/<nick>.npz` (§7a), then run `<nick>_dsa.yaml`.

---

## 7a. CANFAR / arc — the authoritative remote copy (verified 2026-06-24)

The **whole** scintillation working tree is mirrored on CANFAR under
**`arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/`** (call it `$ARC`).
Access is **container-only**: `vls`/`vcp` live at `/opt/pysetup/.venv/bin/` inside the
`chimefrb/baseband-analysis:latest` image on h17; the `cadcproxy.pem` is staged at
`h17:~/.ssl/cadcproxy.pem`. Probe pattern (**[verified]** this works):

```bash
ssh h17
docker run --rm -v "$HOME/.ssl:/ssl:ro" chimefrb/baseband-analysis:latest bash -lc \
  '/opt/pysetup/.venv/bin/vls --certfile=/ssl/cadcproxy.pem \
     arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scintillation/data'
# vcp <arc:...path> <local>  to download.
```

| What | arc path (under `$ARC`) | band / resolution |
|------|--------------------------|-------------------|
| **DSA-band scint npz — ALL 12 bursts** (incl. the locally-missing `casey.npz`, 123 MB) | `FLITS/scintillation/data/{casey,whitney,phineas,mahi,isha,chromatica,freya,hamilton,wilhelm,zach,johndoeII,oran}.npz` | DSA (~1.3–1.5 GHz) |
| DSA native cubes | `FLITS/scintillation/data/*_dsa_I_*.npy` and `data/DSA_bursts/` | DSA native |
| **Up-channelized CHIME ACF products** (legacy `1_lorenz`/`2_lorenz` pkls) | `FLITS/scintillation/chime_acfs/*.pkl` | CHIME up-chan |
| Per-burst ACF/processed-spectrum cache | `FLITS/scintillation/data/cache/<burst>/`, `*_acf_results.pkl`, `*_processed_spectrum.pkl` | mixed |
| **Native CHIME cubes — ALL 12** (coherently dedispersed, **1024 ch** × 32000, *not* up-channelized) | `data/CHIME_bursts/dmphase/<nick>_chime_I_<dm>_..._32000b_cntr_bpc.npy` | CHIME native |
| Raw CHIME voltages (singlebeam `.h5`) | `arc:projects/chime_frb/data/chime/baseband/processed/<y>/<m>/<d>/astro_<id>/singlebeam_<id>.h5` | voltages |

**Key correction caught here:** `data/CHIME_bursts/dmphase/` is **native 1024-channel**
CHIME (a casey cube is 131,072,128 B = 1024 × 32000 × float32), i.e. the coherently-
dedispersed scattering inputs — **NOT** the up-channelized scintillation spectra. The
up-channelized CHIME **dynamic spectra** (24.4 kHz, like local `casey_chime.npz`) exist
**only for casey** anywhere (h17 `upchan_codetections/` + local npz). What arc adds for
CHIME is the **legacy up-channelized ACF *fit products*** (`chime_acfs/`), not re-fittable
spectra. So generating up-channelized CHIME spectra for the other targets still requires
running `upchannelize_chime.py` from the voltages (§6).

**This resolves the DSA-band gap:** fetch `casey.npz` (and every other DSA npz) from
`$ARC/FLITS/scintillation/data/` rather than regenerating — supersedes the §3 / §6
"missing locally" note.

## 7c. What the analysis notebooks actually used (per-burst, per-band)

So nobody re-asks "did we only do casey?": **no.** Verified from the notebook cells.

| notebook | bursts | band of data loaded |
|----------|--------|---------------------|
| `chime_acfs/pickle.ipynb` (legacy CHIME ACF reader) | **chromatica, freya, hamilton, wilhelm** (4) | CHIME up-chan — `*_subband_acf_fits.pkl` (`1_lorenz`/`2_lorenz`) |
| `notebooks/scintillation_analysis.ipynb` + `analyses/templates/scintillation_template.ipynb` (main; identical) | **all 12** | per-burst `*_acf_results.pkl` — **DSA band** |
| `analyses/bursts/wilhelm/scintillation_manual.ipynb`, `notebooks/debug/wilhelm_manual.ipynb` | wilhelm | `data/cache/wilhelm/wilhelm_acf_results.pkl` (DSA) |

Takeaways:
- **CHIME-band scintillation was analyzed for only 4 bursts** (chromatica, freya, hamilton,
  wilhelm) via the legacy fit pkls — **casey was *not* among them.**
- **casey is the inverse special case**: the only burst with a fresh up-channelized CHIME
  *dynamic spectrum* (`casey_chime.npz`, new `upchannelize_chime.py` method) — the
  proof-of-concept for the voltage→up-channelize path, runnable end-to-end through the
  current pipeline. It was *not* in the old CHIME ACF notebook set.
- The main notebook's all-12 loop is **DSA-band**. So CHIME-band Δν(ν) coverage today is
  ~4 legacy bursts + casey; the other ~7 are **DSA-only** until §6 up-channelizes them.

## 7b. Open / pending (data-provenance gaps)

- **oran** CHIME event ID not located (DSA cube only); **johndoeII** keyed by date code
  `230814aaas` rather than a numeric ID.
- The legacy `chime_acfs/*_subband_acf_fits.pkl` (§2b) hold fits, not raw ACFs, so they
  can't be re-fit by the new selector without recovering their underlying ACFs (or
  re-up-channelizing those four bursts via §6).
- DMs for non-target bursts are **[from filename]** only — reconcile with `configs/bursts.yaml`.
