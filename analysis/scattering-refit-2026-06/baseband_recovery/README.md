# CHIME baseband recovery of the resolution-limited co-detection sightlines

Recover the diffractive scintillation bandwidth `Dnu_d` at CHIME (400–800 MHz) for the sightlines
unresolved at CHIME native channel width (0.390625 MHz), by coherently dedispersing and
**upchannelizing** the CHIME baseband, then feeding the small returned spectra into the existing
FLITS scintillation pipeline.

`upchannelize_chime.py` is the worker (runs in the baseband_analysis image). This file is the plan.

## TL;DR — feasibility: **ready on h17**

Runs on **h17 (`lxd110h17`)** in the already-pulled `chimefrb/baseband-analysis:latest` docker image
— **no CANFAR Science-Platform / Harbor dependency**. Verified live in-container: `baseband_analysis
1.9.0` imports, the CADC `vos` client (`vls`/`vcp`) is present, `~/.ssl/cadcproxy.pem` (`CN=jfaber_1ff`,
valid to **Jul 18 2026**) authenticates, and `vls arc:projects/chime_frb/...` reads the baseband store
directly from h17. `/data` has ~1.8 T free for the ~1 GB-per-target pulls.

**Outcome (casey run):** the route works end-to-end, but the science answer is negative — casey (the
best case) does **not** yield a contract-PASS CHIME `Dnu_d`. See "Empirical result" below; the rest of
this file is the route/plan that produced it.

## Why this work exists (prior art)

All OLD scint notebooks (`scint_{casey,phineas,whitney,mahi,isha}.ipynb` on iacobus) are
**DSA-110-only** (1.3–1.5 GHz, 6144 ch @ 30.5 kHz). **None loaded or analyzed CHIME data**; there is
no CHIME scint notebook for these sources anywhere. A CHIME `upchannel()` helper was staged in a stray
checkpoint but a tree-wide grep finds **zero callers** — never wired in. Of the OLD DSA fits, only
`casey` is a genuine clean recovery (`Dnu_d = 7.5 MHz`, R²=0.99); `phineas` is marginal/degenerate,
and `whitney`/`mahi`/`isha` show **byte-identical stale `phineas` output` (their ACF/fit cells were
never re-run). So at CHIME the answer is simply: never attempted. This is the first attempt. (The
repo's current `scintillation/configs/bursts/<name>_dsa.yaml` fits *are* genuine per-target — used
below as the DSA `Dnu_d` inputs.)

The limiter is CHIME for all of them: the predicted scint scale is at/below one 0.390625 MHz CHIME
coarse channel, so it is unresolved without baseband upchannelization. DSA's 30.5 kHz channels already
resolve the DSA-side scale and are not the limiter.

## HONEST physics caveat — which targets are actually recoverable (skeptic verdict, authoritative)

The feasibility pass first sized the upchannelization factor `U` to resolve the **host** scintle and
used a pulse-non-smearing test (`time_res << FWHM`) as the safety check. The adversarial review
**revised** this. Two binding corrections:

1. **Resolvability ≠ no time-smearing.** `time_res << FWHM` only proves the *pulse* isn't smeared.
   The binding test is whether channels can be made fine enough to resolve the **narrower (dominant)**
   scintle — the finer modulation that dominates the small-lag ACF.
2. **Two-screen logic was inverted.** For phineas/whitney/mahi/isha the predicted host `Dnu_d`
   *exceeds* the NE2025 Galactic floor, so the **Galactic** scintle is the narrower, dominant scale —
   the host is the broad/washed component. Only **casey** is genuinely host-dominated (host 0.187 <
   MW floor 0.207 MHz).

Re-sizing `U` to the dominant scale at ≥4 channels/HWHM:

| target  | U (verified) | resolvable | dominant scale | note |
|---------|:---:|:---:|---|---|
| casey   | **16**  | ✅ yes | host | only genuinely host-dominated CHIME measurement; cleanest DSA input (γ=6.694, redχ²=0.57). |
| whitney | **16**  | ✅ yes | MW floor | native ×16 (24 kHz) resolves the 0.140 MHz floor; Galactic, not host. |
| phineas | **16**  | ✅ yes | MW floor | min factor is 8; native ×16 is cleaner and over-resolves the 0.206 MHz floor (long FWHM). Galactic, not host. |
| mahi    | **512** | ✅ yes | MW floor | floor 0.0036 MHz (high-scattering MW sightline) → ×512 via `_upchannel(fftsize=1024)` (slow). Safe **only** because FWHM=24.3 ms. |
| isha    | 256 (probe) | ❌ **NO** | MW floor | **NOT cleanly resolvable.** DSA input railed (γ floor-pinned); dominant scale needs U≥256–512 but the 1.8 ms burst smears to <3 time elements — at/past the time-bandwidth wall. |

**Bottom line: 4 of 5 recoverable (casey host-dominated; phineas/whitney/mahi MW-floor-dominated),
isha not.** There is **no single global U**; the per-target factor lives in `upchannelize_chime.py`.
`isha` is OFF by default (`--run-unresolvable` → upper bound only, do not publish as a measurement).

> Interpretation: for phineas/whitney/mahi the recovered CHIME `Dnu_d` is the **Galactic** scintle,
> not a host-screen measurement (consistent-with-floor is the expectation, not an excess). Only casey
> yields a clean host-dominated CHIME `Dnu_d`. This bounds what the excess test can claim per sightline.

## API reality (verified in the image, not assumed)

- `baseband_analysis.analysis.waterfall.waterfall_from_beamformed(...)` is **BROKEN** in this image
  (v1.9.0): it feeds `upchannel()`'s 3-tuple straight into `incoherent_dedisp`, which calls
  `matrix_in.copy()` → `AttributeError` on the tuple. The worker does **not** use it for any target.
- The worker drives the package's own primitives directly for **all** factors:
  `BBData.from_file` → `coherent_dedisp(time_shift=True)` → `_upchannel(fftsize=2U, downfreq=2)`
  (returns `(spec, freq, chan_id)`; spec `(npol, nblock, nfine)` complex, freq high→low, factor
  `U = fftsize/downfreq`) → Stokes I = `|X|²+|Y|²` directly from the complex spectrum. **No
  `incoherent_dedisp`** — coherent dedispersion already de-chirps fully.
- **Quirk:** the public `baseband_analysis.core.sampling.upchannel(data, fftsize, downfreq)`
  **ignores its args** and is hard-wired to ×16 — which is why the worker calls the internal
  `_upchannel` with explicit `fftsize=2U`. The worker's `assert` on fine-channel width = 0.390625/U
  MHz is the first-run shape sanity-check (casey: measured df=24.414 kHz at U=16, exact).

## Data — the arc URIs (verified to exist; ~1 GB each)

`vcp`'d from arc to h17 scratch (`/data/jfaber/chime_singlebeam`, idempotent) — h17 does **not** mount
`/arc`, so unlike CANFAR these are pulled, not read in place. ~5 GB total; fits easily in the 1.8 T free.

| target  | CHIME id  | DM (pc cm⁻³) | arc URI |
|---------|-----------|-------------:|---------|
| casey   | 362593221 | 491.207 | `arc:projects/chime_frb/data/chime/baseband/processed/2024/02/29/astro_362593221/singlebeam_362593221.h5` |
| whitney | 215063905 | 462.174 | `arc:projects/chime_frb/data/chime/baseband/processed/2022/03/10/astro_215063905/singlebeam_215063905.h5` |
| phineas | 274819243 | 610.274 | `arc:projects/chime_frb/data/chime/baseband/processed/2023/03/07/astro_274819243/singlebeam_274819243.h5` |
| mahi    | 354049284 | 960.128 | `arc:projects/chime_frb/data/chime/baseband/processed/2024/01/22/astro_354049284/singlebeam_354049284.h5` |
| isha    | 252069198 | 411.568 | `arc:projects/chime_frb/data/chime/baseband/processed/2022/11/13/astro_252069198/singlebeam_252069198.h5` |

DMs/ids/FWHMs from `crossmatching/notebook_reproduction_fixture.json` (DMs also in `configs/bursts.yaml`).

## Run on h17

Stage the worker to h17, then run it inside the image with the cert + scratch + out mounted. The
container has `vcp`, so it pulls the `.h5` itself; the `HOME=/root` + cert mount lets vos authenticate.

```bash
# 1. copy the worker to h17
scp analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py h17:/data/jfaber/

# 2. run the 3 native-x16 targets (casey first = clean host validation case)
ssh h17 'docker run --rm \
  -e HOME=/root \
  -v /home/ubuntu/.ssl:/root/.ssl:ro \
  -v /data/jfaber:/data/jfaber \
  chimefrb/baseband-analysis:latest \
  python /data/jfaber/upchannelize_chime.py casey whitney phineas \
    --scratch /data/jfaber/chime_singlebeam \
    --out /data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections'

# 3. mahi (slow, x512 via _upchannel); isha only as an upper bound
ssh h17 'docker run --rm -e HOME=/root -v /home/ubuntu/.ssl:/root/.ssl:ro -v /data/jfaber:/data/jfaber \
  chimefrb/baseband-analysis:latest \
  python /data/jfaber/upchannelize_chime.py mahi --out /data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections'
# isha:  ... upchannelize_chime.py isha --run-unresolvable ...
```

`vcp` may need the cert at the vos default (`$HOME/.ssl/cadcproxy.pem`); `HOME=/root` + the mount above
satisfies that. If the container runs as non-root, mount to that user's `~/.ssl` instead.

## Return the small products

Each target writes `<name>_chime_upchan.npy` (Stokes-I, `(n_freq, n_time)`, float32) +
`<name>_chime_freq.npy` (ascending MHz), both kB–MB. Pull them to the repo:

```bash
scp 'h17:/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/*_chime_*.npy' \
    analysis/scattering-refit-2026-06/baseband_recovery/products/
```

Shape convention matches FLITS `BurstDataset`: `(n_freq, n_time)`, **frequency ascending** (the worker
flips both axes). Do **not** pull the gigabyte `.h5` files.

## Feed the FLITS scint pipeline (the co-detection excess test)

1. Time-integrate the burst spectrum, compute the 1D spectral ACF (scinttools-style; same machinery as
   `scintillation/scint_analysis`).
2. Fit `L(x) = m₁²/(1+(x/γ₁)²) + c`; HWHM `γ₁` **is** the CHIME `Dnu_d` (kHz). Build
   `scintillation/configs/bursts/<name>_chime.yaml` mirroring `<name>_dsa.yaml`.
3. **Excess test:** with CHIME `Dnu_d` + DSA `Dnu_d`, fit `Dnu_d ∝ ν^α` and compare the CHIME scintle
   to the **NE2025 Galactic floor** at ~600 MHz (`scintillation/ne2025/query_ne2025_scint.py`
   `galactic_floor`, α=4.4). Per-target meaning (see caveat): casey = clean **host** excess test;
   phineas/whitney/mahi = **Galactic**-scintle recovery (consistent-with-floor expected); isha excluded.

## Gates that apply

- **Fit-validation contract** (`.cursor/rules/AGENT_CONFIGURATION_FLITS.md`): every Lorentzian ACF fit
  gets PASS/MARGINAL/FAIL (bounds, χ²_red/R²/residuals, physics). Report failures; `isha` if run is
  expected MARGINAL/FAIL.
- **Verify gate** (`.claude/workflows/fit-verify.js`): a separate judge agent re-checks each
  `*_fit_results.json` against the runtime gate.
- **Figure-review Stop gate** (`.claude/hooks/figure-review-gate.sh`): any ACF/waterfall PNG by a
  `figures.manifest.json` must be **Read** (rendered) and recorded in `figures.review.json`.

## Upchannelization status — all 5 targets generated (2026-06-24)

All five targets were upchannelized end-to-end on h17 via the wrapper script
`/data/research/astrophysics/frbs/chime-dsa-codetections/bin/baseband_analysis_python.sh`
(which wraps `docker run --rm --entrypoint python -v /data:/data
chimefrb/baseband-analysis:latest`). The h17 copy of the worker is byte-identical to
the repo copy (sha256 `cd5f6e35…`, git rev `8732a695`).

### Exact run commands (as executed on h17)

```bash
# recoverable targets (whitney + phineas native x16; mahi x512 slow path)
ssh h17 '/data/research/astrophysics/frbs/chime-dsa-codetections/bin/baseband_analysis_python.sh \
  /data/jfaber/upchannelize_chime.py whitney phineas mahi'

# isha — lower-confidence upper bound (flagged NOT cleanly resolvable)
ssh h17 '/data/research/astrophysics/frbs/chime-dsa-codetections/bin/baseband_analysis_python.sh \
  /data/jfaber/upchannelize_chime.py isha --run-unresolvable'
```

(casey was generated earlier on 2026-06-23 via the same route; see provenance table below.)

### Generated products

All products at `/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/` on h17. Shape convention:
`(n_freq, n_time)`, frequency ascending, float32 Stokes-I.

| target  | U   | shape (nfreq, ntime) | df (kHz) | dt (ms) | finite | upchan.npy sha256 (prefix) | generated |
|---------|----:|----------------------|---------:|--------:|-------:|----------------------------|-----------|
| casey   |  16 | (12336, 1748)        | 24.414   | 0.0819  | 92.2%  | `c6cdfa28…`                | 2026-06-23 16:47 |
| whitney |  16 | (13808, 1748)        | 24.414   | 0.0819  | 92.3%  | `a50048d9…`                | 2026-06-24 12:01 |
| phineas |  16 | (14512, 2276)        | 24.414   | 0.0819  | 88.2%  | `e77019ef…`                | 2026-06-24 12:10 |
| mahi    | 512 | (387072, 54)         | 0.763    | 2.6214  | 92.3%  | `e7f96fa1…`                | 2026-06-24 12:14 |
| isha*   | 256 | (210688, 109)        | 1.526    | 1.3107  | 91.7%  | `a4461e21…`                | 2026-06-24 12:18 |

\* isha generated with `--run-unresolvable`; lower-confidence upper bound only (railed DSA
input + 1.8 ms burst smears to <3 time elements at U=256). Do not publish as a measurement.

Each target also has a matching `<name>_chime_freq.npy` (ascending MHz, 400–800 MHz band).

### Casey fit result (the best case) — FAILS the contract

casey was converted to `scintillation/data/casey_chime.npz` and fit through the FLITS scint
pipeline. Both the **full band** (`casey_chime.yaml`) and a **high-band focus** slice
(`casey_chime_hi.yaml`, 711–799 MHz) were run:

| run | best ACF model | γ₀ (Δν_d) | α (scaling) | χ²_red | verdict |
|-----|---|---|---|---|---|
| full 400–800 | power-law | 0.086 ± 0.052 MHz | 8.00 ± 5.10 | 17.2 | **FAIL** |
| high-band 711–799 | power-law | 2.624 ± 1.146 MHz | 7.96 ± 13.56 | 5.25 | **FAIL** |

Independent `fit-validation` agent verdict on the high-band run: **FAIL** — α=7.96 violates the L1
bound (1.5<α<6.0), χ²_red=5.25>3, σ(α)/α=1.70>1 (unconstrained), and BIC prefers a power-law over a
Lorentzian by ΔBIC≈17 (i.e. there is no clean diffractive Lorentzian to fit). Disabling the self-noise
template leaves the 2D fit numerically identical (best model flips to Gaussian, still not Lorentzian) —
the failure is **structural**, not a self-noise artefact. The narrow ~117 kHz "peak" a crude
lag-skipping ACF reported (R²=0.94) was self-noise, correctly rejected by the pipeline.

**Implication:** casey is the *only* genuinely host-dominated CHIME case and the cleanest DSA input —
if its CHIME scintle is not contract-recoverable, the phineas/whitney/mahi cases (MW-floor-dominated,
narrower predicted scales) are very unlikely to fare better.

### All four remaining targets fit (2026-06-24) — all FAIL, as predicted

The other four products were converted (`npy_to_npz.py <name> --U <U>`, no de-ripple — flip+slice
only; byte-identical roundtrip verified for casey) and fit through the FLITS scint pipeline with a
config mirroring `casey_chime.yaml` (full 400–800 MHz, num_subbands=4, fit_lagrange=1.0 MHz, poly-1
baseline, self-noise template on; per-burst on/off windows auto-set from the nan-aware burst profile):

| target | U | best ACF model | bw / γ₀ | α (scaling) | χ²_red | verdict |
|--------|--:|----------------|---------|-------------|--------|---------|
| whitney | 16 | *no successful fit* (sub-bands too narrow → defaulted to `lorentzian_component`) | — | — | — | **FAIL** |
| phineas | 16 | Gaussian (`fit_sn_tpl_gauss`) | 8.9e5 MHz | 59.4 ± 24.2 | 16.6 | **FAIL** |
| mahi | 512 | *unfittable* — 54 time bins, burst at frame edge (peak@1) → insufficient off-pulse for baseline/noise modeling (pipeline `NumbaTypeError` on the degenerate masked frame) | — | — | — | **FAIL** |
| isha | 256 | power-law (`fit_sn_power`) | 49.7 MHz | −8.5 ± 6.3 | 7.7e5 | **FAIL** |

None yields a clean diffractive Lorentzian: whitney has no successful sub-band ACF fit; phineas/isha
prefer Gaussian/power-law with α far outside the physical 1.5<α<6.0 band and χ²_red ≫ 3; mahi is
degenerate (the U=512 frame has only 54 time bins with the burst at the edge). **Conclusion: the
CHIME-band diffractive scintle is not contract-recoverable for any of the five co-detection
sightlines** at the achievable up-channelization. The two-screen Δν(ν) analysis therefore rests on the
**DSA-band** measurements (where Δν_d *is* resolved at 30.5 kHz); the CHIME side contributes
non-detections / upper limits consistent with the NE2025 Galactic floor, not measurements. This is the
documented justification for excluding the CHIME band from the per-sightline Δν(ν) fits.

Raw per-target fit outputs: `products/campaign_results.json` (this run); pipeline artifacts under the
scratch run dir. The four `<name>_chime.npz` were produced by `npy_to_npz.py` from the h17 products
(checksums above); they are gitignored (`*.npz` under `scintillation/data/` / `products/`).

## Data provenance — full chain

### Software

| component | version / identifier |
|-----------|---------------------|
| worker script (repo) | `analysis/scattering-refit-2026-06/baseband_recovery/upchannelize_chime.py` |
| worker git rev | `8732a695a2ef4b6e0fad5caf3c14893958a9a09f` (2026-06-23 18:33 -0700) |
| worker sha256 | `cd5f6e353e0e49b8410b30bd21ba59fa24a4e52564a7ffbfb7ac6e4b8c9a851b` |
| h17 copy sha256 | `cd5f6e35…` (byte-identical to repo) |
| docker image | `chimefrb/baseband-analysis:latest` |
| image digest | `sha256:f510909d892d0d5224c982c590cbe80967a49a59b79c396ab72bb710105c4c41` |
| image id | `sha256:8c903ec6a5a836e8a97fe3468fd3ee02177c220ead84e6d1d25e8f41b735db4b` |
| image created | 2026-03-25T14:19:50Z |
| baseband_analysis | 1.9.0 |
| h17 wrapper | `/data/research/astrophysics/frbs/chime-dsa-codetections/bin/baseband_analysis_python.sh` |

### Raw baseband (input) — h17 staged copies

All at `/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam/`.
Source: CADC arc `arc:projects/chime_frb/data/chime/baseband/processed/<date>/astro_<id>/singlebeam_<id>.h5`.

| target  | CHIME id  | DM (pc cm⁻³) | size | h5 sha256 (prefix) |
|---------|-----------|-------------:|-----:|---------------------|
| casey   | 362593221 | 491.207 | 990M  | `ea15c60b…` |
| whitney | 215063905 | 462.174 | 1.1G  | `e76950cc…` |
| phineas | 274819243 | 610.274 | 1.5G  | `3ce7ab34…` |
| mahi    | 354049284 | 960.128 | 970M  | `bcf3b157…` |
| isha    | 252069198 | 411.568 | 1.1G  | `0dc5ec98…` |

DMs/ids/FWHMs from `crossmatching/notebook_reproduction_fixture.json` (DMs also in `configs/bursts.yaml`).

### Processing chain (per target)

```
staged singlebeam_<id>.h5
  → BBData.from_file
  → coherent_dedisp(dm, time_shift=True)        # exact de-chirp at burst DM
  → _upchannel(fftsize=2U, downfreq=2)           # U = fftsize/downfreq; internal primitive
  → Stokes I = |X|² + |Y|²                       # (npol, nblock, nfine) → (nfine, n_time)
  → flip to frequency-ascending
  → assert n_fine ≥ 1024, finite_frac > 0.3, df = 0.390625/U MHz (rtol 5%)
  → np.save(<name>_chime_upchan.npy, float32)
  → np.save(<name>_chime_freq.npy, float64)
```

No `incoherent_dedisp` step (coherent dedisp already de-chirps fully). No `waterfall_from_beamformed`
(broken in v1.9.0 — see "API reality" above).

### Products (output) — h17

All at `/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/`. Full sha256 checksums:

| file | sha256 |
|------|--------|
| casey_chime_upchan.npy | `c6cdfa2812b79693215b889a87cee0d351a74cf4582783388378612fcfb23d73` |
| casey_chime_freq.npy | `045c266606d2364479a55c4c108fb0d2f46001062eabb27e311e77bc21836e33` |
| whitney_chime_upchan.npy | `a50048d9bcb8a0d1c6dae499b3bd8ab5f06ee18cfacc23b9963da4249eeafe81` |
| whitney_chime_freq.npy | `3367d19d3507f959ac2fbdf786c8f2b96448f1c10435a8570dd0396c9f421e0f` |
| phineas_chime_upchan.npy | `e77019ef756555996feb48cb585959129cabcf7163f43e678eb0a350fbdbb0f2` |
| phineas_chime_freq.npy | `38efd87c3935a986c98d56bb1e7184e02207e88c9dd71132e107fbb2d3bef9e9` |
| mahi_chime_upchan.npy | `e7f96fa18dcf67a3a6e1cae9a0ceba037595b8e1e0c8eb13cff0dffd8453a072` |
| mahi_chime_freq.npy | `bddf6341c97d77bfb7ada3e62a3a95509858304bd9484233cf2a2a0265f41590` |
| isha_chime_upchan.npy | `a4461e21f27172d60c2de373755f850f925a26c252e898eb4b2401fb54a8767d` |
| isha_chime_freq.npy | `0f43476df77e4edbcad3e91818002d917dd88a924eec788005e3eaabcdc588db` |

### Repo-local copies

casey products are pulled to `analysis/scattering-refit-2026-06/baseband_recovery/products/`
(gitignored — `*.npy`). The other four are on h17 only; pull with:

```bash
scp 'h17:/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/{whitney,phineas,mahi,isha}_chime_*.npy' \
    analysis/scattering-refit-2026-06/baseband_recovery/products/
```

Do **not** pull the gigabyte `.h5` baseband files — they stay on h17 (staged copies, not in repo).

### Reproducing from scratch

1. Ensure h17 has the docker image: `docker pull chimefrb/baseband-analysis:latest`.
2. Stage the worker: `scp upchannelize_chime.py h17:/data/jfaber/`.
3. Verify the 5 singlebeam `.h5` are staged under
   `/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam/` (or let the script
   `vcp` them from arc — requires `~/.ssl/cadcproxy.pem`).
4. Run the commands in "Exact run commands" above.
5. Verify each output: `n_fine ≥ 1024`, `df = 0.390625/U MHz`, `finite_frac > 0.3` (the script's
   built-in asserts enforce this; a clean exit means all gates passed).
