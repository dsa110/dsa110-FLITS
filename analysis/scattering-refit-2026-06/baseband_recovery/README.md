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
    --out /data/jfaber/upchan_codetections'

# 3. mahi (slow, x512 via _upchannel); isha only as an upper bound
ssh h17 'docker run --rm -e HOME=/root -v /home/ubuntu/.ssl:/root/.ssl:ro -v /data/jfaber:/data/jfaber \
  chimefrb/baseband-analysis:latest \
  python /data/jfaber/upchannelize_chime.py mahi --out /data/jfaber/upchan_codetections'
# isha:  ... upchannelize_chime.py isha --run-unresolvable ...
```

`vcp` may need the cert at the vos default (`$HOME/.ssl/cadcproxy.pem`); `HOME=/root` + the mount above
satisfies that. If the container runs as non-root, mount to that user's `~/.ssl` instead.

## Return the small products

Each target writes `<name>_chime_upchan.npy` (Stokes-I, `(n_freq, n_time)`, float32) +
`<name>_chime_freq.npy` (ascending MHz), both kB–MB. Pull them to the repo:

```bash
scp 'h17:/data/jfaber/upchan_codetections/*_chime_*.npy' \
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

## Empirical result so far — casey (the best case) FAILS the contract

casey was upchannelized end-to-end on h17 (U=16, `_upchannel` direct path): `casey_chime_upchan.npy`
= (12336, 1748), df=24.414 kHz exact, 92.2% finite. Converted to `scintillation/data/casey_chime.npz`
and fit through the FLITS scint pipeline. Both the **full band** (`casey_chime.yaml`) and a
**high-band focus** slice (`casey_chime_hi.yaml`, 711–799 MHz where a crude ACF looked cleanest) were
run:

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
narrower predicted scales) are very unlikely to fare better. Per the gating plan, the other targets
were **not** upchannelized. The honest conclusion for task #3: CHIME diffractive-scintillation recovery
for these co-detection sightlines is **not contract-feasible** at the resolution/SNR available — the
skeptic verdict (above) borne out by data. casey stands as the empirical demonstration.

## What this plan deliberately does NOT do

Only casey has been upchannelized + fit (above). whitney/phineas/mahi/isha were not run (casey's FAIL
made proceeding unjustified) and no gigabyte `.h5` files were pulled to the repo. No commit.
