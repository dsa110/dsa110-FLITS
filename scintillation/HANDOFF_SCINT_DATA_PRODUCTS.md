# HANDOFF — Generate all data products needed to complete the scintillation analysis

**Date:** 2026-07-07. **Author:** audit session (Claude + JF).
**Goal:** produce a pipeline-ready dynamic-spectrum npz for **every burst × both bands**
(12 CHIME + 12 DSA = 24 products), so the shared ACF pipeline
(`scint_analysis/pipeline.py`, `analyze_scintillation` / `analyze_scintillation_from_acfs`)
can be run uniformly across the full sample. Same algorithm both bands: frequency ACF →
Lorentzian / gen-Lorentzian / composite fits, BIC + nested-F component selection,
sub-band Δν_d(ν) power law → α, NE2025 Galactic floor.

> ✓ **Path update applied (2026-07-08).** `DATA_PROVENANCE.md` previously pointed at
> the stale `h17:/data/jfaber/` staging tree, which was emptied ~2026-06-27 by the
> arc_cleanup/migrate scripts. It has since been updated: everything now lives under
> `h17:/data/research/astrophysics/frbs/chime-dsa-codetections/` (call it `$COD` below),
> and `DATA_PROVENANCE.md` carries a "2026-07-07/08 staging update" section, the
> recovered oran/johndoeII event IDs, the local Mac inventory, and the CANFAR-pull
> provenance. Both documents now agree on `$COD`; either can be trusted on paths.

---

## 1. The sample (12 bursts)

| nickname | TNS | CHIME event ID | DSA date code |
|---|---|---|---|
| zach | FRB 20220207C | 210456524 | 220207aabh |
| whitney | FRB 20220310F | 215063905 | 220310aaam |
| oran | FRB 20220506D | **224263996** (recovered from filterbank dir name) | 220506aabd |
| isha | FRB 20221113A | 252069198 | 221113aaao |
| wilhelm | FRB 20221203A | 253635173 | 221203aaaa |
| phineas | FRB 20230307A | 274819243 | 230307aaao |
| freya | FRB 20230325A | 278720455 | 230325aaag |
| johndoeII | FRB 20230814B | **311723353** (recovered) | 230814aaas |
| hamilton | FRB 20230913A | 318353610 | 230913aaao |
| mahi | FRB 20240122A | 354049284 | 240122aaag |
| chromatica | FRB 20240203A | 356959136 | 240203aacl |
| casey | FRB 20240229A | 362593221 | 240229aaad |

(oran and johndoeII IDs were the two missing entries in DATA_PROVENANCE.md §7b — resolved
by `$COD/dsa_filterbanks/Codetections_DSA_Filterbanks/<nick>_<datecode>_<eventid>/` naming.)

---

## 2. Current inventory (verified on disk 2026-07-07)

### CHIME band
- **h17, complete for all 12:** `$COD/upchan_codetections/{nick}_chime_upchan.npy` +
  `{nick}_chime_freq.npy` + `{nick}_time0_metadata.json`, regenerated 2026-07-06/07
  (`regen_20260706.log`, `regen_batch2_20260707.log`). Quarantined bad runs:
  `DEFECTIVE_nodedisp_20260703/`, `SUPERSEDED_timeshift_20260704/` — do NOT use.
- **Local Mac, packaged npz for 6/12:** `~/Data/Faber2026/dsa110/scintillation-data/`
  has `{casey,freya,isha,mahi,phineas,whitney}_chime.npz` (+ `_hi.npz`), built 2026-07-06.
  **Missing locally: chromatica, hamilton, wilhelm, zach, johndoeII, oran.**
- Packaging scripts + provenance: `~/Data/Faber2026/dsa110/upchan_codetections/`
  (`build_npz_aligned_generic_20260706.py`, `PROVENANCE.md`).

### DSA band
- **Local packaged npz: 1/12** — `~/Data/Faber2026/dsa110/scintillation/data/freya.npz`
  (57 MB, 2026-07-04; symlinked into `scintillation-data/`). Other 11 absent from Mac.
- **CANFAR (authoritative): all 12** — `$ARC/FLITS/scintillation/data/{nick}.npz`,
  `$ARC = arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/`.
- **h17 raw filterbanks: all 12 (+ extras)** —
  `$COD/dsa_filterbanks/Codetections_DSA_Filterbanks/<nick>_<datecode>_<id>/<datecode>_dev_polcal_I.fil`.
- **Local raw waterfalls: all 12** — `~/Data/Faber2026/dsa110/DSA_bursts/{nick}_dsa_I_*_2500b_cntr_bpc.npy`
  (centered, bandpass-corrected Stokes-I; preprocessing lineage vs CANFAR npz unverified).
- **Pipeline-ready DSA ACF products (3):**
  `h17:$COD/archive/arc_trash_2026-06/acf_results/{chromatica,freya,wilhelm}_acf_results.pkl`
  (⚠ sitting in an `arc_trash` dir — copy out before any further cleanup deletes them).

### Fit products already committed in repo
- `configs/bursts/freya_dsa.yaml`, `wilhelm_dsa.yaml` — `stored_fits` (keep; will be
  re-validated, not regenerated).
- `chime_acfs/{chromatica,freya,hamilton,wilhelm}_{id}_subband_acf_fits.pkl` — legacy
  fit products (`1_lorenz`/`2_lorenz` schema), NOT re-fittable under the BIC selector;
  superseded by the fresh upchannelization. Keep for cross-checks only.
- `freya_analysis_results.json` — **degenerate** (bw_at_ref=3.84e12 MHz, α=4.0±0.0,
  empty subbands). Do not cite; regenerate.

---

## 3. DSA npz: pull from CANFAR (recommended) — NOT regenerate from .fil

**Decision: `vcp` the 11 missing npz from CANFAR.** Rationale:
1. The CANFAR npz are the exact files the 12 `configs/bursts/{nick}_dsa.yaml` already
   point to (`${FLITS_ROOT}/scintillation/data/{nick}.npz`) — schema guaranteed to match
   what the pipeline consumes, and consistent with the already-committed freya/wilhelm fits.
2. Regenerating from `.fil` would require reverse-engineering the original preprocessing
   (on-pulse selection, RFI mask, bandpass) and would create a second DSA lineage —
   exactly the legacy-vs-new fork we're trying to eliminate on the CHIME side.
3. Transfer cost trivial: ~12 × O(50–120 MB).

Fallback only if CANFAR is unreachable: package `.fil` (h17) or the local
`DSA_bursts/*.npy` via a build script, then **validate against `freya.npz`**
(same shape/keys/values) before trusting.

**How (verified pattern, run from h17 inside the baseband container with the CADC proxy):**
```bash
ssh h17
docker run --rm -it \
  -v /data/research:/data/research \
  -v ~/.ssl:/root/.ssl:ro \
  chimefrb/baseband-analysis:latest bash
# inside container:
for n in casey whitney phineas mahi isha chromatica hamilton wilhelm zach johndoeII oran; do
  vcp arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/FLITS/scintillation/data/$n.npz \
      /data/research/astrophysics/frbs/chime-dsa-codetections/results/dsa_scint_npz/
done
```
Then rsync `dsa_scint_npz/` to the Mac →
`~/Data/Faber2026/dsa110/scintillation/data/` (where `freya.npz` already lives).

**Immediately also:** copy the 3 `*_acf_results.pkl` out of
`archive/arc_trash_2026-06/acf_results/` into `results/` (they are one cleanup pass
away from deletion).

---

## 4. CHIME npz: package the remaining 6 bursts

Inputs already exist on h17 (all 12 regenerated Jul 6–7). Local packaging pipeline
already worked for 6 bursts on Jul 6 — repeat for the rest:

1. Pull raw upchan products for **chromatica, hamilton, wilhelm, zach, johndoeII, oran**:
   ```bash
   for n in chromatica hamilton wilhelm zach johndoeII oran; do
     rsync -avP h17:/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/${n}_chime_{upchan,freq}.npy \
                h17:/data/research/astrophysics/frbs/chime-dsa-codetections/upchan_codetections/${n}_time0_metadata.json \
                ~/Data/Faber2026/dsa110/upchan_codetections/
   done
   ```
2. Package with `~/Data/Faber2026/dsa110/upchan_codetections/build_npz_aligned_generic_20260706.py`
   (read its `PROVENANCE.md` first; it synthesizes `times_s` from time0 metadata —
   this is the fix for the DEFECTIVE/SUPERSEDED runs). Output
   `{nick}_chime.npz` (+ `_hi.npz`) → `~/Data/Faber2026/dsa110/scintillation-data/`.
3. Sanity check each npz against `casey_chime.npz` (same keys, freq axis 400–800 MHz,
   expected upchan resolution 0.390625/U MHz).

---

## 5. Configs to create/verify (repo: `pipeline/scintillation/configs/bursts/`)

- DSA: all 12 `{nick}_dsa.yaml` exist. Verify each `data_path` resolves to the
  newly-staged npz (set `FLITS_ROOT` or edit paths). Do NOT clobber `stored_fits`
  in freya/wilhelm.
- CHIME: only `freya_chime{,_hi}.yaml` and `casey_chime{,_hi}.yaml` exist.
  **Create the other 10** by templating `casey_chime.yaml`: telescope `chime`,
  `reference_frequency_mhz: 600`, `max_lag_mhz: 5.0`, `fit_lagrange_mhz: 1.0`,
  `num_subbands: 4` (adjust per SNR), and **keep the harmonic mask**
  (`harmonic_mask: spacing_mhz: 0.390625`) — it removes the coarse-channel comb
  from upchannelization. Per-burst upchan factor U differs (16–512) — read it from
  the upchannelize logs / metadata and record it in the yaml comment.
- isha: previous run was `--run-unresolvable` (upper bound). Decide per new-data SNR
  whether it stays an upper bound; don't silently promote it.

---

## 6. Definition of done

- [ ] 12 DSA npz staged locally (11 pulled + freya) and readable by
      `scint_analysis` DynamicSpectrum loader.
- [ ] 12 CHIME npz local (6 existing + 6 newly packaged), same schema.
- [ ] 3 DSA `acf_results.pkl` rescued out of `arc_trash_2026-06`.
- [ ] 24 configs resolve and `flits-scint --config <yaml> --dry-run` (or equivalent
      loader smoke test) passes for every burst × band.
- [x] `DATA_PROVENANCE.md` updated (2026-07-08): new h17 root `$COD`, recovered
      oran/johndoeII IDs, local Mac inventory (§2 above), CANFAR-pull provenance for
      the 11 DSA npz.
- [ ] Nothing fitted yet — fitting/re-validation is the NEXT campaign step
      (per `sections/results.tex` validation contract). This handoff is data-staging only.

## 7. Known traps
- Anything under `DEFECTIVE_nodedisp_20260703/` or `SUPERSEDED_timeshift_20260704/`
  (both h17 and Mac) is poison — dedispersion/time-axis bugs.
- Legacy `chime_acfs/*.pkl` cannot go through the BIC selector; don't try.
- `freya_analysis_results.json` is degenerate; regenerate, don't cite.
- CANFAR access requires the `cadcproxy.pem` at `h17:~/.ssl/` inside the
  `chimefrb/baseband-analysis:latest` container; check cert expiry if vcp auth fails.
- h17 disk at 87% (11T/13T) — stage DSA npz (~1 GB) is fine, but don't duplicate
  the singlebeam archive.
