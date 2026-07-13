# reference_arc — verbatim capture of the CANFAR-era scintillation analysis originals

Captured 2026-07-12 (PDT) from two sources, following the
`scattering/scat_analysis/builders_arc/` precedent: these are the
collaborator-era scripts and worked notebooks that contain the working
CHIME recipe — up-channelize → clean/RFI-excise → ACF → Lorentzian fit →
scintillation bandwidth — whose step ordering and cleaning choices are not
documented anywhere else. Methodology lineage: Kenzi Nimmo's analysis
sequence; `code/analysis-Copy1.py` cites the Nimmo scintillation paper
(bib entry `nimmo2025` in `../references.bib`), and
`code/frb_scintillator_wAnisotropy-Copy1.py` is the redshift-aware
two-screen simulator (bib entry `pradeep2025`, ibid.).
Several notebooks are grep-positive for the Nimmo attribution.

**Files are captured verbatim — do not edit in place.** Port logic into
`scint_analysis/` proper; treat this directory as read-only evidence.
`SHA256SUMS` in this directory covers every captured file.

## Sources

1. **h17 arc-trash rescue** (`h17:/data/research/astrophysics/frbs/chime-dsa-codetections/archive/arc_trash_2026-06/`),
   itself the 2026-06 rescue of the arc VOSpace trash. Pulled 2026-07-12 via
   scp; all 19 files sha256-verified against the h17 originals (0 mismatches).
   A second copy of the same tree exists at
   `iacobus:~/Research/CHIME_DSA_Codetections/archive/arc_trash_2026-06/`
   (D3 rsync 2026-06-27) — not independently re-verified here.
   - `code/` — all 11 `*.py` from `arc_trash_2026-06/code/`:
     `scinttools_old.py`, `scinttools_new.py`, `scinttools_v3.py`
     (refactor chain; v3 docstring: ACF computation + Lorentzian
     scintillation-bandwidth fitting), `frb_scintillator_wAnisotropy-Copy1.py`,
     `baseband_analysis_core.py` / `baseband_analysis_analysis.py`
     (CHIME baseband upchan/cleaning layer), `burstfit_subband.py`,
     `burstfittools.py`, `burstfit_utils.py`, `analysis-Copy1.py`,
     `untitled.py`.
   - `notebooks/` — the 8 `scint_*.ipynb` from `arc_trash_2026-06/notebooks/`:
     casey (empty stub, kept verbatim), chromatica (+`_v2`),
     freya (+`-Copy1`), hamilton, wilhelm (+`_v2`). These are the per-burst
     worked ACF→fit sequences.
2. **arc live home** (`arc:home/jfaber/`), NOT part of the trash rescue —
   pulled 2026-07-12 via `vcp` (CADC cert valid to 2026-07-18; VOSpace
   exposes no MD5 node property, so the SHA256SUMS entries are the
   post-download provenance hashes):
   - `arc_home/scint_freya_trash.ipynb`
   - `arc_home/scint_chromatica_trash.ipynb`

## Not captured (deliberately)

- `arc:home/jfaber/burst_search/` (`frb-ops`, `L4_databases`) — CHIME ops
  tooling, not scintillation analysis.
- The other 55 `arc_trash_2026-06/notebooks/*.ipynb` (scattering/DM/TOA
  work) — remain in the h17 + iacobus archive copies; pull on demand.

## Why this exists

`DATA_PROVENANCE.md` §7c flags that rediscovered historical CHIME products
are retired context "unless their preprocessing provenance is
reconstructed". This capture is that reconstruction path: the recipe
deltas between these originals and the current `scint_analysis/` pipeline
(cleaning steps, ACF windowing/lag selection, fit form, modulation-index
handling) are the input to the CHIME γ / modulation-index campaign.

## Addendum — capture 2 (arc_live/, same day)

A second sweep of the **live** arc tree
(`arc:home/jfaber/baseband_morphologies/chime_dsa_codetections/`) found the
canonical Nimmo pipeline itself, absent from the trash rescue. Captured
2026-07-12 via `vcp` into `arc_live/` (hashes: `SHA256SUMS.arc_live`;
`.ipynb_checkpoints` excluded):

- `Nimmo_scripts/` — the named source: `scint_pipe_new_Nimmo2025.py` (76K),
  `scintillation_funcs_new_Nimmo2025.py`, `scint_pipeline_tutorial_new_Nimmo2025.ipynb`
  (worked tutorial), `scint_toolkit.py`/`.ipynb`, `run_scint_pipe.py`,
  `scint_analysis.ipynb`, `subband_results.png`.
- `scint_toolkit/` — `scint_toolkit_v1.py`, `scint_toolkit_v2.py` (the
  version lineage RECIPE.md §7.5 flagged as missing).
- `scattering_root/` — `prep_data.py`, `make_acf.py`, `fit_acf.py`,
  `burstfittools.py`, `run_scint_pipe.ipynb`; `scint_pipe.py` is 0 bytes
  **on arc itself** (kept as evidence).
- `old_scattering_scintillation/` — complete worked zach example
  (`measure_scintillation.ipynb`, `test_upchan_spec.ipynb`, ACF npz,
  scintbw/modind/subband-fit PNGs) + `utilities/`: **`kenzie_funcs.py`,
  `kenzie_functions.py`, `scint_funcs.py`** (the exact `scint_funcs` module
  `scint_freya-Copy1.ipynb` imports), `scint_functions.py`,
  `scint_utilities.py`, `scint_utils_compiled.py`, `upchan_spec.py`.

Resolves RECIPE.md ambiguity #5: the offset-bearing fit function the
notebooks bound at runtime exists here — `kenzie_funcs.py:446`
`lorentz(x, gamma, m, c)`; `scint_funcs.py:269/275` carries both
`lorentz_w_c` and no-offset `lorentz` variants.

Still not captured: `scattering/scint_analysis/` package on arc (ancestor of
this repo's `scint_analysis/`; reachable in git history), `simulation/`
(Pradeep simulator variants; one copy already in `code/`), and the
`OLD_scattering/scintillation/` bulk PNG set beyond the zach exemplars.
