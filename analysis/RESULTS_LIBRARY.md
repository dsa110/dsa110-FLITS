# Results library pointers

Campaign **result** trees are materialized under
`$FABER2026_RESULTS_LIBRARY` (default `~/Data/Faber2026/results-library/`).
Driver scripts stay in this `analysis/` tree.

| Campaign | Library slot |
|----------|--------------|
| `scattering-dm-locked-2026-07-14/results` | `scattering/2026-07-14_dm-locked` |
| `scattering-refit-2026-06/_a1_fits` | `scattering/2026-06_refit/_a1_fits` |
| `scattering-refit-2026-06/joint_json` | `scattering/2026-06_refit/joint_json` |
| `beta_campaign/fits` (+ verdicts JSON) | `scattering/2026-07_beta-campaign/` |
| `scintillation-dsa-lorentzian-2026-07-07/results` | `scintillation/2026-07-07_dsa-lorentzian` (DSA-only catalog; predates the CHIME objective-window rerun) |
| `window-tuning-campaign-2026-07-17/results` | committed in-tree (PR #192 final objective-window campaign; one `chromatica_hi` measurement, 23 diagnostic records) |
| `../.archive/outdated-science/2026-07-17/` | committed in-tree archive; obsolete science bytes retained for re-examination, never current claims |

Recreate local symlinks from the Faber2026 parent checkout:

```bash
python3 scripts/materialize_results_library.py
```

Catalog: `scripts/results_library_catalog.yaml` (parent repo).
