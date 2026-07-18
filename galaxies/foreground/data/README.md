# Foreground data artifacts (SSOT)

| File | Role |
|------|------|
| `intervening_census_registry.csv` | 49-object validated census + `budget_eligible` |
| `tau_consistency_catalog.csv` | Dual-τ track: free-α joint + α=4 consistency refit status |
| `../../../../quarantine/2026-07-17-outdated-science/galaxies/foreground/data/sightline_attribution_matrix.csv` | Quarantined historical two-screen × foreground cross-check; not current science |

Regenerate:

```bash
python -m galaxies.foreground.build_artifacts
```

α=4 joint refits (needs FLITS_RUNS data on HPC):

```bash
python -m galaxies.foreground.run_tau_consistency_refits casey
```

Outputs land in `tau_consistency/*.json` and refresh the catalog via `build_artifacts`.
The historical attribution matrix is regenerated only into the dated
quarantine's `regenerated/` subtree; the path-preserving snapshot is immutable.
