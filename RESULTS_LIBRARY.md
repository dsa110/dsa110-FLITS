# Results library (relocated)

Bulk products for this tree live under:

```text
$FABER2026_RESULTS_LIBRARY/dispersion/pipeline-results-root/
```

Default library root: `~/Data/Faber2026/results-library/`.

From the Faber2026 checkout:

```bash
python3 scripts/materialize_results_library.py
# creates pipeline/results → library slot
```

See `scripts/results_library_catalog.yaml` entry `dispersion.pipeline-results-root`.
