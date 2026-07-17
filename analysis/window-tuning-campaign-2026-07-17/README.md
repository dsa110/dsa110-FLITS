# CHIME objective-window campaign gate closure

Reproduce from the repository root:

```bash
uv sync --frozen
FLITS_ROOT="$PWD" uv run python analysis/window-tuning-campaign-2026-07-17/run_injection_gate.py
FLITS_ROOT="$PWD" uv run python scintillation/scripts/run_window_campaign.py all analysis/window-tuning-campaign-2026-07-17/results
FLITS_ROOT="$PWD" uv run python scintillation/scripts/run_window_campaign.py all_hi analysis/window-tuning-campaign-2026-07-17/results
```

After rendering every entry in `results/figures.manifest.json`, write
`results/figures.review.json` following `docs/dev/figure-review-protocol.md`, then run
`finalize_campaign.py`. The finalizer never promotes a failed or inconclusive record.

