# Holoviz Evaluation for FLITS Batch-Results Exploration

**Question**: Should FLITS adopt HoloViz ecosystem tools (hvPlot, HoloViews, Datashader, Panel) for batch-results exploration?

**Verdict**: Yes, selectively — as an **additive exploratory layer**, not a replacement for the existing matplotlib validation figures. The fit is narrow: hvPlot/HoloViews/Panel for the batch DB; Datashader only for dynamic spectra and full MCMC chains.

## Current state

- `flits/batch/results_db.py` — SQLite with `scattering_results` + `scintillation_results` tables; `to_dataframe()` + `export_latex_table()`.
- `flits/batch/summary_plots.py` — static matplotlib figures: sample overview, tau/deltanu/alpha distributions, burst comparison grid, tau–deltanu scatter. Publication-quality; gated by `figure-review-gate.sh`.
- Per-fit JSON (`*_fit_results.json`): `best_params`, `goodness_of_fit` (chi2_reduced, R², residual_autocorr, durbin_watson, normality_pvalue), `convergence.ncall` (emcee per-step calls — ~2.3×10⁴ entries observed), `all_results` (per-model BIC).

## Data-size regime (decides which tool fits)

| Surface | Size | Right tool |
|---------|------|-----------|
| Batch results DB | ~24 rows (12 bursts × 2 telescopes) | hvPlot / HoloViews — tiny, interactive is free |
| Per-fit goodness-of-fit scalars | ~10 metrics × 24 runs | hvPlot bars/scatter |
| MCMC `convergence.ncall` arrays | ~2.3×10⁴ entries per burst (23475, 22909 observed) | HoloViews; Datashader only if overlaying all 12 bursts (~280k points) |
| Dynamic spectra `(n_freq, n_time)` | up to ~32000 × ~256 ≈ 8M pixels | **Datashader** — this is its regime |
| Residual autocorrelation | ~30–50 lags | hvPlot |

Datashader's 100M+ point regime is only hit if you render multiple dynamic spectra or full chains simultaneously. Single-spectrum renders are borderline — HoloViews + Datashader rasterization is still the right call for faithful large-array display (matplotlib's `imshow` on 8M pixels is slower and less honest about downsampling).

## Where it helps the science goals

1. **α-measurement (CHIME–DSA lever arm)**: interactive hvPlot scatter of `alpha` vs `freq_min_ghz`/`freq_max_ghz` per burst, colored by `best_model` and sized by `chi2_reduced`. Lets you spot which bursts are pulling the shared-α fit and whether M3 (full model) vs M2 (α fixed) tracks the 1 GHz lever arm differently across the 12 sightlines.
2. **Sightline attribution**: Panel dashboard joining `scattering_results` to `galaxies/` + `crossmatching/` outputs — partition τ and DM across host/MW/foreground for the 49 candidate intervening systems. The existing `results/galaxy_sightlines_report.html` is static; a Panel version with linked brushing across sightlines is the natural upgrade.
3. **Fit-quality triage**: linked views of `chi2_reduced` / `gelman_rubin_max` / `acceptance_fraction` across the batch, with click-to-load the diagnostic figure for a burst. Pairs with the figure-review stop gate — interactive triage of which bursts need figure review first.

## What NOT to do with holoviz here

- **Do not replace `summary_plots.py`** — those are publication-quality and figure-review-gated. Holoviz outputs are exploratory, not publication artifacts.
- **Do not put holoviz figures through the figure-review stop gate** — the gate keys on `figures.manifest.json` + PNG review; holoviz produces interactive HTML/JS, not PNGs. Keep the two surfaces separate.
- **Do not add holoviz as a hard dependency** — it's a heavy stack (panel, bokeh, datashader, holoviews, dask). Make it an optional extra (`pip install -e ".[exploratory]"`) so the core fit/validation pipeline stays lean (per the ponytail principle).

## Recommended minimal integration

1. Add optional extra in `pyproject.toml`:
   ```toml
   exploratory = ["hvplot>=0.10", "holoviews>=1.18", "panel>=1.4", "datashader>=0.16"]
   ```
2. One module: `flits/batch/exploratory.py` — `load_results_df()` (wraps `ResultsDatabase.to_dataframe`), `alpha_leverarm_scatter()`, `fit_quality_overview()`, `sightline_dashboard()`. hvPlot one-liners backed by the existing DataFrame; Panel for the multi-panel dashboard.
3. CLI hook: `flits-batch explore` → `panel serve` the dashboard over the batch DB.
4. No changes to `summary_plots.py`, `results_db.py`, or the figure-review gate.

## Decision

Adopt hvPlot + HoloViews + Panel as an optional `[exploratory]` extra for batch-results exploration. Adopt Datashader only for dynamic-spectrum and full-chain rendering. Keep it out of the validation/gated-figure path. Defer until the batch DB is actually populated (currently 2 bursts) — building the exploratory layer before there's a batch to explore is premature (YAGNI, per ponytail).

## Cross-references

- Upstream skill: `uw-ssec/rse-plugins` → `community-plugins/holoviz-visualization` (panel-dashboards, plotting-fundamentals, data-visualization, advanced-rendering skills). Vendor those SKILL.md files into `.claude/skills/` if/when this integration is built.
- Validation contract: `.cursor/rules/AGENT_CONFIGURATION_FLITS.md` (PASS/MARGINAL/FAIL gates — orthogonal to this exploratory layer).
- Ponytail: this eval biases toward the smallest viable integration; revisit if the batch grows beyond ~50 bursts.
