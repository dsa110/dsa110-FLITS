# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FLITS (Fitting Likelihoods In Time-Frequency Spectra) fits pulse-broadening (scattering) and scintillation in FRB dynamic spectra. Telescope-agnostic; primary use is the CHIME–DSA co-detection analysis.

## Commands

Conda env is `flits` (Python 3.12). `pip install -e ".[nested,perf]"` for extras; `galactic` extra (pygedm/NE2001) needs a manual macOS build — see `environment.yml`.

```bash
# Tests — pytest config + testpaths live in pyproject.toml (tests/, the scat/scint/batch test dirs)
pytest                                  # full suite
pytest -m "not slow"                    # skip slow-marked tests
pytest tests/test_foo.py::test_bar      # single test
nox                                     # tests in an isolated uv/virtualenv (installs .[nested,perf])

# Lint (ruff, line-length 100; defaults to noxfile.py only — pass paths to widen)
ruff check .
ruff format --check .
nox -s lint -- flits scattering          # lint a broader surface

# Entry points (also installed as console scripts)
flits-scat      # scattering.run_scat_analysis:main
flits-scint     # scintillation.scint_analysis.run_analysis:main
flits-batch     # flits.batch.cli:main  — run / generate-configs / joint-analysis / summary / export
flits-configs   # flits.batch.config_generator:main
```

`xfail_strict = true` and `--strict-markers` are set: an xfail that passes is a failure, and unknown pytest markers error.

## Architecture

Three analysis surfaces plus a shared package. The **canonical physics kernel** is `scattering/scat_analysis/burstfit.py` — `flits/` wraps it (e.g. `flits/models.py` imports `FRBModel` from there), so when changing model physics, edit `burstfit.py`, not the `flits/` wrapper.

- **`scattering/scat_analysis/`** — scattering pipeline. `burstfit.py` (`FRBModel`, `FRBFitter`, `FRBParams`, `build_priors` — emcee MCMC), `burstfit_modelselect.py` (`fit_models_bic`), `burstfit_robust.py` (sub-band consistency, leave-one-out influence), `burstfit_nested.py` (dynesty, optional), `burstfit_joint.py` (joint multi-telescope fits). `pipeline/` is the OO orchestrator (core/io/optimization/diagnostics). CLI: `scattering/run_scat_analysis.py`.
- **`scintillation/scint_analysis/`** — scintillation pipeline (ACF, 2D fitting, NE2001). CLI: `run_analysis.py`.
- **`simulation/`** — forward simulator (`engine.py`, `wave_optics.py`) + sim↔fit bridge (`sim_fit_bridge.py`) and validation scripts. Used to validate the fitter against known-truth injections.
- **`flits/`** — shared package: `batch/` (batch runner over many bursts → SQLite `results_db`, `joint_analysis`, summary plots), `fitting/` (diagnostics, `VALIDATION_THRESHOLDS`), `orchestration/`, plus thin model/param/sampler re-exports.
- **`galaxies/`**, **`crossmatching/`**, **`dispersion/`** — host-galaxy / multi-telescope cross-match (TOA + geometric delay) / DM-budget tooling.

### Scattering models (`FRBFitter`)
`M0` unresolved (delta source; DM fixed, β=2) → `M1` adds intrinsic width `zeta` → `M2` adds scattering tail `tau_1ghz` (α fixed, default 4.4) → `M3` full model (adds `alpha`, `delta_dm`). `mixed` fits multiple components each with its own model (params suffixed by component index, e.g. `tau_1ghz_2`). Model selection picks the winner by BIC.

### Data & metadata
- Raw data: `.npy`, shape `(n_freq, n_time)`, **frequency standardized to ascending** on load (`BurstDataset`). Large `.npy` are gitignored / external — see `DATA_LOCATIONS.md`, `DATA_SOURCES.md`, `data-manifest.csv`, `codetections_manifest.yaml`.
- Burst registry: `configs/bursts.yaml` (source of truth) + `chimedsa_burst_specs.csv` (nickname↔TNS). Internal **nicknames** (`casey`, `freya`) key filenames/configs; **TNS names** (`FRB 20240229A`) are for publication. Convert via `scattering.scat_analysis.burst_metadata`.
- Results: JSON (`best_params`, `best_key`, `goodness_of_fit`, `chain_stats`); batch results also in a SQLite DB.

## Fit validation is mandatory (do not rationalize fits)

`.cursor/rules/AGENT_CONFIGURATION_FLITS.md` is the binding contract for anyone writing/running fits here. Every fit gets a PASS/MARGINAL/FAIL flag from three levels; never declare a fit good without running them, and report failures explicitly rather than rationalizing.

- **Level 1 gates (any failure ⇒ FAIL, stop):** optimizer converged; physical bounds (0.0001 < τ < 100 ms; 1.5 < α < 6.0); Jacobian well-conditioned (cond < 1e6).
- **Level 2 quality:** χ²_red (good 0.8–1.5, fail >3 or <0.3), R² (good >0.85), residuals random/normal/uncorrelated (Durbin-Watson ≈2), parameter rel-err < 0.5.
- **Level 3 physics:** τ×Δν in [0.1, 2.0] (≈0.159 thin screen, ≈1.0 extended); α near 4.0 = Kolmogorov.

Thresholds in code: `flits/fitting/VALIDATION_THRESHOLDS.py`, `scattering/scat_analysis/validation_thresholds.py`.

## Code style: lazy-minimalist (ponytail)

<important if="you are writing, refactoring, or reviewing code in this repo">
Invoke the `ponytail` skill (level: full) and follow its ladder: does this need to exist at all (YAGNI) → stdlib → native / already-installed dep → one line → minimum that works. Shortest working diff, delete over add, no speculative abstractions or single-use config. This repo carries over-engineering worth resisting — the `flits/` wrapper over `scattering/`, several parallel `burstfit_*.py` variants.

Ponytail governs STRUCTURE and SIZE, not scientific rigor — the two are orthogonal. It does NOT relax the validation contract above: always run the PASS/MARGINAL/FAIL gates, generate diagnostic plots, keep calibration/physics knobs (the physical world needs tuning a minimal model can't see), and report failures explicitly. "Shortest diff" never means dropping a validation level or a diagnostic. In numerical code, derivation/"why" comments stay; only restate-what comments are cut.

On demand: `/ponytail-audit` (whole-repo bloat scan) · `/ponytail-review` (over-engineering review of a diff).
</important>

## Figure-review Stop gate (will block you)

`.claude/settings.json` registers a `Stop` hook (`.claude/hooks/figure-review-gate.sh`). Any dir with a `figures.manifest.json` newer than its `figures.review.json` blocks end-of-turn. To clear: actually **Read** each PNG (so it renders) and visually compare it to the manifest's stated expectation, then write `figures.review.json` with per-figure verdicts (`match` / `anomaly` / `skipped:<why>`). Fastest path: dispatch the `figure-reviewer` subagent on each unreviewed dir. A produced plot is never "validated" until it has been looked at.
