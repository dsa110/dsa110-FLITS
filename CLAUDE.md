# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FLITS (Fitting Likelihoods In Time-Frequency Spectra) fits pulse-broadening (scattering) and scintillation in FRB dynamic spectra. Telescope-agnostic; primary use is the CHIME–DSA co-detection analysis.

### Long-View Science Goals
- **Accurate Scattering (\(\alpha\)):** Simultaneously fit CHIME (400–800 MHz) and DSA-110 (1.2–1.5 GHz) to measure the shared scattering index \(\alpha\) using the \(\sim 1\) GHz lever arm.
- **Profile Bias Mitigation:** Model hidden temporal sub-components. Left unmodeled, they bias \(\alpha\) high (e.g. \(\alpha \approx 3.3 \to 2.7\) for `zach`).
- **Sightline Attribution:** Reconstruct DM and scattering budgets across the 12 co-detected sightlines, probing the CGM/groups/clusters of 49 candidate intervening systems.


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

Thresholds in code: `flits/fitting/VALIDATION_THRESHOLDS.py` (canonical, single source of truth).

## Code style: lazy-minimalist (ponytail)

<important if="you are writing, refactoring, or reviewing code in this repo">
Invoke the `ponytail` skill (level: full) and follow its ladder: does this need to exist at all (YAGNI) → stdlib → native / already-installed dep → one line → minimum that works. Shortest working diff, delete over add, no speculative abstractions or single-use config. This repo carries over-engineering worth resisting — the `flits/` wrapper over `scattering/`, several parallel `burstfit_*.py` variants.

Ponytail governs STRUCTURE and SIZE, not scientific rigor — the two are orthogonal. It does NOT relax the validation contract above: always run the PASS/MARGINAL/FAIL gates, generate diagnostic plots, keep calibration/physics knobs (the physical world needs tuning a minimal model can't see), and report failures explicitly. "Shortest diff" never means dropping a validation level or a diagnostic. In numerical code, derivation/"why" comments stay; only restate-what comments are cut.

On demand: `/ponytail-audit` (whole-repo bloat scan) · `/ponytail-review` (over-engineering review of a diff).
</important>

<important if="you are adding an import, fixture, or any new symbol in an Edit/Write">
A post-edit autoformatter reformats files after every Edit/Write (observed: it removed an import that was unused at edit-time, then it stayed once a consumer referenced it). So any import/symbol unused *at the moment that reformat runs* can be silently stripped — a later reference then fails with `NameError`. Add an import or definition in the SAME edit as its first consumer (or add the consumer first); never land an import-only edit ahead of the code that uses it. The verify-gate's mandatory test is the backstop that catches a strip before it ships.
</important>

## Tackling larger work

<important if="you are starting a substantial or multi-step task here — a fit campaign, a refactor, a new analysis surface — before you start editing">
Plan the approach before editing: name the target, the files in play, and how you'll prove it worked (which gate or diagnostic). Then execute end-to-end. Commit to one approach rather than narrating alternatives, and keep the diff minimal (the `ponytail` block governs size).
</important>

<important if="a task spans many bursts, many files, or many fits — a batch fit campaign, a cross-file refactor or migration, or a sweep over the results tree">
Use a dynamic workflow (say "use a workflow") instead of a serial pass, and pair it with the verification already built here. `.claude/workflows/fit-verify.js` adversarially re-checks every `*_fit_results.json` against the runtime gate with a *separate* judge agent (so a fit can't pass its own work). Pattern: `flits-batch` fans out the fits → the workflow verifies them in parallel. Set a completion condition with `/goal` so coverage is every burst, not a partial set, and run auto mode so it doesn't stall on permission prompts.
</important>

## Figure-review Stop gate (will block you)

`.claude/settings.json` registers a `Stop` hook (`.claude/hooks/figure-review-gate.sh`). Any dir with a `figures.manifest.json` newer than its `figures.review.json` blocks end-of-turn. To clear: actually **Read** each PNG (so it renders) and visually compare it to the manifest's stated expectation, then write `figures.review.json` with per-figure verdicts (`match` / `anomaly` / `skipped:<why>`). Fastest path: dispatch the `figure-reviewer` subagent on each unreviewed dir. A produced plot is never "validated" until it has been looked at.

## Deferred tasks gate completion (will block you)

A session shall not be completed while deferred tasks remain that the agent can execute or implement itself. Loose ends are tracked in `.agents/deferred-tasks.md` — a markdown checklist where every open `- [ ]` item carries exactly one tag:

- `@agent` — the agent can do it now → **blocks** end-of-turn until it is finished (`- [x]`).
- `@human` — needs a person or a one-way door (push / PR / publish) → does not block.
- `@decision` — a pending product/science choice → does not block.
- `@separate-lane` — belongs to another task's git lane → does not block.

`.claude/settings.json` registers a `Stop` hook (`.claude/hooks/deferred-task-gate.sh`) that blocks while any unchecked `@agent` item exists. To clear: **finish the work** and check it off, or — only if it genuinely cannot be done by the agent now — retag it `@human` / `@decision` / `@separate-lane`. Do not retag agent-doable work just to pass the gate; that defeats the policy. Add new follow-ups to the ledger as they arise, tagged honestly.

## Protected-branch commit guard (will block the commit)

`.claude/settings.json` registers a `PreToolUse` Bash hook (`.claude/hooks/no-commit-to-protected-branch.sh`) that **refuses `git commit` while `HEAD` is `main`/`master`**. Branch hygiene was otherwise prose-only, and `origin/main` already carries direct non-PR commits. To proceed: branch first (`git switch -c <feature-branch>`), then commit. The guard fails open when it cannot prove the branch is protected (not a repo, detached HEAD) and only sees the agent's own commits — an external auto-committer is a separate path.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on the `origin` fork (`jakobtfaber/dsa110-FLITS`), via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical vocabulary: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
