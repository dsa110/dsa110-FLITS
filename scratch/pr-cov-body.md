## What

Wires `pytest-cov` (coverage.py) as a **standing** tool so dead-code sweeps are repeatable.

- `pytest-cov>=4.0` in the `test` dependency-group.
- `[tool.coverage]` config: `source = flits/scattering/scintillation`, omit tests+scripts, `show_missing`. **No `fail_under` gate** — coverage here *nominates* dead code, it does not police CI.
- `nox -s cov` session mirroring the `tests` extras (`-m "not slow"` by default).

`pytest --cov` now works with no flags.

## Why

The `validation_thresholds` re-export shim removed in #40 was found via an ad-hoc `pytest --cov` run. This makes that sweep a one-liner (`nox -s cov`) instead of hand-passed flags.

A first sweep's 0%/low-coverage candidates were then run through an adversarial verification pass — all turned out to be **live** (runnable `__main__` diagnostic CLIs + a documented `burstfit_robust` diagnostic API with real importers), confirming the deliberate **no-`fail_under`** choice: in this repo low coverage flags *untested entry points*, not dead code.

## Verification

- `pytest --cov` renders a per-file report scoped to the three packages (config `source` resolves).
- `nox` sessions = `['tests', 'cov', 'lint']` (AST-parsed).
- `pyproject.toml` valid; `coverage.run.source` correct; `pytest-cov` present in `test` group.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GQbpq4cd1zoAEhKx3D6Krd
