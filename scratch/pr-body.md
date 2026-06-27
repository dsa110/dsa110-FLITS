## What

Removes a dead re-export shim found via a `pytest-cov` dead-code sweep.

- **Delete** `scattering/scat_analysis/validation_thresholds.py` — a 1-line `from flits.fitting.VALIDATION_THRESHOLDS import *` re-export with **zero importers** anywhere in the repo (0% coverage).
- **Doc** `CLAUDE.md` — name `flits/fitting/VALIDATION_THRESHOLDS.py` as the single source of truth (the shim's removal makes the old "thresholds live in two places" note stale).

## Why

Coverage sweep (`pytest --cov`) flagged the shim at 0%. It carries no unique content — pure forward of the canonical module.

## Verification

Independent Codex adversarial review (read-only, gpt-5.5 high) tried to refute "safe to delete" and confirmed it:

- No static import of `scattering.scat_analysis.validation_thresholds` (absolute or relative) in code/tests/notebooks/configs.
- No dynamic/string reference (`importlib`, `__import__`, entry_points/console_scripts, plugin registries).
- Not re-exported via `scattering/scat_analysis/__init__.py`.
- Runtime export equality vs canonical: `shim_minus_canon=[]`, `canon_minus_shim=[]`, `value_mismatches=[]` (36 == 36 constants).

Post-deletion: `import scattering.scat_analysis` clean; `pytest -k "valid or threshold or quality or classify"` → **39 passed**.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GQbpq4cd1zoAEhKx3D6Krd
