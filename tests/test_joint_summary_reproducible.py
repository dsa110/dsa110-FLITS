"""Drift guard: results/joint_fit_summary.md must be byte-reproducible from the
committed joint-fit JSONs via the in-repo generator. Fails if a JSON or the
generator changed without regenerating the summary."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "analysis" / "scattering-refit-2026-06" / "gen_joint_summary.py"
SUMMARY = REPO / "results" / "joint_fit_summary.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_joint_summary", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_matches_generator_output():
    gen = _load_generator()
    assert SUMMARY.read_text() == gen.render(), (
        "results/joint_fit_summary.md is stale — re-run "
        "`python analysis/scattering-refit-2026-06/gen_joint_summary.py`"
    )
