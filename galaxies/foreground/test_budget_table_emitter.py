"""Parity tests for the budget_table.tex emitter.

Three layers of protection against the value drift that motivated generating the
table (the DR8/DR9 mismatch caught in language_audit.md):

1. structural  — the emitter produces a deluxetable;
2. regression  — the emitter matches its committed canonical export
                 (exports/budget_table.tex); byte-exact, catches ANY change;
3. cross-check — the DM_host posterior column agrees, value-for-value, with the
                 independent forward model in scripts/dm_budget_uncertainty.py.

Layer 3 is the substantive one: it ties the manuscript column to a recomputable
source rather than a hand transcription.
"""
from __future__ import annotations

import csv
import json
import pathlib

import pytest

from galaxies.foreground.budget_table_emitter import (
    DATA_PATH,
    EXPORT_PATH,
    format_budget_table_tex,
)
from galaxies.foreground.sightline_budget import (
    format_budget_table_tex as reexport_tex,
)

# scripts/dm_budget_uncertainty.csv lives in the manuscript (super-)repo, two
# levels above the pipeline submodule root (…/Faber2026/pipeline/…).
_PIPELINE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POSTERIOR_CSV = _PIPELINE_ROOT.parent / "scripts" / "dm_budget_uncertainty.csv"


def test_emitter_produces_deluxetable():
    tex = format_budget_table_tex()
    assert r"\begin{deluxetable" in tex
    assert r"\startdata" in tex and r"\enddata" in tex


def test_reexport_matches_direct():
    """sightline_budget.format_budget_table_tex is the same output."""
    assert reexport_tex() == format_budget_table_tex()


def test_emitted_tex_matches_committed_export():
    """Regression anchor: emitter == exports/budget_table.tex (byte-exact).

    Regenerate with `python -m galaxies.foreground.budget_table_emitter`
    after any intentional change to budget_table_data.json or the markup.
    """
    assert EXPORT_PATH.exists(), (
        f"{EXPORT_PATH} missing — run "
        "`python -m galaxies.foreground.budget_table_emitter` to generate it."
    )
    assert format_budget_table_tex() == EXPORT_PATH.read_text()


def test_row_count_and_columns():
    data = json.loads(DATA_PATH.read_text())
    assert len(data["rows"]) == 12  # the twelve co-detections
    assert data["columns"][0] == "burst"


@pytest.mark.skipif(
    not _POSTERIOR_CSV.exists(),
    reason="dm_budget_uncertainty.csv not present (manuscript-repo artifact)",
)
def test_dm_host_matches_forward_model():
    """DM_host [median, +, -] in the table == p50/p84/p16 from the forward model.

    Placeholder-z sightlines (dm_host is null) carry no host posterior and are
    skipped; every other row must match the independently-computed CSV exactly.
    """
    posteriors = {}
    with _POSTERIOR_CSV.open() as fh:
        for row in csv.DictReader(fh):
            posteriors[row["burst"]] = row

    data = json.loads(DATA_PATH.read_text())
    checked = 0
    for r in data["rows"]:
        if r["dm_host"] is None:
            continue
        med, plus, minus = r["dm_host"]
        c = posteriors.get(r["burst"])
        assert c is not None, f"no forward-model row for {r['burst']}"
        p16, p50, p84 = (round(float(c[k])) for k in ("dm_host_p16", "dm_host_p50", "dm_host_p84"))
        assert (med, plus, minus) == (p50, p84 - p50, p50 - p16), (
            f"{r['burst']}: table {med}^+{plus}_-{minus} vs "
            f"forward-model {p50}^+{p84 - p50}_-{p50 - p16}"
        )
        checked += 1
    assert checked == 9  # nine non-placeholder sightlines
