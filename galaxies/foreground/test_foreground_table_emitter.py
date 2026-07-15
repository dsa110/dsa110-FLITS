"""Parity tests for the foreground_table.tex emitter.

Three layers, mirroring the budget-table emitter:

1. structural  — the emitter produces a deluxetable;
2. regression  — emitter output == committed exports/foreground_table.tex
                 (byte-exact; catches ANY change);
3. cross-check — every numeric object ID in the table carries the same verdict
                 as the census registry (data/intervening_census_registry.csv),
                 tying the manuscript table to the recomputable validation
                 product rather than a hand transcription.
"""
from __future__ import annotations

import csv
import json
import re

from galaxies.foreground.foreground_table_emitter import (
    DATA_PATH,
    EXPORT_PATH,
    REGISTRY_PATH,
    format_foreground_table_tex,
)


def test_emitter_produces_deluxetable():
    tex = format_foreground_table_tex()
    assert r"\begin{deluxetable" in tex
    assert r"\startdata" in tex and r"\enddata" in tex


def test_emitted_tex_matches_committed_export():
    """Regression anchor: emitter == exports/foreground_table.tex (byte-exact).

    Regenerate with `python -m galaxies.foreground.foreground_table_emitter`
    after any intentional change to foreground_table_data.json or the markup.
    """
    assert EXPORT_PATH.exists(), (
        f"{EXPORT_PATH} missing — run "
        "`python -m galaxies.foreground.foreground_table_emitter` to generate it."
    )
    assert format_foreground_table_tex() == EXPORT_PATH.read_text()


def test_rows_have_ten_cells():
    data = json.loads(DATA_PATH.read_text())
    # 23 tabulated physical census systems (28 rows minus the five confirmed
    # cross-listed duplicate pairs merged by the 2026-07-15 remediation)
    # + 3 explicit no-candidate sightline rows (FRB 20221113A, FRB 20230814B,
    # FRB 20240122A) so every co-detection sightline appears in the table.
    assert len(data["rows"]) == 26
    for r in data["rows"]:
        assert len(r) == 10, f"row has {len(r)} cells, expected 10: {r!r}"
    empties = [r for r in data["rows"] if r[8] == "no candidates"]
    assert [r[0] for r in empties] == [
        "FRB 20221113A",
        "FRB 20230814B",
        "FRB 20240122A",
    ]
    merged = [r for r in data["rows"] if "," in r[2] and not r[2].startswith("J")]
    assert len(merged) == 5  # the five dual-listed confirmed systems


def test_verdicts_match_census_registry():
    """Every numeric object ID in the table agrees with the registry verdict.

    Cluster-catalog IDs (WenHan2024, e.g. the FRB 20230307A cluster) are not in
    the intervening-halo registry and are skipped. Every registry-resident row
    must match exactly.
    """
    assert REGISTRY_PATH.exists(), f"{REGISTRY_PATH} missing"
    reg = {}
    with REGISTRY_PATH.open() as fh:
        for row in csv.DictReader(fh):
            reg[row["obj"]] = row["final_verdict"]

    data = json.loads(DATA_PATH.read_text())
    checked = 0
    for cells in data["rows"]:
        obj_field, verdict = cells[2], cells[8]
        for tok in re.split(r"[,\s]+", obj_field):
            tok = tok.strip()
            if not tok.isdigit() or tok not in reg:
                continue  # cluster-catalog id or non-registry token
            assert reg[tok] == verdict, (
                f"obj {tok}: table '{verdict}' vs registry '{reg[tok]}'"
            )
            checked += 1
    assert checked == 27  # 27 of 28 rows are registry-resident intervening halos
