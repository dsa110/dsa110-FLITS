import pathlib

from galaxies.foreground.budget_table_emitter import format_budget_table_tex

ROOT = pathlib.Path(__file__).parents[1]


def test_emitter_produces_deluxetable():
    tex = format_budget_table_tex()
    assert r"\begin{deluxetable" in tex


def test_emitted_tex_matches_committed_copy():
    committed = (ROOT / "exports/budget_table.tex").read_text()
    assert format_budget_table_tex() == committed
