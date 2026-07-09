import csv
import pathlib

ROOT = pathlib.Path(__file__).parents[1]
TAB = ROOT / "crossmatching/dm_provenance.csv"

REQUIRED = {"nickname",
            "dm_dsa", "dm_dsa_err", "dm_dsa_method", "dm_dsa_source",
            "dm_chime", "dm_chime_err", "dm_chime_method",
            "dm_chime_source", "delta_dm", "delta_dm_sigma"}


def test_dm_provenance_covers_all_twelve():
    rows = list(csv.DictReader(TAB.open()))
    assert len(rows) == 12
    assert REQUIRED <= set(rows[0])
    assert all(r["dm_dsa_method"] and r["dm_chime_method"]
               for r in rows)
    assert all(r["dm_dsa_source"] and r["dm_chime_source"]
               for r in rows)
