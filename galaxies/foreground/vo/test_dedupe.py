import pandas as pd
import pytest

from .dedupe import deduplicate_candidates


def _row(**kwargs):
    base = {
        "frb_name": "FRB1",
        "ra": 10.0,
        "dec": 20.0,
        "z": 0.1,
        "z_type": "photo",
        "m_delta": None,
        "rank_key": 2.0,
        "catalog_id": "glade2",
        "service": "glade2",
        "provenance_json": "{}",
    }
    base.update(kwargs)
    return base


@pytest.mark.unit
def test_dedupe_merges_nearby_same_z_prefers_spec():
    df = pd.DataFrame(
        [
            _row(ra=10.0, z_type="photo", catalog_id="glade2", m_delta=1e12),
            _row(ra=10.0001, z_type="spec", catalog_id="hecate-v2", m_delta=None, rank_key=1.0),
        ]
    )
    out = deduplicate_candidates(df)
    assert len(out) == 1
    assert out.iloc[0]["z_type"] == "spec"
    assert "glade2" in out.iloc[0]["catalog_sources"]
    assert "hecate-v2" in out.iloc[0]["catalog_sources"]


@pytest.mark.unit
def test_dedupe_keeps_separate_galaxies_far_apart():
    df = pd.DataFrame(
        [
            _row(ra=10.0, catalog_id="glade2"),
            _row(ra=12.0, catalog_id="hecate-v2"),
        ]
    )
    out = deduplicate_candidates(df)
    assert len(out) == 2


@pytest.mark.unit
def test_dedupe_tolerates_missing_mass_and_rank_columns():
    df = pd.DataFrame(
        [
            {"frb_name": "FRB1", "ra": 10.0, "dec": 20.0, "z": 0.1, "catalog_id": "a"},
            {"frb_name": "FRB1", "ra": 10.0001, "dec": 20.0, "z": 0.1005, "catalog_id": "b"},
        ]
    )
    out = deduplicate_candidates(df)
    assert len(out) == 1


@pytest.mark.unit
def test_dedupe_is_per_sightline():
    df = pd.DataFrame(
        [
            _row(frb_name="A", ra=10.0, catalog_id="glade2"),
            _row(frb_name="B", ra=10.0001, catalog_id="hecate-v2"),
        ]
    )
    out = deduplicate_candidates(df)
    assert len(out) == 2
