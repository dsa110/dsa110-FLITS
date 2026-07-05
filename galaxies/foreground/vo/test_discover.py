from unittest.mock import Mock

import pandas as pd
import pytest

from . import discover as dmod
from .discover import _is_probable_z, _normalize_service_url, discover_tables, find_columns


@pytest.mark.unit
def test_is_probable_z_positive_cases():
    assert _is_probable_z("col1", "", "src.redshift") == (True, "ucd")
    assert _is_probable_z("z", "", "") == (True, "name")
    assert _is_probable_z("z_spec", "", "") == (True, "name")
    assert _is_probable_z("redshift", "", "") == (True, "name")
    assert _is_probable_z("cz", "", "") == (True, "velocity")


@pytest.mark.unit
def test_is_probable_z_negative_cases():
    assert _is_probable_z("access_estsize", "", "") == (False, "excluded_name")
    assert _is_probable_z("size", "", "") == (False, "excluded_name")
    assert _is_probable_z("obscore_z", "", "") == (False, "excluded_name")
    assert _is_probable_z("XZ80Q", "", "") == (False, "excluded_name")  # random ID pattern


@pytest.mark.unit
def test_normalize_service_url():
    assert _normalize_service_url("https://test.com/tap") == "https://test.com/tap"
    assert _normalize_service_url("https://test.com/tap/sync") == "https://test.com/tap"
    assert _normalize_service_url("https://test.com/tap?") == "https://test.com/tap"
    assert _normalize_service_url("http://test.com/tap") == "https://test.com/tap"
    assert _normalize_service_url("not-a-url") is None
    assert _normalize_service_url("") is None


@pytest.mark.unit
def test_find_columns_offline_returns_none(tmp_path):
    # Unresolvable endpoint: metadata lookups fail, all picks come back None
    out = find_columns("https://example.invalid/tap", "some.table")
    assert {"ra_col", "dec_col", "z_col"} <= set(out.keys())
    assert out["ra_col"] is None and out["dec_col"] is None and out["z_col"] is None


@pytest.mark.unit
def test_discover_tables_invalid_endpoint_empty(tmp_path):
    df = discover_tables("http://invalid.invalid/tap", limit=10, cache_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["table", "ra_col", "dec_col", "z_col"]
    assert len(df) == 0


@pytest.mark.unit
def test_discover_tables_success(monkeypatch, tmp_path):
    # patch by module object (galaxies/ is a namespace dir; see test_cone_query_success)
    mock_safe_search = Mock()
    monkeypatch.setattr(dmod, "TAPService", Mock())
    monkeypatch.setattr(dmod, "safe_search", mock_safe_search)
    prefilter_response = pd.DataFrame({"table_name": ["galaxy.main"]})
    columns_response = pd.DataFrame(
        {
            "table_name": ["galaxy.main"] * 3,
            "name": ["ra", "dec", "z_spec"],
            "ucd": ["pos.eq.ra", "pos.eq.dec", "src.redshift"],
            "description": ["Right ascension", "Declination", "Spectroscopic redshift"],
        }
    )
    z_sample = pd.DataFrame({"z": [0.1, 0.2, 0.3]})
    mock_safe_search.side_effect = [prefilter_response, columns_response, z_sample]

    result = discover_tables("https://test.com/tap", limit=10, cache_dir=tmp_path)
    assert len(result) == 1
    assert result.loc[0, "table"] == "galaxy.main"
    assert result.loc[0, "ra_col"] == "ra"
    assert result.loc[0, "dec_col"] == "dec"
    assert result.loc[0, "z_col"] == "z_spec"


@pytest.mark.unit
def test_discover_tables_cache_roundtrip(monkeypatch, tmp_path):
    mock_safe_search = Mock()
    monkeypatch.setattr(dmod, "TAPService", Mock())
    monkeypatch.setattr(dmod, "safe_search", mock_safe_search)
    mock_safe_search.side_effect = [
        pd.DataFrame({"table_name": ["galaxy.main"]}),
        pd.DataFrame(
            {
                "table_name": ["galaxy.main"] * 3,
                "name": ["ra", "dec", "z_spec"],
                "ucd": ["pos.eq.ra", "pos.eq.dec", "src.redshift"],
                "description": [None, None, None],
            }
        ),
        pd.DataFrame({"z": [0.1, 0.2]}),
    ]
    first = discover_tables("https://test.com/tap", limit=10, cache_dir=tmp_path)
    # Second call must be served from the parquet cache (safe_search side_effect exhausted)
    second = discover_tables("https://test.com/tap", limit=10, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert second.attrs["provenance"]["cached"] is True
