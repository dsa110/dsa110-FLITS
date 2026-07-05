import types
from unittest.mock import Mock

import pandas as pd
import pytest

from . import query as qmod
from .query import build_cone_adql, cone_query, quote_table, safe_search


@pytest.mark.unit
def test_quote_table_variants():
    assert quote_table("simple") == "simple"
    assert quote_table("schema.table") == "schema.table"
    assert quote_table("sch.t-b") == '"sch"."t-b"'  # special chars quote per part
    assert quote_table("with space") == '"with space"'
    assert quote_table("a/b") == '"a/b"'
    assert quote_table('"quoted_table"') == '"quoted_table"'


@pytest.mark.unit
def test_build_cone_adql_contains_geometry():
    adql = build_cone_adql(table="my.table", ra_col="ra", dec_col="dec", ra_deg=10.0, dec_deg=0.0, radius_deg=0.1)
    assert "SELECT TOP 10000 * FROM my.table" in adql
    assert "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10.0, 0.0, 0.1))" in adql


@pytest.mark.unit
def test_cone_query_attaches_provenance_offline(monkeypatch):
    monkeypatch.setattr(qmod, "query_sync", lambda *a, **k: pd.DataFrame())
    df = cone_query(
        access_url="https://example.invalid/tap",
        table="t", ra_col="ra", dec_col="dec",
        ra_deg=10.0, dec_deg=0.0, radius_deg=0.1, maxrec=10,
    )
    prov = df.attrs.get("provenance")
    assert isinstance(prov, str)
    assert '"service": "https://example.invalid/tap"' in prov
    assert '"adql":' in prov


@pytest.mark.unit
def test_cone_query_success(monkeypatch, sample_cone_query_result):
    # patch by module object: galaxies/ is a namespace dir, so string targets can
    # hit a twin module imported under a different package identity
    mock_service = Mock()
    mock_result = Mock()
    mock_table = Mock()
    mock_table.to_pandas.return_value = sample_cone_query_result
    mock_result.to_table.return_value = mock_table
    mock_service.run_sync.return_value = mock_result
    monkeypatch.setattr(qmod, "TAPService", Mock(return_value=mock_service))

    result = cone_query("https://test.com/tap", "catalog.galaxies", "ra", "dec", 150.0, 2.0, 0.1)
    assert len(result) == 3
    assert "source_id" in result.columns
    assert "provenance" in result.attrs


class _FakeResult:
    def to_table(self):
        class _T:
            def to_pandas(self):
                return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

            def as_array(self):
                return []

        return _T()


class _FakeTapSync:
    def __init__(self):
        self._session = types.SimpleNamespace(request=lambda *a, **k: None)

    def run_sync(self, adql):
        return _FakeResult()


@pytest.mark.unit
def test_safe_search_sync_happy_path():
    df = safe_search(_FakeTapSync(), "SELECT 1")
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


class _FakeJob:
    def __init__(self):
        self._phase = "EXECUTING"

    @property
    def phase(self):
        return self._phase

    def fetch_result(self):
        raise RuntimeError("no result")

    def delete(self):
        self._phase = "ABORTED"


class _FakeTapAsyncOnly:
    def __init__(self):
        self._session = types.SimpleNamespace(request=lambda *a, **k: None)

    def run_async(self, adql):
        return _FakeJob()


@pytest.mark.unit
def test_safe_search_timeout():
    with pytest.raises(TimeoutError):
        safe_search(_FakeTapAsyncOnly(), "SELECT 1", call_timeout_s=0.01, max_wall_s=0.1, poll_interval_s=0.01)
