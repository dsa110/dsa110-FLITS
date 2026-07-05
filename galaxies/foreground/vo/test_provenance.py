import json

import pytest

from .provenance import make_provenance, parse_provenance, utc_now_iso


@pytest.mark.unit
def test_make_provenance_basic():
    prov = json.loads(make_provenance("SELECT * FROM test", service="https://test.com/tap", table="test_table"))
    assert prov["service"] == "https://test.com/tap"
    assert prov["table"] == "test_table"
    assert prov["adql"] == "SELECT * FROM test"


@pytest.mark.unit
def test_make_provenance_with_extra_and_column_mapping():
    prov = json.loads(
        make_provenance(
            None,
            service="svc",
            table="tab",
            column_mapping={"ra": "RAJ2000"},
            extra={"ra_deg": 150.0, "radius_deg": 0.1},
        )
    )
    assert prov["adql"] is None
    assert prov["column_mapping"]["ra"] == "RAJ2000"
    assert prov["ra_deg"] == 150.0
    assert prov["radius_deg"] == 0.1


@pytest.mark.unit
def test_make_provenance_sorted_keys():
    prov_str = make_provenance("SELECT *", service="zzz", table="aaa", extra={"zzz": 1, "aaa": 2})
    assert prov_str == json.dumps(json.loads(prov_str), sort_keys=True)


@pytest.mark.unit
def test_parse_provenance_roundtrip_and_empty():
    assert parse_provenance(None) == {}
    assert parse_provenance("") == {}
    payload = make_provenance("SELECT 1", service="s", table="t")
    assert parse_provenance(payload)["table"] == "t"


@pytest.mark.unit
def test_utc_now_iso_parses():
    from datetime import datetime

    datetime.fromisoformat(utc_now_iso())
