import json
from datetime import datetime

import pytest

from .utils import make_provenance, set_tap_timeout


class _Sess:
    def __init__(self):
        self.timeout_seen = []

    def request(self, method, url, **kwargs):
        self.timeout_seen.append(kwargs.get("timeout"))
        return None


class _FakeTap:
    def __init__(self):
        self._session = _Sess()


@pytest.mark.unit
def test_set_tap_timeout_injects_default():
    svc = _FakeTap()
    set_tap_timeout(svc, timeout_seconds=3.5)
    svc._session.request("GET", "https://x")
    assert svc._session.timeout_seen[-1] == 3.5
    # explicit timeout preserved
    svc._session.request("GET", "https://x", timeout=1.2)
    assert svc._session.timeout_seen[-1] == 1.2


@pytest.mark.unit
def test_make_provenance_basic():
    prov = json.loads(make_provenance(adql="SELECT * FROM test", service="https://test.com/tap", table="test_table"))
    assert prov["service"] == "https://test.com/tap"
    assert prov["table"] == "test_table"
    assert prov["adql"] == "SELECT * FROM test"
    datetime.fromisoformat(prov["timestamp_utc"])  # valid ISO timestamp


@pytest.mark.unit
def test_make_provenance_with_extra_and_no_adql():
    prov = json.loads(
        make_provenance(adql=None, service="svc", table="tab", extra={"ra_deg": 150.0, "radius_deg": 0.1})
    )
    assert prov["ra_deg"] == 150.0
    assert prov["radius_deg"] == 0.1
    assert "adql" not in prov


@pytest.mark.unit
def test_make_provenance_sorted_keys():
    prov_str = make_provenance(adql="SELECT *", service="zzz", table="aaa", extra={"zzz": 1, "aaa": 2})
    assert prov_str == json.dumps(json.loads(prov_str), sort_keys=True)
