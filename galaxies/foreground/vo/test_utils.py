import pytest

from .utils import set_tap_timeout


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
