import pandas as pd
import pytest

from .registry import _dedupe_endpoints, _normalize_access_urls, discover_tap_services


@pytest.mark.unit
def test_discover_tap_services_offline_anchor_merge():
    # Bogus RegTAP URL forces the exception path and avoids live network
    df = discover_tap_services(regtap_url="https://example.invalid/tap", include_anchors=True, max_services=5)
    urls = set(df["access_url"].tolist())
    assert "https://tapvizier.cds.unistra.fr/TAPVizieR/tap" in urls
    assert "https://mast.stsci.edu/vo-tap/" in urls
    assert "https://datalab.noirlab.edu/tap" in urls
    prov = df.attrs.get("provenance", {})
    assert isinstance(prov.get("adql"), str)
    assert prov.get("anchors_added") is True


@pytest.mark.unit
def test_normalize_access_urls_trims_and_upgrades():
    df = pd.DataFrame(
        {
            "access_url": [
                " http://example.org/tap/sync? ",
                "https://good.org/tap/async",
                "not_a_url",
            ]
        }
    )
    out = _normalize_access_urls(df)
    urls = out["access_url"].tolist()
    # strips '?', upgrades http->https, drops /sync|/async suffix (once each, in order)
    assert "https://example.org/tap/sync" in urls
    assert "https://good.org/tap" in urls
    assert len(out) == 2  # malformed dropped


@pytest.mark.unit
def test_dedupe_endpoints_unique_sorted():
    df = pd.DataFrame(
        {
            "ivoid": ["a", "b"],
            "title": ["B", "A"],
            "access_url": ["https://x", "https://x"],
        }
    )
    out = _dedupe_endpoints(df)
    assert list(out.columns) == ["ivoid", "title", "access_url"]
    assert len(out) == 1
