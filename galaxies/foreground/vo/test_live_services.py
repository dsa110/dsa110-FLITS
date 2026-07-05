"""Live-service integration tests (require network; run with -m network)."""

import time

import pandas as pd
import pytest

from .discover import discover_tables
from .normalize import SCHEMA_COLUMNS, to_common_schema
from .query import cone_query
from .reduce import merge_and_rank
from .registry import discover_tap_services
from .targets import get_cosmology

VIZIER = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"


@pytest.mark.integration
@pytest.mark.network
def test_discover_tap_services_regtap():
    services = discover_tap_services(keywords=("galaxy", "catalog"), max_services=20, include_anchors=True)
    assert len(services) > 0
    assert {"access_url", "title"} <= set(services.columns)
    assert any("tapvizier.cds.unistra.fr" in url for url in services["access_url"])


@pytest.mark.integration
@pytest.mark.network
def test_discover_tables_vizier(tmp_path):
    tables = discover_tables(VIZIER, limit=20, cache_dir=tmp_path)
    if len(tables) > 0:
        assert {"table", "ra_col", "dec_col", "z_col"} <= set(tables.columns)
        for z_col in tables["z_col"].dropna():
            assert "access" not in z_col.lower()
            assert "size" not in z_col.lower()


@pytest.mark.integration
@pytest.mark.network
def test_discovery_caching(tmp_path):
    t0 = time.time()
    tables1 = discover_tables(VIZIER, limit=10, cache_dir=tmp_path)
    first_duration = time.time() - t0

    assert list(tmp_path.glob("tables_*.parquet"))

    t0 = time.time()
    tables2 = discover_tables(VIZIER, limit=10, cache_dir=tmp_path)
    second_duration = time.time() - t0

    assert second_duration < first_duration / 2
    pd.testing.assert_frame_equal(tables1, tables2)
    assert tables2.attrs["provenance"]["cached"] is True


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
def test_end_to_end_pipeline_vizier(tmp_path, sample_targets):
    tables = discover_tables(VIZIER, limit=20, cache_dir=tmp_path)
    if len(tables) == 0:
        pytest.skip("No suitable tables found for end-to-end test")

    test_table = tables.iloc[0]
    target = sample_targets[0]

    result = cone_query(
        VIZIER, test_table["table"], test_table["ra_col"], test_table["dec_col"],
        ra_deg=target.ra, dec_deg=target.dec, radius_deg=0.1, maxrec=50,
    )
    if len(result) == 0:
        pytest.skip("No sources found in cone search")

    normalized = to_common_schema(
        result,
        ra_col=test_table["ra_col"], dec_col=test_table["dec_col"], z_col=test_table["z_col"],
        service=VIZIER, table=test_table["table"],
    )
    assert list(normalized.columns) == SCHEMA_COLUMNS

    ranked = merge_and_rank(normalized, frb_ra_deg=target.ra, frb_dec_deg=target.dec, cosmo=get_cosmology())
    assert {"theta_arcmin", "b_kpc", "rank_key"} <= set(ranked.columns)
    assert ranked["rank_key"].is_monotonic_increasing
