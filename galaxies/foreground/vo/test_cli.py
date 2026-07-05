from pathlib import Path

import pandas as pd
import pytest

from . import cli


@pytest.mark.unit
def test_cli_requires_subcommand(capsys):
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.unit
def test_cli_services_offline_prints_anchors(capsys):
    # Bogus RegTAP endpoint -> registry degrades to anchors, no live network needed
    cli.main(["services", "--regtap-url", "https://example.invalid/tap", "--max-services", "5"])
    out = capsys.readouterr().out
    assert "access_url" in out
    assert "tapvizier.cds.unistra.fr" in out


@pytest.mark.unit
def test_cli_run_targets_prints_counts(sample_targets_yaml, monkeypatch, capsys):
    fake_rows = pd.DataFrame({"ra": [1.0, 2.0], "dec": [0.0, 0.1]})
    monkeypatch.setattr(cli, "cone_query", lambda *a, **k: fake_rows)
    cli.main(["run-targets", "https://svc.example/tap", "galaxy.main", "ra", "dec",
              "--targets", str(sample_targets_yaml)])
    out = capsys.readouterr().out
    assert "# FRB_Test_A: 2 rows" in out
    assert out.count("rows") == 3


@pytest.mark.unit
def test_cli_discover_hash_matches_tables_cache(tmp_path, monkeypatch):
    # service_hash written to services.parquet must equal the hash discover_tables
    # uses to name its cache file, even for un-normalized input URLs
    fake_tables = pd.DataFrame([{"table": "t", "ra_col": "ra", "dec_col": "dec", "z_col": "z"}])

    def fake_discover_tables(url, limit, cache_dir):
        from .discover import _normalize_service_url
        h = cli._hash(_normalize_service_url(url) or url)
        fake_tables.to_parquet(Path(cache_dir) / f"tables_{h}_lim{limit}.parquet", index=False)
        return fake_tables

    monkeypatch.setattr(cli, "discover_tables", fake_discover_tables)
    monkeypatch.setattr(cli, "_SERVICE_ALIASES", {"weird": "http://svc.example/tap/sync"})
    cache = tmp_path / ".cache"
    cli.main(["--cache-dir", str(cache), "discover", "--services", "weird", "--limit", "5"])

    svc = pd.read_parquet(cache / "services.parquet")
    h = svc.loc[0, "service_hash"]
    assert list(cache.glob(f"tables_{h}_lim*.parquet")), "service_hash must match the tables cache filename"


@pytest.mark.unit
def test_cli_query_reduce_roundtrip(tmp_path, sample_targets_yaml, monkeypatch):
    # Seed a fake discover cache, monkeypatch cone_query, and run query -> reduce offline
    cache = tmp_path / ".cache"
    cache.mkdir()
    svc_url = "https://svc.example/tap"
    svc_hash = cli._hash(svc_url)
    pd.DataFrame([{"access_url": svc_url, "service_hash": svc_hash, "num_tables": 1}]).to_parquet(
        cache / "services.parquet"
    )
    pd.DataFrame([{"table": "galaxy.main", "ra_col": "ra", "dec_col": "dec", "z_col": "z_spec"}]).to_parquet(
        cache / f"tables_{svc_hash}_lim500.parquet"
    )

    fake_rows = pd.DataFrame({"ra": [150.115], "dec": [2.205], "z_spec": [0.12]})
    monkeypatch.setattr(cli, "cone_query", lambda *a, **k: fake_rows)

    cli.main(["--cache-dir", str(cache), "query", "--targets", str(sample_targets_yaml)])
    assert (cache / "queries" / svc_hash / cli._hash("galaxy.main") / "FRB_Test_A.parquet").exists()

    out = tmp_path / "results"
    cli.main(["--cache-dir", str(cache), "reduce", "--targets", str(sample_targets_yaml), "--out", str(out)])
    candidates = pd.read_parquet(out / "FRB_Test_A" / "candidates.parquet")
    assert len(candidates) == 1
    assert candidates.loc[0, "frb_name"] == "FRB_Test_A"
    assert (out / "FRB_Test_A" / "summary.md").exists()
