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
    monkeypatch.setattr(
        cli, "cone_query", lambda *a, **k: (fake_rows, {"adql": "SELECT 1", "truncated": False})
    )
    cli.main(
        [
            "run-targets",
            "https://svc.example/tap",
            "galaxy.main",
            "ra",
            "dec",
            "--targets",
            str(sample_targets_yaml),
        ]
    )
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
    assert list(cache.glob(f"tables_{h}_lim*.parquet")), (
        "service_hash must match the tables cache filename"
    )


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
    pd.DataFrame(
        [{"table": "galaxy.main", "ra_col": "ra", "dec_col": "dec", "z_col": "z_spec"}]
    ).to_parquet(cache / f"tables_{svc_hash}_lim500.parquet")

    fake_rows = pd.DataFrame({"ra": [150.115], "dec": [2.205], "z_spec": [0.12]})
    monkeypatch.setattr(
        cli, "cone_query", lambda *a, **k: (fake_rows, {"adql": "SELECT 1", "truncated": False})
    )

    cli.main(["--cache-dir", str(cache), "query", "--targets", str(sample_targets_yaml)])
    assert (cache / "queries" / svc_hash / cli._hash("galaxy.main") / "FRB_Test_A.parquet").exists()

    out = tmp_path / "results"
    cli.main(
        [
            "--cache-dir",
            str(cache),
            "reduce",
            "--targets",
            str(sample_targets_yaml),
            "--out",
            str(out),
        ]
    )
    candidates = pd.read_parquet(out / "FRB_Test_A" / "candidates.parquet")
    assert len(candidates) == 1
    assert candidates.loc[0, "frb_name"] == "FRB_Test_A"
    assert (out / "FRB_Test_A" / "summary.md").exists()


# --- frb-foreground-halos lineage: run-catalog / run-plan / plan-queries ---
# Ported from the ffh click CLI test suite; invocations rewritten to argparse
# (cli.main([...]) + capsys), error paths assert SystemExit.


@pytest.fixture
def fake_glade2_rows():
    """A 2-row GLADE-2-shaped DataFrame: one populated, one missing Kmag."""
    import numpy as np

    return pd.DataFrame(
        {
            "RAJ2000": [150.0, 150.05],
            "DEJ2000": [2.0, 2.05],
            "z": [0.05, 0.05],
            "Kmag": [12.0, np.nan],
            "Dist": [200.0, 200.0],
            "PGC": ["1001", "1002"],
            "HyperLEDA": ["NGC0001", "NGC0002"],
        }
    )


def _fake_meta(ra, dec, radius_deg, maxrec=200, truncated=False):
    return {
        "adql": "SELECT * FROM mock",
        "ts_start_utc": "2026-01-01T00:00:00+00:00",
        "ts_end_utc": "2026-01-01T00:00:01+00:00",
        "ra_deg": ra,
        "dec_deg": dec,
        "radius_deg": radius_deg,
        "maxrec": maxrec,
        "truncated": truncated,
    }


@pytest.mark.unit
def test_run_catalog_smoke(tmp_path, monkeypatch, fake_glade2_rows):
    """End-to-end CLI smoke: targets YAML -> mocked cone_query -> ranked CSV + summary CSV."""
    import json

    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n  - name: TEST\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.5\n"
    )

    def fake_cone_query(*args, **kwargs):
        return fake_glade2_rows.copy(), _fake_meta(
            args[4], args[5], args[6], kwargs.get("maxrec", 200)
        )

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    assert output.exists()
    assert summary.exists()

    candidates = pd.read_csv(output)
    expected_cols = {
        "name",
        "id",
        "ra",
        "dec",
        "z",
        "z_type",
        "richness",
        "m_delta",
        "r_delta",
        "delta_def",
        "service",
        "table",
        "provenance_json",
        "theta_arcmin",
        "b_kpc",
        "r_delta_computed",
        "rank_key",
        "b_over_rvir",
        "intersects_nominal",
        "intersects_loose",
        "intersects_strict",
        "is_foreground",
        "frb_name",
        "frb_z",
    }
    assert expected_cols.issubset(set(candidates.columns)), (
        f"missing: {expected_cols - set(candidates.columns)}"
    )
    # First row should have populated halo mass; provenance includes mass section.
    populated = candidates[candidates["m_delta"].notna()].iloc[0]
    prov = json.loads(populated["provenance_json"])
    assert "mass" in prov
    assert prov["mass"]["mass_method"] == "kband_bell03_moster13"

    summary_df = pd.read_csv(summary)
    assert list(summary_df["frb_name"]) == ["TEST"]
    assert summary_df.iloc[0]["n_candidates"] == 2


@pytest.mark.unit
def test_run_catalog_unknown_catalog_raises(tmp_path):
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text("targets:\n  - name: T\n    ra: 0.0\n    dec: 0.0\n")

    with pytest.raises(SystemExit, match="Unknown catalog"):
        cli.main(["run-catalog", str(targets_yaml), "made-up-catalog"])


@pytest.mark.unit
def test_run_catalog_preserves_zero_row_targets(tmp_path, monkeypatch, fake_glade2_rows):
    """A target whose TAP query returns zero rows must still appear in the summary.

    Regression for the FRB 20240122A (mahi) case: the previous run dropped that
    sightline from the summary, silently shrinking the denominator from 12 to
    11. Now every target_records entry gets a row, with status='no_candidates_in_cone'
    flagging the zero-row case for downstream readers.

    This docstring said "FRB 20240119A" until 2026-07-09. That name belongs to a
    different DSA burst (nickname `nikhil`, candname 240119aacg, MJD 60328.6,
    DM 483) which is not a CHIME co-detection and was never in the 12. The
    mislabel came from the derived sheets `chimedsa_burst_specs.csv` /
    `DSA_CHIME_BurstProps.csv`, whose row for mahi carries mahi's own RA/Dec
    (39.7665, +71.0179) under nikhil's TNS name. `configs/bursts.yaml` is the
    registry: mahi = chime_id 354049284, MJD 60331.104, 2024-01-22T02:30:33.
    """
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n"
        "  - name: WITH_CANDS\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.5\n"
        "  - name: ZERO_ROW\n    ra: 30.0\n    dec: 70.0\n    redshift: 1.0\n"
    )

    def fake_cone_query(access_url, table, ra_col, dec_col, ra, dec, radius_deg, **kwargs):
        rows = fake_glade2_rows.copy() if abs(ra - 150.0) < 0.01 else pd.DataFrame()
        return rows, _fake_meta(ra, dec, radius_deg, kwargs.get("maxrec", 200))

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    summary_df = pd.read_csv(summary)
    # Both targets must be present — denominator is 2, never silently 1
    assert sorted(summary_df["frb_name"]) == ["WITH_CANDS", "ZERO_ROW"]
    zero_row = summary_df[summary_df.frb_name == "ZERO_ROW"].iloc[0]
    assert zero_row["status"] == "no_candidates_in_cone"
    assert zero_row["n_candidates"] == 0
    assert zero_row["n_foreground"] == 0
    assert zero_row["n_intersects_strict"] == 0
    assert zero_row["n_intersects_nominal"] == 0
    assert zero_row["n_intersects_loose"] == 0
    # And the populated target should NOT carry the zero-row status
    populated = summary_df[summary_df.frb_name == "WITH_CANDS"].iloc[0]
    assert populated["status"] == "ok"


@pytest.mark.unit
def test_run_catalog_excludes_likely_host_from_intersections(tmp_path, monkeypatch, capsys):
    """Host-confusion regression: a candidate at small theta + small dz from
    the FRB position must be flagged likely_host and NOT counted as a
    foreground intersection, even if it would otherwise pass the b/R_vir test.
    """
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n  - name: HOST_LIKE\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.27\n"
    )

    rows = pd.DataFrame(
        {
            "RAJ2000": [150.00056, 150.10],  # ~2 arcsec vs ~6 arcmin
            "DEJ2000": [2.0, 2.0],
            "z": [0.234, 0.10],  # row 0: dz=0.036, row 1: dz=0.17
            "Kmag": [12.0, 14.0],
            "Dist": [1100.0, 470.0],
            "PGC": ["1", "2"],
            "HyperLEDA": ["host_galaxy", "real_fg"],
        }
    )

    def fake_cone_query(*args, **kwargs):
        return rows.copy(), _fake_meta(args[4], args[5], args[6], kwargs.get("maxrec", 200))

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    candidates = pd.read_csv(output)
    host_row = candidates[candidates["name"] == "host_galaxy"].iloc[0]
    assert bool(host_row["likely_host"]) is True
    assert bool(host_row["is_foreground"]) is False

    summary_df = pd.read_csv(summary)
    row = summary_df.iloc[0]
    assert row["n_intersects_strict"] == 0
    assert row["n_intersects_nominal"] == 0
    assert "likely_host" in capsys.readouterr().out  # warning was echoed


@pytest.mark.unit
def test_host_dz_tolerance_split_by_z_type(tmp_path, monkeypatch):
    """Spec-z catalog rows must use a tighter dz tolerance than photo/unknown.

    Two candidates at identical theta = 2 arcsec and dz = 0.02 from the host:
    one row has f_dL=3 (spec-z, tolerance 0.005), the other f_dL=1 (photo-z,
    tolerance 0.05). The spec row is a real foreground; the photo row could be
    the host with photo-z scatter.
    """
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n  - name: HOST_LIKE\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.27\n"
    )

    rows = pd.DataFrame(
        {
            "RAJ2000": [150.00056, 150.00056],
            "DEJ2000": [2.0, 2.0],
            "zhelio": [0.25, 0.25],
            "M*": [1.0, 1.0],
            "dL": [1100.0, 1100.0],
            "Kmag": [12.0, 12.0],
            "f_dL": [3, 1],  # row 0 spec, row 1 photo
            "HyperLEDA": ["spec_galaxy", "photo_galaxy"],
            "GLADE+": [101, 102],
        }
    )

    def fake_cone_query(*args, **kwargs):
        return rows.copy(), _fake_meta(args[4], args[5], args[6], kwargs.get("maxrec", 200))

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade-plus",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    candidates = pd.read_csv(output)
    spec_row = candidates[candidates["name"] == "spec_galaxy"].iloc[0]
    photo_row = candidates[candidates["name"] == "photo_galaxy"].iloc[0]

    assert spec_row["z_type"] == "spec"
    assert photo_row["z_type"] == "photo"

    # The asymmetric outcome is the whole point of the fix:
    assert bool(spec_row["likely_host"]) is False
    assert bool(spec_row["is_foreground"]) is True
    assert bool(photo_row["likely_host"]) is True
    assert bool(photo_row["is_foreground"]) is False


@pytest.mark.unit
def test_run_catalog_warning_lines_match_final_summary_status(
    tmp_path, monkeypatch, capsys, fake_glade2_rows
):
    """The CLI's tail warning lines must reflect the resolved summary status.

    A sightline that has BOTH no spec-z host AND zero TAP rows should appear
    only under no_candidates_in_cone (the higher-priority status), not under
    z_frb_unknown.
    """
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n"
        "  - name: WITH_CANDS\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.5\n"
        "  - name: NO_Z_NO_ROWS\n    ra: 30.0\n    dec: 70.0\n"  # both conditions
    )

    def fake_cone_query(access_url, table, ra_col, dec_col, ra, dec, radius_deg, **kwargs):
        rows = fake_glade2_rows.copy() if abs(ra - 150.0) < 0.01 else pd.DataFrame()
        return rows, _fake_meta(ra, dec, radius_deg, kwargs.get("maxrec", 200))

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    summary_df = pd.read_csv(summary)
    no_z_row = summary_df[summary_df.frb_name == "NO_Z_NO_ROWS"].iloc[0]
    assert no_z_row["status"] == "no_candidates_in_cone"

    out = capsys.readouterr().out
    no_cand_section = (
        out.split("no_candidates_in_cone")[1] if "no_candidates_in_cone" in out else ""
    )
    z_unknown_section = out.split("z_frb_unknown")[1] if "z_frb_unknown" in out else ""
    assert "NO_Z_NO_ROWS" in no_cand_section
    assert "NO_Z_NO_ROWS" not in z_unknown_section


@pytest.mark.unit
def test_run_catalog_marks_z_frb_unknown_when_redshift_omitted(
    tmp_path, monkeypatch, capsys, fake_glade2_rows
):
    """Targets without a host redshift must produce status='z_frb_unknown'
    and zero foreground intersection counts."""
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text("targets:\n  - name: NO_Z\n    ra: 150.0\n    dec: 2.0\n")

    def fake_cone_query(*args, **kwargs):
        return fake_glade2_rows.copy(), _fake_meta(
            args[4], args[5], args[6], kwargs.get("maxrec", 200)
        )

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
    )

    summary_df = pd.read_csv(summary)
    row = summary_df.iloc[0]
    assert row["status"] == "z_frb_unknown"
    assert row["n_foreground"] == 0
    assert row["n_intersects_strict"] == 0
    assert row["n_intersects_nominal"] == 0
    assert row["n_intersects_loose"] == 0
    assert "z_frb_unknown" in capsys.readouterr().out

    candidates = pd.read_csv(output)
    assert (candidates["is_foreground"] == False).all()  # noqa: E712
    assert (candidates["z_frb_known"] == False).all()  # noqa: E712


@pytest.mark.unit
def test_run_catalog_marks_possibly_truncated_sightlines(
    tmp_path, monkeypatch, capsys, fake_glade2_rows
):
    """A sightline whose cone hits maxrec must be tagged status='possibly_truncated'."""
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n  - name: TRUNC\n    ra: 150.0\n    dec: 2.0\n    redshift: 0.5\n"
    )

    def fake_cone_query(access_url, table, ra_col, dec_col, ra, dec, radius_deg, **kwargs):
        return fake_glade2_rows.copy(), _fake_meta(
            ra, dec, radius_deg, kwargs.get("maxrec", 200), truncated=True
        )

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)

    output = tmp_path / "candidates.csv"
    summary = tmp_path / "summary.csv"
    cli.main(
        [
            "run-catalog",
            str(targets_yaml),
            "glade2",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--maxrec",
            "2",
        ]
    )

    summary_df = pd.read_csv(summary)
    assert summary_df.iloc[0]["status"] == "possibly_truncated"
    assert "possibly_truncated" in capsys.readouterr().out  # warning was echoed


@pytest.mark.unit
def test_run_plan_uses_per_row_radius_and_maxrec(tmp_path, monkeypatch, fake_glade2_rows):
    plan_csv = tmp_path / "plan.csv"
    plan_csv.write_text(
        "frb_name,catalog_id,access_url,table_id,ra_deg,dec_deg,radius_arcmin,frb_z,maxrec\n"
        "TEST,glade2,http://tap/vizier,VII/281/glade2,150.0,2.0,42.5,0.5,7777\n"
    )
    captured: list[dict] = []

    def fake_cone_query(access_url, table, ra_col, dec_col, ra, dec, radius_deg, **kwargs):
        captured.append({"radius_deg": radius_deg, "maxrec": kwargs.get("maxrec")})
        return fake_glade2_rows.copy(), _fake_meta(ra, dec, radius_deg, kwargs.get("maxrec", 200))

    monkeypatch.setattr(cli, "cone_query", fake_cone_query)
    output = tmp_path / "candidates.csv"
    cli.main(["run-plan", str(plan_csv), "--output", str(output)])
    assert captured[0]["radius_deg"] == pytest.approx(42.5 / 60.0)
    assert captured[0]["maxrec"] == 7777
    assert output.exists()


@pytest.mark.unit
def test_plan_queries_writes_maxrec_column(tmp_path):
    targets_yaml = tmp_path / "targets.yaml"
    targets_yaml.write_text(
        "targets:\n  - name: T\n    ra: 10.0\n    dec: 20.0\n    redshift: 0.1\n"
    )
    output = tmp_path / "plan.csv"
    cli.main(
        [
            "plan-queries",
            str(targets_yaml),
            "--solver",
            "greedy",
            "--budget",
            "5",
            "--maxrec",
            "9000",
            "--output",
            str(output),
        ]
    )
    plan = pd.read_csv(output)
    assert "maxrec" in plan.columns
    assert (plan["maxrec"] == 9000).all()
