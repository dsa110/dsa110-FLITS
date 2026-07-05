from pathlib import Path

import pandas as pd
import pytest

from .io import normalize_host_redshift, read_table, read_targets_yaml, write_table


@pytest.mark.unit
def test_normalize_host_redshift_rejects_placeholders():
    assert normalize_host_redshift(1.0) is None
    assert normalize_host_redshift(1.5) is None
    assert normalize_host_redshift(0.27) == 0.27
    assert normalize_host_redshift(None) is None


@pytest.mark.unit
def test_read_targets_yaml_null_and_placeholder_hosts(tmp_path: Path):
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(
        "targets:\n"
        "  - name: KNOWN\n    ra: 1.0\n    dec: 2.0\n    redshift: 0.3\n"
        "  - name: NULL_Z\n    ra: 3.0\n    dec: 4.0\n    redshift: null\n"
        "  - name: PLACEHOLDER\n    ra: 5.0\n    dec: 6.0\n    redshift: 1.0\n"
    )
    sightlines = read_targets_yaml(yaml_path)
    by_name = {s.name: s.redshift for s in sightlines}
    assert by_name["KNOWN"] == 0.3
    assert by_name["NULL_Z"] is None
    assert by_name["PLACEHOLDER"] is None


@pytest.mark.unit
def test_read_targets_yaml_accepts_legacy_z_host_key(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("targets:\n- {name: A, ra: 1.0, dec: 2.0, z_host: 0.25}\n")
    (s,) = read_targets_yaml(p)
    assert s.redshift == 0.25 and s.name == "A"
    assert "z_host" not in s.metadata


@pytest.mark.unit
def test_read_write_table_suffix_dispatch(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_path = tmp_path / "t.csv"
    pq_path = tmp_path / "sub" / "t.parquet"
    write_table(df, csv_path)
    write_table(df, pq_path)  # creates parent dir
    assert read_table(csv_path)["a"].tolist() == [1, 2]
    assert read_table(pq_path)["b"].tolist() == ["x", "y"]
