import json

import pandas as pd
import pytest

from .normalize import SCHEMA_COLUMNS, ColumnMapping, normalize, to_common_schema


@pytest.mark.unit
def test_normalize_basic():
    df = pd.DataFrame(
        {
            "obj": ["A", "B"],
            "ra_deg": [10.0, 20.0],
            "dec_deg": [-1.0, 2.0],
            "z_spec": [0.1, None],
            "cluster_richness": [10, 20],
        }
    )
    mapping = ColumnMapping(name="obj", ra="ra_deg", dec="dec_deg", z="z_spec", richness="cluster_richness")
    out = normalize(df, mapping, service="svc", table="tab")

    assert list(out.columns) == SCHEMA_COLUMNS
    assert out["name"].tolist() == ["A", "B"]
    assert out["ra"].tolist() == [10.0, 20.0]
    assert out["richness"].tolist() == [10, 20]
    assert all(out["service"] == "svc") and all(out["table"] == "tab")
    # z_type inferred per value: first has a value, second is none
    assert out.loc[0, "z_type"] in ("unknown", "spec", "photo")
    assert out.loc[1, "z_type"] == "none"

    prov = json.loads(out.loc[0, "provenance_json"])
    assert prov["service"] == "svc" and prov["table"] == "tab"
    assert "column_mapping" in prov


@pytest.mark.unit
def test_normalize_missing_columns():
    df = pd.DataFrame({"name": ["obj1"], "ra": [150.0], "dec": [2.0], "z": [0.1]})
    out = normalize(df, ColumnMapping(name="name", ra="ra", dec="dec", z="z"), "service", "table")
    assert pd.isna(out["richness"].iloc[0])
    assert pd.isna(out["m_delta"].iloc[0])
    assert pd.isna(out["r_delta"].iloc[0])


@pytest.mark.unit
def test_normalize_with_explicit_types():
    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "ra": [10.0, 20.0],
            "dec": [-1.0, 2.0],
            "z_phot": [0.12, 0.34],
            "z_type": ["photo", "photo"],
        }
    )
    mapping = ColumnMapping(name="name", ra="ra", dec="dec", z="z_phot", z_type="z_type")
    out = normalize(df, mapping, service="svc", table="tab")
    assert all(out["z_type"] == "photo")


@pytest.mark.unit
def test_normalize_sorted_provenance():
    df = pd.DataFrame({"obj": ["A"], "ra": [10.0], "dec": [0.0], "z_phot": [0.2]})
    out = normalize(df, ColumnMapping(name="obj", ra="ra", dec="dec", z="z_phot"), service="svc", table="tab")
    prov = out.loc[0, "provenance_json"]
    assert prov.index('"service"') < prov.index('"table"')  # sorted keys


@pytest.mark.unit
def test_to_common_schema_ztype_variants():
    base = {"ra": [10.0], "dec": [0.0]}
    out_spec = to_common_schema(
        pd.DataFrame({**base, "z_spec": [0.1]}), ra_col="ra", dec_col="dec", z_col="z_spec", service="svc", table="tab"
    )
    assert (out_spec["z_type"] == "spec").all()

    out_phot = to_common_schema(
        pd.DataFrame({**base, "z_phot": [0.3]}), ra_col="ra", dec_col="dec", z_col="z_phot", service="svc", table="tab"
    )
    assert (out_phot["z_type"] == "photo").all()

    out_plain = to_common_schema(
        pd.DataFrame({**base, "z": [0.5]}), ra_col="ra", dec_col="dec", z_col="z", service="svc", table="tab"
    )
    assert out_plain["z_type"].isin(["unknown", "none"]).any()


@pytest.mark.unit
def test_to_common_schema_id_and_z_prior():
    df = pd.DataFrame({"galaxy_id": ["gal1"], "ra": [1.0], "dec": [2.0], "z_phot": [0.3]})
    out = to_common_schema(df, ra_col="ra", dec_col="dec", z_col="z_phot", service="svc", table="tab", id_col="galaxy_id")
    assert out["id"].tolist() == ["gal1"]
    assert out.loc[0, "z_type"] == "photo"
    prov = json.loads(out.loc[0, "provenance_json"])
    assert prov.get("z_prior") is True
