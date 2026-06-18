import math

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table

from galaxies.v2_0 import engines_extra as ee


def test_tap_box_query_filters_circle_masks_sentinels_and_uses_between(monkeypatch):
    captured = {}

    class FakeResult:
        def to_table(self):
            return Table(
                {
                    "src_ra": [10.0, 10.09, 10.25],
                    "src_dec": [20.0, 20.09, 20.0],
                    "flux_g": MaskedColumn([1.0, -9999.0, 3.0], mask=[False, False, True]),
                }
            )

    class FakeTapService:
        def __init__(self, tap_url):
            captured["tap_url"] = tap_url

        def search(self, adql):
            captured["adql"] = adql
            return FakeResult()

    monkeypatch.setattr(ee.pyvo.dal, "TAPService", FakeTapService)

    result = ee._tap_box_query(
        "https://example.invalid/tap",
        "example.sources",
        SkyCoord(10.0 * u.deg, 20.0 * u.deg),
        0.15 * u.deg,
        ["src_ra", "src_dec", "flux_g"],
        where="flux_g IS NOT NULL",
        ra_col="src_ra",
        dec_col="src_dec",
    )

    assert captured["tap_url"] == "https://example.invalid/tap"
    assert "BETWEEN" in captured["adql"]
    assert "CONTAINS" not in captured["adql"]
    assert "POINT" not in captured["adql"]
    assert "flux_g IS NOT NULL" in captured["adql"]
    assert result["ra"].tolist() == [10.0, 10.09]
    assert result["dec"].tolist() == [20.0, 20.09]
    assert math.isnan(result.loc[1, "flux_g"])
    assert "src_ra" not in result.columns
    assert "src_dec" not in result.columns


def test_desi_dr1_engine_query_standardizes_and_preserves_columns(monkeypatch):
    canned = pd.DataFrame(
        {
            "ra": [177.78],
            "dec": [71.70],
            "z": [0.241],
            "zerr": [0.0001],
            "zwarn": [0],
            "spectype": ["GALAXY"],
            "deltachi2": [25.0],
            "targetid": [12345],
        }
    )

    monkeypatch.setattr(ee, "_tap_box_query", lambda *args, **kwargs: canned.copy())

    result = ee.DesiDr1Engine().query(SkyCoord(177.78 * u.deg, 71.70 * u.deg), 1 * u.arcmin)

    assert result.loc[0, "ra"] == 177.78
    assert result.loc[0, "dec"] == 71.70
    assert result.loc[0, "z"] == 0.241
    assert result.loc[0, "catalog"] == "DESI_DR1"
    for column in ["zerr", "zwarn", "spectype", "deltachi2", "targetid"]:
        assert column in result.columns


def test_desi_dr1_empty_paths_do_not_query(monkeypatch):
    monkeypatch.setattr(ee, "_tap_box_query", lambda *args, **kwargs: pd.DataFrame())

    result = ee.DesiDr1Engine().query(SkyCoord(177.78 * u.deg, 71.70 * u.deg), 1 * u.arcmin)

    assert result.empty
    assert ee.DesiDr1Engine().query_emfit([]).empty
    assert ee.DesiDr1Engine().query_agngal([]).empty


def test_desi_ls_dr10_engine_sets_nan_redshifts_and_excludes_i_band(monkeypatch):
    captured = {}
    canned = pd.DataFrame(
        {
            "ra": [177.78],
            "dec": [71.70],
            "type": ["EXP"],
            "flux_g": [10.0],
            "flux_r": [20.0],
            "flux_z": [30.0],
            "flux_w1": [4.0],
            "flux_w2": [3.0],
            "flux_w3": [2.0],
            "flux_w4": [1.0],
            "dered_flux_g": [11.0],
            "dered_flux_r": [21.0],
            "dered_flux_z": [31.0],
            "shape_e1": [0.1],
            "shape_e2": [0.2],
            "shape_r": [1.4],
            "sersic": [2.0],
            "ref_cat": ["R1"],
        }
    )

    def fake_tap_box_query(tap_url, table, coord, radius, columns, **kwargs):
        captured["columns"] = columns
        return canned.copy()

    monkeypatch.setattr(ee, "_tap_box_query", fake_tap_box_query)

    result = ee.DesiLsDr10Engine().query(SkyCoord(177.78 * u.deg, 71.70 * u.deg), 1 * u.arcmin)

    assert result.loc[0, "catalog"] == "DESI_LS_DR10"
    assert result["z"].isna().all()
    assert "flux_i" not in result.columns
    assert "dered_flux_i" not in result.columns
    assert "flux_i" not in captured["columns"]
    assert "dered_flux_i" not in captured["columns"]


def test_allwise_engine_postprocesses_vizier_dataframe():
    eng = ee.AllWiseEngine()
    eng._vizier_engine.query = lambda coord, radius: pd.DataFrame(
        {"ra": [10.0], "dec": [20.0], "W1mag": [14.1], "W2mag": [13.8], "W3mag": [12.0], "W4mag": [8.0]}
    )

    result = eng.query(SkyCoord(10.0 * u.deg, 20.0 * u.deg), 1 * u.arcmin)

    assert result.loc[0, "catalog"] == "ALLWISE"
    assert result["z"].isna().all()
    for column in ["W1mag", "W2mag", "W3mag", "W4mag"]:
        assert column in result.columns


def test_galex_ais_engine_renames_parenthesized_ebv_column():
    eng = ee.GalexAisEngine()
    eng._vizier_engine.query = lambda coord, radius: pd.DataFrame(
        {"ra": [10.0], "dec": [20.0], "FUVmag": [20.1], "NUVmag": [19.2], "E(B-V)": [0.03]}
    )

    result = eng.query(SkyCoord(10.0 * u.deg, 20.0 * u.deg), 1 * u.arcmin)

    assert result.loc[0, "catalog"] == "GALEX_AIS"
    assert "FUVmag" in result.columns
    assert "NUVmag" in result.columns
    assert "ebv" in result.columns
    assert "E(B-V)" not in result.columns


def test_xsc_engine_renames_literal_vizier_columns():
    eng = ee.XscEngine()
    eng._vizier_engine.query = lambda coord, radius: pd.DataFrame(
        {"ra": [10.0], "dec": [20.0], "K.K20e": [11.4], "Kb/a": [0.7], "Kpa": [35.0]}
    )

    result = eng.query(SkyCoord(10.0 * u.deg, 20.0 * u.deg), 1 * u.arcmin)

    assert result.loc[0, "catalog"] == "2MASS_XSC"
    for column in ["Kmag", "axis_ratio", "pa_deg"]:
        assert column in result.columns


def test_vizier_wrappers_return_empty_dataframes_for_empty_canned_results():
    coord = SkyCoord(10.0 * u.deg, 20.0 * u.deg)
    for engine_cls in [ee.AllWiseEngine, ee.GalexAisEngine, ee.XscEngine]:
        eng = engine_cls()
        eng._vizier_engine.query = lambda coord, radius: pd.DataFrame()

        result = eng.query(coord, 1 * u.arcmin)

        assert result.empty
