import math

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table

from galaxies.foreground import engines_extra as ee


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
        {
            "ra": [10.0],
            "dec": [20.0],
            "W1mag": [14.1],
            "W2mag": [13.8],
            "W3mag": [12.0],
            "W4mag": [8.0],
        }
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


def test_standardize_cluster_columns_psz2():
    raw = pd.DataFrame({"RAdeg": [10.0], "DEdeg": [72.0], "z": [0.20], "MSZ": [5.0]})
    out = ee._standardize_cluster_columns(raw, "J/A+A/594/A27/psz2")
    assert out["ra"].iloc[0] == 10.0 and out["dec"].iloc[0] == 72.0
    assert out["z"].iloc[0] == 0.20
    assert abs(out["m500_msun"].iloc[0] - 5.0e14) < 1.0  # MSZ is in 1e14 Msun
    assert math.isnan(out["r500_kpc"].iloc[0])  # PSZ2 has no R500
    assert out["classification"].iloc[0] == "cluster"


def test_standardize_cluster_columns_mcxc_uses_r500():
    raw = pd.DataFrame(
        {"RAJ2000": [10.0], "DEJ2000": [72.0], "z": [0.20], "M500": [3.0], "R500": [1.2]}
    )
    out = ee._standardize_cluster_columns(raw, "J/A+A/534/A109/mcxc")
    assert abs(out["m500_msun"].iloc[0] - 3.0e14) < 1.0
    assert abs(out["r500_kpc"].iloc[0] - 1200.0) < 1.0  # R500 Mpc -> kpc


def test_cluster_engine_concatenates_and_standardizes(monkeypatch):
    class FakeVizier:
        def __init__(self, cat_id):
            self.cat_id = cat_id

        def query(self, coord, radius):
            if self.cat_id == "J/A+A/594/A27/psz2":
                return pd.DataFrame({"RAdeg": [10.0], "DEdeg": [72.0], "z": [0.2], "MSZ": [5.0]})
            return pd.DataFrame()

    monkeypatch.setattr(ee, "VizierEngine", FakeVizier)
    eng = ee.ClusterEngine(catalogs={"PSZ2": "J/A+A/594/A27/psz2", "MCXC": "x"})
    out = eng.query(SkyCoord(10.0, 72.0, unit="deg"), 1.0 * u.deg)
    assert len(out) == 1
    assert out["classification"].iloc[0] == "cluster"
    assert abs(out["m500_msun"].iloc[0] - 5.0e14) < 1.0


def test_cluster_engine_empty_when_no_catalogs():
    eng = ee.ClusterEngine(catalogs={})
    assert eng.query(SkyCoord(10.0, 72.0, unit="deg"), 1.0 * u.deg).empty


def test_standardize_ned_tap_maps_columns():
    raw = pd.DataFrame(
        {"prefname": ["NGC 1"], "ra": [10.0], "dec": [72.0], "z": [0.1], "prefphytype": ["G"]}
    )
    out = ee._standardize_ned_tap(raw)
    assert out["name"].iloc[0] == "NGC 1"
    assert out["ra"].iloc[0] == 10.0 and out["dec"].iloc[0] == 72.0
    assert out["z"].iloc[0] == 0.1
    assert out["classification"].iloc[0] == "G"  # NED prefphytype -> classification
    assert out["catalog"].iloc[0] == "NED"
    assert list(out.columns) == ["name", "ra", "dec", "z", "classification", "catalog"]


def test_standardize_ned_tap_drops_stars_and_untyped_keeps_galaxies():
    # NED objdir is a mixed catalog: stars (type '*') carry junk near-zero
    # redshifts and the FRB transient itself appears untyped (blank). Both pass a
    # bare z<z_FRB foreground cut and corrupt the intervening DM budget, so the
    # engine must keep only extragalactic galaxy types (G/GClstr/...).
    raw = pd.DataFrame(
        {
            "prefname": ["a star", "the FRB", "a galaxy", "a cluster", "a quasar"],
            "ra": [10.0, 10.1, 10.2, 10.3, 10.4],
            "dec": [72.0, 72.0, 72.0, 72.0, 72.0],
            "z": [0.0001, 0.25, 0.04, 0.2, 1.5],
            "prefphytype": ["*", "", "G", "GClstr", "QSO"],
        }
    )
    out = ee._standardize_ned_tap(raw)
    assert out["name"].tolist() == ["a galaxy", "a cluster", "a quasar"]
    assert "*" not in out["classification"].tolist()


def test_ned_tap_engine_query_standardizes(monkeypatch):
    # Sync TAP plumbing: search -> to_table -> standardize.
    captured = {}

    class FakeResult:
        def to_table(self):
            return Table(
                {
                    "prefname": ["GClstr X"],
                    "ra": [10.0],
                    "dec": [72.0],
                    "z": [0.2],
                    "prefphytype": ["GClstr"],
                }
            )

    class FakeSvc:
        def __init__(self, url):
            pass

        def search(self, adql):
            captured["adql"] = adql
            return FakeResult()

    monkeypatch.setattr(ee.pyvo.dal, "TAPService", FakeSvc)
    df = ee.NedTapEngine().query(SkyCoord(10.0, 72.0, unit="deg"), 0.3 * u.deg)
    assert "NEDTAP.objdir" in captured["adql"] and "CIRCLE('ICRS'" in captured["adql"]
    assert len(df) == 1
    assert df["catalog"].iloc[0] == "NED"
    assert df["classification"].iloc[0] == "GClstr"  # cluster classification preserved
    assert df["name"].iloc[0] == "GClstr X"


def test_ned_tap_engine_returns_empty_on_error(monkeypatch):
    class BoomSvc:
        def __init__(self, url):
            raise RuntimeError("NED TAP down")

    monkeypatch.setattr(ee.pyvo.dal, "TAPService", BoomSvc)
    df = ee.NedTapEngine().query(SkyCoord(10.0, 72.0, unit="deg"), 0.3 * u.deg)
    assert df.empty


def test_ned_tap_engine_caps_cone_radius(monkeypatch):
    # Sync NED TAP caps near 60s; a 2deg request must shrink to max_radius_deg.
    captured = {}

    class FakeResult:
        def to_table(self):
            return Table({"prefname": [], "ra": [], "dec": [], "z": [], "prefphytype": []})

    class FakeSvc:
        def __init__(self, url):
            pass

        def search(self, adql):
            captured["adql"] = adql
            return FakeResult()

    monkeypatch.setattr(ee.pyvo.dal, "TAPService", FakeSvc)
    ee.NedTapEngine(max_radius_deg=0.5).query(SkyCoord(10.0, 72.0, unit="deg"), 2.0 * u.deg)
    assert "0.50000000)" in captured["adql"]  # cone capped at 0.5deg
    assert "2.00000000)" not in captured["adql"]  # not the requested 2deg
