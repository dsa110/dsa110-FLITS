"""Additional opt-in catalog engines for galaxy searches."""

import numpy as np
import pandas as pd
import pyvo
from astropy import units as u
from astropy.coordinates import SkyCoord

from .engines import BaseEngine, VizierEngine


NOIRLAB_TAP_URL = "https://datalab.noirlab.edu/tap"


def _mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert survey sentinel values to NaN without changing nonnumeric fields."""
    if df.empty:
        return df
    return df.replace(-9999, np.nan)


def _tap_search_to_dataframe(tap_url: str, adql: str) -> pd.DataFrame:
    svc = pyvo.dal.TAPService(tap_url)
    table = svc.search(adql).to_table()
    return _mask_sentinels(table.to_pandas())


def _ensure_standard_columns(df: pd.DataFrame, catalog: str) -> pd.DataFrame:
    df = df.copy()
    for column in ["ra", "dec", "z"]:
        if column not in df.columns:
            df[column] = np.nan
    df["catalog"] = catalog
    return df


def _tap_box_query(
    tap_url,
    table,
    coord,
    radius,
    columns,
    where=None,
    ra_col="ra",
    dec_col="dec",
) -> pd.DataFrame:
    """Query Data Lab TAP with a box prefilter, then enforce a true sky circle.

    NOIRLab Astro Data Lab rejects ADQL geometry functions for these tables
    ("function point does not exist"), so use a portable RA/Dec BETWEEN box and
    perform the physically correct circular cut with Astropy on the client side.
    """
    try:
        radius_deg = radius.to(u.deg).value
        ra0 = coord.ra.deg
        dec0 = coord.dec.deg
        ddec = radius_deg
        cos_dec = max(abs(np.cos(np.radians(dec0))), 1.0e-6)
        dra = radius_deg / cos_dec

        predicates = [
            f"{dec_col} BETWEEN {dec0 - ddec:.12g} AND {dec0 + ddec:.12g}",
            f"{ra_col} BETWEEN {ra0 - dra:.12g} AND {ra0 + dra:.12g}",
        ]
        if where is not None:
            predicates.append(f"({where})")

        adql = (
            f"SELECT {', '.join(columns)} "
            f"FROM {table} "
            f"WHERE {' AND '.join(predicates)}"
        )
        df = _tap_search_to_dataframe(tap_url, adql)
        if df.empty:
            return pd.DataFrame()

        matches = SkyCoord(df[ra_col].to_numpy() * u.deg, df[dec_col].to_numpy() * u.deg)
        df = df.loc[coord.separation(matches) <= radius].copy()
        if df.empty:
            return pd.DataFrame()
        return df.rename(columns={ra_col: "ra", dec_col: "dec"})
    except Exception:
        return pd.DataFrame()


class DesiDr1Engine(BaseEngine):
    def __init__(self, tap_url=NOIRLAB_TAP_URL, require_primary=True):
        self.tap_url = tap_url
        self.require_primary = require_primary

    def query(self, coord, radius) -> pd.DataFrame:
        columns = [
            "mean_fiber_ra",
            "mean_fiber_dec",
            "z",
            "zerr",
            "zwarn",
            "spectype",
            "deltachi2",
            "targetid",
            "zcat_primary",
        ]
        where = "zcat_primary='true' AND zwarn=0" if self.require_primary else "zwarn=0"
        df = _tap_box_query(
            self.tap_url,
            "desi_dr1.zpix",
            coord,
            radius,
            columns,
            where=where,
            ra_col="mean_fiber_ra",
            dec_col="mean_fiber_dec",
        )
        if df.empty:
            return pd.DataFrame()

        return _ensure_standard_columns(df, "DESI_DR1")

    def query_emfit(self, targetids) -> pd.DataFrame:
        return self._query_targetid_table("desi_dr1.emfit", targetids)

    def query_agngal(self, targetids) -> pd.DataFrame:
        return self._query_targetid_table("desi_dr1.agngal", targetids)

    def _query_targetid_table(self, table, targetids) -> pd.DataFrame:
        ids = [int(targetid) for targetid in targetids]
        if not ids:
            return pd.DataFrame()

        try:
            adql = f"SELECT * FROM {table} WHERE targetid IN ({', '.join(map(str, ids))})"
            return _tap_search_to_dataframe(self.tap_url, adql)
        except Exception:
            return pd.DataFrame()


class DesiLsDr10Engine(BaseEngine):
    def __init__(self, tap_url=NOIRLAB_TAP_URL):
        self.tap_url = tap_url

    def query(self, coord, radius) -> pd.DataFrame:
        # At Dec +70..+74, Legacy Survey DR10 i-band fields are -9999 sentinels;
        # selecting only grz/Wise avoids surfacing nonphysical placeholder data.
        columns = [
            "ra",
            "dec",
            "type",
            "flux_g",
            "flux_r",
            "flux_z",
            "flux_w1",
            "flux_w2",
            "flux_w3",
            "flux_w4",
            "dered_flux_g",
            "dered_flux_r",
            "dered_flux_z",
            "shape_e1",
            "shape_e2",
            "shape_r",
            "sersic",
            "ref_cat",
        ]
        df = _tap_box_query(self.tap_url, "ls_dr10.tractor", coord, radius, columns)
        if df.empty:
            return pd.DataFrame()

        # Data Lab's ls_dr10.photo_z table is empty at these DSA-110 declinations.
        df = df.copy()
        df["z"] = np.nan
        return _ensure_standard_columns(df, "DESI_LS_DR10")


class AllWiseEngine(BaseEngine):
    def __init__(self):
        self._vizier_engine = VizierEngine("II/328/allwise")

    def query(self, coord, radius) -> pd.DataFrame:
        df = self._vizier_engine.query(coord, radius)
        if df.empty:
            return pd.DataFrame()

        return _ensure_standard_columns(df, "ALLWISE")


class GalexAisEngine(BaseEngine):
    def __init__(self):
        self._vizier_engine = VizierEngine("II/335/galex_ais")

    def query(self, coord, radius) -> pd.DataFrame:
        df = self._vizier_engine.query(coord, radius)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df = df.rename(columns={col: "ebv" for col in ["E(B-V)"] if col in df.columns})
        return _ensure_standard_columns(df, "GALEX_AIS")


class XscEngine(BaseEngine):
    def __init__(self):
        self._vizier_engine = VizierEngine("VII/233/xsc")

    def query(self, coord, radius) -> pd.DataFrame:
        df = self._vizier_engine.query(coord, radius)
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df = df.rename(
            columns={
                col: new_col
                for col, new_col in {
                    "K.K20e": "Kmag",
                    "Kb/a": "axis_ratio",
                    "Kpa": "pa_deg",
                }.items()
                if col in df.columns
            }
        )
        return _ensure_standard_columns(df, "2MASS_XSC")
