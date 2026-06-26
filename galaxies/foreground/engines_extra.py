"""Additional opt-in catalog engines for galaxy searches."""

import os

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

        adql = f"SELECT {', '.join(columns)} FROM {table} WHERE {' AND '.join(predicates)}"
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


# Per-catalog column maps for all-sky cluster catalogs:
# catalog_id -> (ra_col, dec_col, z_col, m500_col_1e14, r500_col_mpc_or_None).
_CLUSTER_COLUMN_MAPS = {
    "J/A+A/594/A27/psz2": ("RAdeg", "DEdeg", "z", "MSZ", None),
    "J/A+A/534/A109/mcxc": ("RAJ2000", "DEJ2000", "z", "M500", "R500"),
    "J/A+A/688/A187/mcxcii": ("RAJ2000", "DEJ2000", "z", "M500", "R500"),
}


def _standardize_cluster_columns(df: pd.DataFrame, catalog_id: str) -> pd.DataFrame:
    """Standardize a cluster catalog frame to ra/dec/z/m500_msun/r500_kpc/classification."""
    out = df.copy()
    cols = _CLUSTER_COLUMN_MAPS.get(catalog_id)
    if cols is None:
        out["classification"] = "cluster"
        out["catalog"] = catalog_id
        return out
    ra_c, dec_c, z_c, m_c, r_c = cols
    lower = {c.lower(): c for c in out.columns}
    rename = {}
    for src, std in ((ra_c, "ra"), (dec_c, "dec"), (z_c, "z")):
        if src.lower() in lower:
            rename[lower[src.lower()]] = std
    out = out.rename(columns=rename)
    m500 = pd.to_numeric(out.get(m_c), errors="coerce") if m_c in out.columns else np.nan
    out["m500_msun"] = m500 * 1.0e14
    if r_c is not None and r_c in out.columns:
        out["r500_kpc"] = pd.to_numeric(out[r_c], errors="coerce") * 1000.0  # Mpc -> kpc
    else:
        out["r500_kpc"] = np.nan
    out["classification"] = "cluster"
    out["catalog"] = catalog_id
    return out


class ClusterEngine(BaseEngine):
    """All-sky galaxy-cluster engine (PSZ2 + MCXC + MCXC-II via Vizier).

    Only all-sky cluster catalogs cover the sample's high declination; each
    catalog supplies redshift + M500 (and R500 where available) so the search can
    apply an r200-relative impact cut and a beta-model ICM dispersion measure.
    """

    def __init__(self, catalogs=None):
        if catalogs is None:
            from .config import CLUSTER_VIZIER_CATALOGS

            catalogs = CLUSTER_VIZIER_CATALOGS
        self.catalogs = dict(catalogs)

    def query(self, coord, radius) -> pd.DataFrame:
        frames = []
        for cat_id in self.catalogs.values():
            raw = VizierEngine(cat_id).query(coord, radius)
            if raw.empty:
                continue
            std = _standardize_cluster_columns(raw, cat_id)
            keep = [
                c
                for c in ("ra", "dec", "z", "m500_msun", "r500_kpc", "classification", "catalog")
                if c in std.columns
            ]
            frames.append(std[keep])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


NED_TAP_URL = "https://ned.ipac.caltech.edu/tap/"


def _is_extragalactic_ned_type(classification) -> bool:
    """True for NED object types that are galaxies / galaxy systems.

    NEDTAP.objdir is a mixed catalog: stars (prefphytype '*'), the FRB transient
    itself (untyped/blank), and Galactic sources sit alongside galaxies. Stars
    carry junk near-zero redshifts (~1e-4) and the FRB self-entry sits at z_FRB,
    so both slip past a bare z<z_FRB foreground cut and inflate the intervening DM
    budget. Keep only galaxy types: NED's galaxy/group/cluster/pair/triple codes
    all start with 'G' (excluding 'GammaS', a gamma-ray source) plus QSO/AGN.
    """
    s = str(classification).strip().upper()
    if s in ("", "NAN", "NONE", "GAMMAS"):
        return False
    return s.startswith("G") or s in {"QSO", "QGROUP", "AGN", "Q", "ABLS", "EMLS"}


def _standardize_ned_tap(df: pd.DataFrame) -> pd.DataFrame:
    """Map NED TAP objdir columns to the search schema (name/ra/dec/z/classification).

    prefname -> name, prefphytype -> classification (NED object type, e.g. 'GClstr');
    ra/dec/z pass through; catalog tagged 'NED'. Non-galaxy types (stars, the FRB
    self-entry, other Galactic sources) are dropped — see _is_extragalactic_ned_type.
    """
    out = df.rename(columns={"prefname": "name", "prefphytype": "classification"})
    out["catalog"] = "NED"
    if "classification" in out.columns:
        out = out[out["classification"].map(_is_extragalactic_ned_type)].reset_index(drop=True)
    keep = [c for c in ("name", "ra", "dec", "z", "classification", "catalog") if c in out.columns]
    return out[keep]


class NedTapEngine(BaseEngine):
    """NED foreground engine via the VO TAP service (synchronous).

    Replaces the deprecated astroquery legacy objsearch path, which omits data
    ingested after 2026-01 and was unreachable during testing. NED's async TAP
    result host (rc.ned.ipac.caltech.edu) is itself unreachable, so this uses the
    synchronous endpoint. Sync NED TAP caps server-side near 60s, so the cone is
    capped at FLITS_NED_TAP_MAX_DEG (default 0.5deg; NEDTAP.objdir timings:
    0.3deg~9s, 0.5deg~27s, 0.7deg~39s, >=1deg fails at 60s). That cap is >10x the
    search's foreground-galaxy footprint (100 kpc impact <= ~0.05deg even at the
    sample's lowest z, z~0.04), and clusters now come from ClusterEngine, so the
    cap drops no galaxy of interest. Output schema matches NedEngine.
    """

    def __init__(self, tap_url: str = NED_TAP_URL, max_radius_deg: float | None = None):
        self.tap_url = tap_url
        self.max_radius_deg = (
            max_radius_deg
            if max_radius_deg is not None
            else float(os.environ.get("FLITS_NED_TAP_MAX_DEG", "0.5"))
        )

    def query(self, coord, radius) -> pd.DataFrame:
        ra0, dec0 = coord.ra.deg, coord.dec.deg
        sr = min(radius.to(u.deg).value, self.max_radius_deg)
        adql = (
            "SELECT prefname, ra, dec, z, prefphytype FROM NEDTAP.objdir "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra0:.8f}, {dec0:.8f}, {sr:.8f})) = 1"
        )
        try:
            df = _tap_search_to_dataframe(self.tap_url, adql)
        except Exception as e:
            print(f"NED TAP query failed: {e}")
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        return _standardize_ned_tap(df)
