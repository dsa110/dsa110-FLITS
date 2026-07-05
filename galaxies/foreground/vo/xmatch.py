"""CDS X-Match API helpers."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

_XMATCH_URL = "https://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"


def cds_xmatch(
    left: pd.DataFrame,
    right: pd.DataFrame,
    colnames: tuple[str, str, str, str] = ("ra", "dec", "ra", "dec"),
    radius_arcsec: float = 1.0,
) -> pd.DataFrame:
    """Cross-match two position tables via the CDS X-Match service (network)."""
    l_ra, l_dec, r_ra, r_dec = colnames
    files = {
        "cat1": ("left.csv", left[[l_ra, l_dec]].to_csv(index=False), "text/csv"),
        "cat2": ("right.csv", right[[r_ra, r_dec]].to_csv(index=False), "text/csv"),
    }
    data = {"sepArcSec": str(radius_arcsec), "format": "csv"}
    resp = requests.post(_XMATCH_URL, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content))


def cds_xmatch_positions(
    positions_df: pd.DataFrame,
    catalog: str,
    ra_col: str = "ra",
    dec_col: str = "dec",
    radius_arcsec: float = 60.0,
    response_format: str = "csv",
) -> pd.DataFrame:
    """Cross-match positions against a remote CDS catalog (e.g. ``vizier:II/246``)."""
    if ra_col not in positions_df.columns or dec_col not in positions_df.columns:
        raise ValueError("positions_df must contain RA/Dec columns")

    csv_left = positions_df[[ra_col, dec_col]].rename(columns={ra_col: "ra", dec_col: "dec"}).to_csv(index=False)
    files = {"cat1": ("positions.csv", csv_left, "text/csv")}
    data = {"cat2": catalog, "sepArcSec": str(radius_arcsec), "format": response_format}
    resp = requests.post(_XMATCH_URL, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content))
