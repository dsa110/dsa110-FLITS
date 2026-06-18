"""Network-optional cross-survey enrichment helpers for matched galaxies."""

import math

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


DESI_LS_MATCH_RADIUS = 1.5 * u.arcsec
WISE_MATCH_RADIUS = 2.5 * u.arcsec
GALEX_MATCH_RADIUS = 4.0 * u.arcsec
XSC_MATCH_RADIUS = 5.0 * u.arcsec

DESI_LS_COLUMNS = (
    "desi_ls_gmag",
    "desi_ls_rmag",
    "desi_ls_zmag",
    "desi_ls_w1mag",
    "desi_ls_w2mag",
    "desi_ls_type",
    "desi_ls_sersic",
    "desi_ls_shape_r",
    "desi_ls_sep_arcsec",
)
ALLWISE_COLUMNS = (
    "W1mag",
    "W2mag",
    "W3mag",
    "W4mag",
    "wise_W1_W2",
    "wise_agn",
    "wise_sep_arcsec",
)
GALEX_COLUMNS = ("galex_fuv", "galex_nuv", "galex_ebv", "galex_sep_arcsec")
XSC_COLUMNS = ("xsc_kmag", "xsc_axis_ratio", "xsc_pa", "xsc_sep_arcsec")
DESI_EMISSION_COLUMNS = (
    "desi_oii_flux",
    "desi_halpha_flux",
    "desi_is_agn",
    "desi_emission_matched",
)


def _blank_object_array(n_rows: int) -> np.ndarray:
    values = np.empty(n_rows, dtype=object)
    values[:] = np.nan
    return values


def _ensure_columns(matches: pd.DataFrame, columns: tuple[str, ...], overwrite: bool = False) -> pd.DataFrame:
    out = matches.copy()
    for column in columns:
        if not overwrite and column in out.columns:
            continue
        if column == "desi_emission_matched":
            out[column] = np.array([False] * len(out), dtype=object)
        elif column in {"desi_ls_type", "wise_agn", "desi_is_agn"}:
            out[column] = _blank_object_array(len(out))
        else:
            out[column] = np.full(len(out), np.nan, dtype="float64")
    return out


def _safe_float(value) -> float:
    if value is None:
        return math.nan
    try:
        if np.ma.is_masked(value):
            return math.nan
    except TypeError:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(numeric) or numeric <= -9990.0:
        return math.nan
    return numeric


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    return numeric.mask((~np.isfinite(numeric)) | (numeric <= -9990.0))


def _safe_bool(value):
    if value is None:
        return np.nan
    try:
        if np.ma.is_masked(value):
            return np.nan
    except TypeError:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n"}:
            return False
        return np.nan
    numeric = _safe_float(value)
    if math.isnan(numeric):
        return np.nan
    return bool(numeric)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        column = columns.get(candidate.lower())
        if column is not None:
            return column
    return None


def _candidate_coord_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    for ra_name, dec_name in (
        ("ra", "dec"),
        ("RAJ2000", "DEJ2000"),
        ("mean_fiber_ra", "mean_fiber_dec"),
    ):
        ra_col = _find_column(df, (ra_name,))
        dec_col = _find_column(df, (dec_name,))
        if ra_col is not None and dec_col is not None:
            return ra_col, dec_col
    return None


def _candidate_coords(df: pd.DataFrame) -> SkyCoord | None:
    coord_columns = _candidate_coord_columns(df)
    if coord_columns is None or df.empty:
        return None
    ra_col, dec_col = coord_columns
    ra = _safe_numeric_series(df[ra_col])
    dec = _safe_numeric_series(df[dec_col])
    valid = ra.notna() & dec.notna()
    if not valid.all():
        return None
    return SkyCoord(ra=ra.to_numpy() * u.deg, dec=dec.to_numpy() * u.deg)


def _valid_coordinate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    coord_columns = _candidate_coord_columns(df)
    if coord_columns is None or df.empty:
        return pd.DataFrame()
    ra_col, dec_col = coord_columns
    ra = _safe_numeric_series(df[ra_col])
    dec = _safe_numeric_series(df[dec_col])
    valid = ra.notna() & dec.notna()
    if not valid.any():
        return pd.DataFrame()
    return df.loc[valid].reset_index(drop=True)


def _query_cone_engine(engine, coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    if hasattr(engine, "query"):
        result = engine.query(coord, radius)
    else:
        result = engine(coord, radius)
    if result is None:
        return pd.DataFrame()
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)


def _query_targetid_engine(engine, targetid) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    if hasattr(engine, "query_by_targetid"):
        result = engine.query_by_targetid(targetid)
    else:
        result = engine(targetid)
    if result is None:
        return pd.DataFrame()
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)


def _nearest_candidate(coord: SkyCoord, candidates: pd.DataFrame) -> tuple[pd.Series | None, float]:
    valid_candidates = _valid_coordinate_candidates(candidates)
    coords = _candidate_coords(valid_candidates)
    if coords is None:
        return None, math.nan

    seps = coord.separation(coords).arcsec
    if len(seps) == 0 or not np.isfinite(seps).any():
        return None, math.nan
    nearest_pos = int(np.nanargmin(seps))
    return valid_candidates.iloc[nearest_pos], float(seps[nearest_pos])


def _within_radius(sep_arcsec: float, radius: u.Quantity) -> bool:
    return np.isfinite(sep_arcsec) and sep_arcsec <= radius.to(u.arcsec).value


def _row_coord(row: pd.Series) -> SkyCoord:
    return SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)


def _row_value(row: pd.Series, candidates: tuple[str, ...]):
    column = _find_column(pd.DataFrame(columns=row.index), candidates)
    if column is None:
        return None
    return row[column]


def _mag_from_flux(row: pd.Series, preferred_col: str, fallback_col: str) -> float:
    value = _safe_float(_row_value(row, (preferred_col,)))
    if math.isnan(value):
        value = _safe_float(_row_value(row, (fallback_col,)))
    if not (value > 0.0):
        return math.nan
    # Nanomaggies use the SDSS/DESI AB zeropoint convention: m_AB=22.5-2.5log10(flux).
    return 22.5 - 2.5 * math.log10(value)


def enrich_with_desi_ls(matches: pd.DataFrame, engine=None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()

    n_rows = len(matches)
    gmag = np.full(n_rows, np.nan, dtype="float64")
    rmag = np.full(n_rows, np.nan, dtype="float64")
    zmag = np.full(n_rows, np.nan, dtype="float64")
    w1mag = np.full(n_rows, np.nan, dtype="float64")
    w2mag = np.full(n_rows, np.nan, dtype="float64")
    types = _blank_object_array(n_rows)
    sersic = np.full(n_rows, np.nan, dtype="float64")
    shape_r = np.full(n_rows, np.nan, dtype="float64")
    seps = np.full(n_rows, np.nan, dtype="float64")

    if engine is not None:
        for pos in range(n_rows):
            row = matches.iloc[pos]
            coord = _row_coord(row)
            candidate, sep = _nearest_candidate(
                coord,
                _query_cone_engine(engine, coord, DESI_LS_MATCH_RADIUS),
            )
            seps[pos] = sep
            if candidate is None or not _within_radius(sep, DESI_LS_MATCH_RADIUS):
                continue

            gmag[pos] = _mag_from_flux(candidate, "dered_flux_g", "flux_g")
            rmag[pos] = _mag_from_flux(candidate, "dered_flux_r", "flux_r")
            zmag[pos] = _mag_from_flux(candidate, "dered_flux_z", "flux_z")
            w1mag[pos] = _mag_from_flux(candidate, "dered_flux_w1", "flux_w1")
            w2mag[pos] = _mag_from_flux(candidate, "dered_flux_w2", "flux_w2")
            source_type = _row_value(candidate, ("type",))
            if source_type is not None and not pd.isna(source_type):
                types[pos] = str(source_type)
            sersic[pos] = _safe_float(_row_value(candidate, ("sersic",)))
            shape_r[pos] = _safe_float(_row_value(candidate, ("shape_r",)))

    out = matches.copy()
    out["desi_ls_gmag"] = gmag
    out["desi_ls_rmag"] = rmag
    out["desi_ls_zmag"] = zmag
    out["desi_ls_w1mag"] = w1mag
    out["desi_ls_w2mag"] = w2mag
    out["desi_ls_type"] = types
    out["desi_ls_sersic"] = sersic
    out["desi_ls_shape_r"] = shape_r
    out["desi_ls_sep_arcsec"] = seps
    return out


def enrich_with_allwise(matches: pd.DataFrame, engine=None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()

    n_rows = len(matches)
    w1 = np.full(n_rows, np.nan, dtype="float64")
    w2 = np.full(n_rows, np.nan, dtype="float64")
    w3 = np.full(n_rows, np.nan, dtype="float64")
    w4 = np.full(n_rows, np.nan, dtype="float64")
    color = np.full(n_rows, np.nan, dtype="float64")
    agn = _blank_object_array(n_rows)
    seps = np.full(n_rows, np.nan, dtype="float64")

    if engine is not None:
        for pos in range(n_rows):
            row = matches.iloc[pos]
            coord = _row_coord(row)
            candidate, sep = _nearest_candidate(coord, _query_cone_engine(engine, coord, WISE_MATCH_RADIUS))
            seps[pos] = sep
            if candidate is None or not _within_radius(sep, WISE_MATCH_RADIUS):
                continue

            w1[pos] = _safe_float(_row_value(candidate, ("W1mag",)))
            w2[pos] = _safe_float(_row_value(candidate, ("W2mag",)))
            w3[pos] = _safe_float(_row_value(candidate, ("W3mag",)))
            w4[pos] = _safe_float(_row_value(candidate, ("W4mag",)))
            if np.isfinite(w1[pos]) and np.isfinite(w2[pos]):
                color[pos] = w1[pos] - w2[pos]
                # Stern+2012 defines the mid-IR AGN cut in WISE Vega colors: W1-W2 >= 0.8 mag.
                agn[pos] = bool(color[pos] >= 0.8)

    out = matches.copy()
    out["W1mag"] = w1
    out["W2mag"] = w2
    out["W3mag"] = w3
    out["W4mag"] = w4
    out["wise_W1_W2"] = color
    out["wise_agn"] = agn
    out["wise_sep_arcsec"] = seps
    return out


def enrich_with_galex(matches: pd.DataFrame, engine=None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()

    n_rows = len(matches)
    fuv = np.full(n_rows, np.nan, dtype="float64")
    nuv = np.full(n_rows, np.nan, dtype="float64")
    ebv = np.full(n_rows, np.nan, dtype="float64")
    seps = np.full(n_rows, np.nan, dtype="float64")

    if engine is not None:
        for pos in range(n_rows):
            row = matches.iloc[pos]
            coord = _row_coord(row)
            candidate, sep = _nearest_candidate(coord, _query_cone_engine(engine, coord, GALEX_MATCH_RADIUS))
            seps[pos] = sep
            if candidate is None or not _within_radius(sep, GALEX_MATCH_RADIUS):
                continue

            fuv[pos] = _safe_float(_row_value(candidate, ("FUVmag",)))
            nuv[pos] = _safe_float(_row_value(candidate, ("NUVmag",)))
            ebv[pos] = _safe_float(_row_value(candidate, ("E(B-V)",)))

    out = matches.copy()
    out["galex_fuv"] = fuv
    out["galex_nuv"] = nuv
    out["galex_ebv"] = ebv
    out["galex_sep_arcsec"] = seps
    return out


def enrich_with_xsc(matches: pd.DataFrame, engine=None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()

    n_rows = len(matches)
    kmag = np.full(n_rows, np.nan, dtype="float64")
    axis_ratio = np.full(n_rows, np.nan, dtype="float64")
    pa = np.full(n_rows, np.nan, dtype="float64")
    seps = np.full(n_rows, np.nan, dtype="float64")

    if engine is not None:
        for pos in range(n_rows):
            row = matches.iloc[pos]
            coord = _row_coord(row)
            candidate, sep = _nearest_candidate(coord, _query_cone_engine(engine, coord, XSC_MATCH_RADIUS))
            seps[pos] = sep
            if candidate is None or not _within_radius(sep, XSC_MATCH_RADIUS):
                continue

            kmag[pos] = _safe_float(_row_value(candidate, ("K.K20e",)))
            axis_ratio[pos] = _safe_float(_row_value(candidate, ("Kb/a",)))
            pa[pos] = _safe_float(_row_value(candidate, ("Kpa",)))

    out = matches.copy()
    out["xsc_kmag"] = kmag
    out["xsc_axis_ratio"] = axis_ratio
    out["xsc_pa"] = pa
    out["xsc_sep_arcsec"] = seps
    return out


def _targetid_value(value):
    numeric = _safe_float(value)
    if math.isnan(numeric):
        return None
    return int(numeric)


def enrich_with_desi_emission(matches: pd.DataFrame, engine=None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()

    n_rows = len(matches)
    oii = np.full(n_rows, np.nan, dtype="float64")
    halpha = np.full(n_rows, np.nan, dtype="float64")
    is_agn = _blank_object_array(n_rows)
    matched = np.array([False] * n_rows, dtype=object)

    if engine is not None and "targetid" in matches.columns:
        for pos in range(n_rows):
            row = matches.iloc[pos]
            if row.get("catalog") != "DESI_DR1":
                continue
            targetid = _targetid_value(row.get("targetid"))
            if targetid is None:
                continue

            result = _query_targetid_engine(engine, targetid)
            if result.empty:
                continue

            emission_row = result.iloc[0]
            matched[pos] = True
            oii[pos] = _safe_float(_row_value(emission_row, ("oii_3727_flux", "oii_flux")))
            halpha[pos] = _safe_float(_row_value(emission_row, ("halpha_flux",)))
            agn_value = _row_value(emission_row, ("is_agn", "agn"))
            parsed_agn = _safe_bool(agn_value)
            if not pd.isna(parsed_agn):
                is_agn[pos] = parsed_agn

    out = matches.copy()
    out["desi_oii_flux"] = oii
    out["desi_halpha_flux"] = halpha
    out["desi_is_agn"] = is_agn
    out["desi_emission_matched"] = matched
    return out


def enrich_all_catalogs(matches: pd.DataFrame, engines: dict | None = None) -> pd.DataFrame:
    result = matches.copy()
    if result.empty:
        return result

    engines = engines or {}
    steps = (
        ("desi_ls", enrich_with_desi_ls, DESI_LS_COLUMNS),
        ("allwise", enrich_with_allwise, ALLWISE_COLUMNS),
        ("galex", enrich_with_galex, GALEX_COLUMNS),
        ("xsc", enrich_with_xsc, XSC_COLUMNS),
        ("desi_emission", enrich_with_desi_emission, DESI_EMISSION_COLUMNS),
    )

    for key, func, columns in steps:
        try:
            result = func(result, engine=engines.get(key))
        except Exception as exc:
            print(f"Warning: {key} enrichment failed: {exc}")
            result = _ensure_columns(result, columns, overwrite=True)
    return result
