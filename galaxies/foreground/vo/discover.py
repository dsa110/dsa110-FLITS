"""Discover TAP tables carrying RA/Dec/redshift columns, via UCDs then name heuristics."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

try:
    from pyvo.dal import TAPService
except Exception:  # pragma: no cover - optional import for offline tests
    TAPService = None  # type: ignore[assignment]

from .query import safe_search
from .utils import set_tap_timeout

_REDZ_NAME_HINTS = [
    "redshift", "z_spec", "zphot", "z_phot", "photoz", "z_phot_median",
    "photoz_best", "zml", "z_pdf", "zbest", "z_best", "z",
]
_RA_NAME_HINTS = ["ra", "raj2000", "ra_icrs", "alpha"]
_DEC_NAME_HINTS = ["dec", "dej2000", "dec_icrs", "delta"]

_UCD_RA_MAIN = "pos.eq.ra;meta.main"
_UCD_DEC_MAIN = "pos.eq.dec;meta.main"
_UCD_RA = "pos.eq.ra"
_UCD_DEC = "pos.eq.dec"
_UCD_REDSHIFT = "src.redshift"

PROBABLE_Z_NAME_PATTERNS = (
    r"^z$", r"^z_", r"_z$", r"^redshift$", r"^z(spec|_spec|spec1)?$",
    r"^(zsp|zspec|z_spec)$",
    r"^(z_phot|zphot|photoz|photoz_best|z_phot_median|z_ml|z_best|z_pdf|zpdf)$",
    r"^cz$", r"^(v_rad|vrad|radial_velocity)$",
)

Z_EXCLUDE_NAME_PATTERNS = (
    r"size", r"estsize", r"access_", r"scale", r"tile", r"pix", r"obscore", r"obs_", r"^xz",
    r"file", r"path", r"url", r"uri", r"mime", r"format",
    r"exposure", r"^exp_", r"filter", r"band", r"wavelength", r"freq",
    r"energy", r"flux", r"mag", r"error", r"^err", r"uncertainty", r"^unc",
    r"flag", r"quality", r"^qual", r"status", r"^stat", r"time", r"date",
    r"epoch", r"mjd", r"jd", r"coord", r"offset", r"separation", r"^sep$",
    r"^(access_estsize|size|.*size.*)$",
    r"^[A-Z0-9]{4,}[A-Z0-9]*$",
)

_IN_LIST_CHUNK = 50  # batch TAP_SCHEMA IN-lists; oversized clauses fail on VizieR


def _is_probable_z(name: str | None, description: str | None, ucd: str | None = None) -> tuple[bool, str]:
    """Best-effort classifier for redshift-like columns by name/description.

    Returns (is_z, reason), reason in {ucd,name,desc,velocity,excluded_name,none}.
    Excludes common non-z terms to avoid false matches.
    """
    n = (name or "").strip().lower()
    d = (description or "").strip().lower()
    u = (ucd or "").strip().lower()
    if not n and not d and not u:
        return False, "none"
    if u.startswith(_UCD_REDSHIFT):
        return True, "ucd"
    for pat in Z_EXCLUDE_NAME_PATTERNS:
        if re.search(pat, n):
            return False, "excluded_name"
    for pat in PROBABLE_Z_NAME_PATTERNS:
        if re.search(pat, n):
            if n in {"cz", "vrad", "v_rad", "radial_velocity"}:
                return True, "velocity"
            return True, "name"
    if "redshift" in d:
        return True, "desc"
    return False, "none"


def _normalize_service_url(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    s = url.strip()
    for suffix in ("/async", "/sync"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    if s.endswith("?"):
        s = s[:-1]
    if s.startswith("http://"):
        s = "https://" + s[len("http://") :]
    p = urlparse(s)
    if not p.scheme or not p.netloc:
        return None
    return s


def _safe_run_tap(svc: TAPService, adql: str) -> pd.DataFrame:
    try:
        if hasattr(svc, "run_sync"):
            res = svc.run_sync(adql)
        else:  # legacy pyvo
            res = svc.launch_job_sync(adql)
        tab = res.to_table()
        try:
            return tab.to_pandas()
        except Exception:
            return pd.DataFrame(tab.as_array())
    except Exception:
        return pd.DataFrame()


def _choose_by_ucd(cols: pd.DataFrame, ucd_exact: str, ucd_prefix: str) -> str | None:
    if cols.empty or "ucd" not in cols.columns:
        return None
    u = cols["ucd"].astype(str).str.lower()
    exact = cols.loc[u == ucd_exact, "name"]
    if not exact.empty:
        return exact.iloc[0]
    pref = cols.loc[u.str.startswith(ucd_prefix), "name"]
    if not pref.empty:
        return pref.iloc[0]
    contains = cols.loc[u.str.contains(ucd_prefix), "name"]
    return contains.iloc[0] if not contains.empty else None


def _choose_by_name_or_desc(cols: pd.DataFrame, name_hints: list[str], desc_keywords: list[str]) -> str | None:
    if cols.empty:
        return None
    names = cols["name"].astype(str).str.lower()
    for h in name_hints:
        cand = cols.loc[names == h, "name"]
        if not cand.empty:
            return cand.iloc[0]
    for h in name_hints:
        cand = cols.loc[names.str.contains(h), "name"]
        if not cand.empty:
            return cand.iloc[0]
    if "description" in cols.columns:
        desc = cols["description"].astype(str).str.lower()
        for k in desc_keywords:
            cand = cols.loc[desc.str.contains(k), "name"]
            if not cand.empty:
                return cand.iloc[0]
    return None


def _with_meta_columns(cols: pd.DataFrame) -> pd.DataFrame:
    out = cols.copy()
    out["name"] = out["name"].astype(str)
    for c in ("ucd", "description"):
        if c not in out.columns:
            out[c] = None
    return out


def _pick_ra_dec_z(cols: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    """Choose (ra, dec, z, z_reason) for one table's column metadata."""
    ra = _choose_by_ucd(cols, _UCD_RA_MAIN, _UCD_RA) or _choose_by_name_or_desc(
        cols, _RA_NAME_HINTS, ["right ascension", "icrs", "ra"]
    )
    dec = _choose_by_ucd(cols, _UCD_DEC_MAIN, _UCD_DEC) or _choose_by_name_or_desc(
        cols, _DEC_NAME_HINTS, ["declination", "icrs", "dec"]
    )
    z = None
    z_reason = None
    for _, row in cols[["name", "description", "ucd"]].iterrows():
        ok, reason = _is_probable_z(row.get("name"), row.get("description"), row.get("ucd"))
        if ok:
            z = row.get("name")
            z_reason = reason
            break
    return ra, dec, z, z_reason


def _z_value_stats(svc: TAPService, table: str, z_col: str) -> dict[str, object]:
    """Sample up to 50 z values; flag columns that are non-numeric or out of range."""
    stats: dict[str, object] = {"z_ok": True, "z_frac_numeric": 0.0, "z_min": None, "z_max": None}
    try:
        q = f'SELECT TOP 50 "{z_col}" AS z FROM "{table}" WHERE "{z_col}" IS NOT NULL'
        sample = safe_search(svc, q, call_timeout_s=8.0, max_wall_s=10.0)
        if not sample.empty and "z" in sample.columns:
            zz = pd.to_numeric(sample["z"], errors="coerce")
            numeric = int(zz.notna().sum())
            frac = numeric / len(sample) if len(sample) else 0.0
            z_min = float(zz.min()) if numeric else None
            z_max = float(zz.max()) if numeric else None
            stats.update(z_frac_numeric=round(frac, 3), z_min=z_min, z_max=z_max)
            if numeric == 0 or frac < 0.10 or (z_min is not None and z_min < -0.01) or (z_max is not None and z_max > 10.0):
                stats["z_ok"] = False
    except Exception:
        pass
    return stats


def _columns_for_tables(svc: TAPService, tables: list[str]) -> pd.DataFrame:
    """Fetch TAP_SCHEMA column metadata for the given tables, chunking the IN-list."""
    frames = []
    for i in range(0, len(tables), _IN_LIST_CHUNK):
        in_list = ",".join(f"'{t}'" for t in tables[i : i + _IN_LIST_CHUNK])
        adql = (
            "SELECT table_name, column_name AS name, ucd, description "
            f"FROM TAP_SCHEMA.columns WHERE table_name IN ({in_list})"
        )
        chunk = safe_search(svc, adql, call_timeout_s=20.0, max_wall_s=45.0)
        if not chunk.empty:
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def find_columns(tap: TAPService | str, table_name: str) -> dict[str, str | None]:
    """Infer RA/Dec/Z columns for one table using UCDs first, then heuristics.

    Tries VOSI tables metadata, falling back to TAP_SCHEMA.columns.
    Returns {"ra_col", "dec_col", "z_col", "z_pick_reason", "z_source"} (values may be None).
    """
    if TAPService is None:
        return {"ra_col": None, "dec_col": None, "z_col": None}

    svc = tap if isinstance(tap, TAPService) else TAPService(str(tap))
    set_tap_timeout(svc, timeout_seconds=10.0)

    cols_df = pd.DataFrame()
    try:
        tinfo = svc.tables.get(table_name)
        if tinfo is not None:
            cols_df = pd.DataFrame(
                [
                    {
                        "name": getattr(c, "name", None),
                        "ucd": getattr(c, "ucd", None),
                        "description": getattr(c, "description", None),
                    }
                    for c in getattr(tinfo, "columns", [])
                    if getattr(c, "name", None)
                ]
            )
    except Exception:
        cols_df = pd.DataFrame()

    if cols_df.empty:
        adql = (
            "SELECT column_name AS name, ucd, description "
            f"FROM TAP_SCHEMA.columns WHERE table_name = '{table_name}'"
        )
        cols_df = _safe_run_tap(svc, adql)

    if cols_df.empty or "name" not in cols_df.columns:
        return {"ra_col": None, "dec_col": None, "z_col": None}

    ra, dec, z, z_reason = _pick_ra_dec_z(_with_meta_columns(cols_df))
    return {
        "ra_col": ra,
        "dec_col": dec,
        "z_col": z,
        "z_pick_reason": z_reason,
        "z_source": "velocity" if z_reason == "velocity" else None,
    }


def _build_prefilter_adql() -> str:
    """Single grouped TAP_SCHEMA scan for tables having RA, Dec, and z-like columns."""
    ra_list = ",".join(f"'{n.lower()}'" for n in [*_RA_NAME_HINTS, "_raj2000"])
    dec_list = ",".join(f"'{n.lower()}'" for n in [*_DEC_NAME_HINTS, "_dej2000", "de"])
    z_list = ",".join(f"'{n.lower()}'" for n in [*_REDZ_NAME_HINTS, "zsp", "zspec", "z_ml", "zml", "zpdf"])
    has_ra = (
        "MAX(CASE WHEN LOWER(c.ucd) LIKE 'pos.eq.ra%' THEN 1\n"
        f"      WHEN LOWER(c.column_name) IN ({ra_list}) THEN 1\n"
        "      WHEN LOWER(c.column_name) LIKE 'ra_%' THEN 1 ELSE 0 END)"
    )
    has_dec = (
        "MAX(CASE WHEN LOWER(c.ucd) LIKE 'pos.eq.dec%' THEN 1\n"
        f"      WHEN LOWER(c.column_name) IN ({dec_list}) THEN 1\n"
        "      WHEN LOWER(c.column_name) LIKE 'dec_%' OR LOWER(c.column_name) LIKE 'de_%' THEN 1 ELSE 0 END)"
    )
    has_z = (
        "MAX(CASE WHEN LOWER(c.ucd) LIKE 'src.redshift%' THEN 1\n"
        f"      WHEN LOWER(c.column_name) IN ({z_list}) THEN 1\n"
        "      WHEN LOWER(c.description) LIKE '%redshift%' THEN 1 ELSE 0 END)"
    )
    return (
        f"SELECT TOP 5000 c.table_name, {has_ra} AS has_ra, {has_dec} AS has_dec, {has_z} AS has_z\n"
        "FROM TAP_SCHEMA.columns AS c\n"
        "GROUP BY c.table_name\n"
        f"HAVING {has_ra} = 1 AND {has_dec} = 1 AND {has_z} = 1\n"
        "ORDER BY c.table_name"
    )


def discover_tables(access_url: str, limit: int = 500, cache_dir: str | Path = ".cache") -> pd.DataFrame:
    """Discover tables with RA/Dec and redshift-like columns for a TAP endpoint.

    Returns a DataFrame with columns ``table``, ``ra_col``, ``dec_col``, ``z_col``,
    cached to ``{cache_dir}/tables_{service_hash}_lim{limit}.parquet``. A provenance
    dict is attached at ``df.attrs['provenance']``.
    """
    norm_url = _normalize_service_url(access_url) or access_url
    svc_hash = hashlib.sha1(norm_url.encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"tables_{svc_hash}_lim{int(limit)}.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df = df.sort_values(["table", "ra_col", "dec_col", "z_col"], kind="mergesort").reset_index(drop=True)
            df.attrs["provenance"] = {"access_url": norm_url, "service_hash": svc_hash, "cached": True}
            return df
        except Exception:
            pass

    empty = pd.DataFrame(columns=["table", "ra_col", "dec_col", "z_col"])
    if TAPService is None:
        return empty

    svc = TAPService(norm_url)
    set_tap_timeout(svc, timeout_seconds=20.0)
    rows: list[dict[str, str | None]] = []
    per_table_meta: dict[str, dict[str, object]] = {}
    scanned = 0
    t0 = time.time()

    pre = safe_search(svc, _build_prefilter_adql(), call_timeout_s=20.0, max_wall_s=30.0)
    candidate_tables: list[str] = []
    if not pre.empty and "table_name" in pre.columns:
        candidate_tables = sorted({str(t) for t in pre["table_name"].astype(str)})[: max(0, int(limit))]

    if not candidate_tables:
        # Fallback (e.g. service rejects the grouped prefilter): sample all tables
        tables_df = safe_search(
            svc, "SELECT table_name FROM TAP_SCHEMA.tables ORDER BY table_name", call_timeout_s=15.0, max_wall_s=20.0
        )
        if not tables_df.empty and "table_name" in tables_df.columns:
            candidate_tables = [str(x) for x in tables_df["table_name"].astype(str)][: max(0, int(limit))]

    if candidate_tables:
        cols_df = _columns_for_tables(svc, candidate_tables)
        if not cols_df.empty:
            for tname, sub in cols_df.groupby("table_name"):
                scanned += 1
                ra, dec, z, z_reason = _pick_ra_dec_z(_with_meta_columns(sub))
                if not (ra and dec and z):
                    continue
                # Exclude obvious OBSCore false positives unless z re-passes the name gate
                if "obscore" in str(tname).lower() and not _is_probable_z(z, None)[0]:
                    continue
                stats = _z_value_stats(svc, str(tname), z)
                per_table_meta[str(tname)] = {"z_pick_reason": z_reason, "z_value_checked": True, **{k: v for k, v in stats.items() if k != "z_ok"}}
                if not stats["z_ok"]:
                    continue
                rows.append({"table": str(tname), "ra_col": ra, "dec_col": dec, "z_col": z})

    df = pd.DataFrame.from_records(rows, columns=["table", "ra_col", "dec_col", "z_col"]) if rows else empty
    df = df.sort_values(["table", "ra_col", "dec_col", "z_col"], kind="mergesort").reset_index(drop=True)

    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    elapsed = time.time() - t0
    prov: dict[str, object] = {
        "access_url": norm_url,
        "service_hash": svc_hash,
        "cached": cache_path.exists(),
        "num_tables_scanned": scanned,
        "num_candidates": len(df),
        "prefilter_rows": int(len(pre)),
        "elapsed_s": round(elapsed, 2),
        "partial": elapsed > 90.0,  # service-level budget
        "limit": int(limit),
    }
    if per_table_meta:
        prov["per_table_meta"] = per_table_meta
    df.attrs["provenance"] = prov
    return df
