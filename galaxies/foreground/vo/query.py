"""ADQL cone queries against TAP services, with retries and wall-time budgets."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd

try:
    from pyvo.dal import TAPService
except Exception:  # pragma: no cover - optional import for offline tests
    TAPService = None  # type: ignore[assignment]

from .utils import make_provenance, set_tap_timeout


def quote_table(table: str) -> str:
    # Quote only when necessary; avoid over-quoting schema.table for services that
    # expect bare identifiers (e.g., Data Lab).
    t = table.strip()
    if t.startswith('"') and t.endswith('"'):
        return t
    if "/" in t or " " in t:
        return f'"{t}"'
    if "." in t:
        schema, tbl = t.split(".", 1)

        def _simple(identifier: str) -> bool:
            return identifier.replace("_", "").isalnum()

        if _simple(schema) and _simple(tbl):
            return f"{schema}.{tbl}"
        return f'"{schema}"."{tbl}"'
    if t.replace("_", "").isalnum():
        return t
    return f'"{t}"'


def build_cone_adql(
    table: str,
    ra_col: str,
    dec_col: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    columns: str = "*",
) -> str:
    t = quote_table(table)
    return (
        f"SELECT TOP 10000 {columns} FROM {t} "
        f"WHERE 1=CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
    )


def _with_retries(fn, attempts: int = 5, base_delay: float = 0.5, max_delay: float = 8.0):
    # ponytail: replaces the former tenacity dependency (absent from the flits env)
    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(min(base_delay * 2**k, max_delay))


def query_sync(access_url: str, adql: str, maxrec: int = 10000) -> pd.DataFrame:
    if TAPService is None:
        return pd.DataFrame()

    def _run() -> pd.DataFrame:
        svc = TAPService(access_url)
        set_tap_timeout(svc, timeout_seconds=10.0)
        if hasattr(svc, "run_sync"):
            res = svc.run_sync(adql, MAXREC=maxrec)
        else:  # legacy pyvo
            res = svc.launch_job_sync(adql)
        table = res.to_table()
        try:
            return table.to_pandas()
        except Exception:
            # astropy <-> pandas conversions can fail; fallback safely
            return pd.DataFrame(table.as_array())

    return _with_retries(_run)


def cone_query(
    access_url: str,
    table: str,
    ra_col: str,
    dec_col: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    columns: str = "*",
    maxrec: int = 10000,
) -> pd.DataFrame:
    ts_start = datetime.now(UTC).isoformat()
    adql = build_cone_adql(table, ra_col, dec_col, ra_deg, dec_deg, radius_deg, columns)
    df = query_sync(access_url, adql, maxrec=maxrec)
    ts_end = datetime.now(UTC).isoformat()
    prov = make_provenance(
        adql,
        service=access_url,
        table=table,
        extra={
            "ts_start_utc": ts_start,
            "ts_end_utc": ts_end,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "radius_deg": radius_deg,
            "maxrec": maxrec,
        },
    )
    try:
        df.attrs["provenance"] = prov
    except Exception:
        pass
    return df


def safe_search(
    tap: TAPService | str,
    adql: str,
    call_timeout_s: float = 20.0,
    max_wall_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> pd.DataFrame:
    """Run ADQL sync when possible, else async with a max wall-time budget.

    Returns an empty DataFrame on errors; raises ``TimeoutError`` when the async
    job exceeds ``max_wall_s``.
    """
    if TAPService is None:
        return pd.DataFrame()

    # Accept either a TAPService-like object with run_async/run_sync, or a URL
    if hasattr(tap, "run_async") or hasattr(tap, "run_sync"):
        svc = tap  # test double or real service
    else:
        svc = tap if isinstance(tap, TAPService) else TAPService(str(tap))
        set_tap_timeout(svc, timeout_seconds=call_timeout_s)

    try:
        if hasattr(svc, "run_sync"):
            res = svc.run_sync(adql)
            tab = res.to_table()
            try:
                return tab.to_pandas()
            except Exception:
                return pd.DataFrame(tab.as_array())
    except Exception:
        pass

    try:
        job = svc.run_async(adql)
    except Exception:
        return pd.DataFrame()
    start = time.time()
    while True:
        phase = getattr(job, "phase", None)
        if callable(phase):  # pyvo version differences
            try:
                phase = phase()
            except Exception:
                phase = None
        if isinstance(phase, str) and phase.upper() in {"COMPLETED", "ERROR", "ABORTED"}:
            break
        if time.time() - start > max_wall_s:
            try:
                job.delete()
            except Exception:
                pass
            raise TimeoutError("TAP async query exceeded wall-time budget")
        time.sleep(poll_interval_s)

    try:
        tab = job.fetch_result().to_table()
    except Exception:
        return pd.DataFrame()
    try:
        return tab.to_pandas()
    except Exception:
        try:
            return pd.DataFrame(tab.as_array())
        except Exception:
            return pd.DataFrame()
