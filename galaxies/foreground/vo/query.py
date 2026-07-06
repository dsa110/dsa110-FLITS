"""ADQL cone queries against TAP services, with retries and wall-time budgets."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd

try:
    from pyvo.dal import TAPService
except Exception:  # pragma: no cover - optional import for offline tests
    TAPService = None  # type: ignore[assignment]

from .utils import set_tap_timeout


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
    *,
    columns: str = "*",
    top: int = 10000,
    order_by_distance: bool = True,
) -> str:
    """Build deterministic ADQL for a cone search around a sightline.

    By default the query computes per-row angular separation (in degrees) as a
    SELECT alias ``sep_deg`` and orders by it, so that a ``TOP {top}``
    truncation keeps the *closest* sources rather than an arbitrary subset.
    This is the completeness guarantee for impact-parameter analysis: without
    distance-ordered truncation, a TAP service is free to return any matching
    rows it likes, and the closest source could be silently excluded.

    The alias form (``DISTANCE(...) AS sep_deg ... ORDER BY sep_deg``) is used
    rather than placing the function call directly in ORDER BY, because some
    TAP parsers (VizieR's ``vollt`` among them) are strict about ORDER BY
    accepting only a column reference or an integer position. The result of
    distance-ordered queries therefore carries an extra ``sep_deg`` column,
    which downstream code may use as a diagnostic but is not part of the
    normalized schema.

    Pass ``order_by_distance=False`` for queries where you do not care about
    completeness or for services that do not support the DISTANCE function.
    """
    quoted_table = quote_table(table)
    if order_by_distance:
        select_list = (
            f"{columns}, DISTANCE("
            f"POINT('ICRS', {ra_col}, {dec_col}), "
            f"POINT('ICRS', {ra_deg}, {dec_deg})) AS sep_deg"
        )
        order_clause = " ORDER BY sep_deg"
    else:
        select_list = columns
        order_clause = ""
    return (
        f"SELECT TOP {top} {select_list} FROM {quoted_table} "
        f"WHERE 1=CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
        f"{order_clause}"
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


def _table_to_dataframe(result: object) -> pd.DataFrame:
    table = result.to_table()  # type: ignore[attr-defined]
    try:
        return table.to_pandas()
    except Exception:
        # astropy <-> pandas conversions can fail; fallback safely
        return pd.DataFrame(table.as_array())


def run_tap_sync(access_url: str, adql: str, *, maxrec: int = 10000) -> pd.DataFrame:
    """Run one TAP query with retry behavior."""
    if TAPService is None:
        return pd.DataFrame()

    def _run() -> pd.DataFrame:
        service = TAPService(access_url)
        set_tap_timeout(service, timeout_seconds=10.0)
        result = service.run_sync(adql, MAXREC=maxrec)
        return _table_to_dataframe(result)

    return _with_retries(_run)


def cone_query(
    access_url: str,
    table: str,
    ra_col: str,
    dec_col: str,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    columns: str = "*",
    maxrec: int = 10000,
) -> tuple[pd.DataFrame, dict[str, str | float | int | bool]]:
    """Run a TAP cone query and return the raw rows plus query metadata.

    Metadata includes ``truncated``: True when the returned row count meets or
    exceeds ``maxrec``, signalling that there may have been more matching rows
    than were returned. Combined with the distance-ordered ADQL emitted by
    ``build_cone_adql``, ``truncated=True`` means "the {maxrec} closest rows
    in angular separation were returned, but the cone may contain more sources
    at larger separation." This is not a completeness guarantee for derived
    quantities such as ``b_kpc / R_vir``, which also depend on redshift and
    halo mass.
    """
    started = datetime.now(UTC).isoformat()
    adql = build_cone_adql(
        table,
        ra_col,
        dec_col,
        ra_deg,
        dec_deg,
        radius_deg,
        columns=columns,
        top=maxrec,
    )
    df = run_tap_sync(access_url, adql, maxrec=maxrec)
    ended = datetime.now(UTC).isoformat()
    truncated = bool(len(df) >= maxrec)
    return df, {
        "adql": adql,
        "ts_start_utc": started,
        "ts_end_utc": ended,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "radius_deg": radius_deg,
        "maxrec": maxrec,
        "truncated": truncated,
    }


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
