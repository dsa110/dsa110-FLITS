"""Shared helpers: TAP session timeout injection and provenance records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def set_tap_timeout(tap_service: object, timeout_seconds: float = 10.0) -> None:
    """Set a default HTTP timeout on a pyvo TAPService by wrapping its session.

    PyVO does not enforce a default timeout; this monkeypatches the underlying
    requests session's ``request`` to inject one when the caller passes none.
    Safe no-op if the service structure is unexpected.
    """
    try:
        session = getattr(tap_service, "_session", None)
        if session is None:
            return
        original_request = session.request

        def request_with_timeout(method, url, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = timeout_seconds
            return original_request(method, url, **kwargs)

        session.request = request_with_timeout
    except Exception:
        return


def make_provenance(adql: str | None, service: str, table: str, extra: dict[str, Any] | None = None) -> str:
    """JSON provenance record (sorted keys) for one query."""
    meta: dict[str, Any] = {
        "service": service,
        "table": table,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if adql is not None:
        meta["adql"] = adql
    if extra:
        meta.update(extra)
    return json.dumps(meta, sort_keys=True)
