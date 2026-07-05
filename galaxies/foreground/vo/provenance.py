"""Deterministic provenance helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def make_provenance(
    adql: str | None,
    *,
    service: str,
    table: str,
    column_mapping: dict[str, str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create sorted-key JSON provenance for a normalized output row."""
    payload: dict[str, Any] = {
        "adql": adql,
        "service": service,
        "table": table,
    }
    if column_mapping is not None:
        payload["column_mapping"] = column_mapping
    if extra:
        payload.update(extra)
    return json.dumps(payload, sort_keys=True)


def parse_provenance(value: str | None) -> dict[str, Any]:
    """Parse provenance JSON, returning an empty dict for missing values."""
    if not value:
        return {}
    return json.loads(value)
