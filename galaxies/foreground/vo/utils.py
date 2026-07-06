"""Shared helpers: TAP session timeout injection."""

from __future__ import annotations


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
