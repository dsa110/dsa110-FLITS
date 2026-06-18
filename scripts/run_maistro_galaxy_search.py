#!/usr/bin/env python3
"""Run galaxy search and emit Maistro provenance sidecar records."""

from flits.orchestration.maistro import main


if __name__ == "__main__":
    raise SystemExit(main())
