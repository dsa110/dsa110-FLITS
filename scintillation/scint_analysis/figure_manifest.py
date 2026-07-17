"""Shared figure-manifest writer for window-campaign diagnostics."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def register_figure(outdir, filename, expectation, *, campaign):
    """Register one generated figure as pending without writing a review verdict."""
    root = Path(outdir)
    path = root / "figures.manifest.json"
    figures = {}
    if path.exists():
        figures = {
            item["file"]: item
            for item in json.loads(path.read_text()).get("figures", [])
        }
    figures[str(filename)] = {
        "file": str(filename),
        "expectation": expectation,
        "review_status": "pending",
    }
    path.write_text(
        json.dumps(
            {
                "directory": str(root),
                "generated": str(date.today()),
                "campaign": campaign,
                "figures": [figures[key] for key in sorted(figures)],
            },
            indent=2,
        )
        + "\n"
    )
    return path


def campaign_is_validated(record):
    """Return whether a campaign record has passed every measurement gate."""
    return bool(
        record
        and record.get("science_status") == "measurement"
        and record.get("artifact_validation_status") == "pass"
        and record.get("figure_review_status") == "pass"
    )
