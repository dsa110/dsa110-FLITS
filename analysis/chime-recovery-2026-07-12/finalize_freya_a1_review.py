#!/usr/bin/env python3
"""Bind the completed visual review into the Freya A1 validation verdict."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "analysis/chime-recovery-2026-07-12/results/a1"


def main() -> int:
    validation_path = RESULT_DIR / "validation.json"
    manifest = json.loads((RESULT_DIR / "figures.manifest.json").read_text())
    review = json.loads((RESULT_DIR / "figures.review.json").read_text())
    validation = json.loads(validation_path.read_text())

    expected = {item["path"] for item in manifest["figures"]}
    reviewed = {item["path"] for item in review["figures"]}
    if expected != reviewed:
        raise ValueError("figure review does not cover the complete manifest")

    manual_pass = all(item["verdict"] == "match" for item in review["figures"])
    validation["checks"]["manual_review"] = {
        "pass": manual_pass,
        "overall_verdict": review["overall_verdict"],
        "review_file": "figures.review.json",
        "reason": review["science_disposition"],
    }
    failed = [name for name, check in validation["checks"].items() if check.get("pass") is False]
    pending = [
        name for name, check in validation["checks"].items() if check.get("pass") is not True
    ]
    validation["failed_checks"] = failed
    validation["pending_checks"] = pending
    validation["qualification_status"] = "fail" if failed else "pass"
    validation["science_status"] = "diagnostic_only" if failed else "measurement"
    validation["experiment_status"] = "documented_fail" if failed else "pass"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    verdict = {
        "experiment_status": validation["experiment_status"],
        "failed_checks": failed,
        "on_pulse_fit_performed": validation["on_pulse_fit_performed"],
        "science_status": validation["science_status"],
    }
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if validation["experiment_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
