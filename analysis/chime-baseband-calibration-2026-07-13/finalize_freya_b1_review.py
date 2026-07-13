#!/usr/bin/env python3
"""Compact the remote B1 result and bind the completed figure review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "analysis/chime-baseband-calibration-2026-07-13/results/b1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_review_pass(manifest: dict, review: dict) -> bool:
    expected = {item["path"] for item in manifest["figures"]}
    reviewed = {item["path"] for item in review["figures"]}
    if expected != reviewed:
        raise ValueError("figure review does not cover the complete manifest")
    return (
        all(item["verdict"] == "match" for item in review["figures"])
        and review["overall_verdict"] == "match"
        and review.get("qualification_authorized") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.raw.read_text())
    manifest = json.loads((RESULT_DIR / "figures.manifest.json").read_text())
    review = json.loads((RESULT_DIR / "figures.review.json").read_text())
    manual_pass = _manual_review_pass(manifest, review)

    for record in validation["records"]:
        record.pop("target_spectrum", None)
        record.pop("recovered_spectrum", None)
    validation["raw_validation"] = {
        "path": "/data/research/astrophysics/frbs/chime-dsa-codetections/experiments/freya_endtoend_calibration_v1/results/validation.json",
        "sha256": _sha256(args.raw),
        "retention": "full spectra retained on h17; compact record committed",
    }
    validation["checks"]["manual_review"] = {
        "pass": manual_pass,
        "overall_verdict": review["overall_verdict"],
        "qualification_authorized": review.get("qualification_authorized") is True,
        "review_file": str((RESULT_DIR / "figures.review.json").resolve()),
        "reason": review["science_disposition"],
    }
    failed = [name for name, check in validation["checks"].items() if check.get("pass") is False]
    validation["failed_checks"] = failed
    validation["pending_checks"] = [
        name for name, check in validation["checks"].items() if check.get("pass") is not True
    ]
    validation["qualification_status"] = "fail" if failed else "pass"
    validation["experiment_status"] = "documented_fail" if failed else "pass"
    validation["science_status"] = "diagnostic_only" if failed else "calibration_only"
    output = RESULT_DIR / "validation.json"
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "experiment_status": validation["experiment_status"],
                "failed_checks": failed,
                "on_pulse_fit_performed": validation["on_pulse_fit_performed"],
                "raw_validation_sha256": validation["raw_validation"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["experiment_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
