#!/usr/bin/env python3
"""Bind the completed visual review into the Freya H2 validation verdict."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scintillation.scint_analysis.chime_correction_validation import (  # noqa: E402
    adjudicate_chime_result,
    combine_science_status,
)

RESULT_DIR = ROOT / "analysis/chime-recovery-2026-07-12/results/h2"


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
    checks = validation["correction_validation"]["checks"]
    checks["manual_review"] = {
        "pass": manual_pass,
        "overall_verdict": review["overall_verdict"],
        "review_file": "figures.review.json",
        "reason": review["science_disposition"],
    }
    widths = [
        item["selected_components"][0]["dnu_mhz"]
        for item in validation["subbands"]
        if item["selected_components"] and not item["selected_components"][0]["quality_flags"]
    ]
    correction = adjudicate_chime_result(
        checks,
        fitted_dnu_mhz=sum(widths) / len(widths) if widths else None,
    )
    correction["science_status"] = combine_science_status(
        artifact_status=validation["artifact_control"]["measurement_status"],
        correction_status=correction,
    )
    validation["correction_validation"] = {"checks": checks, **correction}
    validation["product_correction_status"] = correction["product_correction_status"]
    validation["science_status"] = correction["science_status"]
    validation["measurement_status"] = correction["science_status"]
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(correction, indent=2, sort_keys=True))
    return 0 if correction["product_correction_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
