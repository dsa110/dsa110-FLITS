#!/usr/bin/env python3
"""Apply the frozen p1 mechanism gate and select at most one variant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(measurement: dict) -> dict:
    return {
        "variant": measurement.get("variant"),
        "lag1_cross_correlation": measurement["cross_correlation"][0],
        "lorentzian_fit": measurement["lorentzian_fit"],
        "mechanism_gate": measurement["mechanism_gate"],
        "measurement_artifact": measurement.get("measurement_artifact"),
    }


def select_variant(measurements: list[dict]) -> dict:
    if not measurements:
        raise ValueError("at least one measurement is required")
    eligible = [item for item in measurements if item["mechanism_gate"]["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["lorentzian_fit"]["amplitude"],
            abs(item["cross_correlation"][0]),
            item.get("variant", {}).get("oversample", 10**9),
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "schema_version": 1,
        "experiment": "p1-window-upchan",
        "status": "mechanism-qualified" if selected is not None else "DOCUMENTED-FAIL",
        "selection_rule": [
            "lowest_lorentzian_amplitude",
            "lowest_lag1_cross_correlation",
            "lowest_oversample",
        ],
        "selected_variant": selected.get("variant") if selected is not None else None,
        "eligible_count": len(eligible),
        "measurements": [_summary(item) for item in measurements],
        "next_step": (
            "run unchanged blinded C1 calibration"
            if selected is not None
            else "stop without on-pulse fitting or C1 calibration"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing verdict: {args.output}")
    payloads = []
    for path in args.measurements:
        payload = json.loads(path.read_text())
        payload["measurement_artifact"] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        payloads.append(payload)
    result = select_variant(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_variant", "next_step")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
