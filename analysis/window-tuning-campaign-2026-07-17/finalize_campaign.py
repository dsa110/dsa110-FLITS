"""Promote only campaign records that pass injection, artifact, and figure gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    injection = json.loads((result_dir / "injection_recovery.json").read_text())
    review = json.loads((result_dir / "figures.review.json").read_text())
    verdicts = {item["path"]: item["verdict"] for item in review["verdicts"]}
    manifest = json.loads((result_dir / "figures.manifest.json").read_text())
    required = {item["file"] for item in manifest["figures"]}
    missing = sorted(required - set(verdicts))
    if missing:
        raise SystemExit(f"missing figure verdicts: {missing}")

    records = []
    for path in sorted(result_dir.glob("*_campaign.json")):
        record = json.loads(path.read_text())
        figure = f"{record['name']}_acf_fits.png"
        record["figure_review_status"] = "pass" if verdicts.get(figure) == "match" else "fail"
        gates_pass = (
            injection["gate_status"] == "pass"
            and record["artifact_validation_status"] == "pass"
            and record["figure_review_status"] == "pass"
        )
        record["science_status"] = "measurement" if gates_pass else "diagnostic_only"
        path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)

    with (result_dir / "campaign_results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "status": "closed",
        "injection_gate": injection["gate_status"],
        "figure_gate": "pass" if all(verdicts[p] == "match" for p in required) else "fail",
        "n_records": len(records),
        "n_measurements": sum(r["science_status"] == "measurement" for r in records),
        "n_diagnostic_only": sum(r["science_status"] == "diagnostic_only" for r in records),
        "artifact_failures": [
            r["name"] for r in records if r["artifact_validation_status"] != "pass"
        ],
    }
    (result_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
