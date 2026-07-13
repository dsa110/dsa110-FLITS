from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml


def synthesize(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    root = Path(config["output_dir"])
    product_paths = sorted((root / "products").glob("*/*/result.json"))
    products = [json.loads(path.read_text()) for path in product_paths]
    if len(products) != 24:
        raise RuntimeError(f"expected 24 product results, found {len(products)}")
    measurement_fields = [
        "burst", "telescope", "status", "dm_reference", "dm_residual", "dm_absolute",
        "sigma_method", "sigma_injection", "sigma_total", "input_sha256", "code_commit",
    ]
    with (root / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=measurement_fields)
        writer.writeheader()
        for product in products:
            writer.writerow({field: product.get(field) for field in measurement_fields})
    events = []
    for burst in sorted({product["burst"] for product in products}):
        by_band = {p["telescope"]: p for p in products if p["burst"] == burst}
        chime, dsa = by_band["chime"], by_band["dsa"]
        passing = [p for p in (chime, dsa) if p["status"] == "PASS"]
        event = {
            "burst": burst,
            "chime_status": chime["status"],
            "chime_dm": chime.get("dm_absolute"),
            "chime_sigma": chime.get("sigma_total"),
            "dsa_status": dsa["status"],
            "dsa_dm": dsa.get("dm_absolute"),
            "dsa_sigma": dsa.get("sigma_total"),
            "event_support": "none",
            "adopted_dm": None,
            "adopted_sigma": None,
            "adoption_verdict": "no new measured DM",
        }
        if len(passing) == 1:
            event["event_support"] = "single-band"
            event["adoption_verdict"] = "single-band evidence; adoption not automatic"
        elif len(passing) == 2:
            difference = abs(chime["dm_absolute"] - dsa["dm_absolute"])
            combined_sigma = float(np.hypot(chime["sigma_total"], dsa["sigma_total"]))
            event["band_difference"] = difference
            event["band_combined_sigma"] = combined_sigma
            if difference <= 2.0 * combined_sigma:
                event["event_support"] = "two-band-consistent"
                weights = np.array([1 / chime["sigma_total"] ** 2, 1 / dsa["sigma_total"] ** 2])
                values = np.array([chime["dm_absolute"], dsa["dm_absolute"]])
                event["adopted_dm"] = float(np.sum(weights * values) / np.sum(weights))
                event["adopted_sigma"] = float(np.sqrt(1.0 / np.sum(weights)))
                event["adoption_verdict"] = "candidate common DM; downstream dry-run required"
            else:
                event["event_support"] = "two-band-tension"
                event["adoption_verdict"] = "do not combine; investigate frequency dependence"
        events.append(event)
    fields = sorted({key for event in events for key in event})
    with (root / "event_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    report = {
        "product_count": len(products),
        "pass_count": sum(product["status"] == "PASS" for product in products),
        "unconstrained_count": sum(product["status"] == "UNCONSTRAINED" for product in products),
        "events": events,
    }
    (root / "event_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
