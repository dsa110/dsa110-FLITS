#!/usr/bin/env python3
"""Run the fail-closed Freya CHIME gate on one corrected product."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "analysis/scintillation-dsa-lorentzian-2026-07-07/run_dsa_lorentzian_fits.py"
BASE_CONFIG = ROOT / "scintillation/configs/bursts/freya_chime.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subbands", type=int, default=2, choices=(2, 3, 4))
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("freya_recovery_driver", DRIVER)
    driver = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(driver)

    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["input_data_path"] = str(args.product.resolve())
    analysis = config.setdefault("analysis", {})
    analysis["bandpass_normalization"] = {"enable": True}
    analysis["instrumental_background_correction"] = {
        "enable": True,
        "manifest_path": str(args.manifest.resolve()),
        "validation": {},
    }

    with tempfile.TemporaryDirectory(prefix="flits-freya-gate-") as temp_dir:
        prepared = driver._config_for_fresh_acf(config, output_dir=Path(temp_dir))
        prepared = driver._config_with_subband_count(prepared, args.subbands)
        result, _ = driver._fit_prepared_config(
            prepared,
            BASE_CONFIG,
            output_dir=Path(temp_dir),
            max_components=3,
        )

    payload = driver._jsonable(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact_control": payload["artifact_control"],
                "product_correction_status": payload["product_correction_status"],
                "science_status": payload["science_status"],
                "subbands": [
                    {
                        "index": item["index"],
                        "center_freq_mhz": item["center_freq_mhz"],
                        "off_pulse_null": item["off_pulse_null"],
                        "low_lag_stability": item["low_lag_stability"],
                    }
                    for item in payload["subbands"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["science_status"] == "measurement" else 2


if __name__ == "__main__":
    raise SystemExit(main())
