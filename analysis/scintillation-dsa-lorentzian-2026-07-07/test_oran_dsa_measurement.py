"""Regression tests for the qualified Oran DSA measurement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATOR = HERE / "validate_oran_dsa_measurement.py"
SPEC = importlib.util.spec_from_file_location("oran_dsa_validator", VALIDATOR)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {VALIDATOR}")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_interval_inversion_enforces_monotonic_tail_probability():
    truths = np.array([0.1, 0.2, 0.4, 0.8])
    noisy_probabilities = np.array([0.92, 0.80, 0.82, 0.10])
    monotonic = validator._monotonic_nonincreasing(noisy_probabilities)
    assert np.all(np.diff(monotonic) <= 0)
    assert np.isclose(
        validator._invert_decreasing_grid(truths, noisy_probabilities, 0.81),
        0.19166666666666668,
    )


def test_published_oran_measurement_is_fully_qualified():
    output = HERE / "results/oran_qualified"
    result = json.loads((output / "validation.json").read_text())
    measurement = result["calibrated_measurement"]
    assert result["machine_status"] == "pass"
    assert all(gate["pass"] for gate in result["gates"].values())
    assert np.isclose(measurement["dnu_mhz"], 0.44624819758756973)
    assert np.allclose(measurement["confidence_interval_68_mhz"], [0.1962, 0.6845])
    assert measurement["confidence_interval_68_mhz"][0] > 4 * result["channel_width_mhz"]
    assert len(result["records"]) == 1024

    review = json.loads((output / "figures.review.json").read_text())
    figure = output / review["figure"]
    assert review["review_status"] == "pass"
    assert _sha256(figure) == review["figure_sha256"]
