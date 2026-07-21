"""Unit checks for off-pulse-only CHIME preprocessing diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).with_name("audit_chime_preprocessing.py")


def _module():
    spec = importlib.util.spec_from_file_location("audit_chime_preprocessing", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bandpass_model_uses_only_requested_interval_and_masks_bad_scales():
    module = _module()
    data = np.array(
        [
            [1.0, 3.0, 1000.0, 1000.0],
            [5.0, 5.0, 7.0, 9.0],
            [1.0, np.nan, 3.0, 4.0],
        ]
    )
    mean, scale, valid = module._bandpass_model(
        data, np.ones(3, dtype=bool), (0, 2), minimum_fraction=0.8
    )

    assert mean[0] == 2.0
    assert np.isclose(scale[0], np.sqrt(2.0))
    assert valid.tolist() == [True, False, False]


def test_normalize_never_fills_invalid_rows_with_zero():
    module = _module()
    data = np.arange(12, dtype=float).reshape(3, 4)
    result = module._normalize(
        data,
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([True, False, True]),
    )

    assert np.isnan(result[1]).all()
    np.testing.assert_array_equal(result[0], (data[0] - 1.0) / 2.0)


def test_disjoint_interval_and_frequency_metrics_are_deterministic():
    module = _module()
    rng = np.random.default_rng(20260721)
    data = rng.normal(size=(16, 20))
    valid = np.ones(16, dtype=bool)

    first = module._metrics(data, valid, (10, 20), 4)
    second = module._metrics(data, valid, (10, 20), 4)

    assert first == second
    assert first["valid_fine_positions"] == 16
    assert set(first["validation_frequency_lag_correlations"]) == {
        str(value) for value in range(1, 11)
    }
