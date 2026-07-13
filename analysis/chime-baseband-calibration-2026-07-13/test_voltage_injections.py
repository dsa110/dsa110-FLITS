"""Local regression tests for the Freya B1 voltage-injection harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("run_voltage_injections.py")


def _module():
    spec = importlib.util.spec_from_file_location("freya_b1", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_periodogram_target_realizes_nominal_width():
    module = _module()
    target = module._make_target(np.random.default_rng(20260713), 8192, 8.0)
    fit = module._fit_width(target)

    assert np.all(target > 0)
    assert fit is not None
    assert np.isclose(
        fit["dnu_mhz"],
        8.0 * module.CHANNEL_WIDTH_MHZ,
        rtol=0.01,
    )


def test_padded_alignment_never_wraps():
    module = _module()
    power = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    aligned = module._align(power, np.asarray([0, 2]))

    np.testing.assert_allclose(aligned[0, :2], [1.0, 2.0])
    np.testing.assert_allclose(aligned[1, 2:], [3.0, 4.0])
    assert np.isnan(aligned[0, 2:]).all()
    assert np.isnan(aligned[1, :2]).all()


def test_acf_preserves_absolute_channel_gaps():
    module = _module()
    full = module._make_target(np.random.default_rng(7), 8192, 8.0)
    ids = np.concatenate((np.arange(3500), np.arange(3600, 8192)))
    fit = module._fit_width(full[ids], ids)

    assert fit is not None
    assert np.isclose(
        fit["dnu_mhz"],
        8.0 * module.CHANNEL_WIDTH_MHZ,
        rtol=0.03,
    )
