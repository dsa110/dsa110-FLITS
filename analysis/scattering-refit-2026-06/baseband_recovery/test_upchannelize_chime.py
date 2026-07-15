"""Unit checks for provenance-preserving CHIME detected products."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).with_name("upchannelize_chime.py")


def _module():
    spec = importlib.util.spec_from_file_location("upchannelize_chime", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detected_products_preserve_independent_polarizations_and_stokes_sum():
    module = _module()
    rng = np.random.default_rng(20260714)
    voltages = rng.normal(size=(2, 7, 11)) + 1j * rng.normal(size=(2, 7, 11))

    stokes_i, per_pol = module._detected_products(voltages)

    assert per_pol.shape == (2, 11, 7)
    np.testing.assert_allclose(per_pol[0], np.abs(voltages[0]).T ** 2)
    np.testing.assert_allclose(per_pol[1], np.abs(voltages[1]).T ** 2)
    np.testing.assert_allclose(stokes_i, per_pol.sum(axis=0))


def test_detected_products_rejects_missing_polarization_axis():
    module = _module()

    with np.testing.assert_raises_regex(ValueError, "shape"):
        module._detected_products(np.ones((8, 16), dtype=complex))
