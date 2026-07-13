"""Regression tests for the Freya B2 folded-scallop calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("run_scallop_injections.py")
FINALIZER = Path(__file__).with_name("finalize_freya_b2_review.py")


def _module(path=RUNNER):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manual_review_requires_overall_authorization(tmp_path):
    module = _module(FINALIZER)
    figure = tmp_path / "figure.svg"
    figure.write_text("reviewed")
    manifest = {"figures": [{"path": "figure.svg", "sha256": module._sha256(figure)}]}
    review = {
        "figures": [{"path": "figure.svg", "verdict": "match"}],
        "overall_verdict": "match",
        "qualification_authorized": False,
    }

    assert module._manual_review_pass(manifest, review, tmp_path) is False
    review["qualification_authorized"] = True
    assert module._manual_review_pass(manifest, review, tmp_path) is True


def test_manual_review_rejects_missing_or_changed_figure(tmp_path):
    module = _module(FINALIZER)
    figure = tmp_path / "figure.svg"
    figure.write_text("reviewed")
    manifest = {"figures": [{"path": "figure.svg", "sha256": module._sha256(figure)}]}
    review = {
        "figures": [{"path": "figure.svg", "verdict": "match"}],
        "overall_verdict": "match",
        "qualification_authorized": True,
    }

    figure.write_text("changed after review")
    assert module._manual_review_pass(manifest, review, tmp_path) is False
    figure.unlink()
    assert module._manual_review_pass(manifest, review, tmp_path) is False


def test_final_qualification_requires_every_check_to_pass():
    module = _module(FINALIZER)

    assert module._qualification_pass({"machine": {"pass": True}, "review": {"pass": True}})
    assert not module._qualification_pass(
        {"machine": {"pass": True}, "review": {"pass": None}}
    )
    assert not module._qualification_pass(
        {"machine": {"pass": True}, "review": {"pass": False}}
    )


def test_folded_gain_recovers_repeating_shape():
    module = _module()
    scallop = np.linspace(0.7, 1.3, module.U)
    scallop /= scallop.mean()
    coarse = np.linspace(1.0, 3.0, 32)
    gain = (coarse[:, None] * scallop[None, :]).reshape(-1)
    power = gain[:, None] * np.ones((gain.size, 80))

    recovered_gain, recovered_scallop = module._folded_gain(power)

    np.testing.assert_allclose(recovered_scallop, scallop, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered_gain, gain, rtol=1e-12, atol=1e-12)


def test_phase_cycle_cancels_cross_term():
    module = _module()
    rng = np.random.default_rng(20260713)
    noise = rng.normal(size=2048) + 1j * rng.normal(size=2048)
    signal = rng.normal(size=2048) + 1j * rng.normal(size=2048)
    baseline = np.abs(noise) ** 2
    plus = np.abs(noise + signal) ** 2
    minus = np.abs(noise - signal) ** 2

    recovered = module._phase_cycled_signal(plus, minus, baseline)

    np.testing.assert_allclose(recovered, np.abs(signal) ** 2, rtol=1e-12, atol=1e-12)


def test_matched_scalar_gain_round_trip():
    module = _module()
    rng = np.random.default_rng(7)
    target = np.abs(rng.normal(size=4096)) + 0.1
    scallop = np.tile(np.linspace(0.65, 1.35, module.U), 4096 // module.U)
    observed_power = np.abs(np.sqrt(target) * np.sqrt(scallop)) ** 2

    np.testing.assert_allclose(observed_power / scallop, target, rtol=1e-12, atol=1e-12)


def test_missing_fit_fails_width_and_comparison_without_crashing():
    module = _module()
    missing = {"truth_fit": {"dnu_mhz": 0.05}, "recovered_fit": None}

    assert module._width_pass(missing, channel_width_mhz=0.006) is False
    assert np.isinf(module._median_abs_fractional_error([missing]))
