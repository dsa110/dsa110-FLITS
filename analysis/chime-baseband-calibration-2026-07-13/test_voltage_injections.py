"""Local regression tests for the Freya B1 voltage-injection harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).with_name("run_voltage_injections.py")
FINALIZER = Path(__file__).with_name("finalize_freya_b1_review.py")


def _module(path=RUNNER):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replay_provenance_rejects_a_different_hdf5(tmp_path):
    module = _module()
    expected_h5 = tmp_path / "expected.h5"
    other_h5 = tmp_path / "other.h5"
    canonical_waterfall = tmp_path / "canonical-waterfall.npy"
    canonical_frequency = tmp_path / "canonical-frequency.npy"
    replay_waterfall = tmp_path / "replay-waterfall.npy"
    replay_frequency = tmp_path / "replay-frequency.npy"
    expected_h5.write_bytes(b"expected voltage")
    other_h5.write_bytes(b"different voltage")
    canonical_waterfall.write_bytes(b"waterfall")
    replay_waterfall.write_bytes(b"waterfall")
    canonical_frequency.write_bytes(b"frequency")
    replay_frequency.write_bytes(b"frequency")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "input_h5": {"sha256": module._sha256(expected_h5)},
                "baseline_replay": {
                    "waterfall_sha256": module._sha256(canonical_waterfall),
                    "frequency_sha256": module._sha256(canonical_frequency),
                },
            }
        )
    )

    with np.testing.assert_raises_regex(ValueError, "frozen provenance"):
        module._verify_replay_provenance(
            other_h5,
            provenance,
            canonical_waterfall,
            canonical_frequency,
            replay_waterfall,
            replay_frequency,
        )


def test_manual_review_requires_overall_authorization():
    module = _module(FINALIZER)
    manifest = {"figures": [{"path": "figure.svg"}]}
    review = {
        "figures": [{"path": "figure.svg", "verdict": "match"}],
        "overall_verdict": "match",
        "qualification_authorized": False,
    }

    assert module._manual_review_pass(manifest, review) is False
    review["qualification_authorized"] = True
    assert module._manual_review_pass(manifest, review) is True


def test_pending_manual_review_cannot_pass_qualification():
    module = _module()
    checks = {
        "automated": {"pass": True},
        "manual_review": {"pass": None},
    }

    assert module._qualification_pass(checks) is False
    checks["manual_review"]["pass"] = True
    assert module._qualification_pass(checks) is True


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
