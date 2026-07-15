from __future__ import annotations

import numpy as np

from scintillation.scint_analysis.reference_arc.replay_falsification import (
    legacy_positive_acf,
    matched_off_windows,
)


def _slow_old_helper(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(x, dtype=float) - np.mean(x)
    ks = np.arange(2, max_lag + 1)
    values = []
    for k in ks:
        left = centered[:-k]
        right = centered[k:]
        values.append(np.sum(left * right) / np.sqrt(np.sum(left**2) * np.sum(right**2)))
    return ks, np.asarray(values)


def test_fft_acf_matches_archived_helper_arithmetic():
    rng = np.random.default_rng(20260714)
    spectrum = rng.normal(size=128)
    expected_k, expected = _slow_old_helper(spectrum, 32)
    actual_k, actual = legacy_positive_acf(spectrum, 32)
    np.testing.assert_array_equal(actual_k, expected_k)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_matched_windows_exclude_burst_and_buffer():
    windows = matched_off_windows(6250, (3050, 3200), max_windows=24)
    assert len(windows) == 24
    assert all(stop <= 2750 or start >= 3500 for start, stop in windows)
    assert all(stop - start == 150 for start, stop in windows)
