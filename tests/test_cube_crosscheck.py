import numpy as np

from scripts.cube_crosscheck import best_lag_bins, resample_profile


def test_resample_profile_preserves_endpoints_and_length():
    profile = np.array([1.0, 3.0, 5.0])

    out = resample_profile(profile, 5)

    assert out.shape == (5,)
    assert out[0] == profile[0]
    assert out[-1] == profile[-1]


def test_best_lag_bins_recovers_small_positive_shift():
    x = np.linspace(-1.0, 1.0, 101)
    ref = np.exp(-0.5 * (x / 0.08) ** 2)
    cube = np.zeros_like(ref)
    cube[3:] = ref[:-3]

    assert best_lag_bins(cube, ref, max_lag=10) == 3
