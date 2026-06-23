"""Regression tests for `_JointPriorTransformOrdered` per-band `dt_min`.

The ordered transform sorts each band's t0 group and enforces a minimum component
separation. `dt_min` may be a scalar (broadcast to every group) or one float per
group (per-band floors). These tests assert the realized prior draws honor each
group's own floor, that the scalar path is unchanged, the degenerate-width branch
still collapses, and a length-mismatch sequence fails fast.
"""

import numpy as np
import pytest

from scattering.scat_analysis.burstfit_joint import _JointPriorTransformOrdered


def _spec(ndim, lo=0.0, hi=10.0):
    # spec entry = (name, (lo, hi), is_log_flag); only bounds + flag are read.
    return [(f"t0_{i}", (lo, hi), False) for i in range(ndim)]


def _min_gap(x, grp):
    g = np.sort(x[grp])
    return float(np.min(np.diff(g))) if g.size > 1 else np.inf


def test_per_group_floor_honored():
    """Each group's realized gaps respect ITS OWN dt_min, not the other group's."""
    groups = [np.array([0, 1]), np.array([2, 3, 4])]  # n=2 (C), n=3 (D)
    dtC, dtD = 2.0, 1.0
    tf = _JointPriorTransformOrdered(_spec(5), groups, dt_min=[dtC, dtD])
    assert tf.dt_min == [dtC, dtD]

    rng = np.random.default_rng(0)
    for _ in range(20000):
        x = tf(rng.random(5))
        assert _min_gap(x, groups[0]) >= dtC - 1e-9
        assert _min_gap(x, groups[1]) >= dtD - 1e-9


def test_scalar_broadcast_matches_uniform_sequence():
    """A scalar dt_min applies the same floor to every group (backward-compat)."""
    groups = [np.array([0, 1]), np.array([2, 3, 4])]
    s = 1.5
    tf_scalar = _JointPriorTransformOrdered(_spec(5), groups, dt_min=s)
    tf_seq = _JointPriorTransformOrdered(_spec(5), groups, dt_min=[s, s])
    assert tf_scalar.dt_min == [s, s]

    rng = np.random.default_rng(7)
    for _ in range(2000):
        u = rng.random(5)
        np.testing.assert_allclose(tf_scalar(u), tf_seq(u), rtol=0, atol=0)
        x = tf_scalar(u)
        assert _min_gap(x, groups[0]) >= s - 1e-9
        assert _min_gap(x, groups[1]) >= s - 1e-9


def test_degenerate_width_collapses_group():
    """A group too narrow for n separated comps collapses to a single point."""
    groups = [np.array([0, 1, 2])]  # n=3
    # usable = hi - lo - (n-1)*dtm = 1 - 0 - 2*1 = -1 <= 0 -> collapse branch.
    tf = _JointPriorTransformOrdered(_spec(3, lo=0.0, hi=1.0), groups, dt_min=1.0)
    rng = np.random.default_rng(3)
    for _ in range(1000):
        x = tf(rng.random(3))
        np.testing.assert_allclose(x[groups[0]], x[groups[0]][0], rtol=0, atol=0)


def test_length_mismatch_raises():
    """A per-group sequence whose length != #groups fails fast."""
    groups = [np.array([0, 1]), np.array([2, 3])]
    with pytest.raises(ValueError, match="dt_min sequence length"):
        _JointPriorTransformOrdered(_spec(4), groups, dt_min=[1.0, 2.0, 3.0])


def test_single_component_group_is_skipped():
    """An n<2 group is passed through untouched (no separation to enforce)."""
    groups = [np.array([0]), np.array([1, 2])]
    tf = _JointPriorTransformOrdered(_spec(3), groups, dt_min=[5.0, 1.0])
    u = np.array([0.5, 0.2, 0.8])
    x = tf(u)
    # group 0 (n=1): linear map of u[0] -> lo + u*(hi-lo) = 0 + 0.5*10 = 5.0, untouched.
    assert x[0] == pytest.approx(5.0)
    assert _min_gap(x, groups[1]) >= 1.0 - 1e-9
