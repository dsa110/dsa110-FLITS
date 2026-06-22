"""Regression tests for `_gain_marginal_multi_band` (per-channel gain-marginal
evidence for one band).

The function integrates per-component gains g ~ N(0, s2 I_N) analytically. The
key correctness statement (asserted by `test_brute_force_woodbury`) is the
Woodbury identity it is built on: for each channel f with kernels K_f (N, T),
data d_f (T,), noise variance sigma_f^2, the analytic per-channel evidence equals
the direct Gaussian evidence of d_f under the marginal covariance

    Sigma_f = sigma_f^2 I_T + s2 * K_f^T K_f          (T, T)
    lnZ_f   = -0.5 d_f^T Sigma_f^-1 d_f - 0.5 ln det(2 pi Sigma_f)

summed over valid channels. We use a DUCK-TYPED fake model that returns
PRESCRIBED component kernels so the brute side is exactly computable and decoupled
from the real forward model. s2 is passed as a fixed float so the s2-profiling
path is never exercised.
"""

import numpy as np
import pytest

from scattering.scat_analysis.burstfit import FRBParams
from scattering.scat_analysis.burstfit_joint import _gain_marginal_multi_band


class _FakeModel:
    """Minimal duck-typed stand-in for FRBModel.

    `_gain_marginal_multi_band` only touches `.data`, `.noise_std`, `.valid`, and
    `model(p, mk, freq_subset=valid)`. We ignore the params/model_key entirely and
    return a pre-baked per-component kernel cube indexed by call order, so the
    kernels in the evidence math are exactly what the test prescribes.

    kernels: (N, F, T) -- component, full-frequency, time. data: (F, T).
    noise_std: (F,). valid: (F,) bool.
    """

    def __init__(self, kernels, data, noise_std, valid):
        self._kernels = np.asarray(kernels, dtype=float)  # (N, F, T)
        self.data = np.asarray(data, dtype=float)  # (F, T)
        self.noise_std = np.asarray(noise_std, dtype=float)  # (F,)
        self.valid = np.asarray(valid, dtype=bool)  # (F,)
        self._i = 0

    def __call__(self, p, model_key, freq_subset=None):
        # The function calls us once per component, in params_list order. Hand back
        # that component's kernel rows for the requested frequency subset.
        k = self._kernels[self._i]
        self._i += 1
        if freq_subset is not None:
            k = k[freq_subset]
        return k


def _brute_lnZ(kernels, data, noise_std, valid, s2):
    """Direct (non-Woodbury) Gaussian evidence summed over valid channels.

    Per channel: Sigma = sigma^2 I_T + s2 * K^T K  (K is (N, T)); the marginal of
    d ~ N(0, Sigma). Uses slogdet + solve so it is an INDEPENDENT implementation of
    the same quantity the function computes via the Woodbury form.
    """
    K_all = np.asarray(kernels, dtype=float)  # (N, F, T)
    N, F, T = K_all.shape
    total = 0.0
    for f in range(F):
        if not valid[f]:
            continue
        K = K_all[:, f, :]  # (N, T)
        d = np.asarray(data)[f]  # (T,)
        sig2 = float(noise_std[f]) ** 2
        Sigma = sig2 * np.eye(T) + s2 * (K.T @ K)  # (T, T)
        sign, logdet = np.linalg.slogdet(Sigma)
        assert sign > 0, "Sigma must be SPD"
        quad = d @ np.linalg.solve(Sigma, d)
        lnZ_f = -0.5 * quad - 0.5 * (logdet + T * np.log(2.0 * np.pi))
        total += lnZ_f
    return total


def _make_well_conditioned_case(seed, N, F, T, scale=1.0):
    """Seeded random kernels/data/noise with full-rank M_f on every channel.

    Random (N, T) Gaussian kernels are generically full row-rank for T > N, so the
    eigenvalue guard does not fire and the well-conditioned path is exercised.
    """
    rng = np.random.default_rng(seed)
    kernels = rng.standard_normal((N, F, T)) * scale
    data = rng.standard_normal((F, T)) * scale
    noise_std = 0.5 + rng.random(F)  # in [0.5, 1.5), strictly > 1e-9
    valid = np.ones(F, dtype=bool)
    return kernels, data, noise_std, valid


# Dummy params/keys: the fake model ignores their CONTENT, but the function calls
# `replace(p, c0=1.0, gamma=0.0)` on each, so they must be real FRBParams dataclass
# instances (and there must be N of them to drive N component calls).
def _dummy_params_keys(N):
    p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.1, tau_1ghz=0.1, alpha=4.0)
    return [p] * N, ["M3"] * N


def test_brute_force_woodbury():
    """(a) Total lnZ equals the direct brute-force Gaussian evidence.

    This is the load-bearing correctness check: the function's Woodbury-collapsed
    per-channel evidence must equal -0.5 d^T Sigma^-1 d - 0.5 ln det(2 pi Sigma)
    summed over valid channels, with Sigma = sigma^2 I + s2 K^T K. If this fails,
    investigate the kernel/construction or a real discrepancy -- do NOT loosen tol.
    """
    N, F, T = 2, 3, 8
    s2 = 0.7
    kernels, data, noise_std, valid = _make_well_conditioned_case(seed=1234, N=N, F=F, T=T)
    model = _FakeModel(kernels, data, noise_std, valid)
    params, keys = _dummy_params_keys(N)

    lnZ, diag = _gain_marginal_multi_band(model, params, keys, s2=s2)
    brute = _brute_lnZ(kernels, data, noise_std, valid, s2)

    assert np.isfinite(lnZ)
    assert diag["frac_culled"] == 0.0
    assert diag["n_supported"] == F
    assert diag["s2"] == pytest.approx(s2)
    np.testing.assert_allclose(lnZ, brute, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_brute_force_woodbury_varied(seed):
    """(a, extended) Woodbury cross-check holds across several seeds and a larger
    component count -- guards against an accidental construction that only matches
    for one particular draw."""
    N, F, T = 3, 4, 12
    s2 = 1.3
    kernels, data, noise_std, valid = _make_well_conditioned_case(seed=seed, N=N, F=F, T=T)
    model = _FakeModel(kernels, data, noise_std, valid)
    params, keys = _dummy_params_keys(N)

    lnZ, _ = _gain_marginal_multi_band(model, params, keys, s2=s2)
    brute = _brute_lnZ(kernels, data, noise_std, valid, s2)
    np.testing.assert_allclose(lnZ, brute, rtol=1e-9, atol=1e-9)


def test_label_swap_invariance():
    """(b) Permuting the component order leaves total lnZ unchanged.

    The marginal evidence depends on the components only through K^T K (and b, M),
    which are symmetric under relabelling. Swapping the component axis of the kernel
    cube (and the params/keys we feed in the same order) must give the same lnZ.
    """
    N, F, T = 3, 4, 10
    s2 = 0.9
    kernels, data, noise_std, valid = _make_well_conditioned_case(seed=2024, N=N, F=F, T=T)
    perm = [2, 0, 1]
    kernels_perm = kernels[perm]

    lnZ_a, _ = _gain_marginal_multi_band(
        _FakeModel(kernels, data, noise_std, valid), *_dummy_params_keys(N), s2=s2
    )
    lnZ_b, _ = _gain_marginal_multi_band(
        _FakeModel(kernels_perm, data, noise_std, valid),
        *_dummy_params_keys(N),
        s2=s2,
    )
    np.testing.assert_allclose(lnZ_a, lnZ_b, rtol=1e-12, atol=1e-12)


def test_rank1_fallback_on_collinear_channel():
    """(c) A collinear channel triggers the eigenvalue guard's rank-1 fallback.

    We make the two components IDENTICAL on channel 0 (K_2 == K_1 there), so M_0 is
    rank-deficient (min/max eig ~ 0 < eig_rel_floor) and the channel is culled to a
    rank-1 proper-prior evidence on its top eigenpair (NOT the gain=0 baseline).

    Crucially, on an EXACTLY-collinear channel the fallback is not an approximation:
    with both components equal to k(t), K_0^T K_0 = 2 k k^T = k_eff k_eff^T for the
    single effective kernel k_eff = sqrt(2) k = v_top . K, so the channel's true
    marginal covariance Sigma_0 = sigma^2 I + s2 K^T K is itself rank-1-plus-diagonal.
    The full (T,T) brute force -- the SAME `_brute_lnZ` used in (a), with no reference
    to the kernel's reduced rank-1 formula -- is therefore the exact ground truth on
    every channel, culled or not. We also assert the culled channel's contribution
    sits strictly BELOW its gain=0 baseline (the merge is Occam-penalized, not
    rewarded), the invariant the rank-1 branch exists to protect.
    """
    N, F, T = 2, 3, 9
    s2 = 1e6  # large fixed s2: the regime where culling-to-baseline would REWARD a merge
    # Well-conditioned channels 1,2; channel 0 collinear (component 2 == component 1).
    kernels, data, noise_std, valid = _make_well_conditioned_case(seed=99, N=N, F=F, T=T)
    kernels[1, 0, :] = kernels[0, 0, :]  # collapse component 2 onto 1 on channel 0

    model = _FakeModel(kernels, data, noise_std, valid)
    lnZ, diag = _gain_marginal_multi_band(model, *_dummy_params_keys(N), s2=s2)

    assert np.isfinite(lnZ)
    # Exactly one channel (channel 0) is not full-rank-N -> culled.
    assert diag["frac_culled"] == pytest.approx(1.0 / F)
    assert diag["n_supported"] == F - 1  # the two well-conditioned channels

    # Independent ground truth: the FULL (T,T) brute force over all valid channels
    # (same _brute_lnZ as test (a)), exact even on the rank-1-collapsed channel 0
    # because Sigma_0 = sigma^2 I + s2 K^T K is itself rank-1-plus-diagonal there.
    brute = _brute_lnZ(kernels, data, noise_std, valid, s2)
    np.testing.assert_allclose(lnZ, brute, rtol=1e-7, atol=1e-7)

    # Occam invariant: the culled channel's own contribution is strictly below its
    # gain=0 baseline. Evaluate that single channel via the same independent brute.
    cull_only = np.zeros(F, dtype=bool)
    cull_only[0] = True
    brute_cull = _brute_lnZ(kernels, data, noise_std, cull_only, s2)
    sig2_0 = float(noise_std[0]) ** 2
    S_dd0 = float(data[0] @ data[0])
    baseline_cull = -0.5 * S_dd0 / sig2_0 - 0.5 * T * np.log(2.0 * np.pi * sig2_0)
    assert brute_cull < baseline_cull


def test_s2_profiling_finds_interior_optimum():
    """The production-default `s2=None` ML-profiling path (1-D max over the gain-prior
    variance) satisfies two implementation-independent properties:

      1. Profiled evidence >= evidence at ANY fixed s2 (profiling maximizes).
      2. The optimum is INTERIOR: perturbing the profiled s2 up or down does not
         increase lnZ -- i.e. it is not pinned to a search bound.

    A fresh _FakeModel per call is required (the fake's component-call counter is
    consumed once per _gain_marginal_multi_band invocation).
    """
    N, F, T = 2, 4, 16
    kernels, data, noise_std, valid = _make_well_conditioned_case(seed=7, N=N, F=F, T=T)

    def fn(s2):
        return _gain_marginal_multi_band(
            _FakeModel(kernels, data, noise_std, valid), *_dummy_params_keys(N), s2=s2
        )

    lnZ_prof, diag = fn(None)
    s2_star = diag["s2"]
    assert np.isfinite(lnZ_prof) and s2_star > 0.0

    # (1) >= a couple of arbitrary fixed values near the optimum's scale.
    for s2_fixed in (s2_star * 3.0, s2_star / 3.0):
        lnZ_fixed, _ = fn(s2_fixed)
        assert lnZ_prof >= lnZ_fixed - 1e-6
    # (2) interior: a 10x step either way (well inside the +-18-in-log search range)
    # does not beat the profiled optimum.
    for s2_pert in (s2_star * 10.0, s2_star / 10.0):
        lnZ_pert, _ = fn(s2_pert)
        assert lnZ_prof >= lnZ_pert - 1e-6


def test_no_valid_channels_returns_neg_inf():
    """All-invalid band -> -inf with frac_culled=1.0 (boundary contract)."""
    N, F, T = 2, 3, 6
    kernels, data, noise_std, _ = _make_well_conditioned_case(seed=5, N=N, F=F, T=T)
    valid = np.zeros(F, dtype=bool)
    model = _FakeModel(kernels, data, noise_std, valid)
    lnZ, diag = _gain_marginal_multi_band(model, *_dummy_params_keys(N), s2=1.0)
    assert lnZ == -np.inf
    assert diag["frac_culled"] == 1.0
    assert diag["n_supported"] == 0


def test_valid_mask_subsets_channels():
    """A partial `valid` mask must restrict the evidence to the unmasked channels.

    Confirms the function honours `valid` (passes it as freq_subset to the model
    and indexes data/noise by it) -- lnZ equals the brute force over valid only.
    """
    N, F, T = 2, 4, 8
    s2 = 1.1
    kernels, data, noise_std, _ = _make_well_conditioned_case(seed=321, N=N, F=F, T=T)
    valid = np.array([True, False, True, True])
    model = _FakeModel(kernels, data, noise_std, valid)
    lnZ, diag = _gain_marginal_multi_band(model, *_dummy_params_keys(N), s2=s2)
    brute = _brute_lnZ(kernels, data, noise_std, valid, s2)
    np.testing.assert_allclose(lnZ, brute, rtol=1e-9, atol=1e-9)
    assert diag["n_supported"] == int(np.count_nonzero(valid))
