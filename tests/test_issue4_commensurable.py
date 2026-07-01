"""Issue #4 acceptance: N=1 joint evidence is commensurable with N>=2.

`burstfit_joint._gain_marginal_multi_band` routes BOTH the single-component (N=1)
and multi-component (N>=2) cases through the same proper-prior path, which carries
the FULL data normalization ``-0.5*T*ln(2 pi sigma^2)`` per channel plus the
finite-variance Occam term. `fit_joint_scattering(force_multi=True)` opts N=1 into
this path (the gate at ``burstfit_joint.py:878``:
``multi = bool(force_multi) or int(components_C) > 1 or int(components_D) > 1``).

Issue #4's Acceptance is "N=1 produces a lnZ on the same additive scale as N>=2 via
the multi path", proven by "a regression test that asserts N=1-via-multi-path agrees
with a direct N=1 brute-force Gaussian evidence at small T". The N>=2 Woodbury
regression lives in ``tests/test_gain_marginal_multi_band.py``; this file pins the
N=1 case it does not cover, reusing the same duck-typed model + independent
brute-force evidence so the check is decoupled from the real forward model.
"""

import numpy as np
import pytest

from scattering.scat_analysis.burstfit import FRBParams
from scattering.scat_analysis.burstfit_joint import _gain_marginal_multi_band


class _FakeModel:
    """Duck-typed FRBModel: returns prescribed per-component kernels in call order.

    `_gain_marginal_multi_band` only touches `.data`, `.noise_std`, `.valid`, and
    `model(p, mk, freq_subset=valid)`. Mirrors the stand-in in
    `tests/test_gain_marginal_multi_band.py` so the brute side is exactly computable.
    """

    def __init__(self, kernels, data, noise_std, valid):
        self._kernels = np.asarray(kernels, dtype=float)  # (N, F, T)
        self.data = np.asarray(data, dtype=float)  # (F, T)
        self.noise_std = np.asarray(noise_std, dtype=float)  # (F,)
        self.valid = np.asarray(valid, dtype=bool)  # (F,)
        self._i = 0

    def __call__(self, p, model_key, freq_subset=None):
        k = self._kernels[self._i]
        self._i += 1
        return k if freq_subset is None else k[freq_subset]


def _brute_lnZ(kernels, data, noise_std, valid, s2):
    """Direct Gaussian evidence summed over valid channels (independent of Woodbury).

    Per channel f: marginal of d_f ~ N(0, Sigma_f), Sigma_f = sigma_f^2 I_T + s2 K^T K.
    Carries the full ``-0.5*(ln det(2 pi Sigma_f))`` normalization, so matching it
    proves the multi-path evidence is on the proper additive scale.
    """
    K_all = np.asarray(kernels, dtype=float)
    N, F, T = K_all.shape
    total = 0.0
    for f in range(F):
        if not valid[f]:
            continue
        K = K_all[:, f, :]  # (N, T)
        d = np.asarray(data)[f]  # (T,)
        sig2 = float(noise_std[f]) ** 2
        Sigma = sig2 * np.eye(T) + s2 * (K.T @ K)
        sign, logdet = np.linalg.slogdet(Sigma)
        assert sign > 0, "Sigma must be SPD"
        total += -0.5 * (d @ np.linalg.solve(Sigma, d)) - 0.5 * (logdet + T * np.log(2.0 * np.pi))
    return total


def _params_keys(N):
    # Content is ignored (the fake model returns pre-baked kernels), but the function
    # calls replace(p, c0=1.0, gamma=0.0) so they must be real FRBParams, N of them.
    p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.1, tau_1ghz=0.1, beta=4.0)
    return [p] * N, ["M3"] * N


def _case(seed, N, F, T):
    """Seeded random kernels/data/noise, full-rank M_f on every channel (T > N)."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((N, F, T)),
        rng.standard_normal((F, T)),
        0.5 + rng.random(F),  # noise_std in [0.5, 1.5)
        np.ones(F, dtype=bool),
    )


def test_n1_multi_matches_brute_force_small_T():
    """N=1 via the multi path equals the direct N=1 brute-force Gaussian evidence."""
    N, F, T, s2 = 1, 3, 8, 0.7
    ker, dat, nstd, valid = _case(1234, N, F, T)
    lnZ, diag = _gain_marginal_multi_band(
        _FakeModel(ker, dat, nstd, valid), *_params_keys(N), s2=s2
    )
    brute = _brute_lnZ(ker, dat, nstd, valid, s2)
    assert np.isfinite(lnZ)
    assert diag["n_supported"] == F
    assert diag["s2"] == pytest.approx(s2)
    np.testing.assert_allclose(lnZ, brute, rtol=1e-9, atol=1e-9)


def test_n1_and_n2_share_additive_scale():
    """N=1 and N=2 on the SAME data each equal their brute-force evidence, so both
    carry the full ``-0.5*T*ln(2 pi sigma^2)`` term and their difference is a pure
    Occam/Bayes factor -- the commensurability issue #4 requires. (Pre-fix, the N=1
    flat-improper path omitted that term, putting N=1 on a different additive scale.)
    """
    F, T, s2 = 4, 10, 0.9
    ker2, dat, nstd, valid = _case(2025, 2, F, T)
    ker1 = ker2[:1]  # N=1 is the first component of the same data
    z1, _ = _gain_marginal_multi_band(_FakeModel(ker1, dat, nstd, valid), *_params_keys(1), s2=s2)
    z2, _ = _gain_marginal_multi_band(_FakeModel(ker2, dat, nstd, valid), *_params_keys(2), s2=s2)
    np.testing.assert_allclose(z1, _brute_lnZ(ker1, dat, nstd, valid, s2), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(z2, _brute_lnZ(ker2, dat, nstd, valid, s2), rtol=1e-9, atol=1e-9)
    assert np.isfinite(z1) and np.isfinite(z2)


def test_force_multi_routes_n1_through_multi_loglike(monkeypatch):
    """Routing acceptance for #4: ``fit_joint_scattering(force_multi=True)`` with
    ``components_C=components_D=1`` must hand the *multi-component* gain likelihood
    (``_JointLogLikelihoodGainMulti``, ndim ``= len(JOINT_PARAM_NAMES_GAIN_MULTI(1,1))``)
    to the sampler -- NOT the single-component path. The brute-force tests above pin
    the evidence algebra; this pins the router gate (``burstfit_joint.py:878``) so the
    fix can't silently fall back to a different additive scale. We stub
    ``dynesty.NestedSampler`` to capture the loglike object + ndim it would receive,
    then abort before any real fit runs.
    """
    import dynesty

    from scattering.scat_analysis import burstfit_joint as bj

    captured = {}

    class _RoutedToSampler(Exception):
        pass

    class _StubSampler:
        def __init__(self, loglike, ptform, ndim, **kw):
            captured["loglike_cls"] = type(loglike)
            captured["ndim"] = ndim
            raise _RoutedToSampler

    monkeypatch.setattr(dynesty, "NestedSampler", _StubSampler)

    T = 16

    class _RouteModel:
        data = np.zeros((3, T))  # only `is None` is checked pre-sampler

        def __init__(self):
            self.time = np.arange(T, dtype=float) * 1e-4  # drives the multi-path dt_min floor

    p = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.1, tau_1ghz=0.1, beta=4.0)
    common = dict(
        model_C=_RouteModel(),
        init_C=p,
        model_D=_RouteModel(),
        init_D=p,
        components_C=1,
        components_D=1,
        gain_s2=1.0,
        nlive=5,
        verbose=False,
    )

    with pytest.raises(_RoutedToSampler):
        bj.fit_joint_scattering(force_multi=True, **common)
    assert captured["loglike_cls"] is bj._JointLogLikelihoodGainMulti
    assert captured["ndim"] == len(bj.JOINT_PARAM_NAMES_GAIN_MULTI(1, 1))

    # Contrast: without force_multi, N=1 does NOT take the multi path -- proving the
    # gate flip is force_multi, not an unconditional N=1 reroute.
    captured.clear()
    with pytest.raises(_RoutedToSampler):
        bj.fit_joint_scattering(force_multi=False, **common)
    assert captured["loglike_cls"] is not bj._JointLogLikelihoodGainMulti
