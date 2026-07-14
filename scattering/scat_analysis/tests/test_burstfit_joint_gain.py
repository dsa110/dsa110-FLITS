import sys
from types import SimpleNamespace

import numpy as np

from scattering.scat_analysis.burstfit import FRBModel, FRBParams
from scattering.scat_analysis.burstfit_joint import (
    JOINT_PARAM_NAMES_GAIN,
    JOINT_PARAM_NAMES_GAIN_MULTI,
    _gain_marginal_multi_band,
    fit_joint_scattering,
)


def _toy_model() -> tuple[FRBModel, list[FRBParams]]:
    time = np.linspace(-1.5, 1.5, 17)
    freq = np.array([0.9, 1.1, 1.35])
    base = FRBModel(time=time, freq=freq, dm_init=0.0)
    params = [
        FRBParams(c0=1.0, t0=-0.22, gamma=0.0, zeta=0.16, tau_1ghz=0.08, beta=3.5),
        FRBParams(c0=1.0, t0=0.31, gamma=0.0, zeta=0.13, tau_1ghz=0.08, beta=3.5),
    ]
    data = (
        0.7 * base(params[0], "M3")
        - 0.25 * base(params[1], "M3")
        + 0.03 * np.arange(freq.size)[:, None]
        + 0.01 * np.arange(time.size)[None, :]
    )
    model = FRBModel(
        time=time,
        freq=freq,
        data=data,
        dm_init=0.0,
        noise_std=np.array([0.19, 0.23, 0.31]),
    )
    return model, params


def _brute_force_gain_evidence(
    model: FRBModel,
    params: list[FRBParams],
    s2: float,
) -> float:
    kernels = np.stack([model(p, "M3", freq_subset=model.valid) for p in params])
    data = model.data[model.valid]
    var = np.square(model.noise_std[model.valid])

    total = 0.0
    for channel in range(data.shape[0]):
        k = kernels[:, channel, :].T
        cov = var[channel] * np.eye(data.shape[1]) + s2 * (k @ k.T)
        sign, logdet = np.linalg.slogdet(2.0 * np.pi * cov)
        assert sign > 0.0
        total += -0.5 * data[channel] @ np.linalg.solve(cov, data[channel])
        total += -0.5 * logdet
    return float(total)


def test_gain_marginal_multi_band_matches_brute_force_gaussian_evidence():
    model, params = _toy_model()
    s2 = 0.8

    got, diag = _gain_marginal_multi_band(model, params, ["M3", "M3"], s2=s2)
    expected = _brute_force_gain_evidence(model, params, s2=s2)

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
    assert diag["frac_culled"] == 0.0
    assert diag["n_supported"] == model.freq.size


def test_gain_marginal_multi_band_is_label_swap_invariant():
    model, params = _toy_model()

    lnz, _ = _gain_marginal_multi_band(model, params, ["M3", "M3"], s2=1.3)
    swapped, _ = _gain_marginal_multi_band(model, list(reversed(params)), ["M3", "M3"], s2=1.3)

    np.testing.assert_allclose(swapped, lnz, rtol=1e-12, atol=1e-12)


def _install_fake_dynesty(monkeypatch):
    captured = {}

    class FakeNestedSampler:
        def __init__(self, loglike, prior_transform, ndim, **kwargs):
            captured["loglike"] = loglike
            captured["prior_transform"] = prior_transform
            captured["ndim"] = ndim
            self.prior_transform = prior_transform

        def run_nested(self, **kwargs):
            sample = self.prior_transform(np.full(captured["ndim"], 0.5))
            self.results = SimpleNamespace(
                samples=np.array([sample]),
                logwt=np.array([0.0]),
                logz=np.array([0.0]),
                logzerr=np.array([0.0]),
                ncall=np.array([1]),
            )

    monkeypatch.setitem(sys.modules, "dynesty", SimpleNamespace(NestedSampler=FakeNestedSampler))
    return captured


def test_fixed_gain_s2_routes_single_component_fit_to_proper_gain_prior(monkeypatch):
    captured = _install_fake_dynesty(monkeypatch)

    model_C, params_C = _toy_model()
    model_D, params_D = _toy_model()
    result = fit_joint_scattering(
        model_C=model_C,
        init_C=params_C[0],
        model_D=model_D,
        init_D=params_D[0],
        marginalize_gain=True,
        components_C=1,
        components_D=1,
        gain_s2=0.8,
        nlive=10,
        verbose=False,
    )

    assert result["param_names"] == list(JOINT_PARAM_NAMES_GAIN_MULTI(1, 1))
    assert captured["ndim"] == 8


def test_single_component_gain_marginal_path_stays_legacy_without_proper_prior(monkeypatch):
    captured = _install_fake_dynesty(monkeypatch)

    model_C, params_C = _toy_model()
    model_D, params_D = _toy_model()
    result = fit_joint_scattering(
        model_C=model_C,
        init_C=params_C[0],
        model_D=model_D,
        init_D=params_D[0],
        marginalize_gain=True,
        components_C=1,
        components_D=1,
        nlive=10,
        verbose=False,
    )

    assert result["param_names"] == list(JOINT_PARAM_NAMES_GAIN)
    assert captured["ndim"] == 8


def test_fixed_delta_dms_are_removed_from_sampling_and_injected_into_likelihood(monkeypatch):
    captured = _install_fake_dynesty(monkeypatch)

    model_C, params_C = _toy_model()
    model_D, params_D = _toy_model()
    result = fit_joint_scattering(
        model_C=model_C,
        init_C=params_C[0],
        model_D=model_D,
        init_D=params_D[0],
        marginalize_gain=True,
        components_C=2,
        components_D=2,
        fixed_delta_dm_C=0.125,
        fixed_delta_dm_D=-0.375,
        nlive=10,
        verbose=False,
    )

    assert "delta_dm_C" not in result["param_names"]
    assert "delta_dm_D" not in result["param_names"]
    assert captured["ndim"] == len(JOINT_PARAM_NAMES_GAIN_MULTI(2, 2)) - 2
    assert result["fixed_parameters"] == {
        "delta_dm_C": 0.125,
        "delta_dm_D": -0.375,
    }
    assert result["percentiles"]["delta_dm_C"]["median"] == 0.125
    assert result["percentiles"]["delta_dm_D"]["median"] == -0.375

    # The reduced midpoint vector must be accepted by the wrapped likelihood;
    # internally it is expanded back to the original multi-component layout.
    theta = captured["prior_transform"](np.full(captured["ndim"], 0.5))
    assert np.isfinite(captured["loglike"](theta))
