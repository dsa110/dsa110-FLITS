from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scattering.scat_analysis.burstfit import FRBModel, FRBParams
from scattering.scat_analysis.burstfit_joint import fit_joint_scattering
from scattering.scat_analysis.controlled_run import (
    canonical_npz_sha256,
    identity_sha256,
    sha256,
)
from scattering.scat_analysis.joint_fit_diagnostics import (
    build_diagnostics,
    render_fit_panel,
    write_diagnostics,
)
from scattering.scat_analysis.joint_model_grid import build_model_grid_arrays


def _model_and_init() -> tuple[FRBModel, FRBParams]:
    time = np.linspace(-1.0, 1.0, 12)
    freq = np.array([0.9, 1.1])
    init = FRBParams(
        c0=1.0,
        t0=0.0,
        gamma=0.0,
        zeta=0.1,
        tau_1ghz=0.1,
        beta=3.5,
    )
    empty = FRBModel(time=time, freq=freq, dm_init=0.0)
    model = FRBModel(
        time=time,
        freq=freq,
        data=empty(init, "M3"),
        dm_init=0.0,
        noise_std=np.full(freq.size, 0.1),
    )
    return model, init


def test_seed_reaches_serial_dynesty_sampler(monkeypatch) -> None:
    captured: dict[str, object] = {}
    resolved: dict[str, object] = {}

    class FakeNestedSampler:
        def __init__(self, loglike, prior_transform, ndim, **kwargs):
            captured.update(kwargs)
            self.prior_transform = prior_transform
            self.ndim = ndim

        def run_nested(self, **kwargs):
            sample = self.prior_transform(np.full(self.ndim, 0.5))
            self.results = SimpleNamespace(
                samples=np.array([sample]),
                logwt=np.array([0.0]),
                logz=np.array([0.0]),
                logzerr=np.array([0.0]),
                ncall=np.array([1]),
            )

    monkeypatch.setitem(
        sys.modules,
        "dynesty",
        SimpleNamespace(NestedSampler=FakeNestedSampler),
    )
    model_c, init_c = _model_and_init()
    model_d, init_d = _model_and_init()

    result = fit_joint_scattering(
        model_C=model_c,
        init_C=init_c,
        model_D=model_d,
        init_D=init_d,
        nlive=10,
        nproc=1,
        seed=20260722,
        resolved_identity_callback=resolved.update,
        verbose=False,
    )

    expected = np.random.default_rng(20260722).integers(0, 2**32, size=4)
    actual = captured["rstate"].integers(0, 2**32, size=4)
    np.testing.assert_array_equal(actual, expected)
    assert result["seed"] == 20260722
    assert resolved["likelihood_class"] == "_JointLogLikelihood"
    assert resolved["sampler"]["seed"] == 20260722
    assert [prior["name"] for prior in resolved["prior_spec"]] == result["param_names"]
    assert resolved["processed_support"]["C"]["arrays"]["data"]["shape"] == [2, 12]


def test_seed_reaches_multiprocess_dynesty_sampler(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePool:
        def __init__(self, nproc, loglike, prior_transform):
            self.loglike = loglike
            self.prior_transform = prior_transform

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeNestedSampler:
        def __init__(self, loglike, prior_transform, ndim, **kwargs):
            captured.update(kwargs)
            self.prior_transform = prior_transform
            self.ndim = ndim

        def run_nested(self, **kwargs):
            sample = self.prior_transform(np.full(self.ndim, 0.5))
            self.results = SimpleNamespace(
                samples=np.array([sample]),
                logwt=np.array([0.0]),
                logz=np.array([0.0]),
                logzerr=np.array([0.0]),
                ncall=np.array([1]),
            )

    monkeypatch.setitem(
        sys.modules,
        "dynesty",
        SimpleNamespace(
            NestedSampler=FakeNestedSampler,
            pool=SimpleNamespace(Pool=FakePool),
        ),
    )
    model_c, init_c = _model_and_init()
    model_d, init_d = _model_and_init()

    result = fit_joint_scattering(
        model_C=model_c,
        init_C=init_c,
        model_D=model_d,
        init_D=init_d,
        nlive=10,
        nproc=2,
        seed=20260723,
        verbose=False,
    )

    expected = np.random.default_rng(20260723).integers(0, 2**32, size=4)
    actual = captured["rstate"].integers(0, 2**32, size=4)
    np.testing.assert_array_equal(actual, expected)
    assert result["seed"] == 20260723


def test_proper_gain_model_grid_matches_analytic_ridge_solution() -> None:
    model_c, init_c = _model_and_init()
    model_d, _ = _model_and_init()
    model_c.noise_std = model_c.noise_std.astype(np.float32)
    model_c.noise_std[0] = 0.0
    model_c.valid[0] = False

    def percentile(value: float) -> dict[str, float]:
        return {
            "median": value,
            "lower": value,
            "upper": value,
            "err_minus": 0.0,
            "err_plus": 0.0,
        }

    summary = {
        "burst": "test",
        "components_C": 1,
        "components_D": 1,
        "shared_zeta": False,
        "gain_model": "proper_gaussian",
        "gain_s2": 2.0,
        "alpha": percentile(4.0),
        "percentiles": {
            "tau_1ghz": percentile(init_c.tau_1ghz),
            "beta": percentile(init_c.beta),
            "t0_C": percentile(init_c.t0),
            "zeta_C": percentile(init_c.zeta),
            "delta_dm_C": percentile(0.0),
            "t0_D": percentile(init_c.t0),
            "zeta_D": percentile(init_c.zeta),
            "delta_dm_D": percentile(0.0),
        },
    }

    grid = build_model_grid_arrays(model_c, model_d, summary)

    valid = np.asarray(model_c.valid, dtype=bool)
    kernel = model_c(init_c, "M3", freq_subset=valid)
    data = np.asarray(model_c.data)[valid]
    variance = np.asarray(model_c.noise_std)[valid] ** 2
    gain = np.sum(kernel * data, axis=1) / (
        np.sum(kernel**2, axis=1) + variance / summary["gain_s2"]
    )
    expected = gain[:, None] * kernel
    np.testing.assert_allclose(grid["modelC"][valid], expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(grid["noiseC"], model_c.noise_std)
    assert grid["noiseC"].dtype == model_c.noise_std.dtype
    assert grid["noiseC"][0] == 0.0
    assert grid["gain_s2_C"] == summary["gain_s2"]

    ordinary_summary = {**summary, "gain_model": "ordinary_least_squares"}
    ordinary_grid = build_model_grid_arrays(model_c, model_d, ordinary_summary)
    np.testing.assert_array_equal(ordinary_grid["noiseC"], model_c.noise_std)
    assert ordinary_grid["noiseC"].dtype == model_c.noise_std.dtype
    assert ordinary_grid["noiseC"][0] == 0.0


def _write_packet(
    root: Path,
    result: dict,
    identity: dict,
    model_c: FRBModel,
    model_d: FRBModel,
) -> dict[str, Path]:
    root.mkdir()
    paths = {
        "summary": root / "summary.json",
        "samples": root / "samples.npz",
        "model": root / "model.npz",
        "diagnostics": root / "diagnostics.json",
        "panel": root / "panel.svg",
    }
    percentiles = result["percentiles"]
    summary = {
        "burst": "test",
        "components_C": 1,
        "components_D": 1,
        "shared_zeta": False,
        "percentiles": percentiles,
        "beta": percentiles["beta"],
        "beta_bounds": list(result["beta_bounds"]),
        "alpha": percentiles["alpha"],
        "alpha_bounds": list(result["alpha_bounds"]),
        "tau_1ghz": percentiles["tau_1ghz"],
        "log_evidence": result["log_evidence"],
        "log_evidence_err": result["log_evidence_err"],
        "ncall": result["ncall"],
        "seed": result["seed"],
        "source_revision": "test-revision",
        "controlled_contract_sha256": "0" * 64,
        "resolved_fit_identity_sha256": identity_sha256(identity),
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez(
        paths["samples"],
        samples=result["samples"],
        weights=result["weights"],
        log_weight=result["log_weight"],
        log_evidence_history=result["log_evidence_history"],
        log_evidence_error_history=result["log_evidence_error_history"],
        ncall_history=result["ncall_history"],
        param_names=np.array(result["param_names"], dtype=object),
        beta_bounds=np.array(result["beta_bounds"]),
        alpha_bounds=np.array(result["alpha_bounds"]),
    )
    model_grid = build_model_grid_arrays(model_c, model_d, summary)
    np.savez_compressed(paths["model"], **model_grid)
    diagnostics = build_diagnostics(
        summary,
        model_grid,
        samples=result["samples"],
        weights=result["weights"],
        param_names=result["param_names"],
    )
    write_diagnostics(paths["diagnostics"], diagnostics)
    render_fit_panel(model_grid, paths["panel"])
    return paths


def test_real_seeded_fit_repeats_complete_packet(tmp_path: Path) -> None:
    outputs = []
    identities = []
    models = []
    for _ in range(2):
        model_c, init_c = _model_and_init()
        model_d, init_d = _model_and_init()
        models.append((model_c, model_d))
        with pytest.warns(UserWarning):
            outputs.append(
                fit_joint_scattering(
                    model_C=model_c,
                    init_C=init_c,
                    model_D=model_d,
                    init_D=init_d,
                    nlive=12,
                    dlogz=10.0,
                    nproc=1,
                    seed=20260722,
                    resolved_identity_callback=identities.append,
                    verbose=False,
                    maxcall=40,
                )
            )

    np.testing.assert_array_equal(outputs[0]["samples"], outputs[1]["samples"])
    np.testing.assert_array_equal(outputs[0]["weights"], outputs[1]["weights"])
    for name in (
        "percentiles",
        "log_evidence",
        "log_evidence_err",
        "ncall",
        "seed",
    ):
        assert outputs[0][name] == outputs[1][name]
    assert identities[0] == identities[1]

    packets = [
        _write_packet(tmp_path / f"run-{index}", output, identity, *model_pair)
        for index, (output, identity, model_pair) in enumerate(
            zip(outputs, identities, models, strict=True)
        )
    ]
    for name in ("summary", "diagnostics", "panel"):
        assert sha256(packets[0][name]) == sha256(packets[1][name])
    for name in ("samples", "model"):
        assert canonical_npz_sha256(packets[0][name]) == canonical_npz_sha256(packets[1][name])
    diagnostics = json.loads(packets[0]["diagnostics"].read_text(encoding="utf-8"))
    assert diagnostics["component_counts"] == {"C": 1, "D": 1}
    assert set(diagnostics["residual_morphology"]) == {"C", "D"}
    assert diagnostics["prior_rail"]["method"] == "posterior_mass"
    assert diagnostics["fit_value_trust"] == "pending"


def test_real_multiprocess_seeded_fit_repeats() -> None:
    outputs = []
    for _ in range(2):
        model_c, init_c = _model_and_init()
        model_d, init_d = _model_and_init()
        with pytest.warns(UserWarning):
            outputs.append(
                fit_joint_scattering(
                    model_C=model_c,
                    init_C=init_c,
                    model_D=model_d,
                    init_D=init_d,
                    nlive=12,
                    dlogz=10.0,
                    nproc=2,
                    seed=20260724,
                    verbose=False,
                    maxcall=40,
                )
            )

    np.testing.assert_array_equal(outputs[0]["samples"], outputs[1]["samples"])
    np.testing.assert_array_equal(outputs[0]["weights"], outputs[1]["weights"])
    assert outputs[0]["percentiles"] == outputs[1]["percentiles"]
    assert outputs[0]["log_evidence"] == outputs[1]["log_evidence"]
