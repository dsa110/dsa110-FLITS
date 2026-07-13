import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scint_analysis import analysis
from scint_analysis.core import DynamicSpectrum
from scint_analysis.revalidation import _mean_normalized_acf

FIXTURES = Path(__file__).parent / "fixtures"
ZACH_ACF_FIXTURE = FIXTURES / "zach_acf_codetections_fftsize64_downfreq1.npz"
ZACH_ACF_SHA256 = "6965c50d40d1ba67671f4537fab983a8f4af30f609feaf253b38a22aef29c667"
ZACH_FCENTS_MHZ = np.array(
    [
        766.8635050688563,
        700.1998902065691,
        633.5362753442816,
        566.8726604819944,
        500.209045619707,
        433.5454307574197,
    ]
)
ZACH_FIRST_ACF_VALUES = np.array(
    [0.08595040, 0.06366222, 0.07017246, 0.13167876, 0.09638528, 0.06162758]
)
ZACH_FIRST_POSITIVE_LAG_MHZ = 0.012207217517470781
ZACH_RECOMPUTED_GAMMA_HWHM_MHZ = np.array(
    [0.21456356, 0.08414576, 0.05479846, 0.09827691, 0.02422339, 0.02155347]
)
FREYA_RECOMPUTED_GAMMA_KHZ = 35.19
FREYA_TOL_KHZ = 15.0


def _clean_spectrum(seed=7, n=512, scale_chan=10):
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n + scale_chan)
    corr = np.convolve(white, np.ones(scale_chan) / scale_chan, mode="valid")[:n]
    return np.ma.MaskedArray(100.0 + 20.0 * corr, mask=np.zeros(n, dtype=bool))


def test_first_fit_lag_two_matches_reference_pairloop():
    """CHIME first_fit_lag=2 matches the reference ACF pair-loop."""
    spec = _clean_spectrum()
    cw = 0.006103515625
    acf = analysis.calculate_acf(spec, cw, max_lag_bins=64, first_fit_lag=2)
    pos = acf.lags > 0
    first_pos_lag_bins = round(float(acf.lags[pos].min()) / cw)
    assert first_pos_lag_bins == 2
    assert not np.any(np.isclose(acf.lags, 0.0))
    keep = np.ones(spec.size)
    _, ref_acf, _ = _mean_normalized_acf(
        spec.data, keep, cw, max_lag_mhz=64 * cw, first_lag=2
    )
    ref_pos = ref_acf[len(ref_acf) // 2 :]
    np.testing.assert_allclose(
        acf.acf[pos], ref_pos[: pos.sum()], rtol=1e-6, atol=1e-9
    )


def test_2d_fitter_excludes_zero_lag():
    from scint_analysis import fitting_2d

    lags = np.linspace(-5, 5, 101)
    acf = 0.4 / (1 + (lags / 0.3) ** 2)
    acf_results = {
        "subband_lags_mhz": [lags, lags],
        "subband_acfs": [acf, acf],
        "subband_acfs_err": [
            np.full_like(acf, 0.02),
            np.full_like(acf, 0.02),
        ],
        "subband_center_freqs_mhz": [600.0, 700.0],
    }
    model = fitting_2d.Scintillation2DModel(acf_results, fit_range_mhz=2.0)
    for lags_i, mask in zip(model.lags_list, model.masks, strict=True):
        assert not np.any(mask & np.isclose(lags_i, 0.0))


def test_result_reports_hwhm_and_fwhm_labels():
    from scint_analysis.analysis import _bandwidth_fields

    out = _bandwidth_fields(gamma_hwhm_mhz=0.25)
    assert out["reported_dnu_definition"] == "HWHM"
    assert out["gamma_hwhm_mhz"] == 0.25
    assert np.isclose(out["fwhm_mhz"], 0.5)


def test_canfar_reference_snr_normalization():
    from scint_analysis import freya_scintillation as fs

    rng = np.random.default_rng(3)
    nchan, nt = 64, 300
    gain = np.linspace(1.0, 3.0, nchan)[:, None]
    power = gain * rng.normal(0.0, 1.0, (nchan, nt)) + 10.0 * gain
    ds = DynamicSpectrum(
        np.ma.MaskedArray(power, mask=np.zeros_like(power, bool)),
        np.linspace(600, 800, nchan),
        np.arange(nt) * 8e-5,
    )
    out = fs.normalize_snr_per_channel(ds, off_pulse_lims=(0, 200))
    off = out.power[:, 0:200]
    assert np.allclose(np.ma.mean(off, axis=1).filled(0), 0.0, atol=0.15)
    assert np.allclose(np.ma.std(off, axis=1).filled(1), 1.0, atol=0.15)


def test_canfar_reference_mode_masks_lte_band():
    freqs = np.linspace(600, 800, 400)
    power = np.ma.MaskedArray(
        np.ones((400, 100)), mask=np.zeros((400, 100), bool)
    )
    ds = DynamicSpectrum(power, freqs, np.arange(100) * 8e-5)
    cfg = {
        "analysis": {
            "preprocessing": {
                "mode": "canfar_reference",
                "lte_exclude_mhz": [730.0, 760.0],
            },
            "rfi_masking": {
                "manual_burst_window": [40, 60],
                "manual_noise_window": [0, 30],
            },
        }
    }
    out = ds.mask_rfi(cfg)
    lte = (freqs >= 730.0) & (freqs <= 760.0)
    assert out.power.mask[lte].all()


def test_pipeline_canfar_reference_mode_uses_snr_normalization():
    from scint_analysis.pipeline import ScintillationAnalysis

    rng = np.random.default_rng(31)
    gain = np.linspace(1.0, 2.0, 16)[:, None]
    power = gain * rng.normal(size=(16, 120)) + 8.0 * gain
    ds = DynamicSpectrum(power, np.linspace(600, 700, 16), np.arange(120))
    pipeline = ScintillationAnalysis(
        {"analysis": {"preprocessing": {"mode": "canfar_reference"}}}
    )
    pipeline.masked_spectrum = ds

    pipeline._apply_bandpass_normalization((0, 80))

    off = pipeline.masked_spectrum.power[:, :80]
    assert np.allclose(np.ma.mean(off, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.ma.std(off, axis=1), 1.0, atol=1e-12)


def test_canfar_reference_writes_inspectable_cleaning_intermediate(
    tmp_path, monkeypatch
):
    from scint_analysis.pipeline import ScintillationAnalysis

    rng = np.random.default_rng(37)
    freqs = np.linspace(700, 780, 80)
    ds = DynamicSpectrum(rng.normal(size=(80, 100)), freqs, np.arange(100))
    monkeypatch.setattr(
        DynamicSpectrum, "from_numpy_file", staticmethod(lambda _path: ds)
    )
    cfg = {
        "burst_id": "parity-test",
        "input_data_path": "unused.npy",
        "pipeline_options": {
            "cache_directory": str(tmp_path),
            "save_intermediate_steps": True,
            "force_recalc": True,
        },
        "analysis": {
            "preprocessing": {"mode": "canfar_reference"},
            "rfi_masking": {
                "manual_burst_window": [40, 60],
                "manual_noise_window": [0, 30],
                "enable_time_domain_flagging": False,
            },
        },
    }

    ScintillationAnalysis(cfg).prepare_data()

    artifact = tmp_path / "parity-test_canfar_clean.npz"
    assert artifact.exists()
    with np.load(artifact) as saved:
        assert set(saved.files) >= {"cleaned_spectrum", "channel_mask", "time_mask"}
        assert saved["cleaned_spectrum"].shape == ds.power.shape
        lte = (freqs >= 730.0) & (freqs <= 760.0)
        assert saved["channel_mask"][lte].all()


class _SeededNoiseDescriptor:
    def __init__(self, nchan, seed=41):
        self.kind = "flux_gauss"
        self.nt = 1
        self.nchan = nchan
        self.mu = 0.0
        self.sigma = 0.2
        self.gamma_k = 0.0
        self.gamma_theta = 0.0
        self.phi_t = 0.0
        self.phi_f = 0.0
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.normal(0.0, self.sigma, (self.nt, self.nchan))


@pytest.mark.parametrize("first_fit_lag", [2, 1])
def test_noise_template_subband_fit_respects_first_fit_lag(first_fit_lag):
    """Enabled noise templates match centerless CHIME and DSA ACF grids."""
    rng = np.random.default_rng(29)
    nchan, nt = 256, 24
    channel_width_mhz = 0.02
    freqs = 1200.0 + np.arange(nchan) * channel_width_mhz
    times = np.arange(nt) * 0.001
    white = rng.normal(size=nchan + 12)
    correlated = np.convolve(white, np.ones(12) / 12, mode="valid")[:nchan]
    spectrum = 10.0 * (1.0 + 0.25 * correlated / np.std(correlated))
    profile = np.zeros(nt)
    profile[8:16] = 1.0
    power = rng.normal(0.0, 0.2, (nchan, nt)) + spectrum[:, None] * profile
    dynamic_spectrum = DynamicSpectrum(power, freqs, times)
    config = {
        "analysis": {
            "acf": {
                "num_subbands": 2,
                "max_lag_mhz": 1.28,
                "first_fit_lag": first_fit_lag,
            },
            "noise": {"disable_template": False, "template_n_draws": 3},
            "self_noise": {"disable": True},
            "fitting": {
                "fit_lagrange_mhz": 0.8,
                "reference_frequency_mhz": 1202.5,
                "force_model": "fit_tpl_lor",
            },
        }
    }

    analysis.clear_noise_acf_cache()
    acf_results = analysis.calculate_acfs_for_subbands(
        dynamic_spectrum,
        config,
        burst_lims=(8, 16),
        noise_desc=_SeededNoiseDescriptor(nchan),
    )
    _final, fits, _power_law = analysis.analyze_scintillation_from_acfs(
        acf_results, config
    )

    assert len(fits) == 2
    for template, fit in zip(acf_results["noise_template"], fits, strict=True):
        assert np.max(template) == pytest.approx(1.0)
        result = fit["fit_tpl_lor"]
        assert result is not None and result.success
        assert np.all(np.isfinite([param.value for param in result.params.values()]))


def test_zach_executed_notebook_acf_fixture_is_unchanged():
    """The rescued executed-notebook ACF artifact is byte-for-byte unchanged."""
    assert hashlib.sha256(ZACH_ACF_FIXTURE.read_bytes()).hexdigest() == ZACH_ACF_SHA256
    with np.load(ZACH_ACF_FIXTURE, allow_pickle=False) as saved:
        assert saved["sub_acfs"].shape == (6, 3274)
        assert saved["sub_lags"].shape == (6, 3274)
        assert saved["sub_fcents"].shape == (6,)
        np.testing.assert_allclose(saved["sub_fcents"], ZACH_FCENTS_MHZ, rtol=0, atol=1e-12)

        mid = saved["sub_lags"].shape[1] // 2
        first_positive_lags = saved["sub_lags"][:, mid]
        np.testing.assert_allclose(
            first_positive_lags, ZACH_FIRST_POSITIVE_LAG_MHZ, rtol=0, atol=2e-13
        )
        np.testing.assert_allclose(
            saved["sub_acfs"][:, mid], ZACH_FIRST_ACF_VALUES, rtol=1e-7, atol=1e-9
        )
        np.testing.assert_allclose(
            saved["sub_lags"][:, :mid],
            -saved["sub_lags"][:, mid:][:, ::-1],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            saved["sub_acfs"][:, :mid],
            saved["sub_acfs"][:, mid:][:, ::-1],
            rtol=0,
            atol=0,
        )


def _fit_zach_subband_gammas(sub_acfs, sub_lags):
    """Recompute γ with the rescued single-Lorentzian, default ±1 MHz fit."""
    from lmfit import Model

    from scint_analysis.revalidation import _lorentz_w_c

    gammas = []
    for acf, lags in zip(sub_acfs, sub_lags, strict=True):
        fit_mask = (lags > 0) & (lags <= 1.0)
        x = np.asarray(lags[fit_mask], dtype=float)
        y = np.asarray(acf[fit_mask], dtype=float)
        below_half = np.flatnonzero(y < y[0] / 2.0)
        gamma_init = float(x[below_half[0]]) if below_half.size else float(x[1])

        model = Model(_lorentz_w_c)
        model.set_param_hint("gamma", min=1e-5, max=100.0)
        model.set_param_hint("m", min=-100.0, max=100.0)
        model.set_param_hint("c", min=-100.0, max=100.0)
        result = model.fit(
            y,
            x=x,
            gamma=gamma_init,
            m=float(np.sqrt(max(y[0], 1e-3))),
            c=0.0,
        )
        assert result.success
        gammas.append(abs(float(result.params["gamma"].value)))
    return np.asarray(gammas)


def test_zach_subband_gamma_recompute_from_notebook_acfs():
    """Recomputed HWHM γ values retain the Zach notebook's positive γ(ν) trend."""
    with np.load(ZACH_ACF_FIXTURE, allow_pickle=False) as saved:
        gammas = _fit_zach_subband_gammas(saved["sub_acfs"], saved["sub_lags"])
        fcents = saved["sub_fcents"]

    assert gammas.shape == (6,)
    assert np.all(gammas > 0)
    np.testing.assert_allclose(gammas, ZACH_RECOMPUTED_GAMMA_HWHM_MHZ, rtol=0.20)
    alpha = np.polyfit(np.log10(fcents), np.log10(gammas), 1)[0]
    assert alpha > 0.0


def test_gamma_scaling_recovers_seeded_injection_with_three_named_estimators():
    """Analytic power-law injection anchors both 1D estimators; 2D stays distinct."""
    rng = np.random.default_rng(20260713)
    freqs = np.array([450.0, 550.0, 650.0, 750.0, 850.0, 950.0])
    alpha_true = 4.0
    gammas = 0.05 * (freqs / 600.0) ** alpha_true
    gammas *= 10 ** rng.normal(0.0, 0.05, freqs.size)

    out = analysis.estimate_gamma_scaling(
        freqs,
        gammas,
        gamma_errs=0.05 * gammas,
        ref_freq=600.0,
        joint_2d={
            "alpha": 4.05,
            "alpha_err": 0.12,
            "gamma_0": 0.051,
            "gamma_0_err": 0.004,
            "nu_ref": 600.0,
            "success": True,
        },
    )

    assert set(out) == {"odr_logspace", "loglog_unweighted", "joint_2d"}
    assert abs(out["odr_logspace"]["alpha"] - alpha_true) < 0.3
    assert abs(out["loglog_unweighted"]["alpha"] - alpha_true) < 0.3
    assert out["odr_logspace"]["method"] == "log-space ODR (analysis-Copy1.py:844)"
    assert out["loglog_unweighted"]["method"].startswith("unweighted log-log")
    assert out["joint_2d"]["method"] == "joint 2D ACF fit (fitting_2d.py)"
    assert all(np.isfinite(method["bw_at_ref_mhz"]) for method in out.values())


def test_gamma_scaling_ill_conditioned_two_subbands_never_returns_inf():
    """Nearly coincident frequencies may fail explicitly, but never overflow to inf."""
    out = analysis.estimate_gamma_scaling(
        np.array([600.0, 600.0 + 1e-9]),
        np.array([1e-250, 1e250]),
        gamma_errs=np.array([1e-251, 1e249]),
        ref_freq=600.0,
    )

    for result in out.values():
        numeric = [value for value in result.values() if isinstance(value, (int, float))]
        assert not any(np.isinf(value) for value in numeric)
        assert result["status"] != "ok" or np.isfinite(result["bw_at_ref_mhz"])


def test_pipeline_attaches_joint_2d_as_third_named_estimator(monkeypatch):
    from scint_analysis import fitting_2d
    from scint_analysis.pipeline import ScintillationAnalysis

    pipeline = ScintillationAnalysis(
        {"analysis": {"fitting": {"reference_frequency_mhz": 600.0}}}
    )
    pipeline.acf_results = {"subband_acfs": [np.array([0.1])]}
    pipeline.final_results = {
        "components": {
            "scint_scale": {
                "gamma_scaling": analysis.estimate_gamma_scaling(
                    [500.0, 700.0], [0.02, 0.08], ref_freq=600.0
                )
            }
        }
    }
    result = SimpleNamespace(
        gamma_0=0.05,
        gamma_0_err=0.005,
        alpha=4.1,
        alpha_err=0.2,
        m_0=0.9,
        m_0_err=0.1,
        nu_ref=600.0,
        redchi=1.1,
        success=True,
    )
    monkeypatch.setattr(fitting_2d, "fit_2d_scintillation", lambda *args, **kwargs: result)

    pipeline._run_2d_scintillation_fit({})

    scaling = pipeline.final_results["components"]["scint_scale"]["gamma_scaling"]
    assert set(scaling) == {"odr_logspace", "loglog_unweighted", "joint_2d"}
    assert scaling["joint_2d"]["status"] == "ok"
    assert scaling["joint_2d"]["alpha"] == pytest.approx(4.1)


def test_intra_pulse_runs_with_single_lorentzian(monkeypatch):
    """The registry's fit_lor/l_gamma/l_m contract drives each time slice."""
    rng = np.random.default_rng(43)
    nchan, nt = 64, 16
    ds = DynamicSpectrum(
        10.0 + rng.normal(size=(nchan, nt)),
        np.linspace(600.0, 620.0, nchan),
        np.arange(nt) * 0.001,
    )
    cfg = {
        "analysis": {
            "acf": {"intra_pulse_time_bins": 4, "max_lag_mhz": 2.0},
            "fitting": {"fit_lagrange_mhz": 1.0},
            "self_noise": {"disable": True},
        }
    }

    def fake_fit(acf_obj, **_kwargs):
        fit_mask = np.abs(acf_obj.lags) <= 1.0
        return {
            "fit_lor": SimpleNamespace(
                success=True,
                params={
                    "l_gamma": SimpleNamespace(value=0.25, stderr=0.02),
                    "l_m": SimpleNamespace(value=0.7, stderr=0.05),
                },
                best_fit=np.zeros(np.count_nonzero(fit_mask)),
            )
        }

    monkeypatch.setattr(analysis, "_fit_acf_models", fake_fit)
    result = analysis.analyze_intra_pulse_scintillation(ds, (0, nt), cfg, None)

    assert isinstance(result, list) and len(result) == 4
    assert all(item["bw"] == pytest.approx(0.25) for item in result)
    assert all(item["mod"] == pytest.approx(0.7) for item in result)
    assert all(item["gamma_hwhm_mhz"] == pytest.approx(0.25) for item in result)


def test_direct_modulation_over_time_matches_analytic_std_over_mean():
    """RECIPE section 4: direct m is std/mean after frequency averaging."""
    profile = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
    power = np.ma.MaskedArray(
        np.repeat(profile[None, :], 32, axis=0),
        mask=np.zeros((32, profile.size), dtype=bool),
    )

    result = analysis.modulation_index_over_time(power, (0, profile.size))

    expected = [
        np.std(profile[start : min(start + 3, profile.size)])
        / np.mean(profile[start : min(start + 3, profile.size)])
        for start in range(profile.size - 1)
    ]
    assert result["method"] == (
        "direct std/mean (scinttools_v3.analyze_modulation_over_time)"
    )
    assert result["definition"].startswith(
        "direct m = std/mean of frequency-averaged intensity"
    )
    assert result["chunk_bins"] == 3
    assert result["overlap_bins"] == 2
    np.testing.assert_allclose(result["m"], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["time_idx"], [1.0, 2.0, 3.0, 3.5])


def test_modulation_results_keep_frequency_and_time_definitions_separate():
    assert analysis.lorentzian_component(0.0, gamma=0.25, m=0.4) == pytest.approx(
        0.4**2
    )
    final = {
        "components": {
            "scint_scale": {
                "subband_measurements": [
                    {"freq_mhz": 500.0, "mod": 0.4, "mod_err": 0.05},
                    {"freq_mhz": 700.0, "mod": 0.8, "mod_err": 0.06},
                ]
            }
        }
    }

    analysis.attach_modulation_index_frequency(final)

    reported = final["modulation_index_frequency"]["acf_amplitude"]
    assert reported["definition"].startswith(
        "ACF-amplitude m = sqrt(fitted Lorentzian amplitude)"
    )
    assert reported["components"]["scint_scale"][0]["m"] == pytest.approx(0.4)
    assert "direct_std_mean" not in final["modulation_index_frequency"]


def test_pipeline_reports_all_modulation_branches_side_by_side(monkeypatch):
    from galaxies.foreground import scintillation_bridge
    from scint_analysis.pipeline import ScintillationAnalysis

    ds = DynamicSpectrum(
        np.repeat(np.array([[1.0, 3.0, 1.0, 3.0]]), 16, axis=0),
        np.linspace(600.0, 620.0, 16),
        np.arange(4) * 0.001,
    )
    cfg = {
        "burst_id": "modulation-test",
        "analysis": {
            "rfi_masking": {
                "manual_burst_window": [0, 4],
                "manual_noise_window": [0, 0],
            },
            "acf": {
                "enable_intra_pulse_analysis": True,
                "time_chunk_size_bins": 3,
                "time_overlap_bins": 2,
            },
            "noise": {"disable": True},
            "fit_2d": {"enable": False},
        },
    }
    pipeline = ScintillationAnalysis(cfg)

    def prepare():
        pipeline.masked_spectrum = ds
        pipeline.data_prepared = True

    monkeypatch.setattr(pipeline, "prepare_data", prepare)
    monkeypatch.setattr(
        analysis,
        "calculate_acfs_for_subbands",
        lambda *_args, **_kwargs: {"subband_acfs": [np.array([0.2])]},
    )
    monkeypatch.setattr(
        analysis,
        "analyze_intra_pulse_scintillation",
        lambda *_args, **_kwargs: [
            {
                "time_s": 0.0015,
                "bw": 0.25,
                "bw_err": 0.02,
                "mod": 0.7,
                "mod_err": 0.05,
                "fit_success": True,
            }
        ],
    )
    monkeypatch.setattr(
        analysis,
        "analyze_scintillation_from_acfs",
        lambda *_args, **_kwargs: (
            {
                "components": {
                    "scint_scale": {
                        "subband_measurements": [
                            {"freq_mhz": 610.0, "mod": 0.6, "mod_err": 0.04}
                        ]
                    }
                }
            },
            [],
            {},
        ),
    )
    monkeypatch.setattr(
        scintillation_bridge,
        "attach_interpretation_with_bridge",
        lambda _results, config, **_kwargs: config,
    )

    pipeline.run()

    m_nu = pipeline.final_results["modulation_index_frequency"]["acf_amplitude"]
    m_t = pipeline.final_results["modulation_index_time"]
    assert m_nu["definition"] == analysis.ACF_AMPLITUDE_MODULATION_DEFINITION
    assert set(m_t) == {"acf_fitted", "direct_std_mean"}
    assert m_t["acf_fitted"]["definition"] == (
        analysis.INTRA_PULSE_ACF_MODULATION_DEFINITION
    )
    assert m_t["direct_std_mean"]["definition"] == (
        analysis.DIRECT_MODULATION_DEFINITION
    )


def test_intra_pulse_plot_writes_figure_manifest(tmp_path):
    from scint_analysis import plotting

    result = {
        "time_s": 0.001,
        "bw": 0.25,
        "bw_err": 0.02,
        "mod": 0.7,
        "mod_err": 0.05,
        "acf_lags": np.array([-1.0, -0.5, 0.5, 1.0]),
        "acf_data": np.array([0.1, 0.4, 0.4, 0.1]),
        "fit_success": True,
    }
    save_path = tmp_path / "intra_pulse.png"

    later = {**result, "time_s": 0.002}
    plotting.plot_intra_pulse_evolution(
        [result, later],
        np.array([1.0, 2.0]),
        np.array([0.0, 0.001]),
        save_path=save_path,
    )

    assert save_path.exists()
    manifest = (tmp_path / "figures.manifest.json").read_text()
    assert "intra_pulse.png" in manifest
    assert "ACF-fitted" in manifest


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "freya CHIME dnu_d is instrument-dominated/unconstrained: the 35.19 kHz "
        "target is the documented instrumental-artifact value "
        "(chime_artifact_guards.py:7) and the rescued notebook's own fit is "
        "unconstrained (scint_freya.ipynb cell 13: 3836 +/- 2132 kHz); point "
        "parity is not expected. Real driver TBD via factor-isolation sweep "
        "(canfar_reference on/off, first_fit_lag 1-3, f_res grid check)."
    ),
    strict=True,
)
def test_freya_chime_gamma_brackets_legacy_recomputation(tmp_path, monkeypatch):
    """The current parity path should recover the legacy Freya HWHM recomputation."""
    from scint_analysis.config import load_config
    from scint_analysis.pipeline import ScintillationAnalysis

    repo_root = Path(__file__).parents[3]
    data_candidates = (
        repo_root / "scintillation/data/freya_chime_hi.npz",
        Path.home() / "Data/Faber2026/dsa110/scintillation/data/freya_chime_hi.npz",
    )
    data_path = next((path for path in data_candidates if path.exists()), None)
    if data_path is None:
        pytest.skip("freya_chime_hi.npz is not available")

    config_path = repo_root / "scintillation/configs/bursts/freya_chime_hi.yaml"
    cfg = load_config(config_path, workspace_root=repo_root)
    cfg["input_data_path"] = str(data_path)
    cfg.setdefault("analysis", {}).setdefault("preprocessing", {})["mode"] = (
        "canfar_reference"
    )
    cfg["analysis"].setdefault("fit_2d", {})["enable"] = False
    cfg.setdefault("pipeline_options", {}).update(
        {
            "cache_directory": str(tmp_path),
            "force_recalc": True,
            "save_intermediate_steps": False,
        }
    )
    monkeypatch.setattr(ScintillationAnalysis, "_create_diagnostic_plots", lambda *args, **kwargs: None)

    pipeline = ScintillationAnalysis(cfg)
    pipeline.run()
    gamma_khz = (
        pipeline.final_results["components"]["scint_scale"]["gamma_hwhm_mhz"] * 1e3
    )
    assert abs(gamma_khz - FREYA_RECOMPUTED_GAMMA_KHZ) < FREYA_TOL_KHZ
