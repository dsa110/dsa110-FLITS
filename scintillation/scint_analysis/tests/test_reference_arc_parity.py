import numpy as np
import pytest

from scint_analysis import analysis
from scint_analysis.core import DynamicSpectrum
from scint_analysis.revalidation import _mean_normalized_acf


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
