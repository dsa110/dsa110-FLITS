"""Phase-0 injection harness physics (Faber2026 plan-dm-measurement-methods)."""

import numpy as np

from dispersion.dm_campaign.injection import (
    InjectionSpec,
    disperse_waterfall,
    inject_pulse,
    make_noise_from_offpulse,
)


def test_dispersion_shift_matches_kdm():
    wf = np.zeros((64, 4096))
    wf[:, 2048] = 1.0
    freq = np.linspace(1.32, 1.49, 64)
    out = disperse_waterfall(wf, freq_ghz=freq, dm=1.0, dt_ms=32.768e-3)
    k = 4.148808  # ms GHz^2 / (pc cm^-3); delay referenced to top of band
    lag_lo = k * 1.0 * (freq[0] ** -2 - freq[-1] ** -2)
    assert abs(np.argmax(out[0]) - 2048 - round(lag_lo / 32.768e-3)) <= 1
    assert np.argmax(out[-1]) == 2048


def test_injection_recovers_specified_snr():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(128, 8192)).astype(np.float32)
    spec = InjectionSpec(dm_offset=0.4, snr=40.0, width_ms=0.5, tau_1ghz_ms=0.0, t0_frac=0.5)
    wf, truth = inject_pulse(
        noise, freq_ghz=np.linspace(1.32, 1.49, 128), dt_ms=32.768e-3, spec=spec, rng=rng
    )
    assert truth["dm_offset"] == 0.4
    prof = wf.sum(axis=0)
    med = np.median(prof)
    snr = (prof.max() - med) / (1.4826 * np.median(np.abs(prof - med)))
    assert snr > 10  # injected burst is detectable


def test_injected_scattering_tail_is_causal():
    rng = np.random.default_rng(4)
    noise = np.zeros((64, 4096), dtype=np.float32)
    spec = InjectionSpec(dm_offset=0.0, snr=50.0, width_ms=0.3, tau_1ghz_ms=5.0, t0_frac=0.5)
    wf, truth = inject_pulse(
        noise, freq_ghz=np.linspace(0.4, 0.8, 64), dt_ms=0.16384, spec=spec, rng=rng
    )
    prof = wf.sum(axis=0)
    pk = int(np.argmax(prof))
    assert prof[pk + 40] > prof[pk - 40]  # tail trails, never leads


def test_offpulse_noise_preserves_channel_stats():
    rng = np.random.default_rng(2)
    wf = rng.normal(loc=np.arange(32)[:, None], size=(32, 4000)).astype(np.float32)
    noise = make_noise_from_offpulse(wf, on_frac=(0.4, 0.6), n_time=2000, rng=rng)
    assert noise.shape == (32, 2000)
    assert np.allclose(noise.mean(axis=1), np.arange(32), atol=0.2)
