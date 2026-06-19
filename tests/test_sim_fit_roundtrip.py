"""Integration: `simulation/` simulator -> `scattering` burstfit fitter.

Proves the two suites are wired end-to-end. The deterministic adapter test always
runs; the sim->fit smoke test is gated on emcee.

NOTE on scope: this asserts the *wiring* works (sim output flows through the
fitter and yields a finite, positive scattering time), NOT quantitative tau
recovery. The simulator emits a fully scintillated burst whose band-integrated
profile is a spiky multipath IRF, while the M2 fitter assumes a smooth-spectrum
exponential pulse-broadening function. Quantitative tau recovery needs ensemble
averaging over screen realisations to suppress scintillation — out of scope here.
"""
import os
import sys

import numpy as np
import astropy.units as u
import pytest

_SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)


def _cfg():
    from engine import SimCfg
    from screen import ScreenCfg

    # Host-dominated single screen (MW L tiny -> negligible scattering); equal N
    # is required by the cross-screen Doppler term in _precompute_doppler_terms.
    return SimCfg(
        peak_flux=5 * u.Jy,
        nu0=800 * u.MHz,
        bw=25.0 * u.MHz,
        nchan=256,
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=128, L=0.2 * u.AU, rng_seed=1234),
        host=ScreenCfg(N=128, L=20.0 * u.AU, rng_seed=5678),
        intrinsic_pulse="delta",
    )


def test_sim_grids_to_fitter_adapter():
    """Deterministic: simulator (I[t,f], s, Hz) -> fitter (data[f,t], ms, GHz, MHz)."""
    from sim_fit_bridge import sim_grids_to_fitter

    I_t_nu = np.arange(6 * 4, dtype=float).reshape(6, 4)  # [time=6, freq=4]
    time_s = np.linspace(0, 5e-3, 6)  # 0..5 ms
    freq_hz = np.linspace(8.0e8, 8.25e8, 4)  # 800..825 MHz

    data, time_ms, freq_ghz, df_MHz = sim_grids_to_fitter(I_t_nu, time_s, freq_hz)

    assert data.shape == (4, 6)  # transposed to [freq, time]
    np.testing.assert_allclose(data, I_t_nu.T)
    np.testing.assert_allclose(time_ms[-1], 5.0)
    np.testing.assert_allclose(freq_ghz[0], 0.8)
    np.testing.assert_allclose(df_MHz, (freq_hz[1] - freq_hz[0]) / 1e6)


def test_sim_to_fit_smoke():
    """sim -> fit runs end-to-end and returns a finite, positive tau_1ghz."""
    pytest.importorskip("emcee")
    from sim_fit_bridge import simulate_scattered_burst, fit_tau

    rng = np.random.default_rng(0)
    _, (data, t_ms, f_ghz, df), tau_true_ms = simulate_scattered_burst(
        _cfg(), duration=2.0 * u.ms, rng=rng
    )
    assert data.shape[0] == 256 and data.shape[1] > 10
    assert np.all(np.isfinite(data))
    assert tau_true_ms > 0

    tau_fit_ms, sampler = fit_tau(data, t_ms, f_ghz, df, n_steps=300, n_walkers_mult=4)
    assert np.isfinite(tau_fit_ms) and tau_fit_ms > 0
    assert sampler.get_chain().shape[0] == 300


if __name__ == "__main__":
    test_sim_grids_to_fitter_adapter()
    test_sim_to_fit_smoke()
    print("ok")
