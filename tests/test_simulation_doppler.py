"""Physics regression: screen-velocity (Doppler) time-evolution of the delay.

Pradeep+2025 (arXiv:2505.04576) Eq. A.2: the scattering angle drifts as
dtheta_n/dt = V_n / [D_n (1+z_n)] with V_n a physical transverse velocity.
The simulator has two code paths for the time-evolved delay:
  - 'fast': a precomputed 2nd-order Taylor expansion (engine._precompute_doppler_terms)
  - 'slow': an exact per-step angular recompute (engine._delays, speed='slow')
Because the delay is exactly quadratic in t, the Taylor form is exact, so the two
paths must agree to floating point. A regression here previously diverged by ~1e40
(the fast path used the linear velocity V instead of the angular drift V/[D(1+z)]).
"""
import os
import sys

import numpy as np
import astropy.units as u

_SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)


def _sim(v_mw=(0.0, 0.0), v_host=(0.0, 0.0)):
    from engine import SimCfg, FRBScintillator
    from screen import ScreenCfg

    cfg = SimCfg(
        peak_flux=5 * u.Jy, nu0=800 * u.MHz, bw=25.0 * u.MHz, nchan=64,
        z_host=0.192, D_mw=2.3 * u.kpc, D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=16, L=3.5 * u.AU, rng_seed=1234, v_perp=v_mw),
        host=ScreenCfg(N=16, L=20.0 * u.AU, rng_seed=5678, v_perp=v_host),
        intrinsic_pulse="delta",
    )
    return FRBScintillator(cfg)


def test_doppler_fast_slow_consistency():
    """Taylor (fast) and exact-recompute (slow) delays agree for nonzero velocity."""
    sim = _sim(v_mw=(50.0, 0.0), v_host=(0.0, 30.0))  # km/s
    static = sim._delays(0.0)
    for dt in (1e-6, 1e-5, 1e-4, 1e-3):
        fast = sim._delays(dt, speed="fast")          # combined (N, N)
        slow = sim._delays(dt, speed="slow")
        drift = np.max(np.abs(slow - static))
        resid = np.max(np.abs(fast - slow))
        # absolute residual at the f64 cancellation floor; sanity-bound it relative
        # to the drift magnitude so a real divergence (the old 1e40 bug) trips.
        assert resid <= 1e-3 * max(drift, 1e-30) + 1e-18, (dt, resid, drift)


def test_doppler_zero_velocity_reduces_to_static():
    """v_perp=0 -> Doppler coefficients vanish -> time-evolved delay == static."""
    sim = _sim()  # zero velocity
    static = sim._delays(0.0)
    assert np.allclose(sim._tau_linear_coeff, 0.0)
    assert sim._tau_quad_coeff == 0.0
    np.testing.assert_allclose(sim._delays(1.0, speed="fast"), static, atol=0, rtol=0)


def test_delays_return_shape_uniform():
    """All _delays branches return the combined (N_mw, N_host) array (no 3-tuple)."""
    sim = _sim(v_mw=(50.0, 0.0), v_host=(0.0, 30.0))
    shape = (sim.cfg.mw.N, sim.cfg.host.N)
    assert np.shape(sim._delays(0.0)) == shape
    assert np.shape(sim._delays(1e-3, speed="fast")) == shape
    assert np.shape(sim._delays(1e-3, speed="slow")) == shape


def test_2d_engine_runs_for_non_triple_n():
    """The 2D engine previously crashed unpacking the fast-path array for N!=3 / dt!=0."""
    sim = _sim(v_mw=(50.0, 0.0), v_host=(0.0, 30.0))  # N=16
    I_t_nu, t, f = sim.synthesise_dynamic_spectrum_2d(
        duration=0.3 * u.ms, time_res=0.05 * u.ms
    )
    assert I_t_nu.shape == (t.size, sim.n_chan)
    assert t.size > 1  # exercises at least one dt != 0 step
    assert np.all(np.isfinite(I_t_nu)) and np.all(I_t_nu >= 0)


if __name__ == "__main__":
    test_doppler_fast_slow_consistency()
    test_doppler_zero_velocity_reduces_to_static()
    test_delays_return_shape_uniform()
    test_2d_engine_runs_for_non_triple_n()
    print("ok")
