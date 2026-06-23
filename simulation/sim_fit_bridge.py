"""Bridge: feed `simulation/` (FRBScintillator) output into the `scattering` fitter.

The two suites were decoupled — the forward simulator and the burstfit MCMC never
shared a closed loop. This wires them: simulate a scattered burst with a known
screen scattering time, fit it, recover tau. Used by tests/test_sim_fit_roundtrip.py.

Unit conventions differ across the boundary:
  simulator  -> I[time, freq], time in s, freq in Hz
  burstfit   -> data[freq, time], time in ms, freq in GHz, df in MHz (native channel)
"""

from __future__ import annotations

import os
import sys

import astropy.units as u
import numpy as np

# simulation/ is a bare-import dir (engine does `from screen import ...`, not relative),
# so it must be on sys.path for `from engine import ...` to resolve regardless of caller cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine import FRBScintillator, SimCfg  # noqa: E402

from scattering.scat_analysis.burstfit import (  # noqa: E402
    FRBFitter,
    FRBModel,
    FRBParams,
    build_priors,
)


def sim_grids_to_fitter(I_t_nu, time_axis_s, freq_axis_hz):
    """Adapt simulator dynamic-spectrum output to the burstfit convention.

    Returns (data[freq, time], time_ms, freq_ghz, df_MHz).
    """
    data = np.ascontiguousarray(np.asarray(I_t_nu, dtype=float).T)  # [time,freq] -> [freq,time]
    time_ms = np.asarray(time_axis_s, dtype=float) * 1e3
    freq_ghz = np.asarray(freq_axis_hz, dtype=float) / 1e9
    df_MHz = float(abs(freq_ghz[1] - freq_ghz[0]) * 1e3) if freq_ghz.size > 1 else 1.0
    return data, time_ms, freq_ghz, df_MHz


def simulate_scattered_burst(cfg: SimCfg, duration: u.Quantity, rng=None):
    """Run the simulator and return (sim, (data, time_ms, freq_ghz, df_MHz), tau_true_ms).

    tau_true is the dominant screen's theoretical scattering time at nu0, in ms.
    """
    sim = FRBScintillator(cfg)
    I_t_nu, t_s, f_hz = sim.synthesise_dynamic_spectrum(duration=duration, rng=rng)
    if I_t_nu.size == 0:
        raise ValueError("simulator returned empty spectrum; increase `duration`")
    obs = sim.calculate_theoretical_observables()
    tau_true_ms = max(obs["tau_s_mw_s"], obs["tau_s_host_s"]) * 1e3
    return sim, sim_grids_to_fitter(I_t_nu, t_s, f_hz), tau_true_ms


def fit_tau(
    data,
    time_ms,
    freq_ghz,
    df_MHz,
    *,
    dm_init: float = 0.0,
    tau_init_ms: float = 1.0,
    n_steps: int = 600,
    n_walkers_mult: int = 4,
    discard_frac: float = 0.5,
):
    """Fit model M2 (c0, t0, gamma, tau_1ghz) and return recovered tau_1ghz in ms.

    The sim has no dispersion, so dm_init defaults to 0. tau_1ghz is sampled in
    log-space by the fitter, hence the exp() on the chain column.
    """
    model = FRBModel(time=time_ms, freq=freq_ghz, data=data, dm_init=dm_init, df_MHz=df_MHz)

    # Peak-anchored t0 init so build_priors' dynamic t0 window brackets the pulse.
    prof = np.nansum(data, axis=0)
    t0_init = float(time_ms[int(np.argmax(prof))])
    init = FRBParams(
        c0=float(np.nanmax(prof)) or 1.0,
        t0=t0_init,
        gamma=0.0,
        zeta=float(time_ms[1] - time_ms[0]),
        tau_1ghz=tau_init_ms,
        alpha=4.0,
        delta_dm=0.0,
    )
    priors, _ = build_priors(init)

    fitter = FRBFitter(model, priors, n_steps=n_steps, n_walkers_mult=n_walkers_mult)
    sampler = fitter.sample(init, model_key="M2")

    names = FRBFitter._ORDER["M2"]
    tau_col = names.index("tau_1ghz")
    discard = int(discard_frac * n_steps)
    chain = sampler.get_chain(discard=discard, flat=True)
    tau_1ghz_ms = float(np.exp(np.median(chain[:, tau_col])))  # tau_1ghz sampled in log
    return tau_1ghz_ms, sampler


def roundtrip(cfg: SimCfg, duration: u.Quantity, *, alpha: float = 4.0, rng=None, **fit_kw):
    """Simulate -> fit -> compare. Returns (tau_true_ms, tau_fit_at_nu0_ms).

    tau_true is at nu0; the fitter reports tau at 1 GHz, so scale to nu0 by nu0^-alpha
    before comparing (tau ∝ nu^-alpha).

    CAVEAT (narrow band): the nu0^-alpha rescale is only valid when the band is
    wide enough to constrain the tau-nu lever. For a narrow fractional band (e.g.
    25 MHz at 800 MHz) the fitter cannot separate tau_1ghz from alpha, so its
    tau_1ghz comes out ~= tau at the data band, not at 1 GHz -- multiplying again
    by nu0^-alpha then double-counts it (a constant ~nu0^-alpha bias). For
    narrow-band recovery compare the raw tau_1ghz to tau_true; see
    recovery_campaign.recover_stacked.
    """
    sim, (data, time_ms, freq_ghz, df_MHz), tau_true_ms = simulate_scattered_burst(
        cfg, duration, rng=rng
    )
    tau_1ghz_ms, _ = fit_tau(data, time_ms, freq_ghz, df_MHz, **fit_kw)
    nu0_ghz = cfg.nu0.to_value(u.GHz)
    tau_fit_at_nu0_ms = tau_1ghz_ms * nu0_ghz ** (-alpha)
    return tau_true_ms, tau_fit_at_nu0_ms


__all__ = ["sim_grids_to_fitter", "simulate_scattered_burst", "fit_tau", "roundtrip"]
