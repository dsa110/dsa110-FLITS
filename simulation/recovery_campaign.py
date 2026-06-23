"""Quantitative tau-recovery campaign (B.5).

The single-realisation sim->fit roundtrip is a spiky multipath IRF
(tests/test_sim_fit_roundtrip.py); ensemble-averaging the dynamic spectra over
screen realisations suppresses the scintillation and leaves the smooth
broadening the fitter's exponential PBF can latch onto. We average n_real
spectra, fit once, and compare the recovered tau to the injected geometric tau.

Finding (probed 2026-06-22): recovery is *linear* -- recovered tau tracks
injected tau with a constant ratio across ~80x in tau (scale_L 1/3/9 ->
ratio 2.47/2.49/2.50), so the fitter recovers *relative* tau (the science:
cross-band tau ratios, tau-nu slope) faithfully. The comparison is made at the
data band: the raw fitter tau_1ghz ~= tau_true(nu0) to ~2%, because a 25 MHz
band at 800 MHz (3% fractional) cannot constrain the 1 GHz extrapolation. Do NOT
also multiply by nu0^-alpha -- that double-counts the reference-frequency
convention and inflates the ratio by ~2.44 (= 0.8^-4); see the narrow-band
caveat in sim_fit_bridge.roundtrip.

Dnu recovery (recover_dnu_stacked / dnu_recovery_curve) is the scintillation-
bandwidth analogue and is likewise *linear*: recovered Dnu tracks injected
nu_s_host with a constant ratio ~0.29 over ~5x in Dnu (the half-max-HWHM
estimator vs the Eq.-4.14 nu_s definition -- the Dnu analogue of tau's nu0^-alpha
constant). Here the ACFs are averaged, not the spectra (averaging spectra would
erase the scintillation), and a fine-channel, small-host-L config is required to
keep Dnu resolvable and well-sampled.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from sim_fit_bridge import fit_tau, simulate_scattered_burst


def default_cfg(scale_L=1.0):
    """Host-dominated single screen; scale_L scales the host scattering-disk size
    (=> injected tau) while keeping geometry fixed."""
    from engine import SimCfg
    from screen import ScreenCfg

    return SimCfg(
        peak_flux=5 * u.Jy,
        nu0=800 * u.MHz,
        bw=25.0 * u.MHz,
        nchan=256,
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=128, L=0.2 * u.AU, rng_seed=1234),
        host=ScreenCfg(N=128, L=20.0 * scale_L * u.AU, rng_seed=5678),
        intrinsic_pulse="delta",
    )


def recover_stacked(cfg, *, n_real=16, seed0=200, duration=12.0 * u.ms, **fit_kw):
    """Average n_real dynamic spectra, fit the stack once.

    Returns (tau_true_ms, tau_fit_ms). tau_fit is the raw fitter tau_1ghz,
    compared at the data band (no nu0^-alpha rescale -- see module docstring).
    """
    stack, grids, tau_true = None, None, None
    for k in range(n_real):
        rng = np.random.default_rng(seed0 + k)
        _, (data, t_ms, f_ghz, df), tau_true = simulate_scattered_burst(
            cfg, duration=duration, rng=rng
        )
        stack = data if stack is None else stack + data
        grids = (t_ms, f_ghz, df)
    stack /= n_real
    t_ms, f_ghz, df = grids
    tau_fit, _ = fit_tau(stack, t_ms, f_ghz, df, tau_init_ms=max(tau_true, 0.05), **fit_kw)
    return float(tau_true), float(tau_fit)


def recovery_curve(scale_L=(1.0, 3.0, 9.0), *, n_real=16, n_steps=800, **kw):
    """Recovery across a few host-screen sizes (=> a few injected tau values)."""
    rows = []
    for s in scale_L:
        tt, tf = recover_stacked(default_cfg(s), n_real=n_real, n_steps=n_steps, **kw)
        rows.append(
            {"scale_L": s, "tau_true_ms": tt, "tau_fit_ms": tf, "ratio": tf / tt if tt else np.nan}
        )
    return pd.DataFrame(rows)


def dnu_cfg(host_L_AU=1.5, nchan=2048):
    """Host-dominated, fine-channel config for resolving the scintillation
    bandwidth Dnu. MW screen tiny-L so nu_s_mw >> band (flat, no structure);
    host L sets the (resolvable) decorrelation bandwidth. Fine nchan because Dnu
    here is sub-MHz; coarse channels (the tau config) leave it unresolved -- the
    inverse tau<->Dnu tension (big L: big tau, tiny Dnu)."""
    from engine import SimCfg
    from screen import ScreenCfg

    return SimCfg(
        peak_flux=5 * u.Jy,
        nu0=800 * u.MHz,
        bw=25.0 * u.MHz,
        nchan=nchan,
        z_host=0.192,
        D_mw=2.3 * u.kpc,
        D_host_src=2.0 * u.kpc,
        mw=ScreenCfg(N=128, L=0.05 * u.AU, rng_seed=1234),
        host=ScreenCfg(N=128, L=host_L_AU * u.AU, rng_seed=5678),
        intrinsic_pulse="delta",
    )


def _acf_hwhm_hz(mean_acf, dnu_hz):
    """Half-max half-width of an ACF, in Hz (interpolated half-crossing lag)."""
    c0 = mean_acf[0]
    if c0 <= 0:
        return np.nan
    below = np.where(mean_acf < c0 / 2.0)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    frac = (mean_acf[i - 1] - c0 / 2.0) / (mean_acf[i - 1] - mean_acf[i])
    return (i - 1 + frac) * dnu_hz


def recover_dnu_stacked(cfg, *, n_real=16, seed0=400, duration=4.0 * u.ms):
    """Average n_real frequency ACFs, measure the half-max width.

    Returns (dnu_true_hz, dnu_fit_hz). The *ACFs* are averaged, not the spectra:
    the scintillation IS the per-realisation speckle that spectrum-averaging
    would erase, but every realisation's ACF estimates the same underlying Dnu,
    so averaging ACFs suppresses estimator noise. Stay in the well-sampled regime
    (Dnu << band, >~25 scintles): too few scintles bias the width low; Dnu near a
    few channels is channel-limited.
    """
    from engine import FRBScintillator

    sim = FRBScintillator(cfg)
    dnu_true = sim.calculate_theoretical_observables()["nu_s_host_hz"]
    acc = None
    for k in range(n_real):
        rng = np.random.default_rng(seed0 + k)
        I_t_nu, _, _ = sim.synthesise_dynamic_spectrum(duration=duration, rng=rng)
        corr, _ = sim.acf(np.nansum(I_t_nu, axis=0))
        acc = corr if acc is None else acc + corr
    return float(dnu_true), float(_acf_hwhm_hz(acc / n_real, sim.dnu_hz))


def dnu_recovery_curve(host_L=(1.0, 1.5, 2.2), *, n_real=16, **kw):
    """Dnu recovery across a few host-screen sizes (=> a few injected Dnu)."""
    rows = []
    for L in host_L:
        tt, tf = recover_dnu_stacked(dnu_cfg(L), n_real=n_real, **kw)
        rows.append(
            {
                "host_L_AU": L,
                "dnu_true_MHz": tt / 1e6,
                "dnu_fit_MHz": tf / 1e6,
                "ratio": tf / tt if tt else np.nan,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import logging

    logging.disable(logging.WARNING)

    df = recovery_curve(n_real=16, n_steps=800)
    print(df.to_string(index=False))
    r = df["ratio"].to_numpy()
    print(f"tau ratio mean={r.mean():.3f} spread={r.std() / r.mean():.2%}")
    df.to_csv("results/recovery_campaign.csv", index=False)
    assert df["tau_fit_ms"].gt(0).all() and np.isfinite(r).all()
    assert r.std() / r.mean() < 0.15, "tau recovery not linear (ratio not constant)"

    dn = dnu_recovery_curve(n_real=16)
    print(dn.to_string(index=False))
    rd = dn["ratio"].to_numpy()
    print(f"dnu ratio mean={rd.mean():.3f} spread={rd.std() / rd.mean():.2%}")
    dn.to_csv("results/recovery_campaign_dnu.csv", index=False)
    assert dn["dnu_fit_MHz"].gt(0).all() and np.isfinite(rd).all()
    assert rd.std() / rd.mean() < 0.15, "dnu recovery not linear (ratio not constant)"
    print("ok: linear tau + dnu recovery -> results/recovery_campaign{,_dnu}.csv")
