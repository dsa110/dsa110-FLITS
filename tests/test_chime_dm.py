"""Validate the custom DM tool (dispersion/chime_dm.py) on synthetic known-truth bursts.

The injector is INDEPENDENT of the estimator: each channel profile is an intrinsic Gaussian
numerically convolved with a one-sided scattering exponential (NOT the module's erfcx exgauss),
then dispersed by the residual DM. So a pass means the tool recovers a known DM from a realistic
scattered, dispersed, noisy waterfall — not that it agrees with its own model.

Covers the regime that broke structure-max (smooth pulses, scattering tails, low S/N), a wide
reference-independent offset (de-circularisation), a non-detection floor, and sigma calibration,
in BOTH the CHIME (400-800 MHz) and DSA-110 (~1.28-1.53 GHz) bands.
"""

import numpy as np
import pytest

from dispersion.chime_dm import K_DM, measure_dm


def _inject(freqs, dt, n_t, dm_ref, dm_true, t0_base, sigma_ms, tau_ref_ms, snr, rng):
    """Waterfall (n_freq, n_t): scattered pulse dispersed by (dm_true - dm_ref)."""
    nu_ref = freqs.max()
    t = np.arange(n_t) * dt
    wf = np.zeros((freqs.size, n_t))
    ddm = dm_true - dm_ref
    sig = sigma_ms * 1e-3
    for j, nu in enumerate(freqs):
        arr = t0_base + ddm * K_DM * (1.0 / nu**2 - 1.0 / nu_ref**2)
        g = np.exp(-0.5 * ((t - arr) / sig) ** 2)
        tau = tau_ref_ms * 1e-3 * (nu / nu_ref) ** -4
        if tau > 1e-6:
            k = np.exp(-np.arange(0, 8 * tau, dt) / tau)
            g = np.convolve(g, k, mode="full")[:n_t]
        wf[j] = snr * g / (g.max() + 1e-12) + rng.standard_normal(n_t)
    return wf


def _run(band, dm_ref, dm_off, tau_ref_ms, snr=4.0, seed=0, dm_window=50.0):
    """Size the window to the residual sweep, inject, and measure. ``band`` = (f_lo, f_hi)."""
    flo, fhi = band
    freqs = np.linspace(flo, fhi, 96)
    dt = 2.0e-4  # 0.2 ms — pulses well resolved (real CHIME is 2.56 us; DSA finer still)
    sweep = abs(dm_off) * K_DM * (1.0 / flo**2 - 1.0 / fhi**2)  # s, full band
    pad = sweep + 0.1
    n_t = int((2 * pad + 0.2) / dt)
    t0_base = pad + 0.05
    wf = _inject(
        freqs,
        dt,
        n_t,
        dm_ref,
        dm_ref + dm_off,
        t0_base,
        1.5,
        tau_ref_ms,
        snr,
        np.random.default_rng(seed),
    )
    return measure_dm(wf, freqs, dt, dm_ref, dm_window=dm_window, dm_step=1.0)


CHIME = (400.0, 800.0)
DSA = (1281.0, 1531.0)


@pytest.mark.parametrize("band,dm_ref", [(CHIME, 500.0), (DSA, 500.0)], ids=["chime", "dsa"])
@pytest.mark.parametrize("dm_off,tau_ref_ms", [(5.0, 0.0), (-8.0, 0.0), (12.0, 2.0), (-15.0, 5.0)])
def test_recovers_known_dm(band, dm_ref, dm_off, tau_ref_ms):
    res = _run(band, dm_ref, dm_off, tau_ref_ms, snr=4.0)
    assert res["constrains_dm"], f"should constrain: {res['reason']} (n={res['n_good_subbands']})"
    assert abs(res["dm"] - (dm_ref + dm_off)) < max(3.0 * res["dm_err"], 2.0), (
        f"dm={res['dm']:.2f} truth={dm_ref + dm_off} sigma={res['dm_err']:.2f}"
    )


def test_wide_search_finds_large_offset_decircularizes():
    # a +30 pc/cm^3 offset from the reference must be FOUND, not pinned at dm_ref -> the agreement
    # test is a real exclusion, not a circular delta~0.
    res = _run(CHIME, 500.0, 30.0, 0.0, snr=4.0)
    assert res["constrains_dm"], res["reason"]
    assert abs(res["dm"] - 530.0) < max(3.0 * res["dm_err"], 2.0), f"dm={res['dm']:.2f} (want 530)"


def test_flags_nondetection_at_low_snr():
    # per-channel S/N ~1: nothing clears the floor -> not constrained, not fabricated
    res = _run(CHIME, 500.0, 10.0, 0.0, snr=0.7, seed=1)
    assert not res["constrains_dm"]
    assert res["dm"] is None or res["dm_err"] is None


def test_dsa_narrow_band_has_larger_sigma_but_recovers():
    # DSA's tiny dispersive lever arm (~25x worse than CHIME) -> honest, larger sigma_DM.
    chime = _run(CHIME, 500.0, 6.0, 1.0, snr=6.0, seed=3)
    dsa = _run(DSA, 500.0, 6.0, 1.0, snr=6.0, seed=3)
    assert chime["constrains_dm"] and dsa["constrains_dm"]
    assert dsa["dm_err"] > chime["dm_err"], (
        f"dsa sigma {dsa['dm_err']:.2f} !> chime {chime['dm_err']:.2f}"
    )


def test_sigma_is_calibrated_across_realizations():
    pulls = []
    for s in range(12):
        res = _run(CHIME, 500.0, 10.0, 1.0, snr=6.0, seed=100 + s)
        if res["constrains_dm"]:
            pulls.append((res["dm"] - 510.0) / res["dm_err"])
    assert len(pulls) >= 8, f"too few constrained fits ({len(pulls)}/12)"
    assert np.std(pulls) < 3.0, f"pull scatter {np.std(pulls):.2f} — sigma over-confident"
