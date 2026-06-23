"""Tests for analysis/chime_beam.py — documented CHIME cylinder-beam approximation + SEFD."""

import numpy as np

from analysis.chime_beam import (
    FWHM_NS_400,
    beam_gain,
    chime_sigma_jy,
    load_chime_sefd,
    sefd_zenith_jy,
)


def test_chime_gain_boresight():
    # source at its own formed-beam centre (baseband case) -> gain 1; off-axis falls below 1
    assert abs(beam_gain(120.0, 45.0, 600.0) - 1.0) < 1e-12
    g_off = beam_gain(120.0, 45.3, 600.0, ra0_deg=120.0, dec0_deg=45.0)
    assert 0.0 < g_off < 1.0


def test_chime_beam_half_power_at_fwhm_over_2():
    # Gaussian: gain = 0.5 exactly at an offset of FWHM/2 (N-S)
    fwhm_ns = FWHM_NS_400 * 400.0 / 600.0  # deg at 600 MHz
    g = beam_gain(120.0, 45.0 + fwhm_ns / 2, 600.0, ra0_deg=120.0, dec0_deg=45.0)
    assert abs(g - 0.5) < 1e-6


def test_chime_beam_is_chromatic():
    # higher frequency -> narrower beam -> lower gain at a fixed angular offset
    g_hi = beam_gain(120.0, 45.3, 800.0, ra0_deg=120.0, dec0_deg=45.0)
    g_lo = beam_gain(120.0, 45.3, 400.0, ra0_deg=120.0, dec0_deg=45.0)
    assert g_hi < g_lo


def test_chime_sigma_jy_matches_radiometer():
    freq_hz = np.linspace(400e6, 800e6, 16)
    dnu_hz, dt_s = 390625.0, 9.83e-4
    sig = chime_sigma_jy(freq_hz, dnu_hz, sefd_jy=34.5, dt_s=dt_s, g=1.0)
    expect = 34.5 / np.sqrt(2 * dnu_hz * dt_s)  # n_pol=2, G=1
    assert np.allclose(sig, expect)
    # beam attenuation G<1 raises the noise
    assert chime_sigma_jy(freq_hz, dnu_hz, 34.5, dt_s, g=0.5) == expect * 2.0


def test_chime_sefd_derivation_and_csv():
    # 2 k_B Tsys / A_eff with the documented defaults -> ~34.5 Jy
    assert abs(sefd_zenith_jy() - 34.5) < 0.5
    sefd = load_chime_sefd()
    assert 20.0 < sefd < 100.0  # sane CHIME zenith SEFD
    assert abs(sefd - sefd_zenith_jy()) < 1.0  # csv matches the derivation
