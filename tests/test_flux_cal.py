"""Tests for analysis/flux_cal.py — radiometer flux calibration (S/N -> Jy)."""

import numpy as np

from analysis.flux_cal import calibrated_band_integral_jy_ms_hz, radiometer_sigma_jy


def test_sigma_jy_analytic():
    # SEFD=2000 Jy, n_pol=2, dnu=1e6 Hz, dt=1e-3 s, G=1 -> 2000/sqrt(2*1e6*1e-3) = sqrt(2000)
    s = radiometer_sigma_jy(2000.0, n_pol=2, dnu_hz=1e6, dt_s=1e-3, g=1.0)
    assert abs(s - 2000.0 / np.sqrt(2000.0)) < 1e-9
    # beam attenuation G=0.5 doubles the noise
    assert abs(radiometer_sigma_jy(2000.0, 2, 1e6, 1e-3, 0.5) - 2.0 * s) < 1e-9


def test_band_integral_flat_oracle():
    # flat per-channel S/N integral A, flat sigma_S=s0, band [nu1,nu2]:
    # integral = A*s0*dt_ms*(nu2-nu1)
    nf = 64
    freq_hz = np.linspace(1.311e9, 1.499e9, nf)
    sn_integrated = np.full(nf, 3.0)  # per-channel sum_onpulse(S/N)
    sigma_jy = np.full(nf, 5.0)  # per-channel sigma_S [Jy]
    dt_ms = 0.131072
    i_band = calibrated_band_integral_jy_ms_hz(sn_integrated, sigma_jy, freq_hz, dt_ms)
    oracle = 3.0 * 5.0 * dt_ms * (freq_hz[-1] - freq_hz[0])
    assert abs(i_band - oracle) / oracle < 1e-9


def test_sn_spectrum_synthetic(tmp_path):
    # synthetic (n_freq, n_time) S/N data: unit noise + a centred Gaussian pulse on every channel
    from analysis.flux_cal import sn_spectrum_from_npy

    rng = np.random.default_rng(0)
    nf, nt = 64, 512
    t = np.arange(nt)
    prof = np.exp(-0.5 * ((t - nt // 2) / 4.0) ** 2)
    data = rng.standard_normal((nf, nt)) + 8.0 * prof[None, :]  # S/N peak ~8
    p = tmp_path / "synth.npy"
    np.save(p, data)
    freq_hz, sn_int, dt_ms, dnu_hz = sn_spectrum_from_npy(
        p, telescope="dsa", f_factor=1, t_factor=1
    )
    assert sn_int.shape == (nf,)
    assert np.all(np.isfinite(sn_int))
    assert np.median(sn_int) > 5.0  # integrated on-pulse S/N well above zero
    assert freq_hz[0] < freq_hz[-1] and dt_ms > 0 and dnu_hz > 0


def test_dsa_sigma_jy_constant_at_boresight():
    from analysis.flux_cal import dsa_sigma_jy

    freq_hz = np.linspace(1.311e9, 1.499e9, 32)
    dnu_hz, dt_s = 30517.578, 1.31072e-4
    sig = dsa_sigma_jy(
        freq_hz,
        dnu_hz,
        sefd_jy=4000.0,
        dt_s=dt_s,
        theta_deg=0.0,
        phi_deg=0.0,
        beam_gain_fn=lambda th, ph, f: 1.0,
    )
    expect = 4000.0 / np.sqrt(2 * dnu_hz * dt_s)  # G=1, n_pol=2 -> constant analytic value
    assert np.allclose(sig, expect)


def test_dsa_beam_offset_is_dec_difference():
    from analysis.flux_cal import dsa_beam_offset

    # transit geometry: same-RA boresight -> separation is exactly the dec difference
    theta, phi = dsa_beam_offset(dec_src=70.31, dec_pointing=72.0)
    assert abs(theta - 1.69) < 1e-6 and phi == 0.0


def test_burst_epoch_position_from_yaml():
    from analysis.flux_cal import burst_epoch_position

    mjd, ra, dec = burst_epoch_position("casey")
    assert abs(mjd - 60369.371) < 1e-3
    assert abs(ra - 169.983542) < 1e-6 and abs(dec - 70.676222) < 1e-6


SAMPLE = [
    "casey",
    "chromatica",
    "freya",
    "hamilton",
    "isha",
    "johndoeii",
    "mahi",
    "oran",
    "phineas",
    "whitney",
    "wilhelm",
    "zach",
]


def test_dsa_pointing_csv_and_offsets():
    # every sample burst has a pointing Dec near the DSA survey strip; offset is within the beam
    from analysis.flux_cal import burst_epoch_position, dsa_beam_offset, dsa_pointing_dec

    for n in SAMPLE:
        pdec = dsa_pointing_dec(n)
        assert 70.0 < pdec < 73.0, (n, pdec)  # DSA transit pointing ~71.6 deg
        _, _ra, dec = burst_epoch_position(n)
        theta, phi = dsa_beam_offset(dec, pdec)
        assert 0.0 <= theta < 5.0 and phi == 0.0, (n, theta)  # inside the primary beam


def test_dsa_fluence_matches_catalog_scale():
    # absolute-scale gate: with the coherent-beam SEFD (per-element / N_ant), the calibrated
    # band-avg fluences must land near the Law+2024 catalog values (16.2/26.2/13.2 Jy*ms). They
    # differ by up to ~2x because the per-channel linear integral and the catalog boxcar
    # matched-filter are different fluence estimators -- so the gate is a factor-of-3 sanity band,
    # not equality. Guards against re-introducing the per-element-SEFD ~100x error.
    import pytest

    from analysis.flux_cal import _dsa_burst_config, dsa_band_fluence_jy_ms_hz

    CATALOG = {"zach": 16.2, "whitney": 26.2, "oran": 13.2}  # Law+2024 (arXiv:2307.03344) Table 1
    band_hz = 1.499e9 - 1.311e9
    for nick, f_cat in CATALOG.items():
        npy, _, _ = _dsa_burst_config(nick)
        if not npy.exists():
            pytest.skip(f"{npy.name} not staged (data/dsa/ external)")
        band_avg = dsa_band_fluence_jy_ms_hz(nick) / band_hz
        assert 1.0 / 3 < band_avg / f_cat < 3.0, (nick, band_avg, f_cat)


def test_joint_band_fluence_matches_catalog_scale():
    # the model-based DSA fluence actually used for E_iso (joint c0/gamma, measured beam,
    # coherent-beam SEFD, /noise_std bandpass-unit correction over the fit's valid channels) must
    # land within ~3x of the Law+2024 catalog. Guards the c0/noise_std correction (without it the
    # bandpass-unit c0 reads ~15-25x low) and the valid-channel mask (without it RFI channels with
    # noise_std~0 blow the integral up by orders of magnitude).
    import pytest

    from analysis.flux_cal import _band_burst_config, joint_band_fluence_jy_ms_hz

    CATALOG = {"zach": 16.2, "whitney": 26.2, "oran": 13.2}  # Law+2024 (arXiv:2307.03344) Table 1
    band_hz = 1.499e9 - 1.311e9
    for nick, f_cat in CATALOG.items():
        npy, _, _, _ = _band_burst_config(nick, "D")
        if not npy.exists():
            pytest.skip(f"{npy.name} not staged (data/dsa/ external)")
        band_avg = joint_band_fluence_jy_ms_hz(nick, "D") / band_hz
        assert 1.0 / 3 < band_avg / f_cat < 3.0, (nick, band_avg, f_cat)


def test_dsa_sefd_csv_present_and_sane():
    import csv
    from pathlib import Path

    from analysis.flux_cal import load_dsa_sefd

    p = Path("analysis/burst_energies/dsa_sefd.csv")
    assert p.exists(), "run analysis/burst_energies/fetch_dsa_sefd.py"
    rows = list(csv.DictReader(p.open()))
    assert rows and {"burst", "mjd", "sefd_jy", "source"} <= set(rows[0])
    for r in rows:
        assert 500.0 < float(r["sefd_jy"]) < 20000.0, r  # sane DSA array SEFD
    for n in SAMPLE:
        assert 500.0 < load_dsa_sefd(n) < 20000.0
