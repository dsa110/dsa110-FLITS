"""Flux calibration wired into the energetics gate (Phases 5-7).

The "fluxcal" sentinel routes a band through the model-based radiometer integral
(analysis.flux_cal.joint_band_fluence_jy_ms_hz(nick, band)) instead of native-units x scalar.
The both-bands-or-nothing gate only emits a summed E_iso when both bands carry a real Jy scale.
"""

import numpy as np

import analysis.calculate_burst_energies as E


def test_dsa_calibrated_but_gate_closed(monkeypatch):
    # D via the model-based integral, C still uncalibrated -> gate closed (no summed energy), but
    # the DSA band's Jy*ms*Hz integral is still surfaced on the row.
    monkeypatch.setattr(E, "joint_band_fluence_jy_ms_hz", lambda n, band: 1.234e6)
    rows = E.compute(scales={"C": None, "D": "fluxcal"})
    assert rows
    assert all("E_iso_erg" not in r for r in rows)
    assert any(abs(r.get("I_DSA_jy_ms_hz", 0) - 1.234e6) < 1 for r in rows)


def test_both_bands_emit(monkeypatch):
    # Phase 7: both bands routed through the model-based radiometer integral ("fluxcal") -> the gate
    # opens, E_iso is summed from two Jy integrals, the (1+z) k-correction identity holds exactly,
    # and the propagated 1-sigma error column is finite and positive (stat c0 + SEFD/beam systematic).
    monkeypatch.setattr(
        E, "joint_band_fluence_jy_ms_hz", lambda n, band: 2.0e6 if band == "C" else 1.5e6
    )
    monkeypatch.setattr(E, "joint_c0_gamma", lambda n, band: (1.0, -3.0, 0.1))
    rows = E.compute(scales={"C": "fluxcal", "D": "fluxcal"})
    assert rows and all("E_iso_erg" in r for r in rows)
    r = rows[0]
    assert abs(r["E_iso_erg"] - r["E_iso_erg_no_kcorr"] / (1.0 + r["z"])) < 1e-9 * r["E_iso_erg"]
    assert r["I_CHIME_jy_ms_hz"] == 2.0e6 and r["I_DSA_jy_ms_hz"] == 1.5e6
    assert np.isfinite(r["E_iso_erg_err"]) and r["E_iso_erg_err"] > 0


def test_mixed_scalar_and_fluxcal_opens_gate(monkeypatch):
    # a legacy float scale on C plus the model-based integral on D also opens the gate
    monkeypatch.setattr(E, "joint_band_fluence_jy_ms_hz", lambda n, band: 2.0e6)
    monkeypatch.setattr(E, "joint_c0_gamma", lambda n, band: (1.0, -3.0, 0.1))
    rows = E.compute(scales={"C": 5.0, "D": "fluxcal"})
    assert rows and all("E_iso_erg" in r for r in rows)
    r = rows[0]
    assert abs(r["E_iso_erg"] - r["E_iso_erg_no_kcorr"] / (1.0 + r["z"])) < 1e-9 * r["E_iso_erg"]
    assert "E_iso_CHIME_erg" in r and "E_iso_DSA_erg" in r and r["I_DSA_jy_ms_hz"] == 2.0e6


def test_dsa_burst_config_resolves():
    # data-independent: the batch config maps nick -> the canonical .npy name + the fit's binning
    from analysis.flux_cal import _dsa_burst_config

    npy, f_factor, t_factor = _dsa_burst_config("chromatica")
    assert npy.name == "chromatica_dsa_I_272_368_2500b_cntr_bpc.npy"
    assert f_factor == 384 and t_factor == 2
