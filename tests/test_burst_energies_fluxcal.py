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


def test_energy_gate_is_alpha_independent():
    # Trust boundary (ADR-0003/0004; 3-expert panel 2026-06-24). E_iso is
    # alpha-INDEPENDENT: band_integral uses only per-band c0/gamma + band edges, so
    # energy citability must NOT gate on the scattering joint-fit quality_flag (the
    # shared-alpha L1 verdict). The gate is on the energy's OWN inputs: a fit is kept
    # iff its per-band c0/gamma are finite and c0 > 0; quality_flag rides along as an
    # informational column. Reads JSON only -> data-independent.
    flags = E.load_gate_flags()
    params = E.load_joint_params()
    kept = set(params)
    live_fail = {b for b, f in flags.items() if f == "FAIL"}
    if live_fail:
        # decouple is real: at least one FAIL-flagged burst is retained (physical c0/gamma)
        assert live_fail & kept, (
            "expected FAIL-but-physical bursts retained for E_iso (energy is alpha-independent), "
            f"but every FAIL burst was dropped: live_fail={sorted(live_fail)}"
        )
    else:
        assert kept, "no joint c0/gamma fits loaded"
    # the actual gate: every kept burst has physical energy inputs, and quality_flag
    # is metadata only (never the exclusion criterion)
    for nick, p in params.items():
        assert all(np.isfinite(p[k]) for k in ("c0_C", "gamma_C", "c0_D", "gamma_D")), nick
        assert p["c0_C"] > 0 and p["c0_D"] > 0, f"non-physical c0 kept: {nick}"
        assert p["quality_flag"] == flags.get(nick)


def test_gate_drops_nonphysical_and_skips_amplitudeless(tmp_path, monkeypatch):
    # The live sidecars are all physical, so --check / the test above never hit the
    # drop or skip branches. Synthetic fits exercise them directly: a FAIL-but-physical
    # fit is KEPT (quality_flag is metadata, not an exclusion); c0<=0 or non-finite is
    # DROPPED; a fit with no per-band c0/gamma keys (all-exp/scattering-only) is SKIPPED.
    import json

    def _fit(burst, c0c=1.0, gc=-3.0, c0d=1.0, gd=-3.0, ampless=False):
        pct = (
            {}
            if ampless
            else {
                "c0_C": {"median": c0c},
                "gamma_C": {"median": gc},
                "c0_D": {"median": c0d},
                "gamma_D": {"median": gd},
            }
        )
        return {"burst": burst, "percentiles": pct}

    fits = {
        "keepfail": _fit("keepfail"),  # physical -> kept even though flagged FAIL
        "dropneg": _fit("dropneg", c0c=-1.0),  # c0 <= 0 -> dropped
        "dropnan": _fit("dropnan", gd=float("nan")),  # non-finite -> dropped
        "skipnoamp": _fit("skipnoamp", ampless=True),  # no c0/gamma -> skipped
    }
    for nick, d in fits.items():
        (tmp_path / f"{nick}_joint_fit.json").write_text(json.dumps(d))
    monkeypatch.setattr(E, "JOINT_DIR", tmp_path)
    monkeypatch.setattr(E, "load_gate_flags", lambda: {"keepfail": "FAIL"})

    kept = E.load_joint_params()
    assert set(kept) == {"keepfail"}, f"expected only the physical fit kept, got {sorted(kept)}"
    assert kept["keepfail"]["quality_flag"] == "FAIL"  # FAIL retained as metadata


def test_z_provenance_flags_unpublished_hosts():
    # every E_iso host has a redshift-provenance entry; only hamilton/chromatica are provisional
    # (no published host paper). Guards against silently presenting an unpublished z as catalog spec.
    energy = {"zach", "whitney", "oran", "isha", "phineas", "wilhelm", "hamilton", "chromatica"}
    assert energy <= set(E.Z_PROVENANCE)
    provisional = {n for n, (q, _) in E.Z_PROVENANCE.items() if q.endswith("provisional")}
    assert provisional == {"hamilton", "chromatica"}
