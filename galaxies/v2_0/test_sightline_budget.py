import json
import math

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

from galaxies.v2_0 import sightline_budget as sb


def _stub_dm_mw(dm_ne=80.0, dm_yw=85.0, tau_ms=5.0e-4):
    """A deterministic stand-in for the pygedm-backed Galactic model."""
    return lambda l_deg, b_deg, method="ne2001": (dm_ne, dm_yw, tau_ms)


def test_dm_cosmic_macquart_zero_monotonic_and_scale():
    assert sb.dm_cosmic_macquart(0.0) == pytest.approx(0.0, abs=1e-6)
    zs = [0.05, 0.25, 0.5, 1.0]
    vals = [sb.dm_cosmic_macquart(z) for z in zs]
    assert all(np.diff(vals) > 0)
    # Macquart+2020 relation: <DM_cosmic> ~ 850-1000 z; check the z=1 anchor.
    assert 700.0 < sb.dm_cosmic_macquart(1.0) < 1100.0
    # Near-linear low-z slope in the canonical range.
    assert 750.0 < sb.dm_cosmic_macquart(0.5) / 0.5 < 1000.0


def test_parse_dm_obs_from_filenames():
    assert sb.parse_dm_obs("data/chime/casey_chime_I_491_2085_32000b_cntr_bpc.npy") == pytest.approx(491.0)
    # DSA paths used a lowercase 'l' where Stokes I was meant.
    assert sb.parse_dm_obs("zach_dsa_l_262_368_2500b_cntr_bpc.npy") == pytest.approx(262.0)
    assert sb.parse_dm_obs("phineas_chime_I_610_2894_32000b_cntr_bpc.npy") == pytest.approx(610.0)
    assert sb.parse_dm_obs("not_a_burst_file.txt") is None


def test_read_measured_tau_ms(tmp_path):
    good = tmp_path / "good_fit_results.json"
    good.write_text(json.dumps({"best_model": "M3", "best_params": {"tau_1ghz": 0.1944}}))
    assert sb.read_measured_tau_ms(str(good)) == pytest.approx(0.1944)

    nokey = tmp_path / "nokey_fit_results.json"
    nokey.write_text(json.dumps({"best_params": {"c0": 1.0}}))
    assert sb.read_measured_tau_ms(str(nokey)) is None

    assert sb.read_measured_tau_ms(str(tmp_path / "missing.json")) is None


def _write_glade_csv(path, z=0.10, impact_kpc=20.0, mstar=10.8):
    pd.DataFrame(
        [{"ra": 310.20, "dec": 72.87, "z": z, "impact_kpc": impact_kpc,
          "catalog": "VII/291/glade", "M_star": mstar}]
    ).to_csv(path, index=False)


def test_build_sightline_budget_dm_closure(tmp_path):
    _write_glade_csv(tmp_path / "aaa_galaxies.csv")
    b = sb.build_sightline_budget(
        "Aaa", "20h40m47.886s", "+72d52m56.378s", 0.30,
        results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(dm_ne=80.0),
        dm_obs=400.0,
        tau_obs=0.05,
        enrich=False,
    )
    # Intervening DM is the sum of the foreground galaxy's hot + cool columns.
    assert math.isfinite(b["dm_intervening"])
    assert b["dm_intervening"] == pytest.approx(b["dm_intervening_hot"] + b["dm_intervening_cool"], rel=1e-9)
    assert b["dm_mw_ism"] == pytest.approx(80.0)
    assert b["dm_cosmic"] == pytest.approx(sb.dm_cosmic_macquart(0.30))
    # The residual host DM closes the observed budget exactly (raw intervening).
    expected_host = 400.0 - b["dm_mw_ism"] - b["dm_mw_halo"] - b["dm_cosmic"] - b["dm_intervening"]
    assert b["dm_host"] == pytest.approx(expected_host, rel=1e-9)
    # Observer-frame host residual deredshifted to the host rest frame.
    assert b["dm_host_rest"] == pytest.approx(b["dm_host"] * (1.0 + 0.30), rel=1e-9)
    assert b["n_foreground"] == 1

    # Capped intervening DM never exceeds the raw value and closes its own budget.
    assert b["dm_intervening_capped"] <= b["dm_intervening"] + 1e-9
    assert b["dm_intervening_regime"] in {"CGM", "GALAXY_INTERIOR", "none"}
    expected_host_cap = 400.0 - b["dm_mw_ism"] - b["dm_mw_halo"] - b["dm_cosmic"] - b["dm_intervening_capped"]
    assert b["dm_host_capped"] == pytest.approx(expected_host_cap, rel=1e-9)

    # The dominant intervening screen here has a measured (GLADE) mass.
    assert b["z_is_placeholder"] is False
    assert b["intervening_mass_source"] == "glade_catalog"
    assert b["intervening_mass_confidence"] == "measured"


def test_placeholder_redshift_withholds_cosmic_and_host(tmp_path):
    # z_frb == 1.0 is the unknown-host-redshift placeholder in this sample.
    _write_glade_csv(tmp_path / "ddd_galaxies.csv", z=0.30, impact_kpc=30.0)
    b = sb.build_sightline_budget(
        "Ddd", "20h40m47.886s", "+72d52m56.378s", 1.0,
        results_dir=str(tmp_path), dm_mw_fn=_stub_dm_mw(), dm_obs=900.0, enrich=False,
    )
    assert b["z_is_placeholder"] is True
    assert math.isnan(b["dm_cosmic"])
    assert math.isnan(b["dm_host"])
    assert math.isnan(b["dm_host_capped"])
    assert "placeholder" in b["verdict_dm"].lower()
    assert b["cgm_budget_flags"]["z_frb"] == "PLACEHOLDER"

    # A real redshift still produces a finite cosmic term.
    b2 = sb.build_sightline_budget(
        "Eee", "20h40m47.886s", "+72d52m56.378s", 0.30,
        results_dir=str(tmp_path), dm_mw_fn=_stub_dm_mw(), dm_obs=400.0, enrich=False,
    )
    assert b2["z_is_placeholder"] is False
    assert math.isfinite(b2["dm_cosmic"])


def test_no_screen_has_no_mass_confidence(tmp_path):
    b = sb.build_sightline_budget(
        "Fff", "20h40m47.886s", "+72d52m56.378s", 0.30,
        results_dir=str(tmp_path), dm_mw_fn=_stub_dm_mw(), dm_obs=300.0, enrich=False,
    )
    assert b["n_foreground"] == 0
    assert b["intervening_mass_confidence"] == "none"


def test_scattering_verdict_host_dominated_when_intervening_tiny(tmp_path):
    # Low-mass foreground galaxy well outside its virial radius -> negligible
    # predicted intervening tau, while the measured burst tau is sizeable ->
    # host/MW must dominate.
    _write_glade_csv(tmp_path / "bbb_galaxies.csv", impact_kpc=250.0, mstar=9.5)
    b = sb.build_sightline_budget(
        "Bbb", "20h40m47.886s", "+72d52m56.378s", 0.30,
        results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(tau_ms=1.0e-4),
        dm_obs=400.0,
        tau_obs=0.20,
        enrich=False,
    )
    assert b["tau_obs_ms"] == pytest.approx(0.20)
    assert b["tau_intervening_ms"] < b["tau_obs_ms"]
    assert "host" in b["verdict_scattering"].lower()


def test_scattering_verdict_no_measurement_and_no_screen(tmp_path):
    # No galaxies CSV -> no intervening screen; no tau_obs -> no measurement.
    b = sb.build_sightline_budget(
        "Ccc", "20h40m47.886s", "+72d52m56.378s", 0.30,
        results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(),
        dm_obs=300.0,
        tau_obs=None,
        enrich=False,
    )
    assert b["n_foreground"] == 0
    assert b["tau_intervening_ms"] == pytest.approx(0.0)
    assert b["tau_obs_ms"] is None or (isinstance(b["tau_obs_ms"], float) and math.isnan(b["tau_obs_ms"]))
    assert "no" in b["verdict_scattering"].lower()  # "no scattering measurement"


def test_build_all_budgets_covers_targets(tmp_path):
    _write_glade_csv(tmp_path / "aaa_galaxies.csv", impact_kpc=15.0)
    targets = [
        ("Aaa", "20h40m47.886s", "+72d52m56.378s", 0.30),
        ("Bbb", "11h51m07.52s", "+71d41m44.3s", 0.30),  # no CSV
    ]
    df = sb.build_all_budgets(
        targets=targets, results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(), dm_obs_map={"Aaa": 400.0, "Bbb": 350.0},
        tau_obs_map={"Aaa": 0.05}, enrich=False,
    )
    assert len(df) == 2
    for col in ("name", "dm_obs", "dm_mw_ism", "dm_cosmic", "dm_intervening",
                "dm_host", "tau_obs_ms", "tau_intervening_ms", "verdict_scattering"):
        assert col in df.columns
    aaa = df[df["name"] == "Aaa"].iloc[0]
    assert aaa["n_foreground"] == 1
    bbb = df[df["name"] == "Bbb"].iloc[0]
    assert bbb["n_foreground"] == 0


def test_format_budget_table_is_markdown(tmp_path):
    df = sb.build_all_budgets(
        targets=[("Zzz", "20h40m47.886s", "+72d52m56.378s", 0.30)],
        results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(), dm_obs_map={"Zzz": 300.0}, enrich=False,
    )
    table = sb.format_budget_table(df)
    assert "|" in table and "---" in table and "Zzz" in table


def test_make_budget_figure_smoke(tmp_path):
    df = sb.build_all_budgets(
        targets=[("Zzz", "20h40m47.886s", "+72d52m56.378s", 0.30),
                 ("Yyy", "11h51m07.52s", "+71d41m44.3s", 0.50)],
        results_dir=str(tmp_path),
        dm_mw_fn=_stub_dm_mw(), dm_obs_map={"Zzz": 300.0, "Yyy": 500.0}, enrich=False,
    )
    fig = sb.make_budget_figure(df)
    out = tmp_path / "budget.png"
    fig.savefig(out, dpi=80)
    assert out.exists() and out.stat().st_size > 0


def test_galactic_dm_tau_pygedm_offline():
    # NB: a raw ``import pygedm`` fails on SciPy>=1.14 (removed integrate.simps);
    # _load_pygedm applies the shim, so gate on it rather than importorskip.
    if not sb._load_pygedm():
        pytest.skip("pygedm unavailable")
    dm_ne, dm_yw, tau_ms = sb.galactic_dm_tau(106.94, 18.39, method="ne2001")
    # Plausible Galactic DM at this latitude and a small positive scattering time.
    assert 20.0 < dm_ne < 300.0
    assert dm_yw > 0.0
    assert 0.0 <= tau_ms < 1.0
