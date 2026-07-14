"""Driver-level guard wiring tests for run_dsa_lorentzian_fits.py.

These exercise the actual driver helper functions (imported from the hyphenated
analysis directory) on synthetic ACFs, so the wiring between the driver and
``chime_artifact_guards`` is covered without a full pipeline run:

  - ``_fit_width`` recovers a known Lorentzian width;
  - ``_low_lag_excision_widths`` keeps the width for a resolved wing and
    collapses it for a low-lag-only bump (the arm-B1 discriminator);
  - the harmonic-mask path in the driver removes comb lag bins and changes the
    fit only when a comb is actually present.

Run: NUMBA_DISABLE_JIT=1 python -m pytest \
     analysis/scintillation-dsa-lorentzian-2026-07-07/test_driver_guards.py
from the pipeline root.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # pipeline/
sys.path.insert(0, str(_ROOT))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "rdlf_under_test", _HERE / "run_dsa_lorentzian_fits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drv = _load_driver()


def _lorentzian_acf(gamma_mhz, m, *, dch=0.01, nch=400, noise=1e-3, seed=0):
    """Symmetric mean-normalized ACF: m^2/(1+(lag/gamma)^2) + noise, lag0=1."""
    rng = np.random.default_rng(seed)
    pos = np.arange(1, nch + 1) * dch
    lags = np.concatenate((-pos[::-1], [0.0], pos))
    acf = m**2 / (1 + (lags / gamma_mhz) ** 2)
    acf[lags == 0] = 1.0
    acf = acf + rng.normal(0, noise, lags.size)
    err = np.full(lags.size, noise)
    return lags, acf, err, dch


def test_fit_width_recovers_known_lorentzian():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    w = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    assert w is not None
    assert abs(w - 0.2) < 0.05  # within 25% of the injected 0.2 MHz


def test_pbf_bandwidth_conversion_has_correct_millisecond_to_megahertz_units():
    # Analytic oracle at 1 GHz: 1.16 / (2 pi 0.12 ms) = 1.5385 kHz.
    predicted = drv._pbf_bandwidth_mhz(
        np.asarray([1000.0]), tau_1ghz_ms=0.12, alpha=4.3, c1=1.16
    )
    assert predicted[0] == pytest.approx(0.0015385, rel=2e-5)


def test_pbf_loader_follows_locked_roster_and_excludes_gate_failures():
    assert drv._load_pbf_fit_for_burst("mahi")["_source"].endswith(
        "mahi_joint_fit_C1D2.json"
    )
    for excluded in ("zach", "wilhelm", "hamilton", "chromatica", "casey"):
        assert drv._load_pbf_fit_for_burst(excluded) is None
    assert not drv._pbf_roster_entry_is_eligible(
        {
            "nickname": "synthetic",
            "gate_final": "FAIL",
            "rail_class": "interior",
            "fit_json": "would_otherwise_load.json",
        }
    )


def test_component_tracks_sort_widths_within_each_subband():
    rows = [
        {
            "subband": subband,
            "center_freq_mhz": frequency,
            "dnu_mhz": gamma,
            "dnu_err_mhz": 0.05,
            "usable": True,
        }
        for subband, frequency, gamma in (
            (0, 1325.0, 1.0),
            (0, 1325.0, 8.0),
            (1, 1465.0, 1.6),
        )
    ]
    tracked = drv._assign_gamma_tracks(rows)
    assert [row["gamma_track"] for row in tracked if row["subband"] == 0] == [1, 2]
    assert [row["dnu_mhz"] for row in tracked if row["subband"] == 0] == [1.0, 8.0]


def test_gamma_power_law_uses_gamma_1_only():
    rows = [
        {
            "subband": subband,
            "center_freq_mhz": frequency,
            "dnu_mhz": gamma,
            "dnu_err_mhz": 0.05,
            "usable": True,
        }
        for subband, frequency, gamma in (
            (0, 1325.0, 1.0),
            (0, 1325.0, 8.0),
            (1, 1465.0, 1.6),
            (1, 1465.0, 12.0),
        )
    ]
    fit = drv._fit_gamma_power_law(drv._assign_gamma_tracks(rows))
    assert fit is not None
    assert fit["selection_policy"] == "gamma_1_only"
    assert fit["n_fit_components"] == 2
    assert fit["included_tracks"] == [1]


def test_low_lag_excision_keeps_resolved_wing():
    # A real, well-resolved Lorentzian (gamma >> channel): width survives
    # dropping the first few channel lags.
    lags, acf, err, dch = _lorentzian_acf(0.3, 0.8, dch=0.01, nch=400)
    keep = np.abs(lags) <= 3.0
    full = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    excised = drv._low_lag_excision_widths(
        lags[keep], acf[keep], err[keep], dch, max_components=1, ks=(1, 2, 3)
    )
    verdict = drv.guards.low_lag_stability_verdict(full, excised)
    assert verdict["stable"] is True


def test_low_lag_excision_collapses_on_no_wing_artifact():
    # The freya CHIME failure signature: correlated power carried ENTIRELY by the
    # first couple of channel lags with NO Lorentzian wing (flat noise beyond).
    # Excising the low lags must collapse the fitted width.
    dch = 0.01
    rng = np.random.default_rng(0)
    pos = np.arange(1, 401) * dch
    lags = np.concatenate((-pos[::-1], [0.0], pos))
    acf = np.zeros_like(lags)
    lag_ch = np.round(np.abs(lags) / dch).astype(int)
    for k in (1, 2):  # power only at |lag| = 1, 2 channels
        acf[np.abs(lag_ch - k) < 0.5] = 0.25 * (0.5 ** (k - 1))
    acf[lags == 0] = 1.0
    acf = acf + rng.normal(0, 5e-4, lags.size)
    err = np.full(lags.size, 5e-4)

    keep = np.abs(lags) <= 0.5
    full = drv._fit_width(lags[keep], acf[keep], err[keep], max_components=1)
    excised = drv._low_lag_excision_widths(
        lags[keep], acf[keep], err[keep], dch, max_components=1, ks=(1, 2, 3)
    )
    verdict = drv.guards.low_lag_stability_verdict(full, excised)
    assert verdict["stable"] is False
    assert set(verdict["failed_ks"]) & {1, 2}  # collapsed at the low-lag excisions


def test_driver_harmonic_mask_removes_comb_bins():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    L, A, E, rec = drv.guards.apply_harmonic_mask_to_fit(
        lags[keep], acf[keep], err[keep],
        {"enable": True, "spacing_mhz": 0.390625, "halfwidth_mhz": 0.05},
    )
    assert rec["enabled"] is True
    assert rec["n_bins_removed"] > 0
    assert L.size < int(np.sum(keep))


def test_driver_harmonic_mask_disabled_is_passthrough():
    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8)
    keep = np.abs(lags) <= 2.0
    L, A, E, rec = drv.guards.apply_harmonic_mask_to_fit(
        lags[keep], acf[keep], err[keep], {"enable": False}
    )
    assert rec["n_bins_removed"] == 0
    assert L.size == int(np.sum(keep))


def test_public_diagnostic_plot_uses_experiment_style_contract(tmp_path):
    """The tracked producer emits a spacious, explicitly diagnostic figure."""
    import matplotlib.image as mpimg

    lags, acf, err, _ = _lorentzian_acf(0.2, 0.8, dch=0.01, nch=80, noise=1e-3)
    component = {
        "dnu_mhz": 0.2,
        "dnu_err": 0.02,
        "m": 0.8,
        "m_err": 0.04,
        "quality_flags": [],
    }
    payloads = []
    for index, center_freq_mhz in enumerate((1325.0, 1365.0, 1410.0, 1465.0)):
        scaled_component = {
            **component,
            "dnu_mhz": 0.2 * (center_freq_mhz / 1400.0) ** 4.2,
        }
        broad_component = {
            **component,
            "dnu_mhz": 1.5 * (center_freq_mhz / 1400.0) ** 2.0,
        }
        payloads.append(
            {
                "lags": lags,
                "acf": acf,
                "err": err,
                "summary": {
                    "index": index,
                    "center_freq_mhz": center_freq_mhz,
                    "channel_width_mhz": 0.01,
                    "fit_range_mhz": 0.8,
                    "selected_redchi": 1.05,
                    "selected_components": [broad_component, scaled_component],
                },
                "fit": {
                    "constant": 0.0,
                    "components": [broad_component, scaled_component],
                },
            }
        )

    outputs = drv.plot_burst_acf_diagnostic(
        "freya",
        payloads,
        figure_dir=tmp_path,
        band="dsa",
        pbf_fit={
            "tau_1ghz": {"median": 0.12, "err_minus": 0.01, "err_plus": 0.02},
            "alpha": {"median": 4.3, "err_minus": 0.2, "err_plus": 0.3},
        },
    )

    svg = Path(outputs["figure_svg"]).read_text()
    assert "scaling fit" in svg
    assert "excluded from scaling fit" in svg
    assert "PBF-derived" in svg
    assert "C_1=1.16" in svg
    assert "Frequency Lag" in svg
    assert "Validation context" not in svg
    assert "Positive frequency lag" not in svg
    assert outputs["gamma_power_law_fit"]["alpha"] == pytest.approx(4.2, abs=0.05)
    assert outputs["gamma_power_law_fit"]["included_tracks"] == [1]
    assert outputs["pbf_overlay"]["alpha"] == pytest.approx(4.3)
    assert outputs["pbf_overlay"]["tau_1ghz_ms"] == pytest.approx(0.12)
    assert outputs["pbf_overlay"]["c1"] == pytest.approx(1.16)
    image = mpimg.imread(outputs["figure_png"])
    assert image.shape[1] > image.shape[0]
    assert Path(outputs["figure_pdf"]).read_bytes().startswith(b"%PDF")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
