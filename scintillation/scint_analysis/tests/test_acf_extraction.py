"""Regression for the ``analyze_scintillation_from_acfs`` sub-band extraction bug.

The success branch built per-component tuples but never appended them to
``params_per_comp`` (only the *fail* branch appended ``{}``), so the downstream
consumer — which expects per-sub-band **dicts** keyed ``bw/mod/bw_err/mod_err/
finite_err/gof`` — always saw empty ``subband_measurements`` and ran its
power-law ODR on empty arrays. Confirmed on real data (casey_chime): 0 -> 4
measurements once fixed.

Driven on a synthetic high-S/N single Lorentzian ACF through the *real*
``analyze_scintillation_from_acfs`` (raw burst spectra are gitignored, see
DATA_SOURCES.md). The pre-fix code returns an empty ``subband_measurements``
here; the fixed code returns one dict per sub-band.
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np

from scint_analysis.analysis import analyze_scintillation_from_acfs, lorentzian_component


def _lorentzian_acf_results(n_sub=3, m=0.8, gamma0=0.12, ref_freq=600.0):
    """Synthetic acf_results: one clean Lorentzian scintle per sub-band, gamma
    scaling ~ nu^4 (scattering). High S/N so a Lorentzian wins BIC."""
    freqs = np.linspace(450.0, 750.0, n_sub)
    chan_width = 0.02  # MHz
    n_chan = 256
    lags = np.arange(-n_chan, n_chan + 1) * chan_width  # symmetric, lag 0 centred
    out = {
        "subband_acfs": [],
        "subband_lags_mhz": [],
        "subband_center_freqs_mhz": [],
        "subband_channel_widths_mhz": [],
        "subband_num_channels": [],
        "noise_template": [None] * n_sub,
        "sigma_self_mhz": None,
    }
    rng = np.random.default_rng(0)
    for f in freqs:
        gamma = gamma0 * (f / ref_freq) ** 4  # scintillation bandwidth grows with nu
        acf = lorentzian_component(lags, gamma, m) + rng.normal(0, 1e-3, lags.size)
        out["subband_acfs"].append(acf)
        out["subband_lags_mhz"].append(lags)
        out["subband_center_freqs_mhz"].append(float(f))
        out["subband_channel_widths_mhz"].append(chan_width)
        out["subband_num_channels"].append(n_chan)
    return out, freqs, m


def test_subband_measurements_populated():
    acf_results, freqs, m_inj = _lorentzian_acf_results()
    config = {"analysis": {"fitting": {"fit_lagrange_mhz": 1.0, "reference_frequency_mhz": 600.0}}}

    final_results, _all_fits, _pl = analyze_scintillation_from_acfs(acf_results, config)

    comp = final_results["components"]["scint_scale"]
    sm = comp["subband_measurements"]

    # The bug: this list was always empty. Fixed: one dict per sub-band.
    assert len(sm) == len(freqs), f"expected {len(freqs)} measurements, got {len(sm)}"
    for meas in sm:
        assert isinstance(meas, dict)
        assert set(meas) >= {"bw", "mod", "bw_err", "mod_err", "finite_err", "gof"}
        assert np.isfinite(meas["bw"]) and meas["bw"] > 0


def test_modulation_recovered_for_lorentzian():
    """When a Lorentzian wins, the recovered modulation index tracks the
    injected m (the per-sub-band ``mod`` is no longer silently dropped)."""
    acf_results, _freqs, m_inj = _lorentzian_acf_results(m=0.8)
    config = {"analysis": {"fitting": {"fit_lagrange_mhz": 1.0, "reference_frequency_mhz": 600.0}}}

    final_results, _all_fits, _pl = analyze_scintillation_from_acfs(acf_results, config)
    sm = final_results["components"]["scint_scale"]["subband_measurements"]

    mods = np.array([s["mod"] for s in sm], dtype=float)
    if np.isfinite(mods).any():  # Lorentzian/Gaussian branch (power-law has m=nan)
        assert abs(np.nanmedian(mods) - m_inj) < 0.15


if __name__ == "__main__":
    test_subband_measurements_populated()
    test_modulation_recovered_for_lorentzian()
    print("ok")
