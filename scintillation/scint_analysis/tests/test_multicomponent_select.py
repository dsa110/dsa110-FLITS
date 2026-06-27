"""Pipeline wiring: analyze_scintillation_from_acfs determines and uses the
statistically-justified number of Lorentzian components (BIC + nested F-test, via
revalidation.compare_lorentzian_components), instead of the dead "2c"/"3c"-in-name
heuristic that no model ever emitted.

Driven on synthetic acf_results (raw burst spectra are gitignored, DATA_SOURCES.md):
a known one- vs two-component ACF in every sub-band must come back as
n_components 1 vs 2, with the two screens recovered as component_1 (narrow) and
component_2 (wide).
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np

from scint_analysis.analysis import (
    _components_ambiguous,
    _determine_n_components,
    analyze_scintillation_from_acfs,
    lorentzian_component,
)

_CFG = {
    "analysis": {
        "fitting": {
            "fit_lagrange_mhz": 2.0,
            "reference_frequency_mhz": 600.0,
            "force_model": "fit_lor",  # deterministic Lorentzian gate
        }
    }
}


def _acf_results(component_sets, dch=0.01, nch=256, noise=2e-3, seed=0):
    """acf_results dict (no noise template/self-noise -> single-prefix model labels)
    whose every sub-band ACF is the sum of the given (gamma, m) Lorentzians."""
    rng = np.random.default_rng(seed)
    pos = np.arange(1, nch + 1) * dch
    lags = np.concatenate((-pos[::-1], [0.0], pos))
    out = {
        "subband_acfs": [],
        "subband_lags_mhz": [],
        "subband_center_freqs_mhz": [],
        "subband_channel_widths_mhz": [],
        "subband_num_channels": [],
        "noise_template": None,
        "sigma_self_mhz": None,
    }
    freqs = np.linspace(450.0, 750.0, len(component_sets))
    for f, comps in zip(freqs, component_sets, strict=True):
        acf = np.zeros(lags.size)
        for g, m in comps:
            acf = acf + lorentzian_component(lags, g, m)
        acf = acf + rng.normal(0, noise, lags.size)
        out["subband_acfs"].append(acf)
        out["subband_lags_mhz"].append(lags)
        out["subband_center_freqs_mhz"].append(float(f))
        out["subband_channel_widths_mhz"].append(dch)
        out["subband_num_channels"].append(nch)
    return out


def test_determine_n_components_counts():
    assert _determine_n_components(_acf_results([[(0.1, 0.8)]] * 4))[0] == 1
    assert _determine_n_components(_acf_results([[(0.04, 0.6), (0.6, 0.6)]] * 4))[0] == 2


def test_two_components_wired_into_output():
    fr, _fits, _pl = analyze_scintillation_from_acfs(
        _acf_results([[(0.04, 0.6), (0.6, 0.6)]] * 4), _CFG
    )
    assert fr["n_components"] == 2
    assert set(fr["components"]) == {"component_1", "component_2"}
    assert fr["component_selection"]["n_per_subband"] == [2, 2, 2, 2]
    # component_1 = narrow (≈0.04), component_2 = wide (≈0.6), recovered at ref freq.
    assert abs(fr["components"]["component_1"]["bw_at_ref_mhz"] - 0.04) < 0.02
    assert abs(fr["components"]["component_2"]["bw_at_ref_mhz"] - 0.6) < 0.15
    for name in ("component_1", "component_2"):
        assert len(fr["components"][name]["subband_measurements"]) == 4


def test_single_component_unchanged():
    fr, _fits, _pl = analyze_scintillation_from_acfs(_acf_results([[(0.1, 0.8)]] * 4), _CFG)
    assert fr["n_components"] == 1
    assert list(fr["components"]) == ["scint_scale"]


def test_non_lorentzian_best_model_stays_single():
    """The component-count search is gated on a Lorentzian best model; force a
    power-law and the burst must stay single-component (no spurious multi-fit)."""
    cfg = {
        "analysis": {
            "fitting": {
                "fit_lagrange_mhz": 2.0,
                "reference_frequency_mhz": 600.0,
                "force_model": "fit_power",
            }
        }
    }
    fr, _fits, _pl = analyze_scintillation_from_acfs(
        _acf_results([[(0.04, 0.6), (0.6, 0.6)]] * 4), cfg
    )
    assert fr["n_components"] == 1
    assert "component_selection" not in fr  # determination not run for non-Lorentzian


def test_underjustified_subband_dropped():
    """Plurality picks 2 components, but a sub-band that itself only justifies 1 must
    not have a forced 2-split read out of it — it drops to {} for every component
    (per-sub-band justification guard), so each component loses that point."""
    acf = _acf_results([[(0.04, 0.6), (0.6, 0.6)]] * 3 + [[(0.1, 0.8)]])
    fr, _fits, _pl = analyze_scintillation_from_acfs(acf, _CFG)
    assert fr["n_components"] == 2
    assert fr["component_selection"]["n_per_subband"] == [2, 2, 2, 1]
    for name in ("component_1", "component_2"):
        assert len(fr["components"][name]["subband_measurements"]) == 3


def test_components_ambiguous_unit():
    sep = [{"dnu_mhz": 0.04, "dnu_err": 0.002}, {"dnu_mhz": 0.6, "dnu_err": 0.01}]
    assert _components_ambiguous(sep) is False
    overlap = [{"dnu_mhz": 0.40, "dnu_err": 0.1}, {"dnu_mhz": 0.45, "dnu_err": 0.1}]
    assert _components_ambiguous(overlap) is True  # |Δ|=0.05 < err sum 0.2
    close_noerr = [{"dnu_mhz": 0.40, "dnu_err": np.nan}, {"dnu_mhz": 0.50, "dnu_err": np.nan}]
    assert _components_ambiguous(close_noerr) is True  # ratio 1.25 < 2.0, errors unknown


if __name__ == "__main__":
    test_determine_n_components_counts()
    test_two_components_wired_into_output()
    test_single_component_unchanged()
    test_non_lorentzian_best_model_stays_single()
    test_underjustified_subband_dropped()
    test_components_ambiguous_unit()
    print("ok")
