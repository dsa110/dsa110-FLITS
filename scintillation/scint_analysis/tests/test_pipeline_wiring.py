"""Phase 4: two-screen / emission-size / consistency interpretation attaches to the
``analyze_scintillation_from_acfs`` output (`attach_scintillation_interpretation`).

Tested as a pure function on a synthetic `final_results` dict + config rather than via
a full `ScintillationAnalysis.run()`, because a real run needs gitignored raw burst
spectra (DATA_SOURCES.md). The wiring point in `pipeline.py` is a single call to this
function, so pinning the function pins the wiring contract without data.
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import numpy as np

from scint_analysis.analysis import (
    attach_scintillation_interpretation,
    interpret_modulation_index,
    scattering_scintillation_consistency,
)


def _final_results_one_component():
    """A minimal `analyze_scintillation_from_acfs`-shaped result: one component with a
    power-law Δν(ref) + two per-subband modulation indices."""
    return {
        "best_model": "lorentzian_component",
        "components": {
            "scint_scale": {
                "bw_at_ref_mhz": 0.318,  # Δν_dc at the reference frequency
                "bw_at_ref_mhz_err": 0.02,
                "scaling_index": 4.1,
                "subband_measurements": [
                    {"freq_mhz": 500.0, "bw": 0.20, "mod": 0.80, "mod_err": 0.05},
                    {"freq_mhz": 700.0, "bw": 0.45, "mod": 0.78, "mod_err": 0.05},
                ],
            }
        },
    }


def test_modulation_interpretation_keys():
    """Contract guard: the wiring depends on these keys from `interpret_modulation_index`."""
    r = interpret_modulation_index(0.9, 0.05)
    assert {"interpretation", "emission_resolved", "resolution_regime"} <= set(r)


def test_consistency_single_screen_flag():
    """Contract guard: `scattering_scintillation_consistency` returns C_implied + a flag."""
    r = scattering_scintillation_consistency(0.5, 0.318, C=1.0)  # 2π·τ·Δν
    assert "C_implied" in r and r["consistent"] in (True, False)


def test_attach_wires_all_keys_with_source():
    """With a full `source` block, all three interpretations attach to the component."""
    fr = _final_results_one_component()
    config = {
        "analysis": {"fitting": {"reference_frequency_mhz": 600.0}},
        "source": {"tau_d_ms": 0.5, "d_source_screen_pc": 11000.0, "distance_mpc": 65.0},
    }
    attach_scintillation_interpretation(fr, config)
    comp = fr["components"]["scint_scale"]
    # m = median(0.80, 0.78) = 0.79 -> marginally_resolved (not the "unknown" default)
    assert "modulation" in comp and comp["modulation"]["resolution_regime"] != "unknown"
    assert "consistency" in comp and "C_implied" in comp["consistency"]
    assert "emission_size" in comp and np.isfinite(comp["emission_size"]["R_obs_km"])


def test_attach_noop_without_source():
    """No `source` block -> only `modulation` attaches (m is intrinsic); the
    distance/τ-gated interpretations are cleanly skipped."""
    fr = _final_results_one_component()
    attach_scintillation_interpretation(fr, {"analysis": {"fitting": {}}})
    comp = fr["components"]["scint_scale"]
    assert "modulation" in comp  # m always available
    assert "consistency" not in comp  # needs τ
    assert "emission_size" not in comp  # needs screen distance


def test_attach_skips_failed_component():
    """A failed component (no `subband_measurements`) is left exactly as-is."""
    failed = {"power_law_fit_report": "Fit failed: Non-positive BWs"}
    fr = {"best_model": "x", "components": {"bad": dict(failed)}}
    attach_scintillation_interpretation(fr, {"source": {"tau_d_ms": 0.5}})
    assert fr["components"]["bad"] == failed
