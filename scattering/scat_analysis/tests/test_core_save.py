"""Tests for build_safe_results: posterior percentiles must survive to JSON."""

import json
from dataclasses import dataclass

from flits.scattering.scat_analysis.pipeline.core import build_safe_results


@dataclass
class _StubParams:
    c0: float = 1.0
    tau_1ghz: float = 0.19


class _StubNested:
    """Minimal stand-in for a NestedSamplingResult."""

    def __init__(self, logz, tau, err_minus, err_plus):
        self.log_evidence = logz
        self.log_evidence_err = 0.1
        self.percentiles = {
            "tau_1ghz": {
                "median": tau,
                "lower": tau - err_minus,
                "upper": tau + err_plus,
                "err_minus": err_minus,
                "err_plus": err_plus,
            }
        }


def _results():
    m3 = _StubNested(11697.0, 0.194, 0.02, 0.03)
    m0 = _StubNested(9938.0, 0.0, 0.0, 0.0)
    return {
        "best_key": "M3",
        "best_params": _StubParams(),
        "param_names": ["c0", "tau_1ghz"],
        "goodness_of_fit": {"chi2_reduced": 1.36, "quality_flag": "PASS"},
        "dm_init": 0.0,
        "loop_stats": {"ncall": [1, 2, 3]},
        "all_results": {"M3": m3, "M0": m0},
    }


def test_best_params_percentiles_persisted():
    safe = build_safe_results(_results())
    bpp = safe["best_params_percentiles"]
    assert bpp is not None
    assert "tau_1ghz" in bpp
    assert bpp["tau_1ghz"]["median"] == 0.194
    assert bpp["tau_1ghz"]["err_minus"] == 0.02
    assert bpp["tau_1ghz"]["err_plus"] == 0.03


def test_per_model_percentiles_persisted():
    safe = build_safe_results(_results())
    assert safe["all_results"]["M3"]["percentiles"]["tau_1ghz"]["median"] == 0.194
    assert safe["all_results"]["M3"]["log_evidence"] == 11697.0
    assert safe["all_results"]["M0"]["log_evidence"] == 9938.0


def test_safe_results_is_json_serializable():
    safe = build_safe_results(_results())
    # Must round-trip through JSON (the actual save does json.dump).
    text = json.dumps(safe)
    back = json.loads(text)
    assert back["best_model"] == "M3"
    assert back["best_params"]["tau_1ghz"] == 0.19
    assert back["goodness_of_fit"]["quality_flag"] == "PASS"


def test_missing_percentiles_degrades_to_none():
    # The emcee/BIC branch yields objects without a percentiles attribute.
    class _Bare:
        log_evidence = None
        log_evidence_err = None

    results = _results()
    results["all_results"] = {"M3": _Bare()}
    safe = build_safe_results(results)
    assert safe["best_params_percentiles"] is None
    assert safe["all_results"]["M3"]["percentiles"] is None
