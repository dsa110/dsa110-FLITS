import json

from scripts.build_scint_results_table import (
    combined_verdict,
    extract_record,
    gamma_verdict,
)


def test_gamma_verdict_applies_requested_fail_and_pass_invariants():
    """The explicit campaign thresholds, not current campaign outputs, are the oracle."""
    assert gamma_verdict("odr_logspace", {"alpha": 4.4, "alpha_err": 0.4}, 4, True) == (
        "PASS",
        [],
    )
    verdict, reasons = gamma_verdict("joint_2d", {"alpha": 7.9995, "alpha_err": 0.2}, 4, True)
    assert verdict == "FAIL"
    assert "optimizer_bound_saturated" in reasons
    verdict, reasons = gamma_verdict("odr_logspace", {"alpha": 4.4, "alpha_err": None}, 4, True)
    assert verdict == "FAIL"
    assert "missing_or_nonfinite_alpha_err" in reasons


def test_combined_verdict_never_promotes_diagnostic_only():
    """A diagnostic-only measurement remains diagnostic even when ODR passes."""
    assert combined_verdict("diagnostic_only", "PASS") == "DIAGNOSTIC"
    assert combined_verdict("measurement", "PASS") == "PASS"
    assert combined_verdict("measurement", "MARGINAL") == "MARGINAL"
    assert combined_verdict("measurement", "FAIL") == "FAIL"


def test_combined_verdict_never_drops_provenance_caveat():
    """A caveated PASS must not render as a clean PASS in derived tables."""
    assert combined_verdict("measurement", "PASS", "single_block") == "PASS [single_block]"
    assert combined_verdict("diagnostic_only", "PASS", "single_block") == (
        "DIAGNOSTIC [single_block]"
    )
    assert combined_verdict("measurement", "PASS", None) == "PASS"


def test_extract_record_carries_provenance_caveat(tmp_path):
    campaign = tmp_path / "campaign"
    result_dir = campaign / "hamilton"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "hamilton_analysis_results.json"
    result_path.write_text(
        json.dumps(
            {
                "measurement_status": {
                    "status": "measurement",
                    "failed_checks": {},
                    "provenance_caveat": "single_block",
                },
                "components": {
                    "scint_scale": {
                        "gamma_scaling": {
                            "odr_logspace": {"alpha": 4.4, "alpha_err": 0.4},
                            "loglog_unweighted": {"alpha": 4.5, "alpha_err": 0.5},
                            "joint_2d": {"alpha": 4.3, "alpha_err": 0.4},
                        },
                        "subband_measurements": [{"bw": 1.0}, {"bw": 2.0}, {"bw": 3.0}],
                    }
                },
            }
        )
    )

    record = extract_record(result_path, campaign)

    assert record["measurement_status"]["provenance_caveat"] == "single_block"
    assert record["verdict"] == "PASS [single_block]"


def test_extract_record_uses_scaling_component_and_sanitizes_nonfinite(tmp_path):
    """Nested producer schema is normalized and non-finite JSON becomes null-safe None."""
    campaign = tmp_path / "campaign"
    result_dir = campaign / "example"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "source_name_analysis_results.json"
    result_path.write_text(
        json.dumps(
            {
                "measurement_status": {"status": "measurement", "failed_checks": {}},
                "components": {
                    "scint_scale": {
                        "gamma_scaling": {
                            "odr_logspace": {"alpha": 4.4, "alpha_err": 0.4},
                            "loglog_unweighted": {"alpha": 4.5, "alpha_err": 0.5},
                            "joint_2d": {"alpha": 4.3, "alpha_err": 0.4},
                        },
                        "subband_measurements": [
                            {"bw": 1.0},
                            {"bw": 2.0},
                            {"bw": 3.0},
                            {"bw": float("nan")},
                        ],
                    }
                },
                "modulation_index_frequency": {
                    "acf_amplitude": {
                        "components": {"scint_scale": [{"freq_mhz": 600, "m": 0.5, "m_err": 0.1}]}
                    }
                },
                "modulation_index_time": {"direct_std_mean": {"time_s": [0.0], "m": [0.2]}},
            },
            allow_nan=True,
        )
    )

    record = extract_record(result_path, campaign)

    assert record["burst"] == "example"
    assert record["source_burst"] == "source_name"
    assert record["selected_component"] == "scint_scale"
    assert record["n_valid_subbands"] == 3
    assert record["subband_measurements"][-1]["bw"] is None
    assert record["modulation_index_frequency"][0]["mod"] == 0.5
    assert record["modulation_index_time"]["available"] is True
    assert record["verdict"] == "PASS"


def test_malformed_json_is_recorded_without_stopping_campaign(tmp_path):
    """A malformed job produces a FAIL record with its parse error."""
    campaign = tmp_path / "campaign"
    result_dir = campaign / "broken"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "broken_analysis_results.json"
    result_path.write_text("{not-json")

    record = extract_record(result_path, campaign)

    assert record["parse_error"].startswith("JSONDecodeError:")
    assert record["verdict"] == "FAIL"
    assert record["n_valid_subbands"] == 0
