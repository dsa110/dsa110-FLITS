"""Escalation-trigger verdict logic (A1, Phase 5).

Combines the injection-calibrated dlnZ limb (i) with the rung-iv PPC limb
(ii); a railed second component is model-family rejection, never a detection;
a non-escalation is censored, never a single-screen claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

_test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_test_dir.parent.parent.parent))  # FLITS root
sys.path.insert(0, str(_test_dir.parent.parent))  # scintillation dir

import pytest  # noqa: E402

from scint_analysis.acf_evidence import escalation_trigger_verdict  # noqa: E402

CAL = {"dlnz_threshold": 6.2, "rate": 0.01}


def test_escalates_on_dlnz():
    v = escalation_trigger_verdict(dlnz=9.0, rail_flags=[],
                                   ppc_pvalues={"lag1_acf": 0.5},
                                   calibration=CAL)
    assert v["escalate"] and v["reasons"] == ["dlnz"]
    assert v["verdict"] == "escalate"


def test_rail_is_model_family_rejection_not_detection():
    v = escalation_trigger_verdict(dlnz=9.0, rail_flags=["f"],
                                   ppc_pvalues={"lag1_acf": 0.5},
                                   calibration=CAL)
    assert not v["escalate"]
    assert v["verdict"] == "model_family_rejection"
    assert v["rail_flags"] == ["f"]


def test_ppc_limb_escalates_independently():
    v = escalation_trigger_verdict(dlnz=1.0, rail_flags=[],
                                   ppc_pvalues={"lag1_acf": 0.01},
                                   calibration=CAL)
    assert v["escalate"] and v["reasons"] == ["ppc_lag1_acf"]


def test_ppc_upper_band_edge_also_fires():
    v = escalation_trigger_verdict(dlnz=1.0, rail_flags=[],
                                   ppc_pvalues={"lag1_acf": 0.99},
                                   calibration=CAL)
    assert v["escalate"] and v["reasons"] == ["ppc_lag1_acf"]


def test_non_detection_is_censored_not_single_screen():
    v = escalation_trigger_verdict(dlnz=1.0, rail_flags=[],
                                   ppc_pvalues={"lag1_acf": 0.5},
                                   calibration=CAL)
    assert not v["escalate"]
    assert v["verdict"] == "no_escalation_censored"


def test_missing_ppc_is_tolerated():
    v = escalation_trigger_verdict(dlnz=9.0, rail_flags=[], ppc_pvalues={},
                                   calibration=CAL)
    assert v["escalate"] and v["reasons"] == ["dlnz"]


def test_both_limbs_reported():
    v = escalation_trigger_verdict(dlnz=9.0, rail_flags=[],
                                   ppc_pvalues={"lag1_acf": 0.01},
                                   calibration=CAL)
    assert v["reasons"] == ["dlnz", "ppc_lag1_acf"]


def test_calibration_must_carry_threshold():
    with pytest.raises(KeyError):
        escalation_trigger_verdict(dlnz=1.0, rail_flags=[],
                                   ppc_pvalues={}, calibration={})
