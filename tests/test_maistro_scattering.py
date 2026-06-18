"""Network-free tests for the DM/scattering-budget Maistro sidecar."""

import math

import pandas as pd

from flits.orchestration import maistro_scattering as ms


class _FakeClient:
    def __init__(self):
        self.ready = 0
        self.batches = []
        self.staged = []

    def ensure_ready(self):
        self.ready += 1
        return {"ready": True}

    def write_batch(self, body):
        self.batches.append(body)
        return {"ok": True}

    def stage(self, body):
        self.staged.append(body)
        return {"ok": True}


def _rows():
    return [
        # locked-in (PASS) foreground-dominated scattering
        {
            "name": "Wilhelm", "z_frb": 0.51, "l_deg": 107.1, "b_deg": 16.7,
            "dm_obs": 602.0, "dm_mw_ism": 83.0, "dm_cosmic": 456.0,
            "dm_intervening_capped": 103.0, "dm_host_capped": -80.0,
            "tau_obs_ms": 0.194, "tau_obs_err_minus": 0.02, "tau_obs_err_plus": 0.03,
            "tau_obs_quality": "PASS", "tau_obs_chi2_reduced": 1.36,
            "tau_intervening_ms": 0.008, "n_foreground": 2, "n_intersecting": 2,
            "intervening_mass_confidence": "assumed", "z_is_placeholder": False,
            "verdict_scattering": "intervening negligible; host / Milky-Way dominated (pred/obs=0.041)",
            "verdict_dm": "cosmic/IGM-dominated",
        },
        # withheld fit (FAIL) -> should be staged for refit
        {
            "name": "Casey", "z_frb": 0.287, "dm_obs": 491.0,
            "dm_intervening_capped": 0.0, "dm_host_capped": 163.0,
            "tau_obs_ms": float("nan"), "tau_obs_quality": "FAIL", "tau_obs_chi2_reduced": 69.2,
            "tau_intervening_ms": 0.0, "n_foreground": 0, "n_intersecting": 0,
            "z_is_placeholder": False,
            "verdict_scattering": "scattering fit present but quality_flag=FAIL (chi2_red=69.2); tau not locked in",
            "verdict_dm": "host-dominated",
        },
        # no measurement, foreground present
        {
            "name": "Phineas", "z_frb": 0.271, "dm_obs": 610.0,
            "dm_intervening_capped": 367.5, "dm_host_capped": -70.0,
            "tau_obs_ms": float("nan"), "tau_obs_quality": None,
            "tau_intervening_ms": 1.36, "n_foreground": 1, "n_intersecting": 1,
            "intervening_mass_confidence": "measured", "z_is_placeholder": False,
            "verdict_scattering": "no scattering measurement (predicted intervening tau=1.4 ms)",
            "verdict_dm": "intervening galaxy-dominated",
        },
    ]


def test_build_payloads_state_and_staging():
    p = ms.build_budget_payloads(_rows(), git_sha="abc123", git_dirty=False)
    assert p["run_id"] == ms.STATE_RUN_ID
    items = {it["key"]: it["value"] for it in p["write_batch"]["items"]}

    # Per-sightline canonical state present.
    assert "budget.wilhelm" in items and "budget.casey" in items and "budget.phineas" in items
    assert items["budget.wilhelm"]["tau_obs_ms"] == 0.194
    # NaN tau coerced to None (JSON-safe).
    assert items["budget.casey"]["tau_obs_ms"] is None
    # Summary counts.
    assert items["summary.n_sightlines"] == 3
    assert items["summary.n_tau_locked_in"] == 1            # only Wilhelm (PASS + finite)
    assert items["pipeline.git_sha"] == "abc123"

    # Only the FAIL/MARGINAL fit is staged for refit.
    assert len(p["stage"]) == 1
    st = p["stage"][0]
    assert st["kind"] == ms.REFIT_KIND
    assert st["payload"]["sightline"] == "Casey"
    assert st["payload"]["tau_obs_quality"] == "FAIL"


def test_ingest_dry_run_does_not_call_client():
    df = pd.DataFrame(_rows())
    fake = _FakeClient()
    out = ms.ingest_budget(df, client=fake, dry_run=True)
    assert fake.ready == 0 and not fake.batches and not fake.staged
    assert out["run_id"] == ms.STATE_RUN_ID


def test_ingest_posts_state_then_stage():
    df = pd.DataFrame(_rows())
    fake = _FakeClient()
    ms.ingest_budget(df, client=fake, dry_run=False)
    assert fake.ready == 1
    assert len(fake.batches) == 1
    assert len(fake.staged) == 1
    # write_batch carries setState ops only.
    assert all(it["op"] == "setState" for it in fake.batches[0]["items"])


def test_tau_locked_predicate():
    assert ms._tau_locked({"tau_obs_quality": "PASS", "tau_obs_ms": 0.1}) is True
    assert ms._tau_locked({"tau_obs_quality": "INJECTED", "tau_obs_ms": 0.1}) is True
    assert ms._tau_locked({"tau_obs_quality": "FAIL", "tau_obs_ms": float("nan")}) is False
    assert ms._tau_locked({"tau_obs_quality": "PASS", "tau_obs_ms": float("nan")}) is False
