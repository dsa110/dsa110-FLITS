import hashlib
import json
from pathlib import Path

import pandas as pd

from galaxies.foreground.freeze_candidate_redshift_provenance import (
    PAYLOADS,
    _canonical_json,
    verify_frozen_payloads,
)

DATA = Path(__file__).parent / "data"
REGISTRY = DATA / "intervening_census_registry.csv"
PROVENANCE = DATA / "candidate_redshift_provenance.csv"
REPLAY = DATA / "candidate_redshift_replay_2026-07-22.json"


REQUIRED_COLUMNS = {
    "nickname",
    "type",
    "obj",
    "source_family",
    "source_release",
    "retrieved_at_utc",
    "stable_source_id",
    "source_row_sha256",
    "query_response_sha256",
    "adopted_z",
    "adopted_z_err",
    "measurement_kind",
    "source_redshift_flag",
    "source_refcode",
    "source_reported_z_err",
    "source_disposition",
    "final_verdict",
    "budget_eligible",
}


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["obj"] = out["obj"].astype(str)
    return out.set_index(["nickname", "type", "obj"]).sort_index()


def test_candidate_redshift_provenance_covers_registry_without_reclassification():
    registry = _keyed(pd.read_csv(REGISTRY))
    provenance = _keyed(pd.read_csv(PROVENANCE, dtype=str).fillna(""))

    assert REQUIRED_COLUMNS <= set(provenance.reset_index().columns)
    assert list(provenance.index) == list(registry.index)
    assert provenance.index.is_unique

    assert (
        provenance["final_verdict"].astype(str).tolist()
        == registry["final_verdict"].astype(str).tolist()
    )
    assert (
        provenance["budget_eligible"].str.lower().tolist()
        == registry["budget_eligible"].astype(str).str.lower().tolist()
    )


def test_every_adopted_candidate_redshift_has_frozen_source_identity():
    provenance = pd.read_csv(PROVENANCE, dtype=str).fillna("")
    adopted = provenance[provenance["adopted_z"].str.strip() != ""]
    assert len(adopted) == 46

    for column in [
        "source_family",
        "source_release",
        "retrieved_at_utc",
        "stable_source_id",
        "source_row_sha256",
        "adopted_z",
        "measurement_kind",
    ]:
        assert adopted[column].str.strip().ne("").all(), column

    assert adopted["source_row_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert adopted["query_response_sha256"].str.fullmatch(r"[0-9a-f]{64}|not_applicable").all()
    assert set(adopted["measurement_kind"]) <= {
        "photometric",
        "spectroscopic",
        "catalog_cluster",
        "no_trustworthy_redshift",
    }


def test_candidate_redshift_ledger_uses_portable_line_endings():
    assert b"\r\n" not in PROVENANCE.read_bytes()


def test_frozen_payloads_replay_every_source_hash():
    verify_frozen_payloads()


def test_frozen_query_rows_have_deterministic_order():
    payload = json.loads(PAYLOADS.read_text())
    for entry in payload["entries"]:
        response = entry["query_response"]
        if response is None:
            continue
        for key in ("rows", "region_rows", "redshift_rows"):
            if key in response:
                assert response[key] == sorted(response[key], key=_canonical_json)


def test_chromatica_ned_photo_z_is_fail_closed_without_uncertainty():
    provenance = _keyed(pd.read_csv(PROVENANCE, dtype=str).fillna(""))
    row = provenance.loc[("chromatica", "halo", "196733128040225775")]
    assert row["measurement_kind"] == "photometric"
    assert row["source_redshift_flag"] == "PUN"
    assert row["source_refcode"] == "2014ApJS..210....9B"
    assert row["source_reported_z_err"] == ""
    assert row["source_disposition"] == "frozen_not_admitted"
    assert row["final_verdict"] == "inconclusive"
    assert row["budget_eligible"].lower() == "false"


def test_every_confirmed_photometric_redshift_clears_its_uncertainty_gate():
    registry = _keyed(pd.read_csv(REGISTRY))
    provenance = _keyed(pd.read_csv(PROVENANCE, dtype=str).fillna(""))
    confirmed_photo = provenance[
        (provenance["final_verdict"] == "confirmed")
        & (provenance["measurement_kind"] == "photometric")
    ]
    assert len(confirmed_photo) > 0
    for key in confirmed_photo.index:
        row = registry.loc[key]
        assert pd.notna(row.best_z_err), key
        assert row.best_z + row.best_z_err < row.host_z_spec, key


def test_live_replay_receipt_binds_the_current_ledger():
    receipt = json.loads(REPLAY.read_text())
    assert receipt["rows"] == 52
    assert receipt["adopted_redshift_rows"] == 46
    assert receipt["stable_source_id_changes"] == 0
    assert receipt["source_row_sha256_changes"] == 2
    assert receipt["adopted_redshift_changes"] == 0
    assert receipt["measurement_kind_changes"] == 1
    assert receipt["verdict_changes"] == 1
    assert receipt["budget_eligibility_changes"] == 1
    assert receipt["query_response_sha256_changes"] == 20
    assert hashlib.sha256(PROVENANCE.read_bytes()).hexdigest() == receipt["ledger_sha256"]
    assert hashlib.sha256(PAYLOADS.read_bytes()).hexdigest() == receipt["payload_sha256"]
