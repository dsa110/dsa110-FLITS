import hashlib
import json
from pathlib import Path

import pandas as pd

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
    assert (
        adopted["query_response_sha256"].str.fullmatch(r"[0-9a-f]{64}|not_applicable").all()
    )
    assert set(adopted["measurement_kind"]) <= {
        "photometric",
        "spectroscopic",
        "catalog_cluster",
        "no_trustworthy_redshift",
    }


def test_candidate_redshift_ledger_uses_portable_line_endings():
    assert b"\r\n" not in PROVENANCE.read_bytes()


def test_live_replay_receipt_binds_the_current_ledger():
    receipt = json.loads(REPLAY.read_text())
    assert receipt["rows"] == 52
    assert receipt["adopted_redshift_rows"] == 46
    assert receipt["stable_source_id_changes"] == 0
    assert receipt["source_row_sha256_changes"] == 0
    assert receipt["adopted_redshift_changes"] == 0
    assert receipt["verdict_changes"] == 0
    assert receipt["budget_eligibility_changes"] == 0
    assert receipt["query_response_sha256_changes"] == 7
    assert hashlib.sha256(PROVENANCE.read_bytes()).hexdigest() == receipt["ledger_sha256"]
