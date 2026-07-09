import csv
import pathlib

MANIFEST = pathlib.Path(__file__).parents[1] / "data-manifest.csv"


def _rows():
    with MANIFEST.open() as fh:
        return list(csv.DictReader(fh))


def test_manifest_parses_and_is_nonempty():
    rows = _rows()
    assert len(rows) == 24


# xfail removed 2026-07-06: P2.2 closed as an arc<->local byte cross-check
# (h17 holds no manifest cubes; all 24 downloaded from arc and sha256-matched
# against the local replica, fail=0).
def test_manifest_has_no_pending_checksums():
    pending = [r["filename"] for r in _rows()
               if r["sha256"].strip().upper().startswith("PENDING")]
    assert pending == []


def test_manifest_rows_are_byte_verified_against_arc():
    bad = [(r["filename"], r["status"]) for r in _rows()
           if r["status"] != "ARC_BYTE_MATCH"]
    assert bad == []
