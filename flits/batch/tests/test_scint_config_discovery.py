"""Check that the batch runner resolves the existing hand-tuned scint configs."""

from pathlib import Path

from flits.batch.batch_runner import discover_scint_configs

REPO = Path(__file__).resolve().parents[3]  # flits/batch/tests -> repo root


def test_discovers_all_scint_configs():
    found = discover_scint_configs(REPO / "configs" / "batch", ["chime", "dsa"])
    assert len(found) == 12
    assert all(set(v) == {"chime", "dsa"} for v in found.values())
    p = found["casey"]["dsa"]
    assert p.exists() and p.name == "casey_dsa.yaml"


def test_missing_telescope_dir_is_empty_not_error():
    found = discover_scint_configs(REPO / "configs" / "batch", ["nope"])
    assert found == {}
