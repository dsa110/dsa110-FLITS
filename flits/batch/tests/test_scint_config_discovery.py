"""Check that the batch runner resolves the existing hand-tuned scint configs."""

from pathlib import Path

from flits.batch.batch_runner import discover_scint_configs

REPO = Path(__file__).resolve().parents[3]  # flits/batch/tests -> repo root
SCINT_CONFIGS = REPO / "scintillation" / "configs" / "bursts"


def test_discovers_hand_tuned_scint_configs():
    found = discover_scint_configs(SCINT_CONFIGS, ["chime", "dsa"])
    # every co-detection burst has a hand-tuned DSA config
    dsa_bursts = {b for b, tels in found.items() if "dsa" in tels}
    assert len(dsa_bursts) == 12
    assert found["casey"].keys() >= {"chime", "dsa"}
    p = found["casey"]["dsa"]
    assert p.exists() and p.name == "casey_dsa.yaml"
    # variant configs (_hi, _temp) must not be swallowed as bursts
    assert "casey_chime_hi" not in found
    assert all("_temp" not in b and not b.endswith("_hi") for b in found)
    # keys are lowercased to match BurstInfo.from_filename burst names
    assert "johndoeii" in dsa_bursts
    assert "johndoeII" not in found
    assert found["johndoeii"]["dsa"].name == "johndoeII_dsa.yaml"


def test_missing_telescope_dir_is_empty_not_error():
    found = discover_scint_configs(SCINT_CONFIGS / "nope", ["chime", "dsa"])
    assert found == {}
