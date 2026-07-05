import pytest

from .targets import get_cosmology, load_targets


@pytest.mark.unit
def test_load_targets_valid_yaml(sample_targets_yaml):
    targets = load_targets(sample_targets_yaml)
    assert len(targets) == 3
    assert targets[0].name == "FRB_Test_A"
    assert targets[0].ra == 150.1149
    assert targets[0].dec == 2.2058
    assert targets[0].z_host == 0.225
    assert targets[2].z_host is None  # optional field


@pytest.mark.unit
def test_load_targets_empty_file(tmp_path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("targets: []")
    assert load_targets(empty_file) == []


@pytest.mark.unit
def test_load_targets_missing_file():
    with pytest.raises(FileNotFoundError):
        load_targets("nonexistent.yaml")


@pytest.mark.unit
def test_get_cosmology_default():
    assert get_cosmology().name == "Planck18"
    assert get_cosmology("Planck18").name == "Planck18"


@pytest.mark.unit
def test_get_cosmology_invalid():
    with pytest.raises(ValueError, match="Unsupported cosmology"):
        get_cosmology("InvalidCosmology")
