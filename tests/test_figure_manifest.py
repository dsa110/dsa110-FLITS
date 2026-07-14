import json

from tools.figure_manifest import write_manifest


def test_write_manifest_merges_separate_figure_owners(tmp_path):
    write_manifest(tmp_path, [("analysis_overview.png", "three gamma estimators")])
    write_manifest(tmp_path, [("intra_pulse.png", "ACF-fitted time evolution")])

    manifest = json.loads((tmp_path / "figures.manifest.json").read_text())
    assert {figure["path"] for figure in manifest["figures"]} == {
        "analysis_overview.png",
        "intra_pulse.png",
    }
