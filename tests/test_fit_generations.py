import pathlib

import yaml

ROOT = pathlib.Path(__file__).parents[1]
INV = ROOT / "analysis/fit_generations.yaml"


def test_inventory_exists_and_covers_known_artifacts():
    inv = yaml.safe_load(INV.read_text())
    gens = {g["name"] for g in inv["generations"]}
    assert {"mixed-legacy-2026-06", "beta-campaign-2026-07"} <= gens
    consumers = {c["artifact"]: c["generation"]
                 for c in inv["consumers"]}
    assert consumers["tab:burst-energies"] == "mixed-legacy-2026-06"
    assert consumers["tab:beta"] == "beta-campaign-2026-07"


def test_generation_paths_exist():
    inv = yaml.safe_load(INV.read_text())
    for g in inv["generations"]:
        for p in g["artifact_globs"]:
            assert list(ROOT.glob(p)), f"{g['name']}: no match for {p}"


def test_mixed_legacy_inventory_includes_gate_inputs():
    inv = yaml.safe_load(INV.read_text())
    mixed = next(g for g in inv["generations"]
                 if g["name"] == "mixed-legacy-2026-06")
    assert "analysis/scattering-refit-2026-06/joint_json/*_joint_ppc.json" in (
        mixed["artifact_globs"]
    )
    assert "analysis/scattering-refit-2026-06/_a1_fits/*.json" in (
        mixed["artifact_globs"]
    )


def test_budget_uses_promoted_beta_campaign_generation():
    inv = yaml.safe_load(INV.read_text())
    consumers = {c["artifact"]: c["generation"] for c in inv["consumers"]}
    assert consumers["fig:budget"] == "beta-campaign-2026-07"
