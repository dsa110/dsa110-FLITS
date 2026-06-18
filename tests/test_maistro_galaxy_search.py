import json
from pathlib import Path

import pandas as pd
import pytest

from flits.orchestration import maistro


def write_fake_outputs(output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "name": "Alpha",
                "target_id": 1,
                "ra": "01h00m00s",
                "dec": "+02d00m00s",
                "z_frb": 0.1,
                "num_galaxies": 2,
            },
            {
                "name": "Beta",
                "target_id": 2,
                "ra": "03h00m00s",
                "dec": "+04d00m00s",
                "z_frb": 0.2,
                "num_galaxies": 0,
            },
        ]
    ).to_csv(output_dir / "search_summary.csv", index=False)
    pd.DataFrame(
        [
            {"name": "farther", "impact_kpc": 55.0, "z": 0.08},
            {"name": "nearest", "impact_kpc": 12.5, "z": 0.09},
        ]
    ).to_csv(output_dir / "alpha_galaxies.csv", index=False)


def test_build_galaxy_payloads_include_state_and_one_candidate_per_target(tmp_path):
    write_fake_outputs(tmp_path)

    payloads = maistro.build_galaxy_payloads(
        run_id="run-1",
        impact_kpc=100.0,
        output_dir=tmp_path,
        z_eps=0.01,
        command=["python", "scripts/run_maistro_galaxy_search.py", "--dry-run"],
        git_sha="abc123",
        git_dirty=False,
        targets=[("Alpha", "01h00m00s", "+02d00m00s", 0.1), ("Beta", "03h00m00s", "+04d00m00s", 0.2)],
        catalogs={"GLADE+": "VII/291/glade"},
        pipeline_status="passed",
    )

    write_batch = payloads["write_batch"]
    state = {item["key"]: item["value"] for item in write_batch["items"]}
    assert write_batch["run_id"] == "run-1"
    assert all(item["op"] == "setState" for item in write_batch["items"])
    assert state["pipeline.kind"] == "flits-galaxy-search"
    assert state["pipeline.status"] == "passed"
    assert state["pipeline.git_sha"] == "abc123"
    assert state["pipeline.git_dirty"] is False
    assert state["galaxy.params.impact_kpc"] == 100.0
    assert state["galaxy.params.z_eps"] == 0.01
    assert state["galaxy.catalogs"] == {"GLADE+": "VII/291/glade"}
    assert state["galaxy.targets"]["hash"]
    assert len(state["galaxy.targets"]["items"]) == 2
    assert state["artifact.search_summary_csv"] == str(tmp_path / "search_summary.csv")
    assert state["artifact.output_dir"] == str(tmp_path)
    assert state["summary.total_targets"] == 2
    assert state["summary.targets_with_matches"] == 1
    assert state["summary.total_matches"] == 2

    stages = payloads["stage"]
    assert len(stages) == 2
    assert {stage["kind"] for stage in stages} == {"flits.galaxy.match_summary"}
    alpha = stages[0]["payload"]
    assert alpha["target_name"] == "Alpha"
    assert alpha["csv_path"] == str(tmp_path / "alpha_galaxies.csv")
    assert alpha["review_note"] == "foreground matches found"
    assert alpha["best_match"]["name"] == "nearest"
    assert alpha["best_match"]["impact_kpc"] == 12.5
    beta = stages[1]["payload"]
    assert beta["target_name"] == "Beta"
    assert beta["csv_path"] == str(tmp_path / "beta_galaxies.csv")
    assert beta["review_note"] == "no foreground matches found"
    assert "best_match" not in beta


def test_dry_run_prints_payloads_and_skips_rpc(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run_search(impact_kpc, output_dir, z_eps):
        calls.append((impact_kpc, output_dir, z_eps))
        print("search progress")
        write_fake_outputs(Path(output_dir))

    monkeypatch.setattr(maistro.galaxy_search, "run_search", fake_run_search)
    monkeypatch.setattr(maistro.galaxy_config, "TARGETS", [("Alpha", "01h00m00s", "+02d00m00s", 0.1), ("Beta", "03h00m00s", "+04d00m00s", 0.2)])
    monkeypatch.setattr(maistro.galaxy_config, "VIZIER_CATALOGS", {"GLADE+": "VII/291/glade"})
    monkeypatch.setattr(maistro, "git_metadata", lambda repo_root=None: ("abc123", False))

    result = maistro.run_galaxy_provenance(
        run_id="dry-run-1",
        impact_kpc=100.0,
        output_dir=tmp_path,
        z_eps=0.01,
        command=["dry"],
        dry_run=True,
    )

    assert result == 0
    assert calls == [(100.0, str(tmp_path), 0.01)]
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert "search progress" in captured.err
    assert printed["run_id"] == "dry-run-1"
    assert printed["write_batch"]["run_id"] == "dry-run-1"
    assert len(printed["stage"]) == 2


def test_rpc_success_path_checks_ready_writes_state_and_stages_targets(tmp_path, monkeypatch):
    write_fake_outputs(tmp_path)
    posted = []

    class FakeClient:
        def ensure_ready(self):
            posted.append(("ready", None))

        def write_batch(self, body):
            posted.append(("write_batch", body))
            return {"seqs": list(range(1, len(body["items"]) + 1))}

        def stage(self, body):
            posted.append(("stage", body))
            return {"id": f"candidate-{len(posted)}", "status": "review"}

    monkeypatch.setattr(maistro.galaxy_search, "run_search", lambda impact_kpc, output_dir, z_eps: None)
    monkeypatch.setattr(maistro.galaxy_config, "TARGETS", [("Alpha", "01h00m00s", "+02d00m00s", 0.1), ("Beta", "03h00m00s", "+04d00m00s", 0.2)])
    monkeypatch.setattr(maistro.galaxy_config, "VIZIER_CATALOGS", {"GLADE+": "VII/291/glade"})
    monkeypatch.setattr(maistro, "git_metadata", lambda repo_root=None: ("abc123", False))

    result = maistro.run_galaxy_provenance(
        run_id="rpc-run-1",
        impact_kpc=100.0,
        output_dir=tmp_path,
        z_eps=0.01,
        command=["rpc"],
        dry_run=False,
        client=FakeClient(),
    )

    assert result == 0
    assert [name for name, _ in posted] == ["ready", "write_batch", "stage", "stage"]
    assert posted[1][1]["run_id"] == "rpc-run-1"
    assert posted[2][1]["payload"]["target_name"] == "Alpha"
    assert posted[3][1]["payload"]["target_name"] == "Beta"


def test_ready_failure_exits_before_running_search(tmp_path, monkeypatch):
    ran_search = False

    class FailingClient:
        def ensure_ready(self):
            raise maistro.MaistroRpcError("not ready")

    def fake_run_search(impact_kpc, output_dir, z_eps):
        nonlocal ran_search
        ran_search = True

    monkeypatch.setattr(maistro.galaxy_search, "run_search", fake_run_search)

    with pytest.raises(maistro.MaistroRpcError):
        maistro.run_galaxy_provenance(
            run_id="not-ready",
            impact_kpc=100.0,
            output_dir=tmp_path,
            z_eps=0.01,
            command=["rpc"],
            dry_run=False,
            client=FailingClient(),
        )

    assert ran_search is False
