from pathlib import Path

import pandas as pd

from scripts.foreground.aggregate_revalidation_replay import aggregate


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_aggregate_revalidation_replay_outputs_combined_artifacts(tmp_path):
    input_dir = tmp_path / "live_search"
    output_dir = tmp_path / "combined"
    _write(
        input_dir / "casey" / "search_summary.csv",
        "name,target_id,ra,dec,z_frb,num_galaxies\nCasey,1,11h,+70d,0.287,1\n",
    )
    _write(
        input_dir / "casey" / "survey_coverage.csv",
        "nickname,ra,dec,z_frb,survey,engine,in_footprint,queried,raw_count,with_z_count,foreground_count,status\n"
        "Casey,11h,+70d,0.287,NED,NedTapEngine,True,True,1,1,1,foreground\n",
    )
    _write(
        input_dir / "casey" / "casey_galaxies.csv",
        "ra,dec,z,catalog,impact_kpc\n170.0,70.0,0.1,NED,50\n",
    )
    registry = tmp_path / "registry.csv"
    _write(
        registry,
        "nickname,obj,ra_deg,dec_deg,best_z,final_verdict\ncasey,known,170.0,70.0,0.1,confirmed\n"
        "whitney,missing,134.0,73.0,0.2,confirmed\n",
    )

    counts = aggregate(input_dir, output_dir, registry)

    assert counts["sightlines"] == 1
    assert counts["candidate_rows"] == 1
    assert counts["matched_registry_rows"] == 1
    assert counts["registry_not_in_replay_rows"] == 1
    candidates = pd.read_csv(output_dir / "foreground_live_replay_candidates.csv")
    diff = pd.read_csv(output_dir / "foreground_live_replay_registry_diff.csv")
    assert candidates["nickname"].tolist() == ["casey"]
    assert set(diff["status"]) == {"matched_registry", "registry_not_in_replay"}
